/*
 * BQQ fused forward CUDA kernels.
 *
 * Computes  out = X @ W.T  where
 *   W = Σ_p (a_p·Y_p@Z_p + b_p·Ysum_p + c_p·Zsum_p) + d
 * without materialising W.
 *
 * Two kernel variants:
 *
 *   v1 — warp-shuffle kernel (bqq_forward_warp_kernel)
 *     Thread j holds X[j]; warp __shfl_down to compute t_k = Σ_j Z[k,j]·X[j].
 *     Then thread i conditionally adds t_aug if Y[i,k]=1.
 *     Bottleneck: k8*8 warp shuffles per (p,c).
 *
 *   v2 — popcount kernel (bqq_forward_popcount_kernel)   ← default
 *     Reformulates: core[i] = Σ_j X[j] · popcount(Y[i,:] AND Z[:,j]).
 *     Each thread independently computes its own output row via AND+popcount.
 *     NO warp shuffles, NO bit→float conversion.
 *     All Z bytes in shared memory; Y bytes in registers.
 *     Bottleneck: z_col * k8 popcounts per (p,c) — pure ALU, no shuffle unit.
 *
 * Grid: (row_width, batch)    Block: 32 threads (one warp)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>


/* ── warp-level sum (used by v1 only) ──────────────────────────── */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


/* ═══════════════════════════════════════════════════════════════════
 * v1: warp-shuffle kernel
 * ═══════════════════════════════════════════════════════════════════ */

template <int N_J_TILES, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_warp_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const float*   __restrict__ X,
    const float*   __restrict__ a_ptr,
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int r    = blockIdx.x;
    const int n    = blockIdx.y;
    const int lane = threadIdx.x;

    extern __shared__ float x_smem[];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = 0; ci < col_width; ci++) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        #pragma unroll
        for (int jt = 0; jt < N_J_TILES; jt++) {
            int j = jt * 32 + lane;
            if (j < z_col) x_smem[j] = X[x_base + j];
        }
        __syncwarp();

        float xsum = 0.0f;
        #pragma unroll
        for (int jt = 0; jt < N_J_TILES; jt++) {
            int j = jt * 32 + lane;
            float xj = (j < z_col) ? x_smem[j] : 0.0f;
            xsum += warp_reduce_sum(xj);
        }
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];
            float t_sum = 0.0f;

            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;

                uint8_t zb[N_J_TILES], yb[N_I_TILES];
                #pragma unroll
                for (int jt = 0; jt < N_J_TILES; jt++) {
                    int j = jt * 32 + lane;
                    zb[jt] = (j < z_col)
                        ? Z[(size_t)B_idx * z_col * k8 + j * k8 + bk] : 0;
                }
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    int i = it * 32 + lane;
                    yb[it] = (i < y_row)
                        ? Y[(size_t)B_idx * y_row * k8 + i * k8 + bk] : 0;
                }

                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const int shift = 7 - bit;
                    float t_k = 0.0f;
                    #pragma unroll
                    for (int jt = 0; jt < N_J_TILES; jt++) {
                        int j = jt * 32 + lane;
                        int z_bit = (zb[jt] >> shift) & 1;
                        float xj  = (j < z_col) ? x_smem[j] : 0.0f;
                        t_k += warp_reduce_sum(z_bit ? xj : 0.0f);
                    }
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;
                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((yb[it] >> shift) & 1)
                            acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }
            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }
        const float d_term = d_ptr[rc] * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        int i = it * 32 + lane;
        if (i < y_row)
            out[(size_t)n * row_width * y_row + r * y_row + i] = acc[it];
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * v2: popcount kernel — NO warp shuffles
 * ═══════════════════════════════════════════════════════════════════
 *
 * Key idea: W[i,j] involves Y[i,:] · Z[:,j] (binary inner product).
 * This equals popcount(Y_packed[i,:] AND Z_packed[j,:]).
 *
 * So: core[i] = Σ_j X[j] · popcount(Y[i,:] AND Z[:,j])
 *
 * Each thread computes its own i's output by iterating over j.
 * Thread reads Y bytes into registers (private), Z bytes from shared
 * memory (broadcast to all threads). No inter-thread communication!
 *
 * Shared memory layout:
 *   [0 .. z_col*4-1]                : X values (float)
 *   [z_col*4 .. z_col*4+z_col*k8-1] : Z bytes  (uint8)
 */

template <int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_popcount_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const float*   __restrict__ X,
    const float*   __restrict__ a_ptr,
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int r    = blockIdx.x;
    const int n    = blockIdx.y;
    const int lane = threadIdx.x;          /* = output row index within tile */

    /* shared memory */
    extern __shared__ char smem_raw[];
    float*   x_smem = reinterpret_cast<float*>(smem_raw);
    uint8_t* z_smem = reinterpret_cast<uint8_t*>(x_smem + z_col);

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    /* ── col_width loop ──────────────────────────────────────── */
    for (int ci = 0; ci < col_width; ci++) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        /* load X[n, c, :] → shared */
        for (int j = lane; j < z_col; j += 32)
            x_smem[j] = X[x_base + j];
        __syncwarp();

        /* xsum = Σ_j X[j]  (all threads compute identically via smem) */
        float xsum = 0.0f;
        for (int j = 0; j < z_col; j++)
            xsum += x_smem[j];

        /* ── bit_width loop ──────────────────────────────────── */
        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];

            /* load Z[B_idx, :, :] → shared */
            for (int j = lane; j < z_col; j += 32)
                for (int bk = 0; bk < k8; bk++)
                    z_smem[j * k8 + bk] =
                        Z[(size_t)B_idx * z_col * k8 + j * k8 + bk];
            __syncwarp();

            /* load Y bytes → registers;  compute Ysum */
            uint8_t yb[N_I_TILES][K8_MAX];
            float y_sum[N_I_TILES];
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) {
                int i = it * 32 + lane;
                int s = 0;
                #pragma unroll
                for (int bk = 0; bk < K8_MAX; bk++) {
                    if (bk < k8 && i < y_row) {
                        uint8_t v = Y[(size_t)B_idx * y_row * k8 + i * k8 + bk];
                        yb[it][bk] = v;
                        s += __popc(static_cast<unsigned>(v));
                    } else {
                        yb[it][bk] = 0;
                    }
                }
                y_sum[it] = static_cast<float>(s);
            }

            /* ── j-loop: core[i] = Σ_j X[j] · popc(Y[i] AND Z[j]) ── */
            float core_val[N_I_TILES];
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) core_val[it] = 0.0f;
            float z_dot_x = 0.0f;

            for (int j = 0; j < z_col; j++) {
                const float xj = x_smem[j];

                /* read Z bytes for column j from shared memory */
                uint8_t zb_reg[K8_MAX];
                int zs = 0;                             /* Zsum[j] */
                #pragma unroll
                for (int bk = 0; bk < K8_MAX; bk++) {
                    if (bk < k8) {
                        zb_reg[bk] = z_smem[j * k8 + bk];
                        zs += __popc(static_cast<unsigned>(zb_reg[bk]));
                    } else {
                        zb_reg[bk] = 0;
                    }
                }
                z_dot_x += xj * static_cast<float>(zs);

                /* binary inner product for each i-tile */
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    int inner = 0;
                    #pragma unroll
                    for (int bk = 0; bk < K8_MAX; bk++) {
                        if (bk < k8)
                            inner += __popc(static_cast<unsigned>(
                                yb[it][bk] & zb_reg[bk]));
                    }
                    core_val[it] += xj * static_cast<float>(inner);
                }
            }

            /* accumulate: a·core + b·xsum·Ysum + c·Zsum@X */
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++)
                acc[it] += a_val * core_val[it]
                         + b_val * xsum * y_sum[it]
                         + c_val * z_dot_x;
        }

        /* term 4: d · xsum */
        const float d_term = d_ptr[rc] * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

    /* ── store ───────────────────────────────────────────────── */
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        int i = it * 32 + lane;
        if (i < y_row)
            out[(size_t)n * row_width * y_row + r * y_row + i] = acc[it];
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * v3: multi-warp shuffle + col_width split for high occupancy
 * ═══════════════════════════════════════════════════════════════════
 *
 * Problem with v1/v2: only 1 warp per block → 48 warps for 56 SMs
 * → 1.6% occupancy → can't hide memory latency.
 *
 * Solution: split col_width across warps within a block.
 * Each warp handles col_width/N_WARPS c-values independently,
 * then cross-warp reduction via shared memory.
 *
 * Block: N_WARPS × 32 threads    Grid: (row_width, batch)
 */

template <int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_multiw_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const float*   __restrict__ X,
    const float*   __restrict__ a_ptr,
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int r       = blockIdx.x;
    const int n       = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;      /* 0 .. N_WARPS-1 */
    const int lane    = threadIdx.x & 31;

    __shared__ float warp_acc[N_WARPS * 32];   /* cross-warp reduction buffer */

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    /* distribute col_width across warps: warp w handles c = w, w+NW, w+2*NW, ... */
    for (int ci = warp_id; ci < col_width; ci += N_WARPS) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        const float x_val = (lane < z_col) ? X[x_base + lane] : 0.0f;
        float xsum = warp_reduce_sum(x_val);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];
            float t_sum = 0.0f;

            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;
                uint8_t zb = (lane < z_col)
                    ? Z[(size_t)B_idx * z_col * k8 + lane * k8 + bk] : 0;
                uint8_t yb[N_I_TILES];
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    int i = it * 32 + lane;
                    yb[it] = (i < y_row)
                        ? Y[(size_t)B_idx * y_row * k8 + i * k8 + bk] : 0;
                }
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const int shift = 7 - bit;
                    float t_k = warp_reduce_sum(
                        ((zb >> shift) & 1) ? x_val : 0.0f);
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;
                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((yb[it] >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }
            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }
        const float d_term = d_ptr[rc] * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

    /* ── cross-warp reduction ────────────────────────────────── */
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        warp_acc[warp_id * 32 + lane] = acc[it];
        __syncthreads();

        if (warp_id == 0) {
            float total = warp_acc[lane];          /* warp 0 */
            #pragma unroll
            for (int w = 1; w < N_WARPS; w++)
                total += warp_acc[w * 32 + lane];
            int i = it * 32 + lane;
            if (i < y_row)
                out[(size_t)n * row_width * y_row + r * y_row + i] = total;
        }
        __syncthreads();
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * v4: popcount + multi-warp i-split  — minimal shuffles
 * ═══════════════════════════════════════════════════════════════════
 *
 * Combines AND+popcount with multi-warp for occupancy.
 *
 *   Thread = j (z_col direction) for warp reduction
 *   Warp  w handles i = w*IPW .. (w+1)*IPW-1  (y_row split)
 *   → y_row / N_WARPS reductions per warp  (vs y_row in v1)
 *   → NO cross-warp reduction needed (each warp writes independent i rows)
 *
 * Per (p, c, i) iteration:
 *   1. Y bytes broadcast-loaded (all threads read same address → L1 hit)
 *   2. inner = Σ_bk popc(Y[i,bk] AND Z[j=lane,bk])  — k8 AND+popc ops
 *   3. core_i = warp_reduce(X[j] * inner)             — ONE reduction
 *
 * Block: N_WARPS × 32 threads    Grid: (row_width, batch)
 */

template <int N_WARPS, int IPW_MAX, int K8_MAX>
__global__ void bqq_forward_popcount_multiw_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const float*   __restrict__ X,
    const float*   __restrict__ a_ptr,
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int r       = blockIdx.x;
    const int n       = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;     /* = j index for z_col */

    /* i-range for this warp */
    const int ipw     = (y_row + N_WARPS - 1) / N_WARPS;
    const int i_start = warp_id * ipw;
    const int i_end   = min(i_start + ipw, y_row);
    const int my_cnt  = i_end - i_start;

    float acc[IPW_MAX];
    #pragma unroll
    for (int ii = 0; ii < IPW_MAX; ii++) acc[ii] = 0.0f;

    for (int ci = 0; ci < col_width; ci++) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        /* X[n, c, j=lane] — private per thread */
        const float x_val = (lane < z_col) ? X[x_base + lane] : 0.0f;

        float xsum = warp_reduce_sum(x_val);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];

            /* Z bytes for j=lane — private, reused across all i */
            uint8_t zb[K8_MAX];
            int z_sum_j = 0;
            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk < k8 && lane < z_col) {
                    zb[bk] = Z[(size_t)B_idx * z_col * k8 + lane * k8 + bk];
                    z_sum_j += __popc(static_cast<unsigned>(zb[bk]));
                } else {
                    zb[bk] = 0;
                }
            }

            /* z_dot_x = Σ_j X[j]·Zsum[j] — ONE reduction per (p,c) */
            float z_dot_x = warp_reduce_sum(x_val * static_cast<float>(z_sum_j));
            z_dot_x = __shfl_sync(0xffffffff, z_dot_x, 0);

            /* ── i-loop: only this warp's i-range ────────────── */
            for (int ii = 0; ii < my_cnt; ii++) {
                const int i = i_start + ii;

                /* Y bytes — broadcast (all threads read same addr → L1 hit) */
                int y_sum_i = 0;
                int inner   = 0;
                #pragma unroll
                for (int bk = 0; bk < K8_MAX; bk++) {
                    if (bk < k8) {
                        uint8_t yv = Y[(size_t)B_idx * y_row * k8 + i * k8 + bk];
                        y_sum_i += __popc(static_cast<unsigned>(yv));
                        inner   += __popc(static_cast<unsigned>(yv & zb[bk]));
                    }
                }

                /* core_i = Σ_j X[j] · popc(Y[i]&Z[j]) — ONE reduction */
                float core_i = warp_reduce_sum(x_val * static_cast<float>(inner));
                core_i = __shfl_sync(0xffffffff, core_i, 0);

                acc[ii] += a_val * core_i
                         + b_val * xsum * static_cast<float>(y_sum_i)
                         + c_val * z_dot_x;
            }
        }

        const float d_term = d_ptr[rc] * xsum;
        for (int ii = 0; ii < my_cnt; ii++) acc[ii] += d_term;
    }

    /* store — all threads have identical acc[], lane 0 writes */
    if (lane == 0) {
        for (int ii = 0; ii < my_cnt; ii++) {
            int i = i_start + ii;
            out[(size_t)n * row_width * y_row + r * y_row + i] = acc[ii];
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * v5: grid-split + uint32 bulk loads + preloaded registers
 * ═══════════════════════════════════════════════════════════════════
 *
 * Three orthogonal optimisations for bandwidth utilisation:
 *
 *   1. Grid splitting over col_width → col_splits × more blocks
 *      → warps/SM jumps from 3.4 to 20+
 *      → memory latency hidden by interleaving warp execution
 *      → atomicAdd to accumulate partial results
 *
 *   2. uint32 bulk loads for Z/Y bytes
 *      → k8=4 bytes loaded in 1 instruction instead of 4
 *      → 32 threads × 4 bytes = 128 bytes = 1 cache line (perfect coalescing)
 *
 *   3. All Z/Y data preloaded to registers before bit loop
 *      → hardware pipelines loads while compute runs
 *      → no load-use stalls in the inner loop
 *
 * Grid: (row_width × col_splits, batch)
 * Block: N_WARPS × 32 threads
 * Output: pre-zeroed, accumulated via atomicAdd
 */

template <int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_v5_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const float*   __restrict__ X,
    const float*   __restrict__ a_ptr,
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,
    float*         __restrict__ out,        /* PRE-ZEROED */
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits)
{
    const int combined = blockIdx.x;
    const int r  = combined / col_splits;
    const int cs = combined % col_splits;
    const int n  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    /* c-range for this block */
    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    /* distribute c within block across warps */
    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        const float x_val = (lane < z_col) ? X[x_base + lane] : 0.0f;
        float xsum = warp_reduce_sum(x_val);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];
            float t_sum = 0.0f;

            /* ── uint32 bulk load: Z[B_idx, lane, 0..k8-1] ────── */
            /*    k8 contiguous bytes per thread → k8/4 uint32's   */
            /*    32 threads × k8 bytes = coalesced cache line(s)  */
            constexpr int N_WORDS = (K8_MAX + 3) / 4;
            uint32_t z_words[N_WORDS];
            if (lane < z_col) {
                auto zp = reinterpret_cast<const uint32_t*>(
                    Z + (size_t)B_idx * z_col * k8 + lane * k8);
                #pragma unroll
                for (int w = 0; w < N_WORDS; w++)
                    z_words[w] = (w * 4 < k8) ? zp[w] : 0;
            } else {
                #pragma unroll
                for (int w = 0; w < N_WORDS; w++) z_words[w] = 0;
            }

            /* ── uint32 bulk load: Y[B_idx, i_tile, 0..k8-1] ──── */
            uint32_t y_words[N_I_TILES][N_WORDS];
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) {
                int i = it * 32 + lane;
                if (i < y_row) {
                    auto yp = reinterpret_cast<const uint32_t*>(
                        Y + (size_t)B_idx * y_row * k8 + i * k8);
                    #pragma unroll
                    for (int w = 0; w < N_WORDS; w++)
                        y_words[it][w] = (w * 4 < k8) ? yp[w] : 0;
                } else {
                    #pragma unroll
                    for (int w = 0; w < N_WORDS; w++) y_words[it][w] = 0;
                }
            }

            /* ── inner loop: all data in registers ─────────────── */
            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;
                const int w_idx = bk >> 2;           /* bk / 4 */
                const int b_shift = (bk & 3) << 3;   /* (bk % 4) * 8 */
                const uint8_t zb = (z_words[w_idx] >> b_shift) & 0xFF;

                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const int shift = 7 - bit;
                    float t_k = warp_reduce_sum(
                        ((zb >> shift) & 1) ? x_val : 0.0f);
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;

                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        const uint8_t yb =
                            (y_words[it][w_idx] >> b_shift) & 0xFF;
                        if ((yb >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }

            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }

        const float d_term = d_ptr[rc] * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

    /* ── cross-warp reduction + atomicAdd ────────────────────── */
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        warp_acc[warp_id * 32 + lane] = acc[it];
        __syncthreads();
        if (warp_id == 0) {
            float total = warp_acc[lane];
            #pragma unroll
            for (int w = 1; w < N_WARPS; w++)
                total += warp_acc[w * 32 + lane];
            int i = it * 32 + lane;
            if (i < y_row) {
                size_t idx = (size_t)n * row_width * y_row + r * y_row + i;
                if (col_splits > 1)
                    atomicAdd(&out[idx], total);
                else
                    out[idx] = total;   /* no atomics needed for single split */
            }
        }
        __syncthreads();
    }
}


/* ═══════════════════════════════════════════════════════════════════
 * dispatch
 * ═══════════════════════════════════════════════════════════════════ */

torch::Tensor bqq_forward_cuda(
    torch::Tensor Y_packed,
    torch::Tensor Z_packed,
    torch::Tensor X,
    torch::Tensor a, torch::Tensor b,
    torch::Tensor c, torch::Tensor d,
    int batch, int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int mode)     /* 0=auto, 1=shfl, 2=popc-1w, 3=mw-shfl, 4=mw-popc, 5=grid-split */
{
    auto out = torch::empty({batch, row_width, y_row},
        torch::dtype(torch::kFloat32).device(X.device()));

    auto yp = Y_packed.data_ptr<uint8_t>();
    auto zp = Z_packed.data_ptr<uint8_t>();
    auto xp = X.data_ptr<float>();
    auto ap = a.data_ptr<float>();
    auto bp = b.data_ptr<float>();
    auto cp = c.data_ptr<float>();
    auto dp = d.data_ptr<float>();
    auto op = out.data_ptr<float>();

    int nj = (z_col + 31) / 32;
    int ni = (y_row + 31) / 32;

    if (mode == 0) mode = 5;   /* default = grid-split + uint32 + preload */

    if (mode == 5) {
        /* ── v5: grid-split + uint32 bulk loads ────────────────── */
        constexpr int NW = 4;
        /* auto-tune col_splits for max occupancy (~48 warps/SM) */
        int sm_count = 56;  /* TODO: query at runtime */
        int col_splits = max(1, (48 * sm_count + row_width * NW - 1)
                                / (row_width * NW));
        col_splits = min(col_splits, col_width);
        /* prefer divisors of col_width for even distribution */
        while (col_splits > 1 && col_width % col_splits != 0)
            col_splits--;

        /* pre-zero output (atomicAdd accumulates into it) */
        out = torch::zeros({batch, row_width, y_row},
            torch::dtype(torch::kFloat32).device(X.device()));
        op = out.data_ptr<float>();   /* update pointer! */

        dim3 grid(row_width * col_splits, batch);
        dim3 block(NW * 32);
        int smem = NW * 32 * sizeof(float);

        #define LAUNCH_V5(NI, K8M) \
            bqq_forward_v5_kernel<NW, NI, K8M><<<grid, block, smem>>>( \
                yp, zp, xp, ap, bp, cp, dp, op, \
                row_width, col_width, bit_width, y_row, z_col, k8, \
                col_splits)

        if (ni == 1) {
            if      (k8 <= 4)  LAUNCH_V5(1, 4);
            else if (k8 <= 8)  LAUNCH_V5(1, 8);
            else               LAUNCH_V5(1, 16);
        } else if (ni <= 2) {
            if      (k8 <= 8)  LAUNCH_V5(2, 8);
            else               LAUNCH_V5(2, 16);
        } else {
            LAUNCH_V5(4, 16);
        }
        #undef LAUNCH_V5

    } else

    if (mode == 4) {
        /* ── popcount + multi-warp i-split ─────────────────────── */
        constexpr int NW = 8;
        dim3 grid(row_width, batch);
        dim3 block(NW * 32);
        int ipw = (y_row + NW - 1) / NW;  /* i per warp */

        #define LAUNCH_V4(IPW, K8M) \
            bqq_forward_popcount_multiw_kernel<NW, IPW, K8M> \
                <<<grid, block>>>(yp, zp, xp, ap, bp, cp, dp, op, \
                    row_width, col_width, bit_width, y_row, z_col, k8)

        if (ipw <= 4) {
            if      (k8 <= 4)  LAUNCH_V4(4, 4);
            else if (k8 <= 8)  LAUNCH_V4(4, 8);
            else               LAUNCH_V4(4, 16);
        } else if (ipw <= 8) {
            if      (k8 <= 4)  LAUNCH_V4(8, 4);
            else if (k8 <= 8)  LAUNCH_V4(8, 8);
            else               LAUNCH_V4(8, 16);
        } else if (ipw <= 16) {
            if      (k8 <= 8)  LAUNCH_V4(16, 8);
            else               LAUNCH_V4(16, 16);
        } else {
            LAUNCH_V4(32, 16);
        }
        #undef LAUNCH_V4

    } else if (mode == 3) {
        /* ── multi-warp kernel (N_WARPS=4 → 128 threads/block) ── */
        constexpr int NW = 4;
        dim3 grid(row_width, batch);
        dim3 block(NW * 32);
        int smem = NW * 32 * sizeof(float);

        #define LAUNCH_MW(NI, K8M) \
            bqq_forward_multiw_kernel<NW, NI, K8M><<<grid, block, smem>>>( \
                yp, zp, xp, ap, bp, cp, dp, op, \
                row_width, col_width, bit_width, y_row, z_col, k8)

        if (ni == 1) {
            if      (k8 <= 4)  LAUNCH_MW(1, 4);
            else if (k8 <= 8)  LAUNCH_MW(1, 8);
            else               LAUNCH_MW(1, 16);
        } else if (ni <= 2) {
            if      (k8 <= 8)  LAUNCH_MW(2, 8);
            else               LAUNCH_MW(2, 16);
        } else {
            LAUNCH_MW(4, 16);
        }
        #undef LAUNCH_MW

    } else if (mode == 2) {
        /* ── popcount kernel ───────────────────────────────────── */
        dim3 grid(row_width, batch);
        dim3 block(32);
        int smem = z_col * sizeof(float) + z_col * k8;

        #define LAUNCH_PC(NI, K8M) \
            bqq_forward_popcount_kernel<NI, K8M><<<grid, block, smem>>>( \
                yp, zp, xp, ap, bp, cp, dp, op, \
                row_width, col_width, bit_width, y_row, z_col, k8)

        if (ni == 1) {
            if      (k8 <= 4)  LAUNCH_PC(1, 4);
            else if (k8 <= 8)  LAUNCH_PC(1, 8);
            else               LAUNCH_PC(1, 16);
        } else if (ni <= 2) {
            if      (k8 <= 8)  LAUNCH_PC(2, 8);
            else               LAUNCH_PC(2, 16);
        } else {
            LAUNCH_PC(4, 16);
        }
        #undef LAUNCH_PC

    } else {
        /* ── warp-shuffle kernel (mode=1) ──────────────────────── */
        dim3 grid(row_width, batch);
        dim3 block(32);
        int smem = z_col * sizeof(float);

        #define LAUNCH_WS(NJ, NI, K8M) \
            bqq_forward_warp_kernel<NJ, NI, K8M><<<grid, block, smem>>>( \
                yp, zp, xp, ap, bp, cp, dp, op, \
                row_width, col_width, bit_width, y_row, z_col, k8)

        if (nj == 1 && ni == 1) {
            if      (k8 <= 4)  LAUNCH_WS(1, 1, 4);
            else if (k8 <= 8)  LAUNCH_WS(1, 1, 8);
            else               LAUNCH_WS(1, 1, 16);
        } else if (nj <= 2 && ni <= 2) {
            if      (k8 <= 8)  LAUNCH_WS(2, 2, 8);
            else               LAUNCH_WS(2, 2, 16);
        } else {
            LAUNCH_WS(4, 4, 16);
        }
        #undef LAUNCH_WS
    }

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bqq_forward_cuda", &bqq_forward_cuda,
          "BQQ fused forward CUDA kernel",
          py::arg("Y_packed"), py::arg("Z_packed"), py::arg("X"),
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
          py::arg("batch"), py::arg("row_width"), py::arg("col_width"),
          py::arg("bit_width"), py::arg("y_row"), py::arg("z_col"),
          py::arg("k8"), py::arg("mode") = 0);
}
