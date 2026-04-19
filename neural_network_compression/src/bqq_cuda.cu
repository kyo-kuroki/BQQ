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
    int mode)     /* 0 = auto, 1 = warp-shuffle, 2 = popcount, 3 = multi-warp */
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

    if (mode == 0) mode = 3;   /* default = multi-warp */

    if (mode == 3) {
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
