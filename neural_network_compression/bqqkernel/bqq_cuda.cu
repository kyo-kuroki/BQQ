/*
 * BQQ fused forward CUDA kernel.
 *
 * Computes  out = X @ W.T  where
 *   W = Σ_p (a_p·Y_p@Z_p + b_p·Ysum_p + c_p·Zsum_p) + d
 * without materialising W.
 *
 * Optimisations:
 *   1. Warp-level conditional select + __shfl_down reduction for Z@X
 *      (no bit→float multiply; branch instead of 0/1 multiply for Y)
 *   2. Grid splitting over col_width for high SM occupancy (~72 warps/SM)
 *   3. uint32 bulk loads for Z/Y bytes (4x fewer load instructions)
 *   4. Z/Y preloaded to registers before the 8-bit inner loop
 *   5. b term folded: t_aug = a*t_k + b*xsum (merges terms 1+2)
 *   6. c term free:   sum(t) = Zsum@x (no extra memory load)
 *
 * Thread mapping (one warp = 32 threads):
 *   Phase 1 — thread j holds X[j], reduces Z_bool[k,j]*X[j] via __shfl_down
 *   Phase 2 — thread i checks Y_bool[i,k], conditionally adds t_aug
 *
 * Auto-selection:  seq_len <= 32 → this kernel
 *                  seq_len > 32  → W-reconstruction + cuBLAS (in Python)
 *
 * Grid: (row_width × col_splits, batch)    Block: N_WARPS × 32 threads
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>


/* ── warp-level sum ────────────────────────────────────────────── */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__device__ __forceinline__ float load_activation(const T* ptr, size_t idx);

template <>
__device__ __forceinline__ float load_activation<float>(const float* ptr, size_t idx) {
    return __ldg(&ptr[idx]);
}

template <>
__device__ __forceinline__ float load_activation<__half>(const __half* ptr, size_t idx) {
    return __half2float(__ldg(&ptr[idx]));
}

template <>
__device__ __forceinline__ float load_activation<__nv_bfloat16>(
    const __nv_bfloat16* ptr, size_t idx) {
    return __bfloat162float(__ldg(&ptr[idx]));
}

/* fp32 → OUT_T store used by the fused decode epilogue. */
template <typename T>
__device__ __forceinline__ T store_activation(float v);

template <>
__device__ __forceinline__ __half store_activation<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_activation<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}


/* ═══════════════════════════════════════════════════════════════════
 * Main kernel: grid-split + uint32 bulk loads
 * ═══════════════════════════════════════════════════════════════════ */

template <typename X_T, int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
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

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        const float x_val = (lane < z_col) ? load_activation(X, x_base + lane) : 0.0f;
        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += load_activation(X, x_base + j);
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);
            float t_sum = 0.0f;

            /* uint32 bulk load: Z/Y bytes → registers */
            constexpr int N_WORDS = (K8_MAX + 3) / 4;
            uint32_t z_words[N_WORDS];
            #pragma unroll
            for (int w = 0; w < N_WORDS; w++) z_words[w] = 0;

            /* inner loop: all data in registers */
            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const int shift = 7 - bit;
                    float t_local = 0.0f;
                    if (z_col <= 32) {
                        const uint8_t zb = Z[(size_t)B_idx * z_col * k8 + lane * k8 + bk];
                        t_local = ((zb >> shift) & 1) ? x_val : 0.0f;
                    } else {
                        for (int j = lane; j < z_col; j += 32) {
                            const uint8_t zj =
                                Z[(size_t)B_idx * z_col * k8 + j * k8 + bk];
                            if ((zj >> shift) & 1) {
                                t_local += load_activation(X, x_base + j);
                            }
                        }
                    }
                    float t_k = warp_reduce_sum(t_local);
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;

                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        const int i = it * 32 + lane;
                        const uint8_t yb = (i < y_row)
                            ? Y[(size_t)B_idx * y_row * k8 + i * k8 + bk]
                            : 0;
                        if ((yb >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }

            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

    /* cross-warp reduction + atomicAdd */
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
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}

/* ────────────────────────────────────────────────────────────────────────
 * lane-per-rank kernel  (requires inter_dim <= 32, i.e. k8 <= 4)
 *
 * The other decode kernels map lanes to j (the z_col axis), so every rank
 * accumulator ends up spread across the warp and needs a cross-lane reduction:
 * the byte4 kernel pays 5 shuffle levels x 32 bit planes = 160 __shfl_down
 * plus 32 broadcasts per (column patch, bit) -- ~3x more shuffle traffic than
 * useful adds, which is what pins those kernels ~5x above the memory-bandwidth
 * floor.
 *
 * Here lane k owns rank k outright, so Z@x needs *no* cross-lane reduction at
 * all.  Only two warp reductions remain per (patch, bit): xsum and t_sum.
 *
 *   Phase 1 (Z@x):  t_k = sum_j Z[k,j] * x[j]      -- lane k, zero shuffles
 *   Phase 2 (Y@t):  out_i = sum_k Y[i,k] * t_aug_k -- lane i, t_aug via shared
 *
 * Z and x are staged through shared memory so the global reads stay coalesced
 * (lane k would otherwise re-read the same Z byte as 7 other lanes).
 * ──────────────────────────────────────────────────────────────────────── */
template <typename X_T, int N_WARPS, int N_I_TILES>
__global__ void bqq_forward_lanerank_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits)
{
    constexpr int Z_COL_MAX = 128;

    const int combined = blockIdx.x;
    const int r  = combined / col_splits;
    const int cs = combined % col_splits;
    const int n  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    const int n_inner  = k8 * 8;          /* padded rank, <= 32 */
    const bool active  = (lane < n_inner);
    const int byte_k   = lane >> 3;       /* Z/Y byte holding rank `lane` */
    const int shift_k  = 7 - (lane & 7);

    __shared__ uint8_t z_sh[N_WARPS][Z_COL_MAX * 4];
    __shared__ float   x_sh[N_WARPS][Z_COL_MAX];
    __shared__ float   t_sh[N_WARPS][32];
    __shared__ float   warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        /* stage x once per column patch (reused by every bit plane) */
        for (int j = lane; j < z_col; j += 32)
            x_sh[warp_id][j] = load_activation(X, x_base + j);
        __syncwarp();

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) xsum_local += x_sh[warp_id][j];
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);

            /* coalesced load of this bit plane's Z block into shared */
            const uint8_t* Zb = Z + (size_t)B_idx * z_col * k8;
            const int z_bytes = z_col * k8;
            for (int idx = lane; idx < z_bytes; idx += 32)
                z_sh[warp_id][idx] = Zb[idx];
            __syncwarp();

            /* Phase 1: lane k accumulates rank k — no cross-lane reduction */
            float t_k = 0.0f;
            if (active) {
                for (int j = 0; j < z_col; j++) {
                    const uint8_t zb = z_sh[warp_id][j * k8 + byte_k];
                    if ((zb >> shift_k) & 1) t_k += x_sh[warp_id][j];
                }
            }
            const float t_aug = a_val * t_k + b_val * xsum;
            t_sh[warp_id][lane] = active ? t_aug : 0.0f;

            /* the only remaining reduction: t_sum for the c term */
            float t_sum = warp_reduce_sum(active ? t_k : 0.0f);
            t_sum = __shfl_sync(0xffffffff, t_sum, 0);
            __syncwarp();

            /* Phase 2: lane owns output rows i = it*32 + lane */
            const uint8_t* Yb = Y + (size_t)B_idx * y_row * k8;
            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) {
                const int i = it * 32 + lane;
                if (i >= y_row) continue;
                const uint8_t* yrow = Yb + (size_t)i * k8;
                /* rank kk lives in byte kk>>3, bit 7-(kk&7) */
                uint32_t ymask = 0;
                #pragma unroll
                for (int u = 0; u < 4; u++)
                    if (u < k8) ymask |= ((uint32_t)yrow[u]) << (24 - 8 * u);
                float s = 0.0f;
                for (int kk = 0; kk < n_inner; kk++)
                    if ((ymask >> (31 - kk)) & 1) s += t_sh[warp_id][kk];
                acc[it] += s + c_term;
            }
            __syncwarp();
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
        __syncwarp();
    }

    /* cross-warp reduction + atomicAdd (same epilogue as the other kernels) */
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
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}


/* ────────────────────────────────────────────────────────────────────────
 * nibble-LUT kernel  (requires inter_dim <= 32, z_col % 8 == 0, z_col <= 128)
 *
 * The other kernels spend ~3 instructions per weight (shift + and + predicated
 * add), twice over for a 2-bit model; that instruction count -- not bandwidth
 * (only 17% of peak is reached) and not the shuffles -- is what holds them ~6x
 * above the memory-bandwidth floor.
 *
 * Here the partial sums of x over each 4-element group are precomputed once per
 * column patch (16 masks x z_col/4 groups), so a 4-bit slice of Z becomes a
 * single table lookup:
 *
 *     t_k = sum_g  xlut[g][ Z[k, 4g..4g+3] ]      (z_col adds  ->  z_col/4 lookups)
 *     o_i = sum_g  tlut[g][ Y[i, 4g..4g+3] ]      (rank  adds  ->  rank/4  lookups)
 *
 * Both LUTs are reused across all ranks / all output rows of the patch, and the
 * lookups are independent, so the dependent-add chain that pinned the
 * lane-per-rank kernel disappears too (4 accumulators => ILP 4).
 *
 * Z must arrive rank-major (packed along j): [B, n_inner, z_col/8].  The Python
 * flat cache builds that layout when BQQ_CUDA_DECODE_KERNEL=lut, exactly like
 * the bitplane path swaps in its own layout.  Y keeps its usual [B, y_row, k8]
 * (already rank-major per row, which is what the phase-2 LUT wants).
 *
 * Bit convention (both LUTs): within a nibble, bit (3-m) selects element 4g+m,
 * matching the MSB-first bit packing.
 * ──────────────────────────────────────────────────────────────────────── */
template <typename X_T, int N_WARPS, int N_I_TILES>
__global__ void bqq_forward_lut_kernel(
    const uint8_t* __restrict__ Y,   /* [B, y_row, k8]        rank-major bits */
    const uint8_t* __restrict__ Zt,  /* [B, n_inner, z_col/8] j-major bits    */
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits)
{
    constexpr int Z_COL_MAX  = 128;
    constexpr int XLUT_MAX   = (Z_COL_MAX / 4) * 16;   /* 512 floats */
    constexpr int TLUT_MAX   = (32 / 4) * 16;          /* 128 floats */

    const int combined = blockIdx.x;
    const int r  = combined / col_splits;
    const int cs = combined % col_splits;
    const int n  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int c_per_split   = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    const int n_inner  = k8 * 8;          /* padded rank, <= 32 */
    const int zc8      = z_col >> 3;      /* bytes per Zt row   */
    const int n_grp_x  = z_col >> 2;      /* x nibble groups    */
    const int n_grp_t  = n_inner >> 2;    /* rank nibble groups */
    const bool active  = (lane < n_inner);

    __shared__ float x_sh   [N_WARPS][Z_COL_MAX];
    __shared__ float xlut   [N_WARPS][XLUT_MAX];
    __shared__ float t_sh   [N_WARPS][32];
    __shared__ float tlut   [N_WARPS][TLUT_MAX];
    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        for (int j = lane; j < z_col; j += 32)
            x_sh[warp_id][j] = load_activation(X, x_base + j);
        __syncwarp();

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) xsum_local += x_sh[warp_id][j];
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        /* build the x nibble LUT once per column patch (reused by every bit) */
        for (int g = lane; g < n_grp_x; g += 32) {
            const float* xg = &x_sh[warp_id][g << 2];
            float* L = &xlut[warp_id][g << 4];
            const float x0 = xg[0], x1 = xg[1], x2 = xg[2], x3 = xg[3];
            L[0]  = 0.0f;          L[1]  = x3;            L[2]  = x2;
            L[3]  = x2 + x3;       L[4]  = x1;            L[5]  = x1 + x3;
            L[6]  = x1 + x2;       L[7]  = x1 + x2 + x3;  L[8]  = x0;
            L[9]  = x0 + x3;       L[10] = x0 + x2;       L[11] = x0 + x2 + x3;
            L[12] = x0 + x1;       L[13] = x0 + x1 + x3;  L[14] = x0 + x1 + x2;
            L[15] = x0 + x1 + x2 + x3;
        }
        __syncwarp();

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);

            /* ---- Phase 1: lane k owns rank k; z_col/4 independent lookups ----
             * The whole Zt row (zc8 bytes) is pulled in as uint32 words in one
             * shot: reading it a byte at a time makes the 32 lanes stride by
             * zc8, which over-fetches ~8x.  Every nibble is then extracted from
             * registers. */
            float t_k = 0.0f;
            if (active) {
                const uint8_t* zrow = Zt + (size_t)B_idx * n_inner * zc8
                                         + (size_t)lane * zc8;
                uint32_t zw[4];
                const int nwords = zc8 >> 2;
                const uint32_t* zrow32 = reinterpret_cast<const uint32_t*>(zrow);
                #pragma unroll
                for (int w = 0; w < 4; w++) zw[w] = (w < nwords) ? zrow32[w] : 0u;

                float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
                int g = 0;
                for (; g + 3 < n_grp_x; g += 4) {
                    const int b0 = (g + 0) >> 1, b1 = (g + 1) >> 1;
                    const int b2 = (g + 2) >> 1, b3 = (g + 3) >> 1;
                    const uint32_t v0 = (zw[b0 >> 2] >> (8 * (b0 & 3))) & 0xFFu;
                    const uint32_t v1 = (zw[b1 >> 2] >> (8 * (b1 & 3))) & 0xFFu;
                    const uint32_t v2 = (zw[b2 >> 2] >> (8 * (b2 & 3))) & 0xFFu;
                    const uint32_t v3 = (zw[b3 >> 2] >> (8 * (b3 & 3))) & 0xFFu;
                    const int n0 = ((g + 0) & 1) ? (int)(v0 & 0xFu) : (int)(v0 >> 4);
                    const int n1 = ((g + 1) & 1) ? (int)(v1 & 0xFu) : (int)(v1 >> 4);
                    const int n2 = ((g + 2) & 1) ? (int)(v2 & 0xFu) : (int)(v2 >> 4);
                    const int n3 = ((g + 3) & 1) ? (int)(v3 & 0xFu) : (int)(v3 >> 4);
                    t0 += xlut[warp_id][((g + 0) << 4) + n0];
                    t1 += xlut[warp_id][((g + 1) << 4) + n1];
                    t2 += xlut[warp_id][((g + 2) << 4) + n2];
                    t3 += xlut[warp_id][((g + 3) << 4) + n3];
                }
                for (; g < n_grp_x; g++) {
                    const int bb = g >> 1;
                    const uint32_t vv = (zw[bb >> 2] >> (8 * (bb & 3))) & 0xFFu;
                    const int nb = (g & 1) ? (int)(vv & 0xFu) : (int)(vv >> 4);
                    t0 += xlut[warp_id][(g << 4) + nb];
                }
                t_k = (t0 + t1) + (t2 + t3);
            }
            const float t_aug = a_val * t_k + b_val * xsum;
            t_sh[warp_id][lane] = active ? t_aug : 0.0f;

            float t_sum = warp_reduce_sum(active ? t_k : 0.0f);
            t_sum = __shfl_sync(0xffffffff, t_sum, 0);
            __syncwarp();

            /* build the t_aug nibble LUT for this bit plane */
            for (int g = lane; g < n_grp_t; g += 32) {
                const float* tg = &t_sh[warp_id][g << 2];
                float* L = &tlut[warp_id][g << 4];
                const float t0 = tg[0], t1 = tg[1], t2 = tg[2], t3 = tg[3];
                L[0]  = 0.0f;          L[1]  = t3;            L[2]  = t2;
                L[3]  = t2 + t3;       L[4]  = t1;            L[5]  = t1 + t3;
                L[6]  = t1 + t2;       L[7]  = t1 + t2 + t3;  L[8]  = t0;
                L[9]  = t0 + t3;       L[10] = t0 + t2;       L[11] = t0 + t2 + t3;
                L[12] = t0 + t1;       L[13] = t0 + t1 + t3;  L[14] = t0 + t1 + t2;
                L[15] = t0 + t1 + t2 + t3;
            }
            __syncwarp();

            /* ---- Phase 2: lane owns rows i = it*32 + lane; rank/4 lookups ---- */
            const uint8_t* Yb = Y + (size_t)B_idx * y_row * k8;
            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) {
                const int i = it * 32 + lane;
                if (i >= y_row) continue;
                const uint8_t* yrow = Yb + (size_t)i * k8;
                /* one 32-bit load instead of 4 byte loads; __byte_perm puts
                 * byte 0 (ranks 0-7) back in the MSBs after the LE load. */
                uint32_t ymask;
                if (k8 == 4) {
                    ymask = __byte_perm(
                        *reinterpret_cast<const uint32_t*>(yrow), 0u, 0x0123);
                } else {
                    ymask = 0;
                    #pragma unroll
                    for (int u = 0; u < 4; u++)
                        if (u < k8) ymask |= ((uint32_t)yrow[u]) << (24 - 8 * u);
                }
                float s0 = 0.0f, s1 = 0.0f;
                int g = 0;
                for (; g + 1 < n_grp_t; g += 2) {
                    const int nb0 = (ymask >> (28 - 4 * (g + 0))) & 0xF;
                    const int nb1 = (ymask >> (28 - 4 * (g + 1))) & 0xF;
                    s0 += tlut[warp_id][((g + 0) << 4) + nb0];
                    s1 += tlut[warp_id][((g + 1) << 4) + nb1];
                }
                for (; g < n_grp_t; g++) {
                    const int nb = (ymask >> (28 - 4 * g)) & 0xF;
                    s0 += tlut[warp_id][(g << 4) + nb];
                }
                acc[it] += (s0 + s1) + c_term;
            }
            __syncwarp();
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
        __syncwarp();
    }

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
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}


/* ────────────────────────────────────────────────────────────────────────
 * 8-bit-LUT + row-tiled kernel  (inter_dim <= 32, z_col % 8 == 0, z_col <= 128)
 *
 * Two changes over bqq_forward_lut_kernel:
 *
 *  1. The x partial-sum table is built over 8-element groups, so a whole Zt byte
 *     *is* the table index -- no nibble extraction at all, and phase 1 costs
 *     z_col/8 lookups instead of z_col/4.  The 256-entry tables are filled with a
 *     doubling DP (255 adds per group, spread over the block) rather than
 *     2^8 x 8 conditional adds.
 *
 *  2. That bigger table only pays for itself if it is reused, and it depends on
 *     the *column* patch alone -- yet the 4-bit kernel rebuilds it once per row
 *     patch (row_width times over).  So a block now owns N_WARPS *row* patches
 *     (warp w -> row patch r_base + w) and walks the column range once, building
 *     each x table a single time for all of them.
 *
 *     Per unit of work that turns  build 120 + lookups 512  into
 *     build 128 + lookups 256 (~40% fewer ops at N_WARPS = 8).
 *
 * Because each warp owns a distinct row patch, the cross-warp reduction in the
 * epilogue disappears too -- a warp's accumulator is already the full sum for
 * its rows over this block's columns.
 *
 * Phase 2 keeps the 4-bit table: t_aug changes per (row patch, bit), so a
 * 256-entry version would spend more on the build than it saves on lookups.
 * ──────────────────────────────────────────────────────────────────────── */
template <typename X_T, int N_WARPS, int N_I_TILES>
__global__ void bqq_forward_lut8_kernel(
    const uint8_t* __restrict__ Y,   /* [B, y_row, k8]        rank-major bits */
    const uint8_t* __restrict__ Zt,  /* [B, n_inner, z_col/8] j-major bits    */
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits)
{
    constexpr int Z_COL_MAX = 128;
    constexpr int XLUT8_MAX = (Z_COL_MAX / 8) * 256;  /* 4096 floats = 16 KB */
    constexpr int TLUT_MAX  = (32 / 4) * 16;          /* 128 floats          */

    const int combined = blockIdx.x;
    const int rt = combined / col_splits;     /* row-patch tile */
    const int cs = combined % col_splits;
    const int n  = blockIdx.y;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int r       = rt * N_WARPS + warp_id;   /* this warp's row patch */
    const bool r_valid = (r < row_width);

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c0 = cs * c_per_split;
    const int c1 = min(c0 + c_per_split, col_width);

    const int n_inner = k8 * 8;
    const int zc8     = z_col >> 3;   /* bytes per Zt row == 8-bit groups */
    const int n_grp_t = n_inner >> 2;
    const bool active = (lane < n_inner);

    __shared__ float x_sh[Z_COL_MAX];
    __shared__ float xlut8[XLUT8_MAX];
    __shared__ float tlut[N_WARPS][TLUT_MAX];
    __shared__ float t_sh[N_WARPS][32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c0; ci < c1; ci++) {
        const int x_base = n * col_width * z_col + ci * z_col;

        for (int j = tid; j < z_col; j += blockDim.x)
            x_sh[j] = load_activation(X, x_base + j);
        __syncthreads();

        float xs_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) xs_local += x_sh[j];
        float xsum = warp_reduce_sum(xs_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        /* 8-bit x table, doubling DP: after bit b, indices [0,2^b) are filled.
         * Index bit b selects element (7-b), matching the MSB-first packing. */
        for (int g = tid; g < zc8; g += blockDim.x) xlut8[g << 8] = 0.0f;
        __syncthreads();
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            const int half  = 1 << b;
            const int total = zc8 * half;
            const int melem = 7 - b;
            for (int t = tid; t < total; t += blockDim.x) {
                const int g = t >> b;
                const int i = t - (g << b);
                xlut8[(g << 8) + i + half] = xlut8[(g << 8) + i]
                                           + x_sh[(g << 3) + melem];
            }
            __syncthreads();
        }

        if (r_valid) {
            const int rc = r * col_width + ci;
            for (int p = 0; p < bit_width; p++) {
                const int B_idx = p * row_width * col_width + rc;
                const float a_val = __half2float(a_ptr[B_idx]);
                const float b_val = __half2float(b_ptr[B_idx]);
                const float c_val = __half2float(c_ptr[B_idx]);

                /* Phase 1: the Zt byte is the table index directly */
                float t_k = 0.0f;
                if (active) {
                    const uint8_t* zrow = Zt + (size_t)B_idx * n_inner * zc8
                                             + (size_t)lane * zc8;
                    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
                    int g = 0;
                    for (; g + 3 < zc8; g += 4) {
                        t0 += xlut8[((g + 0) << 8) + zrow[g + 0]];
                        t1 += xlut8[((g + 1) << 8) + zrow[g + 1]];
                        t2 += xlut8[((g + 2) << 8) + zrow[g + 2]];
                        t3 += xlut8[((g + 3) << 8) + zrow[g + 3]];
                    }
                    for (; g < zc8; g++)
                        t0 += xlut8[(g << 8) + zrow[g]];
                    t_k = (t0 + t1) + (t2 + t3);
                }
                const float t_aug = a_val * t_k + b_val * xsum;
                t_sh[warp_id][lane] = active ? t_aug : 0.0f;

                float t_sum = warp_reduce_sum(active ? t_k : 0.0f);
                t_sum = __shfl_sync(0xffffffff, t_sum, 0);
                __syncwarp();

                for (int g = lane; g < n_grp_t; g += 32) {
                    const float* tg = &t_sh[warp_id][g << 2];
                    float* L = &tlut[warp_id][g << 4];
                    const float u0 = tg[0], u1 = tg[1], u2 = tg[2], u3 = tg[3];
                    L[0]  = 0.0f;          L[1]  = u3;            L[2]  = u2;
                    L[3]  = u2 + u3;       L[4]  = u1;            L[5]  = u1 + u3;
                    L[6]  = u1 + u2;       L[7]  = u1 + u2 + u3;  L[8]  = u0;
                    L[9]  = u0 + u3;       L[10] = u0 + u2;       L[11] = u0 + u2 + u3;
                    L[12] = u0 + u1;       L[13] = u0 + u1 + u3;  L[14] = u0 + u1 + u2;
                    L[15] = u0 + u1 + u2 + u3;
                }
                __syncwarp();

                /* Phase 2: rank nibbles of Y index the t table */
                const uint8_t* Yb = Y + (size_t)B_idx * y_row * k8;
                const float c_term = c_val * t_sum;
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    const int i = it * 32 + lane;
                    if (i >= y_row) continue;
                    const uint8_t* yrow = Yb + (size_t)i * k8;
                    uint32_t ymask;
                    if (k8 == 4) {
                        ymask = __byte_perm(
                            *reinterpret_cast<const uint32_t*>(yrow), 0u, 0x0123);
                    } else {
                        ymask = 0;
                        #pragma unroll
                        for (int u = 0; u < 4; u++)
                            if (u < k8) ymask |= ((uint32_t)yrow[u]) << (24 - 8 * u);
                    }
                    float s0 = 0.0f, s1 = 0.0f;
                    int g = 0;
                    for (; g + 1 < n_grp_t; g += 2) {
                        const int nb0 = (ymask >> (28 - 4 * (g + 0))) & 0xF;
                        const int nb1 = (ymask >> (28 - 4 * (g + 1))) & 0xF;
                        s0 += tlut[warp_id][((g + 0) << 4) + nb0];
                        s1 += tlut[warp_id][((g + 1) << 4) + nb1];
                    }
                    for (; g < n_grp_t; g++) {
                        const int nb = (ymask >> (28 - 4 * g)) & 0xF;
                        s0 += tlut[warp_id][(g << 4) + nb];
                    }
                    acc[it] += (s0 + s1) + c_term;
                }
                __syncwarp();
            }

            const float d_term = __half2float(d_ptr[rc]) * xsum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
        }
        __syncthreads();   /* x_sh / xlut8 are reused by the next column patch */
    }

    /* Each warp owns a distinct row patch -> no cross-warp reduction. */
    if (r_valid) {
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) {
            const int i = it * 32 + lane;
            if (i < y_row) {
                const size_t idx = (size_t)n * row_width * y_row + r * y_row + i;
                if (col_splits > 1) atomicAdd(&out[idx], acc[it]);
                else                out[idx] = acc[it];
            }
        }
    }
}


template <typename X_T, int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_byte_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
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

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += load_activation(X, x_base + j);
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);
            float t_sum = 0.0f;

            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;

                float t_local[8];
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) t_local[bit] = 0.0f;

                for (int j = lane; j < z_col; j += 32) {
                    const uint8_t zb =
                        Z[(size_t)B_idx * z_col * k8 + j * k8 + bk];
                    const float xv = load_activation(X, x_base + j);
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int shift = 7 - bit;
                        if ((zb >> shift) & 1) t_local[bit] += xv;
                    }
                }

                uint8_t y_bytes[N_I_TILES];
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    const int i = it * 32 + lane;
                    y_bytes[it] = (i < y_row)
                        ? Y[(size_t)B_idx * y_row * k8 + i * k8 + bk]
                        : 0;
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        t_local[bit] += __shfl_down_sync(
                            0xffffffff, t_local[bit], offset);
                    }
                }

                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    float t_k = t_local[bit];
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;
                    const int shift = 7 - bit;

                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((y_bytes[it] >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }

            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

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
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}

template <typename X_T, int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_byte2_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
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

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += load_activation(X, x_base + j);
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);
            float t_sum = 0.0f;

            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk += 2) {
                if (bk >= k8) break;

                float t_local[16];
                #pragma unroll
                for (int bit = 0; bit < 16; bit++) t_local[bit] = 0.0f;

                for (int j = lane; j < z_col; j += 32) {
                    const size_t z_base = (size_t)B_idx * z_col * k8 + j * k8 + bk;
                    const uint8_t z0 = Z[z_base];
                    const uint8_t z1 = (bk + 1 < k8) ? Z[z_base + 1] : 0;
                    const float xv = load_activation(X, x_base + j);
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int shift = 7 - bit;
                        if ((z0 >> shift) & 1) t_local[bit] += xv;
                        if ((z1 >> shift) & 1) t_local[bit + 8] += xv;
                    }
                }

                uint8_t y0[N_I_TILES];
                uint8_t y1[N_I_TILES];
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    const int i = it * 32 + lane;
                    if (i < y_row) {
                        const size_t y_base =
                            (size_t)B_idx * y_row * k8 + i * k8 + bk;
                        y0[it] = Y[y_base];
                        y1[it] = (bk + 1 < k8) ? Y[y_base + 1] : 0;
                    } else {
                        y0[it] = 0;
                        y1[it] = 0;
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    #pragma unroll
                    for (int bit = 0; bit < 16; bit++) {
                        t_local[bit] += __shfl_down_sync(
                            0xffffffff, t_local[bit], offset);
                    }
                }

                #pragma unroll
                for (int bit = 0; bit < 16; bit++) {
                    float t_k = t_local[bit];
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;
                    const int shift = 7 - (bit & 7);

                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        const uint8_t yb = (bit < 8) ? y0[it] : y1[it];
                        if ((yb >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }

            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

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
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}

template <typename X_T, int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_byte4_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits,
    bool interleave_bit_reductions)
{
    const int combined = blockIdx.x;
    const int r  = combined / col_splits;
    const int cs = combined % col_splits;
    const int n  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += load_activation(X, x_base + j);
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);
            float t_sum = 0.0f;

            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk += 4) {
                if (bk >= k8) break;

                float t_local[32];
                #pragma unroll
                for (int bit = 0; bit < 32; bit++) t_local[bit] = 0.0f;

                for (int j = lane; j < z_col; j += 32) {
                    const size_t z_base = (size_t)B_idx * z_col * k8 + j * k8 + bk;
                    uint8_t z_bytes[4];
                    #pragma unroll
                    for (int u = 0; u < 4; u++) {
                        z_bytes[u] = (bk + u < k8) ? Z[z_base + u] : 0;
                    }
                    const float xv = load_activation(X, x_base + j);
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int shift = 7 - bit;
                        #pragma unroll
                        for (int u = 0; u < 4; u++) {
                            if ((z_bytes[u] >> shift) & 1) {
                                t_local[u * 8 + bit] += xv;
                            }
                        }
                    }
                }

                uint8_t y_bytes[N_I_TILES][4];
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    const int i = it * 32 + lane;
                    if (i < y_row) {
                        const size_t y_base =
                            (size_t)B_idx * y_row * k8 + i * k8 + bk;
                        #pragma unroll
                        for (int u = 0; u < 4; u++) {
                            y_bytes[it][u] = (bk + u < k8) ? Y[y_base + u] : 0;
                        }
                    } else {
                        #pragma unroll
                        for (int u = 0; u < 4; u++) y_bytes[it][u] = 0;
                    }
                }

                /* All 32 bit planes are independent.  Interleave each
                 * shuffle-tree level so the scheduler can hide shuffle
                 * latency across bit planes without reloading X/Z. */
                if (interleave_bit_reductions) {
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        #pragma unroll
                        for (int bit = 0; bit < 32; bit++) {
                            t_local[bit] += __shfl_down_sync(
                                0xffffffff, t_local[bit], offset);
                        }
                    }
                }

                #pragma unroll
                for (int bit = 0; bit < 32; bit++) {
                    float t_k = interleave_bit_reductions
                        ? t_local[bit]
                        : warp_reduce_sum(t_local[bit]);
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;
                    const int byte_idx = bit >> 3;
                    const int shift = 7 - (bit & 7);

                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((y_bytes[it][byte_idx] >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }

            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

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
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}

/* Quantization-plane-interleaved decode for 2/3-bit BQQ.
 * Y/Z layout: [row_width, col_width, row, k8, BIT_PLANES]. */
template <typename X_T, int N_WARPS, int N_I_TILES, int K8_MAX,
          int BIT_PLANES>
__global__ void bqq_forward_bitplane_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width,
    int y_row, int z_col, int k8,
    int col_splits)
{
    const int combined = blockIdx.x;
    const int r = combined / col_splits;
    const int cs = combined % col_splits;
    const int n = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];
    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += load_activation(X, x_base + j);
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        float a_val[BIT_PLANES];
        float b_val[BIT_PLANES];
        float c_val[BIT_PLANES];
        float t_sum[BIT_PLANES];
        #pragma unroll
        for (int p = 0; p < BIT_PLANES; p++) {
            const int B_idx = p * row_width * col_width + rc;
            a_val[p] = __half2float(a_ptr[B_idx]);
            b_val[p] = __half2float(b_ptr[B_idx]);
            c_val[p] = __half2float(c_ptr[B_idx]);
            t_sum[p] = 0.0f;
        }

        /* One packed K byte per tile keeps 8*BIT_PLANES partials in
         * registers while sharing each X load across all planes. */
        #pragma unroll
        for (int bk = 0; bk < K8_MAX; bk++) {
            if (bk >= k8) break;

            float t_local[BIT_PLANES][8];
            #pragma unroll
            for (int p = 0; p < BIT_PLANES; p++) {
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) t_local[p][bit] = 0.0f;
            }

            for (int j = lane; j < z_col; j += 32) {
                const float xv = load_activation(X, x_base + j);
                const size_t z_base =
                    (((size_t)rc * z_col + j) * k8 + bk) * BIT_PLANES;
                #pragma unroll
                for (int p = 0; p < BIT_PLANES; p++) {
                    const uint8_t zb = Z[z_base + p];
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int shift = 7 - bit;
                        if ((zb >> shift) & 1) t_local[p][bit] += xv;
                    }
                }
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int p = 0; p < BIT_PLANES; p++) {
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        t_local[p][bit] += __shfl_down_sync(
                            0xffffffff, t_local[p][bit], offset);
                    }
                }
            }

            uint8_t y_bytes[N_I_TILES][BIT_PLANES];
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) {
                const int i = it * 32 + lane;
                #pragma unroll
                for (int p = 0; p < BIT_PLANES; p++) {
                    if (i < y_row) {
                        const size_t y_base =
                            (((size_t)rc * y_row + i) * k8 + bk) * BIT_PLANES;
                        y_bytes[it][p] = Y[y_base + p];
                    } else {
                        y_bytes[it][p] = 0;
                    }
                }
            }

            #pragma unroll
            for (int p = 0; p < BIT_PLANES; p++) {
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const float t_k = __shfl_sync(
                        0xffffffff, t_local[p][bit], 0);
                    const float t_aug = a_val[p] * t_k + b_val[p] * xsum;
                    const int shift = 7 - bit;
                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((y_bytes[it][p] >> shift) & 1)
                            acc[it] += t_aug;
                    }
                    t_sum[p] += t_k;
                }
            }
        }

        float c_term = 0.0f;
        #pragma unroll
        for (int p = 0; p < BIT_PLANES; p++) c_term += c_val[p] * t_sum[p];
        const float common_term = c_term + __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += common_term;
    }

    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        warp_acc[warp_id * 32 + lane] = acc[it];
        __syncthreads();
        if (warp_id == 0) {
            float total = warp_acc[lane];
            #pragma unroll
            for (int w = 1; w < N_WARPS; w++)
                total += warp_acc[w * 32 + lane];
            const int i = it * 32 + lane;
            if (i < y_row) {
                const size_t idx =
                    (size_t)n * row_width * y_row + r * y_row + i;
                if (col_splits > 1)
                    atomicAdd(&out[idx], total);
                else
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}

template <typename X_T, int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_byte4_rowtile_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits, int row_tiles)
{
    constexpr int ROW_TILE = N_I_TILES * 32;
    const int combined = blockIdx.x;
    const int cs = combined % col_splits;
    const int tmp = combined / col_splits;
    const int rt = tmp % row_tiles;
    const int r = tmp / row_tiles;
    const int row_start = rt * ROW_TILE;

    const int n  = blockIdx.y;
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end   = min(c_block_start + c_per_split, col_width);

    __shared__ float warp_acc[N_WARPS * 32];

    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    for (int ci = c_block_start + warp_id; ci < c_block_end; ci += N_WARPS) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += load_activation(X, x_base + j);
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);
            float t_sum = 0.0f;

            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk += 4) {
                if (bk >= k8) break;

                float t_local[32];
                #pragma unroll
                for (int bit = 0; bit < 32; bit++) t_local[bit] = 0.0f;

                for (int j = lane; j < z_col; j += 32) {
                    const size_t z_base = (size_t)B_idx * z_col * k8 + j * k8 + bk;
                    uint8_t z_bytes[4];
                    #pragma unroll
                    for (int u = 0; u < 4; u++) {
                        z_bytes[u] = (bk + u < k8) ? Z[z_base + u] : 0;
                    }
                    const float xv = load_activation(X, x_base + j);
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int shift = 7 - bit;
                        #pragma unroll
                        for (int u = 0; u < 4; u++) {
                            if ((z_bytes[u] >> shift) & 1) {
                                t_local[u * 8 + bit] += xv;
                            }
                        }
                    }
                }

                uint8_t y_bytes[N_I_TILES][4];
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    const int i = row_start + it * 32 + lane;
                    if (i < y_row) {
                        const size_t y_base =
                            (size_t)B_idx * y_row * k8 + i * k8 + bk;
                        #pragma unroll
                        for (int u = 0; u < 4; u++) {
                            y_bytes[it][u] = (bk + u < k8) ? Y[y_base + u] : 0;
                        }
                    } else {
                        #pragma unroll
                        for (int u = 0; u < 4; u++) y_bytes[it][u] = 0;
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    #pragma unroll
                    for (int bit = 0; bit < 32; bit++) {
                        t_local[bit] += __shfl_down_sync(
                            0xffffffff, t_local[bit], offset);
                    }
                }

                #pragma unroll
                for (int bit = 0; bit < 32; bit++) {
                    float t_k = t_local[bit];
                    t_k = __shfl_sync(0xffffffff, t_k, 0);
                    const float t_aug = a_val * t_k + b_val * xsum;
                    const int byte_idx = bit >> 3;
                    const int shift = 7 - (bit & 7);

                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((y_bytes[it][byte_idx] >> shift) & 1) acc[it] += t_aug;
                    }
                    t_sum += t_k;
                }
            }

            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++) acc[it] += c_term;
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++) acc[it] += d_term;
    }

    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        warp_acc[warp_id * 32 + lane] = acc[it];
        __syncthreads();
        if (warp_id == 0) {
            float total = warp_acc[lane];
            #pragma unroll
            for (int w = 1; w < N_WARPS; w++)
                total += warp_acc[w * 32 + lane];
            const int i = row_start + it * 32 + lane;
            if (i < y_row) {
                const size_t idx = (size_t)n * row_width * y_row + r * y_row + i;
                if (col_splits > 1)
                    atomicAdd(&out[idx], total);
                else
                    out[idx] = total;
            }
        }
        __syncthreads();
    }
}

template <typename X_T, int N_WARPS>
__global__ void bqq_stage1_ztx_kernel(
    const uint8_t* __restrict__ Z,
    const X_T*     __restrict__ X,
    float*         __restrict__ T,
    float*         __restrict__ Tsum,
    float*         __restrict__ Xsum,
    int B_total, int col_width, int z_col, int k8,
    bool z_rank_major)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int n_inner = k8 * 8;
    const int item = (int)blockIdx.x * N_WARPS + warp_id;
    const int total = B_total * n_inner;
    if (item >= total) return;

    const int B_idx = item / n_inner;
    const int l = item - B_idx * n_inner;
    const int c = B_idx % col_width;
    const int byte_k = l >> 3;
    const int shift = 7 - (l & 7);
    const int x_base = c * z_col;

    /* Two Z layouts.  j-major [B, z_col, k8] is the historical one; rank-major
     * [B, n_inner, z_col/8] is what the LUT decode kernels need, and a module
     * stores only one of them (see PackedBinaryQuadratic._get_flat_cache), so
     * prefill has to read whichever it got.  Rank-major is also the friendlier
     * layout here: this warp owns one rank l, and its j bits are contiguous. */
    const int zc8 = z_col >> 3;
    const uint8_t* zrow_rm = z_rank_major
        ? (Z + (size_t)B_idx * n_inner * zc8 + (size_t)l * zc8)
        : nullptr;

    float t_local = 0.0f;
    float xsum_local = 0.0f;
    for (int j = lane; j < z_col; j += 32) {
        const float xv = load_activation(X, x_base + j);
        uint8_t zb;
        int sh;
        if (z_rank_major) {
            zb = zrow_rm[j >> 3];
            sh = 7 - (j & 7);
        } else {
            zb = Z[(size_t)B_idx * z_col * k8 + j * k8 + byte_k];
            sh = shift;
        }
        if ((zb >> sh) & 1) t_local += xv;
        if (l == 0) xsum_local += xv;
    }
    float t = warp_reduce_sum(t_local);
    float xs = warp_reduce_sum(xsum_local);
    if (lane == 0) {
        T[(size_t)B_idx * n_inner + l] = t;
        atomicAdd(&Tsum[B_idx], t);
        if (l == 0) Xsum[B_idx] = xs;
    }
}

template <int N_I_TILES>
__global__ void bqq_stage2_yt_kernel(
    const uint8_t* __restrict__ Y,
    const float*   __restrict__ T,
    const float*   __restrict__ Tsum,
    const float*   __restrict__ Xsum,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int k8, int row_tiles)
{
    constexpr int ROW_TILE = N_I_TILES * 32;
    const int combined = blockIdx.x;
    const int rt = combined % row_tiles;
    const int r = combined / row_tiles;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n = blockIdx.y;
    const int n_inner = k8 * 8;
    const int row_start = rt * ROW_TILE;

    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        const int i = row_start + it * 32 + lane;
        if (warp_id != it || i >= y_row) continue;

        float acc = 0.0f;
        for (int ci = 0; ci < col_width; ci++) {
            const int rc = r * col_width + ci;
            float d_xsum = 0.0f;
            for (int p = 0; p < bit_width; p++) {
                const int B_idx = p * row_width * col_width + rc;
                const uint8_t* yp = Y + (size_t)B_idx * y_row * k8 + i * k8;
                const float* tptr = T + (size_t)B_idx * n_inner;
                float part = 0.0f;
                int y_count = 0;
                for (int bk = 0; bk < k8; bk++) {
                    const uint8_t yb = yp[bk];
                    y_count += __popc((unsigned)yb);
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int shift = 7 - bit;
                        if ((yb >> shift) & 1) {
                            part += tptr[(bk << 3) + bit];
                        }
                    }
                }
                const float xsum = Xsum[B_idx];
                acc += __half2float(a_ptr[B_idx]) * part
                     + __half2float(b_ptr[B_idx]) * xsum * (float)y_count
                     + __half2float(c_ptr[B_idx]) * Tsum[B_idx];
                d_xsum = __half2float(d_ptr[rc]) * xsum;
            }
            acc += d_xsum;
        }

        out[(size_t)n * row_width * y_row + r * y_row + i] = acc;
    }
}

template <int N_I_TILES>
__global__ void bqq_stage2_yt_partial_kernel(
    const uint8_t* __restrict__ Y,
    const float*   __restrict__ T,
    const float*   __restrict__ Tsum,
    const float*   __restrict__ Xsum,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    float*         __restrict__ partial,
    int row_width, int col_width, int bit_width,
    int y_row, int k8, int row_tiles)
{
    constexpr int ROW_TILE = N_I_TILES * 32;
    const int combined = blockIdx.x;
    const int rt = combined % row_tiles;
    const int tmp = combined / row_tiles;
    const int r = tmp % row_width;
    const int p = tmp / row_width;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_inner = k8 * 8;
    const int row_start = rt * ROW_TILE;

    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        const int i = row_start + it * 32 + lane;
        if (warp_id != it || i >= y_row) continue;

        float acc = 0.0f;
        for (int ci = 0; ci < col_width; ci++) {
            const int rc = r * col_width + ci;
            const int B_idx = p * row_width * col_width + rc;
            const uint8_t* yp = Y + (size_t)B_idx * y_row * k8 + i * k8;
            const float* tptr = T + (size_t)B_idx * n_inner;
            float part = 0.0f;
            int y_count = 0;
            for (int bk = 0; bk < k8; bk++) {
                const uint8_t yb = yp[bk];
                y_count += __popc((unsigned)yb);
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const int shift = 7 - bit;
                    if ((yb >> shift) & 1) {
                        part += tptr[(bk << 3) + bit];
                    }
                }
            }
            const float xsum = Xsum[B_idx];
            acc += __half2float(a_ptr[B_idx]) * part
                 + __half2float(b_ptr[B_idx]) * xsum * (float)y_count
                 + __half2float(c_ptr[B_idx]) * Tsum[B_idx];
        }

        partial[(size_t)p * row_width * y_row + r * y_row + i] = acc;
    }
}

__global__ void bqq_stage3_reduce_kernel(
    const float* __restrict__ partial,
    const float* __restrict__ Xsum,
    const __half* __restrict__ d_ptr,
    float*       __restrict__ out,
    int row_width, int col_width, int bit_width, int y_row)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = row_width * y_row;
    if (idx >= total) return;

    const int r = idx / y_row;
    const int i = idx - r * y_row;
    float acc = 0.0f;
    for (int p = 0; p < bit_width; p++) {
        acc += partial[(size_t)p * row_width * y_row + r * y_row + i];
    }
    for (int ci = 0; ci < col_width; ci++) {
        const int rc = r * col_width + ci;
        acc += __half2float(d_ptr[rc]) * Xsum[rc];
    }
    out[idx] = acc;
}

__global__ void bqq_build_t_lut_kernel(
    const float* __restrict__ T,
    float*       __restrict__ TLut,
    int B_total, int k8)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B_total * k8 * 256;
    if (idx >= total) return;

    const int mask = idx & 255;
    const int tmp = idx >> 8;
    const int bk = tmp % k8;
    const int B_idx = tmp / k8;
    const float* tptr = T + (size_t)B_idx * k8 * 8 + bk * 8;

    float sum = 0.0f;
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {
        const int shift = 7 - bit;
        if ((mask >> shift) & 1) sum += tptr[bit];
    }
    TLut[idx] = sum;
}

template <int N_WARPS>
__global__ void bqq_stage2_yt_rowwarp_lut_partial_kernel(
    const uint8_t* __restrict__ Y,
    const float*   __restrict__ TLut,
    const float*   __restrict__ Tsum,
    const float*   __restrict__ Xsum,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    float*         __restrict__ partial,
    int row_width, int col_width, int bit_width,
    int y_row, int k8, int row_tiles)
{
    const int combined = blockIdx.x;
    const int rt = combined % row_tiles;
    const int tmp = combined / row_tiles;
    const int r = tmp % row_width;
    const int p = tmp / row_width;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_start = rt * N_WARPS;
    const int i = row_start + warp_id;
    if (warp_id >= N_WARPS || i >= y_row) return;

    float acc = 0.0f;
    for (int ci = 0; ci < col_width; ci++) {
        const int rc = r * col_width + ci;
        const int B_idx = p * row_width * col_width + rc;
        const uint8_t* yp = Y + (size_t)B_idx * y_row * k8 + i * k8;
        const float* lut = TLut + (size_t)B_idx * k8 * 256;

        float part_local = 0.0f;
        float ycount_local = 0.0f;
        for (int bk = lane; bk < k8; bk += 32) {
            const uint8_t yb = yp[bk];
            ycount_local += (float)__popc((unsigned)yb);
            part_local += lut[bk * 256 + (int)yb];
        }

        const float part = warp_reduce_sum(part_local);
        const float y_count = warp_reduce_sum(ycount_local);
        if (lane == 0) {
            const float xsum = Xsum[B_idx];
            acc += __half2float(a_ptr[B_idx]) * part
                 + __half2float(b_ptr[B_idx]) * xsum * y_count
                 + __half2float(c_ptr[B_idx]) * Tsum[B_idx];
        }
    }

    if (lane == 0) {
        partial[(size_t)p * row_width * y_row + r * y_row + i] = acc;
    }
}


/* Warp-parallel explicit Z@x then Y@t fallback for float32 activations. */
template <int N_WARPS>
__global__ void bqq_decode_two_stage_warp_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const float*   __restrict__ X,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,
    float*         __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8,
    int col_splits)
{
    constexpr int K_MAX = 128;
    __shared__ float t_vals[K_MAX];
    __shared__ float t_sum_shared;

    const int combined = blockIdx.x;
    const int r = combined / col_splits;
    const int cs = combined % col_splits;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int c_per_split = (col_width + col_splits - 1) / col_splits;
    const int c_block_start = cs * c_per_split;
    const int c_block_end = min(c_block_start + c_per_split, col_width);
    const int n_inner = k8 * 8;

    float acc = 0.0f;
    const bool has_i = tid < y_row;
    const int i = tid;

    for (int ci = c_block_start; ci < c_block_end; ci++) {
        const int rc = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        float xsum_local = 0.0f;
        for (int j = lane; j < z_col; j += 32) {
            xsum_local += X[x_base + j];
        }
        float xsum = warp_reduce_sum(xsum_local);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);

            for (int l_base = 0; l_base < K_MAX; l_base += N_WARPS) {
                const int l = l_base + warp_id;
                float t_local = 0.0f;
                if (l < n_inner) {
                    const int byte_k = l >> 3;
                    const int shift = 7 - (l & 7);
                    for (int j = lane; j < z_col; j += 32) {
                        const uint8_t zb = Z[(size_t)B_idx * z_col * k8 + j * k8 + byte_k];
                        if ((zb >> shift) & 1) t_local += X[x_base + j];
                    }
                }
                float t_l = warp_reduce_sum(t_local);
                if (lane == 0 && l < n_inner) t_vals[l] = t_l;
            }
            __syncthreads();

            if (warp_id == 0) {
                float ts = 0.0f;
                for (int l = lane; l < n_inner; l += 32) {
                    ts += t_vals[l];
                }
                ts = warp_reduce_sum(ts);
                if (lane == 0) t_sum_shared = ts;
            }
            __syncthreads();

            if (has_i) {
                const uint8_t* yp = Y + (size_t)B_idx * y_row * k8 + i * k8;
                float part = 0.0f;
                int y_count = 0;
                for (int byte_k = 0; byte_k < k8; byte_k++) {
                    uint8_t yb = yp[byte_k];
                    y_count += __popc(static_cast<unsigned>(yb));
                    #pragma unroll
                    for (int bit = 0; bit < 8; bit++) {
                        const int l = (byte_k << 3) + bit;
                        if (l >= n_inner) break;
                        const int shift = 7 - bit;
                        if ((yb >> shift) & 1) {
                            part += a_val * t_vals[l];
                        }
                    }
                }
                acc += part + b_val * xsum * (float)y_count + c_val * t_sum_shared;
            }
            __syncthreads();
        }

        if (has_i) acc += __half2float(d_ptr[rc]) * xsum;
    }

    if (has_i) {
        size_t idx = (size_t)n * row_width * y_row + r * y_row + i;
        if (col_splits > 1)
            atomicAdd(&out[idx], acc);
        else
            out[idx] = acc;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Fused W-reconstruction kernel
 * ═══════════════════════════════════════════════════════════════════
 *
 * Reconstructs the full weight matrix W via AND+popcount in one
 * kernel launch (replaces 4-5 separate ops in the Python path).
 *
 * W[r*yr+i, c*zc+j] = Σ_p { a_p · popc(Y_p[i,:] & Z_p[:,j])
 *                          + b_p · popc(Y_p[i,:])
 *                          + c_p · popc(Z_p[:,j]) } + d[r,c]
 *
 * Each thread computes one W element independently.
 * Output in FP16 → enables cuBLAS FP16 Tensor Core for X @ W.T.
 */

/* Transpose a rank-major Z ([B, n_inner, z_col/8]) back to the j-major layout
 * ([B, z_col, k8]) that reconstruct_W_kernel needs.
 *
 * reconstruct_W builds W[i][j] as __popc(Y_row[i] & Z_col[j]), which requires
 * the *rank* bits of one column j to be contiguous -- i.e. exactly the j-major
 * layout.  A module that feeds the LUT decode kernels stores only the
 * rank-major Z, so prefill materialises the j-major view into a scratch buffer
 * instead of keeping a second copy resident.  The buffer is transient and tiny
 * next to the dense fp16 W that reconstruct_W already allocates. */
__global__ void bqq_z_rank_to_jmajor_kernel(
    const uint8_t* __restrict__ Zt,   /* [B_total, n_inner, z_col/8] */
    uint8_t*       __restrict__ Z,    /* [B_total, z_col, k8]        */
    int B_total, int z_col, int k8)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B_total * z_col * k8;
    if (idx >= total) return;

    const int byte_k = idx % k8;
    const int j      = (idx / k8) % z_col;
    const int B_idx  = idx / (k8 * z_col);

    const int n_inner = k8 * 8;
    const int zc8     = z_col >> 3;
    const int j_byte  = j >> 3;
    const int j_shift = 7 - (j & 7);

    uint8_t out = 0;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const int l = byte_k * 8 + b;          /* rank index */
        if (l >= n_inner) break;
        const uint8_t zb =
            Zt[(size_t)B_idx * n_inner * zc8 + (size_t)l * zc8 + j_byte];
        out |= (uint8_t)(((zb >> j_shift) & 1) << (7 - b));
    }
    Z[(size_t)B_idx * z_col * k8 + (size_t)j * k8 + byte_k] = out;
}


__global__ void reconstruct_W_kernel(
    const uint8_t* __restrict__ Y,      /* [B_total, y_row, k8] */
    const uint8_t* __restrict__ Z,      /* [B_total, z_col, k8] */
    const __half*  __restrict__ a_ptr,  /* [B_total] */
    const __half*  __restrict__ b_ptr,
    const __half*  __restrict__ c_ptr,
    const __half*  __restrict__ d_ptr,  /* [row_width * col_width] */
    __half*        __restrict__ W_out,  /* [out_features, in_features] fp16 */
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int in_idx  = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_features = row_width * y_row;
    const int in_features  = col_width * z_col;

    if (out_idx >= out_features || in_idx >= in_features) return;

    const int r = out_idx / y_row, i = out_idx % y_row;
    const int c = in_idx  / z_col, j = in_idx  % z_col;

    float val = 0.0f;
    for (int p = 0; p < bit_width; p++) {
        const int B = p * row_width * col_width + r * col_width + c;
        const auto* yp = Y + (size_t)B * y_row * k8 + i * k8;
        const auto* zp = Z + (size_t)B * z_col * k8 + j * k8;

        int inner = 0, ys = 0, zs = 0;
        /* k8 bytes per row/col — load as uint32 when k8=4 */
        if (k8 == 4) {
            uint32_t yw = *reinterpret_cast<const uint32_t*>(yp);
            uint32_t zw = *reinterpret_cast<const uint32_t*>(zp);
            inner = __popc(yw & zw);
            ys    = __popc(yw);
            zs    = __popc(zw);
        } else {
            for (int bk = 0; bk < k8; bk++) {
                inner += __popc(static_cast<unsigned>(yp[bk] & zp[bk]));
                ys    += __popc(static_cast<unsigned>(yp[bk]));
                zs    += __popc(static_cast<unsigned>(zp[bk]));
            }
        }
        val += __half2float(a_ptr[B]) * (float)inner
             + __half2float(b_ptr[B]) * (float)ys
             + __half2float(c_ptr[B]) * (float)zs;
    }
    val += __half2float(d_ptr[r * col_width + c]);
    W_out[out_idx * in_features + in_idx] = __float2half(val);
}

/* ═══════════════════════════════════════════════════════════════════
 * Unified forward: one call from Python, zero tensor manipulation
 * ═══════════════════════════════════════════════════════════════════
 *
 * Accepts raw packed tensors in their storage shapes.  Internally
 * handles all flatten/reshape, dispatches the right kernel, and
 * returns the result in the same leading shape as X.
 *
 * seq_len <= 32 → fused warp-shuffle kernel (no W materialisation)
 * seq_len >  32 → reconstruct_W_kernel (popcount) + cuBLAS FP16 matmul
 */

static inline const __half* half_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());
}

static inline const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

/* Decode-path epilogue: emit the fp32 workspace as OUT_T (fp16/bf16) and
 * re-zero it in the same pass, so the accumulation workspace can live in
 * the layer's flat cache and no per-call zeros/fill/cast aten ops are
 * needed. */
template <typename OUT_T>
__global__ void bqq_ws_store_zero_kernel(
    float* __restrict__ ws, OUT_T* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = store_activation<OUT_T>(ws[i]);
        ws[i] = 0.0f;
    }
}

/* Cached multiprocessor count per device (cudaDeviceGetAttribute is
 * surprisingly expensive to call on every decode step). */
static int cached_sm_count() {
    constexpr int kMaxDevices = 64;
    static int sm_cache[kMaxDevices] = {0};
    int device = 0;
    cudaGetDevice(&device);
    if (device < 0 || device >= kMaxDevices) device = 0;
    if (sm_cache[device] == 0) {
        int sm = 0;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device);
        sm_cache[device] = sm > 0 ? sm : 1;
    }
    return sm_cache[device];
}

/* Core entry point taking pre-flattened weights.  Callers that hold the
 * layer state (PackedBinaryQuadratic) cache these flat fp32 tensors once,
 * so per-call reshape/dtype-conversion work disappears from the decode
 * hot path. */
torch::Tensor bqq_forward_core_impl(
    torch::Tensor Y_flat,       /* [B_total, y_row, k8]  uint8, contiguous */
    torch::Tensor Z_flat,       /* [B_total, z_col, k8]  uint8, contiguous */
    torch::Tensor X,            /* [..., in_features]                      */
    torch::Tensor a_flat,       /* [B_total]             float16           */
    torch::Tensor b_flat,       /* [B_total]             float16           */
    torch::Tensor c_flat,       /* [B_total]             float16           */
    torch::Tensor d_flat,       /* [row_width*col_width] float16           */
    torch::Tensor bias,         /* [out_features] or empty                 */
    torch::Tensor ws,           /* [1, row_width, y_row] float32 zeroed
                                   workspace, or undefined.  Must be zero on
                                   entry; the fused epilogue re-zeroes it.  */
    int64_t bit_width_, int64_t row_width_, int64_t col_width_,
    int64_t y_row_, int64_t z_col_,
    bool z_rank_major,
    torch::Tensor output, int64_t output_offset)
{
    TORCH_CHECK(a_flat.scalar_type() == torch::kHalf &&
                b_flat.scalar_type() == torch::kHalf &&
                c_flat.scalar_type() == torch::kHalf &&
                d_flat.scalar_type() == torch::kHalf,
                "bqq_forward_flat: a/b/c/d must be float16");
    const int bit_width = (int)bit_width_;
    const int row_width = (int)row_width_;
    const int col_width = (int)col_width_;
    const int y_row     = (int)y_row_;
    const int z_col     = (int)z_col_;
    const bool has_bitplane_layout = Y_flat.dim() == 5;
    const int k8        = has_bitplane_layout ? Y_flat.size(3) : Y_flat.size(2);
    const int out_f     = row_width * y_row;
    const int in_f      = col_width * z_col;
    const int B_total   = bit_width * row_width * col_width;
    const int ni        = (y_row + 31) / 32;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    /* ── flatten X to [batch, in_features] ─────────────────── */
    auto X_shape = X.sizes().vec();
    int64_t batch = 1;
    for (int i = 0; i < (int)X_shape.size() - 1; i++) batch *= X_shape[i];
    auto X_2d = X.reshape({batch, in_f});

    const bool direct_output = output.defined();
    if (direct_output) {
        TORCH_CHECK(batch == 1,
                    "bqq_forward_flat_out currently supports batch=1 only");
        TORCH_CHECK(output.is_cuda() && output.is_contiguous(),
                    "bqq_forward_flat_out: output must be contiguous CUDA tensor");
        TORCH_CHECK(output.scalar_type() == X.scalar_type(),
                    "bqq_forward_flat_out: output dtype must match X");
        TORCH_CHECK(output_offset >= 0 && output.dim() >= 1 &&
                    output.size(-1) >= output_offset + out_f,
                    "bqq_forward_flat_out: output slice is too small");
    }

    torch::Tensor result;
    bool wrote_direct_output = false;

    const bool x_large_bf16 = X.scalar_type() == torch::kBFloat16;
    if (batch <= 1 && y_row > 128 &&
        (X.scalar_type() == torch::kFloat16 || x_large_bf16) &&
        z_col > 32 && k8 > 3) {
        constexpr int NW = 4;
        constexpr int NI = 4;
        constexpr int ROW_TILE = NI * 32;
        const int row_tiles = (y_row + ROW_TILE - 1) / ROW_TILE;
        const char* large_kernel = std::getenv("BQQ_CUDA_LARGE_KERNEL");
        const bool use_large_rowtile =
            !z_rank_major &&
            large_kernel && std::strcmp(large_kernel, "rowtile") == 0 && k8 <= 16;

        const int sm_count = cached_sm_count();

        auto X_view_half = X_2d.reshape({(int)batch, col_width, z_col}).contiguous();

        if (use_large_rowtile) {
            int col_splits = max(1, (72 * sm_count + row_width * row_tiles * NW - 1)
                                    / (row_width * row_tiles * NW));
            col_splits = min(col_splits, col_width);
            while (col_splits > 1 && col_width % col_splits != 0) col_splits--;
            if (const char* env = std::getenv("BQQ_CUDA_COL_SPLITS")) {
                int requested = std::atoi(env);
                if (requested >= 1) {
                    col_splits = min(requested, col_width);
                    while (col_splits > 1 && col_width % col_splits != 0) col_splits--;
                }
            }

            auto out = (col_splits > 1)
                ? torch::zeros({(int)batch, row_width, y_row},
                      torch::dtype(torch::kFloat32).device(X.device()))
                : torch::empty({(int)batch, row_width, y_row},
                      torch::dtype(torch::kFloat32).device(X.device()));

            dim3 grid(row_width * row_tiles * col_splits, batch);
            dim3 block(NW * 32);
            int smem = NW * 32 * sizeof(float);

            #define LAUNCH_ROWTILE(K8M) \
                do { \
                    if (x_large_bf16) \
                        bqq_forward_byte4_rowtile_kernel<__nv_bfloat16, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                            bf16_ptr(X_view_half), \
                            half_ptr(a_flat), half_ptr(b_flat), \
                            half_ptr(c_flat), half_ptr(d_flat), \
                            out.data_ptr<float>(), \
                            row_width, col_width, bit_width, y_row, z_col, k8, col_splits, row_tiles); \
                    else \
                        bqq_forward_byte4_rowtile_kernel<__half, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                            half_ptr(X_view_half), \
                            half_ptr(a_flat), half_ptr(b_flat), \
                            half_ptr(c_flat), half_ptr(d_flat), \
                            out.data_ptr<float>(), \
                            row_width, col_width, bit_width, y_row, z_col, k8, col_splits, row_tiles); \
                } while (0)

            if      (k8 <= 4)  LAUNCH_ROWTILE(4);
            else if (k8 <= 8)  LAUNCH_ROWTILE(8);
            else               LAUNCH_ROWTILE(16);
            #undef LAUNCH_ROWTILE

            result = out.reshape({(int)batch, out_f});
        } else {
            const int n_inner = k8 * 8;
            auto T = torch::empty({B_total, n_inner},
                torch::dtype(torch::kFloat32).device(X.device()));
            auto Tsum = torch::zeros({B_total},
                torch::dtype(torch::kFloat32).device(X.device()));
            auto Xsum = torch::empty({B_total},
                torch::dtype(torch::kFloat32).device(X.device()));
            auto out = torch::empty({(int)batch, row_width, y_row},
                torch::dtype(torch::kFloat32).device(X.device()));

            dim3 s1_block(NW * 32);
            dim3 s1_grid((B_total * n_inner + NW - 1) / NW);
            if (x_large_bf16)
                bqq_stage1_ztx_kernel<__nv_bfloat16, NW><<<s1_grid, s1_block, 0, stream>>>(
                    Z_flat.data_ptr<uint8_t>(),
                    bf16_ptr(X_view_half),
                    T.data_ptr<float>(),
                    Tsum.data_ptr<float>(),
                    Xsum.data_ptr<float>(),
                    B_total, col_width, z_col, k8, z_rank_major);
            else
                bqq_stage1_ztx_kernel<__half, NW><<<s1_grid, s1_block, 0, stream>>>(
                    Z_flat.data_ptr<uint8_t>(),
                    half_ptr(X_view_half),
                    T.data_ptr<float>(),
                    Tsum.data_ptr<float>(),
                    Xsum.data_ptr<float>(),
                    B_total, col_width, z_col, k8, z_rank_major);

            const bool use_parallel_bits = bit_width >= 2 && k8 >= 64;
            const bool use_t_lut = k8 >= 128;
            dim3 s2_block(NI * 32);
            if (use_parallel_bits) {
                torch::Tensor TLut;
                if (use_t_lut) {
                    TLut = torch::empty({B_total, k8, 256},
                        torch::dtype(torch::kFloat32).device(X.device()));
                    const int lut_total = B_total * k8 * 256;
                    dim3 lut_block(256);
                    dim3 lut_grid((lut_total + lut_block.x - 1) / lut_block.x);
                    bqq_build_t_lut_kernel<<<lut_grid, lut_block, 0, stream>>>(
                        T.data_ptr<float>(),
                        TLut.data_ptr<float>(),
                        B_total, k8);
                }
                if (use_t_lut) {
                    constexpr int rowwarp_warps = 8;
                    const int rowwarp_row_tiles = (y_row + rowwarp_warps - 1) / rowwarp_warps;
                    auto partial = torch::empty({bit_width, row_width, y_row},
                        torch::dtype(torch::kFloat32).device(X.device()));
                    dim3 s2_grid(bit_width * row_width * rowwarp_row_tiles, batch);
                    dim3 s2_block(rowwarp_warps * 32);
                    bqq_stage2_yt_rowwarp_lut_partial_kernel<rowwarp_warps><<<s2_grid, s2_block, 0, stream>>>(
                        Y_flat.data_ptr<uint8_t>(),
                        TLut.data_ptr<float>(),
                        Tsum.data_ptr<float>(),
                        Xsum.data_ptr<float>(),
                        half_ptr(a_flat), half_ptr(b_flat),
                        half_ptr(c_flat),
                        partial.data_ptr<float>(),
                        row_width, col_width, bit_width, y_row, k8, rowwarp_row_tiles);

                    const int total_out = row_width * y_row;
                    dim3 s3_block(256);
                    dim3 s3_grid((total_out + s3_block.x - 1) / s3_block.x);
                    bqq_stage3_reduce_kernel<<<s3_grid, s3_block, 0, stream>>>(
                        partial.data_ptr<float>(),
                        Xsum.data_ptr<float>(),
                        half_ptr(d_flat),
                        out.data_ptr<float>(),
                        row_width, col_width, bit_width, y_row);
                } else {
                    auto partial = torch::empty({bit_width, row_width, y_row},
                        torch::dtype(torch::kFloat32).device(X.device()));
                    dim3 s2_grid(bit_width * row_width * row_tiles, batch);
                    bqq_stage2_yt_partial_kernel<NI><<<s2_grid, s2_block, 0, stream>>>(
                        Y_flat.data_ptr<uint8_t>(),
                        T.data_ptr<float>(),
                        Tsum.data_ptr<float>(),
                        Xsum.data_ptr<float>(),
                        half_ptr(a_flat), half_ptr(b_flat),
                        half_ptr(c_flat),
                        partial.data_ptr<float>(),
                        row_width, col_width, bit_width, y_row, k8, row_tiles);

                    const int total_out = row_width * y_row;
                    dim3 s3_block(256);
                    dim3 s3_grid((total_out + s3_block.x - 1) / s3_block.x);
                    bqq_stage3_reduce_kernel<<<s3_grid, s3_block, 0, stream>>>(
                        partial.data_ptr<float>(),
                        Xsum.data_ptr<float>(),
                        half_ptr(d_flat),
                        out.data_ptr<float>(),
                        row_width, col_width, bit_width, y_row);
                }
            } else {
                dim3 s2_grid(row_width * row_tiles, batch);
                bqq_stage2_yt_kernel<NI><<<s2_grid, s2_block, 0, stream>>>(
                    Y_flat.data_ptr<uint8_t>(),
                    T.data_ptr<float>(),
                    Tsum.data_ptr<float>(),
                    Xsum.data_ptr<float>(),
                    half_ptr(a_flat), half_ptr(b_flat),
                    half_ptr(c_flat), half_ptr(d_flat),
                    out.data_ptr<float>(),
                    row_width, col_width, bit_width, y_row, k8, row_tiles);
            }

            result = out.reshape({(int)batch, out_f});
        }

    } else if (batch <= 1) {
        /* ── small seq: fused warp-shuffle kernel ──────────────── */
        const char* decode_kernel = std::getenv("BQQ_CUDA_DECODE_KERNEL");
        /* fp16 and bf16 activations both take the fused 16-bit kernels;
         * load_activation converts to fp32 in-register either way. */
        const bool x_is_bf16 = X.scalar_type() == torch::kBFloat16;
        const bool x_is_16bit =
            X.scalar_type() == torch::kFloat16 || x_is_bf16;
        const bool use_bitplane_packed =
            !z_rank_major && decode_kernel &&
            std::strcmp(decode_kernel, "bitplane_packed") == 0 &&
            has_bitplane_layout && bit_width >= 2 && bit_width <= 3 &&
            x_is_16bit && z_col > 32 && k8 > 1;
        const bool serial_byte4_reduction =
            !z_rank_major && decode_kernel &&
            std::strcmp(decode_kernel, "bitblas_byte4_serial") == 0 &&
            x_is_16bit && z_col > 32 && k8 > 3;
        /* nibble-LUT: Z arrives rank-major (built by the Python flat cache). */
        /* 8-bit LUT + row tiling; shares the rank-major Zt layout with "lut". */
        const bool use_lut8 =
            z_rank_major &&
            decode_kernel && std::strcmp(decode_kernel, "lut8") == 0 &&
            k8 <= 4 && z_col <= 128 && (z_col % 8) == 0;
        /* Nibble-LUT is the default decode kernel: it replaces the per-weight
         * shift+and+predicated-add of the byte kernels with one table lookup per
         * 4 weights, which measured ~2x faster than bitblas_byte4 on every layer
         * shape we tried.  z_col % 32 == 0 keeps zc8 a multiple of 4 so a Zt row
         * loads as whole uint32 words.  Anything outside those bounds falls
         * through to the older kernels below. */
        const bool lut_capable = k8 <= 4 && z_col <= 128 && (z_col % 32) == 0;
        const bool use_lut =
            !use_lut8 && lut_capable && z_rank_major &&
            (decode_kernel == nullptr || std::strcmp(decode_kernel, "lut") == 0);
        /* lane-per-rank: needs inter_dim <= 32 (k8 <= 4) so one lane owns one
         * rank, and z_col <= 128 for the shared-memory staging buffers. */
        const bool use_lanerank =
            !z_rank_major && !use_lut8 && !use_lut &&
            decode_kernel && std::strcmp(decode_kernel, "lanerank") == 0 &&
            k8 <= 4 && z_col <= 128;
        const bool use_bitblas_byte4 =
            !z_rank_major && !use_lut8 && !use_lut && !use_lanerank &&
            ((decode_kernel && std::strcmp(decode_kernel, "bitblas_byte4") == 0) ||
             (decode_kernel == nullptr && bit_width <= 2)) &&
            x_is_16bit && z_col > 32 && k8 > 3;
        const bool use_byte4_path = use_bitblas_byte4 || serial_byte4_reduction;
        const bool use_bitblas_byte2 =
            !z_rank_major && !use_lut8 && !use_lut && !use_lanerank &&
            ((decode_kernel && std::strcmp(decode_kernel, "bitblas_byte2") == 0) ||
             (decode_kernel == nullptr && !use_bitblas_byte4 && bit_width <= 2)) &&
            x_is_16bit && z_col > 32 && k8 > 1;
        const bool use_bitblas_byte =
            !z_rank_major && !use_lut8 && !use_lut && !use_lanerank &&
            ((decode_kernel && std::strcmp(decode_kernel, "bitblas_byte") == 0) ||
             (decode_kernel == nullptr && !use_bitblas_byte4 && !use_bitblas_byte2)) &&
            x_is_16bit && z_col > 32;

        const bool half_x_path =
            use_bitblas_byte || use_bitblas_byte2 || use_byte4_path ||
            use_bitplane_packed || ((use_lanerank || use_lut || use_lut8) && x_is_16bit);
        auto X_view = half_x_path
            ? torch::Tensor()
            : X_2d.reshape({(int)batch, col_width, z_col})
                  .to(torch::kFloat32).contiguous();
        auto X_view_half = half_x_path
            ? X_2d.reshape({(int)batch, col_width, z_col}).contiguous()
            : torch::Tensor();

        constexpr int NW = 4;
        const int sm_count = cached_sm_count();

        int col_splits = max(1, (72 * sm_count + row_width * NW - 1)
                                / (row_width * NW));
        col_splits = min(col_splits, col_width);
        while (col_splits > 1 && col_width % col_splits != 0) col_splits--;

        const char* col_splits_env = std::getenv("BQQ_CUDA_COL_SPLITS");
        if (col_splits_env) {
            int requested = std::atoi(col_splits_env);
            if (requested >= 1) {
                col_splits = min(requested, col_width);
                while (col_splits > 1 && col_width % col_splits != 0) col_splits--;
            }
        } else if (use_bitblas_byte || use_bitblas_byte2 || use_byte4_path ||
                   use_bitplane_packed) {
            /* The 70-94 register byte kernels need enough independent CTAs
             * to cover the register-limited occupancy on GA10x. */
            constexpr int max_splits = 16;
            col_splits = min(col_splits, max_splits);
            while (col_splits > 1 && col_width % col_splits != 0) col_splits--;
        }

        /* Fused-output path: accumulate into the caller-cached, pre-zeroed
         * fp32 workspace, then one epilogue kernel emits fp16 and re-zeroes.
         * Replaces per-call zeros+fill and the fp32→fp16 aten cast. */
        const bool fuse_out =
            ws.defined() && batch == 1 && x_is_16bit &&
            ws.is_cuda() && ws.scalar_type() == torch::kFloat32 &&
            ws.dim() == 3 && ws.size(0) == 1 &&
            ws.size(1) == row_width && ws.size(2) == y_row;

        torch::Tensor out;
        if (fuse_out) {
            out = ws;
        } else {
            out = (col_splits > 1)
                ? torch::zeros({(int)batch, row_width, y_row},
                      torch::dtype(torch::kFloat32).device(X.device()))
                : torch::empty({(int)batch, row_width, y_row},
                      torch::dtype(torch::kFloat32).device(X.device()));
        }

        dim3 grid(row_width * col_splits, batch);
        const bool use_auto_decode_kernel = decode_kernel == nullptr;
        const bool use_two_stage_warp =
            ((decode_kernel && std::strcmp(decode_kernel, "two_stage_warp") == 0) ||
             (use_auto_decode_kernel && !use_bitblas_byte && !use_bitblas_byte2 &&
              !use_byte4_path && !use_bitplane_packed && z_col >= 128)) &&
            y_row <= 256 && k8 <= 16;

        if (use_lut8) {
            /* 8 warps == 8 row patches per block; grid tiles over row patches. */
            constexpr int NW8 = 8;
            dim3 grid8((row_width + NW8 - 1) / NW8 * col_splits, batch);
            dim3 block8(NW8 * 32);
            #define LAUNCH_LUT8(TYPE, PTR, NI) \
                bqq_forward_lut8_kernel<TYPE, NW8, NI><<<grid8, block8, 0, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), PTR, \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits)
            #define DISPATCH_LUT8(TYPE, PTR) \
                do { \
                    if (ni == 1)      LAUNCH_LUT8(TYPE, PTR, 1); \
                    else if (ni <= 2) LAUNCH_LUT8(TYPE, PTR, 2); \
                    else if (ni <= 4) LAUNCH_LUT8(TYPE, PTR, 4); \
                    else              LAUNCH_LUT8(TYPE, PTR, 8); \
                } while (0)
            if (x_is_bf16)      DISPATCH_LUT8(__nv_bfloat16, bf16_ptr(X_view_half));
            else if (x_is_16bit) DISPATCH_LUT8(__half, half_ptr(X_view_half));
            else                 DISPATCH_LUT8(float, X_view.data_ptr<float>());
            #undef DISPATCH_LUT8
            #undef LAUNCH_LUT8
        } else if (use_lut) {
            dim3 block(NW * 32);
            #define LAUNCH_LUT(TYPE, PTR, NI) \
                bqq_forward_lut_kernel<TYPE, NW, NI><<<grid, block, 0, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), PTR, \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits)
            #define DISPATCH_LUT(TYPE, PTR) \
                do { \
                    if (ni == 1)      LAUNCH_LUT(TYPE, PTR, 1); \
                    else if (ni <= 2) LAUNCH_LUT(TYPE, PTR, 2); \
                    else if (ni <= 4) LAUNCH_LUT(TYPE, PTR, 4); \
                    else              LAUNCH_LUT(TYPE, PTR, 8); \
                } while (0)
            if (x_is_bf16)
                DISPATCH_LUT(__nv_bfloat16, bf16_ptr(X_view_half));
            else if (x_is_16bit)
                DISPATCH_LUT(__half, half_ptr(X_view_half));
            else
                DISPATCH_LUT(float, X_view.data_ptr<float>());
            #undef DISPATCH_LUT
            #undef LAUNCH_LUT
        } else if (use_lanerank) {
            dim3 block(NW * 32);
            #define LAUNCH_LANERANK(TYPE, PTR, NI) \
                bqq_forward_lanerank_kernel<TYPE, NW, NI><<<grid, block, 0, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), PTR, \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits)
            #define DISPATCH_LANERANK(TYPE, PTR) \
                do { \
                    if (ni == 1)      LAUNCH_LANERANK(TYPE, PTR, 1); \
                    else if (ni <= 2) LAUNCH_LANERANK(TYPE, PTR, 2); \
                    else if (ni <= 4) LAUNCH_LANERANK(TYPE, PTR, 4); \
                    else              LAUNCH_LANERANK(TYPE, PTR, 8); \
                } while (0)
            if (x_is_bf16)
                DISPATCH_LANERANK(__nv_bfloat16, bf16_ptr(X_view_half));
            else if (x_is_16bit)
                DISPATCH_LANERANK(__half, half_ptr(X_view_half));
            else
                DISPATCH_LANERANK(float, X_view.data_ptr<float>());
            #undef DISPATCH_LANERANK
            #undef LAUNCH_LANERANK
        } else if (use_bitplane_packed) {
            dim3 block(NW * 32);
            #define LAUNCH_BITPLANE(TYPE, PTR, PLANES, NI, K8M) \
                bqq_forward_bitplane_kernel<TYPE, NW, NI, K8M, PLANES><<<grid, block, 0, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), PTR, \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, y_row, z_col, k8, col_splits)
            #define DISPATCH_BITPLANE(TYPE, PTR, PLANES) \
                do { \
                    if (ni == 1) { \
                        if (k8 <= 4) LAUNCH_BITPLANE(TYPE, PTR, PLANES, 1, 4); \
                        else if (k8 <= 8) LAUNCH_BITPLANE(TYPE, PTR, PLANES, 1, 8); \
                        else LAUNCH_BITPLANE(TYPE, PTR, PLANES, 1, 16); \
                    } else if (ni <= 2) { \
                        if (k8 <= 8) LAUNCH_BITPLANE(TYPE, PTR, PLANES, 2, 8); \
                        else LAUNCH_BITPLANE(TYPE, PTR, PLANES, 2, 16); \
                    } else { \
                        LAUNCH_BITPLANE(TYPE, PTR, PLANES, 4, 16); \
                    } \
                } while (0)
            if (x_is_bf16) {
                if (bit_width == 2)
                    DISPATCH_BITPLANE(__nv_bfloat16, bf16_ptr(X_view_half), 2);
                else
                    DISPATCH_BITPLANE(__nv_bfloat16, bf16_ptr(X_view_half), 3);
            } else {
                if (bit_width == 2)
                    DISPATCH_BITPLANE(__half, half_ptr(X_view_half), 2);
                else
                    DISPATCH_BITPLANE(__half, half_ptr(X_view_half), 3);
            }
            #undef DISPATCH_BITPLANE
            #undef LAUNCH_BITPLANE
        } else if (use_two_stage_warp) {
            constexpr int TW = 8;
            dim3 block(TW * 32);
            bqq_decode_two_stage_warp_kernel<TW><<<grid, block, 0, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                    X_view.data_ptr<float>(), \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits);
        } else {
            dim3 block(NW * 32);
            int smem = NW * 32 * sizeof(float);

            #define LAUNCH_FLOAT(NI, K8M) \
                bqq_forward_kernel<float, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                    X_view.data_ptr<float>(), \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits)

            /* Each 16-bit macro dispatches on the runtime activation dtype;
             * the kernels only differ in the X_T template argument. */
            #define LAUNCH_X16(KERNEL, NI, K8M) \
                do { \
                    if (x_is_bf16) \
                        KERNEL<__nv_bfloat16, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                            bf16_ptr(X_view_half), \
                            half_ptr(a_flat), half_ptr(b_flat), \
                            half_ptr(c_flat), half_ptr(d_flat), \
                            out.data_ptr<float>(), \
                            row_width, col_width, bit_width, y_row, z_col, k8, col_splits); \
                    else \
                        KERNEL<__half, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                            half_ptr(X_view_half), \
                            half_ptr(a_flat), half_ptr(b_flat), \
                            half_ptr(c_flat), half_ptr(d_flat), \
                            out.data_ptr<float>(), \
                            row_width, col_width, bit_width, y_row, z_col, k8, col_splits); \
                } while (0)

            #define LAUNCH_BYTE_HALF(NI, K8M)  LAUNCH_X16(bqq_forward_byte_kernel, NI, K8M)
            #define LAUNCH_BYTE2_HALF(NI, K8M) LAUNCH_X16(bqq_forward_byte2_kernel, NI, K8M)
            #define LAUNCH_BYTE4_HALF(NI, K8M) \
                do { \
                    if (x_is_bf16) \
                        bqq_forward_byte4_kernel<__nv_bfloat16, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                            bf16_ptr(X_view_half), \
                            half_ptr(a_flat), half_ptr(b_flat), \
                            half_ptr(c_flat), half_ptr(d_flat), \
                            out.data_ptr<float>(), \
                            row_width, col_width, bit_width, y_row, z_col, k8, col_splits, \
                            !serial_byte4_reduction); \
                    else \
                        bqq_forward_byte4_kernel<__half, NW, NI, K8M><<<grid, block, smem, stream>>>( \
                            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                            half_ptr(X_view_half), \
                            half_ptr(a_flat), half_ptr(b_flat), \
                            half_ptr(c_flat), half_ptr(d_flat), \
                            out.data_ptr<float>(), \
                            row_width, col_width, bit_width, y_row, z_col, k8, col_splits, \
                            !serial_byte4_reduction); \
                } while (0)

            if (use_byte4_path) {
                if (ni == 1) {
                    if      (k8 <= 4)  LAUNCH_BYTE4_HALF(1, 4);
                    else if (k8 <= 8)  LAUNCH_BYTE4_HALF(1, 8);
                    else               LAUNCH_BYTE4_HALF(1, 16);
                } else if (ni <= 2) {
                    if      (k8 <= 8)  LAUNCH_BYTE4_HALF(2, 8);
                    else               LAUNCH_BYTE4_HALF(2, 16);
                } else { LAUNCH_BYTE4_HALF(4, 16); }
            } else if (use_bitblas_byte2) {
                if (ni == 1) {
                    if      (k8 <= 4)  LAUNCH_BYTE2_HALF(1, 4);
                    else if (k8 <= 8)  LAUNCH_BYTE2_HALF(1, 8);
                    else               LAUNCH_BYTE2_HALF(1, 16);
                } else if (ni <= 2) {
                    if      (k8 <= 8)  LAUNCH_BYTE2_HALF(2, 8);
                    else               LAUNCH_BYTE2_HALF(2, 16);
                } else { LAUNCH_BYTE2_HALF(4, 16); }
            } else if (use_bitblas_byte) {
                if (ni == 1) {
                    if      (k8 <= 4)  LAUNCH_BYTE_HALF(1, 4);
                    else if (k8 <= 8)  LAUNCH_BYTE_HALF(1, 8);
                    else               LAUNCH_BYTE_HALF(1, 16);
                } else if (ni <= 2) {
                    if      (k8 <= 8)  LAUNCH_BYTE_HALF(2, 8);
                    else               LAUNCH_BYTE_HALF(2, 16);
                } else { LAUNCH_BYTE_HALF(4, 16); }
            } else {
                if (ni == 1) {
                    if      (k8 <= 4)  LAUNCH_FLOAT(1, 4);
                    else if (k8 <= 8)  LAUNCH_FLOAT(1, 8);
                    else               LAUNCH_FLOAT(1, 16);
                } else if (ni <= 2) {
                    if      (k8 <= 8)  LAUNCH_FLOAT(2, 8);
                    else               LAUNCH_FLOAT(2, 16);
                } else { LAUNCH_FLOAT(4, 16); }
            }
            #undef LAUNCH_BYTE4_HALF
            #undef LAUNCH_BYTE2_HALF
            #undef LAUNCH_BYTE_HALF
            #undef LAUNCH_X16
            #undef LAUNCH_FLOAT
        }

        if (fuse_out) {
            auto out_h = direct_output
                ? output
                : torch::empty({1, out_f},
                      torch::dtype(x_is_bf16 ? torch::kBFloat16 : torch::kFloat16)
                          .device(X.device()));
            const int threads = 256;
            const int blocks = (out_f + threads - 1) / threads;
            if (x_is_bf16) {
                auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(
                    out_h.data_ptr<at::BFloat16>()) +
                    (direct_output ? output_offset : 0);
                bqq_ws_store_zero_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                    ws.data_ptr<float>(), out_ptr, out_f);
            } else {
                auto* out_ptr = reinterpret_cast<__half*>(
                    out_h.data_ptr<at::Half>()) +
                    (direct_output ? output_offset : 0);
                bqq_ws_store_zero_kernel<__half><<<blocks, threads, 0, stream>>>(
                    ws.data_ptr<float>(), out_ptr, out_f);
            }
            result = direct_output
                ? output.narrow(-1, output_offset, out_f)
                : out_h;
            wrote_direct_output = direct_output;
        } else {
            result = out.reshape({(int)batch, out_f});
        }

    } else {
        /* ── large seq: reconstruct W (popcount) + cuBLAS FP16 ── */
        auto W_half = torch::empty({out_f, in_f},
            torch::dtype(torch::kFloat16).device(X.device()));

        /* reconstruct_W wants Z j-major (the rank bits of a column contiguous).
         * LUT-backed modules only keep the rank-major Z, so transpose it into a
         * scratch buffer here -- transient, and negligible next to W_half. */
        torch::Tensor Z_recon = Z_flat;
        if (z_rank_major) {
            const int64_t B_total = bit_width * row_width * col_width;
            Z_recon = torch::empty({B_total, z_col, k8},
                torch::dtype(torch::kUInt8).device(X.device()));
            const int64_t n_bytes = B_total * z_col * k8;
            const int tblock = 256;
            const int tgrid = (int)((n_bytes + tblock - 1) / tblock);
            bqq_z_rank_to_jmajor_kernel<<<tgrid, tblock, 0, stream>>>(
                Z_flat.data_ptr<uint8_t>(), Z_recon.data_ptr<uint8_t>(),
                (int)B_total, z_col, k8);
        }

        dim3 rblock(16, 16);
        dim3 rgrid((out_f + 15) / 16, (in_f + 15) / 16);

        reconstruct_W_kernel<<<rgrid, rblock, 0, stream>>>(
            Y_flat.data_ptr<uint8_t>(), Z_recon.data_ptr<uint8_t>(),
            half_ptr(a_flat), half_ptr(b_flat),
            half_ptr(c_flat), half_ptr(d_flat),
            reinterpret_cast<__half*>(W_half.data_ptr<at::Half>()),
            row_width, col_width, bit_width, y_row, z_col, k8);

        result = torch::mm(X_2d.to(torch::kFloat16), W_half.t());
    }

    /* ── bias ──────────────────────────────────────────────── */
    if (bias.numel() > 0) {
        if (wrote_direct_output)
            result.add_(bias.to(result.dtype()).to(result.device()));
        else
            result = result + bias.to(result.dtype()).to(result.device());
    }

    if (direct_output && !wrote_direct_output) {
        auto target = output.narrow(-1, output_offset, out_f);
        target.copy_(result.to(X.dtype()).reshape(target.sizes()));
        result = target;
    }

    /* ── restore original leading shape ────────────────────── */
    auto out_shape = X_shape;
    out_shape.back() = out_f;
    return result.reshape(out_shape).to(X.dtype());
}

torch::Tensor bqq_forward_core(
    torch::Tensor Y_flat, torch::Tensor Z_flat, torch::Tensor X,
    torch::Tensor a_flat, torch::Tensor b_flat, torch::Tensor c_flat,
    torch::Tensor d_flat, torch::Tensor bias, torch::Tensor ws,
    int64_t bit_width, int64_t row_width, int64_t col_width,
    int64_t y_row, int64_t z_col, bool z_rank_major)
{
    return bqq_forward_core_impl(
        Y_flat, Z_flat, X, a_flat, b_flat, c_flat, d_flat, bias, ws,
        bit_width, row_width, col_width, y_row, z_col, z_rank_major,
        torch::Tensor(), 0);
}

torch::Tensor bqq_forward_core_out(
    torch::Tensor Y_flat, torch::Tensor Z_flat, torch::Tensor X,
    torch::Tensor a_flat, torch::Tensor b_flat, torch::Tensor c_flat,
    torch::Tensor d_flat, torch::Tensor bias, torch::Tensor ws,
    torch::Tensor output, int64_t output_offset,
    int64_t bit_width, int64_t row_width, int64_t col_width,
    int64_t y_row, int64_t z_col, bool z_rank_major)
{
    bqq_forward_core_impl(
        Y_flat, Z_flat, X, a_flat, b_flat, c_flat, d_flat, bias, ws,
        bit_width, row_width, col_width, y_row, z_col, z_rank_major,
        output, output_offset);
    return output;
}

/* Legacy entry point: takes the 5-D packed layout and per-call converts
 * the coefficients to flat fp16.  Prefer bqq_forward_flat + caching the
 * flat tensors on the module — the conversions here cost ~6 extra CUDA
 * ops per call, which dominates single-token decode. */
torch::Tensor bqq_forward(
    torch::Tensor Y_packed,     /* [bit, row, col, y_row, k8]  uint8 */
    torch::Tensor Z_packed,     /* [bit, row, col, z_col, k8]  uint8 */
    torch::Tensor X,            /* [..., in_features]                */
    torch::Tensor a,            /* [bit, row, col, 1, 1]             */
    torch::Tensor b,            /* [bit, row, col, 1, 1]             */
    torch::Tensor c,            /* [bit, row, col, 1, 1]             */
    torch::Tensor d,            /* [row, col, 1, 1]                  */
    torch::Tensor bias)         /* [out_features] or empty           */
{
    const int bit_width = Y_packed.size(0);
    const int row_width = Y_packed.size(1);
    const int col_width = Y_packed.size(2);
    const int y_row     = Y_packed.size(3);
    const int k8        = Y_packed.size(4);
    const int z_col     = Z_packed.size(3);
    const int B_total   = bit_width * row_width * col_width;

    auto Y_flat = Y_packed.reshape({B_total, y_row, k8}).contiguous();
    auto Z_flat = Z_packed.reshape({B_total, z_col, k8}).contiguous();
    auto a_flat = a.reshape({B_total}).to(torch::kFloat16).contiguous();
    auto b_flat = b.reshape({B_total}).to(torch::kFloat16).contiguous();
    auto c_flat = c.reshape({B_total}).to(torch::kFloat16).contiguous();
    auto d_flat = d.reshape({row_width * col_width}).to(torch::kFloat16).contiguous();

    return bqq_forward_core(Y_flat, Z_flat, X,
                            a_flat, b_flat, c_flat, d_flat, bias,
                            torch::Tensor(),
                            bit_width, row_width, col_width, y_row, z_col,
                            /*z_rank_major=*/false);   /* builds j-major above */
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bqq_forward", &bqq_forward,
          "BQQ forward: one call, no Python tensor manipulation",
          py::arg("Y_packed"), py::arg("Z_packed"), py::arg("X"),
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
          py::arg("bias"));
    m.def("bqq_forward_flat", &bqq_forward_core,
          "BQQ forward with pre-flattened uint8 weights and fp16 coefficients "
          "(cache these on the module to keep dtype conversions off the decode path)",
          py::arg("Y_flat"), py::arg("Z_flat"), py::arg("X"),
          py::arg("a_flat"), py::arg("b_flat"), py::arg("c_flat"),
          py::arg("d_flat"), py::arg("bias"), py::arg("ws"),
          py::arg("bit_width"), py::arg("row_width"), py::arg("col_width"),
          py::arg("y_row"), py::arg("z_col"),
          py::arg("z_rank_major") = false);
    m.def("bqq_forward_flat_out", &bqq_forward_core_out,
          "BQQ forward writing into a caller-provided output slice",
          py::arg("Y_flat"), py::arg("Z_flat"), py::arg("X"),
          py::arg("a_flat"), py::arg("b_flat"), py::arg("c_flat"),
          py::arg("d_flat"), py::arg("bias"), py::arg("ws"),
          py::arg("output"), py::arg("output_offset"),
          py::arg("bit_width"), py::arg("row_width"), py::arg("col_width"),
          py::arg("y_row"), py::arg("z_col"),
          py::arg("z_rank_major") = false);
}
