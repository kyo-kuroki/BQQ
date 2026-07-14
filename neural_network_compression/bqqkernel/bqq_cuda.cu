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

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    return smem[0];
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
                for (int bit = 0; bit < 8; bit++) {
                    float t_k = warp_reduce_sum(t_local[bit]);
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
                for (int bit = 0; bit < 16; bit++) {
                    float t_k = warp_reduce_sum(t_local[bit]);
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

                #pragma unroll
                for (int bit = 0; bit < 32; bit++) {
                    float t_k = warp_reduce_sum(t_local[bit]);
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
                for (int bit = 0; bit < 32; bit++) {
                    float t_k = warp_reduce_sum(t_local[bit]);
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
    int B_total, int col_width, int z_col, int k8)
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

    float t_local = 0.0f;
    float xsum_local = 0.0f;
    for (int j = lane; j < z_col; j += 32) {
        const float xv = load_activation(X, x_base + j);
        const uint8_t zb = Z[(size_t)B_idx * z_col * k8 + j * k8 + byte_k];
        if ((zb >> shift) & 1) t_local += xv;
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

template <int N_I_TILES, int K_CHUNK>
__global__ void bqq_stage2_yt_ksplit_lut_kernel(
    const uint8_t* __restrict__ Y,
    const float*   __restrict__ TLut,
    const float*   __restrict__ Xsum,
    const __half*  __restrict__ a_ptr,
    const __half*  __restrict__ b_ptr,
    float*         __restrict__ partial,
    int row_width, int col_width, int bit_width,
    int y_row, int k8, int row_tiles, int k_splits)
{
    constexpr int ROW_TILE = N_I_TILES * 32;
    const int combined = blockIdx.x;
    const int rt = combined % row_tiles;
    const int tmp0 = combined / row_tiles;
    const int ks = tmp0 % k_splits;
    const int tmp1 = tmp0 / k_splits;
    const int r = tmp1 % row_width;
    const int p = tmp1 / row_width;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int row_start = rt * ROW_TILE;
    const int bk_start = ks * K_CHUNK;
    const int bk_end = min(bk_start + K_CHUNK, k8);

    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        const int i = row_start + it * 32 + lane;
        if (warp_id != it || i >= y_row) continue;

        float acc = 0.0f;
        for (int ci = 0; ci < col_width; ci++) {
            const int rc = r * col_width + ci;
            const int B_idx = p * row_width * col_width + rc;
            const uint8_t* yp = Y + (size_t)B_idx * y_row * k8 + i * k8;
            const float* lut = TLut + (size_t)B_idx * k8 * 256;
            float part = 0.0f;
            int y_count = 0;
            for (int bk = bk_start; bk < bk_end; bk++) {
                const uint8_t yb = yp[bk];
                y_count += __popc((unsigned)yb);
                part += lut[bk * 256 + (int)yb];
            }
            const float xsum = Xsum[B_idx];
            acc += __half2float(a_ptr[B_idx]) * part
                 + __half2float(b_ptr[B_idx]) * xsum * (float)y_count;
        }

        partial[(((size_t)p * k_splits + ks) * row_width + r) * y_row + i] = acc;
    }
}

__global__ void bqq_stage3_reduce_ksplit_kernel(
    const float* __restrict__ partial,
    const float* __restrict__ Tsum,
    const float* __restrict__ Xsum,
    const __half* __restrict__ c_ptr,
    const __half* __restrict__ d_ptr,
    float*       __restrict__ out,
    int row_width, int col_width, int bit_width, int y_row, int k_splits)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = row_width * y_row;
    if (idx >= total) return;

    const int r = idx / y_row;
    const int i = idx - r * y_row;
    float acc = 0.0f;

    for (int p = 0; p < bit_width; p++) {
        for (int ks = 0; ks < k_splits; ks++) {
            acc += partial[(((size_t)p * k_splits + ks) * row_width + r) * y_row + i];
        }
        for (int ci = 0; ci < col_width; ci++) {
            const int B_idx = p * row_width * col_width + r * col_width + ci;
            acc += __half2float(c_ptr[B_idx]) * Tsum[B_idx];
        }
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

template <int N_I_TILES>
__global__ void bqq_stage2_yt_partial_lut_kernel(
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
    constexpr int ROW_TILE = N_I_TILES * 32;
    const int combined = blockIdx.x;
    const int rt = combined % row_tiles;
    const int tmp = combined / row_tiles;
    const int r = tmp % row_width;
    const int p = tmp / row_width;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
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
            const float* lut = TLut + (size_t)B_idx * k8 * 256;
            float part = 0.0f;
            int y_count = 0;
            for (int bk = 0; bk < k8; bk++) {
                const uint8_t yb = yp[bk];
                y_count += __popc((unsigned)yb);
                part += lut[bk * 256 + (int)yb];
            }
            const float xsum = Xsum[B_idx];
            acc += __half2float(a_ptr[B_idx]) * part
                 + __half2float(b_ptr[B_idx]) * xsum * (float)y_count
                 + __half2float(c_ptr[B_idx]) * Tsum[B_idx];
        }

        partial[(size_t)p * row_width * y_row + r * y_row + i] = acc;
    }
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


/* ═══════════════════════════════════════════════════════════════════
 * Decode experiment: explicit Z@x then Y@t
 * ═══════════════════════════════════════════════════════════════════ */

template <int K8_MAX>
__global__ void bqq_decode_two_stage_kernel(
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
    constexpr int K_MAX = K8_MAX * 8;
    __shared__ float t_vals[K_MAX];
    __shared__ float reduce_buf[256];

    const int combined = blockIdx.x;
    const int r = combined / col_splits;
    const int cs = combined % col_splits;
    const int n = blockIdx.y;
    const int tid = threadIdx.x;

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
        for (int j = tid; j < z_col; j += blockDim.x) {
            xsum_local += X[x_base + j];
        }
        const float xsum = block_reduce_sum(xsum_local, reduce_buf);

        for (int p = 0; p < bit_width; p++) {
            const int B_idx = p * row_width * col_width + rc;
            const float a_val = __half2float(a_ptr[B_idx]);
            const float b_val = __half2float(b_ptr[B_idx]);
            const float c_val = __half2float(c_ptr[B_idx]);
            float t_sum = 0.0f;

            for (int l = 0; l < K_MAX; l++) {
                if (l >= n_inner) break;
                const int byte_k = l >> 3;
                const int shift = 7 - (l & 7);
                float t_local = 0.0f;
                for (int j = tid; j < z_col; j += blockDim.x) {
                    const uint8_t zb = Z[(size_t)B_idx * z_col * k8 + j * k8 + byte_k];
                    if ((zb >> shift) & 1) t_local += X[x_base + j];
                }
                const float t_l = block_reduce_sum(t_local, reduce_buf);
                if (tid == 0) t_vals[l] = t_l;
                t_sum += t_l;
            }
            __syncthreads();

            if (has_i) {
                const uint8_t* yp = Y + (size_t)B_idx * y_row * k8 + i * k8;
                float part = 0.0f;
                for (int l = 0; l < K_MAX; l++) {
                    if (l >= n_inner) break;
                    const int byte_k = l >> 3;
                    const int shift = 7 - (l & 7);
                    if ((yp[byte_k] >> shift) & 1) {
                        part += a_val * t_vals[l] + b_val * xsum;
                    }
                }
                acc += part + c_val * t_sum;
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

template <int N_WARPS, int K8_MAX>
__global__ void bqq_decode_grouped_scatter_kernel(
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
    constexpr int K_MAX = K8_MAX * 8;
    extern __shared__ float smem[];
    float* row_acc = smem;                         // [y_row]
    float* warp_partials = row_acc + y_row;        // [N_WARPS, y_row]
    float* warp_t_sums = warp_partials + N_WARPS * y_row; // [N_WARPS]

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

    for (int i = tid; i < y_row; i += blockDim.x) {
        row_acc[i] = 0.0f;
    }
    __syncthreads();

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

            for (int i = tid; i < N_WARPS * y_row; i += blockDim.x) {
                warp_partials[i] = 0.0f;
            }
            if (tid < N_WARPS) warp_t_sums[tid] = 0.0f;
            __syncthreads();

            for (int l_base = 0; l_base < K_MAX; l_base += N_WARPS) {
                const int l = l_base + warp_id;
                float t_local = 0.0f;
                if (l < n_inner) {
                    const int byte_k = l >> 3;
                    const int shift = 7 - (l & 7);
                    for (int j = lane; j < z_col; j += 32) {
                        const uint8_t zb =
                            Z[(size_t)B_idx * z_col * k8 + j * k8 + byte_k];
                        if ((zb >> shift) & 1) t_local += X[x_base + j];
                    }
                }
                float t_l = warp_reduce_sum(t_local);

                if (lane == 0 && l < n_inner) {
                    const int byte_k = l >> 3;
                    const int shift = 7 - (l & 7);
                    const float contrib = a_val * t_l + b_val * xsum;
                    float* my_partials = warp_partials + warp_id * y_row;
                    warp_t_sums[warp_id] += t_l;
                    for (int i = 0; i < y_row; i++) {
                        const uint8_t yb =
                            Y[(size_t)B_idx * y_row * k8 + i * k8 + byte_k];
                        if ((yb >> shift) & 1) my_partials[i] += contrib;
                    }
                }
            }
            __syncthreads();

            float t_sum = 0.0f;
            if (tid == 0) {
                for (int w = 0; w < N_WARPS; w++) t_sum += warp_t_sums[w];
                warp_t_sums[0] = t_sum;
            }
            __syncthreads();
            t_sum = warp_t_sums[0];

            for (int i = tid; i < y_row; i += blockDim.x) {
                float row_val = c_val * t_sum;
                #pragma unroll
                for (int w = 0; w < N_WARPS; w++) {
                    row_val += warp_partials[w * y_row + i];
                }
                row_acc[i] += row_val;
            }
            __syncthreads();
        }

        const float d_term = __half2float(d_ptr[rc]) * xsum;
        for (int i = tid; i < y_row; i += blockDim.x) {
            row_acc[i] += d_term;
        }
        __syncthreads();
    }

    for (int i = tid; i < y_row; i += blockDim.x) {
        const size_t idx = (size_t)n * row_width * y_row + r * y_row + i;
        if (col_splits > 1)
            atomicAdd(&out[idx], row_acc[i]);
        else
            out[idx] = row_acc[i];
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

torch::Tensor reconstruct_W(
    torch::Tensor Y_packed, torch::Tensor Z_packed,
    torch::Tensor a, torch::Tensor b,
    torch::Tensor c, torch::Tensor d,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    int out_f = row_width * y_row;
    int in_f  = col_width * z_col;

    auto W = torch::empty({out_f, in_f},
        torch::dtype(torch::kFloat16).device(Y_packed.device()));

    dim3 block(16, 16);
    dim3 grid((out_f + 15) / 16, (in_f + 15) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto a_h = a.to(torch::kFloat16).contiguous();
    auto b_h = b.to(torch::kFloat16).contiguous();
    auto c_h = c.to(torch::kFloat16).contiguous();
    auto d_h = d.to(torch::kFloat16).contiguous();

    reconstruct_W_kernel<<<grid, block, 0, stream>>>(
        Y_packed.data_ptr<uint8_t>(),
        Z_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(a_h.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b_h.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(c_h.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(d_h.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(W.data_ptr<at::Half>()),
        row_width, col_width, bit_width,
        y_row, z_col, k8);

    return W;
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
 * seq_len >  32 → reconstruct_W (popcount) + cuBLAS FP16 matmul
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
torch::Tensor bqq_forward_core(
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
    int64_t y_row_, int64_t z_col_)
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
    const int k8        = Y_flat.size(2);
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

    torch::Tensor result;

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
                    B_total, col_width, z_col, k8);
            else
                bqq_stage1_ztx_kernel<__half, NW><<<s1_grid, s1_block, 0, stream>>>(
                    Z_flat.data_ptr<uint8_t>(),
                    half_ptr(X_view_half),
                    T.data_ptr<float>(),
                    Tsum.data_ptr<float>(),
                    Xsum.data_ptr<float>(),
                    B_total, col_width, z_col, k8);

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
        const bool use_bitblas_fp16 =
            decode_kernel && std::strcmp(decode_kernel, "bitblas_fp16") == 0 &&
            x_is_16bit;
        const bool use_bitblas_byte4 =
            ((decode_kernel && std::strcmp(decode_kernel, "bitblas_byte4") == 0) ||
             (decode_kernel == nullptr && bit_width <= 2)) &&
            x_is_16bit && z_col > 32 && k8 > 3;
        const bool use_bitblas_byte2 =
            ((decode_kernel && std::strcmp(decode_kernel, "bitblas_byte2") == 0) ||
             (decode_kernel == nullptr && !use_bitblas_byte4 && bit_width <= 2)) &&
            x_is_16bit && z_col > 32 && k8 > 1;
        const bool use_bitblas_byte =
            ((decode_kernel && std::strcmp(decode_kernel, "bitblas_byte") == 0) ||
             (decode_kernel == nullptr && !use_bitblas_byte4 && !use_bitblas_byte2)) &&
            x_is_16bit && z_col > 32;

        auto X_view = (use_bitblas_fp16 || use_bitblas_byte || use_bitblas_byte2 || use_bitblas_byte4)
            ? torch::Tensor()
            : X_2d.reshape({(int)batch, col_width, z_col})
                  .to(torch::kFloat32).contiguous();
        auto X_view_half = (use_bitblas_fp16 || use_bitblas_byte || use_bitblas_byte2 || use_bitblas_byte4)
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
        } else if (use_bitblas_byte || use_bitblas_byte2 || use_bitblas_byte4) {
            const int max_splits = use_bitblas_byte ? 16 : 8;
            col_splits = min(col_splits, max_splits);
            while (col_splits > 1 && col_width % col_splits != 0) col_splits--;
        }

        const bool use_grouped_scatter =
            decode_kernel && std::strcmp(decode_kernel, "grouped") == 0 &&
            y_row <= 256 && k8 <= 16;

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
            out = (col_splits > 1 || use_grouped_scatter)
                ? torch::zeros({(int)batch, row_width, y_row},
                      torch::dtype(torch::kFloat32).device(X.device()))
                : torch::empty({(int)batch, row_width, y_row},
                      torch::dtype(torch::kFloat32).device(X.device()));
        }

        dim3 grid(row_width * col_splits, batch);
        const bool use_auto_decode_kernel = decode_kernel == nullptr;
        const bool use_two_stage = decode_kernel &&
            std::strcmp(decode_kernel, "two_stage") == 0 &&
            y_row <= 256;
        const bool use_two_stage_warp =
            ((decode_kernel && std::strcmp(decode_kernel, "two_stage_warp") == 0) ||
             (use_auto_decode_kernel && !use_bitblas_byte && !use_bitblas_byte2 &&
              !use_bitblas_byte4 && z_col >= 128)) &&
            y_row <= 256 && k8 <= 16;

        if (use_grouped_scatter) {
            constexpr int GW = 8;
            dim3 block(GW * 32);
            int smem = (y_row + GW * y_row + GW) * static_cast<int>(sizeof(float));
            #define LAUNCH_GROUPED(K8M) \
                bqq_decode_grouped_scatter_kernel<GW, K8M><<<grid, block, smem, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                    X_view.data_ptr<float>(), \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits)

            if      (k8 <= 4)  LAUNCH_GROUPED(4);
            else if (k8 <= 8)  LAUNCH_GROUPED(8);
            else               LAUNCH_GROUPED(16);
            #undef LAUNCH_GROUPED
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
        } else if (use_two_stage) {
            dim3 block(256);
            #define LAUNCH_TWO_STAGE(K8M) \
                bqq_decode_two_stage_kernel<K8M><<<grid, block, 0, stream>>>( \
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(), \
                    X_view.data_ptr<float>(), \
                    half_ptr(a_flat), half_ptr(b_flat), \
                    half_ptr(c_flat), half_ptr(d_flat), \
                    out.data_ptr<float>(), \
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits)

            if      (k8 <= 4)  LAUNCH_TWO_STAGE(4);
            else if (k8 <= 8)  LAUNCH_TWO_STAGE(8);
            else               LAUNCH_TWO_STAGE(16);
            #undef LAUNCH_TWO_STAGE
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

            #define LAUNCH_HALF(NI, K8M)  LAUNCH_X16(bqq_forward_kernel, NI, K8M)
            #define LAUNCH_BYTE_HALF(NI, K8M)  LAUNCH_X16(bqq_forward_byte_kernel, NI, K8M)
            #define LAUNCH_BYTE2_HALF(NI, K8M) LAUNCH_X16(bqq_forward_byte2_kernel, NI, K8M)
            #define LAUNCH_BYTE4_HALF(NI, K8M) LAUNCH_X16(bqq_forward_byte4_kernel, NI, K8M)

            if (use_bitblas_byte4) {
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
            } else if (use_bitblas_fp16) {
                if (ni == 1) {
                    if      (k8 <= 4)  LAUNCH_HALF(1, 4);
                    else if (k8 <= 8)  LAUNCH_HALF(1, 8);
                    else               LAUNCH_HALF(1, 16);
                } else if (ni <= 2) {
                    if      (k8 <= 8)  LAUNCH_HALF(2, 8);
                    else               LAUNCH_HALF(2, 16);
                } else { LAUNCH_HALF(4, 16); }
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
            #undef LAUNCH_HALF
            #undef LAUNCH_X16
            #undef LAUNCH_FLOAT
        }

        if (fuse_out) {
            auto out_h = torch::empty({1, out_f},
                torch::dtype(x_is_bf16 ? torch::kBFloat16 : torch::kFloat16)
                    .device(X.device()));
            const int threads = 256;
            const int blocks = (out_f + threads - 1) / threads;
            if (x_is_bf16)
                bqq_ws_store_zero_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
                    ws.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(out_h.data_ptr<at::BFloat16>()),
                    out_f);
            else
                bqq_ws_store_zero_kernel<__half><<<blocks, threads, 0, stream>>>(
                    ws.data_ptr<float>(),
                    reinterpret_cast<__half*>(out_h.data_ptr<at::Half>()),
                    out_f);
            result = out_h;
        } else {
            result = out.reshape({(int)batch, out_f});
        }

    } else {
        /* ── large seq: reconstruct W (popcount) + cuBLAS FP16 ── */
        auto W_half = torch::empty({out_f, in_f},
            torch::dtype(torch::kFloat16).device(X.device()));

        dim3 rblock(16, 16);
        dim3 rgrid((out_f + 15) / 16, (in_f + 15) / 16);

        reconstruct_W_kernel<<<rgrid, rblock, 0, stream>>>(
            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(),
            half_ptr(a_flat), half_ptr(b_flat),
            half_ptr(c_flat), half_ptr(d_flat),
            reinterpret_cast<__half*>(W_half.data_ptr<at::Half>()),
            row_width, col_width, bit_width, y_row, z_col, k8);

        result = torch::mm(X_2d.to(torch::kFloat16), W_half.t());
    }

    /* ── bias ──────────────────────────────────────────────── */
    if (bias.numel() > 0)
        result = result + bias.to(result.dtype()).to(result.device());

    /* ── restore original leading shape ────────────────────── */
    auto out_shape = X_shape;
    out_shape.back() = out_f;
    return result.reshape(out_shape).to(X.dtype());
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
                            bit_width, row_width, col_width, y_row, z_col);
}


/* ═══════════════════════════════════════════════════════════════════
 * L2 cache utilities (experimental)
 * ═══════════════════════════════════════════════════════════════════ */

__global__ void prefetch_l2_kernel(const char* __restrict__ ptr, size_t nbytes) {
    const size_t tid = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid * 16; i < nbytes; i += stride * 16) {
        if (i + 16 <= nbytes) {
            float4 v = *reinterpret_cast<const float4*>(ptr + i);
            asm volatile("" :: "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w));
        }
    }
}

void prefetch_tensors_to_l2(
    torch::Tensor Y_packed, torch::Tensor Z_packed,
    torch::Tensor a, torch::Tensor b,
    torch::Tensor c, torch::Tensor d)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto launch = [stream](const void* ptr, size_t nbytes) {
        if (nbytes == 0) return;
        int threads = 256;
        int blocks = min((int)((nbytes + threads * 16 - 1) / (threads * 16)), 256);
        prefetch_l2_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const char*>(ptr), nbytes);
    };
    launch(Y_packed.data_ptr(), Y_packed.nbytes());
    launch(Z_packed.data_ptr(), Z_packed.nbytes());
    launch(a.data_ptr(), a.nbytes());
    launch(b.data_ptr(), b.nbytes());
    launch(c.data_ptr(), c.nbytes());
    launch(d.data_ptr(), d.nbytes());
}

void set_l2_persistence(torch::Tensor tensor, float hit_ratio) {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr  = tensor.data_ptr();
    attr.accessPolicyWindow.num_bytes = tensor.nbytes();
    attr.accessPolicyWindow.hitRatio  = hit_ratio;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(
        at::cuda::getCurrentCUDAStream(),
        cudaStreamAttributeAccessPolicyWindow, &attr);
}

void reset_l2_persistence() {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(
        at::cuda::getCurrentCUDAStream(),
        cudaStreamAttributeAccessPolicyWindow, &attr);
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
          py::arg("y_row"), py::arg("z_col"));
    m.def("prefetch_tensors_to_l2", &prefetch_tensors_to_l2,
          "Prefetch weight tensors into L2 cache");
    m.def("set_l2_persistence", &set_l2_persistence,
          "Pin tensor in L2 cache (Ampere+)",
          py::arg("tensor"), py::arg("hit_ratio") = 1.0f);
    m.def("reset_l2_persistence", &reset_l2_persistence,
          "Remove L2 persistence policy");
}
