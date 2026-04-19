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
 *   2. Grid splitting over col_width for high SM occupancy (~48 warps/SM)
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
#include <cstdint>


/* ── warp-level sum ────────────────────────────────────────────── */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


/* ═══════════════════════════════════════════════════════════════════
 * Main kernel: grid-split + uint32 bulk loads
 * ═══════════════════════════════════════════════════════════════════ */

template <int N_WARPS, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_kernel(
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

        const float x_val = (lane < z_col) ? X[x_base + lane] : 0.0f;
        float xsum = warp_reduce_sum(x_val);
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];
            float t_sum = 0.0f;

            /* uint32 bulk load: Z/Y bytes → registers */
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

            /* inner loop: all data in registers */
            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;
                const int w_idx = bk >> 2;
                const int b_shift = (bk & 3) << 3;
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
    const float*   __restrict__ a_ptr,  /* [B_total] */
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,  /* [row_width * col_width] */
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
        val += a_ptr[B] * (float)inner
             + b_ptr[B] * (float)ys
             + c_ptr[B] * (float)zs;
    }
    val += d_ptr[r * col_width + c];
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

    reconstruct_W_kernel<<<grid, block>>>(
        Y_packed.data_ptr<uint8_t>(),
        Z_packed.data_ptr<uint8_t>(),
        a.data_ptr<float>(), b.data_ptr<float>(),
        c.data_ptr<float>(), d.data_ptr<float>(),
        reinterpret_cast<__half*>(W.data_ptr<at::Half>()),
        row_width, col_width, bit_width,
        y_row, z_col, k8);

    return W;
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
    int mode)
{
    auto yp = Y_packed.data_ptr<uint8_t>();
    auto zp = Z_packed.data_ptr<uint8_t>();
    auto xp = X.data_ptr<float>();
    auto ap = a.data_ptr<float>();
    auto bp = b.data_ptr<float>();
    auto cp = c.data_ptr<float>();
    auto dp = d.data_ptr<float>();

    int ni = (y_row + 31) / 32;

    constexpr int NW = 4;
    int sm_count = 56;
    int col_splits = max(1, (48 * sm_count + row_width * NW - 1)
                            / (row_width * NW));
    col_splits = min(col_splits, col_width);
    while (col_splits > 1 && col_width % col_splits != 0)
        col_splits--;

    auto out = torch::zeros({batch, row_width, y_row},
        torch::dtype(torch::kFloat32).device(X.device()));
    auto op = out.data_ptr<float>();

    dim3 grid(row_width * col_splits, batch);
    dim3 block(NW * 32);
    int smem = NW * 32 * sizeof(float);

    #define LAUNCH(NI, K8M) \
        bqq_forward_kernel<NW, NI, K8M><<<grid, block, smem>>>( \
            yp, zp, xp, ap, bp, cp, dp, op, \
            row_width, col_width, bit_width, y_row, z_col, k8, \
            col_splits)

    if (ni == 1) {
        if      (k8 <= 4)  LAUNCH(1, 4);
        else if (k8 <= 8)  LAUNCH(1, 8);
        else               LAUNCH(1, 16);
    } else if (ni <= 2) {
        if      (k8 <= 8)  LAUNCH(2, 8);
        else               LAUNCH(2, 16);
    } else {
        LAUNCH(4, 16);
    }
    #undef LAUNCH

    return out;
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
    auto launch = [](const void* ptr, size_t nbytes) {
        if (nbytes == 0) return;
        int threads = 256;
        int blocks = min((int)((nbytes + threads * 16 - 1) / (threads * 16)), 256);
        prefetch_l2_kernel<<<blocks, threads>>>(
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
    m.def("bqq_forward_cuda", &bqq_forward_cuda,
          "BQQ fused forward CUDA kernel",
          py::arg("Y_packed"), py::arg("Z_packed"), py::arg("X"),
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("d"),
          py::arg("batch"), py::arg("row_width"), py::arg("col_width"),
          py::arg("bit_width"), py::arg("y_row"), py::arg("z_col"),
          py::arg("k8"), py::arg("mode") = 0);
    m.def("reconstruct_W", &reconstruct_W,
          "Reconstruct W matrix via AND+popcount (returns FP16)");
    m.def("prefetch_tensors_to_l2", &prefetch_tensors_to_l2,
          "Prefetch weight tensors into L2 cache");
    m.def("set_l2_persistence", &set_l2_persistence,
          "Pin tensor in L2 cache (Ampere+)",
          py::arg("tensor"), py::arg("hit_ratio") = 1.0f);
    m.def("reset_l2_persistence", &reset_l2_persistence,
          "Remove L2 persistence policy");
}
