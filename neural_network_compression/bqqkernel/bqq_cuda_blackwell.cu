/*
 * Blackwell-safe fallback BQQ CUDA extension.
 *
 * The optimized bqq_cuda.cu currently fails ptxas register allocation on
 * sm_120 in this environment.  This file keeps a compact reconstruct-W +
 * torch::mm implementation for bqq_forward_flat so older GPUs can continue
 * using the existing optimized kernels unchanged.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

static inline const __half* half_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());
}

__global__ void bqq_blackwell_reconstruct_w_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const __half* __restrict__ a_ptr,
    const __half* __restrict__ b_ptr,
    const __half* __restrict__ c_ptr,
    const __half* __restrict__ d_ptr,
    __half* __restrict__ W_out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int in_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_f = row_width * y_row;
    const int in_f = col_width * z_col;
    if (out_idx >= out_f || in_idx >= in_f) return;

    const int r = out_idx / y_row;
    const int i = out_idx - r * y_row;
    const int c = in_idx / z_col;
    const int j = in_idx - c * z_col;

    float val = __half2float(d_ptr[r * col_width + c]);
    for (int bit = 0; bit < bit_width; ++bit) {
        const int B = (bit * row_width + r) * col_width + c;
        const uint8_t* y_ptr = Y + ((size_t)B * y_row + i) * k8;
        const uint8_t* z_ptr = Z + ((size_t)B * z_col + j) * k8;

        int inner = 0;
        int ys = 0;
        int zs = 0;
        for (int byte_k = 0; byte_k < k8; ++byte_k) {
            const uint8_t yb = y_ptr[byte_k];
            const uint8_t zb = z_ptr[byte_k];
            inner += __popc((unsigned)(yb & zb));
            ys += __popc((unsigned)yb);
            zs += __popc((unsigned)zb);
        }

        val += __half2float(a_ptr[B]) * (float)inner
             + __half2float(b_ptr[B]) * (float)ys
             + __half2float(c_ptr[B]) * (float)zs;
    }
    W_out[(size_t)out_idx * in_f + in_idx] = __float2half(val);
}

torch::Tensor bqq_forward_core(
    torch::Tensor Y_flat,
    torch::Tensor Z_flat,
    torch::Tensor X,
    torch::Tensor a_flat,
    torch::Tensor b_flat,
    torch::Tensor c_flat,
    torch::Tensor d_flat,
    torch::Tensor bias,
    torch::Tensor ws,
    int64_t bit_width_, int64_t row_width_, int64_t col_width_,
    int64_t y_row_, int64_t z_col_)
{
    TORCH_CHECK(Y_flat.is_cuda() && Z_flat.is_cuda() && X.is_cuda(),
                "bqq_forward_flat Blackwell fallback expects CUDA tensors");
    TORCH_CHECK(a_flat.scalar_type() == torch::kHalf &&
                b_flat.scalar_type() == torch::kHalf &&
                c_flat.scalar_type() == torch::kHalf &&
                d_flat.scalar_type() == torch::kHalf,
                "bqq_forward_flat: a/b/c/d must be float16");

    const int bit_width = (int)bit_width_;
    const int row_width = (int)row_width_;
    const int col_width = (int)col_width_;
    const int y_row = (int)y_row_;
    const int z_col = (int)z_col_;
    const int k8 = (int)Y_flat.size(2);
    const int out_f = row_width * y_row;
    const int in_f = col_width * z_col;

    auto x_shape = X.sizes().vec();
    int64_t batch = 1;
    for (int i = 0; i < (int)x_shape.size() - 1; ++i) batch *= x_shape[i];
    auto X_2d = X.reshape({batch, in_f});

    auto W_half = torch::empty({out_f, in_f},
        torch::dtype(torch::kFloat16).device(X.device()));
    dim3 block(16, 16);
    dim3 grid((out_f + block.x - 1) / block.x,
              (in_f + block.y - 1) / block.y);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bqq_blackwell_reconstruct_w_kernel<<<grid, block, 0, stream>>>(
        Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(),
        half_ptr(a_flat), half_ptr(b_flat), half_ptr(c_flat), half_ptr(d_flat),
        reinterpret_cast<__half*>(W_half.data_ptr<at::Half>()),
        row_width, col_width, bit_width, y_row, z_col, k8);

    auto result = torch::mm(X_2d.to(torch::kFloat16), W_half.t());
    if (bias.numel() > 0) {
        result = result + bias.to(result.dtype()).to(result.device());
    }
    x_shape.back() = out_f;
    return result.reshape(x_shape).to(X.dtype());
}

torch::Tensor bqq_forward(
    torch::Tensor Y_packed,
    torch::Tensor Z_packed,
    torch::Tensor X,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c,
    torch::Tensor d,
    torch::Tensor bias)
{
    const int bit_width = Y_packed.size(0);
    const int row_width = Y_packed.size(1);
    const int col_width = Y_packed.size(2);
    const int y_row = Y_packed.size(3);
    const int z_col = Z_packed.size(3);
    const int B_total = bit_width * row_width * col_width;

    auto Y_flat = Y_packed.reshape({B_total, y_row, Y_packed.size(4)}).contiguous();
    auto Z_flat = Z_packed.reshape({B_total, z_col, Z_packed.size(4)}).contiguous();
    auto a_flat = a.reshape({B_total}).to(torch::kFloat16).contiguous();
    auto b_flat = b.reshape({B_total}).to(torch::kFloat16).contiguous();
    auto c_flat = c.reshape({B_total}).to(torch::kFloat16).contiguous();
    auto d_flat = d.reshape({row_width * col_width}).to(torch::kFloat16).contiguous();
    return bqq_forward_core(Y_flat, Z_flat, X, a_flat, b_flat, c_flat, d_flat,
                            bias, torch::Tensor(), bit_width, row_width,
                            col_width, y_row, z_col);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bqq_forward", &bqq_forward, "Blackwell BQQ fallback forward");
    m.def("bqq_forward_flat", &bqq_forward_core,
          "Blackwell BQQ fallback forward with flattened weights");
}
