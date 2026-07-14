/*
 * Blackwell BQQ CUDA extension.
 *
 * The general bqq_cuda.cu exceeds ptxas resource limits on sm_120. This file
 * uses a compact fused packed GEMV for single-token decode and retains
 * reconstruct-W + torch::mm only for prefill.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

static inline const __half* half_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());
}

static inline const __nv_bfloat16* bf16_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

template <typename T>
__device__ __forceinline__ float load_x(const T* ptr, size_t idx);

template <>
__device__ __forceinline__ float load_x<__half>(const __half* ptr, size_t idx) {
    return __half2float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_x<__nv_bfloat16>(
    const __nv_bfloat16* ptr, size_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ T store_x(float value);

template <>
__device__ __forceinline__ __half store_x<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_x<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

/* One block computes one output group and one column split. The packed Z@x
 * intermediates are shared by every output row, avoiding dense W materialization. */
template <typename X_T>
__global__ void bqq_blackwell_decode_kernel(
    const uint8_t* __restrict__ Y,
    const uint8_t* __restrict__ Z,
    const X_T* __restrict__ X,
    const __half* __restrict__ a_ptr,
    const __half* __restrict__ b_ptr,
    const __half* __restrict__ c_ptr,
    const __half* __restrict__ d_ptr,
    float* __restrict__ out,
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8, int col_splits)
{
    extern __shared__ float t_values[];
    __shared__ float x_sum;

    const int split = blockIdx.x % col_splits;
    const int row = blockIdx.x / col_splits;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int n_inner = k8 * 8;
    const int cols_per_split = (col_width + col_splits - 1) / col_splits;
    const int col_begin = split * cols_per_split;
    const int col_end = min(col_begin + cols_per_split, col_width);
    const int output_i = tid;
    float acc = 0.0f;

    for (int col = col_begin; col < col_end; ++col) {
        const int x_base = batch_idx * col_width * z_col + col * z_col;
        if (tid == 0) {
            float sum = 0.0f;
            for (int j = 0; j < z_col; ++j) sum += load_x(X, x_base + j);
            x_sum = sum;
        }

        for (int bit = 0; bit < bit_width; ++bit) {
            const int packed_block = (bit * row_width + row) * col_width + col;
            for (int inner = tid; inner < n_inner; inner += blockDim.x) {
                const int byte_k = inner >> 3;
                const int shift = 7 - (inner & 7);
                float sum = 0.0f;
                for (int j = 0; j < z_col; ++j) {
                    const uint8_t z_byte =
                        Z[((size_t)packed_block * z_col + j) * k8 + byte_k];
                    if ((z_byte >> shift) & 1) sum += load_x(X, x_base + j);
                }
                t_values[inner] = sum;
            }
            __syncthreads();

            if (output_i < y_row) {
                const uint8_t* y_ptr =
                    Y + ((size_t)packed_block * y_row + output_i) * k8;
                float selected_t = 0.0f;
                float t_sum = 0.0f;
                int y_count = 0;
                for (int byte_k = 0; byte_k < k8; ++byte_k) {
                    const uint8_t y_byte = y_ptr[byte_k];
                    y_count += __popc((unsigned)y_byte);
                    #pragma unroll
                    for (int bit_in_byte = 0; bit_in_byte < 8; ++bit_in_byte) {
                        const float t = t_values[byte_k * 8 + bit_in_byte];
                        t_sum += t;
                        if ((y_byte >> (7 - bit_in_byte)) & 1) selected_t += t;
                    }
                }
                acc += __half2float(a_ptr[packed_block]) * selected_t
                     + __half2float(b_ptr[packed_block]) * x_sum * (float)y_count
                     + __half2float(c_ptr[packed_block]) * t_sum;
            }
            __syncthreads();
        }

        if (output_i < y_row) {
            acc += __half2float(d_ptr[row * col_width + col]) * x_sum;
        }
    }

    if (output_i < y_row) {
        const size_t output_idx =
            (size_t)batch_idx * row_width * y_row + row * y_row + output_i;
        if (col_splits > 1) atomicAdd(&out[output_idx], acc);
        else out[output_idx] = acc;
    }
}

template <typename OUT_T>
__global__ void bqq_blackwell_store_zero_kernel(
    float* __restrict__ workspace, OUT_T* __restrict__ output, int count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = store_x<OUT_T>(workspace[idx]);
        workspace[idx] = 0.0f;
    }
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
                "Blackwell bqq_forward_flat expects CUDA tensors");
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool x_is_bf16 = X.scalar_type() == torch::kBFloat16;
    const char* force_reconstruct_env =
        std::getenv("BQQ_BLACKWELL_FORCE_RECONSTRUCT");
    const bool force_reconstruct = force_reconstruct_env != nullptr &&
        std::atoi(force_reconstruct_env) != 0;
    const bool fused_decode = !force_reconstruct && batch == 1 &&
        y_row <= 256 && k8 <= 128 &&
        (X.scalar_type() == torch::kFloat16 || x_is_bf16);
    torch::Tensor result;

    if (fused_decode) {
        const bool valid_workspace =
            ws.defined() && ws.is_cuda() && ws.scalar_type() == torch::kFloat32 &&
            ws.numel() == out_f;
        auto workspace = valid_workspace
            ? ws.reshape({1, row_width, y_row})
            : torch::zeros({1, row_width, y_row},
                  torch::dtype(torch::kFloat32).device(X.device()));

        int col_splits = min(col_width, 4);
        if (const char* value = std::getenv("BQQ_CUDA_COL_SPLITS")) {
            const int requested = std::atoi(value);
            if (requested > 0) col_splits = min(col_width, requested);
        }
        const int n_inner = k8 * 8;
        int threads = max(y_row, min(n_inner, 256));
        threads = min(256, ((threads + 31) / 32) * 32);
        dim3 decode_grid(row_width * col_splits, 1);
        dim3 decode_block(threads);
        const size_t shared_bytes = (size_t)n_inner * sizeof(float);

        if (x_is_bf16) {
            bqq_blackwell_decode_kernel<__nv_bfloat16>
                <<<decode_grid, decode_block, shared_bytes, stream>>>(
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(),
                    bf16_ptr(X_2d), half_ptr(a_flat), half_ptr(b_flat),
                    half_ptr(c_flat), half_ptr(d_flat), workspace.data_ptr<float>(),
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits);
        } else {
            bqq_blackwell_decode_kernel<__half>
                <<<decode_grid, decode_block, shared_bytes, stream>>>(
                    Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(),
                    half_ptr(X_2d), half_ptr(a_flat), half_ptr(b_flat),
                    half_ptr(c_flat), half_ptr(d_flat), workspace.data_ptr<float>(),
                    row_width, col_width, bit_width, y_row, z_col, k8, col_splits);
        }

        result = torch::empty({1, out_f},
            torch::dtype(x_is_bf16 ? torch::kBFloat16 : torch::kFloat16)
                .device(X.device()));
        const int store_threads = 256;
        const int store_blocks = (out_f + store_threads - 1) / store_threads;
        if (x_is_bf16) {
            bqq_blackwell_store_zero_kernel<__nv_bfloat16>
                <<<store_blocks, store_threads, 0, stream>>>(
                    workspace.data_ptr<float>(),
                    reinterpret_cast<__nv_bfloat16*>(
                        result.data_ptr<at::BFloat16>()), out_f);
        } else {
            bqq_blackwell_store_zero_kernel<__half>
                <<<store_blocks, store_threads, 0, stream>>>(
                    workspace.data_ptr<float>(),
                    reinterpret_cast<__half*>(result.data_ptr<at::Half>()), out_f);
        }
    } else {
        auto W_half = torch::empty({out_f, in_f},
            torch::dtype(torch::kFloat16).device(X.device()));
        dim3 block(16, 16);
        dim3 grid((out_f + block.x - 1) / block.x,
                  (in_f + block.y - 1) / block.y);
        bqq_blackwell_reconstruct_w_kernel<<<grid, block, 0, stream>>>(
            Y_flat.data_ptr<uint8_t>(), Z_flat.data_ptr<uint8_t>(),
            half_ptr(a_flat), half_ptr(b_flat), half_ptr(c_flat), half_ptr(d_flat),
            reinterpret_cast<__half*>(W_half.data_ptr<at::Half>()),
            row_width, col_width, bit_width, y_row, z_col, k8);
        result = torch::mm(X_2d.to(torch::kFloat16), W_half.t());
    }

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
    m.def("bqq_forward", &bqq_forward, "Blackwell BQQ forward");
    m.def("bqq_forward_flat", &bqq_forward_core,
          "Blackwell BQQ forward with flattened weights");
}
