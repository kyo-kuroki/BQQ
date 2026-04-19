/*
 * BQQ fused forward CUDA kernel.
 *
 * Computes  out = X @ W.T  where
 *   W = Σ_p (a_p·Y_p@Z_p + b_p·Ysum_p + c_p·Zsum_p) + d
 * without materialising W.
 *
 * Key optimisations over the Triton version:
 *   1. Warp-level shuffle reduction for Z@X (no bit→float multiply)
 *   2. Conditional add for Y@t (branch instead of multiply-by-0-or-1)
 *   3. Z/Y bytes preloaded into registers, reused across 8 bit iterations
 *   4. Template-based tile sizes for compile-time unrolling
 *
 * Thread mapping (one warp = 32 threads per block):
 *   Phase 1 — thread j holds X[j], reduces Z_bool[k,j]*X[j] via __shfl_down
 *   Phase 2 — thread i checks Y_bool[i,k], conditionally adds t_aug
 *
 * Grid: (row_width, batch)    Block: 32 threads (one warp)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>


/* ── warp-level sum ────────────────────────────────────────────── */

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


/* ── main kernel ───────────────────────────────────────────────── */
/*
 * Template parameters:
 *   N_J_TILES — ceil(z_col / 32), number of warp passes for Phase 1
 *   N_I_TILES — ceil(y_row / 32), number of output rows per thread
 *   K8_MAX    — upper bound on k8 (for register array sizing)
 */

template <int N_J_TILES, int N_I_TILES, int K8_MAX>
__global__ void bqq_forward_kernel(
    const uint8_t* __restrict__ Y,      // [B_total, y_row, k8]
    const uint8_t* __restrict__ Z,      // [B_total, z_col, k8]
    const float*   __restrict__ X,      // [batch, col_width, z_col]
    const float*   __restrict__ a_ptr,
    const float*   __restrict__ b_ptr,
    const float*   __restrict__ c_ptr,
    const float*   __restrict__ d_ptr,  // [row_width * col_width]
    float*         __restrict__ out,    // [batch, row_width, y_row]
    int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    const int r    = blockIdx.x;          // output row-block index
    const int n    = blockIdx.y;          // batch index
    const int lane = threadIdx.x;         // 0 .. 31

    /* shared memory for X[n, c, :] — needed when z_col > 32 */
    extern __shared__ float x_smem[];

    /* per-i-tile accumulators */
    float acc[N_I_TILES];
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) acc[it] = 0.0f;

    /* ───── col_width loop ───────────────────────────────────── */
    for (int ci = 0; ci < col_width; ci++) {
        const int rc     = r * col_width + ci;
        const int x_base = n * col_width * z_col + ci * z_col;

        /* load X[n, c, :] to shared memory */
        #pragma unroll
        for (int jt = 0; jt < N_J_TILES; jt++) {
            int j = jt * 32 + lane;
            if (j < z_col) x_smem[j] = X[x_base + j];
        }
        __syncwarp();

        /* xsum = Σ_j X[n,c,j] via warp reduction */
        float xsum = 0.0f;
        #pragma unroll
        for (int jt = 0; jt < N_J_TILES; jt++) {
            int j = jt * 32 + lane;
            float xj = (j < z_col) ? x_smem[j] : 0.0f;
            xsum += warp_reduce_sum(xj);
        }
        xsum = __shfl_sync(0xffffffff, xsum, 0);

        /* ───── bit_width loop ───────────────────────────────── */
        for (int p = 0; p < bit_width; p++) {
            const int   B_idx = p * row_width * col_width + rc;
            const float a_val = a_ptr[B_idx];
            const float b_val = b_ptr[B_idx];
            const float c_val = c_ptr[B_idx];
            float t_sum = 0.0f;

            /* ── byte_k loop (over packed bytes) ─────────────── */
            #pragma unroll
            for (int bk = 0; bk < K8_MAX; bk++) {
                if (bk >= k8) break;

                /* preload Z bytes for all j-tiles */
                uint8_t zb[N_J_TILES];
                #pragma unroll
                for (int jt = 0; jt < N_J_TILES; jt++) {
                    int j = jt * 32 + lane;
                    zb[jt] = (j < z_col)
                        ? Z[(size_t)B_idx * z_col * k8 + j * k8 + bk] : 0;
                }

                /* preload Y bytes for all i-tiles */
                uint8_t yb[N_I_TILES];
                #pragma unroll
                for (int it = 0; it < N_I_TILES; it++) {
                    int i = it * 32 + lane;
                    yb[it] = (i < y_row)
                        ? Y[(size_t)B_idx * y_row * k8 + i * k8 + bk] : 0;
                }

                /* ── 8-bit loop (fully unrolled) ─────────────── */
                #pragma unroll
                for (int bit = 0; bit < 8; bit++) {
                    const int shift = 7 - bit;

                    /*
                     * Phase 1: t_k = Σ_j Z_bool[k,j] · X[j]
                     *
                     * Each thread holds X[j] and Z_bit for its lane.
                     * z_bit ? x_j : 0  →  warp shuffle reduction.
                     * NO bit-to-float conversion or multiply.
                     */
                    float t_k = 0.0f;
                    #pragma unroll
                    for (int jt = 0; jt < N_J_TILES; jt++) {
                        int j = jt * 32 + lane;
                        int z_bit = (zb[jt] >> shift) & 1;
                        float xj  = (j < z_col) ? x_smem[j] : 0.0f;
                        /* conditional select — no multiply */
                        float masked = z_bit ? xj : 0.0f;
                        t_k += warp_reduce_sum(masked);
                    }
                    /* broadcast t_k from lane 0 to all threads */
                    t_k = __shfl_sync(0xffffffff, t_k, 0);

                    /* t_aug = a·t_k + b·xsum (same scalar for all i) */
                    const float t_aug = a_val * t_k + b_val * xsum;

                    /*
                     * Phase 2: acc[i] += Y_bool[i,k] ? t_aug : 0
                     *
                     * Pure conditional add — no multiply, no bit→float.
                     */
                    #pragma unroll
                    for (int it = 0; it < N_I_TILES; it++) {
                        if ((yb[it] >> shift) & 1)
                            acc[it] += t_aug;
                    }

                    t_sum += t_k;
                }   /* bit */
            }       /* bk  */

            /* term 3: c · sum(t) */
            const float c_term = c_val * t_sum;
            #pragma unroll
            for (int it = 0; it < N_I_TILES; it++)
                acc[it] += c_term;

        }           /* p   */

        /* term 4: d · xsum */
        const float d_term = d_ptr[rc] * xsum;
        #pragma unroll
        for (int it = 0; it < N_I_TILES; it++)
            acc[it] += d_term;

    }               /* ci  */

    /* ───── store output ─────────────────────────────────────── */
    #pragma unroll
    for (int it = 0; it < N_I_TILES; it++) {
        int i = it * 32 + lane;
        if (i < y_row)
            out[(size_t)n * row_width * y_row + r * y_row + i] = acc[it];
    }
}


/* ── dispatch ──────────────────────────────────────────────────── */

torch::Tensor bqq_forward_cuda(
    torch::Tensor Y_packed,     // [B_total, y_row, k8] uint8
    torch::Tensor Z_packed,     // [B_total, z_col, k8] uint8
    torch::Tensor X,            // [batch, col_width, z_col] float32
    torch::Tensor a, torch::Tensor b,
    torch::Tensor c, torch::Tensor d,
    int batch, int row_width, int col_width, int bit_width,
    int y_row, int z_col, int k8)
{
    auto out = torch::empty({batch, row_width, y_row},
        torch::dtype(torch::kFloat32).device(X.device()));

    dim3 grid(row_width, batch);
    dim3 block(32);   /* one warp */
    int smem = z_col * sizeof(float);

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

    #define LAUNCH(NJ, NI, K8M) \
        bqq_forward_kernel<NJ, NI, K8M><<<grid, block, smem>>>( \
            yp, zp, xp, ap, bp, cp, dp, op, \
            row_width, col_width, bit_width, y_row, z_col, k8)

    /* dispatch by tile counts (gs=32 → 1,1; gs=64 → 2,2; gs=128 → 4,4) */
    if (nj == 1 && ni == 1) {
        if      (k8 <= 4)  LAUNCH(1, 1, 4);
        else if (k8 <= 8)  LAUNCH(1, 1, 8);
        else               LAUNCH(1, 1, 16);
    } else if (nj <= 2 && ni <= 2) {
        if      (k8 <= 8)  LAUNCH(2, 2, 8);
        else               LAUNCH(2, 2, 16);
    } else {
        LAUNCH(4, 4, 16);
    }

    #undef LAUNCH
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bqq_forward_cuda", &bqq_forward_cuda,
          "BQQ fused forward (warp-level CUDA kernel)");
}
