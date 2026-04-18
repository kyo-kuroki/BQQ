"""
Benchmark BinaryQuadratic vs PackedBinaryQuadratic forward pass speed.

Tests on both CPU and CUDA (if available), with realistic LM inference shapes.
"""
import sys, os, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import bqq_modules  # noqa: F401
from bqq_modules import BinaryQuadratic, PackedBinaryQuadratic

torch.manual_seed(0)


def make_bq(bit_width, row_width, col_width, y_row, inter_dim, z_col, device):
    Y = (torch.rand(bit_width, row_width, col_width, y_row, inter_dim) > 0.5)
    Z = (torch.rand(bit_width, row_width, col_width, inter_dim, z_col) > 0.5)
    A = torch.randn(bit_width, row_width, col_width, 4)
    bq = BinaryQuadratic(Y.float(), Z.float(), A).to(device)
    pbq = PackedBinaryQuadratic.from_unpacked(bq.cpu()).to(device)
    return bq, pbq


def bench(module, X, n_warmup=10, n_iter=100, sync_cuda=False):
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = module(X)
        if sync_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = module(X)
        if sync_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    return (t1 - t0) / n_iter * 1e3  # ms per call


def run_benchmark(device, dtype=torch.bfloat16):
    sync = (device != "cpu")
    print(f"\n{'='*60}")
    print(f"Device: {device}  dtype: {dtype}")
    print(f"{'='*60}")
    print(f"{'Shape':45s} {'BQ(ms)':>8} {'Packed(ms)':>10} {'Speedup':>8}")
    print(f"{'-'*45} {'-'*8} {'-'*10} {'-'*8}")

    # (description, bit_width, row_width, col_width, y_row, inter_dim, z_col, batch_seq)
    # Shapes match Qwen3.5 layers with gs=64:
    #   2B  attn q_proj  2048->2048 => rows=32, cols=32
    #   2B  mlp gate    2048->11008 => rows=32, cols=172
    #   9B  attn q_proj  4096->4096 => rows=64, cols=64
    #   9B  mlp gate    4096->22016 => rows=64, cols=344
    shapes = [
        ("2B q_proj  (r=32,c=32,bw=2,seq=1)",    2, 32, 32,  64, 32, 64, (1, 1,    2048)),
        ("2B q_proj  (r=32,c=32,bw=2,seq=512)",  2, 32, 32,  64, 32, 64, (1, 512,  2048)),
        ("2B mlp_gate(r=32,c=172,bw=2,seq=1)",   2, 32, 172, 64, 32, 64, (1, 1,    2048)),
        ("9B q_proj  (r=64,c=64,bw=2,seq=1)",    2, 64, 64,  64, 32, 64, (1, 1,    4096)),
        ("9B q_proj  (r=64,c=64,bw=2,seq=512)",  2, 64, 64,  64, 32, 64, (1, 512,  4096)),
        ("9B mlp_gate(r=64,c=344,bw=2,seq=1)",   2, 64, 344, 64, 32, 64, (1, 1,    4096)),
    ]

    for desc, bw, rw, cw, yr, id_, zc, xshape in shapes:
        bq, pbq = make_bq(bw, rw, cw, yr, id_, zc, device)
        in_feat = cw * zc
        X = torch.randn(*xshape, dtype=dtype, device=device)
        # adjust last dim to match
        X = torch.randn(*xshape[:-1], in_feat, dtype=dtype, device=device)

        t_bq = bench(bq, X, sync_cuda=sync)
        t_pb = bench(pbq, X, sync_cuda=sync)
        speedup = t_bq / t_pb

        print(f"{desc:45s} {t_bq:8.3f} {t_pb:10.3f} {speedup:7.2f}x")


if __name__ == "__main__":
    run_benchmark("cpu")
    if torch.cuda.is_available():
        run_benchmark("cuda:0")
    else:
        print("\n(CUDA not available)")
