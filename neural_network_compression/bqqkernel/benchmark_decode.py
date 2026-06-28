"""Microbenchmark PackedBinaryQuadratic autoregressive decode latency.

This targets the CUDA fused path used when X is a single decode token:
X.shape == [1, in_features].  It can sweep BQQ_CUDA_COL_SPLITS to measure
the occupancy vs atomicAdd tradeoff in bqq_cuda.cu.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch


def _add_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _make_layer(
    *,
    out_features: int,
    in_features: int,
    bit_width: int,
    y_row: int,
    z_col: int,
    inter_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    from neural_network_compression.bqqkernel.bqq_modules import PackedBinaryQuadratic

    if out_features % y_row != 0:
        raise ValueError("out_features must be divisible by y_row")
    if in_features % z_col != 0:
        raise ValueError("in_features must be divisible by z_col")
    if inter_dim % 8 != 0:
        raise ValueError("inter_dim must be divisible by 8 for this benchmark")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    row_width = out_features // y_row
    col_width = in_features // z_col
    k8 = inter_dim // 8

    shape_y = (bit_width, row_width, col_width, y_row, k8)
    shape_z = (bit_width, row_width, col_width, z_col, k8)
    coeff_shape = (bit_width, row_width, col_width, 1, 1)

    Y_packed = torch.randint(0, 256, shape_y, dtype=torch.uint8, generator=generator)
    Z_packed = torch.randint(0, 256, shape_z, dtype=torch.uint8, generator=generator)

    a = torch.randn(coeff_shape, generator=generator, dtype=torch.float32) / inter_dim**0.5
    b = torch.randn(coeff_shape, generator=generator, dtype=torch.float32) / inter_dim**0.5
    c = torch.randn(coeff_shape, generator=generator, dtype=torch.float32) / inter_dim**0.5
    d = torch.randn((row_width, col_width, 1, 1), generator=generator, dtype=torch.float32) / in_features**0.5

    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    Y_bits = ((Y_packed.unsqueeze(-1) >> shifts) & 1).reshape(
        bit_width, row_width, col_width, y_row, -1)
    Z_bits = ((Z_packed.unsqueeze(-1) >> shifts) & 1).reshape(
        bit_width, row_width, col_width, z_col, -1)
    Y_sum_i16 = Y_bits[..., :inter_dim].sum(dim=-1, keepdim=True).to(torch.int16)
    Z_sum_i16 = Z_bits[..., :inter_dim].sum(dim=-1).unsqueeze(-2).to(torch.int16)

    layer = PackedBinaryQuadratic(
        Y_packed=Y_packed,
        Z_packed=Z_packed,
        a=a,
        b=b,
        c=c,
        d=d,
        Y_sum_i16=Y_sum_i16,
        Z_sum_i16=Z_sum_i16,
        inter_dimension=inter_dim,
        y_row=y_row,
        z_col=z_col,
        bias=None,
    ).to(device=device)

    x = torch.randn((1, in_features), device=device, dtype=dtype)
    return layer, x


def _time_forward(
    fn,
    *,
    warmup: int,
    iters: int,
    inner_iters: int,
) -> tuple[float, float, float]:
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        samples = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            for _ in range(inner_iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / inner_iters)

    mean_ms = statistics.fmean(samples)
    p50_ms = statistics.median(samples)
    p90_ms = sorted(samples)[int(0.9 * (len(samples) - 1))]
    return mean_ms, p50_ms, p90_ms


def _parse_splits(value: str) -> list[int | None]:
    result: list[int | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"auto", "default"}:
            result.append(None)
        else:
            result.append(int(item))
    return result


def _parse_decode_kernels(value: str) -> list[str | None]:
    result: list[str | None] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.lower() in {"default", "current", "fused"}:
            result.append(None)
        elif item in {"original", "two_stage", "two_stage_warp"}:
            result.append(item)
        else:
            raise ValueError(f"unknown decode kernel: {item}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--bit-width", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--y-row", type=int, default=None)
    parser.add_argument("--z-col", type=int, default=None)
    parser.add_argument("--inter-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--inner-iters", type=int, default=1)
    parser.add_argument(
        "--baseline",
        choices=["none", "dense-bf16", "dense-fp16", "all"],
        default="dense-bf16",
        help="Also benchmark dense GEMV baseline with the same [1, in_features] input.",
    )
    parser.add_argument(
        "--col-splits",
        default="auto,1,2,4,8,16,32,64",
        help="Comma-separated BQQ_CUDA_COL_SPLITS values. Use auto for default heuristic.",
    )
    parser.add_argument(
        "--decode-kernels",
        default="default,two_stage,two_stage_warp",
        help="Comma-separated decode kernels: default,original,two_stage,two_stage_warp.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _add_repo_to_path()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    y_row = args.y_row if args.y_row is not None else args.group_size
    z_col = args.z_col if args.z_col is not None else args.group_size
    layer, x = _make_layer(
        out_features=args.out_features,
        in_features=args.in_features,
        bit_width=args.bit_width,
        y_row=y_row,
        z_col=z_col,
        inter_dim=args.inter_dim,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )

    print(
        "decode benchmark: "
        f"out={args.out_features} in={args.in_features} "
        f"bits={args.bit_width} y_row={y_row} z_col={z_col} inter={args.inter_dim} "
        f"dtype={args.dtype}"
    )
    print("method,col_splits,mean_ms,p50_ms,p90_ms,tokens_per_s")

    dense_baselines: list[tuple[str, torch.Tensor]] = []
    if args.baseline in {"dense-bf16", "all"}:
        dense_baselines.append((
            "dense-bf16",
            torch.randn(
                (args.out_features, args.in_features),
                device=device,
                dtype=torch.bfloat16,
            ),
        ))
    if args.baseline in {"dense-fp16", "all"}:
        dense_baselines.append((
            "dense-fp16",
            torch.randn(
                (args.out_features, args.in_features),
                device=device,
                dtype=torch.float16,
            ),
        ))

    for name, weight in dense_baselines:
        dense_x = x.to(weight.dtype)
        mean_ms, p50_ms, p90_ms = _time_forward(
            lambda weight=weight, dense_x=dense_x: torch.nn.functional.linear(dense_x, weight),
            warmup=args.warmup,
            iters=args.iters,
            inner_iters=args.inner_iters,
        )
        toks = 1000.0 / mean_ms
        print(f"{name},n/a,{mean_ms:.4f},{p50_ms:.4f},{p90_ms:.4f},{toks:.2f}")

    old_splits_env = os.environ.get("BQQ_CUDA_COL_SPLITS")
    old_kernel_env = os.environ.get("BQQ_CUDA_DECODE_KERNEL")
    try:
        for kernel in _parse_decode_kernels(args.decode_kernels):
            if kernel is None:
                os.environ.pop("BQQ_CUDA_DECODE_KERNEL", None)
                kernel_label = "default"
            else:
                os.environ["BQQ_CUDA_DECODE_KERNEL"] = kernel
                kernel_label = kernel

            for splits in _parse_splits(args.col_splits):
                if splits is None:
                    os.environ.pop("BQQ_CUDA_COL_SPLITS", None)
                    split_label = "auto"
                else:
                    os.environ["BQQ_CUDA_COL_SPLITS"] = str(splits)
                    split_label = str(splits)

                mean_ms, p50_ms, p90_ms = _time_forward(
                    lambda: layer(x),
                    warmup=args.warmup,
                    iters=args.iters,
                    inner_iters=args.inner_iters,
                )
                toks = 1000.0 / mean_ms
                print(
                    f"bqq-{kernel_label},{split_label},"
                    f"{mean_ms:.4f},{p50_ms:.4f},{p90_ms:.4f},{toks:.2f}"
                )
    finally:
        if old_splits_env is None:
            os.environ.pop("BQQ_CUDA_COL_SPLITS", None)
        else:
            os.environ["BQQ_CUDA_COL_SPLITS"] = old_splits_env
        if old_kernel_env is None:
            os.environ.pop("BQQ_CUDA_DECODE_KERNEL", None)
        else:
            os.environ["BQQ_CUDA_DECODE_KERNEL"] = old_kernel_env


if __name__ == "__main__":
    main()
