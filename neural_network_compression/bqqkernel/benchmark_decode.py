"""Microbenchmark PackedBinaryQuadratic autoregressive decode latency.

This targets the CUDA fused path used when X is a single decode token:
X.shape == [1, in_features].  It can sweep BQQ_CUDA_COL_SPLITS to measure
the occupancy vs atomicAdd tradeoff in bqq_cuda.cu.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from pathlib import Path

import torch
import dill

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from transformers import AutoModelForCausalLM


def _add_repo_to_path() -> None:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    for path in (str(project_root), str(package_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _prime_local_import_shims() -> None:
    """Force local compatibility shims to win over broken site-packages ones."""
    package_root = Path(__file__).resolve().parents[1]

    def _load_package(name: str) -> None:
        pkg_init = package_root / name / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            name,
            pkg_init,
            submodule_search_locations=[str(pkg_init.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local shim package: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    _load_package("causal_conv1d")
    _load_package("fla")


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
        elif item in {"bitblas_byte", "bitblas_byte2", "bitblas_byte4"}:
            result.append(item)
        else:
            raise ValueError(f"unknown decode kernel: {item}")
    return result


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_bqq_model(model_path: str, dtype: torch.dtype | None) -> torch.nn.Module:
    _add_repo_to_path()
    _prime_local_import_shims()
    model = torch.load(model_path, map_location="cpu", weights_only=False, pickle_module=dill)
    if not hasattr(model, "eval"):
        raise TypeError(f"{model_path} did not deserialize to a torch.nn.Module")
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        setattr(model.config, "use_cache", True)
    if dtype is not None:
        model = model.to(dtype=dtype)
    _ensure_qwen35_block_type(model)
    return model


def _load_fp16_model(model_name: str) -> torch.nn.Module:
    _add_repo_to_path()
    _prime_local_import_shims()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config"):
        setattr(model.config, "use_cache", True)
    _ensure_qwen35_block_type(model)
    return model


def _ensure_qwen35_block_type(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "layer_type") and not hasattr(module, "block_type"):
            module.block_type = module.layer_type


def _apply_bqq_kernel_env(kernel: str | None, col_splits: str | None) -> tuple[str | None, str | None]:
    old_kernel = os.environ.get("BQQ_CUDA_DECODE_KERNEL")
    old_splits = os.environ.get("BQQ_CUDA_COL_SPLITS")
    if kernel is None:
        os.environ.pop("BQQ_CUDA_DECODE_KERNEL", None)
    else:
        os.environ["BQQ_CUDA_DECODE_KERNEL"] = kernel
    if col_splits is None:
        os.environ.pop("BQQ_CUDA_COL_SPLITS", None)
    else:
        os.environ["BQQ_CUDA_COL_SPLITS"] = col_splits
    return old_kernel, old_splits


def _restore_bqq_kernel_env(old_kernel: str | None, old_splits: str | None) -> None:
    if old_kernel is None:
        os.environ.pop("BQQ_CUDA_DECODE_KERNEL", None)
    else:
        os.environ["BQQ_CUDA_DECODE_KERNEL"] = old_kernel
    if old_splits is None:
        os.environ.pop("BQQ_CUDA_COL_SPLITS", None)
    else:
        os.environ["BQQ_CUDA_COL_SPLITS"] = old_splits


@torch.inference_mode()
def _run_decode_benchmark(
    *,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    warmup: int,
    iters: int,
    inner_iters: int,
    use_cuda_graph: bool = False,
) -> tuple[float, float, float]:
    model.eval()
    model.to(input_ids.device)

    pref = model(input_ids=input_ids, use_cache=True)
    past_key_values = pref.past_key_values
    step_ids = input_ids[:, -1:]

    def step():
        model(input_ids=step_ids, past_key_values=past_key_values, use_cache=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    if use_cuda_graph:
        # Capture one decode step and replay it.  The DynamicCache is
        # frozen at capture time, so this measures fixed-context decode
        # with zero kernel-launch overhead — what a serving stack with
        # CUDA graphs / static cache would see.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        run_step = graph.replay
        torch.cuda.synchronize()
    else:
        run_step = step

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        for _ in range(inner_iters):
            run_step()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / inner_iters)

    mean_ms = statistics.fmean(samples)
    p50_ms = statistics.median(samples)
    p90_ms = sorted(samples)[int(0.9 * (len(samples) - 1))]
    return mean_ms, p50_ms, p90_ms


def _run_model_benchmark(args) -> None:
    if not args.model_path:
        raise ValueError("--model-path is required in model benchmark mode")
    if not args.model_name:
        raise ValueError("--model-name is required in model benchmark mode")

    device = torch.device("cuda")
    bqq_dtype = None if args.bqq_dtype == "keep" else getattr(torch, args.bqq_dtype)
    bqq_model = _load_bqq_model(args.model_path, bqq_dtype)
    fp16_model = _load_fp16_model(args.model_name)
    _ensure_qwen35_block_type(bqq_model)
    _ensure_qwen35_block_type(fp16_model)

    vocab_size = getattr(getattr(fp16_model, "config", None), "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(getattr(bqq_model, "config", None), "vocab_size", None)
    if vocab_size is None:
        raise RuntimeError("Could not determine vocab_size from either model")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(0, int(vocab_size), (1, args.seq_len), dtype=torch.long, generator=generator)
    input_ids = input_ids.to(device=device)

    old_kernel = old_splits = None
    try:
        old_kernel, old_splits = _apply_bqq_kernel_env(args.bqq_decode_kernel, args.bqq_col_splits)
        for name, model in (("bqq", bqq_model), ("fp16", fp16_model)):
            mean_ms, p50_ms, p90_ms = _run_decode_benchmark(
                model=model,
                input_ids=input_ids,
                warmup=args.warmup,
                iters=args.iters,
                inner_iters=args.inner_iters,
                use_cuda_graph=args.use_cuda_graph,
            )
            toks = 1000.0 / mean_ms
            print(f"{name},{mean_ms:.4f},{p50_ms:.4f},{p90_ms:.4f},{toks:.2f}")
    finally:
        _restore_bqq_kernel_env(old_kernel, old_splits)


def _run_packed_core_benchmark(args) -> None:
    """Benchmark packed binary matmul directly.

    This bypasses the full decode forward path and measures the core packed
    BQQ binary matmul used by PackedBinaryQuadratic._compute_W_core.
    """
    from neural_network_compression.bqqkernel.bqq_modules import PackedBinaryQuadratic

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    default_group = args.group_size if args.group_size is not None else 2 * args.inter_dim
    y_row = args.y_row if args.y_row is not None else default_group
    z_col = args.z_col if args.z_col is not None else default_group
    bit_widths = _parse_ints(args.bit_widths) if args.bit_widths else [args.bit_width]

    print("method,bits,col_splits,mean_ms,p50_ms,p90_ms,tokens_per_s")

    old_flag = PackedBinaryQuadratic.use_packed_kernel
    try:
        for bit_width in bit_widths:
            layer, _x = _make_layer(
                out_features=args.out_features,
                in_features=args.in_features,
                bit_width=bit_width,
                y_row=y_row,
                z_col=z_col,
                inter_dim=args.inter_dim,
                device=device,
                dtype=dtype,
                seed=args.seed + bit_width,
            )

            PackedBinaryQuadratic.use_packed_kernel = False
            mean_ms, p50_ms, p90_ms = _time_forward(
                lambda: layer._compute_W_core(dtype),
                warmup=args.warmup,
                iters=args.iters,
                inner_iters=args.inner_iters,
            )
            toks = 1000.0 / mean_ms
            print(f"packed-unpack,{bit_width},n/a,{mean_ms:.4f},{p50_ms:.4f},{p90_ms:.4f},{toks:.2f}")

            PackedBinaryQuadratic.use_packed_kernel = True
            mean_ms, p50_ms, p90_ms = _time_forward(
                lambda: layer._compute_W_core(dtype),
                warmup=args.warmup,
                iters=args.iters,
                inner_iters=args.inner_iters,
            )
            toks = 1000.0 / mean_ms
            print(f"packed-kernel,{bit_width},n/a,{mean_ms:.4f},{p50_ms:.4f},{p90_ms:.4f},{toks:.2f}")
    finally:
        PackedBinaryQuadratic.use_packed_kernel = old_flag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-mode",
        choices=["microbench", "packed-core", "model"],
        default="microbench",
        help="microbench keeps the existing decode benchmark; packed-core measures packed_binary_matmul directly; model compares BQQ .pth vs fp16 HF model.",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--bqq-dtype",
        choices=["keep", "float16", "bfloat16", "float32"],
        default="keep",
        help="Cast the loaded BQQ model to this dtype. float16 is required for "
        "the fused CUDA decode kernels (bitblas_byte*); keep preserves the saved dtype.",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Capture one decode step as a CUDA graph and time graph replay. "
        "Removes kernel-launch overhead; context is frozen at capture time.",
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--bit-width", type=int, default=1)
    parser.add_argument("--bit-widths", default="1,2,3")
    parser.add_argument("--group-size", type=int, default=None)
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
        default="default,bitblas_byte4,bitblas_byte2,bitblas_byte",
        help="Comma-separated decode kernels: default,bitblas_byte4,bitblas_byte2,bitblas_byte.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bqq-decode-kernel",
        default=None,
        help="Override BQQ_CUDA_DECODE_KERNEL during model benchmark. Useful to force a stable decode kernel such as two_stage_warp.",
    )
    parser.add_argument(
        "--bqq-col-splits",
        default=None,
        help="Override BQQ_CUDA_COL_SPLITS during model benchmark.",
    )
    args = parser.parse_args()

    _add_repo_to_path()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    if args.benchmark_mode == "model":
        print("method,mean_ms,p50_ms,p90_ms,tokens_per_s")
        _run_model_benchmark(args)
        return
    if args.benchmark_mode == "packed-core":
        _run_packed_core_benchmark(args)
        return

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    default_group = args.group_size if args.group_size is not None else 2 * args.inter_dim
    y_row = args.y_row if args.y_row is not None else default_group
    z_col = args.z_col if args.z_col is not None else default_group
    bit_widths = _parse_ints(args.bit_widths) if args.bit_widths else [args.bit_width]

    print("method,bits,col_splits,mean_ms,p50_ms,p90_ms,tokens_per_s")

    old_splits_env = os.environ.get("BQQ_CUDA_COL_SPLITS")
    old_kernel_env = os.environ.get("BQQ_CUDA_DECODE_KERNEL")
    try:
        for bit_width in bit_widths:
            layer, x = _make_layer(
                out_features=args.out_features,
                in_features=args.in_features,
                bit_width=bit_width,
                y_row=y_row,
                z_col=z_col,
                inter_dim=args.inter_dim,
                device=device,
                dtype=dtype,
                seed=args.seed + bit_width,
            )

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
                print(f"{name},{bit_width},n/a,{mean_ms:.4f},{p50_ms:.4f},{p90_ms:.4f},{toks:.2f}")

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
                        f"bqq-{kernel_label},{bit_width},{split_label},"
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
