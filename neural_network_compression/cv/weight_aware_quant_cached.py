from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent

    path_str = str(path)


from quantizer import BinaryQuadraticQuantization as BQQ2
from build_dataset import get_imagenet
from build_model import get_model
from build_bqq_model import save_bqq_model
from utils import test_model_accuracy


DEFAULT_CACHE_ROOT = SCRIPT_DIR / "cache"
WEIGHTS_DIR_NAME = "weights"
TARGETS_FILE_NAME = "targets.txt"
METADATA_FILE_NAME = "metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache-first weight-aware BQQ utilities for CV models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-cache",
        help="Load a model once and cache quantization target weights under ./cache",
    )
    prepare_parser.add_argument("--model_name", type=str, required=True)
    prepare_parser.add_argument("--cache_dir", type=Path, default=None)
    prepare_parser.add_argument("--refresh_cache", action="store_true")

    list_parser = subparsers.add_parser(
        "list-targets",
        help="Print cached target weight names, one per line",
    )
    list_parser.add_argument("--model_name", type=str, default=None)
    list_parser.add_argument("--cache_dir", type=Path, default=None)

    list_patches_parser = subparsers.add_parser(
        "list-patches",
        help="Print cached patch jobs as tab-separated target_name, patch_index, patch_row, patch_col",
    )
    list_patches_parser.add_argument("--model_name", type=str, default=None)
    list_patches_parser.add_argument("--cache_dir", type=Path, default=None)
    list_patches_parser.add_argument("--group_size", type=int, default=32)

    quantize_parser = subparsers.add_parser(
        "quantize-target",
        help="Quantize one cached target weight",
    )
    quantize_parser.add_argument("--target_name", type=str, required=True)
    quantize_parser.add_argument("--model_name", type=str, default=None)
    quantize_parser.add_argument("--bit_width", type=int, default=2)
    quantize_parser.add_argument("--Nstep", type=int, default=5000)
    quantize_parser.add_argument("--main_gpu_id", type=int, default=0)
    quantize_parser.add_argument("--rank_scale", type=float, default=1.0)
    quantize_parser.add_argument("--seed", type=int, default=0)
    quantize_parser.add_argument("--group_size", type=int, default=32)
    quantize_parser.add_argument("--num_workers_per_gpu", type=int, default=16)
    quantize_parser.add_argument("--zeta", type=float, default=4.0)
    quantize_parser.add_argument("--eta", type=float, default=0.06)
    quantize_parser.add_argument("--Tinit", type=float, default=0.2)
    quantize_parser.add_argument("--Tfin", type=float, default=0.005)
    quantize_parser.add_argument("--cache_dir", type=Path, default=None)
    quantize_parser.add_argument("--save_dir", type=Path, default=None)
    quantize_parser.add_argument(
        "--patch_index",
        type=int,
        default=None,
        help="0-based patch index within the cached target; when set, only that patch is quantized",
    )
    quantize_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if they already exist",
    )

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="Rebuild a BQQ model from compressed patches and optionally evaluate it",
    )
    finalize_parser.add_argument("--model_name", type=str, default=None)
    finalize_parser.add_argument("--bit_width", type=int, default=2)
    finalize_parser.add_argument("--Nstep", type=int, default=5000)
    finalize_parser.add_argument("--group_size", type=int, default=32)
    finalize_parser.add_argument("--rank_scale", type=float, default=1.0)
    finalize_parser.add_argument("--main_gpu_id", type=int, default=0)
    finalize_parser.add_argument("--cache_dir", type=Path, default=None)
    finalize_parser.add_argument("--save_dir", type=Path, default=None)
    finalize_parser.add_argument("--data_path", type=str, default=None)
    finalize_parser.add_argument("--evaluate", action="store_true")
    finalize_parser.add_argument("--results_dir", type=Path, default=None)
    finalize_parser.add_argument("--quantization_minutes", type=float, default=None)

    return parser.parse_args()


def model_cache_name(model_name: str) -> str:
    return model_name.rstrip("/").replace("/", "_")


def default_cache_dir(model_name: str) -> Path:
    return DEFAULT_CACHE_ROOT / model_cache_name(model_name)


def default_compressed_data_dir(model_name: str, Nstep: int, group_size: int, rank_scale: float) -> Path:
    return SCRIPT_DIR / "bqq_compressed_data" / f"{model_name}-{Nstep}step-{group_size}gs-{rank_scale}rankscale"


def default_results_dir() -> Path:
    return SCRIPT_DIR / "results"


def resolve_cache_dir(*, cache_dir: Path | None, model_name: str | None) -> Path:
    if cache_dir is not None:
        return cache_dir
    if model_name is None:
        raise ValueError("--cache_dir or --model_name must be specified")
    return default_cache_dir(model_name)


def weights_dir(cache_dir: Path) -> Path:
    return cache_dir / WEIGHTS_DIR_NAME


def targets_file(cache_dir: Path) -> Path:
    return cache_dir / TARGETS_FILE_NAME


def metadata_file(cache_dir: Path) -> Path:
    return cache_dir / METADATA_FILE_NAME


def cached_weight_path(cache_dir: Path, target_name: str) -> Path:
    return weights_dir(cache_dir) / f"{target_name}.pt"


def patch_output_path(save_dir: Path, target_name: str, patch_row: int, patch_col: int) -> Path:
    return save_dir / f"{target_name}_row{patch_row}_col{patch_col}.pth"


def read_targets(cache_dir: Path) -> list[str]:
    path = targets_file(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Cached targets file does not exist: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_metadata(cache_dir: Path) -> dict:
    path = metadata_file(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Cache metadata does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def is_quantization_target(name: str) -> bool:
    return not any(token in name for token in ("norm", "bias", "token", "pos", "emb"))


def cache_is_complete(cache_dir: Path) -> bool:
    if not metadata_file(cache_dir).exists() or not targets_file(cache_dir).exists():
        return False

    try:
        targets = read_targets(cache_dir)
    except FileNotFoundError:
        return False

    if not targets:
        return False

    return all(cached_weight_path(cache_dir, name).exists() for name in targets)


def get_max_divisor(num: int, max_value: int) -> int:
    if max_value <= 0:
        raise ValueError("group_size must be a positive integer")

    limit = max(int(math.sqrt(num)), max_value)
    for candidate in range(limit, 0, -1):
        if num % candidate == 0 and candidate <= max_value:
            return candidate
    return 1


def get_target_metadata(metadata: dict, target_name: str) -> dict:
    for target in metadata.get("targets", []):
        if target.get("name") == target_name:
            return target
    raise ValueError(f"Cache metadata does not contain target: {target_name}")


def get_patch_layout(shape: list[int] | tuple[int, int], group_size: int) -> dict[str, int]:
    if len(shape) != 2:
        raise ValueError(f"Only 2D tensors are supported for patch layout, got shape={tuple(shape)}")

    height, width = int(shape[0]), int(shape[1])
    patch_height = get_max_divisor(height, group_size)
    patch_width = get_max_divisor(width, group_size)
    return {
        "height": height,
        "width": width,
        "patch_height": patch_height,
        "patch_width": patch_width,
        "num_patch_rows": height // patch_height,
        "num_patch_cols": width // patch_width,
    }


def get_patch_spec(
    *,
    shape: list[int] | tuple[int, int],
    group_size: int,
    patch_index: int,
) -> dict[str, int]:
    layout = get_patch_layout(shape, group_size)
    total_patches = layout["num_patch_rows"] * layout["num_patch_cols"]
    if patch_index < 0 or patch_index >= total_patches:
        raise ValueError(f"patch_index must be in [0, {total_patches - 1}], got {patch_index}")

    patch_row, patch_col = divmod(patch_index, layout["num_patch_cols"])
    row_start = patch_row * layout["patch_height"]
    col_start = patch_col * layout["patch_width"]

    return {
        **layout,
        "patch_index": patch_index,
        "patch_row": patch_row,
        "patch_col": patch_col,
        "row_start": row_start,
        "row_end": row_start + layout["patch_height"],
        "col_start": col_start,
        "col_end": col_start + layout["patch_width"],
    }


def iter_patch_specs(
    *,
    shape: list[int] | tuple[int, int],
    group_size: int,
):
    layout = get_patch_layout(shape, group_size)
    total_patches = layout["num_patch_rows"] * layout["num_patch_cols"]
    for patch_index in range(total_patches):
        yield get_patch_spec(shape=shape, group_size=group_size, patch_index=patch_index)


def extract_patch(weight: torch.Tensor, patch_spec: dict[str, int]) -> torch.Tensor:
    return weight[
        patch_spec["row_start"]:patch_spec["row_end"],
        patch_spec["col_start"]:patch_spec["col_end"],
    ].detach().clone()


def quantize_weight(
    weight: torch.Tensor,
    save_prefix: Path,
    *,
    bit_width: int,
    rank_scale: float,
    group_size: int,
    Nstep: int,
    seed: int,
    num_workers_per_gpu: int,
    main_gpu_id: int,
    zeta: float,
    eta: float,
    Tinit: float,
    Tfin: float,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("The input tensor must be 2-dimensional")

    return BQQ2(weight, rank_scale=rank_scale).bqq_large_matrix_multi_worker(
        max_patch_size=group_size,
        bit_width=bit_width,
        consolidated_path=str(save_prefix),
        zeta=zeta,
        eta=eta,
        Tinit=Tinit,
        Tfin=Tfin,
        Nstep=Nstep,
        seed=seed,
        workers_per_gpu=num_workers_per_gpu,
        main_gpu_id=main_gpu_id,
    )


def quantize_patch(
    patch: torch.Tensor,
    save_path: Path,
    *,
    patch_row: int,
    patch_col: int,
    bit_width: int,
    rank_scale: float,
    Nstep: int,
    seed: int,
    main_gpu_id: int,
    zeta: float,
    eta: float,
    Tinit: float,
    Tfin: float,
) -> torch.Tensor:
    if patch.ndim != 2:
        raise ValueError("The input tensor must be 2-dimensional")

    torch.manual_seed(seed)
    torch.cuda.set_device(main_gpu_id)
    device = torch.device(f"cuda:{main_gpu_id}")
    original_x = patch.to(device)
    update_x = original_x.detach().clone()
    decomposed_patches: list[dict] = []

    for bit_idx in range(bit_width):
        decomp_instance = BQQ2(x=update_x.clone(), rank_scale=rank_scale)
        y, z, a = decomp_instance.run_bqq_compile(
            zeta=zeta,
            eta=eta,
            Tinit=Tinit,
            Tfin=Tfin,
            Nstep=Nstep,
            device_id=main_gpu_id,
            seed=seed,
            output_type="torch",
        )
        reconst = a[0] * y @ z
        reconst += a[1] * y.sum(axis=1, keepdim=True)
        reconst += a[2] * z.sum(axis=0, keepdim=True)
        reconst += a[3]
        update_x -= reconst

        decomposed_patches.append(
            {
                "patch_row": patch_row,
                "patch_col": patch_col,
                "coeff": a.cpu(),
                "mat1": y.cpu(),
                "mat2": z.cpu(),
                "bit_idx": bit_idx,
            }
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decomposed_patches, save_path)
    return (original_x - update_x).detach().cpu()


def prepare_cache(args) -> Path:
    cache_dir = resolve_cache_dir(cache_dir=args.cache_dir, model_name=args.model_name)

    if cache_is_complete(cache_dir) and not args.refresh_cache:
        targets = read_targets(cache_dir)
        print(f"Using existing cache: {cache_dir}")
        print(f"Cached targets: {len(targets)}")
        return cache_dir

    if args.refresh_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    target_weights_dir = weights_dir(cache_dir)
    target_weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model for caching: {args.model_name}")
    model = get_model(args.model_name).cpu()

    cached_targets: list[dict] = []
    with torch.inference_mode():
        for name, param in model.named_parameters():
            if not is_quantization_target(name):
                continue
            if param.ndim != 2:
                print(f"Skipping non-2D target {name}: {tuple(param.shape)}")
                continue

            save_path = cached_weight_path(cache_dir, name)
            tensor = param.detach().cpu()
            torch.save(tensor, save_path)
            cached_targets.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "path": str(save_path.relative_to(cache_dir)),
                }
            )
            print(f"Cached {name}: {tuple(tensor.shape)}")

    if not cached_targets:
        raise RuntimeError("No quantization targets matched the current filter settings.")

    targets_file(cache_dir).write_text(
        "\n".join(target["name"] for target in cached_targets) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "target_count": len(cached_targets),
        "targets": cached_targets,
    }
    metadata_file(cache_dir).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Prepared cache: {cache_dir}")
    print(f"Cached targets: {len(cached_targets)}")
    return cache_dir


def list_targets(args) -> None:
    cache_dir = resolve_cache_dir(cache_dir=args.cache_dir, model_name=args.model_name)
    for name in read_targets(cache_dir):
        print(name)


def list_patches(args) -> None:
    cache_dir = resolve_cache_dir(cache_dir=args.cache_dir, model_name=args.model_name)
    metadata = read_metadata(cache_dir)
    for name in read_targets(cache_dir):
        target = get_target_metadata(metadata, name)
        for patch_spec in iter_patch_specs(shape=target["shape"], group_size=args.group_size):
            print(
                f"{name}\t{patch_spec['patch_index']}\t"
                f"{patch_spec['patch_row']}\t{patch_spec['patch_col']}"
            )


def quantize_target(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cached weight-aware BQQ quantization.")

    cache_dir = resolve_cache_dir(cache_dir=args.cache_dir, model_name=args.model_name)
    metadata = read_metadata(cache_dir)
    targets = set(read_targets(cache_dir))
    if args.target_name not in targets:
        raise ValueError(f"Target is not present in cache: {args.target_name}")
    target_metadata = get_target_metadata(metadata, args.target_name)

    model_name = args.model_name or metadata["model_name"]
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(model_name, args.Nstep, args.group_size, args.rank_scale)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_index is not None:
        patch_spec = get_patch_spec(
            shape=target_metadata["shape"],
            group_size=args.group_size,
            patch_index=args.patch_index,
        )
        patch_checkpoint = patch_output_path(
            save_dir,
            args.target_name,
            patch_spec["patch_row"],
            patch_spec["patch_col"],
        )
        if patch_checkpoint.exists() and not args.overwrite:
            print(
                f"Skipping {args.target_name} patch {args.patch_index}: "
                f"already quantized at {patch_checkpoint}"
            )
            return patch_checkpoint

        weight_path = cached_weight_path(cache_dir, args.target_name)
        if not weight_path.exists():
            raise FileNotFoundError(f"Cached tensor does not exist: {weight_path}")

        print(f"Loading cached weight: {weight_path}")
        weight = torch.load(weight_path, map_location="cpu")
        patch = extract_patch(weight, patch_spec)
        print(
            f"Quantizing {args.target_name} patch {args.patch_index} "
            f"(row={patch_spec['patch_row']}, col={patch_spec['patch_col']}): {tuple(patch.shape)}"
        )
        quantize_patch(
            patch,
            patch_checkpoint,
            patch_row=patch_spec["patch_row"],
            patch_col=patch_spec["patch_col"],
            bit_width=args.bit_width,
            rank_scale=args.rank_scale,
            Nstep=args.Nstep,
            seed=args.seed,
            main_gpu_id=args.main_gpu_id,
            zeta=args.zeta,
            eta=args.eta,
            Tinit=args.Tinit,
            Tfin=args.Tfin,
        )
        print(f"Saved compressed patch: {patch_checkpoint}")
        return patch_checkpoint

    tensor_checkpoint = save_dir / f"{args.target_name}.pt"
    if tensor_checkpoint.exists() and not args.overwrite:
        print(f"Skipping {args.target_name}: already quantized at {tensor_checkpoint}")
        return tensor_checkpoint

    weight_path = cached_weight_path(cache_dir, args.target_name)
    if not weight_path.exists():
        raise FileNotFoundError(f"Cached tensor does not exist: {weight_path}")

    print(f"Loading cached weight: {weight_path}")
    weight = torch.load(weight_path, map_location="cpu")

    print(f"Quantizing {args.target_name}: {tuple(weight.shape)}")
    transformed = quantize_weight(
        weight,
        save_dir / args.target_name,
        bit_width=args.bit_width,
        rank_scale=args.rank_scale,
        group_size=args.group_size,
        Nstep=args.Nstep,
        seed=args.seed,
        num_workers_per_gpu=args.num_workers_per_gpu,
        main_gpu_id=args.main_gpu_id,
        zeta=args.zeta,
        eta=args.eta,
        Tinit=args.Tinit,
        Tfin=args.Tfin,
    )

    torch.save(transformed.cpu(), tensor_checkpoint)
    print(f"Saved reconstructed tensor: {tensor_checkpoint}")
    return tensor_checkpoint


def finalize(args) -> Path:
    cache_dir = resolve_cache_dir(cache_dir=args.cache_dir, model_name=args.model_name)
    metadata = read_metadata(cache_dir)
    model_name = args.model_name or metadata["model_name"]
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(model_name, args.Nstep, args.group_size, args.rank_scale)
    save_dir = Path(save_dir)

    device = f"cuda:{args.main_gpu_id}" if torch.cuda.is_available() else "cpu"
    save_bqq_model(
        model_name,
        str(save_dir),
        args.bit_width,
        args.group_size,
        args.Nstep,
        device=device,
    )

    output_model_path = SCRIPT_DIR / "quantized_bqq_model" / f"{model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step-bqq.pth"
    print(f"Saved quantized model to {output_model_path}")

    result = {
        "model": model_name,
        "bit_width": args.bit_width,
        "Nstep": args.Nstep,
        "group_size": args.group_size,
        "rank_scale": args.rank_scale,
    }
    if args.quantization_minutes is not None:
        result["q_time"] = args.quantization_minutes

    if args.evaluate:
        model = torch.load(output_model_path, map_location=device, weights_only=False)
        _, testloader = get_imagenet(model_name, data_path=args.data_path)
        imagenet_accuracy = test_model_accuracy(model, testloader, device)
        result["accuracy"] = imagenet_accuracy
        print(f"Transformed {model_name} ImageNet Accuracy: {imagenet_accuracy:.2f}%")

    if args.evaluate or args.quantization_minutes is not None:
        results_dir = Path(args.results_dir) if args.results_dir is not None else default_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale-{args.bit_width}bit.csv"
        pd.DataFrame([result]).to_csv(result_path, index=False)
        print(f"Saved results to {result_path}")

    return output_model_path


def main():
    args = parse_args()

    if args.command == "prepare-cache":
        prepare_cache(args)
        return
    if args.command == "list-targets":
        list_targets(args)
        return
    if args.command == "list-patches":
        list_patches(args)
        return
    if args.command == "quantize-target":
        quantize_target(args)
        return
    if args.command == "finalize":
        finalize(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
