from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
BQQ_ROOT = CV_DIR.parent.parent
UTILS_DIR = CV_DIR / "utils"

for path in (BQQ_ROOT, UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from quantizer import BinaryQuadraticQuantization2 as BQQ2
from build_dataset import get_imagenet
from build_model import get_model
from make_bqq_model_from_compressed_data import save_bqq_model
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
    quantize_parser.add_argument("--overwrite", action="store_true")

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
        save_name=str(save_prefix),
        zeta=zeta,
        eta=eta,
        Tinit=Tinit,
        Tfin=Tfin,
        Nstep=Nstep,
        seed=seed,
        workers_per_gpu=num_workers_per_gpu,
        main_gpu_id=main_gpu_id,
    )


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


def quantize_target(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cached weight-aware BQQ quantization.")

    cache_dir = resolve_cache_dir(cache_dir=args.cache_dir, model_name=args.model_name)
    metadata = read_metadata(cache_dir)
    targets = set(read_targets(cache_dir))
    if args.target_name not in targets:
        raise ValueError(f"Target is not present in cache: {args.target_name}")

    model_name = args.model_name or metadata["model_name"]
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(model_name, args.Nstep, args.group_size, args.rank_scale)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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
    if args.command == "quantize-target":
        quantize_target(args)
        return
    if args.command == "finalize":
        finalize(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
