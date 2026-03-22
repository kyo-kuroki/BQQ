from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

import torch
from transformers import AutoModelForCausalLM

try:
    from .compressed_data import (
        default_compressed_data_dir,
        ensure_bqq_root_on_path,
        model_basename,
    )
except ImportError:
    from compressed_data import (
        default_compressed_data_dir,
        ensure_bqq_root_on_path,
        model_basename,
    )


ensure_bqq_root_on_path()

from quantizer import BinaryQuadraticQuantization2 as BQQ2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = SCRIPT_DIR / 'cache'
WEIGHTS_DIR_NAME = 'weights'
TARGETS_FILE_NAME = 'targets.txt'
METADATA_FILE_NAME = 'metadata.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cache-first weight-aware BQQ utilities for language models'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser(
        'prepare-cache',
        help='Load a model once and cache quantization target weights under ./cache',
    )
    prepare_parser.add_argument('--model_name', type=str, required=True)
    prepare_parser.add_argument('--layer_threshold', type=int, default=4)
    prepare_parser.add_argument('--cache_dir', type=Path, default=None)
    prepare_parser.add_argument(
        '--refresh_cache',
        action='store_true',
        help='Rebuild the cache even when a complete cache already exists',
    )

    list_parser = subparsers.add_parser(
        'list-targets',
        help='Print cached target weight names, one per line',
    )
    list_parser.add_argument('--model_name', type=str, default=None)
    list_parser.add_argument('--layer_threshold', type=int, default=4)
    list_parser.add_argument('--cache_dir', type=Path, default=None)

    quantize_parser = subparsers.add_parser(
        'quantize-target',
        help='Quantize one cached target weight',
    )
    quantize_parser.add_argument('--target_name', type=str, required=True)
    quantize_parser.add_argument('--model_name', type=str, default=None)
    quantize_parser.add_argument('--bit_width', type=int, default=4)
    quantize_parser.add_argument('--group_size', type=int, default=128)
    quantize_parser.add_argument('--num_steps', type=int, default=50000)
    quantize_parser.add_argument('--rank_scale', type=float, default=1.0)
    quantize_parser.add_argument('--workers_per_gpu', type=int, default=16)
    quantize_parser.add_argument('--main_gpu_id', type=int, default=0)
    quantize_parser.add_argument('--seed', type=int, default=0)
    quantize_parser.add_argument('--layer_threshold', type=int, default=4)
    quantize_parser.add_argument('--cache_dir', type=Path, default=None)
    quantize_parser.add_argument('--save_dir', type=Path, default=None)
    quantize_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the reconstructed tensor checkpoint if it already exists',
    )

    return parser.parse_args()


def is_quantization_target(name: str) -> bool:
    return name.endswith('weight') and all(token not in name for token in ('norm', 'bias', 'emb'))


def passes_layer_threshold(name: str, layer_threshold: int) -> bool:
    layer_ids = [int(num) for num in re.findall(r'\d+', name)]
    return any(layer_id >= layer_threshold for layer_id in layer_ids)


def default_cache_dir(model_name: str, layer_threshold: int) -> Path:
    return DEFAULT_CACHE_ROOT / f'{model_basename(model_name)}-layer{layer_threshold}'


def resolve_cache_dir(
    *,
    cache_dir: Path | None,
    model_name: str | None,
    layer_threshold: int,
) -> Path:
    if cache_dir is not None:
        return cache_dir
    if model_name is None:
        raise ValueError('--cache_dir or --model_name must be specified')
    return default_cache_dir(model_name, layer_threshold)


def weights_dir(cache_dir: Path) -> Path:
    return cache_dir / WEIGHTS_DIR_NAME


def targets_file(cache_dir: Path) -> Path:
    return cache_dir / TARGETS_FILE_NAME


def metadata_file(cache_dir: Path) -> Path:
    return cache_dir / METADATA_FILE_NAME


def cached_weight_path(cache_dir: Path, target_name: str) -> Path:
    return weights_dir(cache_dir) / f'{target_name}.pt'


def read_targets(cache_dir: Path) -> list[str]:
    path = targets_file(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f'Cached targets file does not exist: {path}')
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def read_metadata(cache_dir: Path) -> dict:
    path = metadata_file(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f'Cache metadata does not exist: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def cache_is_complete(cache_dir: Path) -> bool:
    metadata_path = metadata_file(cache_dir)
    targets_path = targets_file(cache_dir)
    if not metadata_path.exists() or not targets_path.exists():
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
    num_steps: int,
    seed: int,
    workers_per_gpu: int,
    main_gpu_id: int,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError('The input tensor must be 2-dimensional')

    quantizer = BQQ2(weight, rank_scale=rank_scale)
    return quantizer.bqq_large_matrix_multi_worker(
        max_patch_size=group_size,
        bit_width=bit_width,
        save_name=str(save_prefix),
        zeta=4,
        eta=0.06,
        Tinit=0.2,
        Tfin=0.005,
        Nstep=num_steps,
        seed=seed,
        workers_per_gpu=workers_per_gpu,
        main_gpu_id=main_gpu_id,
    )


def prepare_cache(args) -> Path:
    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        layer_threshold=args.layer_threshold,
    )

    if cache_is_complete(cache_dir) and not args.refresh_cache:
        targets = read_targets(cache_dir)
        print(f'Using existing cache: {cache_dir}')
        print(f'Cached targets: {len(targets)}')
        return cache_dir

    if args.refresh_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    target_weights_dir = weights_dir(cache_dir)
    target_weights_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading model for caching: {args.model_name}')
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype='auto').cpu()

    cached_targets: list[dict] = []
    with torch.inference_mode():
        for name, param in model.named_parameters():
            if not is_quantization_target(name):
                continue
            if not passes_layer_threshold(name, args.layer_threshold):
                continue

            save_path = cached_weight_path(cache_dir, name)
            tensor = param.detach().cpu()
            torch.save(tensor, save_path)
            cached_targets.append(
                {
                    'name': name,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'path': str(save_path.relative_to(cache_dir)),
                }
            )
            print(f'Cached {name}: {tuple(tensor.shape)}')

    if not cached_targets:
        raise RuntimeError('No quantization targets matched the current filter settings.')

    targets_file(cache_dir).write_text(
        '\n'.join(target['name'] for target in cached_targets) + '\n',
        encoding='utf-8',
    )

    metadata = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'model_name': args.model_name,
        'model_basename': model_basename(args.model_name),
        'layer_threshold': args.layer_threshold,
        'target_count': len(cached_targets),
        'targets': cached_targets,
    }
    metadata_file(cache_dir).write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Prepared cache: {cache_dir}')
    print(f'Cached targets: {len(cached_targets)}')
    return cache_dir


def list_targets(args) -> None:
    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        layer_threshold=args.layer_threshold,
    )
    for name in read_targets(cache_dir):
        print(name)


def quantize_target(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for cached weight-aware BQQ quantization.')

    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        layer_threshold=args.layer_threshold,
    )
    metadata = read_metadata(cache_dir)
    targets = set(read_targets(cache_dir))
    if args.target_name not in targets:
        raise ValueError(f'Target is not present in cache: {args.target_name}')

    model_name = args.model_name or metadata['model_name']
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(model_name, args.group_size, args.num_steps)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tensor_checkpoint = save_dir / f'{args.target_name}.pth'
    if tensor_checkpoint.exists() and not args.overwrite:
        print(f'Skipping {args.target_name}: already quantized at {tensor_checkpoint}')
        return tensor_checkpoint

    weight_path = cached_weight_path(cache_dir, args.target_name)
    if not weight_path.exists():
        raise FileNotFoundError(f'Cached tensor does not exist: {weight_path}')

    print(f'Loading cached weight: {weight_path}')
    weight = torch.load(weight_path, map_location='cpu')

    print(f'Quantizing {args.target_name}: {tuple(weight.shape)}')
    transformed = quantize_weight(
        weight,
        save_dir / args.target_name,
        bit_width=args.bit_width,
        rank_scale=args.rank_scale,
        group_size=args.group_size,
        num_steps=args.num_steps,
        seed=args.seed,
        workers_per_gpu=args.workers_per_gpu,
        main_gpu_id=args.main_gpu_id,
    )

    torch.save(transformed.cpu(), tensor_checkpoint)
    print(f'Saved reconstructed tensor: {tensor_checkpoint}')
    return tensor_checkpoint


def main():
    args = parse_args()

    if args.command == 'prepare-cache':
        prepare_cache(args)
        return
    if args.command == 'list-targets':
        list_targets(args)
        return
    if args.command == 'quantize-target':
        quantize_target(args)
        return

    raise ValueError(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    main()
