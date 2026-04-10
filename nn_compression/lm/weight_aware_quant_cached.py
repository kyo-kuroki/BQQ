from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import re
import shutil

import torch
from transformers import AutoModelForCausalLM

try:
    from .compressed_data import (
        consolidate_target_patches,
        default_compressed_data_dir,
        ensure_bqq_root_on_path,
        get_max_divisor,
        get_patch_layout,
        model_basename,
    )
except ImportError:
    from compressed_data import (
        consolidate_target_patches,
        default_compressed_data_dir,
        ensure_bqq_root_on_path,
        get_max_divisor,
        get_patch_layout,
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
    prepare_parser.add_argument('--layer_threshold', type=int, default=0)
    prepare_parser.add_argument('--layer_max', type=int, default=None,
                                help='Upper bound (inclusive) on layer index; None means no upper limit')
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
    list_parser.add_argument('--layer_threshold', type=int, default=0)
    list_parser.add_argument('--cache_dir', type=Path, default=None)

    list_patches_parser = subparsers.add_parser(
        'list-patches',
        help='Print cached patch jobs as tab-separated target_name, patch_index, patch_row, patch_col',
    )
    list_patches_parser.add_argument('--model_name', type=str, default=None)
    list_patches_parser.add_argument('--layer_threshold', type=int, default=0)
    list_patches_parser.add_argument('--cache_dir', type=Path, default=None)
    list_patches_parser.add_argument('--group_size', type=int, default=128)

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
    quantize_parser.add_argument('--workers_per_gpu', type=int, default=1024)
    quantize_parser.add_argument('--main_gpu_id', type=int, default=0)
    quantize_parser.add_argument('--seed', type=int, default=0)
    quantize_parser.add_argument('--layer_threshold', type=int, default=0)
    quantize_parser.add_argument('--cache_dir', type=Path, default=None)
    quantize_parser.add_argument('--save_dir', type=Path, default=None)
    quantize_parser.add_argument(
        '--patch_index',
        type=int,
        default=None,
        help='0-based patch index within the cached target; when set, only that patch is quantized',
    )
    quantize_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing outputs if they already exist',
    )

    extend_parser = subparsers.add_parser(
        'extend-target',
        help='Extend an existing N-bit quantization by optimizing additional residual bits',
    )
    extend_parser.add_argument('--target_name', type=str, required=True)
    extend_parser.add_argument('--model_name', type=str, default=None)
    extend_parser.add_argument(
        '--source_dir', type=Path, required=True,
        help='Directory containing the existing N-bit quantization patch files',
    )
    extend_parser.add_argument(
        '--save_dir', type=Path, required=True,
        help='Output directory for the extended quantization',
    )
    extend_parser.add_argument(
        '--extra_bits', type=int, default=1,
        help='Number of additional bits to optimise on top of the existing result (default: 1)',
    )
    extend_parser.add_argument('--group_size', type=int, default=32)
    extend_parser.add_argument('--num_steps', type=int, default=10000)
    extend_parser.add_argument('--rank_scale', type=float, default=1.0)
    extend_parser.add_argument('--workers_per_gpu', type=int, default=1024)
    extend_parser.add_argument('--main_gpu_id', type=int, default=0)
    extend_parser.add_argument('--seed', type=int, default=0)
    extend_parser.add_argument('--layer_threshold', type=int, default=0)
    extend_parser.add_argument('--cache_dir', type=Path, default=None)
    extend_parser.add_argument('--overwrite', action='store_true')

    return parser.parse_args()


def is_quantization_target(name: str) -> bool:
    return name.endswith('weight') and all(token not in name for token in ('norm', 'bias', 'emb'))


def passes_layer_threshold(name: str, layer_threshold: int) -> bool:
    layer_ids = [int(num) for num in re.findall(r'\d+', name)]
    return any(layer_id >= layer_threshold for layer_id in layer_ids)


def passes_layer_max(name: str, layer_max: int) -> bool:
    layer_ids = [int(num) for num in re.findall(r'\d+', name)]
    return any(layer_id <= layer_max for layer_id in layer_ids)


def default_cache_dir(model_name: str, layer_threshold: int, layer_max: int | None = None) -> Path:
    if layer_max is not None:
        return DEFAULT_CACHE_ROOT / f'{model_basename(model_name)}-layer{layer_threshold}to{layer_max}'
    return DEFAULT_CACHE_ROOT / f'{model_basename(model_name)}-layer{layer_threshold}'


def resolve_cache_dir(
    *,
    cache_dir: Path | None,
    model_name: str | None,
    layer_threshold: int,
    layer_max: int | None = None,
) -> Path:
    if cache_dir is not None:
        return cache_dir
    if model_name is None:
        raise ValueError('--cache_dir or --model_name must be specified')
    return default_cache_dir(model_name, layer_threshold, layer_max)


def weights_dir(cache_dir: Path) -> Path:
    return cache_dir / WEIGHTS_DIR_NAME


def targets_file(cache_dir: Path) -> Path:
    return cache_dir / TARGETS_FILE_NAME


def metadata_file(cache_dir: Path) -> Path:
    return cache_dir / METADATA_FILE_NAME


def cached_weight_path(cache_dir: Path, target_name: str) -> Path:
    return weights_dir(cache_dir) / f'{target_name}.pt'


def patch_output_path(save_dir: Path, target_name: str, patch_row: int, patch_col: int) -> Path:
    return save_dir / f'{target_name}_row{patch_row}_col{patch_col}.pth'


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


def get_target_metadata(metadata: dict, target_name: str) -> dict:
    for target in metadata.get('targets', []):
        if target.get('name') == target_name:
            return target
    raise ValueError(f'Cache metadata does not contain target: {target_name}')


def get_patch_spec(
    *,
    shape: list[int] | tuple[int, int],
    group_size: int,
    patch_index: int,
) -> dict[str, int]:
    layout = get_patch_layout(shape, group_size)
    total_patches = layout['num_patch_rows'] * layout['num_patch_cols']
    if patch_index < 0 or patch_index >= total_patches:
        raise ValueError(f'patch_index must be in [0, {total_patches - 1}], got {patch_index}')

    patch_row, patch_col = divmod(patch_index, layout['num_patch_cols'])
    row_start = patch_row * layout['patch_height']
    col_start = patch_col * layout['patch_width']

    return {
        **layout,
        'patch_index': patch_index,
        'patch_row': patch_row,
        'patch_col': patch_col,
        'row_start': row_start,
        'row_end': row_start + layout['patch_height'],
        'col_start': col_start,
        'col_end': col_start + layout['patch_width'],
    }


def iter_patch_specs(
    *,
    shape: list[int] | tuple[int, int],
    group_size: int,
):
    layout = get_patch_layout(shape, group_size)
    total_patches = layout['num_patch_rows'] * layout['num_patch_cols']
    for patch_index in range(total_patches):
        yield get_patch_spec(shape=shape, group_size=group_size, patch_index=patch_index)


def extract_patch(weight: torch.Tensor, patch_spec: dict[str, int]) -> torch.Tensor:
    return weight[
        patch_spec['row_start']:patch_spec['row_end'],
        patch_spec['col_start']:patch_spec['col_end'],
    ].detach().clone()


def quantize_weight(
    weight: torch.Tensor,
    consolidated_path: Path,
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
        consolidated_path=str(consolidated_path),
        zeta=4,
        eta=0.06,
        Tinit=0.2,
        Tfin=0.005,
        Nstep=num_steps,
        seed=seed,
        workers_per_gpu=workers_per_gpu,
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
    num_steps: int,
    seed: int,
    main_gpu_id: int,
) -> torch.Tensor:
    if patch.ndim != 2:
        raise ValueError('The input tensor must be 2-dimensional')

    torch.manual_seed(seed)
    torch.cuda.set_device(main_gpu_id)
    device = torch.device(f'cuda:{main_gpu_id}')
    original_x = patch.to(device)
    update_x = original_x.detach().clone()
    decomposed_patches: list[dict] = []

    for bit_idx in range(bit_width):
        decomp_instance = BQQ2(x=update_x.clone(), rank_scale=rank_scale)
        y, z, a = decomp_instance.run_bqq_compile(
            zeta=4,
            eta=0.06,
            Tinit=0.2,
            Tfin=0.005,
            Nstep=num_steps,
            device_id=main_gpu_id,
            seed=seed,
            output_type='torch',
        )
        reconst = a[0] * y @ z
        reconst += a[1] * y.sum(axis=1, keepdim=True)
        reconst += a[2] * z.sum(axis=0, keepdim=True)
        reconst += a[3]
        update_x -= reconst

        decomposed_patches.append(
            {
                'patch_row': patch_row,
                'patch_col': patch_col,
                'coeff': a.cpu(),
                'mat1': y.cpu(),
                'mat2': z.cpu(),
                'bit_idx': bit_idx,
            }
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decomposed_patches, save_path)
    return (original_x - update_x).detach().cpu()


def prepare_cache(args) -> Path:
    layer_max = getattr(args, 'layer_max', None)
    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        layer_threshold=args.layer_threshold,
        layer_max=layer_max,
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
            if layer_max is not None and not passes_layer_max(name, layer_max):
                continue
            if param.ndim != 2:
                print(f"Skipping non-2D target {name}: {tuple(param.shape)}")
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
        'layer_max': layer_max,
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


def list_patches(args) -> None:
    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        layer_threshold=args.layer_threshold,
    )
    metadata = read_metadata(cache_dir)
    for name in read_targets(cache_dir):
        target = get_target_metadata(metadata, name)
        for patch_spec in iter_patch_specs(shape=target['shape'], group_size=args.group_size):
            print(
                f"{name}\t{patch_spec['patch_index']}\t"
                f"{patch_spec['patch_row']}\t{patch_spec['patch_col']}"
            )


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
    target_metadata = get_target_metadata(metadata, args.target_name)

    model_name = args.model_name or metadata['model_name']
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(model_name, args.group_size, args.num_steps)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_index is not None:
        patch_spec = get_patch_spec(
            shape=target_metadata['shape'],
            group_size=args.group_size,
            patch_index=args.patch_index,
        )
        patch_checkpoint = patch_output_path(
            save_dir,
            args.target_name,
            patch_spec['patch_row'],
            patch_spec['patch_col'],
        )
        if patch_checkpoint.exists() and not args.overwrite:
            print(
                'Skipping '
                f"{args.target_name} patch {args.patch_index}: already quantized at {patch_checkpoint}"
            )
            return patch_checkpoint

        weight_path = cached_weight_path(cache_dir, args.target_name)
        if not weight_path.exists():
            raise FileNotFoundError(f'Cached tensor does not exist: {weight_path}')

        print(f'Loading cached weight: {weight_path}')
        weight = torch.load(weight_path, map_location='cpu')
        patch = extract_patch(weight, patch_spec)
        print(
            'Quantizing '
            f"{args.target_name} patch {args.patch_index} "
            f"(row={patch_spec['patch_row']}, col={patch_spec['patch_col']}): {tuple(patch.shape)}"
        )
        quantize_patch(
            patch,
            patch_checkpoint,
            patch_row=patch_spec['patch_row'],
            patch_col=patch_spec['patch_col'],
            bit_width=args.bit_width,
            rank_scale=args.rank_scale,
            num_steps=args.num_steps,
            seed=args.seed,
            main_gpu_id=args.main_gpu_id,
        )
        print(f'Saved compressed patch: {patch_checkpoint}')
        return patch_checkpoint

    consolidated_dir = save_dir / '_consolidated'
    consolidated_file = consolidated_dir / f'{args.target_name}.pth'
    tensor_checkpoint = save_dir / f'{args.target_name}.pth'
    if tensor_checkpoint.exists() and consolidated_file.exists() and not args.overwrite:
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
        consolidated_file,
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


def _reconstruct_from_patch_data(patch_data: list[dict], device: torch.device) -> torch.Tensor:
    """BQQ分解データリストからパッチを再構成する。"""
    y0 = patch_data[0]['mat1'].to(device)
    z0 = patch_data[0]['mat2'].to(device)
    result = torch.zeros(y0.shape[0], z0.shape[1], device=device, dtype=torch.float32)
    for d in patch_data:
        y = d['mat1'].to(device)
        z = d['mat2'].to(device)
        a = d['coeff'].to(device)
        result = result + a[0] * y @ z
        result = result + a[1] * y.sum(axis=1, keepdim=True)
        result = result + a[2] * z.sum(axis=0, keepdim=True)
        result = result + a[3]
    return result


def extend_weight(
    weight: torch.Tensor,
    source_consolidated: Path,
    save_consolidated: Path,
    *,
    extra_bits: int,
    rank_scale: float,
    group_size: int,
    num_steps: int,
    seed: int,
    workers_per_gpu: int,
    main_gpu_id: int,
) -> torch.Tensor:
    """既存 N-bit 分解の残差に extra_bits ビット分の BQQ を追加し、全ビット再構成を返す。

    consolidated ファイルからソースを読み込み、拡張結果も consolidated ファイルに保存する。
    再開時は save_consolidated の既存エントリから完了済みパッチをスキップ。
    """
    torch.manual_seed(seed)
    torch.cuda.set_device(main_gpu_id)
    device = torch.device(f'cuda:{main_gpu_id}')

    quantizer = BQQ2(weight, rank_scale=rank_scale)
    divided_tensor = quantizer.patchify(weight.clone(), max_patch_size=group_size)
    num_patches_h, num_patches_w, _, _ = divided_tensor.shape
    total_patches = num_patches_h * num_patches_w

    # ソースの consolidated ファイルを読み込み、パッチごとにグループ化
    source_data = torch.load(source_consolidated, weights_only=False, map_location='cpu')
    from collections import defaultdict
    source_by_patch = defaultdict(list)
    for p in source_data:
        source_by_patch[(p['patch_row'], p['patch_col'])].append(p)
    source_bits = max(p['bit_idx'] for p in source_data) + 1

    # 保存先の consolidated ファイルが既にあれば再開用に読み込む
    # ただし extend 前のデータ（source_bits 分のみ）の場合はスキップしない
    target_bits = source_bits + extra_bits
    all_decomposed = []
    completed = set()
    if save_consolidated.exists():
        existing = torch.load(save_consolidated, weights_only=False, map_location='cpu')
        existing_max_bit = max(p['bit_idx'] for p in existing) + 1 if existing else 0
        if existing_max_bit >= target_bits:
            # 既に extend 済みのデータ
            all_decomposed = existing
            for p in all_decomposed:
                completed.add((p['patch_row'], p['patch_col']))

    done = 0
    skipped = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if (i, j) in completed:
                # 既に拡張済み: 再構成テンソルを復元
                patches_for_ij = [p for p in all_decomposed if p['patch_row'] == i and p['patch_col'] == j]
                divided_tensor[i, j, :, :] = _reconstruct_from_patch_data(patches_for_ij, device).cpu()
                skipped += 1
                continue

            source_patches = source_by_patch.get((i, j), [])
            if not source_patches:
                raise ValueError(f'Source patch not found: row={i}, col={j}')

            patch = divided_tensor[i, j, :, :].to(device).float()
            existing_reconst = _reconstruct_from_patch_data(source_patches, device)
            update_x = patch - existing_reconst

            new_decompositions = list(source_patches)

            for bit_idx in range(extra_bits):
                decomp = BQQ2(x=update_x.clone(), rank_scale=rank_scale)
                y, z, a = decomp.run_bqq_compile(
                    zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
                    Nstep=num_steps, device_id=main_gpu_id, seed=seed, output_type='torch',
                )
                reconst = a[0] * y @ z
                reconst += a[1] * y.sum(axis=1, keepdim=True)
                reconst += a[2] * z.sum(axis=0, keepdim=True)
                reconst += a[3]
                update_x = update_x - reconst
                new_decompositions.append({
                    'patch_row': i,
                    'patch_col': j,
                    'coeff': a.cpu(),
                    'mat1': y.cpu(),
                    'mat2': z.cpu(),
                    'bit_idx': source_bits + bit_idx,
                })

            all_decomposed.extend(new_decompositions)
            divided_tensor[i, j, :, :] = (patch - update_x).detach().cpu()
            done += 1
            if done % 100 == 0:
                print(f'Progress: {done + skipped}/{total_patches} patches ({skipped} skipped)')
                # 中間保存（walltime kill 対策）
                save_consolidated.parent.mkdir(parents=True, exist_ok=True)
                torch.save(all_decomposed, save_consolidated)

    # 最終保存
    save_consolidated.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_decomposed, save_consolidated)
    print(f'Extend done: {done} processed, {skipped} skipped, total {total_patches}')
    print(f'Saved consolidated: {save_consolidated} ({len(all_decomposed)} entries)')
    return quantizer.unpatchify(divided_tensor, weight.shape)


def extend_target(args) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for BQQ quantization.')

    cache_dir = resolve_cache_dir(
        cache_dir=args.cache_dir,
        model_name=args.model_name,
        layer_threshold=args.layer_threshold,
    )
    metadata = read_metadata(cache_dir)
    targets = set(read_targets(cache_dir))
    if args.target_name not in targets:
        raise ValueError(f'Target is not present in cache: {args.target_name}')

    source_dir = Path(args.source_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    consolidated_dir = save_dir / '_consolidated'
    save_consolidated = consolidated_dir / f'{args.target_name}.pth'
    tensor_checkpoint = save_dir / f'{args.target_name}.pth'
    if tensor_checkpoint.exists() and save_consolidated.exists() and not args.overwrite:
        print(f'Skipping {args.target_name}: already at {tensor_checkpoint}')
        return tensor_checkpoint

    weight_path = cached_weight_path(cache_dir, args.target_name)
    if not weight_path.exists():
        raise FileNotFoundError(f'Cached tensor does not exist: {weight_path}')

    # ソースの consolidated ファイルを確認
    target_metadata = get_target_metadata(metadata, args.target_name)
    source_consolidated = source_dir / '_consolidated' / f'{args.target_name}.pth'
    if not source_consolidated.exists():
        raise FileNotFoundError(
            f'Source consolidated file not found: {source_consolidated}\n'
            f'Run quantize-target with the source bit_width first.'
        )

    print(f'Loading cached weight: {weight_path}')
    weight = torch.load(weight_path, map_location='cpu')

    source_data = torch.load(source_consolidated, weights_only=False, map_location='cpu')
    source_bits = max(p['bit_idx'] for p in source_data) + 1
    print(
        f'Extending {args.target_name}: {tuple(weight.shape)} '
        f'({source_bits}-bit → {source_bits + args.extra_bits}-bit)'
    )

    transformed = extend_weight(
        weight,
        source_consolidated,
        save_consolidated,
        extra_bits=args.extra_bits,
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
    if args.command == 'list-patches':
        list_patches(args)
        return
    if args.command == 'quantize-target':
        quantize_target(args)
        return
    if args.command == 'extend-target':
        extend_target(args)
        return

    raise ValueError(f'Unsupported command: {args.command}')


if __name__ == '__main__':
    main()
