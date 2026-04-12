"""Verify all consolidated files and re-quantize broken ones.

Usage:
    python verify_and_fix.py \
        --cache_dir  cache/Qwen3.5-4B-layer4 \
        --save_dir   bqq_compressed_data/Qwen3.5-4B-32gs-10000step \
        --bit_width  3 --group_size 32 --num_steps 10000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def get_max_divisor(num: int, max_value: int) -> int:
    limit = max(int(math.sqrt(num)), max_value)
    for i in range(limit, 0, -1):
        if num % i == 0 and i <= max_value:
            return i
    return 1


def verify_consolidated(pth_path: Path, shape: list[int], group_size: int, bit_width: int) -> str | None:
    """Return error description if broken, None if OK."""
    if not pth_path.exists():
        return "MISSING"
    try:
        data = torch.load(pth_path, weights_only=False, map_location='cpu')
    except Exception as e:
        return f"CORRUPT: {e}"
    if not data:
        return "EMPTY"

    h, w = shape
    ph = get_max_divisor(h, group_size)
    pw = get_max_divisor(w, group_size)
    expected_rows = h // ph
    expected_cols = w // pw
    expected_patches = expected_rows * expected_cols

    max_bit = max(p['bit_idx'] for p in data) + 1
    unique_patches = set((p['patch_row'], p['patch_col']) for p in data)
    max_row = max(p['patch_row'] for p in data) + 1
    max_col = max(p['patch_col'] for p in data) + 1

    issues = []
    if max_row != expected_rows:
        issues.append(f"rows {max_row}/{expected_rows}")
    if max_col != expected_cols:
        issues.append(f"cols {max_col}/{expected_cols}")
    if len(unique_patches) < expected_patches:
        issues.append(f"patches {len(unique_patches)}/{expected_patches}")
    if max_bit < bit_width:
        issues.append(f"bits {max_bit}/{bit_width}")

    return "; ".join(issues) if issues else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=Path, required=True)
    parser.add_argument('--save_dir', type=Path, required=True)
    parser.add_argument('--bit_width', type=int, default=3)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--fix', action='store_true', help='Delete broken files and re-quantize')
    args = parser.parse_args()

    metadata_path = args.cache_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    targets = metadata['targets']

    consolidated_dir = args.save_dir / '_consolidated'

    broken = []
    ok = 0
    for t in targets:
        name = t['name']
        shape = t['shape']
        pth = consolidated_dir / f'{name}.pth'
        issue = verify_consolidated(pth, shape, args.group_size, args.bit_width)
        if issue:
            print(f"BROKEN: {name} | {issue}")
            broken.append(name)
        else:
            ok += 1

    print(f"\nTotal: OK={ok}, BROKEN={len(broken)}/{len(targets)}")

    if not args.fix or not broken:
        if broken:
            # Write broken list for external use
            broken_file = args.save_dir / '_broken_targets.txt'
            broken_file.write_text('\n'.join(broken) + '\n')
            print(f"Broken list saved to {broken_file}")
        return

    # Delete broken files and re-quantize
    print(f"\nFixing {len(broken)} targets...")
    for name in broken:
        consolidated = consolidated_dir / f'{name}.pth'
        tensor_ckpt = args.save_dir / f'{name}.pth'
        if consolidated.exists():
            consolidated.unlink()
            print(f"  Deleted {consolidated}")
        if tensor_ckpt.exists():
            tensor_ckpt.unlink()
            print(f"  Deleted {tensor_ckpt}")

    # Re-quantize using batch mode
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from quantizer import BinaryQuadraticQuantization as BQQ2

    for i, name in enumerate(broken):
        weight_path = args.cache_dir / 'weights' / f'{name}.pt'
        if not weight_path.exists():
            print(f"  SKIP (no cache): {name}")
            continue

        weight = torch.load(weight_path, map_location='cpu')
        consolidated_path = consolidated_dir / f'{name}.pth'

        print(f"  [{i+1}/{len(broken)}] Quantizing {name}: {tuple(weight.shape)}")
        quantizer = BQQ2(weight, rank_scale=1.0)
        reconstructed = quantizer.bqq_large_matrix_multi_worker(
            max_patch_size=args.group_size,
            bit_width=args.bit_width,
            consolidated_path=str(consolidated_path),
            zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
            Nstep=args.num_steps, seed=0, main_gpu_id=0,
            use_batch=True,
        )

        tensor_ckpt = args.save_dir / f'{name}.pth'
        torch.save(reconstructed.cpu(), tensor_ckpt)
        print(f"  Saved: {tensor_ckpt}")

    print("All fixes complete.")


if __name__ == '__main__':
    main()
