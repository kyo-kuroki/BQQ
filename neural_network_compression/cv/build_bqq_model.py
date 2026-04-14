"""
Build a BQQ quantized vision model.

Two modes:
  1. From compressed patch data (weight_aware_quant output)
  2. From block-wise files (blockwise_quant output)

Usage:
  # From compressed patches
  python build_bqq_model.py --model_name deit-s \
    --compressed_data_dir bqq_compressed_data/...

  # From block-wise files
  python build_bqq_model.py --model_name deit-s \
    --block_dir blockwise_output/deit-s
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from bqq_modules import (  # noqa: F401
    BinaryQuadratic,
    BQQLinear,
    get_matrices,
    transform_A,
    merge_binary_quadratic,
    merge_binaryquadratic_recursive,
)

from build_model import get_model


# ---------------------------------------------------------------------------
# Replace Linear -> BQQLinear (CV uses {-1,+1} representation)
# ---------------------------------------------------------------------------

def _load_patches_from_dir(weights_dir, full_name, device=None):
    """Load patch files matching a layer name from a directory."""
    weight_list = []
    for file in os.listdir(weights_dir):
        if file.endswith('.pth') and (full_name in file) and ('row' in file):
            path = os.path.join(weights_dir, file)
            weight_list += torch.load(path, map_location=device)
    return weight_list


def replace_linear_with_bqq(model, weights_dir, bit_width, prefix='',
                              device=None, show_tqdm=True):
    """Replace Linear layers with BQQLinear ({-1,+1} representation)."""
    iterator = tqdm(model.named_children()) if show_tqdm else model.named_children()
    for name, module in iterator:
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, nn.Linear):
            map_loc = device if device is not None else module.weight.device
            weight_list = _load_patches_from_dir(weights_dir, full_name, device=map_loc)
            A, Y, Z = get_matrices(weight_list, bit_width=bit_width)
            bqq = BQQLinear(2 * Y - 1, 2 * Z - 1, transform_A(A, l=Y.shape[-1]), bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(module, weights_dir, bit_width, prefix=full_name, show_tqdm=False, device=device)

    return model


# ---------------------------------------------------------------------------
# Build from compressed patches
# ---------------------------------------------------------------------------

def save_bqq_model(model_name, compressed_data_dir, bit_width, group_size, Nstep, device, output_dir=None):
    model = get_model(model_name)
    model = replace_linear_with_bqq(model, weights_dir=str(compressed_data_dir), bit_width=bit_width, device=device)

    if output_dir is None:
        output_dir = SCRIPT_DIR / "quantized_bqq_model"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'{model_name}-{bit_width}bit-{group_size}gs-{Nstep}step-bqq.pth'
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved quantized model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Assemble from block-wise files
# ---------------------------------------------------------------------------

def assemble_from_blocks(model_name, block_dir, output_dir=None):
    """Assemble full model from block_*.pth files (blockwise_quant output)."""
    block_dir = Path(block_dir)
    if output_dir is None:
        output_dir = SCRIPT_DIR / "quantized_bqq_model"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {model_name}")
    model = get_model(model_name)
    num_blocks = len(model.blocks)

    replaced = 0
    for i in range(num_blocks):
        block_path = block_dir / f"block_{i}.pth"
        if block_path.exists():
            print(f"  Loading block {i} from {block_path}")
            block = torch.load(block_path, map_location="cpu", weights_only=False)
            model.blocks[i] = block
            replaced += 1

    if replaced == 0:
        raise FileNotFoundError(f"No block_*.pth files found in {block_dir}")

    print(f"Replaced {replaced}/{num_blocks} blocks")

    output_path = output_dir / f'{model_name}-blockwise-{replaced}of{num_blocks}.pth'
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved assembled model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a BQQ vision model from patches or blocks")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=['deit-s', 'deit-b', 'vit-s', 'vit-b', 'swin-t', 'swin-s'])
    parser.add_argument("--bit_width", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--Nstep", type=int, default=50000)
    parser.add_argument("--compressed_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--block_dir", type=str, default=None,
                        help="Directory containing block_*.pth files (block-wise assembly mode)")
    args = parser.parse_args()

    if args.block_dir is not None:
        assemble_from_blocks(model_name=args.model_name, block_dir=args.block_dir, output_dir=args.output_dir)
    else:
        compressed_data_dir = args.compressed_data_dir
        if compressed_data_dir is None:
            compressed_data_dir = SCRIPT_DIR / "bqq_compressed_data" / \
                f"{args.model_name}-{args.Nstep}step-{args.group_size}gs"
        save_bqq_model(
            model_name=args.model_name, compressed_data_dir=compressed_data_dir,
            bit_width=args.bit_width, group_size=args.group_size,
            Nstep=args.Nstep, device=args.device, output_dir=args.output_dir,
        )


if __name__ == '__main__':
    main()
