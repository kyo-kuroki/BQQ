"""
Build a BQQ quantized vision model (DeiT / ViT / Swin).

Two modes:
  1. From compressed patch data (layerwise_quant output)
  2. From block-wise files (blockwise_quant output)

Usage:
  # From compressed patches
  python build_bqq_model.py build --model_name deit-s \
    --compressed_data_dir bqq_compressed_data/... --bit_width 2 --group_size 32

  # From block-wise files
  python build_bqq_model.py assemble --model_name deit-s \
    --block_dir blockwise_output/deit-s
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from .compressed_data import (
        build_consolidated_index,
        build_patch_index,
        consolidate_all_patches,
        default_compressed_data_dir,
        default_quantized_model_dir,
        load_layer_patches,
        model_basename,
    )
    from .model_loader import get_num_blocks, get_vision_model, set_block
except ImportError:
    from compressed_data import (
        build_consolidated_index,
        build_patch_index,
        consolidate_all_patches,
        default_compressed_data_dir,
        default_quantized_model_dir,
        load_layer_patches,
        model_basename,
    )
    from model_loader import get_num_blocks, get_vision_model, set_block

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'bqqkernel'))
from neural_network_compression.bqqkernel.bqq_modules import (  # noqa: F401
    BinaryQuadratic,
    BQQLinear,
    get_matrices,
    merge_binary_quadratic,
    merge_binaryquadratic_recursive,
    transform_A,
)


# ---------------------------------------------------------------------------
# Replace Linear -> BQQLinear (CV uses {-1,+1} representation)
# ---------------------------------------------------------------------------

def replace_linear_with_bqq(model, weights_dir, bit_width, prefix='',
                              device=None, show_tqdm=True, patch_index=None):
    """Recursively replace nn.Linear layers with BQQLinear ({-1,+1} representation)."""
    if patch_index is None:
        patch_index = build_consolidated_index(weights_dir) or build_patch_index(weights_dir)

    iterator = tqdm(model.named_children()) if show_tqdm else model.named_children()
    for name, module in iterator:
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, nn.Linear):
            weight_key = f"{full_name}.weight"
            patch_list = load_layer_patches(
                weight_key, patch_index,
                map_location=device if device is not None else module.weight.device,
            )
            if not patch_list:
                print(f"  [WARN] No patches for {weight_key}, keeping original Linear")
                continue

            A, Y, Z = get_matrices(patch_list, bit_width=bit_width)
            bqq = BQQLinear(2 * Y - 1, 2 * Z - 1, transform_A(A, l=Y.shape[-1]), bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(
                module, weights_dir, bit_width,
                prefix=full_name, show_tqdm=False, device=device, patch_index=patch_index,
            )

    return model


# ---------------------------------------------------------------------------
# Build full model from compressed patches
# ---------------------------------------------------------------------------

def save_bqq_model(model_name, compressed_data_dir, bit_width, group_size, num_steps, device, output_dir=None):
    compressed_data_dir = Path(compressed_data_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_quantized_model_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = get_vision_model(model_name)
    model = replace_linear_with_bqq(model, weights_dir=str(compressed_data_dir), bit_width=bit_width, device=device)

    model_id = model_basename(model_name)
    output_path = output_dir / f"{model_id}-{bit_width}bit-{group_size}gs-{num_steps}step-bqq.pth"
    torch.save(model, output_path)

    print(f"Saved quantized model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Assemble full model from block-wise files
# ---------------------------------------------------------------------------

def assemble_from_blocks(model_name, block_dir, output_dir=None, name_suffix=""):
    """Assemble full model from block_*.pth files (blockwise_quant output)."""
    block_dir = Path(block_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_quantized_model_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {model_name}")
    model = get_vision_model(model_name)
    num_blocks = get_num_blocks(model)

    replaced = 0
    for i in range(num_blocks):
        block_path = block_dir / f"block_{i}.pth"
        if block_path.exists():
            print(f"  Loading block {i} from {block_path}")
            block = torch.load(block_path, map_location="cpu", weights_only=False)
            set_block(model, i, block)
            replaced += 1

    if replaced == 0:
        raise FileNotFoundError(f"No block_*.pth files found in {block_dir}")

    print(f"Replaced {replaced}/{num_blocks} blocks")

    model_id = model_basename(model_name)
    suffix = "-blockwise"
    if name_suffix:
        suffix += f"-{name_suffix.lstrip('-')}"
    output_path = output_dir / f"{model_id}{suffix}.pth"
    torch.save(model, output_path)

    print(f"Saved assembled model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a BQQ vision model from patches or blocks")
    sub = parser.add_subparsers(dest="command")

    # --- build (from compressed patches) ---
    p_build = sub.add_parser("build", help="Build BQQ model from compressed patches")
    p_build.add_argument("--model_name", type=str, required=True)
    p_build.add_argument("--bit_widths", type=int, nargs="+", default=[2])
    p_build.add_argument("--group_size", type=int, default=32)
    p_build.add_argument("--num_steps", type=int, default=20000)
    p_build.add_argument("--compressed_data_dir", type=Path, default=None)
    p_build.add_argument("--output_dir", type=Path, default=None)
    p_build.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # --- assemble (from block files) ---
    p_asm = sub.add_parser("assemble", help="Assemble model from block_*.pth files")
    p_asm.add_argument("--model_name", type=str, required=True)
    p_asm.add_argument("--block_dir", type=Path, required=True)
    p_asm.add_argument("--output_dir", type=Path, default=None)
    p_asm.add_argument("--name_suffix", type=str, default="")

    args = parser.parse_args()

    if args.command == "assemble":
        assemble_from_blocks(model_name=args.model_name, block_dir=args.block_dir,
                             output_dir=args.output_dir, name_suffix=args.name_suffix)
    else:  # build (default)
        if not getattr(args, 'model_name', None):
            parser.print_help()
            return
        compressed_data_dir = args.compressed_data_dir
        for bit_width in args.bit_widths:
            effective_dir = compressed_data_dir
            if effective_dir is None:
                effective_dir = default_compressed_data_dir(args.model_name, bit_width, args.group_size, args.num_steps)
            save_bqq_model(
                model_name=args.model_name, compressed_data_dir=effective_dir,
                bit_width=bit_width, group_size=args.group_size,
                num_steps=args.num_steps, device=args.device, output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
