"""
Build a BQQ quantized language model.

Three modes:
  1. From compressed patch data (weight_aware_quant output)
  2. From block-wise files (block_wise_quant output)
  3. Programmatic: replace_linear_with_bqq() for custom pipelines

Usage:
  # From compressed patches
  python build_bqq_model.py --model_name Qwen/Qwen3-2B \
    --compressed_data_dir bqq_compressed_data/... --bit_widths 2 3 4

  # From block-wise files
  python build_bqq_model.py --model_name Qwen/Qwen3-2B \
    --block_dir blockwise_output/Qwen3-2B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from bqq_modules import (  # noqa: F401
    BinaryQuadratic,
    get_matrices,
    merge_binary_quadratic,
    merge_binaryquadratic_recursive,
)


# ---------------------------------------------------------------------------
# Replace Linear -> BinaryQuadratic
# ---------------------------------------------------------------------------

def _load_layer_matrices(layer_name, patch_index, bit_width, map_location):
    patch_list = load_layer_patches(layer_name, patch_index, map_location=map_location)
    if not patch_list:
        return None
    return get_matrices(patch_list, bit_width=bit_width)


def replace_linear_with_bqq(model, weights_dir, bit_width, prefix='',
                              device=None, show_tqdm=True, patch_index=None):
    """Recursively replace nn.Linear layers with BinaryQuadratic modules."""
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
            matrices = _load_layer_matrices(
                weight_key, patch_index, bit_width,
                map_location=device if device is not None else module.weight.device,
            )
            if matrices is None:
                print(f"  [WARN] No patches for {weight_key}, keeping original Linear")
                continue

            A, Y, Z = matrices
            bqq = BinaryQuadratic(Y, Z, A, bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(
                module, weights_dir, bit_width,
                prefix=full_name, show_tqdm=False, device=device, patch_index=patch_index,
            )

    return model


def replace_weight(model, weights_dir, bit_width):
    """Replace weight data in-place (without changing module type)."""
    patch_index = build_consolidated_index(weights_dir) or build_patch_index(weights_dir)

    for name, param in model.named_parameters():
        if 'head' in name:
            print(f"Skipping {name}")
            continue

        if 'norm' not in name and 'bias' in name:
            print(f"Replacing {name}, shape: {tuple(param.shape)}")
            matrices = _load_layer_matrices(name, patch_index, bit_width, map_location=param.device)
            if matrices is None:
                continue
            A, Y, Z = matrices
            bqq = BinaryQuadratic(Y, Z, A, bias=None)
            param.data.copy_(bqq.get_weight())

    return model


# ---------------------------------------------------------------------------
# Build full model from compressed patches
# ---------------------------------------------------------------------------

def save_bqq_model(model_name, compressed_data_dir, bit_width, group_size, num_steps, device, output_dir=None):
    compressed_data_dir = Path(compressed_data_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_quantized_model_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    model = replace_linear_with_bqq(model, weights_dir=str(compressed_data_dir), bit_width=bit_width, device=device)

    model_id = model_basename(model_name)
    output_path = output_dir / f"{model_id}-{bit_width}bit-{group_size}gs-{num_steps}step.pth"
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved quantized model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Assemble full model from block-wise files
# ---------------------------------------------------------------------------

def assemble_from_blocks(model_name, block_dir, output_dir=None):
    """Assemble full model from block_*.pth files (block_wise_quant output)."""
    block_dir = Path(block_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_quantized_model_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    num_layers = len(model.model.layers)

    replaced = 0
    for i in range(num_layers):
        block_path = block_dir / f"block_{i}.pth"
        if block_path.exists():
            print(f"  Loading block {i} from {block_path}")
            block = torch.load(block_path, map_location="cpu", weights_only=False)
            model.model.layers[i] = block
            replaced += 1

    if replaced == 0:
        raise FileNotFoundError(f"No block_*.pth files found in {block_dir}")

    print(f"Replaced {replaced}/{num_layers} blocks")

    model_id = model_basename(model_name)
    output_path = output_dir / f"{model_id}-blockwise-{replaced}of{num_layers}.pth"
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved assembled model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Export to HuggingFace format
# ---------------------------------------------------------------------------

def export_hf(bqq_model_path, model_name, output_dir):
    """Export a BQQ .pth model to HuggingFace format (trust_remote_code)."""
    from export_hf import export_for_hf

    print(f"Loading BQQ model from {bqq_model_path}")
    bqq_model = torch.load(bqq_model_path, map_location="cpu", weights_only=False)
    export_for_hf(bqq_model, model_name, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a BQQ language model from patches or blocks")
    sub = parser.add_subparsers(dest="command")

    # --- build (default, backward-compat) ---
    p_build = sub.add_parser("build", help="Build BQQ model from compressed patches")
    p_build.add_argument("--model_name", type=str, required=True)
    p_build.add_argument("--bit_widths", type=int, nargs="+", default=[2, 3, 4])
    p_build.add_argument("--group_size", type=int, default=128)
    p_build.add_argument("--num_steps", type=int, default=50000)
    p_build.add_argument("--compressed_data_dir", type=Path, default=None)
    p_build.add_argument("--output_dir", type=Path, default=None)
    p_build.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    # --- assemble ---
    p_asm = sub.add_parser("assemble", help="Assemble model from block_*.pth files")
    p_asm.add_argument("--model_name", type=str, required=True)
    p_asm.add_argument("--block_dir", type=Path, required=True)
    p_asm.add_argument("--output_dir", type=Path, default=None)

    # --- export-hf ---
    p_hf = sub.add_parser("export-hf", help="Export BQQ model to HuggingFace format")
    p_hf.add_argument("--model_name", type=str, required=True,
                       help="Base HuggingFace model name (e.g. Qwen/Qwen3-2B)")
    p_hf.add_argument("--bqq_model", type=Path, required=True,
                       help="Path to BQQ .pth model file")
    p_hf.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for HuggingFace model")

    args = parser.parse_args()

    if args.command == "assemble":
        assemble_from_blocks(model_name=args.model_name, block_dir=args.block_dir, output_dir=args.output_dir)

    elif args.command == "export-hf":
        export_hf(args.bqq_model, args.model_name, args.output_dir)

    else:  # build (default)
        if not hasattr(args, 'model_name') or args.model_name is None:
            parser.print_help()
            return
        compressed_data_dir = args.compressed_data_dir
        if compressed_data_dir is None:
            compressed_data_dir = default_compressed_data_dir(args.model_name, args.group_size, args.num_steps)
        for bit_width in args.bit_widths:
            save_bqq_model(
                model_name=args.model_name, compressed_data_dir=compressed_data_dir,
                bit_width=bit_width, group_size=args.group_size,
                num_steps=args.num_steps, device=args.device, output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
