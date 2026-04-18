"""
Build a BQQ quantized language model.

Three modes:
  1. From compressed patch data (weight_aware_quant output)
  2. From block-wise files (blockwise_quant output)
  3. Programmatic: replace_linear_with_bqq() for custom pipelines

Evaluation utility:
  dequantize_bqq_model(model) — convert BQQ layers to nn.Linear at load time
  for standard FP inference speed (no permanent file written).

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
    PackedBinaryQuadratic,
    PartialBQQLinear,
    get_matrices,
    merge_binary_quadratic,
    merge_binaryquadratic_recursive,
    pack_binaryquadratic_model,
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


def dequantize_bqq_model(model: nn.Module, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Recursively replace BQQ layers with nn.Linear for standard FP inference.

    Call at evaluation time after torch.load — does not modify the saved .pth.
    Safe to call on non-BQQ models (no-op if no BQQ layers are found).
    """
    for child_name, child_module in list(model.named_children()):
        if isinstance(child_module, (BinaryQuadratic, PackedBinaryQuadratic)):
            W = child_module.get_weight(dtype=dtype)          # [out, in]
            has_bias = child_module.bias is not None
            linear = nn.Linear(
                W.shape[1], W.shape[0], bias=has_bias,
                device=W.device, dtype=dtype,
            )
            linear.weight = nn.Parameter(W)
            if has_bias:
                linear.bias = nn.Parameter(child_module.bias.to(dtype=dtype))
            setattr(model, child_name, linear)
        else:
            dequantize_bqq_model(child_module, dtype=dtype)
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
    output_path = output_dir / f"{model_id}-{bit_width}bit-{group_size}gs.pth"
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved quantized model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Assemble full model from block-wise files
# ---------------------------------------------------------------------------

def assemble_from_blocks(model_name, block_dir, bit_width=None, group_size=None,
                          output_dir=None, pack=False, name_suffix=""):
    """Assemble full model from block_*.pth files (blockwise_quant output)."""
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

    # Unify dtype to bfloat16 — blockwise blocks are saved in float32
    # (for training precision) but BinaryQuadratic.forward() casts to
    # input dtype, so bfloat16 works for inference and saves memory.
    model = model.bfloat16()

    if pack:
        print("Packing BinaryQuadratic layers (bool → uint8 packbits) ...")
        pack_binaryquadratic_model(model)

    model_id = model_basename(model_name)
    suffix = "-blockwise"
    if bit_width is not None and group_size is not None:
        suffix = f"-{bit_width}bit-{group_size}gs-blockwise"
    if name_suffix:
        suffix += f"-{name_suffix.lstrip('-')}"
    if pack:
        suffix += "-packed"
    output_path = output_dir / f"{model_id}{suffix}.pth"
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved assembled model to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Pack an existing assembled model
# ---------------------------------------------------------------------------

def pack_existing_model(input_path, output_path=None):
    """Convert all BinaryQuadratic layers in a saved .pth to PackedBinaryQuadratic."""
    input_path = Path(input_path)
    if output_path is None:
        stem = input_path.stem
        output_path = input_path.with_name(stem + "-packed.pth")
    output_path = Path(output_path)

    print(f"Loading model from {input_path} ...")
    model = torch.load(input_path, map_location="cpu", weights_only=False)

    print("Packing BinaryQuadratic layers ...")
    pack_binaryquadratic_model(model)

    torch.save(model, output_path)
    print(f"Saved packed model to {output_path}")

    size_in = input_path.stat().st_size / 1e9
    size_out = output_path.stat().st_size / 1e9
    print(f"File size: {size_in:.2f} GB → {size_out:.2f} GB  ({size_in/size_out:.1f}x reduction)")
    return output_path


# ---------------------------------------------------------------------------
# Export to HuggingFace format
# ---------------------------------------------------------------------------

def export_hf(bqq_model_path, model_name, output_dir=None):
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
    p_asm.add_argument("--bit_width", type=int, default=None)
    p_asm.add_argument("--group_size", type=int, default=None)
    p_asm.add_argument("--output_dir", type=Path, default=None)
    p_asm.add_argument("--pack", action="store_true",
                        help="Convert BinaryQuadratic to PackedBinaryQuadratic (8x smaller Y/Z)")
    p_asm.add_argument("--name_suffix", type=str, default="",
                        help="Extra suffix appended before .pth (e.g. 'revH' -> ...-blockwise-revH.pth)")

    # --- pack ---
    p_pack = sub.add_parser("pack", help="Pack an existing assembled .pth model")
    p_pack.add_argument("--input", type=Path, required=True,
                         help="Path to existing assembled .pth file")
    p_pack.add_argument("--output", type=Path, default=None,
                         help="Output path (default: <input>-packed.pth)")

    # --- export-hf ---
    p_hf = sub.add_parser("export-hf", help="Export BQQ model to HuggingFace format")
    p_hf.add_argument("--model_name", type=str, required=True,
                       help="Base HuggingFace model name (e.g. Qwen/Qwen3-2B)")
    p_hf.add_argument("--bqq_model", type=Path, required=True,
                       help="Path to BQQ .pth model file")
    p_hf.add_argument("--output_dir", type=Path, default=None,
                       help="Output directory (default: bqq/{ModelName}-{N}bit)")

    args = parser.parse_args()

    if args.command == "assemble":
        assemble_from_blocks(model_name=args.model_name, block_dir=args.block_dir,
                             bit_width=args.bit_width, group_size=args.group_size,
                             output_dir=args.output_dir, pack=args.pack,
                             name_suffix=args.name_suffix)

    elif args.command == "pack":
        pack_existing_model(args.input, args.output)

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
