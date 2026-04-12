from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

try:
    from . import binary_quadratic_network
    from .compressed_data import (
        consolidate_all_patches,
        default_compressed_data_dir,
        default_quantized_model_dir,
        model_basename,
    )
except ImportError:
    import binary_quadratic_network
    from compressed_data import (
        consolidate_all_patches,
        default_compressed_data_dir,
        default_quantized_model_dir,
        model_basename,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild a BQQ language model from compressed patches or block-wise files")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--bit_widths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--compressed_data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # Block-wise assembly
    parser.add_argument("--block_dir", type=Path, default=None,
                        help="Directory containing block_*.pth files from block_wise_quant.py. "
                             "If specified, assembles model from blocks instead of compressed patches.")
    return parser.parse_args()


def save_bqq_model(model_name, compressed_data_dir, bit_width, group_size, num_steps, device, output_dir=None):
    compressed_data_dir = Path(compressed_data_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_quantized_model_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model = binary_quadratic_network.replace_linear_with_bqq(
        model,
        weights_dir=str(compressed_data_dir),
        bit_width=bit_width,
        device=device,
    )

    model_id = model_basename(model_name)
    output_path = output_dir / f"{model_id}-{bit_width}bit-{group_size}gs-{num_steps}step.pth"
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")

    print(f"Saved quantized model to {output_path}")
    return output_path


def assemble_from_blocks(model_name, block_dir, output_dir=None):
    """
    block_wise_quant.py が出力した block_*.pth ファイル群から
    フルモデルを組み立てて保存する。

    各 block_*.pth は torch.save(block_module) で保存された
    量子化済みブロック (BinaryQuadratic + 最適化済み連続パラメータ含む)。
    """
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


def main():
    args = parse_args()

    if args.block_dir is not None:
        # Block-wise assembly mode
        assemble_from_blocks(
            model_name=args.model_name,
            block_dir=args.block_dir,
            output_dir=args.output_dir,
        )
    else:
        # Original patch-based assembly
        compressed_data_dir = args.compressed_data_dir
        if compressed_data_dir is None:
            compressed_data_dir = default_compressed_data_dir(args.model_name, args.group_size, args.num_steps)

        for bit_width in args.bit_widths:
            save_bqq_model(
                model_name=args.model_name,
                compressed_data_dir=compressed_data_dir,
                bit_width=bit_width,
                group_size=args.group_size,
                num_steps=args.num_steps,
                device=args.device,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
