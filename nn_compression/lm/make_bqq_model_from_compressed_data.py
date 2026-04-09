from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

try:
    from . import binary_quadratic_network
    from .compressed_data import (
        default_compressed_data_dir,
        default_quantized_model_dir,
        model_basename,
    )
except ImportError:
    import binary_quadratic_network
    from compressed_data import (
        default_compressed_data_dir,
        default_quantized_model_dir,
        model_basename,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild a BQQ language model from compressed patches")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--bit_widths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--compressed_data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_bqq_model(model_name, compressed_data_dir, bit_width, group_size, num_steps, device, output_dir=None):
    """再構成済みターゲットファイル ({target_name}.pth) から直接 Linear 重みを差し替えて保存する。

    パッチファイル (_row{i}_col{j}.pth) ではなく最終ファイルを使うため、
    数百万パッチの torch.load を回避して高速。
    """
    compressed_data_dir = Path(compressed_data_dir)
    output_dir = Path(output_dir) if output_dir is not None else default_quantized_model_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

    replaced = 0
    skipped = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if 'head' in name:
            print(f"Skipping {name} (head)")
            skipped += 1
            continue

        target_file = compressed_data_dir / f"{name}.weight.pth"
        if not target_file.exists():
            print(f"  [WARN] No reconstructed file for {name}.weight, keeping original")
            skipped += 1
            continue

        weight_reconstructed = torch.load(target_file, map_location=device, weights_only=True)
        module.weight.data = weight_reconstructed.to(module.weight.dtype)
        replaced += 1

    print(f"Replaced {replaced} layers, skipped {skipped}")

    model_id = model_basename(model_name)
    output_path = output_dir / f"{model_id}-{bit_width}bit-{group_size}gs-{num_steps}step.pth"
    torch.save(model, output_path)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | dtype: {param.dtype}")

    print(f"Saved quantized model to {output_path}")
    return output_path


def main():
    args = parse_args()
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
