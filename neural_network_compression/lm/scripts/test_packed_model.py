"""
Test that PackedBinaryQuadratic produces the same output as BinaryQuadratic.

Usage:
    python test_packed_model.py \
        --unpacked quantized_model_data/Qwen3.5-2B-2bit-64gs-blockwise.pth \
        --packed   quantized_model_data/Qwen3.5-2B-2bit-64gs-blockwise-packed.pth
"""
import argparse
import sys
import os
import torch

# Make bqq_modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import bqq_modules  # noqa: F401 (needed for torch.load deserialization)
from bqq_modules import BinaryQuadratic, PackedBinaryQuadratic


def count_bqq_layers(model):
    n_bq, n_pbq = 0, 0
    for m in model.modules():
        if isinstance(m, BinaryQuadratic):
            n_bq += 1
        elif isinstance(m, PackedBinaryQuadratic):
            n_pbq += 1
    return n_bq, n_pbq


def compare_models(unpacked_path, packed_path, device="cpu", seq_len=16, seed=42):
    print(f"Loading unpacked model from {unpacked_path} ...")
    model_u = torch.load(unpacked_path, map_location=device, weights_only=False)
    model_u.eval()

    print(f"Loading packed model from {packed_path} ...")
    model_p = torch.load(packed_path, map_location=device, weights_only=False)
    model_p.eval()

    n_bq, _ = count_bqq_layers(model_u)
    _, n_pbq = count_bqq_layers(model_p)
    print(f"Unpacked: {n_bq} BinaryQuadratic layers")
    print(f"Packed:   {n_pbq} PackedBinaryQuadratic layers")

    # Check file sizes
    size_u = os.path.getsize(unpacked_path) / 1e9
    size_p = os.path.getsize(packed_path) / 1e9
    print(f"\nFile size: {size_u:.2f} GB  →  {size_p:.2f} GB  ({size_u/size_p:.2f}x reduction)")

    # Build a small random input_ids
    torch.manual_seed(seed)
    vocab_size = model_u.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    print(f"\nRunning forward pass (device={device}, seq_len={seq_len}) ...")
    with torch.no_grad():
        out_u = model_u(input_ids).logits  # [1, seq_len, vocab]
        out_p = model_p(input_ids).logits

    max_abs_diff = (out_u - out_p).abs().max().item()
    mean_abs_diff = (out_u - out_p).abs().mean().item()
    max_val = out_u.abs().max().item()
    rel_err = max_abs_diff / (max_val + 1e-9)

    print(f"\nOutput comparison:")
    print(f"  max |diff|  = {max_abs_diff:.6e}")
    print(f"  mean |diff| = {mean_abs_diff:.6e}")
    print(f"  max |logit| = {max_val:.4f}")
    print(f"  rel error   = {rel_err:.6e}")

    # Check argmax agreement (token prediction)
    pred_u = out_u.argmax(-1)
    pred_p = out_p.argmax(-1)
    token_match = (pred_u == pred_p).float().mean().item()
    print(f"  argmax match= {token_match*100:.1f}%")

    # Accept if max absolute diff < 1e-3 (numerical noise from int→float cast is ~1e-5)
    tolerance = 1e-3
    passed = max_abs_diff < tolerance
    print(f"\n{'[PASS]' if passed else '[FAIL]'} max |diff| {max_abs_diff:.2e} < {tolerance:.0e}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unpacked", type=str, required=True)
    parser.add_argument("--packed",   type=str, required=True)
    parser.add_argument("--device",   type=str, default="cpu")
    parser.add_argument("--seq_len",  type=int, default=16)
    args = parser.parse_args()

    ok = compare_models(args.unpacked, args.packed, device=args.device, seq_len=args.seq_len)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
