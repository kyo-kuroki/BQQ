"""Benchmark end-to-end inference latency: BQQ (CUDA kernel) vs FP16.

Creates a BQQ model by replacing all Linear layers with random BinaryQuadratic
(no real quantization, just correct shapes for timing).

Usage:
    python benchmark_inference.py --model_name Qwen/Qwen2.5-0.5B --bit_width 4
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _src_dir)
from bqq_modules import BinaryQuadratic, PackedBinaryQuadratic, pack_binaryquadratic_model


def make_random_bqq(out_f, in_f, bit_width, gs, bias=None):
    """Create a BinaryQuadratic with random binary data (for benchmarking only)."""
    rw, cw = out_f // gs, in_f // gs
    Y = torch.randint(0, 2, (bit_width, rw, cw, gs, gs)).bool()
    Z = torch.randint(0, 2, (bit_width, rw, cw, gs, gs)).bool()
    A = torch.randn(bit_width, rw, cw, 4) * 0.01
    bq = BinaryQuadratic.__new__(BinaryQuadratic)
    nn.Module.__init__(bq)
    bq.bit_width, bq.row_width, bq.col_width = bit_width, rw, cw
    bq.y_row, bq.inter_dimension, bq.z_col = gs, gs, gs
    bq.register_buffer('Y', Y)
    bq.register_buffer('Z', Z)
    bq.a = nn.Parameter(A[..., 0].unsqueeze(-1).unsqueeze(-1))
    bq.b = nn.Parameter(A[..., 1].unsqueeze(-1).unsqueeze(-1))
    bq.c = nn.Parameter(A[..., 2].unsqueeze(-1).unsqueeze(-1))
    bq.d = nn.Parameter(A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0))
    bq.bias = nn.Parameter(bias.clone()) if bias is not None else None
    return bq


def replace_linears_with_bqq(model, bit_width, gs, prefix=''):
    """Replace all nn.Linear in model with random BinaryQuadratic."""
    for name, module in list(model.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if 'head' in full or 'embed' in full:
            continue
        if isinstance(module, nn.Linear):
            out_f, in_f = module.weight.shape
            if out_f % gs != 0 or in_f % gs != 0:
                print(f"  Skip {full}: {out_f}x{in_f} not divisible by gs={gs}")
                continue
            bqq = make_random_bqq(out_f, in_f, bit_width, gs, module.bias)
            setattr(model, name, bqq)
        else:
            replace_linears_with_bqq(module, bit_width, gs, prefix=full)


def benchmark_generate(model, tokenizer, prompt, max_new_tokens, warmup=3, repeats=5):
    """Benchmark autoregressive generation latency."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens,
                           do_sample=False, use_cache=True)

    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    total_tokens = out.shape[1] - input_len
    avg_time = sum(times) / len(times)
    return avg_time, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--bit_width", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--pack", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # === 1. FP16 baseline ===
    print(f"Loading FP16 model: {args.model_name}")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype="auto", trust_remote_code=True
    ).to(device).eval()

    print("Benchmarking FP16...")
    t_fp16, n_tokens = benchmark_generate(
        model_fp16, tokenizer, args.prompt, args.max_new_tokens)
    print(f"  FP16: {t_fp16*1000:.1f} ms total, {n_tokens} tokens, "
          f"{t_fp16/n_tokens*1000:.1f} ms/token")

    # === 2. BQQ model ===
    print(f"\nBuilding BQQ model ({args.bit_width}-bit, gs={args.group_size})...")
    model_bqq = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype="auto", trust_remote_code=True
    ).to("cpu")

    replace_linears_with_bqq(model_bqq, args.bit_width, args.group_size)

    if args.pack:
        print("Packing BinaryQuadratic → PackedBinaryQuadratic...")
        pack_binaryquadratic_model(model_bqq)
        PackedBinaryQuadratic.use_zy_x_kernel = True

    model_bqq = model_bqq.to(device).eval()

    # Pre-compile CUDA kernel (first call triggers JIT compilation)
    print("Pre-compiling CUDA kernel...")
    dummy = tokenizer("warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        model_bqq(dummy.input_ids)
    torch.cuda.synchronize()
    print("  CUDA kernel compiled.")

    print("Benchmarking BQQ...")
    t_bqq, n_tokens_bqq = benchmark_generate(
        model_bqq, tokenizer, args.prompt, args.max_new_tokens)
    print(f"  BQQ:  {t_bqq*1000:.1f} ms total, {n_tokens_bqq} tokens, "
          f"{t_bqq/n_tokens_bqq*1000:.1f} ms/token")

    # === 3. Summary ===
    print(f"\n{'='*50}")
    print(f"Model: {args.model_name} ({args.bit_width}-bit BQQ, gs={args.group_size})")
    print(f"Tokens generated: {n_tokens}")
    print(f"  FP16:  {t_fp16/n_tokens*1000:.2f} ms/token")
    print(f"  BQQ:   {t_bqq/n_tokens_bqq*1000:.2f} ms/token")
    print(f"  Ratio: {t_bqq/t_fp16:.2f}x")

    # Memory comparison
    fp16_mem = sum(p.numel() * p.element_size() for p in model_fp16.parameters()) / 1e6
    bqq_params = sum(p.numel() * p.element_size() for p in model_bqq.parameters()) / 1e6
    bqq_buffers = sum(b.numel() * b.element_size() for b in model_bqq.buffers()) / 1e6
    bqq_mem = bqq_params + bqq_buffers
    print(f"  FP16 model size: {fp16_mem:.0f} MB")
    print(f"  BQQ model size:  {bqq_mem:.0f} MB ({fp16_mem/bqq_mem:.1f}x compression)")

    del model_fp16, model_bqq
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
