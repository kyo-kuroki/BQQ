"""
Block-wise BQQ quantization with block output error optimization.

Pipeline:
  1. Load pretrained model, cache each block's input/output via calibration data
  2. For target block, sequentially for each Linear weight:
     a. BQQ quantize the weight
     b. Replace Linear -> BinaryQuadratic (Y,Z fixed; a,b,c,d learnable)
     c. Optimize ALL continuous params in block (BQQ a,b,c,d + remaining
        unquantized Linear weights + LayerNorm params) to minimize
        block output MSE vs pretrained output
  3. Save quantized block

Blocks are independent -> can be parallelized via --block_idx argument.

Usage:
  python block_wise_quant.py --model_name Qwen/Qwen2.5-1.5B --block_idx 0 \
    --dataset wikitext2 --nsamples 128 --seqlen 2048 \
    --bit_width 4 --group_size 128 --num_steps 50000 \
    --epochs 5 --lr 1e-4 --save_dir ./blockwise_output
"""

import argparse
import copy
import os
import sys
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quantizer import BinaryQuadraticQuantization

from build_bqq_model import BinaryQuadratic
from compressed_data import get_bqq_matrices
from datautils import get_loaders
from model_loader import load_causal_lm


# ---------------------------------------------------------------------------
# Block I/O caching
# ---------------------------------------------------------------------------

def _detach_to_cpu(v):
    """Recursively detach and move tensors to CPU (handles tuples/lists)."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    elif isinstance(v, tuple):
        return tuple(_detach_to_cpu(x) for x in v)
    elif isinstance(v, list):
        return [_detach_to_cpu(x) for x in v]
    return v


def _to_device_dtype(v, device, dtype):
    """Recursively move tensors to device/dtype (handles tuples/lists)."""
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
    elif isinstance(v, tuple):
        return tuple(_to_device_dtype(x, device, dtype) for x in v)
    elif isinstance(v, list):
        return [_to_device_dtype(x, device, dtype) for x in v]
    return v


@torch.no_grad()
def cache_block_io(model, block_idx, dataloader, device):
    """
    Forward pretrained model on calibration data.
    Cache hidden_states input and output for the target block.

    Returns:
        inputs_cache:  list of dicts, each with 'hidden_states' + kwargs
        targets_cache: list of tensors (block output hidden_states)
    """
    model.eval()
    model.to(device)

    block = model.model.layers[block_idx]
    inputs_cache = []
    targets_cache = []

    def capture_input(module, args, kwargs):
        cached = {'hidden_states': args[0].detach().cpu()}
        for k, v in kwargs.items():
            cached[k] = _detach_to_cpu(v)
        inputs_cache.append(cached)

    def capture_output(module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        targets_cache.append(out.detach().cpu())

    h_in = block.register_forward_pre_hook(capture_input, with_kwargs=True)
    h_out = block.register_forward_hook(capture_output, with_kwargs=True)

    for batch in tqdm(dataloader, desc=f'Caching block {block_idx} I/O'):
        ids = batch[0].to(device)
        try:
            model(ids)
        except Exception:
            pass

    h_in.remove()
    h_out.remove()

    return inputs_cache, targets_cache


# ---------------------------------------------------------------------------
# BQQ quantization helpers
# ---------------------------------------------------------------------------

def get_quantizable_linears(block):
    """Return names of all Linear layers in block (excluding norm layers)."""
    linears = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            linears.append(name)
    return linears


def quantize_weight_to_bqq(weight, *, bit_width, group_size, num_steps,
                            rank_scale, seed, device_id):
    """Quantize a 2D weight tensor with BQQ. Returns (A, Y, Z) for BinaryQuadratic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidated_path = os.path.join(tmpdir, 'temp.pth')
        quantizer = BinaryQuadraticQuantization(weight, rank_scale=rank_scale)
        quantizer.bqq_large_matrix_multi_worker(
            max_patch_size=group_size,
            bit_width=bit_width,
            consolidated_path=consolidated_path,
            Nstep=num_steps,
            seed=seed,
            main_gpu_id=device_id,
        )
        patches = torch.load(consolidated_path, weights_only=False, map_location='cpu')

    A, Y, Z = get_bqq_matrices(patches, bit_width)
    return A, Y, Z


def _get_submodule(module, dotted_name):
    """Traverse module by dotted name (e.g. 'self_attn.q_proj')."""
    for part in dotted_name.split('.'):
        module = getattr(module, part)
    return module


def _set_submodule(module, dotted_name, new_child):
    """Replace a submodule at dotted path."""
    parts = dotted_name.split('.')
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_child)


def replace_linear_in_block(block, linear_name, A, Y, Z, bias=None):
    """Replace a specific Linear in block with BinaryQuadratic module."""
    bqq_module = BinaryQuadratic(Y, Z, A, bias=bias)
    _set_submodule(block, linear_name, bqq_module)


# ---------------------------------------------------------------------------
# Block output optimization
# ---------------------------------------------------------------------------

def run_block_forward(block, inp, device):
    """Run block forward from cached input dict. Returns output hidden_states."""
    # Infer dtype from block parameters
    dtype = next(block.parameters()).dtype
    hidden_states = inp['hidden_states'].to(device=device, dtype=dtype)
    kwargs = {}
    for k, v in inp.items():
        if k == 'hidden_states':
            continue
        kwargs[k] = _to_device_dtype(v, device, dtype)
    # Prevent KV cache / SSM state from persisting across forward calls.
    kwargs['use_cache'] = False
    kwargs.pop('past_key_values', None)
    output = block(hidden_states, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def compute_block_mse(block, inputs_cache, targets_cache, device):
    """Compute mean block output MSE over all cached samples."""
    block.to(device).eval()
    total_mse = 0.0
    with torch.no_grad():
        for inp, target in zip(inputs_cache, targets_cache):
            output = run_block_forward(block, inp, device)
            total_mse += ((output - target.to(device)) ** 2).mean().item()
    return total_mse / len(inputs_cache)


def optimize_block_params(block, inputs_cache, targets_cache, *,
                          epochs, lr, device, max_grad_norm=1.0):
    """
    Optimize all trainable parameters in block to minimize
    ||block(cached_input) - pretrained_output||^2.

    Keeps the best parameter state (lowest epoch MSE) and restores it
    at the end, so gradient explosions at later epochs are harmless.

    Trainable params include:
      - BinaryQuadratic: a, b, c, d (scale factors), bias
      - Unquantized Linear: weight, bias
      - LayerNorm: weight, bias
    Binary buffers (Y, Z) are NOT parameters and thus excluded.
    """
    block.to(device)
    block.eval()  # keep eval mode (no dropout noise)

    params = [p for p in block.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f'    Optimizing {len(params)} param groups ({n_params:,} elements), '
          f'lr={lr}, epochs={epochs}, max_grad_norm={max_grad_norm}')

    optimizer = torch.optim.AdamW(params, lr=lr)

    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        total_loss = 0.0
        for inp, target in zip(inputs_cache, targets_cache):
            with torch.enable_grad():
                output = run_block_forward(block, inp, device)
                target_hs = target.to(device)
                loss = ((output - target_hs) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(inputs_cache)
        if avg < best_mse:
            best_mse = avg
            best_state = {k: v.cpu().clone() for k, v in block.state_dict().items()}
        print(f'    Epoch {epoch + 1}/{epochs}: MSE={avg:.6f}'
              f'{" *" if avg <= best_mse else ""}')

    # Restore best parameters
    if best_state is not None:
        block.load_state_dict(best_state)
        block.to(device)
        print(f'    Restored best epoch (MSE={best_mse:.6f})')


# ---------------------------------------------------------------------------
# Main block quantization routine
# ---------------------------------------------------------------------------

def quantize_block(
    model_name,
    block_idx,
    dataloader,
    *,
    bit_width,
    group_size,
    num_steps,
    rank_scale,
    seed,
    epochs,
    lr,
    max_grad_norm=1.0,
    device,
    save_dir,
):
    """
    Quantize all Linear weights in a single transformer block.

    Steps:
      1. Cache block I/O from pretrained model
      2. For each Linear (in order):
         a. BQQ quantize weight
         b. Replace Linear -> BinaryQuadratic
         c. Optimize all continuous params via block output MSE
      3. Save result
    """
    dev = torch.device(device)
    device_id = dev.index if dev.type == 'cuda' else 0

    # --- 1. Cache block I/O ---
    print(f'Loading model: {model_name}')
    model = load_causal_lm(model_name)

    print(f'Caching block {block_idx} I/O ...')
    inputs_cache, targets_cache = cache_block_io(model, block_idx, dataloader, dev)
    print(f'  Cached {len(inputs_cache)} samples')

    # Deep-copy target block and free the full model
    # Convert to float32 to avoid dtype mismatch when mixing
    # BinaryQuadratic (float32) with remaining bfloat16 Linears
    block = copy.deepcopy(model.model.layers[block_idx]).float()
    del model
    torch.cuda.empty_cache()

    # --- 2. Sequential quantize-optimize ---
    linear_names = get_quantizable_linears(block)
    print(f'\nBlock {block_idx} quantization targets ({len(linear_names)}):')
    for name in linear_names:
        lin = _get_submodule(block, name)
        print(f'  {name}: {tuple(lin.weight.shape)}')

    # Initial block MSE (before any quantization)
    init_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'\nInitial block MSE (pretrained): {init_mse:.6f}')

    for i, linear_name in enumerate(linear_names):
        linear = _get_submodule(block, linear_name)
        weight = linear.weight.data.clone().float()
        bias = linear.bias.data.clone().float() if linear.bias is not None else None

        print(f'\n--- [{i + 1}/{len(linear_names)}] {linear_name} '
              f'{tuple(weight.shape)} ---')

        # a. BQQ quantize
        A, Y, Z = quantize_weight_to_bqq(
            weight, bit_width=bit_width, group_size=group_size,
            num_steps=num_steps, rank_scale=rank_scale, seed=seed,
            device_id=device_id,
        )

        # b. Replace Linear -> BinaryQuadratic
        replace_linear_in_block(block, linear_name, A, Y, Z, bias=bias)

        # MSE after quantization (before optimization)
        mse_before = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f'  Block MSE after quant: {mse_before:.6f}')

        # c. Optimize continuous params
        optimize_block_params(
            block, inputs_cache, targets_cache,
            epochs=epochs, lr=lr, max_grad_norm=max_grad_norm, device=dev,
        )

        mse_after = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f'  Block MSE after optim: {mse_after:.6f} '
              f'(recovered {(mse_before - mse_after) / (mse_before - init_mse + 1e-12) * 100:.1f}%)')

    # --- 3. Save (モジュール丸ごと保存 — 最適化済み連続パラメータを含む) ---
    save_path = Path(save_dir) / f'block_{block_idx}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(block.cpu(), save_path)

    final_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'\n=== Block {block_idx} done ===')
    print(f'  Initial MSE: {init_mse:.6f}')
    print(f'  Final MSE:   {final_mse:.6f}')
    print(f'  Saved to: {save_path}')

    return block


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Block-wise BQQ quantization with output error optimization')

    # Model
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name (e.g. Qwen/Qwen2.5-1.5B)')
    parser.add_argument('--block_idx', type=int, required=True,
                        help='Transformer block index to quantize')

    # BQQ params
    parser.add_argument('--bit_width', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=128)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    # Dataset
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)

    # Optimization
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')

    # Device / output
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_loader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model=args.model_name,
        tokenizer=tokenizer,
    )

    quantize_block(
        model_name=args.model_name,
        block_idx=args.block_idx,
        dataloader=train_loader,
        bit_width=args.bit_width,
        group_size=args.group_size,
        num_steps=args.num_steps,
        rank_scale=args.rank_scale,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
