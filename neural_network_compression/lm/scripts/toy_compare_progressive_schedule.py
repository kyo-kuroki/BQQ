"""
Toy experiment: linear vs geometric schedule for progressive BQQ.

Both schedules share the same:
  - block I/O cache (calibration data)
  - initial block weights
  - shuffled patch order (same RNG seed)

This isolates the effect of the batch size schedule only.

Usage (inside container, from lm/ directory):
  python scripts/toy_compare_progressive_schedule.py \
      --model_name Qwen/Qwen3.5-2B \
      --block_idx 0 \
      --bit_width 2 --group_size 64 \
      --num_steps 500 --num_rounds 8 \
      --epochs 3 --lr 1e-4 \
      --nsamples 64 --seqlen 512 \
      --device cuda:0
"""
import argparse
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from blockwise_quant import (
    _compute_batch_sizes,
    _get_submodule,
    _set_submodule,
    cache_block_io,
    compute_block_mse,
    get_quantizable_linears,
    optimize_block_params,
    quantize_weight_to_bqq,
)
from build_bqq_model import PartialBQQLinear
from datautils import get_loaders
from model_loader import load_causal_lm
from transformers import AutoTokenizer


def run_progressive_on_block(
    block,
    inputs_cache,
    targets_cache,
    shuffled_patches,
    *,
    bit_width,
    group_size,
    num_steps,
    rank_scale,
    seed,
    epochs,
    lr,
    max_grad_norm,
    num_rounds,
    schedule,
    device,
):
    """Run progressive quantization on a pre-copied block.

    Uses `shuffled_patches` (pre-shuffled list of (lname, i, j)) so that
    both schedules see patches in the same order — only the batch sizes differ.

    Returns final block MSE and per-round MSE history.
    """
    dev = torch.device(device)
    device_id = dev.index if dev.type == 'cuda' else 0

    # Convert all Linear → PartialBQQLinear
    linear_names = get_quantizable_linears(block)
    for lname in linear_names:
        lin = _get_submodule(block, lname)
        partial = PartialBQQLinear(
            lin.weight.data,
            lin.bias.data if lin.bias is not None else None,
            group_size=group_size,
            bit_width=bit_width,
        )
        _set_submodule(block, lname, partial)

    total_patches = len(shuffled_patches)
    batch_sizes = _compute_batch_sizes(total_patches, num_rounds, schedule)
    print(f'\n  Schedule={schedule!r}, num_rounds={num_rounds}')
    print(f'  Batch sizes: {batch_sizes}  (total={sum(batch_sizes)})')

    history = []
    offset = 0

    for round_idx, batch_size in enumerate(batch_sizes, start=1):
        batch = shuffled_patches[offset:offset + batch_size]
        offset += batch_size
        print(f'\n  Round {round_idx}/{num_rounds}: {len(batch)} patches '
              f'({offset}/{total_patches} quantized)')

        # Group by layer → one quantize_weight_to_bqq call per layer (uses multiprocessing)
        patches_by_layer = {}
        for lname, i, j in batch:
            patches_by_layer.setdefault(lname, []).append((i, j))

        for lname, ij_list in patches_by_layer.items():
            layer = _get_submodule(block, lname)
            print(f'    [{lname}] quantize full layer → activate {len(ij_list)} patches')
            A_all, Y_all, Z_all = quantize_weight_to_bqq(
                layer.float_weight.data.clone(),
                bit_width=bit_width,
                group_size=group_size,
                num_steps=num_steps,
                rank_scale=rank_scale,
                seed=seed,
                device_id=device_id,
            )
            for i, j in ij_list:
                layer.quantize_patch(i, j, A_all[:, i, j, :], Y_all[:, i, j], Z_all[:, i, j])

        mse_pre = compute_block_mse(block, inputs_cache, targets_cache, dev)
        optimize_block_params(
            block, inputs_cache, targets_cache,
            epochs=epochs, lr=lr, max_grad_norm=max_grad_norm, device=dev,
        )
        mse_post = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f'  MSE: {mse_pre:.6f} → {mse_post:.6f}  (Δ={mse_post - mse_pre:+.6f})')
        history.append({'round': round_idx, 'mse_pre': mse_pre, 'mse_post': mse_post})

    # Convert PartialBQQLinear → BinaryQuadratic
    for lname in linear_names:
        layer = _get_submodule(block, lname)
        _set_submodule(block, lname, layer.to_binaryquadratic())

    final_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    return final_mse, history


def main():
    parser = argparse.ArgumentParser(
        description='Compare linear vs geometric progressive BQQ schedule on one block')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3.5-2B')
    parser.add_argument('--block_idx', type=int, default=0)
    parser.add_argument('--bit_width', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=500,
                        help='BQQ optimisation steps per patch (reduced for toy experiment)')
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rounds', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--nsamples', type=int, default=64)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    dev = torch.device(args.device)

    # --- Load model and cache block I/O once ---
    print(f'Loading model: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model=args.model_name,
        tokenizer=tokenizer,
    )

    model = load_causal_lm(args.model_name)
    print(f'Caching block {args.block_idx} I/O ...')
    inputs_cache, targets_cache = cache_block_io(model, args.block_idx, train_loader, dev)
    print(f'  Cached {len(inputs_cache)} samples')

    block_orig = copy.deepcopy(model.model.layers[args.block_idx]).float()
    del model
    torch.cuda.empty_cache()

    # --- Compute initial MSE ---
    init_mse = compute_block_mse(block_orig, inputs_cache, targets_cache, dev)
    print(f'\nInitial block MSE (pretrained): {init_mse:.6f}')

    # --- Build shuffled patch list (shared by both schedules) ---
    linear_names = get_quantizable_linears(block_orig)

    def _n_patches(dim, gs):
        return 1 if dim <= gs else dim // gs

    all_patches = []
    for lname in linear_names:
        lin = _get_submodule(block_orig, lname)
        rw = _n_patches(lin.weight.shape[0], args.group_size)
        cw = _n_patches(lin.weight.shape[1], args.group_size)
        for i in range(rw):
            for j in range(cw):
                all_patches.append((lname, i, j))

    rng = torch.Generator()
    rng.manual_seed(args.seed)
    perm = torch.randperm(len(all_patches), generator=rng).tolist()
    shuffled_patches = [all_patches[k] for k in perm]
    print(f'Total patches: {len(shuffled_patches)}')

    # --- Run both schedules ---
    results = {}
    for schedule in ['linear', 'geometric']:
        print(f'\n{"=" * 60}')
        print(f'Running schedule: {schedule.upper()}')
        print('=' * 60)
        block_copy = copy.deepcopy(block_orig)
        final_mse, history = run_progressive_on_block(
            block_copy,
            inputs_cache,
            targets_cache,
            shuffled_patches,
            bit_width=args.bit_width,
            group_size=args.group_size,
            num_steps=args.num_steps,
            rank_scale=args.rank_scale,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            max_grad_norm=args.max_grad_norm,
            num_rounds=args.num_rounds,
            schedule=schedule,
            device=args.device,
        )
        results[schedule] = {'final_mse': final_mse, 'history': history}
        del block_copy
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print('=' * 60)
    print(f'Initial MSE (pretrained): {init_mse:.6f}')
    print()
    for schedule in ['linear', 'geometric']:
        r = results[schedule]
        print(f'{schedule.upper():10s}  final MSE: {r["final_mse"]:.6f}  '
              f'(vs pretrained: {r["final_mse"] / init_mse:.3f}x)')
        for h in r['history']:
            print(f'  Round {h["round"]:2d}: {h["mse_pre"]:.6f} → {h["mse_post"]:.6f}')
        print()


if __name__ == '__main__':
    main()
