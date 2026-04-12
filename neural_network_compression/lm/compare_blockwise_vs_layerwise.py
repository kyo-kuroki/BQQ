"""Quick comparison: blockwise BQQ vs layerwise+refine BQQ (block output MSE)."""
import os
import sys
import copy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from model_loader import load_causal_lm
from datautils import get_loaders
from transformers import AutoTokenizer


def _to_dev(v, device, dtype):
    if isinstance(v, torch.Tensor):
        return v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
    if isinstance(v, tuple):
        return tuple(_to_dev(x, device, dtype) for x in v)
    if isinstance(v, list):
        return [_to_dev(x, device, dtype) for x in v]
    return v


def _detach_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, tuple):
        return tuple(_detach_cpu(x) for x in v)
    if isinstance(v, list):
        return [_detach_cpu(x) for x in v]
    return v


def compute_block_mse(blk, inputs, targets, dev):
    blk.to(dev).eval()
    dtype = next(blk.parameters()).dtype
    total = 0.0
    with torch.no_grad():
        for inp, tgt in zip(inputs, targets):
            hs = inp['hidden_states'].to(device=dev, dtype=dtype)
            kw = {}
            for k, v in inp.items():
                if k == 'hidden_states':
                    continue
                kw[k] = _to_dev(v, dev, dtype)
            kw['use_cache'] = False
            kw.pop('past_key_values', None)
            out = blk(hs, **kw)
            out = out[0] if isinstance(out, tuple) else out
            total += ((out.float() - tgt.to(dev).float()) ** 2).mean().item()
    return total / len(inputs)


def main():
    model_name = 'Qwen/Qwen3.5-2B'
    block_idx = 3  # full_attention block
    device = torch.device('cuda:0')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Paths relative to nn_compression/lm/ (where refined models live)
    nn_comp_lm = os.path.join(base_dir, '..', '..', 'nn_compression', 'lm')

    refined_2bit = os.path.join(nn_comp_lm, 'quantized_model_data',
                                'Qwen3.5-2B-2bit-32gs-10000step-refined-c4.pth')
    unrefined_2bit = os.path.join(nn_comp_lm, 'quantized_model_data',
                                  'Qwen3.5-2B-2bit-32gs-10000step.pth')
    blockwise_2bit = os.path.join(base_dir, 'blockwise_output',
                                  'Qwen3.5-2B-2bit-gs32-20000step-c4',
                                  f'block_{block_idx}.pth')

    # 1. Load pretrained model and cache block IO
    print('Loading pretrained model...')
    model = load_causal_lm(model_name)
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, _ = get_loaders(
        'c4', nsamples=16, seed=0, seqlen=2048,
        model=model_name, tokenizer=tokenizer,
    )

    block = model.model.layers[block_idx]
    inputs_cache = []
    targets_cache = []

    def capture_in(module, args, kwargs):
        cached = {'hidden_states': args[0].detach().cpu()}
        for k, v in kwargs.items():
            cached[k] = _detach_cpu(v)
        inputs_cache.append(cached)

    def capture_out(module, args, kwargs, output):
        out = output[0] if isinstance(output, tuple) else output
        targets_cache.append(out.detach().cpu())

    h1 = block.register_forward_pre_hook(capture_in, with_kwargs=True)
    h2 = block.register_forward_hook(capture_out, with_kwargs=True)
    with torch.no_grad():
        for batch in train_loader:
            try:
                model(batch[0].to(device))
            except Exception:
                pass
    h1.remove()
    h2.remove()
    print(f'Cached {len(inputs_cache)} samples')

    del model
    torch.cuda.empty_cache()

    results = {}

    # 2. Layerwise BQQ (no refine) — load full model, extract block
    if os.path.exists(unrefined_2bit):
        print(f'\nLoading layerwise model: {unrefined_2bit}')
        lw_model = torch.load(unrefined_2bit, map_location='cpu', weights_only=False)
        lw_block = lw_model.model.layers[block_idx]
        del lw_model
        mse = compute_block_mse(lw_block, inputs_cache, targets_cache, device)
        results['Layerwise (10k steps, no refine)'] = mse
        print(f'  Block {block_idx} MSE: {mse:.8f}')
        del lw_block
        torch.cuda.empty_cache()
    else:
        print(f'Not found: {unrefined_2bit}')

    # 3. Layerwise BQQ + refine — load full model, extract block
    if os.path.exists(refined_2bit):
        print(f'\nLoading refined model: {refined_2bit}')
        ref_model = torch.load(refined_2bit, map_location='cpu', weights_only=False)
        ref_block = ref_model.model.layers[block_idx]
        del ref_model
        mse = compute_block_mse(ref_block, inputs_cache, targets_cache, device)
        results['Layerwise + refine (10k steps)'] = mse
        print(f'  Block {block_idx} MSE: {mse:.8f}')
        del ref_block
        torch.cuda.empty_cache()
    else:
        print(f'Not found: {refined_2bit}')

    # 4. Blockwise BQQ
    if os.path.exists(blockwise_2bit):
        print(f'\nLoading blockwise block: {blockwise_2bit}')
        bw_block = torch.load(blockwise_2bit, map_location='cpu', weights_only=False)
        mse = compute_block_mse(bw_block, inputs_cache, targets_cache, device)
        results['Blockwise (20k steps, c4)'] = mse
        print(f'  Block {block_idx} MSE: {mse:.8f}')
        del bw_block
        torch.cuda.empty_cache()
    else:
        print(f'Not found: {blockwise_2bit}')

    # Summary
    print('\n' + '=' * 60)
    print(f'Block {block_idx} (full_attention) — 2-bit comparison')
    print('=' * 60)
    for name, mse in results.items():
        print(f'  {name:40s}: {mse:.8f}')
    if 'Layerwise + refine (10k steps)' in results and 'Blockwise (20k steps, c4)' in results:
        ref = results['Layerwise + refine (10k steps)']
        bw = results['Blockwise (20k steps, c4)']
        pct = (1 - bw / ref) * 100
        print(f'\n  Blockwise vs refined layerwise: {pct:+.1f}%'
              f' ({"better" if pct > 0 else "worse"})')


if __name__ == '__main__':
    main()
