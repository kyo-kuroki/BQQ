"""
Block-wise BQQ quantization for vision transformers (DeiT, ViT, Swin).

Same approach as lm/block_wise_quant.py but adapted for vision models:
  - Blocks accessed via model.blocks[i] (timm convention)
  - Calibration data from ImageNet
  - Input is image tensors, not token IDs

Usage:
  python block_wise_quant.py --model_name deit-s --block_idx 0 \
    --bit_width 2 --group_size 32 --num_steps 20000 \
    --nsamples 256 --epochs 10 --lr 1e-5 \
    --data_path /path/to/imagenet --save_dir ./blockwise_output/deit-s
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from bqq_modules import BinaryQuadratic, get_matrices

from build_model import get_model
from build_dataset import get_imagenet, calibsample_from_trainloader


# ---------------------------------------------------------------------------
# Block I/O caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_block_io(model, block_idx, calib_images, device):
    """
    Forward pretrained model on calibration images, cache input/output
    hidden states for the target block.

    For ViT/DeiT: the model forward is:
      x = patch_embed(images) + pos_embed + cls_token
      for block in blocks:
          x = block(x)
      x = norm(x)
      x = head(x)

    We hook on model.blocks[block_idx] to capture its input/output.
    """
    model.eval()
    model.to(device)

    block = model.blocks[block_idx]
    inputs_cache = []
    targets_cache = []

    def capture_input(module, args):
        inputs_cache.append(args[0].detach().cpu())

    def capture_output(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        targets_cache.append(out.detach().cpu())

    h_in = block.register_forward_pre_hook(capture_input)
    h_out = block.register_forward_hook(capture_output)

    # Process in batches to avoid OOM
    batch_size = 32
    for i in tqdm(range(0, len(calib_images), batch_size), desc=f'Caching block {block_idx} I/O'):
        batch = calib_images[i:i+batch_size].to(device)
        try:
            model(batch)
        except Exception:
            pass

    h_in.remove()
    h_out.remove()

    return inputs_cache, targets_cache


# ---------------------------------------------------------------------------
# BQQ quantization helpers
# ---------------------------------------------------------------------------

def get_quantizable_linears(block):
    """Return names of all Linear layers in block."""
    linears = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            linears.append(name)
    return linears


def quantize_weight_to_bqq(weight, *, bit_width, group_size, num_steps,
                            rank_scale, seed, device_id, H=None,
                            scale_refine=True, damping=1e-6):
    """Quantize a 2D weight tensor with BQQ. Returns (A, Y, Z)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidated_path = os.path.join(tmpdir, 'temp.pth')
        quantizer = BinaryQuadraticQuantization(weight, rank_scale=rank_scale)
        kwargs = dict(
            max_patch_size=group_size, bit_width=bit_width,
            consolidated_path=consolidated_path, Nstep=num_steps,
            seed=seed, main_gpu_id=device_id,
        )
        if H is not None:
            kwargs.update(H=H, hessian_mode='intra-layer',
                          scale_refine=scale_refine, damping=damping)
        quantizer.bqq_large_matrix_multi_worker(**kwargs)
        patches = torch.load(consolidated_path, weights_only=False, map_location='cpu')
    A, Y, Z = get_matrices(patches, bit_width)
    return A, Y, Z


def _get_submodule(module, dotted_name):
    for part in dotted_name.split('.'):
        module = getattr(module, part)
    return module


def _set_submodule(module, dotted_name, new_child):
    parts = dotted_name.split('.')
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_child)


# ---------------------------------------------------------------------------
# Block output optimization
# ---------------------------------------------------------------------------

def compute_block_mse(block, inputs_cache, targets_cache, device):
    block.to(device).eval()
    total_mse = 0.0
    with torch.no_grad():
        for inp, target in zip(inputs_cache, targets_cache):
            output = block(inp.to(device))
            if isinstance(output, tuple):
                output = output[0]
            total_mse += ((output - target.to(device)) ** 2).mean().item()
    return total_mse / len(inputs_cache)


def optimize_block_params(block, inputs_cache, targets_cache, *,
                          epochs, lr, device, max_grad_norm=1.0):
    """Optimize all trainable params to minimize block output MSE.

    Keeps the best parameter state (lowest epoch MSE) and restores it
    at the end, so gradient explosions at later epochs are harmless.
    """
    block.to(device).eval()

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
                output = block(inp.to(device))
                if isinstance(output, tuple):
                    output = output[0]
                loss = ((output - target.to(device)) ** 2).mean()
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

def collect_block_hessians(block, inputs_cache, device):
    """Collect H = X^T X for each Linear layer in block."""
    H_dict = {}
    handles = []
    linear_names = get_quantizable_linears(block)

    def make_hook(name):
        def _hook(module, inp, _out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            h = x.T @ x
            if name not in H_dict:
                H_dict[name] = h
            else:
                H_dict[name].add_(h)
        return _hook

    for name, module in block.named_modules():
        if name in linear_names and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))

    block.to(device).eval()
    with torch.no_grad():
        for inp in inputs_cache:
            try:
                output = block(inp.to(device))
            except Exception:
                pass

    for h in handles:
        h.remove()
    return {k: v.cpu() for k, v in H_dict.items()}


def quantize_block(
    model_name,
    block_idx,
    calib_images,
    *,
    bit_width,
    group_size,
    num_steps,
    rank_scale,
    seed,
    epochs,
    lr,
    max_grad_norm=1.0,
    use_hessian=True,
    hessian_cache_dir=None,
    scale_refine=True,
    damping=1e-6,
    device,
    save_dir,
):
    dev = torch.device(device)
    device_id = dev.index if dev.type == 'cuda' else 0

    # 1. Cache block I/O
    print(f'Loading model: {model_name}')
    model = get_model(model_name)

    print(f'Caching block {block_idx} I/O ...')
    inputs_cache, targets_cache = cache_block_io(model, block_idx, calib_images, dev)
    print(f'  Cached {len(inputs_cache)} batches')

    block = copy.deepcopy(model.blocks[block_idx])
    del model
    torch.cuda.empty_cache()

    # 2. Sequential quantize-optimize
    linear_names = get_quantizable_linears(block)
    print(f'\nBlock {block_idx} quantization targets ({len(linear_names)}):')
    for name in linear_names:
        lin = _get_submodule(block, name)
        print(f'  {name}: {tuple(lin.weight.shape)}')

    init_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'\nInitial block MSE: {init_mse:.6f}')

    # Collect or load Hessians
    H_dict = {}
    if use_hessian:
        cache_path = None
        if hessian_cache_dir is not None:
            cache_path = Path(hessian_cache_dir) / f'hessian_block_{block_idx}.pth'
            if cache_path.exists():
                print(f'Loading Hessian cache from {cache_path}')
                H_dict = torch.load(cache_path, map_location='cpu', weights_only=False)

        if not H_dict:
            print('Collecting Hessians for block Linears...')
            H_dict = collect_block_hessians(block, inputs_cache, dev)
            print(f'  Collected {len(H_dict)} Hessians')

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(H_dict, cache_path)
                print(f'  Saved Hessian cache to {cache_path}')

    for i, linear_name in enumerate(linear_names):
        linear = _get_submodule(block, linear_name)
        weight = linear.weight.data.clone().float()
        bias = linear.bias.data.clone().float() if linear.bias is not None else None

        print(f'\n--- [{i + 1}/{len(linear_names)}] {linear_name} '
              f'{tuple(weight.shape)} ---')

        H = H_dict.get(linear_name) if use_hessian else None
        A, Y, Z = quantize_weight_to_bqq(
            weight, bit_width=bit_width, group_size=group_size,
            num_steps=num_steps, rank_scale=rank_scale, seed=seed,
            device_id=device_id, H=H, scale_refine=scale_refine,
            damping=damping,
        )
        _set_submodule(block, linear_name, BinaryQuadratic(Y, Z, A, bias=bias))

        mse_before = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f'  Block MSE after quant: {mse_before:.6f}')

        optimize_block_params(
            block, inputs_cache, targets_cache,
            epochs=epochs, lr=lr, max_grad_norm=max_grad_norm, device=dev,
        )

        mse_after = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f'  Block MSE after optim: {mse_after:.6f} '
              f'(recovered {(mse_before - mse_after) / (mse_before - init_mse + 1e-12) * 100:.1f}%)')

    # 3. Save
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
        description='Block-wise BQQ quantization for vision transformers')

    parser.add_argument('--model_name', type=str, required=True,
                        choices=['deit-s', 'deit-b', 'vit-s', 'vit-b', 'swin-t', 'swin-s'])
    parser.add_argument('--block_idx', type=int, required=True)

    parser.add_argument('--bit_width', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--nsamples', type=int, default=256,
                        help='Number of calibration images')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--no_hessian', action='store_true',
                        help='Disable Hessian-aware quantization (use standard BQQ)')
    parser.add_argument('--hessian_cache_dir', type=str, default=None,
                        help='Directory to cache/load Hessian matrices')
    parser.add_argument('--no_scale_refine', action='store_true',
                        help='Disable Hessian-aware scale refinement')
    parser.add_argument('--damping', type=float, default=1e-6)

    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to ImageNet. Falls back to IMAGENET_DIR env var.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    train_loader, _ = get_imagenet(
        args.model_name, num_traindatas=args.nsamples,
        data_path=args.data_path, seed=args.seed,
    )
    calib_images, _ = calibsample_from_trainloader(train_loader, args.nsamples, seed=args.seed)

    quantize_block(
        model_name=args.model_name,
        block_idx=args.block_idx,
        calib_images=calib_images,
        bit_width=args.bit_width,
        group_size=args.group_size,
        num_steps=args.num_steps,
        rank_scale=args.rank_scale,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        use_hessian=not args.no_hessian,
        hessian_cache_dir=args.hessian_cache_dir,
        scale_refine=not args.no_scale_refine,
        damping=args.damping,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == '__main__':
    main()
