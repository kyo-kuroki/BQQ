"""
Layer-wise Hessian-aware BQQ quantization for vision models.

1. Load pretrained model (DeiT, ViT, Swin)
2. Collect Hessian (H = X^T X) per linear layer via ImageNet calibration data
3. Quantize each weight with intra-layer Hessian-aware BQQ
4. Save in same format as weight_aware_quant_cached.py (consolidated patch files)

Usage:
  python layerwise_quant.py \
    --model_name deit-s \
    --save_dir bqq_compressed_data/deit-s-2bit-32gs-hessian \
    --bit_width 2 --group_size 32 --num_steps 20000 \
    --nsamples 256 --data_path /path/to/imagenet \
    --scale_refine
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quantizer import BinaryQuadraticQuantization

from build_model import get_model
from build_dataset import get_imagenet


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------

def is_quantization_target(name: str) -> bool:
    return not any(token in name for token in ('norm', 'bias', 'token', 'pos', 'emb', 'head'))


# ---------------------------------------------------------------------------
# Hessian collection
# ---------------------------------------------------------------------------

def collect_hessians(
    model: nn.Module,
    calibration_loader,
    target_names: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run calibration data through model and accumulate H = X^T X per layer."""
    H: Dict[str, Optional[torch.Tensor]] = {n: None for n in target_names}
    handles = []

    def make_hook(name: str):
        def _hook(module, inp, _out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            h = x.T @ x
            if H[name] is None:
                H[name] = h
            else:
                H[name].add_(h)
        return _hook

    for name, module in model.named_modules():
        if name in target_names and isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(calibration_loader, desc="Collecting Hessians"):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            try:
                model(images)
            except Exception:
                pass

    for h in handles:
        h.remove()

    return {k: v for k, v in H.items() if v is not None}


# ---------------------------------------------------------------------------
# Main quantization
# ---------------------------------------------------------------------------

def layerwise_quantize(
    model_name: str,
    save_dir: Path,
    *,
    bit_width: int,
    group_size: int,
    num_steps: int,
    rank_scale: float,
    seed: int,
    main_gpu_id: int,
    scale_refine: bool,
    damping: float,
    calibration_loader,
):
    device = torch.device(f'cuda:{main_gpu_id}' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    consolidated_dir = save_dir / '_consolidated'
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    model = get_model(model_name)

    # Find quantization targets
    target_params = {}
    for name, param in model.named_parameters():
        if is_quantization_target(name) and param.ndim == 2:
            module_name = name[:-len('.weight')] if name.endswith('.weight') else name
            target_params[module_name] = param.detach().cpu().float()

    target_names = list(target_params.keys())
    print(f"Found {len(target_names)} quantization targets")

    # Collect Hessians
    print(f"Collecting Hessians ({len(target_names)} layers)...")
    H_dict = collect_hessians(model, calibration_loader, target_names, device)
    print(f"  Collected {len(H_dict)} Hessians")

    model.cpu()
    del model
    torch.cuda.empty_cache()

    # Quantize each target
    for idx, (module_name, weight) in enumerate(target_params.items()):
        param_name = f"{module_name}.weight"
        consolidated_path = consolidated_dir / f'{param_name}.pth'
        tensor_path = save_dir / f'{param_name}.pth'

        if tensor_path.exists():
            print(f"[{idx+1}/{len(target_params)}] {param_name}: already exists, skipping")
            continue

        if module_name not in H_dict:
            print(f"[{idx+1}/{len(target_params)}] {param_name}: no Hessian, skipping")
            continue

        H = H_dict[module_name].cpu()
        print(f"\n[{idx+1}/{len(target_params)}] {param_name} {tuple(weight.shape)}")

        quantizer = BinaryQuadraticQuantization(weight, rank_scale=rank_scale)
        reconstructed = quantizer.bqq_large_matrix_multi_worker(
            max_patch_size=group_size,
            bit_width=bit_width,
            consolidated_path=str(consolidated_path),
            Nstep=num_steps,
            seed=seed,
            main_gpu_id=main_gpu_id,
            H=H,
            hessian_mode='intra-layer',
            scale_refine=scale_refine,
            damping=damping,
        )

        torch.save(reconstructed.cpu(), tensor_path)
        print(f"  Saved: {tensor_path}")

    print(f"\nDone. Output: {save_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Layer-wise Hessian-aware BQQ quantization for vision models")

    parser.add_argument('--model_name', type=str, required=True,
                        choices=['deit-s', 'deit-b', 'vit-s', 'vit-b', 'swin-t', 'swin-s'])

    parser.add_argument('--bit_width', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--scale_refine', action='store_true')
    parser.add_argument('--damping', type=float, default=1e-6)

    parser.add_argument('--nsamples', type=int, default=256)
    parser.add_argument('--data_path', type=str, default=None)

    parser.add_argument('--main_gpu_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None)

    args = parser.parse_args()

    train_loader, _ = get_imagenet(
        args.model_name, num_traindatas=args.nsamples,
        data_path=args.data_path, seed=args.seed,
    )

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = SCRIPT_DIR / 'bqq_compressed_data' / \
            f'{args.model_name}-{args.num_steps}step-{args.group_size}gs-hessian'

    layerwise_quantize(
        model_name=args.model_name,
        save_dir=save_dir,
        bit_width=args.bit_width,
        group_size=args.group_size,
        num_steps=args.num_steps,
        rank_scale=args.rank_scale,
        seed=args.seed,
        main_gpu_id=args.main_gpu_id,
        scale_refine=args.scale_refine,
        damping=args.damping,
        calibration_loader=train_loader,
    )


if __name__ == '__main__':
    main()
