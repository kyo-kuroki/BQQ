"""
Layer-wise Hessian-aware BQQ quantization for language models.

1. Load pretrained model
2. Collect Hessian (H = X^T X) per linear layer via calibration data
3. Quantize each weight with intra-layer Hessian-aware BQQ
   (column-wise N-bit decomposition + GPTQ-style compensation + optional scale refine)
4. Save in same format as weight_aware_quant_cached.py (consolidated patch files)

Usage (all targets sequentially):
  python layerwise_quant.py \
    --model_name Qwen/Qwen3-2B \
    --save_dir bqq_compressed_data/Qwen3-2B-2bit-32gs-hessian \
    --bit_width 2 --group_size 32 --num_steps 20000 \
    --dataset c4 --nsamples 128 --seqlen 2048

Usage (single target, for parallel SGE array jobs):
  python layerwise_quant.py --model_name Qwen/Qwen3-2B --target_idx 0 ...
  python layerwise_quant.py --model_name Qwen/Qwen3-2B --list_targets  # print count and exit
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quantizer import BinaryQuadraticQuantization

try:
    from .model_loader import load_causal_lm
    from .datautils import get_loaders
    from .compressed_data import default_compressed_data_dir
except ImportError:
    from model_loader import load_causal_lm
    from datautils import get_loaders
    from compressed_data import default_compressed_data_dir


# ---------------------------------------------------------------------------
# Target selection (same as weight_aware_quant_cached)
# ---------------------------------------------------------------------------

def is_quantization_target(name: str) -> bool:
    return name.endswith('weight') and all(
        token not in name for token in ('norm', 'bias', 'emb'))


def passes_layer_threshold(name: str, layer_threshold: int) -> bool:
    layer_ids = [int(num) for num in re.findall(r'\d+', name)]
    return any(layer_id >= layer_threshold for layer_id in layer_ids)


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
            ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            ids = ids.to(device)
            try:
                model(ids)
            except Exception:
                pass

    for h in handles:
        h.remove()

    return {k: v for k, v in H.items() if v is not None}


# ---------------------------------------------------------------------------
# Main quantization
# ---------------------------------------------------------------------------

def get_target_names(model_name: str, layer_threshold: int = 0) -> List[str]:
    """Return ordered list of quantization target module names for the given model."""
    model = load_causal_lm(model_name)
    names = []
    for name, param in model.named_parameters():
        if is_quantization_target(name) and param.ndim == 2:
            if layer_threshold > 0 and not passes_layer_threshold(name, layer_threshold):
                continue
            names.append(name[:-len('.weight')])
    del model
    return names


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
    layer_threshold: int = 0,
    target_idx: Optional[int] = None,
):
    """
    Quantize target weights with intra-layer Hessian-aware BQQ.

    If target_idx is given, only that single weight (0-based index into the
    full target list) is quantized.  This enables parallel SGE array jobs.
    """
    device = torch.device(f'cuda:{main_gpu_id}' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    consolidated_dir = save_dir / '_consolidated'
    consolidated_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {model_name}")
    model = load_causal_lm(model_name)

    # Find quantization targets
    target_params = {}
    for name, param in model.named_parameters():
        if is_quantization_target(name) and param.ndim == 2:
            if layer_threshold > 0 and not passes_layer_threshold(name, layer_threshold):
                continue
            module_name = name[:-len('.weight')]
            target_params[module_name] = param.detach().cpu().float()

    all_target_names = list(target_params.keys())
    print(f"Found {len(all_target_names)} quantization targets")

    # When a specific index is requested, narrow to that single target
    if target_idx is not None:
        if target_idx >= len(all_target_names):
            raise ValueError(
                f"--target_idx {target_idx} is out of range "
                f"(model has {len(all_target_names)} targets)"
            )
        selected_name = all_target_names[target_idx]
        print(f"Single-target mode: [{target_idx+1}/{len(all_target_names)}] {selected_name}")
        work_items = {selected_name: target_params[selected_name]}
    else:
        work_items = target_params

    # Collect Hessians only for the targets we will actually quantize
    hessian_targets = list(work_items.keys())
    print(f"Collecting Hessians ({len(hessian_targets)} layer(s))...")
    H_dict = collect_hessians(model, calibration_loader, hessian_targets, device)
    print(f"  Collected {len(H_dict)} Hessians")

    # Free model from GPU
    model.cpu()
    del model
    torch.cuda.empty_cache()

    # Quantize
    n_total = len(all_target_names)
    for module_name, weight in work_items.items():
        display_idx = all_target_names.index(module_name) + 1
        param_name = f"{module_name}.weight"
        consolidated_path = consolidated_dir / f'{param_name}.pth'
        tensor_path = save_dir / f'{param_name}.pth'

        if tensor_path.exists():
            print(f"[{display_idx}/{n_total}] {param_name}: already exists, skipping")
            continue

        if module_name not in H_dict:
            print(f"[{display_idx}/{n_total}] {param_name}: no Hessian, skipping")
            continue

        H = H_dict[module_name].cpu()
        print(f"\n[{display_idx}/{n_total}] {param_name} {tuple(weight.shape)}")

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
    parser = argparse.ArgumentParser(
        description="Layer-wise Hessian-aware BQQ quantization for LMs")

    # Model
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--layer_threshold', type=int, default=0)

    # BQQ params
    parser.add_argument('--bit_width', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    # Hessian
    parser.add_argument('--scale_refine', action='store_true',
                        help='Apply inter-bit scale refinement after intra-layer compensation')
    parser.add_argument('--damping', type=float, default=1e-6)

    # Dataset
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'])
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)

    # Device / output
    parser.add_argument('--main_gpu_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None)

    # Parallel mode
    parser.add_argument('--target_idx', type=int, default=None,
                        help='0-based index of the single target to quantize. '
                             'Used for parallel SGE array jobs.')
    parser.add_argument('--list_targets', action='store_true',
                        help='Print the number of quantization targets and exit '
                             '(for use by submit scripts).')

    args = parser.parse_args()

    # --list_targets: just print count and exit (no GPU needed)
    if args.list_targets:
        names = get_target_names(args.model_name, args.layer_threshold)
        print(len(names))
        return

    # Data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_loader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=args.seqlen, model=args.model_name, tokenizer=tokenizer,
    )

    # Output dir
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(
            args.model_name, args.group_size, args.num_steps)
    save_dir = Path(save_dir)

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
        layer_threshold=args.layer_threshold,
        target_idx=args.target_idx,
    )


if __name__ == '__main__':
    main()
