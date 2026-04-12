"""
Hessian-based scale factor refinement for BQQ vision models.

Binary parameters (Y, Z) are fixed; only continuous scale factors (a, b, c, d)
are re-optimized per patch using closed-form ridge regression with
input correlation matrices H = X^T X.

Usage:
  python scale_refine_bqq.py \
    --model_name deit-s \
    --bqq_model quantized_model.pth \
    --output refined_model.pth \
    --data_path /path/to/imagenet \
    --nsamples 256
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from bqq_modules import BinaryQuadratic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
from build_model import get_model
from build_dataset import get_imagenet


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
        for batch in tqdm(calibration_loader, desc="Collecting activations"):
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
# Per-patch scale refinement (same algorithm as LM version)
# ---------------------------------------------------------------------------

def _cholesky_safe(C: torch.Tensor, damping: float = 1e-6) -> Optional[torch.Tensor]:
    mean_diag = C.diagonal().mean().item()
    for scale in [damping, 1e-4, 1e-2, 1e-1]:
        try:
            L = torch.linalg.cholesky(C + scale * mean_diag * torch.eye(C.shape[0], device=C.device, dtype=C.dtype))
            return L
        except torch.linalg.LinAlgError:
            continue
    return None


def refine_patch(bqq_layer, i, j, W_block, C_j, damping):
    device = bqq_layer.Y.device
    dtype = torch.float32
    bit_width = bqq_layer.bit_width
    patch_h, patch_w = bqq_layer.y_row, bqq_layer.z_col

    C_j = C_j.to(device=device, dtype=dtype)
    W_block = W_block.to(device=device, dtype=dtype)

    L = _cholesky_safe(C_j, damping)
    if L is None:
        return False
    S_j = L

    R_S = W_block @ S_j
    col_sum_S = S_j.sum(dim=0, keepdim=True)
    ones_col = torch.ones(patch_h, 1, device=device, dtype=dtype)

    G_cols: List[torch.Tensor] = []
    for p in range(bit_width):
        Y_p = bqq_layer.Y[p, i, j].to(dtype=dtype)
        Z_p = bqq_layer.Z[p, i, j].to(dtype=dtype)
        Y_sum_p = Y_p.sum(dim=-1, keepdim=True)
        Z_sum_p = Z_p.sum(dim=-2, keepdim=True)

        G_a = (Y_p @ Z_p) @ S_j
        G_b = Y_sum_p @ col_sum_S
        G_c = ones_col @ (Z_sum_p @ S_j)
        G_cols.extend([G_a.reshape(-1), G_b.reshape(-1), G_c.reshape(-1)])

    G_d = ones_col @ col_sum_S
    G_cols.append(G_d.reshape(-1))

    Phi = torch.stack(G_cols, dim=1)
    rhs = R_S.reshape(-1, 1)

    PtP = Phi.T @ Phi
    Ptr = Phi.T @ rhs
    mean_diag = PtP.diagonal().mean().item()

    theta = None
    for reg_scale in [1e-6, 1e-4, 1e-2, 1e-1]:
        try:
            lam = reg_scale * mean_diag
            A = PtP + lam * torch.eye(PtP.shape[0], device=device, dtype=dtype)
            sol = torch.linalg.solve(A, Ptr)
            t = sol.squeeze(1)
            if t.isfinite().all():
                theta = t
                break
        except Exception:
            continue

    if theta is None:
        return False

    with torch.no_grad():
        for p in range(bit_width):
            bqq_layer.a.data[p, i, j, 0, 0] = theta[3 * p]
            bqq_layer.b.data[p, i, j, 0, 0] = theta[3 * p + 1]
            bqq_layer.c.data[p, i, j, 0, 0] = theta[3 * p + 2]
        bqq_layer.d.data[i, j, 0, 0] = theta[-1]

    return True


def refine_bqq_layer(bqq_layer, W_original, H_full, damping=1e-6):
    device = bqq_layer.Y.device
    row_width, col_width = bqq_layer.row_width, bqq_layer.col_width
    patch_h, patch_w = bqq_layer.y_row, bqq_layer.z_col

    W = W_original.to(device=device, dtype=torch.float32)
    H = H_full.to(device=device, dtype=torch.float32)

    n_ok = 0
    for i in range(row_width):
        for j in range(col_width):
            W_block = W[i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
            C_j = H[j * patch_w:(j + 1) * patch_w, j * patch_w:(j + 1) * patch_w]
            if refine_patch(bqq_layer, i, j, W_block, C_j, damping):
                n_ok += 1

    n_total = row_width * col_width
    if n_ok < n_total:
        print(f"    Warning: {n_total - n_ok}/{n_total} patches skipped")


# ---------------------------------------------------------------------------
# Main refinement pipeline
# ---------------------------------------------------------------------------

def find_bqq_layers(model, prefix=""):
    result = {}
    for name, module in model.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(module, BinaryQuadratic):
            result[full] = module
        else:
            result.update(find_bqq_layers(module, full))
    return result


def run_refinement(model_name, bqq_model, calibration_loader, device, damping=1e-6):
    bqq_layers = find_bqq_layers(bqq_model)
    print(f"Found {len(bqq_layers)} BinaryQuadratic layers to refine.")
    target_names = list(bqq_layers.keys())

    # Collect Hessians from pretrained model
    print(f"Loading pretrained model: {model_name}")
    pretrained = get_model(model_name)
    pretrained.eval()

    print("Collecting input correlations H = X^T X ...")
    H_dict = collect_hessians(pretrained, calibration_loader, target_names, device)

    # Extract original weights
    orig_weights = {}
    for name, param in pretrained.named_parameters():
        if name.endswith(".weight"):
            module_name = name[:-len(".weight")]
            if module_name in target_names:
                orig_weights[module_name] = param.detach().cpu().float()
    del pretrained
    torch.cuda.empty_cache()

    # Refine
    bqq_model.to(device)
    for layer_name, bqq_layer in tqdm(bqq_layers.items(), desc="Refining layers"):
        if layer_name not in H_dict or layer_name not in orig_weights:
            print(f"  Skipping {layer_name}")
            continue
        refine_bqq_layer(bqq_layer, orig_weights[layer_name], H_dict[layer_name], damping=damping)

    return bqq_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Refine BQQ scale factors for vision models")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=['deit-s', 'deit-b', 'vit-s', 'vit-b', 'swin-t', 'swin-s'])
    parser.add_argument("--bqq_model", type=Path, required=True,
                        help="Path to saved BQQ model (.pth)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--damping", type=float, default=1e-6)

    args = parser.parse_args()

    # Load BQQ model
    print(f"Loading BQQ model from {args.bqq_model}")
    bqq_model = torch.load(args.bqq_model, map_location="cpu", weights_only=False)

    # Calibration data
    train_loader, _ = get_imagenet(
        args.model_name, num_traindatas=args.nsamples,
        data_path=args.data_path, seed=args.seed,
    )

    device = torch.device(args.device)
    bqq_model = run_refinement(args.model_name, bqq_model, train_loader, device, args.damping)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bqq_model.cpu(), args.output)
    print(f"Saved refined model to {args.output}")


if __name__ == "__main__":
    main()
