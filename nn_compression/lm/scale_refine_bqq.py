"""scale_refine_bqq.py

Weight-aware scale refinement for BQQ quantized language models.

For each BinaryQuadratic layer (binary parameters Y, Z fixed), refines the
scale factors (a, b, c, d) analytically by minimizing:

    ||WX - W'(a,b,c,d) X||^2

Using the equivalence:

    ||WX - W'X||^2 = ||WS - W'S||^2   where  S S^T = X^T X  (Cholesky)

Since W' is linear in (a, b, c, d), this is a quadratic objective and the
Newton step gives the exact global minimum (one-shot linear solve per patch).

Usage
-----
Prepare a BQQ model (e.g. via make_bqq_model_from_compressed_data.py), then:

    python scale_refine_bqq.py \\
        --model_name  Qwen/Qwen3.5-4B \\
        --bqq_model   /path/to/bqq_model.pth \\
        --output      /path/to/refined_model.pth \\
        --nsamples    128 --seqlen 2048 --device cuda:0

Or rebuild the BQQ model from compressed data:

    python scale_refine_bqq.py \\
        --model_name        Qwen/Qwen3.5-4B \\
        --compressed_data   /path/to/bqq_compressed_data/Qwen3.5-4B-32gs-10000step \\
        --bit_width 2 --group_size 32 --num_steps 10000 \\
        --output   /path/to/refined_model.pth \\
        --nsamples 128 --seqlen 2048 --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .binary_quadratic_network import BinaryQuadratic, replace_linear_with_bqq
    from .compressed_data import default_compressed_data_dir, model_basename
    from .datautils import get_loaders
except ImportError:
    from binary_quadratic_network import BinaryQuadratic, replace_linear_with_bqq
    from compressed_data import default_compressed_data_dir, model_basename
    from datautils import get_loaders


# ---------------------------------------------------------------------------
# Calibration data collection
# ---------------------------------------------------------------------------

def collect_hessians(
    model: nn.Module,
    calibration_loader,
    target_names: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run calibration data through *model* and accumulate H = X^T X per layer.

    Parameters
    ----------
    model:
        Pretrained model with nn.Linear layers.
    calibration_loader:
        Iterable of (input_ids, ...) batches.
    target_names:
        Names of linear sub-modules whose input correlation we want.
    device:
        Device to run inference on.

    Returns
    -------
    Dict mapping layer name -> H  (shape: in_features x in_features, float32).
    """
    H: Dict[str, Optional[torch.Tensor]] = {n: None for n in target_names}
    handles = []

    def make_hook(name: str):
        def _hook(module: nn.Module, inp, _out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                # (batch, seq, in_features) → (batch*seq, in_features)
                x = x.reshape(-1, x.shape[-1])
            h = x.T @ x  # (in_features, in_features)
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
# Per-patch scale refinement
# ---------------------------------------------------------------------------

def _cholesky_safe(C: torch.Tensor, damping: float = 1e-6) -> Optional[torch.Tensor]:
    """Return lower-triangular Cholesky factor of C with adaptive damping."""
    mean_diag = C.diagonal().mean().item()
    for scale in [damping, 1e-4, 1e-2, 1e-1]:
        try:
            L = torch.linalg.cholesky(C + scale * mean_diag * torch.eye(C.shape[0], device=C.device, dtype=C.dtype))
            return L
        except torch.linalg.LinAlgError:
            continue
    return None


def refine_patch(
    bqq_layer: BinaryQuadratic,
    i: int,
    j: int,
    W_block: torch.Tensor,   # (patch_h, patch_w) original weight slice
    C_j: torch.Tensor,        # (patch_w, patch_w) input correlation X_j^T X_j
    damping: float,
) -> bool:
    """Refine scale factors for a single patch (i, j) of a BinaryQuadratic layer.

    Solves min_{a,b,c,d} ||W_block S - W'_block(a,b,c,d) S||^2
    where S is the Cholesky factor of C_j = X_j^T X_j.

    Returns True on success, False if numerical issues prevented solving.
    """
    device = bqq_layer.Y.device
    dtype = torch.float32

    bit_width = bqq_layer.bit_width
    patch_h, patch_w = bqq_layer.y_row, bqq_layer.z_col

    C_j = C_j.to(device=device, dtype=dtype)
    W_block = W_block.to(device=device, dtype=dtype)

    # --- Cholesky: C_j = L L^T, use L as S_j so that ||A X||^2 = ||A S_j||^2 ---
    L = _cholesky_safe(C_j, damping)
    if L is None:
        return False
    S_j = L  # (patch_w, patch_w)

    # Target projected onto S_j
    R_S = W_block @ S_j  # (patch_h, patch_w)

    # Precomputed shared quantities
    col_sum_S = S_j.sum(dim=0, keepdim=True)   # (1, patch_w): 1^T S_j
    ones_col = torch.ones(patch_h, 1, device=device, dtype=dtype)

    # Build design matrix columns: one column per free scalar parameter
    # Order: [a_0, b_0, c_0,  a_1, b_1, c_1,  ...,  d]
    G_cols: List[torch.Tensor] = []

    for p in range(bit_width):
        Y_p = bqq_layer.Y[p, i, j].to(dtype=dtype)   # (patch_h, l)
        Z_p = bqq_layer.Z[p, i, j].to(dtype=dtype)   # (l, patch_w)
        Y_sum_p = Y_p.sum(dim=-1, keepdim=True)       # (patch_h, 1)
        Z_sum_p = Z_p.sum(dim=-2, keepdim=True)       # (1, patch_w)

        # G_a[p] = (Y_p @ Z_p) @ S_j          (rank-l product projected)
        G_a = (Y_p @ Z_p) @ S_j               # (patch_h, patch_w)

        # G_b[p] = Y_sum_p @ (1^T S_j)        (row-sums of Y broadcast)
        G_b = Y_sum_p @ col_sum_S             # (patch_h, patch_w)

        # G_c[p] = 1_col @ (Z_sum_p @ S_j)    (col-sums of Z broadcast)
        G_c = ones_col @ (Z_sum_p @ S_j)     # (patch_h, patch_w)

        G_cols.extend([G_a.reshape(-1), G_b.reshape(-1), G_c.reshape(-1)])

    # G_d = 1_col @ (1^T S_j)                 (scalar offset term)
    G_d = ones_col @ col_sum_S                # (patch_h, patch_w)
    G_cols.append(G_d.reshape(-1))

    # Phi: (patch_h * patch_w,  3*bit_width + 1)
    Phi = torch.stack(G_cols, dim=1)
    rhs = R_S.reshape(-1, 1)

    # Solve via Tikhonov regularization (ridge regression):
    #   theta = (Phi^T Phi + lambda I)^{-1} Phi^T rhs
    # This always produces a finite, bounded solution even when Phi is
    # rank-deficient.  Start with a small lambda and increase adaptively
    # until the solution is numerically stable.
    PtP = Phi.T @ Phi          # (n_params, n_params)
    Ptr = Phi.T @ rhs          # (n_params, 1)
    mean_diag = PtP.diagonal().mean().item()

    theta = None
    for reg_scale in [1e-6, 1e-4, 1e-2, 1e-1]:
        try:
            lam = reg_scale * mean_diag
            A = PtP + lam * torch.eye(PtP.shape[0], device=device, dtype=dtype)
            sol = torch.linalg.solve(A, Ptr)  # (n_params, 1)
            t = sol.squeeze(1)
            if t.isfinite().all():
                theta = t
                break
        except Exception:
            continue

    if theta is None:
        return False

    # Write refined parameters back in-place
    with torch.no_grad():
        for p in range(bit_width):
            bqq_layer.a.data[p, i, j, 0, 0] = theta[3 * p]
            bqq_layer.b.data[p, i, j, 0, 0] = theta[3 * p + 1]
            bqq_layer.c.data[p, i, j, 0, 0] = theta[3 * p + 2]
        bqq_layer.d.data[i, j, 0, 0] = theta[-1]

    return True


def refine_bqq_layer(
    bqq_layer: BinaryQuadratic,
    W_original: torch.Tensor,
    H_full: torch.Tensor,
    damping: float = 1e-6,
) -> None:
    """Refine all patches of a BinaryQuadratic layer.

    Parameters
    ----------
    bqq_layer:
        The BinaryQuadratic module whose scale factors will be updated in-place.
    W_original:
        Original pretrained weight (out_features, in_features).
    H_full:
        Accumulated input correlation matrix X^T X (in_features, in_features).
    damping:
        Relative diagonal damping added to C_j for numerical stability.
    """
    device = bqq_layer.Y.device
    row_width = bqq_layer.row_width
    col_width = bqq_layer.col_width
    patch_h = bqq_layer.y_row
    patch_w = bqq_layer.z_col

    W = W_original.to(device=device, dtype=torch.float32)
    H = H_full.to(device=device, dtype=torch.float32)

    n_ok = 0
    for i in range(row_width):
        for j in range(col_width):
            r_start = i * patch_h
            c_start = j * patch_w

            W_block = W[r_start:r_start + patch_h, c_start:c_start + patch_w]
            C_j = H[c_start:c_start + patch_w, c_start:c_start + patch_w]

            ok = refine_patch(bqq_layer, i, j, W_block, C_j, damping)
            if ok:
                n_ok += 1

    n_total = row_width * col_width
    if n_ok < n_total:
        print(f"    Warning: {n_total - n_ok}/{n_total} patches skipped due to numerical issues")


# ---------------------------------------------------------------------------
# Main refinement pipeline
# ---------------------------------------------------------------------------

def find_bqq_layers(model: nn.Module, prefix: str = "") -> Dict[str, BinaryQuadratic]:
    """Recursively collect all BinaryQuadratic sub-modules and their full names."""
    result = {}
    for name, module in model.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(module, BinaryQuadratic):
            result[full] = module
        else:
            result.update(find_bqq_layers(module, full))
    return result


def run_refinement(
    model_name: str,
    bqq_model: nn.Module,
    calibration_loader,
    device: torch.device,
    damping: float = 1e-6,
) -> nn.Module:
    """Full scale-refinement pass over all BinaryQuadratic layers.

    Parameters
    ----------
    model_name:
        HuggingFace model name for the pretrained model.
    bqq_model:
        BQQ quantized model (contains BinaryQuadratic layers).
    calibration_loader:
        Iterable of token id batches for calibration.
    device:
        Device for computation.
    damping:
        Relative damping for Cholesky decomposition stability.

    Returns
    -------
    bqq_model with refined scale factors (modified in-place and returned).
    """
    # --- Identify all BQQ layers and find matching pretrained linear names ---
    bqq_layers = find_bqq_layers(bqq_model)
    print(f"Found {len(bqq_layers)} BinaryQuadratic layers to refine.")

    target_names = list(bqq_layers.keys())

    # --- Load pretrained model to collect calibration inputs ---
    print(f"Loading pretrained model: {model_name}")
    pretrained = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    pretrained.eval()

    print("Collecting input correlations H = X^T X for each target layer ...")
    H_dict = collect_hessians(pretrained, calibration_loader, target_names, device)

    # Free pretrained model from GPU before refinement
    pretrained.cpu()
    del pretrained
    torch.cuda.empty_cache()

    # --- Extract original weights from pretrained model (CPU) ---
    # We reload just to get weights without keeping the full model on GPU
    print("Reloading pretrained weights for W extraction ...")
    pretrained_weights = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    orig_weights: Dict[str, torch.Tensor] = {}
    for name, param in pretrained_weights.named_parameters():
        if name.endswith(".weight"):
            # Module name = param name without ".weight"
            module_name = name[: -len(".weight")]
            if module_name in target_names:
                orig_weights[module_name] = param.detach().cpu().float()
    del pretrained_weights
    torch.cuda.empty_cache()

    # --- Refine each BQQ layer ---
    bqq_model.to(device)
    for layer_name, bqq_layer in tqdm(bqq_layers.items(), desc="Refining layers"):
        if layer_name not in H_dict:
            print(f"  Skipping {layer_name}: no calibration data collected")
            continue
        if layer_name not in orig_weights:
            print(f"  Skipping {layer_name}: original weight not found")
            continue

        refine_bqq_layer(
            bqq_layer,
            orig_weights[layer_name],
            H_dict[layer_name],
            damping=damping,
        )

    return bqq_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Refine BQQ scale factors via ||WX - W'X||^2 minimization"
    )
    p.add_argument("--model_name", type=str, required=True,
                   help="HuggingFace pretrained model name or path")

    # BQQ model source: either a saved .pth or rebuilt from compressed data
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--bqq_model", type=Path,
                     help="Path to a saved BQQ model (.pth from torch.save)")
    src.add_argument("--compressed_data", type=Path,
                     help="Directory of BQQ compressed patch files")

    p.add_argument("--bit_width", type=int, default=2,
                   help="Bit width (used when rebuilding from compressed_data)")
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=10000)

    p.add_argument("--output", type=Path, required=True,
                   help="Output path for the refined model (.pth)")

    p.add_argument("--dataset", type=str, default="wikitext2",
                   choices=["wikitext2", "ptb", "c4"],
                   help="Calibration dataset")
    p.add_argument("--nsamples", type=int, default=128,
                   help="Number of calibration sequences")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--damping", type=float, default=1e-6,
                   help="Relative diagonal damping for Cholesky stability")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # --- Load or rebuild BQQ model ---
    if args.bqq_model is not None:
        print(f"Loading BQQ model from {args.bqq_model}")
        bqq_model = torch.load(args.bqq_model, map_location="cpu", weights_only=False)
    else:
        print(f"Rebuilding BQQ model from compressed data: {args.compressed_data}")
        base = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
        bqq_model = replace_linear_with_bqq(
            base,
            weights_dir=str(args.compressed_data),
            bit_width=args.bit_width,
            device="cpu",
        )

    # --- Calibration data ---
    print(f"Loading calibration data: {args.dataset} ({args.nsamples} samples, seqlen={args.seqlen})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_loader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        model=args.model_name,
        tokenizer=tokenizer,
    )

    # --- Refinement ---
    bqq_model = run_refinement(
        model_name=args.model_name,
        bqq_model=bqq_model,
        calibration_loader=train_loader,
        device=device,
        damping=args.damping,
    )

    # --- Save ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving refined model to {args.output}")
    torch.save(bqq_model, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
