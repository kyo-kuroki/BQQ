import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
import torch.nn as nn

from quantizer import BinaryQuadraticQuantization as BQQ


INTRA_LAYER_METHOD = '_intra_layer_hessian_aware_large_matrix_batched'


def output_error_energy(W, Wq, X):
    diff = W @ X - Wq @ X
    return torch.sum(diff * diff).item()


def build_synthetic_problem(seed=0, in_features=384, out_features=384, samples=1024):
    g = torch.Generator().manual_seed(seed)
    latent_rank = min(64, in_features, out_features)
    left = torch.randn(out_features, latent_rank, generator=g)
    right = torch.randn(latent_rank, in_features, generator=g)
    W = (left @ right + 0.05 * torch.randn(out_features, in_features, generator=g)).float()

    act_rank = min(96, in_features, samples)
    act_left = torch.randn(in_features, act_rank, generator=g)
    act_right = torch.randn(act_rank, samples, generator=g)
    X = (act_left @ act_right + 0.05 * torch.randn(in_features, samples, generator=g)).float()
    return W, X, 'synthetic'


def build_deit_problem(seed=0, batch_size=8, image_size=224, module_name='blocks.0.attn.proj'):
    try:
        import timm
    except ImportError:
        return None

    torch.manual_seed(seed)
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    model.eval()

    modules = dict(model.named_modules())
    preferred = [
        module_name,
        'blocks.0.attn.proj',
        'blocks.0.mlp.fc1',
        'blocks.0.mlp.fc2',
    ]

    target_name, target_module = None, None
    for name in preferred:
        mod = modules.get(name)
        if isinstance(mod, nn.Linear):
            target_name, target_module = name, mod
            break

    if target_module is None:
        for name, mod in modules.items():
            if isinstance(mod, nn.Linear):
                target_name, target_module = name, mod
                break

    if target_module is None:
        return None

    captured = {}

    def hook(_module, inputs, _output):
        captured['x'] = inputs[0].detach().cpu().float()

    handle = target_module.register_forward_hook(hook)
    with torch.no_grad():
        images = torch.randn(batch_size, 3, image_size, image_size)
        model(images)
    handle.remove()

    act = captured.get('x')
    if act is None:
        return None

    if act.ndim == 3:
        x_mat = act.reshape(-1, act.shape[-1]).T.contiguous()
    elif act.ndim == 2:
        x_mat = act.T.contiguous()
    else:
        return None

    W = target_module.weight.detach().cpu().float()
    return W, x_mat, f'deit_small_patch16_224:{target_name}'


def build_problem(seed=0, batch_size=8, image_size=224, module_name='blocks.0.attn.proj'):
    deit_problem = build_deit_problem(
        seed=seed,
        batch_size=batch_size,
        image_size=image_size,
        module_name=module_name,
    )
    if deit_problem is not None:
        return deit_problem
    return build_synthetic_problem(seed=seed)


def quantize_intra_layer_and_save(W, H, args, consolidated_path):
    quantizer = BQQ(x=W, rank_scale=args.rank_scale)
    method = getattr(quantizer, INTRA_LAYER_METHOD)
    return method(
        max_patch_size=args.max_patch_size,
        bit_width=args.bits,
        H=H,
        consolidated_path=consolidated_path,
        zeta=args.zeta,
        eta=args.eta,
        Tinit=args.Tinit,
        Tfin=args.Tfin,
        Nstep=args.nstep,
        seed=args.seed,
        main_gpu_id=args.device_id,
        damping=args.damping,
        scale_refine=args.scale_refine,
        use_multibqq=args.use_multibqq if hasattr(args, 'use_multibqq') else False,
    ).float().cpu()


def compare_intra_layer_refinement(args):
    W, X, source = build_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        module_name=args.module_name,
    )
    H = (X @ X.T).float()

    print(f'Source: {source}')
    print(f'W shape: {tuple(W.shape)}, X shape: {tuple(X.shape)}')
    print(f'Method: {INTRA_LAYER_METHOD}')

    with tempfile.TemporaryDirectory() as tmpdir:
        consolidated_path = str(Path(tmpdir) / 'intra_layer_decomposition.pt')
        Wq_before = quantize_intra_layer_and_save(W, H, args, consolidated_path)
        intra_layer_err_before = output_error_energy(W, Wq_before, X)

        refine_quantizer = BQQ(x=W, rank_scale=args.rank_scale)
        Wq_after, refined_decomp, refine_history = refine_quantizer.refine_decomposition_with_ste(
            all_decomposed=consolidated_path,
            H=H,
            num_steps=args.refine_steps,
            lr=args.refine_lr,
            weight_decay=args.refine_weight_decay,
            device_id=args.device_id,
            optimize_factors=not args.refine_coeffs_only,
            optimize_coeffs=True,
            optimize_theta=not args.fix_theta,
            row_group_batch_size=args.row_group_batch_size,
            consolidated_path=args.refined_output_path,
            log_interval=args.refine_log_interval,
        )
        intra_layer_err_after = output_error_energy(W, Wq_after, X)

        scratch_quantizer = BQQ(x=W, rank_scale=args.rank_scale)
        Wq_scratch, scratch_decomp, scratch_history = scratch_quantizer.optimize_decomposition_from_scratch_with_ste(
            max_patch_size=args.max_patch_size,
            bit_width=args.bits,
            H=H,
            num_steps=args.scratch_steps,
            lr=args.scratch_lr,
            weight_decay=args.scratch_weight_decay,
            device_id=args.device_id,
            optimize_factors=not args.refine_coeffs_only,
            optimize_coeffs=True,
            optimize_theta=not args.fix_theta,
            row_group_batch_size=args.scratch_row_group_batch_size,
            consolidated_path=args.scratch_output_path,
            log_interval=args.refine_log_interval,
            seed=args.seed,
        )
        scratch_err = output_error_energy(W, Wq_scratch, X)

    result = {
        'source': source,
        'weight_shape': str(tuple(W.shape)),
        'activation_shape': str(tuple(X.shape)),
        'bits': args.bits,
        'rank_scale': args.rank_scale,
        'max_patch_size': args.max_patch_size,
        'nstep': args.nstep,
        'scale_refine': args.scale_refine,
        'method': INTRA_LAYER_METHOD,
        'error_before_refine': intra_layer_err_before,
        'error_after_refine': intra_layer_err_after,
        'improvement_refine': intra_layer_err_before - intra_layer_err_after,
        'after_div_before': intra_layer_err_after / intra_layer_err_before if intra_layer_err_before != 0 else float('inf'),
        'scratch_error': scratch_err,
        'scratch_minus_before': scratch_err - intra_layer_err_before,
        'scratch_div_before': scratch_err / intra_layer_err_before if intra_layer_err_before != 0 else float('inf'),
        'scratch_minus_after_refine': scratch_err - intra_layer_err_after,
        'scratch_div_after_refine': scratch_err / intra_layer_err_after if intra_layer_err_after != 0 else float('inf'),
        'refine_steps': args.refine_steps,
        'refine_lr': args.refine_lr,
        'refine_weight_decay': args.refine_weight_decay,
        'scratch_steps': args.scratch_steps,
        'scratch_lr': args.scratch_lr,
        'scratch_weight_decay': args.scratch_weight_decay,
        'refine_coeffs_only': args.refine_coeffs_only,
        'fix_theta': args.fix_theta,
        'row_group_batch_size': args.row_group_batch_size,
        'scratch_row_group_batch_size': args.scratch_row_group_batch_size,
        'refine_last_loss': refine_history[-1] if len(refine_history) > 0 else None,
        'scratch_last_loss': scratch_history[-1] if len(scratch_history) > 0 else None,
        'num_refined_entries': len(refined_decomp),
        'num_scratch_entries': len(scratch_decomp),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=2)
    parser.add_argument('--rank-scale', type=float, default=1.0)
    parser.add_argument('--max-patch-size', type=int, default=128)
    parser.add_argument('--nstep', type=int, default=200)
    parser.add_argument('--zeta', type=float, default=4.0)
    parser.add_argument('--eta', type=float, default=0.06)
    parser.add_argument('--Tinit', type=float, default=0.2)
    parser.add_argument('--Tfin', type=float, default=0.005)
    parser.add_argument('--damping', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--module-name', type=str, default='blocks.0.attn.proj')
    parser.add_argument('--scale-refine', action='store_true', default=True)
    parser.add_argument('--no-scale-refine', dest='scale_refine', action='store_false')
    parser.add_argument('--refine-steps', type=int, default=200)
    parser.add_argument('--refine-lr', type=float, default=1e-3)
    parser.add_argument('--refine-weight-decay', type=float, default=0.0)
    parser.add_argument('--scratch-steps', type=int, default=200)
    parser.add_argument('--scratch-lr', type=float, default=1e-3)
    parser.add_argument('--scratch-weight-decay', type=float, default=0.0)
    parser.add_argument('--refine-log-interval', type=int, default=20)
    parser.add_argument('--refine-coeffs-only', action='store_true')
    parser.add_argument('--fix-theta', action='store_true')
    parser.add_argument('--row-group-batch-size', type=int, default=0)
    parser.add_argument('--scratch-row-group-batch-size', type=int, default=0)
    parser.add_argument('--refined-output-path', type=str, default='')
    parser.add_argument('--scratch-output-path', type=str, default='')
    parser.add_argument('--output-csv', type=str, default=str(Path(__file__).with_name('results') / 'comparison_of_optimization.csv'))
    parser.add_argument('--use-multibqq', action='store_true')
    args = parser.parse_args()

    refined_output_path = args.refined_output_path.strip()
    scratch_output_path = args.scratch_output_path.strip()
    args.refined_output_path = refined_output_path if refined_output_path else None
    args.scratch_output_path = scratch_output_path if scratch_output_path else None

    result = compare_intra_layer_refinement(args)
    df = pd.DataFrame([result])
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f'Saved results to: {output_path}')


if __name__ == '__main__':
    main()
