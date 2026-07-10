import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch
import torch.nn as nn

from quantizer import BinaryQuadraticQuantization as BQQ


INTRA_LAYER_METHOD = '_hessian_aware_large_matrix_batched'


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


def build_deit_problem(seed=0, batch_size=8, image_size=224, module_name='blocks.0.attn.proj',
                       pretrained=True, timm_model='deit_small_patch16_224'):
    try:
        import timm
    except ImportError:
        return None

    torch.manual_seed(seed)
    model = timm.create_model(timm_model, pretrained=pretrained)
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
    return W, x_mat, f'{timm_model}:{target_name}'


def build_llama_problem(seed=0, module_name='model.layers.0.self_attn.q_proj',
                        model_name='meta-llama/Llama-2-7b-hf', nsamples=16, seqlen=512,
                        device_id=0, dataset='wikitext2', max_tokens=8192):
    """Load a (pretrained) causal LM, hook a target Linear, capture its input
    activations on a small calibration set, and return (W, X[in, tokens], source)."""
    try:
        from neural_network_compression.lm.src.model_loader import load_causal_lm
    except Exception as exc:
        print(f'[llama] could not import LM loader: {exc}')
        return None

    torch.manual_seed(seed)
    model = load_causal_lm(model_name)
    model.eval()
    dev = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model.to(dev)

    modules = dict(model.named_modules())
    target = modules.get(module_name)
    if not isinstance(target, nn.Linear):
        for name, mod in modules.items():
            if isinstance(mod, nn.Linear) and 'q_proj' in name:
                module_name, target = name, mod
                break
    if not isinstance(target, nn.Linear):
        print(f'[llama] no Linear target found (tried {module_name})')
        return None

    captured = []
    tok_count = [0]

    def hook(_module, inputs, _output):
        x = inputs[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        captured.append(x.float().cpu())
        tok_count[0] += x.shape[0]

    handle = target.register_forward_hook(hook)

    ids_list = None
    try:
        from transformers import AutoTokenizer
        from neural_network_compression.lm.src.datautils import get_loaders
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        loader, _ = get_loaders(dataset, nsamples=nsamples, seed=seed, seqlen=seqlen,
                                model=model_name, tokenizer=tokenizer)
        ids_list = [(b[0] if isinstance(b, (list, tuple)) else b) for b in loader]
    except Exception as exc:
        print(f'[llama] dataset load failed ({exc}); using random token ids')

    with torch.no_grad():
        if ids_list:
            for ids in ids_list:
                model(ids.to(dev))
                if tok_count[0] >= max_tokens:
                    break
        else:
            vocab = model.config.vocab_size
            g = torch.Generator().manual_seed(seed)
            for _ in range(nsamples):
                ids = torch.randint(0, vocab, (1, seqlen), generator=g)
                model(ids.to(dev))
                if tok_count[0] >= max_tokens:
                    break

    handle.remove()
    X = torch.cat(captured, 0)  # [tokens, in]
    if X.shape[0] > max_tokens:
        idx = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(seed))[:max_tokens]
        X = X[idx]
    W = target.weight.detach().cpu().float()
    x_mat = X.T.contiguous()  # [in, tokens]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W, x_mat, f'{model_name}:{module_name}'


def build_problem(
    seed=0,
    batch_size=8,
    image_size=224,
    module_name='blocks.0.attn.proj',
    force_synthetic=False,
    synthetic_in_features=384,
    synthetic_out_features=384,
    synthetic_samples=1024,
    model_family='deit',
    timm_model='deit_small_patch16_224',
    llama_model_name='meta-llama/Llama-2-7b-hf',
    llama_nsamples=16,
    llama_seqlen=512,
    llama_dataset='wikitext2',
    device_id=0,
):
    if force_synthetic or model_family == 'synthetic':
        return build_synthetic_problem(
            seed=seed, in_features=synthetic_in_features,
            out_features=synthetic_out_features, samples=synthetic_samples,
        )

    if model_family == 'llama':
        llama_problem = build_llama_problem(
            seed=seed, module_name=module_name, model_name=llama_model_name,
            nsamples=llama_nsamples, seqlen=llama_seqlen, device_id=device_id,
            dataset=llama_dataset,
        )
        if llama_problem is not None:
            return llama_problem
        print('[build_problem] llama unavailable; falling back to synthetic')
        return build_synthetic_problem(
            seed=seed, in_features=synthetic_in_features,
            out_features=synthetic_out_features, samples=synthetic_samples,
        )

    # default: deit (pretrained)
    deit_problem = build_deit_problem(
        seed=seed, batch_size=batch_size, image_size=image_size,
        module_name=module_name, pretrained=True, timm_model=timm_model,
    )
    if deit_problem is not None:
        return deit_problem
    return build_synthetic_problem(
        seed=seed, in_features=synthetic_in_features,
        out_features=synthetic_out_features, samples=synthetic_samples,
    )


def quantize_intra_layer_and_save(W, H, args, consolidated_path, compensation_mode=None,
                                  ldlq_act_order=False, ldlq_act_order_score='maxdiag',
                                  rank_alloc_mode='none', importance_weight=False,
                                  importance_score='diag'):
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
        compensation_mode=compensation_mode or args.compensation_mode,
        ldlq_act_order=ldlq_act_order,
        ldlq_act_order_score=ldlq_act_order_score,
        rank_alloc_mode=rank_alloc_mode,
        importance_weight=importance_weight,
        importance_score=importance_score,
    ).float().cpu()


def refine_bqq_decomposition(W, H, args, consolidated_path):
    refine_quantizer = BQQ(x=W, rank_scale=args.rank_scale)
    Wq_refined, _refined_decomp, refine_history = refine_quantizer.refine_decomposition_with_ste(
        all_decomposed=consolidated_path,
        H=H,
        num_steps=args.refine_steps,
        lr=args.refine_lr,
        factors_lr=args.refine_binary_lr,
        continuous_lr=args.refine_continuous_lr,
        weight_decay=args.refine_weight_decay,
        device_id=args.device_id,
        optimize_factors=not args.refine_coeffs_only,
        optimize_coeffs=True,
        optimize_theta=not args.fix_theta,
        row_group_batch_size=args.row_group_batch_size,
        consolidated_path=args.refined_output_path,
        log_interval=args.refine_log_interval,
    )
    return Wq_refined.float().cpu(), refine_history


def _import_quip_sharp(quip_sharp_root):
    root = Path(quip_sharp_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'QUIP-Sharp root does not exist: {root}')

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if 'quiptools_cuda' not in sys.modules:
        dummy_quiptools = ModuleType('quiptools_cuda')

        def _missing_quiptools(*_args, **_kwargs):
            raise RuntimeError(
                'quiptools_cuda is not installed. This comparison can compute quantization error '
                'from quip.quantize(), but QUIP-Sharp packed decode/matmul is unavailable.'
            )

        dummy_quiptools.decode_matvec_e8p = _missing_quiptools
        dummy_quiptools.decompress_packed_e8p = _missing_quiptools
        dummy_quiptools.decompress_e81b_packed = _missing_quiptools
        dummy_quiptools.lookupmatmul_e81b_k8 = _missing_quiptools
        sys.modules['quiptools_cuda'] = dummy_quiptools

    if 'lm_eval.base' not in sys.modules:
        lm_eval_module = sys.modules.setdefault('lm_eval', ModuleType('lm_eval'))
        lm_eval_base = ModuleType('lm_eval.base')
        lm_eval_base.BaseLM = object
        lm_eval_module.base = lm_eval_base
        sys.modules['lm_eval.base'] = lm_eval_base

    try:
        from lib import codebook, utils
        from lib.algo import quip
    except ModuleNotFoundError as exc:
        if exc.name in {'quiptools_cuda', 'lm_eval.base'}:
            raise RuntimeError(f'Failed to install import shim for optional QUIP-Sharp dependency: {exc.name}') from exc
        raise
    return codebook, quip, utils


def quantize_quip_sharp_ldlq(W, H, args):
    codebook_mod, quip_mod, quip_utils = _import_quip_sharp(args.quip_sharp_root)
    cb = codebook_mod.get_codebook(args.quip_codebook)
    validate_quip_sharp_problem(W, cb, args.quip_codebook)
    cb.maybe_pack_idxs = lambda idxs: idxs

    dtype = torch.float64 if args.quip_use_fp64 else torch.float32
    Hq = H.to(dtype=dtype).cpu()
    Hq = 0.5 * (Hq + Hq.T)
    Hq = quip_utils.regularize_H(Hq, Hq.shape[0], args.quip_sigma_reg)

    quip_args = SimpleNamespace(
        sigma_reg=args.quip_sigma_reg,
        sigma_reg2=args.quip_sigma_reg2,
        incoh_mode=args.quip_incoh_mode,
        lora_rank=args.quip_lora_rank,
        scale_override=args.quip_scale_override,
        resid_scale_override=args.quip_resid_scale_override,
        quip_tune_iters=args.quip_tune_iters,
        use_fp64=args.quip_use_fp64,
        full_svd=args.quip_full_svd,
        no_use_buffered=args.quip_no_use_buffered,
        rescale_WH=args.quip_rescale_wh,
        lowmem_ldlq=args.quip_lowmem_ldlq,
        save_pfx=args.quip_debug_dir,
    )

    device = 'cuda' if torch.cuda.is_available() and args.quip_device_id >= 0 else 'cpu'
    if device == 'cuda':
        device = f'cuda:{args.quip_device_id}'

    with torch.no_grad():
        Wq, _attr = quip_mod.quantize(Hq, W.cpu(), args.quip_lora_rank, cb, quip_args, device=device)
    return Wq.float().cpu()


def validate_quip_sharp_problem(W, cb, codebook_name):
    if W.shape[1] % cb.codesz != 0:
        raise ValueError(
            f'QUIP-Sharp codebook {codebook_name} requires in_features divisible by '
            f'codesz={cb.codesz}, got {W.shape[1]}.'
        )
    if W.shape[0] % 2 != 0:
        raise ValueError(f'QUIP-Sharp quantize asserts out_features is even, got {W.shape[0]}.')


def compare_quip_ldlq(args):
    W, X, source = build_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        module_name=args.module_name,
        force_synthetic=args.force_synthetic,
        synthetic_in_features=args.synthetic_in_features,
        synthetic_out_features=args.synthetic_out_features,
        synthetic_samples=args.synthetic_samples,
    )
    H = (X @ X.T).float()

    print(f'Source: {source}')
    print(f'W shape: {tuple(W.shape)}, X shape: {tuple(X.shape)}')
    print(f'BQQ method: {INTRA_LAYER_METHOD}, compensation=ldlq')
    print(f'QUIP-Sharp root: {args.quip_sharp_root}, codebook={args.quip_codebook}')

    codebook_mod, _quip_mod, _quip_utils = _import_quip_sharp(args.quip_sharp_root)
    validate_quip_sharp_problem(W, codebook_mod.get_codebook(args.quip_codebook), args.quip_codebook)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidated_path = str(Path(tmpdir) / 'bqq_ldlq_decomposition.pt')
        Wq_bqq = quantize_intra_layer_and_save(W, H, args, consolidated_path, compensation_mode='ldlq')
        refine_history = []
        if args.apply_ste_refine:
            Wq_bqq, refine_history = refine_bqq_decomposition(W, H, args, consolidated_path)
        rows.append({
            'source': source,
            'weight_shape': str(tuple(W.shape)),
            'activation_shape': str(tuple(X.shape)),
            'bits': args.bits,
            'method': f'BQQ:{INTRA_LAYER_METHOD}' + ('+ste_refine' if args.apply_ste_refine else ''),
            'compensation_mode': 'ldlq',
            'codebook': None,
            'ste_refine_steps': args.refine_steps if args.apply_ste_refine else 0,
            'ste_refine_last_loss': refine_history[-1] if len(refine_history) > 0 else None,
            'output_error_energy': output_error_energy(W, Wq_bqq, X),
            'weight_mse': torch.mean((W - Wq_bqq).pow(2)).item(),
        })

        Wq_quip = quantize_quip_sharp_ldlq(W, H, args)
        rows.append({
            'source': source,
            'weight_shape': str(tuple(W.shape)),
            'activation_shape': str(tuple(X.shape)),
            'bits': args.bits,
            'method': 'QUIP-Sharp:quip.quantize',
            'compensation_mode': 'ldlq',
            'codebook': args.quip_codebook,
            'ste_refine_steps': 0,
            'ste_refine_last_loss': None,
            'output_error_energy': output_error_energy(W, Wq_quip, X),
            'weight_mse': torch.mean((W - Wq_quip).pow(2)).item(),
        })

    best_error = min(row['output_error_energy'] for row in rows)
    bqq_error = rows[0]['output_error_energy']
    for row in rows:
        row['delta_vs_best'] = row['output_error_energy'] - best_error
        row['ratio_vs_best'] = row['output_error_energy'] / best_error if best_error != 0 else float('inf')
        row['delta_vs_bqq_ldlq'] = row['output_error_energy'] - bqq_error
        row['ratio_vs_bqq_ldlq'] = row['output_error_energy'] / bqq_error if bqq_error != 0 else float('inf')
    return rows



def quantize_bqq_incoherent(W, H, args, consolidated_path, *, compensation_mode='ldlq',
                            ldlq_act_order=False, ldlq_act_order_score='maxdiag',
                            rank_alloc_mode='none', importance_weight=False,
                            importance_score='diag', randomize=True):
    """QUIP-style incoherence (randomized Hadamard) around BQQ.

    Reuses QUIP-Sharp's RHT: pick random sign vectors SU (in-side, shared with H)
    and SV (out-side), transform Wr = RHT_W(W, SU, SV), Hr = RHT_H(H, SU), quantize
    the transformed weight with BQQ (ldlq), then invert with incoherence_process.
    The orthogonal transform leaves tr((W-Wq) H (W-Wq)^T) unchanged in value but
    makes Wr/Hr incoherent so the quantizer's error is spread benignly.
    """
    # Self-contained incoherence (copied from QUIP-Sharp into matrix_compression/incoherence.py).
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from incoherence import RHT_H, RHT_W, incoherence_process

    dev = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    Wd = W.to(dev).float()
    Hd = H.to(dev).float()
    m, n = Wd.shape  # out, in

    if randomize:
        g = torch.Generator().manual_seed(args.seed + 12345)
        SU = (torch.randn(n, generator=g).sign() + 1e-5).sign().to(torch.float32).to(dev)  # in-side
        SV = (torch.randn(m, generator=g).sign() + 1e-5).sign().to(torch.float32).to(dev)  # out-side
    else:
        # Plain Hadamard transform (no random sign flips): deterministic rotation
        # that mixes channels but preserves more structure than the full RHT.
        SU = torch.ones(n, dtype=torch.float32, device=dev)
        SV = torch.ones(m, dtype=torch.float32, device=dev)

    Hr = RHT_H(Hd, SU)
    Wr = RHT_W(Wd, SU, SV)

    hatWr = quantize_intra_layer_and_save(
        Wr.detach().cpu(), Hr.detach().cpu(), args, consolidated_path,
        compensation_mode=compensation_mode,
        ldlq_act_order=ldlq_act_order, ldlq_act_order_score=ldlq_act_order_score,
        rank_alloc_mode=rank_alloc_mode,
        importance_weight=importance_weight, importance_score=importance_score,
    ).to(dev)

    Wq = incoherence_process(hatWr, SU.cpu(), SV.cpu())
    return Wq.detach().cpu().float()


def compare_all_three(args):
    """Compare bqq-gptq, bqq-ldlq, and QUIP-Sharp on one layer's output error."""
    W, X, source = build_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        module_name=args.module_name,
        force_synthetic=args.force_synthetic,
        synthetic_in_features=args.synthetic_in_features,
        synthetic_out_features=args.synthetic_out_features,
        synthetic_samples=args.synthetic_samples,
        model_family=args.model_family,
        timm_model=args.timm_model,
        llama_model_name=args.llama_model_name,
        llama_nsamples=args.llama_nsamples,
        llama_seqlen=args.llama_seqlen,
        llama_dataset=args.llama_dataset,
        device_id=args.device_id,
    )
    H = (X @ X.T).float()
    out_numel = W.shape[0] * X.shape[1]

    print(f'Source: {source}')
    print(f'W shape: {tuple(W.shape)}, X shape: {tuple(X.shape)}')
    print(f'BQQ method: {INTRA_LAYER_METHOD}; bits={args.bits}, '
          f'max_patch_size={args.max_patch_size}, nstep={args.nstep}, '
          f'use_multibqq={args.use_multibqq}, scale_refine={args.scale_refine}')

    def mk_row(method, mode, codebook, Wq):
        oee = output_error_energy(W, Wq, X)
        return {
            'source': source,
            'weight_shape': str(tuple(W.shape)),
            'activation_shape': str(tuple(X.shape)),
            'bits': args.bits,
            'method': method,
            'compensation_mode': mode,
            'codebook': codebook,
            'output_error_energy': oee,
            'output_mse': oee / out_numel,
            'weight_mse': torch.mean((W - Wq).pow(2)).item(),
        }

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode in ['gptq', 'ldlq']:
            cp = str(Path(tmpdir) / f'bqq_{mode}.pt')
            Wq = quantize_intra_layer_and_save(W, H, args, cp, compensation_mode=mode)
            rows.append(mk_row(f'BQQ:{mode}', mode, None, Wq))

        # Importance-weighted BQQ (weighted objective sum((sqrt(H_jj)*(W-Wq))^2))
        # vs its unweighted counterpart above, optional.
        if getattr(args, 'importance_weight', False):
            cp_w = str(Path(tmpdir) / 'bqq_ldlq_wimp.pt')
            Wq_w = quantize_intra_layer_and_save(W, H, args, cp_w, compensation_mode='ldlq',
                                                 importance_weight=True)
            rows.append(mk_row('BQQ:ldlq+wimp', 'ldlq', None, Wq_w))

            cp_w2 = str(Path(tmpdir) / 'bqq_ldlq_ao_maxdiag_wimp.pt')
            Wq_w2 = quantize_intra_layer_and_save(W, H, args, cp_w2, compensation_mode='ldlq',
                                                  ldlq_act_order=True, ldlq_act_order_score='maxdiag',
                                                  importance_weight=True)
            rows.append(mk_row('BQQ:ldlq-ao-maxdiag+wimp', 'ldlq', None, Wq_w2))

        # LDLQ with activation-order: one row per requested group-score metric
        # (static / maxdiag / trace / logdet).
        ao_scores = [s.strip() for s in args.ldlq_act_order_scores.split(',') if s.strip()]
        for score in ao_scores:
            cp_ao = str(Path(tmpdir) / f'bqq_ldlq_ao_{score}.pt')
            Wq_ao = quantize_intra_layer_and_save(W, H, args, cp_ao, compensation_mode='ldlq',
                                                  ldlq_act_order=True, ldlq_act_order_score=score)
            rows.append(mk_row(f'BQQ:ldlq-ao-{score}', 'ldlq', None, Wq_ao))
            # Optional STE refinement on top of the act-order decomposition.
            if args.apply_ste_refine:
                Wq_ref, hist = refine_bqq_decomposition(W, H, args, cp_ao)
                r = mk_row(f'BQQ:ldlq-ao-{score}+ste{args.refine_steps}', 'ldlq', None, Wq_ref)
                r['ste_refine_last_loss'] = hist[-1] if len(hist) else None
                rows.append(r)

        # Incoherence-processed BQQ (QUIP-style RHT around ldlq), optional.
        if args.incoherence:
            try:
                cp_ic = str(Path(tmpdir) / 'bqq_ldlq_incoh.pt')
                Wq_ic = quantize_bqq_incoherent(W, H, args, cp_ic, compensation_mode='ldlq',
                                                ldlq_act_order=False)
                rows.append(mk_row('BQQ:ldlq+incoh', 'ldlq', None, Wq_ic))

                cp_ic2 = str(Path(tmpdir) / 'bqq_ldlq_ao_maxdiag_incoh.pt')
                Wq_ic2 = quantize_bqq_incoherent(W, H, args, cp_ic2, compensation_mode='ldlq',
                                                 ldlq_act_order=True, ldlq_act_order_score='maxdiag')
                rows.append(mk_row('BQQ:ldlq-ao-maxdiag+incoh', 'ldlq', None, Wq_ic2))
            except Exception as exc:
                print(f'Incoherence BQQ rows skipped/failed: {exc}')

        try:
            codebook_mod, _q, _u = _import_quip_sharp(args.quip_sharp_root)
            validate_quip_sharp_problem(W, codebook_mod.get_codebook(args.quip_codebook), args.quip_codebook)
            Wq_quip = quantize_quip_sharp_ldlq(W, H, args)
            rows.append(mk_row('QUIP-Sharp', 'ldlq', args.quip_codebook, Wq_quip))
        except Exception as exc:
            print(f'QUIP-Sharp comparison skipped/failed: {exc}')

    if rows:
        best = min(r['output_error_energy'] for r in rows)
        for r in rows:
            r['ratio_vs_best'] = r['output_error_energy'] / best if best else float('inf')
    return rows


def compare_compensation_modes(args):
    W, X, source = build_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        module_name=args.module_name,
        force_synthetic=args.force_synthetic,
        synthetic_in_features=args.synthetic_in_features,
        synthetic_out_features=args.synthetic_out_features,
        synthetic_samples=args.synthetic_samples,
    )
    H = (X @ X.T).float()

    print(f'Source: {source}')
    print(f'W shape: {tuple(W.shape)}, X shape: {tuple(X.shape)}')
    print(f'Method: {INTRA_LAYER_METHOD}')

    rows = []
    modes = [m.strip() for m in args.compensation_modes.split(',') if m.strip()]
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode in modes:
            if mode not in {'gptq', 'ldlq', 'none'}:
                raise ValueError(f'Unknown compensation mode: {mode}')

            consolidated_path = str(Path(tmpdir) / f'intra_layer_decomposition_{mode}.pt')
            Wq = quantize_intra_layer_and_save(W, H, args, consolidated_path, compensation_mode=mode)
            output_error = output_error_energy(W, Wq, X)
            weight_mse = torch.mean((W - Wq).pow(2)).item()
            rows.append({
                'source': source,
                'weight_shape': str(tuple(W.shape)),
                'activation_shape': str(tuple(X.shape)),
                'bits': args.bits,
                'rank_scale': args.rank_scale,
                'max_patch_size': args.max_patch_size,
                'nstep': args.nstep,
                'scale_refine': args.scale_refine,
                'use_multibqq': args.use_multibqq,
                'method': INTRA_LAYER_METHOD,
                'compensation_mode': mode,
                'output_error_energy': output_error,
                'weight_mse': weight_mse,
            })

    if rows:
        best_error = min(row['output_error_energy'] for row in rows)
        baseline = next((row for row in rows if row['compensation_mode'] == 'none'), None)
        baseline_error = baseline['output_error_energy'] if baseline is not None else None
        for row in rows:
            row['delta_vs_best'] = row['output_error_energy'] - best_error
            row['ratio_vs_best'] = row['output_error_energy'] / best_error if best_error != 0 else float('inf')
            row['delta_vs_none'] = (
                row['output_error_energy'] - baseline_error if baseline_error is not None else None
            )
            row['ratio_vs_none'] = (
                row['output_error_energy'] / baseline_error
                if baseline_error not in (None, 0) else None
            )
    return rows

def compare_intra_layer_refinement(args):
    W, X, source = build_problem(
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        module_name=args.module_name,
        force_synthetic=args.force_synthetic,
        synthetic_in_features=args.synthetic_in_features,
        synthetic_out_features=args.synthetic_out_features,
        synthetic_samples=args.synthetic_samples,
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
    parser.add_argument('--nstep', type=int, default=10000)
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
    parser.add_argument('--force-synthetic', action='store_true')
    parser.add_argument('--synthetic-in-features', type=int, default=384)
    parser.add_argument('--synthetic-out-features', type=int, default=384)
    parser.add_argument('--synthetic-samples', type=int, default=1024)
    parser.add_argument('--scale-refine', action='store_true', default=True)
    parser.add_argument('--compensation-mode', type=str, default='ldlq', choices=['gptq', 'ldlq', 'none'])
    parser.add_argument('--compensation-modes', type=str, default='none,gptq,ldlq',
                        help='Comma-separated modes used by --experiment compensation')
    parser.add_argument('--experiment', type=str, default='ste-refine',
                        choices=['ste-refine', 'compensation', 'quip-ldlq', 'all3'])
    parser.add_argument('--model-family', type=str, default='deit',
                        choices=['deit', 'llama', 'synthetic'],
                        help='Which real-model layer to quantize (used by --experiment all3)')
    parser.add_argument('--timm-model', type=str, default='deit_small_patch16_224')
    parser.add_argument('--llama-model-name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--llama-nsamples', type=int, default=16)
    parser.add_argument('--llama-seqlen', type=int, default=512)
    parser.add_argument('--llama-dataset', type=str, default='wikitext2')
    parser.add_argument('--no-scale-refine', dest='scale_refine', action='store_false')
    parser.add_argument('--refine-steps', type=int, default=1000)
    parser.add_argument('--refine-lr', type=float, default=1e-3)
    parser.add_argument('--refine-binary-lr', type=float, default=1e-3,
                        help='AdamW LR for STE binary factors Y/Z during refine.')
    parser.add_argument('--refine-continuous-lr', type=float, default=1e-4,
                        help='AdamW LR for continuous params (coeff/theta/beta) during refine.')
    parser.add_argument('--refine-weight-decay', type=float, default=0.0)
    parser.add_argument('--scratch-steps', type=int, default=5000)
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
    parser.add_argument('--ldlq-act-order', action='store_true',
                        help='Reorder LDLQ column groups (act-order) before quantizing.')
    parser.add_argument('--ldlq-act-order-score', type=str, default='maxdiag',
                        choices=['static', 'maxdiag', 'trace'],
                        help='Group-score for LDLQ act-order: static=original-order LDL-D '
                             'heuristic; maxdiag/trace=true pivoted (Schur-updated).')
    parser.add_argument('--ldlq-act-order-scores', type=str, default='maxdiag',
                        help='Comma-separated act-order scores to compare as separate rows '
                             'in --experiment all3 (e.g. "static,maxdiag,trace,logdet").')
    parser.add_argument('--apply-ste-refine', action='store_true',
                        help='Apply refine_decomposition_with_ste to the BQQ row in --experiment quip-ldlq')
    parser.add_argument('--incoherence', action='store_true',
                        help='Add QUIP-style incoherence (randomized Hadamard) BQQ rows to --experiment all3')
    parser.add_argument('--importance-weight', action='store_true',
                        help='Add importance-weighted BQQ rows (weighted objective sum((sqrt(H_jj)(W-Wq))^2)) '
                             'to --experiment all3, to compare against the unweighted BQQ rows')
    parser.add_argument('--quip-sharp-root', type=str, default='/work2/k-kuroki/quip-sharp')
    parser.add_argument('--quip-codebook', type=str, default='E8P12',
                        choices=['E8P12', 'E8P12RVQ3B', 'E8P12RVQ4B'])
    parser.add_argument('--quip-sigma-reg', type=float, default=1e-2)
    parser.add_argument('--quip-sigma-reg2', type=float, default=1e-2)
    parser.add_argument('--quip-incoh-mode', type=str, default='had', choices=['had', 'kron'])
    parser.add_argument('--quip-lora-rank', type=int, default=0)
    parser.add_argument('--quip-scale-override', type=float, default=-1)
    parser.add_argument('--quip-resid-scale-override', type=float, default=-1)
    parser.add_argument('--quip-tune-iters', type=int, default=10)
    parser.add_argument('--quip-device-id', type=int, default=0)
    parser.add_argument('--quip-debug-dir', type=str, default='/tmp')
    parser.add_argument('--quip-use-fp64', action='store_true')
    parser.add_argument('--quip-full-svd', action='store_true')
    parser.add_argument('--quip-no-use-buffered', action='store_true')
    parser.add_argument('--quip-rescale-wh', action='store_true')
    parser.add_argument('--quip-lowmem-ldlq', action='store_true')
    args = parser.parse_args()

    refined_output_path = args.refined_output_path.strip()
    scratch_output_path = args.scratch_output_path.strip()
    args.refined_output_path = refined_output_path if refined_output_path else None
    args.scratch_output_path = scratch_output_path if scratch_output_path else None

    if args.experiment == 'compensation':
        result = compare_compensation_modes(args)
        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values('output_error_energy')
    elif args.experiment == 'quip-ldlq':
        result = compare_quip_ldlq(args)
        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values('output_error_energy')
    elif args.experiment == 'all3':
        result = compare_all_three(args)
        df = pd.DataFrame(result)
        if not df.empty:
            df = df.sort_values('output_error_energy')
    else:
        result = compare_intra_layer_refinement(args)
        df = pd.DataFrame([result])
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f'Saved results to: {output_path}')


if __name__ == '__main__':
    main()
