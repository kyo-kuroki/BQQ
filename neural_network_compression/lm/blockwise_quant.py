"""
Block-wise BQQ quantization with block output error optimization.

Pipeline:
  1. Load pretrained model, cache each block's input/output via calibration data
  2. For target block, process Linear weights in REVERSE order (last -> first):
     a. Collect H = X^T X just-in-time from current block state
        (downstream layers already quantized+fine-tuned -> X is accurate)
     b. BQQ quantize the weight using fresh H
     c. Replace Linear -> BinaryQuadratic (Y,Z fixed; a,b,c,d learnable)
     d. Optimize ALL continuous params in block (BQQ a,b,c,d + remaining
        unquantized Linear weights + LayerNorm params) to minimize
        block output MSE vs pretrained output
  3. Save quantized block

Blocks are independent -> can be parallelized via --block_idx argument.

Usage:
  python blockwise_quant.py --model_name Qwen/Qwen2.5-1.5B --block_idx 0 \
    --dataset wikitext2 --nsamples 128 --seqlen 2048 \
    --bit_width 4 --group_size 128 --num_steps 50000 \
    --epochs 5 --lr 1e-4 --save_dir ./blockwise_output
"""

import argparse
import copy
import os
import sys
import tempfile
import dill
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quantizer import BinaryQuadraticQuantization

try:
    from .src.build_bqq_model import BinaryQuadratic, PartialBQQLinear, TrainableSTEBinaryQuadratic, assemble_from_blocks, convert_ste_model_to_binaryquadratic
    from .src.compressed_data import build_consolidated_index, default_compressed_data_dir, get_bqq_matrices, load_layer_patches
    from .src.datautils import get_loaders
    from .src.model_loader import load_causal_lm, get_decoder_layer, get_decoder_block_prefix
    from .layerwise_quant import layerwise_quantize_block
except ImportError:
    from neural_network_compression.lm.src.build_bqq_model import BinaryQuadratic, PartialBQQLinear, TrainableSTEBinaryQuadratic, assemble_from_blocks, convert_ste_model_to_binaryquadratic
    from neural_network_compression.lm.src.compressed_data import build_consolidated_index, default_compressed_data_dir, get_bqq_matrices, load_layer_patches
    from neural_network_compression.lm.src.datautils import get_loaders
    from neural_network_compression.lm.src.model_loader import load_causal_lm, get_decoder_layer, get_decoder_block_prefix
    from neural_network_compression.lm.layerwise_quant import layerwise_quantize_block


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

    block = get_decoder_layer(model, block_idx)
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
                            rank_scale, seed, device_id, H=None,
                            scale_refine=True, damping=1e-6):
    """Quantize a 2D weight tensor with BQQ. Returns (A, Y, Z) for BinaryQuadratic.

    If H is provided, uses intra-layer Hessian-aware BQQ (column-wise compensation).
    Otherwise falls back to standard BQQ.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidated_path = os.path.join(tmpdir, 'temp.pth')
        quantizer = BinaryQuadraticQuantization(weight, rank_scale=rank_scale)

        kwargs = dict(
            max_patch_size=group_size,
            bit_width=bit_width,
            consolidated_path=consolidated_path,
            Nstep=num_steps,
            seed=seed,
            main_gpu_id=device_id,
        )
        if H is not None:
            kwargs.update(H=H, hessian_mode='intra-layer-ste',
                          scale_refine=scale_refine, damping=damping)

        quantizer.bqq_large_matrix_multi_worker(**kwargs)
        patches = torch.load(consolidated_path, weights_only=False, map_location='cpu')

    A, Y, Z = get_bqq_matrices(patches, bit_width)
    return A, Y, Z




def ensure_layerwise_block_available(
    block,
    *,
    model_name,
    block_idx,
    block_prefix,
    layerwise_dir,
    bit_width,
    group_size,
    num_steps,
    rank_scale,
    seed,
    scale_refine,
    damping,
    dataloader,
    device,
    refine_coeffs_only=False,
    fix_theta=False,
    fix_beta=False,
    optimizer_name='adamw',
    momentum=0.0,
    binary_lr=None,
    continuous_lr=None,
):
    """Generate missing layerwise quantization results for a block on demand."""
    layerwise_dir = Path(layerwise_dir)
    patch_index = build_consolidated_index(layerwise_dir)
    linear_names = get_quantizable_linears(block)
    missing = []
    for linear_name in linear_names:
        full_name = f"{block_prefix}.{linear_name}.weight"
        if full_name not in patch_index:
            missing.append(full_name)
    if not missing:
        return

    print(f'Missing {len(missing)} layerwise target(s) for block {block_idx}; running internal layerwise quantization ...')
    layerwise_quantize_block(
        model_name=model_name,
        save_dir=layerwise_dir,
        block_idx=block_idx,
        bit_width=bit_width,
        group_size=group_size,
        num_steps=num_steps,
        rank_scale=rank_scale,
        seed=seed,
        scale_refine=scale_refine,
        damping=damping,
        ste_refine_steps=200,
        ste_refine_lr=1e-3,
        ste_refine_weight_decay=0.0,
        ste_refine_binary_lr=None,
        ste_refine_continuous_lr=None,
        ste_refine_log_interval=20,
        refine_coeffs_only=refine_coeffs_only,
        fix_theta=fix_theta,
        fix_beta=fix_beta,
        row_group_batch_size=None,
        use_multibqq=True,
        workers_per_gpu=1,
        calibration_loader=dataloader,
    )


def load_layerwise_block_from_patches(
    block,
    *,
    block_idx,
    block_prefix,
    layerwise_dir,
    bit_width,
    trainable_ste=True,
    optimize_factors=True,
    optimize_coeffs=True,
    optimize_theta=True,
    optimize_beta=True,
):
    """Replace all Linear layers in a block from precomputed layerwise BQQ patches."""
    patch_index = build_consolidated_index(layerwise_dir)
    if not patch_index:
        raise FileNotFoundError(f'No consolidated layerwise patches found in {layerwise_dir}')

    linear_names = get_quantizable_linears(block)
    for linear_name in linear_names:
        full_name = f"{block_prefix}.{linear_name}.weight"
        patch_list = load_layer_patches(full_name, patch_index, map_location='cpu')
        if not patch_list:
            raise FileNotFoundError(f'Missing layerwise patches for {full_name} in {layerwise_dir}')
        A, Y, Z = get_bqq_matrices(patch_list, bit_width)
        lin = _get_submodule(block, linear_name)
        bias = lin.bias.data.clone().float() if lin.bias is not None else None
        replace_linear_in_block(
            block, linear_name, A, Y, Z, bias=bias,
            trainable_ste=trainable_ste,
            optimize_factors=optimize_factors,
            optimize_coeffs=optimize_coeffs,
            optimize_theta=optimize_theta,
            optimize_beta=optimize_beta,
        )
        print(f' Loaded layerwise BQQ: {full_name}')

    return linear_names


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


def replace_linear_in_block(block, linear_name, A, Y, Z, bias=None, *, trainable_ste=True, optimize_factors=True, optimize_coeffs=True, optimize_theta=True, optimize_beta=True):
    """Replace a specific Linear in block with a BQQ module."""
    if trainable_ste:
        bqq_module = TrainableSTEBinaryQuadratic(
            Y, Z, A, bias=bias,
            optimize_factors=optimize_factors,
            optimize_coeffs=optimize_coeffs,
            optimize_theta=optimize_theta,
            optimize_beta=optimize_beta,
        )
    else:
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
                          epochs, lr, device, max_grad_norm=1.0, optimizer_name='adamw', momentum=0.0,
                          binary_lr=None, continuous_lr=None):
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

    named_params = [(name, p) for name, p in block.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    n_params = sum(p.numel() for p in params)
    binary_lr_value = lr if binary_lr is None else binary_lr
    continuous_lr_value = lr if continuous_lr is None else continuous_lr
    binary_params = [p for name, p in named_params if name.lower().endswith('y_fp') or name.lower().endswith('z_fp')]
    continuous_params = [p for name, p in named_params if not (name.lower().endswith('y_fp') or name.lower().endswith('z_fp'))]
    print(f'    Optimizing {len(params)} param groups ({n_params:,} elements), '
          f'optimizer={optimizer_name}, lr={lr}, binary_lr={binary_lr_value}, continuous_lr={continuous_lr_value}, '
          f'momentum={momentum}, epochs={epochs}, max_grad_norm={max_grad_norm}')

    param_groups = []
    if continuous_params:
        param_groups.append({'params': continuous_params, 'lr': continuous_lr_value})
    if binary_params:
        param_groups.append({'params': binary_params, 'lr': binary_lr_value})

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=momentum)
    else:
        raise ValueError(f'Unsupported optimizer_name={optimizer_name!r}. Use adamw or sgd.')

    best_mse = float('inf')
    best_state = None

    for epoch in tqdm(range(epochs), desc='Blockwise optimization', unit='epoch'):
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

def collect_single_hessian(block, linear_name, inputs_cache, device):
    """Collect H = X^T X for one Linear layer given the current block state.

    Called just before quantizing that layer so H reflects the actual input
    distribution (including the effect of previously quantized + fine-tuned
    layers earlier in the block).
    """
    H = None

    target_module = None
    for name, module in block.named_modules():
        if name == linear_name and isinstance(module, nn.Linear):
            target_module = module
            break
    if target_module is None:
        return None

    def _hook(module, inp, _out):
        nonlocal H
        x = inp[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        h = x.T @ x
        H = h if H is None else H.add_(h)

    handle = target_module.register_forward_hook(_hook)
    block.to(device).eval()
    with torch.no_grad():
        for inp in inputs_cache:
            try:
                run_block_forward(block, inp, device)
            except Exception:
                pass
    handle.remove()

    return H.cpu() if H is not None else None


@torch.no_grad()
def collect_cross_hessians_for_linear(original_block, current_block, linear_name, inputs_cache, device):
    """Collect X^T X' and X'^T X' for one layer under original/current block states."""
    original_block.to(device).eval()
    current_block.to(device).eval()

    original_module = _get_submodule(original_block, linear_name)
    current_module = _get_submodule(current_block, linear_name)

    captured = {'orig': None, 'cur': None}
    H_cross = None
    H_current = None

    def _orig_hook(module, inp, _out):
        x = inp[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        captured['orig'] = x

    def _cur_hook(module, inp, _out):
        x = inp[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        captured['cur'] = x

    h_orig = original_module.register_forward_hook(_orig_hook)
    h_cur = current_module.register_forward_hook(_cur_hook)
    try:
        for inp in inputs_cache:
            captured['orig'] = None
            captured['cur'] = None
            try:
                run_block_forward(original_block, inp, device)
                run_block_forward(current_block, inp, device)
            except Exception:
                continue

            x_orig = captured['orig']
            x_cur = captured['cur']
            if x_orig is None or x_cur is None:
                continue

            cross = x_orig.T @ x_cur
            cur = x_cur.T @ x_cur
            H_cross = cross if H_cross is None else H_cross.add_(cross)
            H_current = cur if H_current is None else H_current.add_(cur)
    finally:
        h_orig.remove()
        h_cur.remove()

    if H_cross is None or H_current is None:
        return None, None
    return H_cross.cpu(), H_current.cpu()


def solve_closed_form_weight(weight, H_cross, H_current, damping=1e-6):
    """Solve min_W' ||W X - W' X'||_F^2 via a damped closed form."""
    if H_cross is None or H_current is None:
        return weight.detach().float().clone()

    W = weight.detach().float()
    solve_device = W.device
    H_cross = H_cross.detach().to(device=solve_device, dtype=torch.float32)
    H_current = H_current.detach().to(device=solve_device, dtype=torch.float32)
    in_features = H_current.shape[0]
    eye = torch.eye(in_features, dtype=H_current.dtype, device=solve_device)
    damp = damping
    if damping > 0:
        damp = damping * torch.mean(torch.diag(H_current)).item()
    H_reg = H_current + damp * eye
    transform = torch.linalg.solve(H_reg.T, H_cross.T).T
    return W @ transform


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
    use_hessian=True,
    hessian_cache_dir=None,
    scale_refine=True,
    damping=1e-6,
    reverse=True,
    device,
    save_dir,
    layerwise_dir,
    refine_coeffs_only=False,
    fix_theta=False,
    fix_beta=False,
    optimizer_name='adamw',
    momentum=0.0,
    binary_lr=None,
    continuous_lr=None,
):
    """
    Load a block from precomputed layerwise BQQ patches, then optimize the
    whole block output error with STE-trainable BQQ layers.
    """
    dev = torch.device(device)

    print(f'Loading model: {model_name}')
    model = load_causal_lm(model_name)

    print(f'Caching block {block_idx} I/O ...')
    inputs_cache, targets_cache = cache_block_io(model, block_idx, dataloader, dev)
    print(f'  Cached {len(inputs_cache)} samples')

    block_prefix = get_decoder_block_prefix(model, block_idx)
    block = copy.deepcopy(get_decoder_layer(model, block_idx)).float()
    ensure_layerwise_block_available(
        block,
        model_name=model_name,
        block_idx=block_idx,
        block_prefix=block_prefix,
        layerwise_dir=layerwise_dir,
        bit_width=bit_width,
        group_size=group_size,
        num_steps=num_steps,
        rank_scale=rank_scale,
        seed=seed,
        scale_refine=scale_refine,
        damping=damping,
        dataloader=dataloader,
        device=dev,
        refine_coeffs_only=refine_coeffs_only,
        fix_theta=fix_theta,
        fix_beta=fix_beta,
    )
    del model
    torch.cuda.empty_cache()

    use_trainable_ste = (not refine_coeffs_only) and (binary_lr != 0)

    print(f'Loading layerwise BQQ patches from: {layerwise_dir}')
    linear_names = load_layerwise_block_from_patches(
        block,
        block_idx=block_idx,
        block_prefix=block_prefix,
        layerwise_dir=layerwise_dir,
        bit_width=bit_width,
        trainable_ste=use_trainable_ste,
        optimize_factors=not refine_coeffs_only,
        optimize_coeffs=True,
        optimize_theta=not fix_theta,
        optimize_beta=not fix_beta,
    )

    init_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'\nBlock MSE after loading layerwise BQQ: {init_mse:.6f}')

    optimize_block_params(
        block, inputs_cache, targets_cache,
        epochs=epochs, lr=lr, max_grad_norm=max_grad_norm, device=dev,
        optimizer_name=optimizer_name, momentum=momentum,
        binary_lr=binary_lr, continuous_lr=continuous_lr,
    )

    final_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'Block MSE after blockwise optimization: {final_mse:.6f} (Δ={final_mse - init_mse:+.6f})')

    for lname in linear_names:
        layer = _get_submodule(block, lname)
        if isinstance(layer, TrainableSTEBinaryQuadratic):
            _set_submodule(block, lname, layer.to_binaryquadratic())

    for lname in linear_names:
        layer = _get_submodule(current_block, lname)
        if isinstance(layer, TrainableSTEBinaryQuadratic):
            _set_submodule(current_block, lname, layer.to_binaryquadratic())

    for lname in linear_names:
        layer = _get_submodule(current_block, lname)
        if isinstance(layer, TrainableSTEBinaryQuadratic):
            _set_submodule(current_block, lname, layer.to_binaryquadratic())

    save_path = Path(save_dir) / f'block_{block_idx}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(block.cpu(), save_path, pickle_module=dill)

    print(f'\n=== Block {block_idx} done ===')
    print(f'  Initial loaded MSE: {init_mse:.6f}')
    print(f'  Final MSE:          {final_mse:.6f}')
    print(f'  Saved to: {save_path}')
    return block


# ---------------------------------------------------------------------------
# Progressive block quantization (patch-wise stochastic)
# ---------------------------------------------------------------------------

def _compute_batch_sizes(total: int, num_rounds: int, schedule: str) -> list:
    """Compute patch batch sizes for each round.

    Parameters
    ----------
    total      : total number of patches
    num_rounds : number of quantization rounds
    schedule   : 'linear'    — equal batches (total // num_rounds each)
                 'geometric' — decreasing batches with 2:1 ratio between
                               consecutive rounds (more patches quantized early,
                               finer adjustments later)

    Geometric formula
    -----------------
    Sizes are proportional to 2^(N-1), 2^(N-2), ..., 2^0
    (sum = 2^N - 1), then scaled to sum to `total`.
    Example for N=4, total=1000: [533, 267, 133, 67]
    """
    if num_rounds == 1:
        return [total]

    if schedule == 'linear':
        base = total // num_rounds
        sizes = [base] * num_rounds
        sizes[-1] += total - sum(sizes)   # absorb rounding remainder into last round
    else:  # geometric
        denom = (1 << num_rounds) - 1     # 2^N - 1
        raw = [(1 << (num_rounds - 1 - i)) * total / denom for i in range(num_rounds)]
        sizes = [max(1, round(s)) for s in raw]
        sizes[-1] = max(1, total - sum(sizes[:-1]))   # ensure exact sum

    return sizes


def quantize_block_progressive(
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
    num_rounds=4,
    schedule='geometric',
    device,
    save_dir,
):
    """
    Quantize all Linear weights in a block via progressive patch-wise BQQ.

    Unlike the sequential approach, no Hessian is collected and no ordering
    assumption is made.  All patches across all layers are quantized in
    random batches, with block-output MSE fine-tuning after each batch:

      1. Convert every Linear → PartialBQQLinear (float weights, no BQQ yet).
      2. Pre-compute batch sizes for all rounds using _compute_batch_sizes().
      3. For each round:
         a. Randomly pick the next batch of unquantized patches.
         b. For each selected patch, extract the current float weight values
            and run BQQ optimisation (Y, Z, A solved jointly as usual).
         c. Register the result: Y/Z frozen as buffers; A values become the
            initial a/b/c/d parameters for the fine-tuning step.
         d. Optimise all trainable params (a/b/c/d of quantized patches +
            float_weight of unquantized patches + LayerNorm) to minimise
            block output MSE.
      4. Convert every PartialBQQLinear → BinaryQuadratic and save.

    Gradient routing in PartialBQQLinear.forward() is handled automatically
    by torch.where: quantized positions route to a/b/c/d; unquantized
    positions route to float_weight.

    Parameters
    ----------
    num_rounds : int
        Number of quantization rounds (default 4).
    schedule : str
        'linear'    — equal-sized batches each round.
        'geometric' — decreasing batches with 2:1 ratio between consecutive
                      rounds (more patches early, finer adjustments later).
    """
    dev = torch.device(device)
    device_id = dev.index if dev.type == 'cuda' else 0

    # --- 1. Cache block I/O ---
    print(f'Loading model: {model_name}')
    model = load_causal_lm(model_name)

    print(f'Caching block {block_idx} I/O ...')
    inputs_cache, targets_cache = cache_block_io(model, block_idx, dataloader, dev)
    print(f'  Cached {len(inputs_cache)} samples')

    block = copy.deepcopy(get_decoder_layer(model, block_idx)).float()
    del model
    torch.cuda.empty_cache()

    # --- 2. Convert all Linear → PartialBQQLinear ---
    linear_names = get_quantizable_linears(block)
    n_layers = len(linear_names)
    print(f'\nBlock {block_idx}: {n_layers} quantizable layers → PartialBQQLinear')

    for lname in linear_names:
        lin = _get_submodule(block, lname)
        partial = PartialBQQLinear(
            lin.weight.data,
            lin.bias.data if lin.bias is not None else None,
            group_size=group_size,
            bit_width=bit_width,
        )
        _set_submodule(block, lname, partial)
        print(f'  {lname}: {tuple(lin.weight.shape)} '
              f'→ {partial.row_width}×{partial.col_width} patches')

    # Build flat list of all (layer_name, i, j) patch identifiers
    all_patches = []
    for lname in linear_names:
        layer = _get_submodule(block, lname)
        for i in range(layer.row_width):
            for j in range(layer.col_width):
                all_patches.append((lname, i, j))

    total_patches = len(all_patches)
    batch_sizes = _compute_batch_sizes(total_patches, num_rounds, schedule)
    print(f'\nTotal patches: {total_patches}, rounds: {num_rounds}, '
          f'schedule: {schedule}')
    print(f'Batch sizes per round: {batch_sizes}  (sum={sum(batch_sizes)})')

    init_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'Initial block MSE (pretrained): {init_mse:.6f}')

    # --- 3. Progressive rounds ---
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Shuffle all patches once; then slice off batch_sizes[r] from the front each round
    perm = torch.randperm(total_patches, generator=rng).tolist()
    shuffled = [all_patches[k] for k in perm]
    offset = 0

    for round_idx, batch_size in enumerate(batch_sizes, start=1):
        batch = shuffled[offset:offset + batch_size]
        offset += batch_size

        print(f'\n=== Round {round_idx}/{num_rounds}: quantizing {len(batch)} patches '
              f'({offset}/{total_patches} total) ===')

        # a. Quantize selected patches — group by layer to call quantize_weight_to_bqq
        #    once per layer (uses multiprocessing across all patches of that layer)
        #    rather than once per patch (avoids process-creation overhead × N_patches).
        patches_by_layer = {}
        for lname, i, j in batch:
            patches_by_layer.setdefault(lname, []).append((i, j))

        for lname, ij_list in patches_by_layer.items():
            layer = _get_submodule(block, lname)
            print(f'  [{lname}] BQQ quantize full layer '
                  f'{tuple(layer.float_weight.shape)} → activating {len(ij_list)} patches')
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

        # b. Block MSE before fine-tuning
        mse_before = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f' Block MSE after quantizing:   {mse_before:.6f}')

        # c. Fine-tune all trainable params (a/b/c/d + float_weight + LayerNorm)
        optimize_block_params(
            block, inputs_cache, targets_cache,
            epochs=epochs, lr=lr, max_grad_norm=max_grad_norm, device=dev,
        )

        mse_after = compute_block_mse(block, inputs_cache, targets_cache, dev)
        print(f'  Block MSE after fine-tuning:  {mse_after:.6f} '
              f'(Δ={mse_after - mse_before:+.6f})')

    # --- 4. Convert PartialBQQLinear → BinaryQuadratic ---
    print('\nConverting PartialBQQLinear → BinaryQuadratic ...')
    for lname in linear_names:
        layer = _get_submodule(block, lname)
        assert isinstance(layer, PartialBQQLinear), f"Expected PartialBQQLinear at {lname}"
        _set_submodule(block, lname, layer.to_binaryquadratic())
        print(f'  {lname}: converted')

    # --- 5. Save ---
    save_path = Path(save_dir) / f'block_{block_idx}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(block.cpu(), save_path, pickle_module=dill)

    final_mse = compute_block_mse(block, inputs_cache, targets_cache, dev)
    print(f'\n=== Block {block_idx} done ===')
    print(f'  Initial MSE: {init_mse:.6f}')
    print(f'  Final MSE:   {final_mse:.6f}')
    print(f'  Saved to: {save_path}')

    return block


def quantize_block_progressive_closed_form(
    model_name,
    block_idx,
    dataloader,
    *,
    bit_width,
    group_size,
    num_steps,
    rank_scale,
    seed,
    damping=1e-6,
    epochs=0,
    lr=1e-5,
    max_grad_norm=1.0,
    optimizer_name='adamw',
    momentum=0.0,
    binary_lr=None,
    continuous_lr=None,
    tune_after_each_layer=False,
    use_closed_form=True,
    fix_theta=False,
    fix_beta=False,
    device,
    save_dir,
):
    """Front-to-back layer quantization with closed-form continuous recentering.

    For each layer we observe the original layer input X and the current input X'
    after previously quantized layers, solve

        min_W' ||W X - W' X'||_F^2

    as

        W' = W (X X'^T) (X' X'^T + λI)^-1

    and then quantize that W' with Hessian-aware BQQ using H = X' X'^T.

    If tune_after_each_layer is True, run block-level gradient descent after each
    layer quantization for the requested number of epochs.

    If use_closed_form is False, skip the cross-Hessian recentering step and quantize
    the current float layer directly with H = X'^T X'. This is the layer-tune mode.
    """
    dev = torch.device(device)
    device_id = dev.index if dev.type == 'cuda' else 0

    print(f'Loading model: {model_name}')
    model = load_causal_lm(model_name)

    print(f'Caching block {block_idx} I/O ...')
    inputs_cache, targets_cache = cache_block_io(model, block_idx, dataloader, dev)
    print(f'  Cached {len(inputs_cache)} samples')

    original_block = copy.deepcopy(get_decoder_layer(model, block_idx)).float()
    current_block = copy.deepcopy(get_decoder_layer(model, block_idx)).float()
    del model
    torch.cuda.empty_cache()

    linear_names = get_quantizable_linears(current_block)
    print(f'\nBlock {block_idx}: {len(linear_names)} quantizable layers -> sequential closed-form progressive')

    init_mse = compute_block_mse(current_block, inputs_cache, targets_cache, dev)
    print(f'Initial block MSE (pretrained): {init_mse:.6f}')

    for layer_idx, lname in enumerate(linear_names, start=1):
        print(f'\n=== Layer {layer_idx}/{len(linear_names)}: {lname} ===')
        if use_closed_form:
            H_cross, H_current = collect_cross_hessians_for_linear(
                original_block, current_block, lname, inputs_cache, dev
            )
            if H_cross is None or H_current is None:
                raise RuntimeError(f'Failed to collect cross Hessians for {lname}')

            original_linear = _get_submodule(original_block, lname)
            quant_weight = solve_closed_form_weight(
                original_linear.weight.data,
                H_cross,
                H_current,
                damping=damping,
            )
            bias = original_linear.bias.data.clone().float() if original_linear.bias is not None else None
        else:
            H_current = collect_single_hessian(current_block, lname, inputs_cache, dev)
            if H_current is None:
                raise RuntimeError(f'Failed to collect current Hessian for {lname}')
            current_linear = _get_submodule(current_block, lname)
            quant_weight = current_linear.weight.data.detach().float().clone()
            bias = current_linear.bias.data.clone().float() if current_linear.bias is not None else None

        A, Y, Z = quantize_weight_to_bqq(
            quant_weight.cpu(),
            bit_width=bit_width,
            group_size=group_size,
            num_steps=num_steps,
            rank_scale=rank_scale,
            seed=seed,
            device_id=device_id,
            H=H_current.cpu(),
            scale_refine=True,
            damping=damping,
        )
        replace_linear_in_block(
            current_block,
            lname,
            A,
            Y,
            Z,
            bias=bias,
            trainable_ste=(binary_lr != 0),
            optimize_factors=True,
            optimize_coeffs=True,
            optimize_theta=not fix_theta,
            optimize_beta=not fix_beta,
        )

        cur_mse = compute_block_mse(current_block, inputs_cache, targets_cache, dev)
        print(f'  Block MSE after quantizing {lname}: {cur_mse:.6f}')

        if tune_after_each_layer and epochs > 0:
            optimize_block_params(
                current_block,
                inputs_cache,
                targets_cache,
                epochs=epochs,
                lr=lr,
                max_grad_norm=max_grad_norm,
                device=dev,
                optimizer_name=optimizer_name,
                momentum=momentum,
                binary_lr=binary_lr,
                continuous_lr=continuous_lr,
            )
            tuned_mse = compute_block_mse(current_block, inputs_cache, targets_cache, dev)
            print(f'  Block MSE after tuning {lname}:     {tuned_mse:.6f} (Δ={tuned_mse - cur_mse:+.6f})')

    save_path = Path(save_dir) / f'block_{block_idx}.pth'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(current_block.cpu(), save_path, pickle_module=dill)

    final_mse = compute_block_mse(current_block, inputs_cache, targets_cache, dev)
    print(f'\n=== Block {block_idx} done ===')
    print(f'  Initial MSE: {init_mse:.6f}')
    print(f'  Final MSE:   {final_mse:.6f}')
    print(f'  Saved to: {save_path}')

    return current_block


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
                        choices=['wikitext2', 'ptb', 'c4', 'redpajama1t', 'slimpajama'])
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)

    # Optimization
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--binary_lr', type=float, default=None,
                        help='Learning rate for trainable binary factors Y/Z during blockwise tuning')
    parser.add_argument('--continuous_lr', type=float, default=None,
                        help='Learning rate for continuous parameters during blockwise tuning')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'],
                        help='Optimizer used for blockwise tuning')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum used when --optimizer sgd')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')

    # Mode
    parser.add_argument('--progressive', action='store_true',
                        help='Use a progressive quantization mode instead of loading layerwise results')
    parser.add_argument('--progressive_mode', type=str, default='patch',
                        choices=['patch', 'closed-form-layer', 'layer-tune'],
                        help='Progressive mode: patch = existing patch-wise alternating quantize/tune; '
                             'closed-form-layer = front-to-back layer quantization with closed-form recentering; '
                             'layer-tune = front-to-back layer quantization of current float layers with block tuning after each layer')
    parser.add_argument('--layerwise_dir', type=str, default=None,
                        help='Directory containing layerwise quantization outputs. Defaults to the standard layerwise output path.')
    parser.add_argument('--refine_coeffs_only', action='store_true',
                        help='Freeze BQQ binary factors during blockwise optimization')
    parser.add_argument('--fix_theta', action='store_true',
                        help='Keep STE thresholds fixed at 0.5 during blockwise optimization')
    parser.add_argument('--fix_beta', action='store_true',
                        help='Keep STE sigmoid temperatures fixed during blockwise optimization')

    # Legacy/on-the-fly quantization args (used by progressive mode)
    parser.add_argument('--no_hessian', action='store_true',
                        help='[progressive legacy] Disable Hessian-aware quantization')
    parser.add_argument('--no_reverse', action='store_true',
                        help='[progressive legacy] Process layers front-to-back instead of back-to-front')
    parser.add_argument('--hessian_cache_dir', type=str, default=None,
                        help='[legacy] Directory to cache/load Hessian matrices')
    parser.add_argument('--no_scale_refine', action='store_true',
                        help='[legacy] Disable Hessian-aware scale refinement')
    parser.add_argument('--damping', type=float, default=1e-6)

    # Progressive mode only
    parser.add_argument('--num_rounds', type=int, default=4,
                        help='[progressive] Number of quantization rounds (default 4)')
    parser.add_argument('--schedule', type=str, default='geometric',
                        choices=['linear', 'geometric'],
                        help='[progressive] Batch size schedule: '
                             '"linear" = equal batches each round; '
                             '"geometric" = 2:1 decreasing ratio '
                             '(more patches early, finer adjustments later) '
                             '(default: geometric)')

    # Device / output
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--assemble_full_model', dest='assemble_full_model', action='store_true', default=True,
                        help='Assemble a full quantized model after blockwise quantization (default: enabled)')
    parser.add_argument('--no_assemble_full_model', dest='assemble_full_model', action='store_false',
                        help='Skip full-model assembly after blockwise quantization')
    parser.add_argument('--assembled_output_dir', type=str, default=None,
                        help='Output directory for the assembled full model')

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

    common_kwargs = dict(
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
        binary_lr=args.binary_lr,
        continuous_lr=args.continuous_lr,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        save_dir=args.save_dir,
    )

    layerwise_dir = args.layerwise_dir
    if layerwise_dir is None:
        layerwise_dir = default_compressed_data_dir(args.model_name, args.group_size, args.num_steps)

    if args.progressive:
        if args.progressive_mode == 'patch':
            quantize_block_progressive(**common_kwargs,
                                       num_rounds=args.num_rounds,
                                       schedule=args.schedule)
        else:
            quantize_block_progressive_closed_form(
                model_name=args.model_name,
                block_idx=args.block_idx,
                dataloader=train_loader,
                bit_width=args.bit_width,
                group_size=args.group_size,
                num_steps=args.num_steps,
                rank_scale=args.rank_scale,
                seed=args.seed,
                damping=args.damping,
                epochs=args.epochs,
                lr=args.lr,
                max_grad_norm=args.max_grad_norm,
                optimizer_name=args.optimizer,
                momentum=args.momentum,
                binary_lr=args.binary_lr,
                continuous_lr=args.continuous_lr,
                tune_after_each_layer=(args.progressive_mode == 'layer-tune'),
                use_closed_form=(args.progressive_mode == 'closed-form-layer'),
                fix_theta=args.fix_theta,
                fix_beta=args.fix_beta,
                device=args.device,
                save_dir=args.save_dir,
            )
    else:
        quantize_block(
            **common_kwargs,
            use_hessian=not args.no_hessian,
            hessian_cache_dir=args.hessian_cache_dir,
            scale_refine=not args.no_scale_refine,
            damping=args.damping,
            reverse=not args.no_reverse,
            layerwise_dir=layerwise_dir,
            refine_coeffs_only=args.refine_coeffs_only,
            fix_theta=args.fix_theta,
            fix_beta=args.fix_beta,
        )

    if args.assemble_full_model:
        assemble_from_blocks(
            model_name=args.model_name,
            block_dir=args.save_dir,
            bit_width=args.bit_width,
            group_size=args.group_size,
            output_dir=args.assembled_output_dir,
        )


if __name__ == '__main__':
    main()
