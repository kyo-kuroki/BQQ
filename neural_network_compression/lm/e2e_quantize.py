"""
End-to-end (e2e) progressive BQQ quantization with KL-divergence fine-tuning.

Loop until the whole model is quantized:
  1. Quantize one unit -- default: one transformer block (all its Linear
     layers in parallel across every GPU); --quant_unit layerwise quantizes
     one Linear at a time -- using Hessians collected from the CURRENT
     (partially quantized + fine-tuned) model
  2. Replace the unit's Linears with frozen BQQ deployment modules
     (BinaryQuadratic / IncoherentBinaryQuadratic / DCTBinaryQuadratic)
  3. Fine-tune the remaining unquantized parameters to minimize
     KL(teacher || student) against the original fp model on calibration
     data, so the quantization error is absorbed by the not-yet-quantized
     layers before they are quantized themselves

Per-unit quantization artifacts are cached under --work_dir, and the student
model is checkpointed every --checkpoint_every units, so an interrupted run
resumes where it left off (use --fresh to start over).

Usage:
  python e2e_quantize.py --model_name Qwen/Qwen2.5-1.5B \
    --bit_width 2 --group_size 64 --num_steps 20000 \
    --dataset slimpajama --nsamples 256 --seqlen 2048 \
    --transform rht --diag_power 0.75 --bqq_opt_mode activation-aware
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import dill
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from .blockwise_quant import (
        _get_submodule,
        get_quantizable_linears,
        quantize_weight_to_bqq,
        replace_linear_in_block,
    )
    from .src.build_bqq_model import BinaryQuadratic, TrainableSTEBinaryQuadratic
    from .src.compressed_data import default_quantized_model_dir, model_basename
    from .src.datautils import get_loaders
    from .src.model_loader import (
        get_decoder_block_prefix,
        get_decoder_layer,
        get_decoder_num_layers,
        load_causal_lm,
    )
except ImportError:
    from neural_network_compression.lm.blockwise_quant import (
        _get_submodule,
        get_quantizable_linears,
        quantize_weight_to_bqq,
        replace_linear_in_block,
    )
    from neural_network_compression.lm.src.build_bqq_model import BinaryQuadratic, TrainableSTEBinaryQuadratic
    from neural_network_compression.lm.src.compressed_data import default_quantized_model_dir, model_basename
    from neural_network_compression.lm.src.datautils import get_loaders
    from neural_network_compression.lm.src.model_loader import (
        get_decoder_block_prefix,
        get_decoder_layer,
        get_decoder_num_layers,
        load_causal_lm,
    )

try:
    from .src.model_loader import resolve_decoder_layers
except ImportError:
    from neural_network_compression.lm.src.model_loader import resolve_decoder_layers

LM_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Simple pipeline model parallelism (decoder blocks spread over all GPUs)
# ---------------------------------------------------------------------------

def _tensors_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_tensors_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_tensors_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _tensors_to_device(v, device) for k, v in obj.items()}
    return obj


class PipelineShard:
    """Spread a decoder LM over several GPUs (naive pipeline parallelism).

    Decoder blocks are split evenly across `gpu_ids`; everything else
    (embeddings, rotary, final norm, lm_head) lives on the primary (first)
    GPU. Forward pre-hooks move activations to each stage's device and the
    last block's output back to the primary device. `.to()` is differentiable,
    so autograd crosses devices and KL fine-tuning works unchanged. With a
    single GPU this degenerates to plain model.to(gpu).

    shard() / unshard() are idempotent bracketing calls: unshard removes the
    hooks and returns the whole model to CPU (so it can be pickled or handed
    to the quantization workers' GPUs).
    """

    def __init__(self, model: nn.Module, gpu_ids: List[int]):
        self.model = model
        self.gpu_ids = list(gpu_ids)
        self.handles: list = []
        self.primary = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')

    def _device_for_block(self, block_idx: int, n_blocks: int) -> torch.device:
        stage = block_idx * len(self.gpu_ids) // max(1, n_blocks)
        return torch.device(f'cuda:{self.gpu_ids[stage]}')

    def shard(self):
        if not self.gpu_ids:
            return self
        model = self.model
        layer_path, layers = resolve_decoder_layers(model)
        n_blocks = len(layers)

        # Blocks to their stage devices
        for i, block in enumerate(layers):
            block.to(self._device_for_block(i, n_blocks))

        # Everything outside the decoder blocks to the primary device
        prefix = layer_path + '.'
        for name, param in model.named_parameters():
            if not name.startswith(prefix):
                param.data = param.data.to(self.primary)
        for name, buf in list(model.named_buffers()):
            if name.startswith(prefix):
                continue
            parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
            key = name.rsplit('.', 1)[-1]
            parent._buffers[key] = buf.to(self.primary)

        # Hooks: move args/kwargs to each stage, and the final block's output
        # back to the primary device for the norm / lm_head.
        def make_pre_hook(device):
            def _pre(module, args, kwargs):
                return _tensors_to_device(args, device), _tensors_to_device(kwargs, device)
            return _pre

        for i, block in enumerate(layers):
            dev = self._device_for_block(i, n_blocks)
            self.handles.append(block.register_forward_pre_hook(make_pre_hook(dev), with_kwargs=True))

        def _out_hook(module, args, kwargs, output):
            return _tensors_to_device(output, self.primary)

        self.handles.append(layers[-1].register_forward_hook(_out_hook, with_kwargs=True))
        return self

    def unshard(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def build_units(model: nn.Module, quant_unit: str):
    """Enumerate quantization units front-to-back.

    Returns a list of (block_idx, [linear_names], label).
    blockwise: one unit = every Linear in one transformer block
    layerwise: one unit = a single Linear layer
    """
    units = []
    n_blocks = get_decoder_num_layers(model)
    for b in range(n_blocks):
        block = get_decoder_layer(model, b)
        prefix = get_decoder_block_prefix(model, b)
        names = get_quantizable_linears(block)
        if quant_unit == 'blockwise':
            units.append((b, names, prefix))
        else:
            for name in names:
                units.append((b, [name], f"{prefix}.{name}"))
    return units


# ---------------------------------------------------------------------------
# Hessian collection on the current (partially quantized) student
# ---------------------------------------------------------------------------

class _EarlyExit(Exception):
    pass


def collect_unit_hessians(
    model: nn.Module,
    calibration_loader,
    block_idx: int,
    linear_names: List[str],
    input_device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Accumulate H = X^T X for the requested Linears of block `block_idx` on
    the CURRENT model (earlier blocks may already be BQQ modules), with an
    early-exit hook after the block so the rest of the model never runs.

    The model must already be placed on its device(s) by the caller (e.g.
    sharded across GPUs); `input_device` is where the input ids go.
    Returns dict keyed by the Linear's name within the block.
    """
    block = get_decoder_layer(model, block_idx)

    H: Dict[str, Optional[torch.Tensor]] = {n: None for n in linear_names}
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

    for name in linear_names:
        module = _get_submodule(block, name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} in block {block_idx} is {type(module).__name__}, expected nn.Linear")
        handles.append(module.register_forward_hook(make_hook(name)))

    def _exit_hook(module, inp, out):
        raise _EarlyExit()

    handles.append(block.register_forward_hook(_exit_hook))

    model.eval()

    with torch.no_grad():
        for batch in tqdm(calibration_loader, desc=f"Collecting Hessians (block {block_idx})"):
            ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            ids = ids.to(input_device)
            try:
                model(ids)
            except _EarlyExit:
                pass
            except Exception:
                pass

    for h in handles:
        h.remove()

    return {k: v.cpu() for k, v in H.items() if v is not None}


# ---------------------------------------------------------------------------
# Parallel quantization worker (top-level for mp.spawn pickling)
# ---------------------------------------------------------------------------

def _e2e_quantize_worker(rank: int, gpu_tasks: list, common: dict):
    tasks = gpu_tasks[rank]
    gpu_ids = common['gpu_ids']
    gpu_id = gpu_ids[rank % len(gpu_ids)] if gpu_ids else 0

    for task in tasks:
        out_path = Path(task['out_path'])
        label = task['label']
        if out_path.exists():
            print(f"[GPU{gpu_id}/W{rank}] {label}: artifact exists, skip")
            continue

        weight = task['weight']
        H = task.get('H')
        print(f"[GPU{gpu_id}/W{rank}] [{task['display_idx']}/{task['n_total']}] {label} {tuple(weight.shape)}")

        A, Y, Z, SU, SV, tdesc = quantize_weight_to_bqq(
            weight,
            bit_width=common['bit_width'],
            group_size=common['group_size'],
            num_steps=common['num_steps'],
            rank_scale=common['rank_scale'],
            seed=common['seed'],
            device_id=gpu_id,
            H=H,
            scale_refine=common['scale_refine'],
            damping=common['damping'],
            use_multibqq=common['use_multibqq'],
            compensation_mode=common['compensation_mode'],
            bqq_opt_mode=common['bqq_opt_mode'],
            diag_power=common['diag_power'],
            transform=common['transform'],
            ldlq_act_order=common['ldlq_act_order'],
            ldlq_act_order_score=common['ldlq_act_order_score'],
            rank_alloc_mode=common['rank_alloc_mode'],
            ste_refine_steps=common['ste_refine_steps'],
            ste_refine_lr=common['ste_refine_lr'],
            ste_refine_weight_decay=common['ste_refine_weight_decay'],
            ste_refine_binary_lr=common['ste_refine_binary_lr'],
            ste_refine_continuous_lr=common['ste_refine_continuous_lr'],
            ste_refine_log_interval=common['ste_refine_log_interval'],
        )

        artifact = {
            'A': A.cpu() if torch.is_tensor(A) else A,
            'Y': Y.cpu() if torch.is_tensor(Y) else Y,
            'Z': Z.cpu() if torch.is_tensor(Z) else Z,
            'SU': SU.cpu() if torch.is_tensor(SU) else SU,
            'SV': SV.cpu() if torch.is_tensor(SV) else SV,
            'transform_desc': tdesc,
        }
        tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
        torch.save(artifact, tmp_path)
        os.replace(tmp_path, out_path)
        print(f"[GPU{gpu_id}/W{rank}] Saved: {out_path}")


def quantize_unit(student, block_idx: int, linear_names: List[str],
                  H_dict: Dict[str, torch.Tensor], unit_dir: Path, common: dict):
    """Quantize the unit's Linears (multi-GPU via mp.spawn) and replace them
    in the student with frozen BQQ deployment modules."""
    block = get_decoder_layer(student, block_idx)
    prefix = get_decoder_block_prefix(student, block_idx)
    unit_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, name in enumerate(linear_names):
        lin = _get_submodule(block, name)
        if not isinstance(lin, nn.Linear):
            print(f"  {prefix}.{name}: already replaced, skip")
            continue
        H = H_dict.get(name)
        if H is None:
            print(f"  WARNING: no Hessian collected for {prefix}.{name}; quantizing without H")
        tasks.append({
            'linear_name': name,
            'label': f"{prefix}.{name}",
            'display_idx': i + 1,
            'n_total': len(linear_names),
            'out_path': str(unit_dir / f"{prefix}.{name}.pth"),
            'weight': lin.weight.detach().cpu().float(),
            'H': H,
        })

    todo = [t for t in tasks if not Path(t['out_path']).exists()]
    if todo:
        for t in todo:
            t['weight'].share_memory_()
            if t['H'] is not None:
                t['H'].share_memory_()
        n_slots = max(1, max(1, len(common['gpu_ids'])) * common['workers_per_gpu'])
        n_workers = min(len(todo), n_slots)
        gpu_tasks: List[List[dict]] = [[] for _ in range(n_workers)]
        for i, t in enumerate(todo):
            gpu_tasks[i % n_workers].append(t)
        if n_workers == 1:
            _e2e_quantize_worker(0, gpu_tasks, common)
        else:
            mp.spawn(_e2e_quantize_worker, args=(gpu_tasks, common), nprocs=n_workers, join=True)

    for t in tasks:
        art = torch.load(t['out_path'], weights_only=False, map_location='cpu')
        lin = _get_submodule(block, t['linear_name'])
        bias = lin.bias.data.clone().float() if getattr(lin, 'bias', None) is not None else None
        replace_linear_in_block(
            block, t['linear_name'], art['A'], art['Y'], art['Z'], bias=bias,
            trainable_ste=False,
            SU=art['SU'], SV=art['SV'], transform_desc=art['transform_desc'],
        )
        print(f"  Replaced {t['label']} -> BQQ")


# ---------------------------------------------------------------------------
# KL fine-tuning of the remaining unquantized parameters
# ---------------------------------------------------------------------------

def set_trainable(model: nn.Module, *, train_embeddings: bool = False,
                  train_quantized_continuous: bool = False) -> int:
    """Mark unquantized parameters trainable and freeze quantized ones.

    Everything trains except (a) parameters inside BQQ modules (Y/Z are
    buffers, so with train_quantized_continuous=True only their continuous
    a/b/c/d/bias train) and (b) the input embeddings unless requested (note:
    with tied word embeddings this also freezes lm_head).
    """
    for p in model.parameters():
        p.requires_grad_(True)
    if not train_quantized_continuous:
        for m in model.modules():
            if isinstance(m, (BinaryQuadratic, TrainableSTEBinaryQuadratic)):
                for p in m.parameters():
                    p.requires_grad_(False)
    emb = model.get_input_embeddings()
    if emb is not None and not train_embeddings:
        for p in emb.parameters():
            p.requires_grad_(False)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")
    return n_train


def _kl_loss(student_logits, teacher_logits, temperature: float):
    T = temperature
    s = F.log_softmax(student_logits.float() / T, dim=-1)
    t = F.softmax(teacher_logits.float() / T, dim=-1)
    B, S, V = s.shape
    return F.kl_div(s.view(B * S, V), t.view(B * S, V), reduction='batchmean') * (T * T)


@torch.no_grad()
def eval_kl(student, teacher, eval_batches, *, student_device, teacher_device, temperature: float) -> float:
    """Mean KL(teacher || student) over a fixed set of calibration batches.

    Both models must already be placed (sharded) on their devices;
    student_device / teacher_device are the respective INPUT devices.
    """
    student.eval()
    teacher.eval()
    total = 0.0
    for ids in eval_batches:
        t_logits = teacher(ids.to(teacher_device)).logits
        s_logits = student(ids.to(student_device)).logits
        total += float(_kl_loss(s_logits, t_logits.to(s_logits.device), temperature))
    return total / max(1, len(eval_batches))


def finetune_kl(
    student, teacher, train_loader, *,
    student_device, teacher_device,
    epochs: int, max_steps: int,
    lr: float, weight_decay: float,
    grad_accum: int, max_grad_norm: float,
    kl_temperature: float, kl_alpha: float, ce_alpha: float,
    log_interval: int,
):
    """Fine-tune the trainable (unquantized) parameters of `student` to match
    the teacher's output distribution on the calibration data.

    Both models must already be placed (sharded) on their devices;
    student_device / teacher_device are the respective INPUT devices.
    """
    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        print("  No trainable parameters left; skipping fine-tune")
        return
    student.train()
    teacher.eval()

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    opt.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    running = []
    done = False
    for epoch in range(epochs):
        if done:
            break
        for batch in train_loader:
            ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            ids = ids.to(student_device)
            with torch.no_grad():
                t_logits = teacher(ids.to(teacher_device)).logits.to(student_device)
            out = student(input_ids=ids, labels=ids if ce_alpha > 0 else None)
            loss = kl_alpha * _kl_loss(out.logits, t_logits, kl_temperature)
            if ce_alpha > 0:
                loss = loss + ce_alpha * out.loss
            (loss / grad_accum).backward()
            running.append(float(loss.detach()))
            micro += 1
            if micro % grad_accum == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_interval == 0:
                    avg = sum(running[-grad_accum * log_interval:]) / len(running[-grad_accum * log_interval:])
                    print(f"  [ft] epoch {epoch + 1}/{epochs} step {step}: loss {avg:.6f}")
                if max_steps > 0 and step >= max_steps:
                    done = True
                    break
    # Flush any leftover accumulated gradients
    if micro % grad_accum != 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
    student.eval()
    if running:
        tail = running[-min(len(running), grad_accum * log_interval):]
        print(f"  [ft] done: {step} optimizer step(s), final loss {sum(tail)/len(tail):.6f}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _atomic_torch_save(obj, path: Path, **kwargs):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp, **kwargs)
    os.replace(tmp, path)


def e2e_quantize(args):
    if torch.cuda.is_available():
        if args.gpu_ids:
            gpu_ids = [int(g) for g in str(args.gpu_ids).split(',') if g != '']
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
    # Student pipeline runs front-to-back over gpu_ids; the teacher uses the
    # reversed order so the two models' heavy stages land on different GPUs.
    student_device = torch.device(f'cuda:{gpu_ids[0]}') if gpu_ids else torch.device('cpu')
    teacher_device = torch.device(f'cuda:{gpu_ids[-1]}') if gpu_ids else torch.device('cpu')

    work_dir = Path(args.work_dir) if args.work_dir else (
        LM_DIR / 'e2e_output' / model_basename(args.model_name)
        / f"{args.bit_width}bit-{args.group_size}gs-{args.quant_unit}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / 'student_checkpoint.pth'
    progress_path = work_dir / 'progress.pt'
    log_path = work_dir / 'e2e_log.csv'

    config_key = dict(model_name=args.model_name, quant_unit=args.quant_unit,
                      bit_width=args.bit_width, group_size=args.group_size)

    # Resume or fresh start
    start_unit = 0
    student = None
    if not args.fresh and ckpt_path.exists() and progress_path.exists():
        progress = torch.load(progress_path, weights_only=False)
        if progress.get('config') != config_key:
            raise RuntimeError(
                f"Existing progress in {work_dir} was made with a different config "
                f"({progress.get('config')} vs {config_key}). Use --fresh or a different --work_dir.")
        start_unit = int(progress['completed_units'])
        print(f"Resuming from checkpoint: {start_unit} unit(s) already completed")
        student = torch.load(ckpt_path, weights_only=False, map_location='cpu', pickle_module=dill)

    print(f"Loading teacher model: {args.model_name}")
    teacher = load_causal_lm(args.model_name)
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    if student is None:
        print(f"Loading student model: {args.model_name}")
        student = load_causal_lm(args.model_name)
    student = student.float()
    student.config.use_cache = False
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        student.enable_input_require_grads()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=args.seqlen, model=args.model_name, tokenizer=tokenizer,
    )

    # Fixed evaluation batches so before/after KL numbers are comparable
    eval_batches = []
    if args.eval_batches > 0:
        for batch in train_loader:
            ids = batch[0] if isinstance(batch, (list, tuple)) else batch
            eval_batches.append(ids.cpu())
            if len(eval_batches) >= args.eval_batches:
                break

    # Enumerate units from the teacher: it always has the original fp
    # structure, so unit indices stay stable across resumes (the student's
    # completed blocks are already BQQ modules and would enumerate differently).
    units = build_units(teacher, args.quant_unit)
    n_units = len(units)
    print(f"{n_units} quantization unit(s) [{args.quant_unit}], "
          f"GPUs {gpu_ids} x {args.workers_per_gpu} worker(s); "
          f"student sharded over {gpu_ids} (inputs on {student_device}), "
          f"teacher sharded over {list(reversed(gpu_ids))} (inputs on {teacher_device})")

    # Both models are spread over every GPU (pipeline sharding), so large
    # models fit; the teacher is sharded in reverse GPU order to balance the
    # per-GPU load against the student.
    student_shard = PipelineShard(student, gpu_ids)
    teacher_shard = PipelineShard(teacher, list(reversed(gpu_ids)))

    common = dict(
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        bit_width=args.bit_width,
        group_size=args.group_size,
        num_steps=args.num_steps,
        rank_scale=args.rank_scale,
        seed=args.seed,
        scale_refine=not args.no_scale_refine,
        damping=args.damping,
        use_multibqq=args.use_multibqq,
        compensation_mode=args.compensation_mode,
        bqq_opt_mode=args.bqq_opt_mode,
        diag_power=args.diag_power,
        transform=args.transform,
        ldlq_act_order=args.ldlq_act_order,
        ldlq_act_order_score=args.ldlq_act_order_score,
        rank_alloc_mode=args.rank_alloc_mode,
        ste_refine_steps=args.ste_refine_steps,
        ste_refine_lr=args.ste_refine_lr,
        ste_refine_weight_decay=args.ste_refine_weight_decay,
        ste_refine_binary_lr=args.ste_refine_binary_lr,
        ste_refine_continuous_lr=args.ste_refine_continuous_lr,
        ste_refine_log_interval=args.ste_refine_log_interval,
    )

    end_unit = n_units if args.max_units <= 0 else min(n_units, args.max_units)
    for k in range(start_unit, end_unit):
        block_idx, linear_names, label = units[k]
        unit_dir = work_dir / 'units' / f'unit{k:04d}'
        print(f"\n=== Unit {k + 1}/{n_units}: {label} ({len(linear_names)} Linear(s)) ===")

        # 1. Hessians from the CURRENT student (sharded across all GPUs)
        student_shard.shard()
        H_dict = collect_unit_hessians(student, train_loader, block_idx, linear_names, student_device)

        # 2. Parallel quantization (both models off-GPU so workers get the memory)
        student_shard.unshard()
        quantize_unit(student, block_idx, linear_names, H_dict, unit_dir, common)
        del H_dict

        # 3. Fine-tune remaining unquantized parameters against the teacher
        set_trainable(student, train_embeddings=args.train_embeddings,
                      train_quantized_continuous=args.train_quantized_continuous)

        student_shard.shard()
        teacher_shard.shard()

        kl_before = kl_after = float('nan')
        if eval_batches:
            kl_before = eval_kl(student, teacher, eval_batches, student_device=student_device,
                                teacher_device=teacher_device, temperature=args.kl_temperature)
            print(f"  KL vs teacher before fine-tune: {kl_before:.6f}")

        if not args.no_finetune:
            finetune_kl(
                student, teacher, train_loader,
                student_device=student_device, teacher_device=teacher_device,
                epochs=args.ft_epochs, max_steps=args.ft_steps,
                lr=args.ft_lr, weight_decay=args.ft_weight_decay,
                grad_accum=args.grad_accum, max_grad_norm=args.max_grad_norm,
                kl_temperature=args.kl_temperature, kl_alpha=args.kl_alpha,
                ce_alpha=args.ce_alpha, log_interval=args.ft_log_interval,
            )
            if eval_batches:
                kl_after = eval_kl(student, teacher, eval_batches, student_device=student_device,
                                   teacher_device=teacher_device, temperature=args.kl_temperature)
                print(f"  KL vs teacher after fine-tune:  {kl_after:.6f}")

        # Back to CPU: frees the GPUs for the next unit's quantization workers
        # and keeps the checkpoint hook-free / CPU-resident
        student_shard.unshard()
        teacher_shard.unshard()

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['unit', 'label', 'kl_before', 'kl_after'])
            writer.writerow([k, label, kl_before, kl_after])

        # 4. Checkpoint for resume
        if args.checkpoint_every > 0 and ((k + 1) % args.checkpoint_every == 0 or k + 1 == n_units):
            print(f"  Saving checkpoint after unit {k + 1} ...")
            _atomic_torch_save(student, ckpt_path, pickle_module=dill)
            _atomic_torch_save({'completed_units': k + 1, 'n_units': n_units, 'config': config_key},
                               progress_path)

    # Final save
    output_path = Path(args.output_path) if args.output_path else (
        default_quantized_model_dir(args.model_name)
        / f"{model_basename(args.model_name)}-{args.bit_width}bit-{args.group_size}gs-e2e.pth"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    student.cpu()
    _atomic_torch_save(student, output_path, pickle_module=dill)
    print(f"\nDone. Saved e2e-quantized model to {output_path}")
    print(f"Per-unit log: {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='End-to-end progressive BQQ quantization with KL fine-tuning')

    # Model / mode
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name (e.g. Qwen/Qwen2.5-1.5B)')
    parser.add_argument('--quant_unit', type=str, default='blockwise',
                        choices=['blockwise', 'layerwise'],
                        help='Quantization unit per iteration: blockwise = one transformer '
                             'block (Linears quantized in parallel on all GPUs, default); '
                             'layerwise = one Linear layer at a time')

    # BQQ params
    parser.add_argument('--bit_width', type=int, default=2)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--rank_scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--damping', type=float, default=1e-6)
    parser.add_argument('--no_scale_refine', action='store_true',
                        help='Disable Hessian-aware scale refinement')
    parser.add_argument('--use_multibqq', dest='use_multibqq', action='store_true', default=True)
    parser.add_argument('--no_use_multibqq', dest='use_multibqq', action='store_false')
    parser.add_argument('--compensation_mode', type=str, default='ldlq',
                        choices=['gptq', 'ldlq', 'none'])
    parser.add_argument('--bqq_opt_mode', type=str, default='plain',
                        choices=['plain', 'activation-aware'],
                        help="BQQ block objective: 'plain' = unweighted reconstruction; "
                             "'activation-aware' = full-matrix Hessian (fullchol) weighted output error")
    parser.add_argument('--diag_power', type=float, default=1.0,
                        help='Metric tempering exponent alpha: quantize with H^alpha')
    parser.add_argument('--transform', type=str, default='none',
                        choices=['none', 'rht', 'ht', 'dct'],
                        help='Input/output transform before quantization')
    parser.add_argument('--ldlq_act_order', action='store_true', default=False)
    parser.add_argument('--ldlq_act_order_score', type=str, default='maxdiag',
                        choices=['maxdiag', 'trace', 'static'])
    parser.add_argument('--rank_alloc_mode', type=str, default='none', choices=['none', 'pivot-log'])
    parser.add_argument('--ste_refine_steps', type=int, default=0)
    parser.add_argument('--ste_refine_lr', type=float, default=1e-3)
    parser.add_argument('--ste_refine_weight_decay', type=float, default=0.0)
    parser.add_argument('--ste_refine_binary_lr', type=float, default=None)
    parser.add_argument('--ste_refine_continuous_lr', type=float, default=None)
    parser.add_argument('--ste_refine_log_interval', type=int, default=20)

    # Dataset (calibration + fine-tuning)
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4', 'redpajama1t', 'slimpajama'])
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)

    # Parallel quantization
    parser.add_argument('--workers_per_gpu', type=int, default=2,
                        help='Quantization worker processes per GPU during the parallel phase')

    # KL fine-tuning
    parser.add_argument('--no_finetune', action='store_true',
                        help='Skip fine-tuning between units (quantize-only ablation)')
    parser.add_argument('--ft_epochs', type=int, default=1,
                        help='Fine-tuning epochs over the calibration set per unit')
    parser.add_argument('--ft_steps', type=int, default=0,
                        help='Cap on optimizer steps per unit fine-tune (0 = no cap)')
    parser.add_argument('--ft_lr', type=float, default=1e-5)
    parser.add_argument('--ft_weight_decay', type=float, default=0.0)
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation micro-batches per optimizer step')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--kl_temperature', type=float, default=1.0)
    parser.add_argument('--kl_alpha', type=float, default=1.0)
    parser.add_argument('--ce_alpha', type=float, default=0.0,
                        help='Weight for next-token CE loss on top of KL (0 = pure KL)')
    parser.add_argument('--ft_log_interval', type=int, default=10)
    parser.add_argument('--train_embeddings', action='store_true',
                        help='Also fine-tune the input embeddings (frozen by default; '
                             'note tied word embeddings share lm_head)')
    parser.add_argument('--train_quantized_continuous', action='store_true',
                        help='Also fine-tune the continuous coefficients (a/b/c/d/bias) of '
                             'already-quantized BQQ modules')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--eval_batches', type=int, default=4,
                        help='Fixed calibration batches for the per-unit before/after KL report (0 disables)')

    # Devices / output / resume
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU ids to use (default: all visible GPUs). Both the '
                             'student and the teacher are pipeline-sharded across these GPUs so '
                             'large models fit; quantization workers use the same GPUs.')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Directory for per-unit artifacts, checkpoint and log '
                             '(default: lm/e2e_output/{model}/{bit}bit-{gs}gs-{unit})')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Final model path (default: quantized_models/{model}-{bit}bit-{gs}gs-e2e.pth)')
    parser.add_argument('--checkpoint_every', type=int, default=1,
                        help='Save a resume checkpoint every N units (0 disables)')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignore any existing checkpoint/progress in work_dir and start over')
    parser.add_argument('--max_units', type=int, default=0,
                        help='Debug: stop after this many units (0 = quantize the whole model)')

    args = parser.parse_args()
    e2e_quantize(args)


if __name__ == '__main__':
    main()
