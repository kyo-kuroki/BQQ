"""
Full-model continuous-param fine-tuning for assembled BQQ models.

Loads an assembled BQQ model (.pth), freezes binary factors (already buffers),
and fine-tunes only continuous params (a/b/c/d, LayerNorm, bias) using
causal LM loss over calibration data.

Usage:
    python continuous_finetune.py \
        --model_name meta-llama/Meta-Llama-3-8B \
        --bqq_model_path /path/to/model.pth \
        --output_path /path/to/output.pth \
        --epochs 5 \
        --lr 1e-4 \
        --continuous_lr 1e-6
"""
import argparse
import os
import sys

import dill
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from neural_network_compression.lm.src.datautils import get_loaders


def get_continuous_params(model):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    binary = [(n, p) for n, p in named if n.lower().endswith('y_fp') or n.lower().endswith('z_fp')]
    continuous = [(n, p) for n, p in named if not (n.lower().endswith('y_fp') or n.lower().endswith('z_fp'))]
    return continuous, binary


def finetune(args):
    device = torch.device(args.device)

    print(f'Loading BQQ model from {args.bqq_model_path}')
    model = torch.load(args.bqq_model_path, map_location='cpu', weights_only=False, pickle_module=dill)
    model = model.to(device)
    model.train()

    continuous_params, binary_params = get_continuous_params(model)
    n_continuous = sum(p.numel() for _, p in continuous_params)
    n_binary = sum(p.numel() for _, p in binary_params)
    print(f'Continuous params: {len(continuous_params)} tensors, {n_continuous:,} elements')
    print(f'Binary (y_fp/z_fp) params: {len(binary_params)} tensors, {n_binary:,} elements (frozen via lr=0)')

    continuous_lr = args.continuous_lr if args.continuous_lr is not None else args.lr
    param_groups = [{'params': [p for _, p in continuous_params], 'lr': continuous_lr}]
    if binary_params:
        param_groups.append({'params': [p for _, p in binary_params], 'lr': 0.0})

    optimizer = torch.optim.AdamW(param_groups)

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

    best_loss = float('inf')
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch[0].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for _, p in continuous_params], args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f'Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f'  Loss did not improve (best={best_loss:.4f}), reverting')
            model.load_state_dict(best_state)
            model.to(device)
            break

    model.load_state_dict(best_state)
    model.cpu()

    import pathlib
    pathlib.Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.output_path, pickle_module=dill)
    print(f'Saved fine-tuned model to {args.output_path}')


def main():
    parser = argparse.ArgumentParser(description='Full-model continuous-param fine-tuning for BQQ models')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--bqq_model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--continuous_lr', type=float, default=None)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='slimpajama')
    parser.add_argument('--nsamples', type=int, default=1024)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    finetune(args)


if __name__ == '__main__':
    main()
