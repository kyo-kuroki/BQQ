"""
Evaluate ImageNet top-1 accuracy of a quantized BQQ vision model.

Parallels lm/src/evaluation.py (the model-evaluation layer), but reports
classification accuracy instead of perplexity.

Usage:
  python evaluation.py --model_name deit-s \
    --model_path quantized_models/deit-s/deit-s-2bit-32gs-20000step-bqq.pth \
    --data_path /path/to/imagenet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Ensure repository root and bqqkernel are importable (needed for torch.load of BQQ models)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
_BQQKERNEL_DIR = os.path.join(_REPO_ROOT, 'neural_network_compression', 'bqqkernel')
for _path in (_REPO_ROOT, _BQQKERNEL_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import neural_network_compression.bqqkernel.bqq_modules as bqq_modules  # noqa: F401

try:
    from .compressed_data import default_quantized_model_dir, model_basename
    from .datautils import get_imagenet
except ImportError:
    from compressed_data import default_quantized_model_dir, model_basename
    from datautils import get_imagenet


@torch.no_grad()
def test_model_accuracy(model, dataloader, device):
    """Top-1 ImageNet accuracy (%) for a model on the given dataloader."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs.logits
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def evaluation(args):
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        flag = '-finetuned' if getattr(args, 'fine_tuned', False) else ''
        model_dir = default_quantized_model_dir(args.model_name)
        model_id = model_basename(args.model_name)
        model_path = model_dir / (
            f"{model_id}-{args.bit_width}bit-{args.group_size}gs-{args.num_steps}step-bqq{flag}.pth"
        )

    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=args.device, weights_only=False)

    _, val_loader = get_imagenet(args.model_name, data_path=args.data_path)
    accuracy = test_model_accuracy(model, val_loader, args.device)
    print(f"ImageNet top-1 accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a quantized BQQ vision model")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model abbreviation (deit-s, vit-b, swin-t, ...)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved BQQ .pth model. If omitted, a default path is built.")
    parser.add_argument("--bit_width", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to ImageNet. Falls back to IMAGENET_DIR env var.")
    parser.add_argument("--fine_tuned", action="store_true")
    args = parser.parse_args()
    evaluation(args)


if __name__ == "__main__":
    main()
