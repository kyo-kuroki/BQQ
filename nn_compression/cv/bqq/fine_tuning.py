"""
Fine-tune a quantized BQQ vision model.

Modes:
  - CE only (default): standard classification cross-entropy loss
  - CE + KL distillation (--teacher_model_name): adds KL divergence loss
    against a teacher model's logits
  - KL only (--teacher_model_name + --ce_alpha 0): pure distillation

Usage:
  # CE only
  python fine_tuning.py --model_name deit-s --model_path model.pth

  # CE + KL distillation
  python fine_tuning.py --model_name deit-s --model_path model.pth \
    --teacher_model_name deit-s --kl_alpha 1.0 --kl_temperature 2.0

  # KL only
  python fine_tuning.py --model_name deit-s --model_path model.pth \
    --teacher_model_name deit-s --ce_alpha 0 --kl_alpha 1.0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
UTILS_DIR = CV_DIR / "utils"

for path in (str(CV_DIR.parent.parent), str(UTILS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_dataset import get_imagenet
from build_model import get_model
from utils import test_model_accuracy


def fine_tuning(args):
    # --- 1. Load quantized model ---
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    else:
        model_path = SCRIPT_DIR / "quantized_bqq_model" / \
            f"{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step-bqq.pth"
        model = torch.load(model_path, map_location=args.device, weights_only=False)
    model.to(args.device)
    model.train()

    # --- 2. Teacher model (optional) ---
    teacher = None
    if args.teacher_model_name is not None:
        print(f"Loading teacher model: {args.teacher_model_name}")
        teacher = get_model(args.teacher_model_name)
        teacher.to(args.device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    # --- 3. Datasets ---
    trainloader, testloader = get_imagenet(
        args.model_name, calib_batchsize=args.batch_size,
        num_workers=args.num_workers, data_path=args.data_path, seed=0,
    )

    # --- 4. Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # --- 5. Train Loop ---
    for epoch in range(args.epochs):
        loop = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        running_loss = 0.0

        for images, labels in loop:
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits

            # CE loss
            loss = args.ce_alpha * criterion(outputs, labels)

            # KL distillation loss
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(images)
                    if not isinstance(teacher_outputs, torch.Tensor):
                        teacher_outputs = teacher_outputs.logits

                T = args.kl_temperature
                student_logp = F.log_softmax(outputs / T, dim=-1)
                teacher_p = F.softmax(teacher_outputs / T, dim=-1)
                kl_loss = F.kl_div(student_logp, teacher_p, reduction="batchmean") * (T * T)
                loss = loss + args.kl_alpha * kl_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    # --- 6. Save ---
    if args.model_path is not None:
        save_path = os.path.splitext(args.model_path)[0] + f'-{args.epochs}epochs-finetuned.pth'
    else:
        save_path = SCRIPT_DIR / "quantized_bqq_model" / \
            f"{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.epochs}epochs-finetuned.pth"
    torch.save(model, save_path)
    print(f"Fine-tuned model saved to: {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a quantized BQQ vision model")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--bit_width", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--Nstep", type=int, default=50000)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to ImageNet")
    # Distillation
    parser.add_argument("--teacher_model_name", type=str, default=None,
                        help="Teacher model for KL distillation (omit for CE only)")
    parser.add_argument("--ce_alpha", type=float, default=1.0,
                        help="Weight for CE loss (0 = KL only)")
    parser.add_argument("--kl_alpha", type=float, default=1.0,
                        help="Weight for KL distillation loss")
    parser.add_argument("--kl_temperature", type=float, default=2.0,
                        help="Temperature for KL distillation")

    args = parser.parse_args()
    fine_tuning(args)
