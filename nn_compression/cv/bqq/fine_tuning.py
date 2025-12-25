import torch
import sys
sys.path.append('./../../../../BQQ')
sys.path.append('./../utils')
from build_dataset import get_imagenet
import os
from utils import test_model_accuracy
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

def fine_tuning(args):
    # ---------------------------
    # 1. Load model
    # ---------------------------
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    else:
        if args.fine_tuned_model:
            model_path = os.path.join(
                os.path.dirname(__file__),
                f'quantized_bqq_model/{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step-bqq-finetuned.pth'
            )
        else:
            model_path = os.path.join(
            os.path.dirname(__file__),
            f'quantized_bqq_model/{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step-bqq.pth'
        )

        model = torch.load(model_path, map_location=args.device, weights_only=False)
    model.to(args.device)
    model.train()

    # ---------------------------
    # 2. Datasets
    # ---------------------------
    trainloader, testloader = get_imagenet(args.model_name, calib_batchsize=args.batch_size, num_workers=args.num_workers, data_path=args.data_path, seed=0)

    # ---------------------------
    # 3. Loss & Optimizer
    # （量子化後の微調整なので、LR は小さめに）
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr or 1e-5)

    # ---------------------------
    # 4. Train Loop
    # ---------------------------
    for epoch in range(args.epochs or 3):
        loop = tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        running_loss = 0

        for images, labels in loop:
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {running_loss / len(trainloader):.4f}")

    # ---------------------------
    # 5. Save fine-tuned model
    # ---------------------------
    if args.model_path is not None:
        save_path = os.path.splitext(args.model_path)[0] + f'-{args.epochs}epochs-fine_tuned.pth'
    else:
        save_path = os.path.join(
            os.path.dirname(__file__), 
            f'quantized_bqq_model/{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.epochs}epochs-finetuned.pth'
        )
    torch.save(model, save_path)

    print(f"Fine-tuned model saved to: {save_path}")

    return model


# 実行
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for evaluation")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name passed to evaluation")
    parser.add_argument("--bit_width", type=int, default=4,
                        help="Bit width for quantization")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Group size for quantization")
    parser.add_argument("--Nstep", type=int, default=50000,
                        help="Number of steps for quantization")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for fine-tuning")  
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for fine-tuning")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default='/ldisk/Shared/Datasets/ILSVRC/ILSVRC2012/',
                        help="Path to ImageNet dataset")
    parser.add_argument("--fine_tuned_model", action='store_true',
                        help="Whether to load an already fine-tuned model")
    parser.add_argument("--model_path", type=str, default=None,)
    args = parser.parse_args()
    model = fine_tuning(args)