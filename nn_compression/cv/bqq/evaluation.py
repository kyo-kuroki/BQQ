import torch
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
BQQ_ROOT = CV_DIR.parent.parent
UTILS_DIR = CV_DIR / "utils"

for path in (BQQ_ROOT, UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_dataset import get_imagenet
import os
from utils import test_model_accuracy

# モデルのテストプロセス
def evaluation(args):
    flag = '-finetuned' if args.fine_tuned else ''
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    else:
        model_path = SCRIPT_DIR / "quantized_bqq_model" / f"{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step-bqq{flag}.pth"
        model = torch.load(model_path, map_location=args.device, weights_only=False)
    trainloader, testloader = get_imagenet(args.model_name, data_path=args.data_path)
    imagenet_accuracy = test_model_accuracy(model, testloader, args.device)
    print(f"ImageNet Accuracy: {imagenet_accuracy:.2f}%")
    return imagenet_accuracy

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
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to ImageNet. If omitted, use IMAGENET_DIR or IMAGENET_ROOT.")
    parser.add_argument("--fine_tuned", action='store_true',)
    parser.add_argument("--model_path", type=str, default=None,)
    args = parser.parse_args()
    evaluation(args)
