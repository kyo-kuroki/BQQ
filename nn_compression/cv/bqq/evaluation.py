import torch
import sys
sys.path.append('./../../../../BQQ')
sys.path.append('./../utils')
from build_dataset import get_imagenet
import os
from utils import test_model_accuracy

# モデルのテストプロセス
def evaluation(args):
    flag = '-finetuned' if args.fine_tuned else ''
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    else:
        model = torch.load(os.path.dirname(__file__)+f'/quantized_bqq_model/{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step-bqq{flag}.pth', map_location=args.device, weights_only=False)
    trainloader, testloader = get_imagenet(args.model_name)
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
    parser.add_argument("--fine_tuned", action='store_true',)
    parser.add_argument("--model_path", type=str, default=None,)
    args = parser.parse_args()
    evaluation(args)