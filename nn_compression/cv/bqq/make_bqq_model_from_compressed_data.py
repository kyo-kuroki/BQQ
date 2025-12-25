import torch
import os
import sys
from tqdm import tqdm
sys.path.append('./../utils')
import pandas as pd
from build_dataset import get_imagenet
from build_model import get_model
import argparse
import binary_quadratic_network

def find_sets(obj, path='model'):
    if isinstance(obj, set):
        print(f"Found set at {path}: {obj}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_sets(v, f'{path}[{i}]')
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_sets(v, f'{path}[{k}]')
    elif hasattr(obj, '__dict__'):
        for k, v in vars(obj).items():
            find_sets(v, f'{path}.{k}')

def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for evaluate_models")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name passed to evaluate_models")

    parser.add_argument("--bit_width", type=int, default=4,
                        help="Bit width for quantization")
    
    parser.add_argument("--compressed_data_dir", type=str, default='',)

    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers")

    parser.add_argument("--rank_scale", type=float, default=1.0,
                        help="Rank scale parameter")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    parser.add_argument("--Nstep", type=int, default=50000,
                        help="Number of steps for quantization")
    
    parser.add_argument("--group_size", type=int, default=32,
                        help="Group size for quantization")
    
    parser.add_argument("--device", type=str, default='cuda:0')
    

    return parser.parse_args()


def save_bqq_model(model_name, compressed_data_dir, bit_width, group_size, Nstep, device):
    model = get_model(model_name)
    model = binary_quadratic_network.replace_linear_with_bqq(model, weights_dir=compressed_data_dir, bit_width=bit_width, device=device)

    for name, param in model.named_parameters():
        print(f"{name:40s} | shape: {tuple(param.shape)} | learnable: {param.requires_grad}")
    os.makedirs(os.path.dirname(__file__)+f'/quantized_bqq_model', exist_ok=True)
    torch.save(model, os.path.dirname(__file__)+f'/quantized_bqq_model/{model_name}-{bit_width}bit-{group_size}gs-{Nstep}step-bqq.pth')

if __name__ == '__main__':
    args = get_parser()
    save_bqq_model(model_name=args.model_name,
                   compressed_data_dir=args.compressed_data_dir,
                   bit_width=args.bit_width,
                   group_size=args.group_size,
                   Nstep=args.Nstep,
                   device=args.device)