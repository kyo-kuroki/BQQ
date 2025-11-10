import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, SwinForImageClassification, DeiTForImageClassificationWithTeacher, DeiTForImageClassification
import os
import sys
from tqdm import tqdm
import pandas as pd
import timm
from pathlib import Path
from PIL import Image
import numpy as np
import math
from build_dataset import get_imagenet
from build_model import get_model
import binary_quadratic_network as BQN
import argparse



    

def get_matrices(patch_list, bit_width):
    # 特定のビット重みのみ
    row_width = max(patch['patch_row'] for patch in patch_list)+1
    col_width = max(patch['patch_col'] for patch in patch_list)+1
    m, l = patch_list[0]['mat1'].shape
    l, n = patch_list[0]['mat2'].shape
    A = torch.zeros((bit_width, row_width, col_width, 4))
    Y = torch.zeros((bit_width, row_width, col_width, m, l))
    Z = torch.zeros((bit_width, row_width, col_width, l, n))
    for patch in patch_list:
        i, j = patch['patch_row'], patch['patch_col']
        a, y, z, k = patch['coeff'], patch['mat1'], patch['mat2'], patch['bit_idx']

        
        if k >= bit_width:
            continue  # bit_idx が範囲外の場合はスキップ
        else:
            A[k,i,j] = a
            Y[k,i,j] = y
            Z[k,i,j] = z
        
    return A, Y, Z









# 精度テスト
def test_model_accuracy(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, torch.Tensor):
                pass
            else:
                outputs = outputs.logits
            # _, predicted = torch.max(outputs, 1)
            predicted = outputs.argmax(dim=1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy




def quantize_to_n_bits(matrix, n_bits):
    num_levels = 2**n_bits
    """
    行列を量子化し、最適な量子化を選ぶ関数。

    Parameters:
        matrix (np.ndarray): 量子化する行列
        num_levels (int): 量子化のビット深度（例えば256なら8ビット量子化）

    Returns:
        quantized_matrix (np.ndarray): 最適な量子化結果
        best_scale (float): 最適なスケール
        min_error (float): 最小誤差 (RMSE)
    """
    # 標準偏差と範囲を計算
    mean = (matrix).mean()
    min = (matrix).min()
    max = (matrix).max()
    
    # 量子化スケールを標準偏差の倍率としていくつか試す
    min_error = float('inf')
    best_scale = None
    quantized_matrix = None
    
    for range_max in (torch.linspace(mean, max, 100)):
        for range_min in torch.linspace(min, mean, 100):
        
            # 行列を量子化
            quantized = torch.clamp(matrix, range_min, range_max)  # 範囲外をクリップ
            if range_max == range_min:buffer=1e-8
            else:buffer=0
            quantized = torch.round(
                (quantized - range_min) / (range_max - range_min + buffer) * (num_levels - 1)
            ) / (num_levels - 1) * (range_max - range_min) + range_min
        
            # MSEを計算
            error = (((matrix - quantized) ** 2).mean())
            
            # 最適スケールを更新
            if error < min_error:
                min_error = error
                best_scale = (range_min, range_max)
                quantized_matrix = quantized

    return quantized_matrix


def channel_wise_quantize_to_n_bits(tensor, n_bits):
    matrix_list = []
    for i in range(tensor.shape[0]):
        matrix_list.append(quantize_to_n_bits(tensor[i], n_bits))
    return (torch.stack(matrix_list, axis=0))


def get_parser():
    parser = argparse.ArgumentParser(description="Arguments for evaluate_models")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name passed to evaluate_models")

    parser.add_argument("--bit_width", type=int, default=4,
                        help="Bit width for quantization")

    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers")

    parser.add_argument("--save_dir", type=str, default=os.getcwd(),
                        help="Directory to save results")

    parser.add_argument("--rank_scale", type=float, default=1.0,
                        help="Rank scale parameter")

    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    parser.add_argument("--Nstep", type=int, default=50000,
                        help="Number of steps for quantization")
    
    parser.add_argument("--group_size", type=int, default=128,
                        help="Group size for quantization")
    

    return parser.parse_args()


def main(args):    # モデルの保存先の設定
    weights_dir = f'./bqq_compressed_data/{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale'

    model = get_model(model_abbreviation=args.model_name)

    # ロード
    model = BQN.replace_linear_with_bqq(model, weights_dir=weights_dir, bit_width=args.bit_width)
    torch.save(model, os.path.dirname(__file__)+f'/quantized_bqq_model/{args.model_name}-{args.bit_width}bit-{args.group_size}gs-{args.Nstep}step.pth')


if __name__ == "__main__":
    args = get_parser()
    main(args)