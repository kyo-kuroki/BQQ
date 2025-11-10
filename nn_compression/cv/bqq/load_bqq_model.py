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
 




class BinaryMatrixProductModel(nn.Module):
    def __init__(self, Y, Z, A):
        super().__init__()
        self.bit_width, self.row_width, self.col_width, self.y_row, self.inter_dimension =  Y.shape
        _, _, _, _, self.z_col =  Z.shape

        # バイナリパラメータ：{-1, +1} ランダム初期化
        self.register_buffer("Y", (Y > 0.5))
        self.register_buffer("Z", (Z > 0.5))
        # スケーリング係数：形状 (bit_width, row_width, col_width)
        self.a = nn.Parameter(A[...,0])
        self.b = nn.Parameter(A[...,1])
        self.c = nn.Parameter(A[...,2])
        self.d = nn.Parameter(A[...,3].sum(dim=0))  # オフセット項


    def forward(self, X):
        """
        X: (j*m, k*n)
        Returns: (j*m, ?)
        """
        # (p, j, k, m, l) @ (p, j, k, l, n) → (p, j, k, m, n)
        W_core = torch.matmul(self.Y.to(torch.int32), self.Z.to(torch.int32))  # Y @ Z

        # 各項のスケーリングとブロードキャスト
        a_scaled = self.a[..., None, None]  # (p, j, k, 1, 1)
        b_scaled = self.b[..., None, None]  # (p, j, k, 1, 1)
        c_scaled = self.c[..., None, None]  # (p, j, k, 1, 1)
        d_scaled = self.d[..., None, None]  # (p, j, k, 1, 1)

        # Y.sum over l → (p, j, k, m)
        Y_sum = self.Y.sum(dim=-1, keepdim=True)  # → (p, j, k, m, 1)
        Z_sum = self.Z.sum(dim=-2, keepdim=True)  # → (p, j, k, 1, n)

        # 総和項
        W = a_scaled * W_core + b_scaled * Y_sum + c_scaled * Z_sum 

        # p に沿って加算 → (j, k, m, n)
        W_sum = W.sum(dim=0) + d_scaled

        # reshape → (j*m, k*n)
        W_reshaped = W_sum.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)

        # 出力
        return  X@W_reshaped 
    
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

def aggregate_matrices(patch_list, output_shape, bit_width, qtz_type='qq'):
    result_matrix = torch.zeros(output_shape)
    # 特定のビット重みのみ
    for patch in patch_list:
        i, j = patch['patch_row'], patch['patch_col']
        a, y, z, k = patch['coeff'], patch['mat1'], patch['mat2'], patch['bit_idx']
        
        if k >= bit_width:
            continue  # bit_idx が範囲外の場合はスキップ
        
        if qtz_type == 'soaq':
            term1 = (a[0] * y + a[1] * (1 - y)) @ z
            term2 = (a[2] * y + a[3] * (1 - y)) @ (1 - z)
            patch_result = term1 + term2
        elif qtz_type == 'qq':
            patch_result = a[0]*y@z + a[1]*y.sum(axis=1, keepdim=True) + a[2]*z.sum(axis=0, keepdim=True) + a[3]

        
        # 結果を i, j の位置に格納
        row_start, row_end = i * patch_result.shape[0], (i + 1) * patch_result.shape[0]
        col_start, col_end = j * patch_result.shape[1], (j + 1) * patch_result.shape[1]
        result_matrix[row_start:row_end, col_start:col_end] += patch_result

    return result_matrix


def load_qq_model(model, weights_dir, bit_width, model_type='deit', qtz_type='qq', emb_qtz=True, head_qtz=True):
    # 元モデルのロード
    for name, param in model.named_parameters():

        # 変換されている重みかどうか(QQで変換されていない重みはスキップ)
        deit_bool = 'weight' in name and not 'norm' in name and not 'head' in name and not 'emb' in name and 'deit' in model_type ## DeiTの場合
        vit_bool = 'weight' in name and not 'norm' in name and not 'head' in name and not 'emb' in name and 'vit' in model_type ## ViTの場合
        swin_bool = 'encoder' in name and 'weight' in name and not 'norm' in name and 'swin' in model_type ## Swin-Transformerの場合
        resnet_bool = 'layer' in name and 'weight' and not 'bn' in name and not 'downsample' in name and 'resnet' in model_type #not 'downsample.1' in name: ### CNN系の場合 'bn'と'downsample.1'はバッチ正規化層
        if (deit_bool or swin_bool or resnet_bool or vit_bool):
            weight_list = []
            # ディレクトリにあるファイルを逐次的に読み込み
            for file in os.listdir(weights_dir):
                # 層の名前が読み込むファイルと一致している場合に処理を行う(ファイル名_row{i}_col{j}.pthとなっているものだけを対象とする)
                if file.endswith('.pth') and (name in file) and ('row' in file):
                    weight_path = os.path.join(weights_dir, file) # フルパスを作成(fileはfile名のみ)
                    weight_list += torch.load(weight_path, weights_only=False, map_location=torch.device('cpu'))
            original_shape = param.shape
            # 2次元行列に変換したときの形状を取得
            conversion_shape = param.reshape(param.shape[0], -1).shape
            # 2次元行列を量子化したものを読み込む
            reconst = aggregate_matrices(weight_list, conversion_shape, bit_width=bit_width, qtz_type=qtz_type)
            # 量子化した行列を元の形に戻す
            reconst = reconst.reshape(original_shape)
            param.data.copy_(reconst) 

        elif 'emb' in name and 'weight' in name and not 'norm' in name and emb_qtz: # Swinのembeddingだけ畳み込みなので通常の量子化の方がいい
            if not 'swin' in weights_dir:
                original_shape = param.shape
                conversion_shape = param.reshape(param.shape[0], -1).shape
                emb_list = []
                dir = os.path.dirname(weights_dir)
                # if 'deit-s' in weights_dir:
                #     emb = r'deit-s-embeddings-50000step-384gs'
                # elif 'deit-b' in weights_dir:
                #     emb = r'deit-b-embeddings-100000step-768gs'
                # elif 'swin-t' in weights_dir:
                #     emb = r'swin-t-embeddings-50000step-48gs'
                # elif 'swin-s' in weights_dir:
                #     emb = r'swin-s-embeddings-50000step-48gs'
                # else: emb = weights_dir
                emb = weights_dir
                emb_dir = os.path.join(dir, emb)


                # ディレクトリにあるファイルを逐次的に読み込み
                for file in os.listdir(emb_dir):
                    # 層の名前が読み込むファイルと一致している場合に処理を行う(ファイル名_row{i}_col{j}.pthとなっているものだけを対象とする)
                    if 'row' in file and file.endswith(".pth") and (name in file) : # 重みをロードする
                        emb_path = os.path.join(emb_dir, file) # フルパスを作成(fileはfile名のみ)
                        emb_list += torch.load(emb_path, weights_only=False, map_location=torch.device('cpu'))
                # 保存したものを読み込む
                reconst = aggregate_matrices(emb_list, conversion_shape, bit_width=bit_width, qtz_type=qtz_type)
                reconst_original_shape = reconst.reshape(original_shape)
                param.data.copy_(reconst_original_shape)
            else:
                print('CNN Embedding: Uniform Quantization')
                param.data.copy_(channel_wise_quantize_to_n_bits(param, n_bits=bit_width))

        elif ('head' in name or 'classifier' in name) and 'weight' in name and not 'norm' in name and head_qtz:
            original_shape = param.shape
            conversion_shape = param.reshape(param.shape[0], -1).shape
            head_list = []
            dir = os.path.dirname(weights_dir)
            if 'deit-s' in weights_dir:
                head = r'deit-s-head-50000step-100gs'
            elif 'deit-b' in weights_dir:
                head = r'deit-b-head-50000step-100gs'
            elif 'swin-t' in weights_dir:
                head = r'swin-t-head-50000step-100gs'
            elif 'swin-s' in weights_dir:
                head = r'swin-s-head-50000step-100gs'
            else: head = weights_dir
            # head = weights_dir
            head_dir = os.path.join(dir, head)

            # ディレクトリにあるファイルを逐次的に読み込み
            for file in os.listdir(head_dir):
                # 層の名前が読み込むファイルと一致している場合に処理を行う(ファイル名_row{i}_col{j}.pthとなっているものだけを対象とする。仮にこれがないと、'_'がついていない元々の重みが読み込まれる)
                if 'row' in file and file.endswith(".pth") and (name in file) : # アテンションの重みをロードする
                    head_path = os.path.join(head_dir, file) # フルパスを作成(fileはfile名のみ)
                    head_list += torch.load(head_path, weights_only=False, map_location=torch.device('cpu'))
            
            # 保存したものを読み込む
            reconst = aggregate_matrices(head_list, conversion_shape, bit_width=bit_width, qtz_type='qq')
            reconst_original_shape = reconst.reshape(original_shape)
            param.data.copy_(reconst_original_shape)

    return model




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


if __name__ == "__main__":
    data_list = []
    models_name = ['vit-s', 'vit-b', 'deit-s', 'deit-b', 'swin-t', 'swin-s']
    step = 50000
    gs = 384
    device_id = 0
    for model_name in ['swin-s']:
        for bit_width in range(2, 5):
            # 事前学習済みモデルのロード
            model = get_model(model_abbreviation=model_name)
            # # 重みが保存されているディレクトリ
            weights_dir = r'/work2/k-kuroki/quadratic_quantization/nn_compression/cv/qq_compressed_data/swin-s-100000step-96gs'
            # weights_dir = "/work2/k-kuroki/quadratic_quantization/nn_compression/cv/qq_compressed_data/{0}-{1}step-{2}gs".format(model_name, step, gs)
            # # ロード
            model_type=model_name.split("-")[0]
            model = load_qq_model(model, weights_dir=weights_dir, bit_width=bit_width, model_type=model_type, qtz_type='qq', emb_qtz=True, head_qtz=True)

            # ImageNetのデータローダーを取得
            train_loader, test_loader = get_imagenet(model_name=model_name, val_batchsize=128, num_workers=10)

            # デバイス (CPU or GPU)
            device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

            # 精度評価
            model.eval()
            model.to(device)
            accuracy = test_model_accuracy(model, test_loader, device)
            print(f"Accuracy:{accuracy}%")
            data_list.append({'model':model_name, 'bit_width':bit_width, 'accuracy':accuracy})
            df = pd.DataFrame(data_list)
            # df.to_csv(f'/work2/k-kuroki/quadratic_quantization/nn_compression/cv/qq_acc/{model_name}-data-free-results.csv')