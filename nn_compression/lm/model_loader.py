import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM
from datautils import get_loaders
from parsers import parse_args
import os
from tqdm import tqdm
import pandas as pd



def get_llama(model):
    '''
    input: llama_model_path
    output: llama model
    '''
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048

    return model

def aggregate_matrices(patch_list, output_shape, bit_width):
    '''
    patach_list: list of dictionary
    output_shape: tuple
    bit_width: int
    '''
    result_matrix = torch.zeros(output_shape)
    # 特定のビット重みのみ
    for patch in patch_list:
        i, j = patch['patch_row'], patch['patch_col']
        a, y, z, k = patch['coeff'], patch['mat1'], patch['mat2'], patch['bit_idx']
        
        if k >= bit_width:
            continue  # bit_idx が範囲外の場合はスキップ
        
        patch_result = a[0]*y@z + a[1]*y.sum(axis=1, keepdim=True) + a[2]*z.sum(axis=0, keepdim=True) + a[3]

        # 結果を i, j の位置に格納
        row_start, row_end = i * patch_result.shape[0], (i + 1) * patch_result.shape[0]
        col_start, col_end = j * patch_result.shape[1], (j + 1) * patch_result.shape[1]
        result_matrix[row_start:row_end, col_start:col_end] += patch_result

    return result_matrix

def get_quantized_model(model, weights_dir, bit_width, weight_qtz=True, emb_qtz=True, head_qtz=True):

    # 元モデルのロード
    for name, param in tqdm(model.named_parameters()):

        # 変換されている重みかどうか(QQで変換されていない重みはスキップ)
        weight_bool = 'weight' in name and not 'norm' in name and not 'head' in name and not 'emb' in name and weight_qtz 
        head_bool = 'head' in name and not 'norm' in name and head_qtz
        emb_bool = 'emb' in name and not 'norm' in name and emb_qtz

        if weight_bool or head_bool or emb_bool:
            print(name, param.shape)
            weight_list = []
            # ディレクトリにあるファイルを逐次的に読み込み
            for file in os.listdir(weights_dir):
                # 層の名前が読み込むファイルと一致している場合に処理を行う(ファイル名_row{i}_col{j}.pthとなっているものだけを対象とする)
                if name in file and not (file==name+'.pth'): # アテンションの重みをロードする
                    weight_path = os.path.join(weights_dir, file) # フルパスを作成(fileはfile名のみ)
                    weight_list += torch.load(weight_path, weights_only=False, map_location=torch.device('cpu'))

            original_shape = param.shape
            # 2次元行列に変換したときの形状を取得
            conversion_shape = param.reshape(param.shape[0], -1).shape
            # 2次元行列を量子化したものを読み込む
            reconst = aggregate_matrices(weight_list, conversion_shape, bit_width=bit_width)
            # 量子化した行列を元の形に戻す
            reconst = reconst.reshape(original_shape)
            param.data.copy_(reconst) 

    return model