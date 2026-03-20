import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.multiprocessing import Process, Queue
from torch.multiprocessing import set_start_method
import torch.nn.functional as F
from transformers import AutoImageProcessor, SwinForImageClassification
import os
from pathlib import Path
import sys
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
BQQ_ROOT = CV_DIR.parent.parent
UTILS_DIR = CV_DIR / "utils"

for path in (BQQ_ROOT, UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# quantizerをインポート
from quantizer import BinaryQuadraticQuantization2 as BQQ
import queue
from multiprocessing import Process, Queue, current_process
import pandas as pd
import timm
from PIL import Image
import time
import copy
import random
from build_dataset import get_imagenet
from build_model import get_model
# ------------------------
# 1. 中間入出力の収集
# ------------------------
def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 再現性を高める（ただし速度が落ちることも）


def find_module_by_param_name(model, param_name):
    module_name = ".".join(param_name.split(".")[:-1])
    return dict(model.named_modules()).get(module_name, None)


def get_calibration(model, param_name, data_num, dataloader, device=torch.device("cuda")):
    model.to(device)
    model.eval()

    target_module = find_module_by_param_name(model, param_name)
    if target_module is None:
        raise ValueError(f"Module for param_name {param_name} not found.")

    inputs_collected = []
    outputs_collected = []

    def hook_fn(module, input, output):
        inputs_collected.append(input[0].detach().cpu())  # GPUメモリから退避
        outputs_collected.append(output.detach().cpu())  # GPUメモリから退避

    handle = target_module.register_forward_hook(hook_fn)

    total = 0

    for i, (images, _) in enumerate(dataloader):
        if total >= data_num:
            break
        images = images.to(device)
        with torch.no_grad():
            model(images)  # この間に計算を行う
        total += images.size(0)

    handle.remove()

    # キャッシュを収集した後、不要なGPUメモリを解放
    torch.cuda.empty_cache()

    # CPU上に転送してからメモリを削減
    input_tensor = torch.cat(inputs_collected, dim=0)[:data_num]
    output_tensor = torch.cat(outputs_collected, dim=0)[:data_num]
    
    # 最後にGPUメモリから完全に解放する
    del inputs_collected
    del outputs_collected

    # 入力と出力はGPUに転送
    return input_tensor.to(device), output_tensor.to(device)


# ------------------------
# 2. normalization parameter の最適化
# ------------------------




def optimize_gamma_beta(x_q, y_fp, eps=1e-5, num_iters=10):
    """
    x_q: (B, L, D) 量子化入力
    y_fp: (B, L, D) FP出力
    """
    B, L, D = x_q.shape

    # (B*L, D) にまとめる
    x_q_flat = x_q.view(-1, D)
    y_fp_flat = y_fp.view(-1, D)

    # レイヤーノーマライズ (各行ごとにmean/stdを計算)
    mean = x_q_flat.mean(dim=-1, keepdim=True)  # (B*L, 1)
    var = x_q_flat.var(dim=-1, unbiased=False, keepdim=True)  # (B*L, 1)
    std = (var + eps).sqrt()
    x_norm = (x_q_flat - mean) / std  # (B*L, D)

    # γ, βを初期化
    gamma = torch.ones(D, device=x_q.device)
    beta = torch.zeros(D, device=x_q.device)

    for _ in range(num_iters):
        # Step A: β固定 → γ更新
        numerator = (x_norm * (y_fp_flat - beta)).sum(dim=0)  # (D,)
        denominator = (x_norm * x_norm).sum(dim=0) + 1e-8     # (D,)
        gamma = numerator / denominator

        # Step B: γ固定 → β更新
        beta = (y_fp_flat - gamma * x_norm).mean(dim=0)       # (D,)

    return gamma, beta

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


def verify_layernorm(module, input_tensor, output_tensor, atol=1e-5, rtol=1e-3):
    """
    module: 対象のLayerNormモジュール (nn.LayerNorm)
    input_tensor: その層への入力 (N, L, D)
    output_tensor: その層からの出力 (N, L, D)
    atol, rtol: 許容誤差 (絶対誤差、相対誤差)
    """

    assert isinstance(module, torch.nn.LayerNorm), "moduleはLayerNormである必要があります"

    weight = module.weight.detach()
    bias = module.bias.detach()
    eps = module.eps

    # 入力の次元を最後の次元に対して正規化
    mean = input_tensor.mean(dim=-1, keepdim=True)    # (N, L, 1)
    var = input_tensor.var(dim=-1, unbiased=False, keepdim=True)  # (N, L, 1)
    std = torch.sqrt(var + eps)

    normalized_input = (input_tensor - mean) / std    # (N, L, D)
    reconstructed_output = normalized_input * weight + bias  # (N, L, D)

    # 一致してるかチェック
    is_close = torch.allclose(reconstructed_output, output_tensor, atol=atol, rtol=rtol)

    # 差分も出しておく
    mse_diff = torch.sqrt(((reconstructed_output - output_tensor)**2).mean()).item()

    return is_close, mse_diff


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
                # embedding 用に別ディレクトリを使う場合は、weights_dir と同じ規約のパスを指定する
                emb_dir = weights_dir


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

                print('CNN Embedding: Uniform Quantizing...')
                quantized_param = channel_wise_quantize_to_n_bits(param, n_bits=bit_width)
                print(f'uniform quantization mse: {torch.sqrt(((quantized_param - param)**2).mean())}')
                param.data.copy_(quantized_param)

        elif ('head' in name or 'classifier' in name) and 'weight' in name and not 'norm' in name and head_qtz:
            original_shape = param.shape
            conversion_shape = param.reshape(param.shape[0], -1).shape
            head_list = []
            # head 用に別ディレクトリを使う場合は、weights_dir と同じ規約のパスを指定する
            head_dir = weights_dir

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





# 全体最適化処理関数

def norm_layer_wise_tuning(model, data_num, bit_width, qq_weight_dir, model_type='deit'):
    """
    model: 元のPyTorchモデル
    data_num: キャリブレーションデータ数
    bit_width: 量子化ビット幅
    qq_weight_dir: 量子化済み重み保存ディレクトリ
    model_type: モデル種類 (例: 'deit', 'vit')
    """

    og_model = copy.deepcopy(model)
    qq_model = load_qq_model(model=model, weights_dir=qq_weight_dir, bit_width=bit_width, model_type=model_type, emb_qtz=True, head_qtz=True)
    qq_model.eval()
    og_model.eval()

    dataloader, _ = get_imagenet(model_name=model_type, calib_batchsize=64, num_traindatas=data_num, seed=1)

    with torch.no_grad():
        for name, module in qq_model.named_modules():
            if 'norm' in name and isinstance(module, torch.nn.LayerNorm):
                print('Optimizing LayerNorm:', name)

                # # --- まだキャッシュしてなければ計算 ---
                og_in, og_out = get_calibration(og_model, name+'.weight', data_num=data_num, dataloader=dataloader)
                q_in, q_out = get_calibration(qq_model, name+'.weight', data_num=data_num, dataloader=dataloader)

                # 検証！
                ok, mse = verify_layernorm(module, q_in, q_out)
                print(f"Layer {name} verification:", "OK" if ok else f"NG (mse={mse:.6f})")
                print(f'before optimization, input error:{torch.sqrt(((q_in - og_in)**2).mean())}, output error:{torch.sqrt(((q_out - og_out)**2).mean())}')

                # --- γとβを最適化 ---
                gamma, beta = optimize_gamma_beta(q_in, og_out, num_iters=10, eps=1e-5)

                # --- γとβを同時に保存 ---
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data.copy_(gamma)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.copy_(beta)
                
                # どれくらいエラーが減ったか検証
                ok, mse = verify_layernorm(module, q_in, og_out)
                print(f'after optimization, output error:{mse}')
    return qq_model

    


def whole_norm_layer_tuning(
    model, data_num, bit_width, qq_weight_dir, batch_size=32, num_epochs=10, lr=1e-3, device=torch.device("cuda"), model_type='deit', loss_type='cross_entropy'
):
    dataloader, _ = get_imagenet(model_name=model_type, calib_batchsize=batch_size, num_traindatas=data_num, seed=1)
    fp_model = copy.deepcopy(model)
    qq_model = load_qq_model(model=model, weights_dir=qq_weight_dir, bit_width=bit_width, model_type=model_type, emb_qtz=True, head_qtz=True)
    fp_model = fp_model.to(device)
    qq_model = qq_model.to(device)

    fp_model.eval()  # 元モデルは推論モード
    qq_model.train()  # qqモデルは微調整モード

    # LayerNormのγとβだけ最適化対象にする
    optim_params = []
    for name, module in qq_model.named_modules():
        if isinstance(module, nn.LayerNorm):
            optim_params.append(module.weight)
            optim_params.append(module.bias)
        elif isinstance(module, nn.Linear):
            if not module.bias is None:
                print(name, module.bias.shape)
                optim_params.append(module.bias)

    optimizer = optim.Adam(optim_params, lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)

            with torch.no_grad():
                if 'deit' in model_type:
                    fp_logits = fp_model(images)
                elif 'swin' in model_type:
                    fp_logits = fp_model(images).logits
                else: fp_logits = fp_model(images)

            if 'deit' in model_type:
                qq_logits = qq_model(images)
            elif 'swin' in model_type:
                qq_logits = qq_model(images).logits
            else: qq_logits = qq_model(images)


            # --- ここで損失を選択 ---
            if loss_type == "mse":
                loss = F.mse_loss(qq_logits, fp_logits)
            elif loss_type == "cross_entropy":
                # fp_logitsをsoft labelとみなしてKL Divergenceで近づける
                ref_probs = F.softmax(fp_logits, dim=-1)
                log_probs = F.log_softmax(qq_logits, dim=-1)
                loss = F.kl_div(log_probs, ref_probs, reduction='batchmean')
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    return qq_model

def process(model, data_num, bit_width, qq_weight_dir, batch_size=32, num_epochs=10, lr=1e-3, device=torch.device("cuda"), model_type='deit', tuning_type='whole', loss_type='mse'):
    if tuning_type=='whole':
        return whole_norm_layer_tuning(model, data_num, bit_width, qq_weight_dir, batch_size=batch_size, num_epochs=num_epochs, lr=lr, device=device, model_type=model_type, loss_type=loss_type)
    elif tuning_type=='layer_wise':
        return norm_layer_wise_tuning(model, data_num, bit_width, qq_weight_dir, model_type=model_type)
    else: raise ValueError('正しいtuning_typeを設定してください')
        




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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy



# モデルのテストプロセス
def evaluate_models(qq_weight_dir, model_name='deit-s', bit_width=4, data_num=1024, batch_size=32, num_epochs=10, lr=0.1, device=torch.device('cuda:0'), loss_type='mse', tuning_type='whole'):

    model = get_model(model_name)

    start = time.time()
    model = process(model=model, data_num=data_num, bit_width=bit_width, batch_size=batch_size, qq_weight_dir=qq_weight_dir, model_type=model_name, num_epochs=num_epochs, lr=lr, device=device, loss_type=loss_type, tuning_type=tuning_type)
    end = time.time()
    print("ImageNetでテスト中...")
    model.eval()
    train, imagenet_loader = get_imagenet(model_name=model_name, val_batchsize=batch_size)
    imagenet_accuracy = test_model_accuracy(model, imagenet_loader, device)
    print(f"{bit_width}Bit ImageNet Accuracy: {imagenet_accuracy:.2f}%")
    print(f'Quantization Time:{(end-start)/60}[min]')
    print(f'lr:{lr}, bit:{bit_width}, acc:{imagenet_accuracy}')
    return imagenet_accuracy



# 実行
if __name__ == "__main__":
    step = 100000
    device_id = 3
    data_num_list = (2048, 1024)
    model_names = ['deit-b']
    rank_scale = 1.5
    # プロセス開始方法を'spawn'に設定
    set_start_method("spawn", force=True)
    data = []
    for model_name in model_names:
        if 'deit' in model_name or 'vit' in model_name:
            gs = 384
            data_num = data_num_list[0]
        else:
            gs = 96
            data_num = data_num_list[1]
        for lr in [0.001]:
            for bit_width in [1]:
                for epoch in [15]:
                    for batch in [16]:
                        # 重みが保存されているディレクトリ
                        weights_dir = r'./bqq_compressed_data/{0}-{1}step-{2}gs-{3}'.format(model_name, step, gs, rank_scale)
                        acc = evaluate_models(qq_weight_dir=weights_dir, model_name=model_name, bit_width=bit_width, batch_size=batch, data_num=data_num, num_epochs=epoch, lr=lr, device=torch.device(f'cuda:{device_id}'), loss_type='mse', tuning_type='whole')
                        data.append({'model':model_name, 'bit':bit_width, 'batch_size':batch, 'lr':lr, 'data_num':data_num, 'epochs':epoch, 'accuracy':acc})
                        df = pd.DataFrame(data)
                        df.to_csv(f'./bqq_acc/{model_name}-{rank_scale}rankscale-{bit_width}bit-norm-bias_correction_{data_num}datas_{epoch}epoch.csv')
