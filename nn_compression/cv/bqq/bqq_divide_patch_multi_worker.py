import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.multiprocessing import Process, Queue
from torch.multiprocessing import set_start_method
from transformers import AutoImageProcessor, SwinForImageClassification
import os
import sys
from tqdm import tqdm
sys.path.append('./../../../quantizer')
sys.path.append('./../utils')
# quantizerをインポート
from quantizer import BinaryQuadraticQuantization2 as BQQ2, BinaryQuadraticQuantization as BQQ, BinaryCodingQuantization as BCQ, UniformQuantization as UQ
import queue
from multiprocessing import Process, Queue, current_process
import pandas as pd
import timm
from PIL import Image
import time
from build_dataset import get_imagenet
from build_model import get_model
import argparse



# 自作の変換関数（例：正規化）
def custom_weight_transform(weight: torch.Tensor, max_patch_size, save_path, Nstep, bit_width, main_gpu_id, rank_scale=1, seed=0) -> torch.Tensor:
    """
    2次元テンソルに変換後、変換を適用する。
    """
    if len(weight.shape) != 2:
        raise ValueError("入力テンソルは2次元である必要があります")
    # x = BQQ().bqq_large_matrix_multi_worker(weight.detach().cpu(), rank_scale=rank_scale, max_patch_size=max_patch_size, bit_width=bit_width, save_name=save_path, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, seed=seed, workers_per_gpu=4, main_gpu_id=main_gpu_id)
    x = BQQ2(weight, rank_scale=rank_scale).bqq_large_matrix_multi_worker(max_patch_size=max_patch_size, bit_width=bit_width, save_name=save_path, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, seed=seed, workers_per_gpu=1, main_gpu_id=main_gpu_id)
    return x




# ワーカー関数
def worker(worker_id, param_queue, transform_fn, save_dir, bit_width=2, model_name='deit', Nstep=50000, main_gpu_id=0, rank_scale=1, seed=0):
    while True:
        try:
            # キューからパラメータを取得
            name, param = param_queue.get(timeout=1)
        except queue.Empty:
            print(f"Worker {worker_id} queue is empty, terminating.")
            break
        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")
            break

        with torch.no_grad():
            if 'emb' in name and 'weight' in name and not 'norm' in name: # エンべディング層を量子化するとき
                print('Embedding', name, param.shape)
                if 'deit' in model_name or 'vit' in model_name:
                    if param.shape[0] == 384:
                        max_patch_size = 384
                        emb_Nstep = Nstep
                    elif param.shape[0] == 768:
                        max_patch_size=768
                        emb_Nstep = Nstep
                    else: raise ValueError('embedding dimention is wrong!!')
                    save_path = os.path.join(save_dir, f"{name}")
                    w = param.data.reshape(param.data.shape[0], -1)
                    transformed_param = transform_fn(w, max_patch_size=max_patch_size, bit_width=bit_width, save_path=save_path, Nstep=emb_Nstep, main_gpu_id=main_gpu_id,rank_scale=rank_scale, seed=seed)
                    transformed_param = transformed_param.reshape_as(param.data)
                elif 'swin' in model_name:
                    print('swin embedding --> uniform quantization')
                    w = param.data.reshape(param.data.shape[0], -1)
                    transformed_param = UQ().channel_wise_uq(w, bit_width)
                    transformed_param = transformed_param.reshape_as(param.data)
                param.data.copy_(transformed_param)
            else:
                if 'norm' in name or 'bias' in name or 'token' in name or 'pos' in name: 
                    print('skip:', name, param.shape)
                    pass
                else:
                    if 'weight' in name and not 'norm' in name and not 'head' in name and not 'emb' in name and ('deit' in model_name or 'vit' in model_name): ## DeiTの場合
                        max_patch_size=384
                    elif 'encoder' in name and 'weight' in name and not 'norm' in name and ('swin' in model_name): ## Swin-Transformerの場合
                        max_patch_size=96
                    elif 'weight' in name and ('head' in name or 'classifier' in name) and not 'norm' in name: # 分類層を量子化するとき
                        max_patch_size=100

                    save_path = os.path.join(save_dir, f"{name}")
                    transformed_param = transform_fn(param.data, max_patch_size=max_patch_size, bit_width=bit_width, save_path=save_path, Nstep=Nstep, main_gpu_id=main_gpu_id,rank_scale=rank_scale, seed=seed)
                    param.data.copy_(transformed_param)

            # パラメータの保存
            save_path = os.path.join(save_dir, f"{name}.pth")
            torch.save(param.data, save_path)
            print(f"{name} was saved by Worker {worker_id}")


# 並列化関数（複数ワーカー）
def transform_model_weights_parallel(model, transform_fn, save_dir, bit_width, model_type, num_workers=4, rank_scale=1, seed=0):
    os.makedirs(save_dir, exist_ok=True)
    param_queue = Queue()
    
    # キューにモデルのパラメータを追加
    for name, param in model.named_parameters():
        if not ('norm' in name or 'bias' in name or 'token' in name or 'pos' in name): 
            param_queue.put((name, param.detach().cpu()))
    
    # プロセスを生成
    processes = []
    for worker_id in range(num_workers):
        # worker_id = len(processes)  # ワーカーIDを一意に設定

        p = Process(
            target=worker,
            args=(worker_id, param_queue, transform_fn, save_dir, bit_width, model_type, Nstep, worker_id, rank_scale, seed)
        )
        processes.append(p)
        p.start()
    
    # プロセスの終了を待つ
    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished")

    




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
def evaluate_models(model_name, bit_width=4, num_workers=8, save_dir=os.getcwd(), rank_scale=1, seed=0):
    model = get_model(model_name).cpu()  # GPU に載せない
    torch.cuda.empty_cache()             # 余計なメモリ解放
    start = time.time()
    transform_model_weights_parallel(model, custom_weight_transform, save_dir=save_dir, num_workers=num_workers, model_type=model_name, bit_width=bit_width, rank_scale=rank_scale, seed=seed)
    end = time.time()
    print(f'Quantization Time:{(end-start)/60}[min]')
    trainloader, testloader = get_imagenet(model_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    imagenet_accuracy = test_model_accuracy(model, testloader, device)
    print(f"Transformed {model_name} ImageNet Accuracy: {imagenet_accuracy:.2f}%")
    return imagenet_accuracy, (end-start)/60



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


def main(args):
    # モデルの保存先の設定
    save_dir = f'./bqq_compressed_data/{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale'
    # プロセス開始方法を'spawn'に設定
    set_start_method("spawn", force=True)
    accuracy, q_time = evaluate_models(model_name=args.model_name, bit_width=args.bit_width, num_workers=args.num_workers, save_dir=save_dir, rank_scale=args.rank_scale, seed=args.seed)
    result = {'model':args.model_name, 'bit_width': args.bit_width, 'q_time':q_time, 'Nstep':args.Nstep, 'accuracy':accuracy, 'num_workers':args.num_workers}
    df = pd.DataFrame(result)
    df.to_csv(f'./results/{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale-{args.bit_width}bit.csv')


# 実行
if __name__ == "__main__":
    args = get_parser()
    main(args)