import torch
from torch.multiprocessing import Process, Queue
from torch.multiprocessing import set_start_method
import sys
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from datautils import get_loaders
from parsers import parse_args
import os
from tqdm import tqdm
sys.path.append('/work2/k-kuroki/BQQ/quantizer')
# quantizerをインポート
from quantizer import BinaryQuadraticQuantization as BQQ, BinaryQuadraticQuantization2 as BQQ2
import queue
from multiprocessing import Process, Queue
import pandas as pd
import time
import model_loader
import binary_quadratic_network
import re


# 自作の変換関数（例：正規化）
def custom_weight_transform(weight: torch.Tensor, save_path, bit_width=4, rank_scale=1, group_size=128, Nstep=50000, seed=0, workers_per_gpu=16) -> torch.Tensor:
    """
    3次元テンソルに変換後、変換を適用する。
    """
    if len(weight.shape) != 2:
        raise ValueError("入力テンソルは2次元である必要があります")
    instance = BQQ2(weight, rank_scale=rank_scale)
    x = instance.bqq_large_matrix_multi_worker(max_patch_size=group_size, bit_width=bit_width, save_name=save_path, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, seed=seed, workers_per_gpu=workers_per_gpu)
    # x = BQQ().hbqq_large_matrix_multi_worker(x=weight, rank_scale=rank_scale, max_patch_size=group_size, bit_width=bit_width, save_name=save_path, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, seed=seed, workers_per_gpu=workers_per_gpu)
    return x



# ワーカー関数
def worker(worker_id, param_queue, transform_fn, save_dir, bit_width=4, rank_scale=1, group_size=128, Nstep=20000, seed=0, workers_per_gpu=16):
    # torch.cuda.set_device(device_id)
    print(f"Worker {worker_id} is starting...")
    while True:
        try:
            # キューからパラメータを取得
            name, param = param_queue.get(timeout=3)
            # print(f"Worker {worker_id} processing: {name} (shape: {param.shape})")
        except queue.Empty:
            print(f"Worker {worker_id} queue is empty, terminating.")
            break
        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")
            break
        
        if not 'norm' in name and not 'bias' in name and not 'emb' in name:
            print(name, param.shape)
            save_path = os.path.join(save_dir, f"{name}")
            transformed_param = transform_fn(param.data, save_path, bit_width=bit_width, rank_scale=rank_scale, group_size=group_size, Nstep=Nstep, seed=seed, workers_per_gpu=workers_per_gpu)
            param.data.copy_(transformed_param)

            # パラメータの保存
            save_path = os.path.join(save_dir, f"{name}.pth")
            torch.save(param.data, save_path)
            print(f"{name} transformed and saved by Worker {worker_id}")

# 並列化関数（複数ワーカー）
def transform_model_weights_parallel(model, transform_fn, save_dir, num_workers=4, bit_width=4, rank_scale=1, group_size=128, Nstep=20000, seed=0, workers_per_gpu=16, layer_threshold=0):
    os.makedirs(save_dir, exist_ok=True)
    param_queue = Queue()

    
    # キューにモデルのパラメータを追加
    for name, param in model.named_parameters():
        numbers = re.findall(r'\d+', name)  # 連続する数字（例: 'layer21.weight' -> ['21']）
        if any(int(num) >= layer_threshold for num in numbers):# レイヤーのインデックスがn以上のもののみ対象
            print(f"Adding {name} to queue with shape {param.shape}")
            # param_queue.put((name, param))
            if not 'norm' in name and not 'bias' in name and not 'emb' in name:
                param_queue.put((name, param))
    
    # プロセスを生成
    processes = []
    for i in range(num_workers):
        worker_id = len(processes)  # ワーカーIDを一意に設定
        p = Process(
            target=worker,
            args=(worker_id, param_queue, transform_fn, save_dir, bit_width, rank_scale, group_size, Nstep, seed, workers_per_gpu)
        )
        processes.append(p)
        p.start()
    
    # プロセスの終了を待つ
    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished")

    

# 量子化
def evaluate_models(model, num_workers=8, save_dir=os.getcwd(), bit_width=4, rank_scale=1, group_size=128, Nstep=20000, seed=0, workers_per_gpu=16, layer_threshold=0):
    print("Start Weight Reparametrization......")
    start = time.time()
    transform_model_weights_parallel(model, custom_weight_transform, save_dir=save_dir, num_workers=num_workers, bit_width=bit_width, rank_scale=rank_scale, group_size=group_size, Nstep=Nstep, seed=seed, workers_per_gpu=workers_per_gpu, layer_threshold=layer_threshold)
    end = time.time()
    print("Finish Weight Reparametrization !!")
    print(f'Quantization Time:{(end-start)/60}[min]')


# 実行
if __name__ == "__main__":
    # args = parse_args()
    # model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_path = "Qwen/Qwen2.5-1.5B"
    bit_width = 4
    gs = 128
    step = 50000
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # モデルの保存先の設定
    save_dir = os.path.dirname(__file__) + r'/bqq_compressed_data/{0}-{1}gs-{2}step'.format(model_path.split("/")[-1], gs, step)

    # プロセス開始方法を'spawn'に設定
    set_start_method("spawn", force=True)
    evaluate_models(model=model, num_workers=3, save_dir=save_dir, bit_width=bit_width, rank_scale=1, group_size=gs, Nstep=step, seed=0, workers_per_gpu=16, layer_threshold=4)
