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
sys.path.append('/work2/k-kuroki/quadratic_quantization/quantizer')
# quantizerをインポート
from quantizer import BinaryQuadraticQuantization as BQQ, QuadraticQuantization as QQ, BinaryCodingQuantization as BCQ, UniformQuantization as UQ
import queue
from multiprocessing import Process, Queue, current_process
import pandas as pd
import timm
from PIL import Image
import time
from build_dataset import get_imagenet, calibsample_from_trainloader
from build_model import get_model

# 自作の変換関数（例：正規化）
def custom_weight_transform(weight: torch.Tensor, max_patch_size, save_path, Nstep, bit_width, main_gpu_id, rank_scale=1, seed=0) -> torch.Tensor:
    """
    2次元テンソルに変換後、変換を適用する。
    """
    if len(weight.shape) != 2:
        raise ValueError("入力テンソルは2次元である必要があります")
    x = BQQ().bqq_large_matrix_multi_worker(weight.detach().cpu(), rank_scale=rank_scale, max_patch_size=max_patch_size, bit_width=bit_width, save_name=save_path, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, seed=seed, workers_per_gpu=4, main_gpu_id=main_gpu_id)
    return x




# ワーカー関数
def worker(worker_id, param_queue, transform_fn, save_dir, bit_width=2, group_size=128, head_group_size=None, Nstep=50000, main_gpu_id=0, rank_scale=1, seed=0):
    while True:
        try:
            # キューからパラメータを取得
            name, param, input = param_queue.get(timeout=1)
        except queue.Empty:
            print(f"Worker {worker_id} queue is empty, terminating.")
            break
        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")
            break

        with torch.no_grad():
            if not 'norm' in name and not 'bias' in name and not 'emb' in name: 
                if ('head' in name or 'classifier' in name): # 分類層を量子化するとき
                    if not (head_group_size is None):
                        max_patch_size = head_group_size
                elif 'weight' in name: 
                    max_patch_size=group_size

                save_path = os.path.join(save_dir, f"{name}")
                transformed_param = transform_fn(param.data, input, max_patch_size=max_patch_size, bit_width=bit_width, save_path=save_path, Nstep=Nstep, main_gpu_id=main_gpu_id,rank_scale=rank_scale, seed=seed)
                param.data.copy_(transformed_param)

            # パラメータの保存
            save_path = os.path.join(save_dir, f"{name}.pth")
            torch.save(param.data, save_path)
            print(f"{name} was saved by Worker {worker_id}")


# 並列化関数（複数ワーカー）
def transform_model_weights_parallel(model, inputs, transform_fn, save_dir, bit_width, group_size, head_group_size=None, num_workers=4, rank_scale=1, seed=0):
    os.makedirs(save_dir, exist_ok=True)
    param_queue = Queue()

    def get_input_hook(name, module):
        def hook(module, input, output):
            weight = module.weight.detach().cpu()
            inp = input[0].detach().cpu()
            param_queue.put((name, weight, inp))
        return hook

    # 再帰的に全モジュールを探索して Linear にhookを登録
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(get_input_hook(name, module))

    # forwardを実行してhookを発火させる
    with torch.no_grad():
        _ = model(inputs)
    
    # queueの中身を確認
    while not param_queue.empty():
        name, weight, inp = param_queue.get()
        print(name, weight.shape, inp.shape)
    
    # プロセスを生成
    processes = []
    for worker_id in range(num_workers):
        # worker_id = len(processes)  # ワーカーIDを一意に設定

        p = Process(
            target=worker,
            args=(worker_id, param_queue, transform_fn, save_dir, bit_width, group_size, head_group_size, Nstep, worker_id, rank_scale, seed)
        )
        processes.append(p)
        p.start()
    
    # プロセスの終了を待つ
    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished")


def transform_model_weights_sequential(model, inputs, transform_fn,
                                       save_dir, bit_width, group_size,
                                       head_group_size=None, rank_scale=1, seed=0):
    os.makedirs(save_dir, exist_ok=True)

    # Linear層のリストを作成
    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)]

    for layer_idx, (name, module) in enumerate(linear_layers):
        print(f"[{layer_idx+1}/{len(linear_layers)}] Processing {name}")

        # hook定義
        results = []
        def hook_fn(mod, inp, out):
            weight = mod.weight.detach().cpu()
            x = inp[0].detach().cpu()
            results.append((name, weight, x))

        # hook登録
        handle = module.register_forward_hook(hook_fn)

        # 1層分のデータを収集
        with torch.no_grad():
            _ = model(inputs)

        # hook解除
        handle.remove()

        # ---- ここで worker に投げる処理を呼ぶ ----
        # 例: 同期で処理する場合
        for weight, inp in results:
            result = transform_fn(weight, inp, bit_width, group_size,
                                  head_group_size, rank_scale, seed)
            torch.save(result, os.path.join(save_dir, f"{name}.pt"))

        # メモリ解放
        del results
        torch.cuda.empty_cache()





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
def evaluate_models(model_name, bit_width=4, group_size=128, head_group_size=None, num_workers=8, save_dir=os.getcwd(), rank_scale=1, seed=0, device='cuda:0'):
    device = torch.device(device) if torch.cuda.is_available() else "cpu"
    model = get_model(model_name).to(device)  # GPU に載せない
    inputs, targets = calibsample_from_trainloader(get_imagenet(model_name)[0], n=128, seed=0)
    torch.cuda.empty_cache()             # 余計なメモリ解放
    start = time.time()
    transform_model_weights_parallel(model, inputs, custom_weight_transform, save_dir=save_dir, bit_width=bit_width, group_size=group_size, head_group_size=head_group_size, num_workers=num_workers, model_type=model_name, rank_scale=rank_scale, seed=seed)
    end = time.time()
    print(f'Quantization Time:{(end-start)/60}[min]')
    trainloader, testloader = get_imagenet(model_name)
    imagenet_accuracy = test_model_accuracy(model, testloader, device)
    print(f"Transformed {model_name} ImageNet Accuracy: {imagenet_accuracy:.2f}%")
    return imagenet_accuracy, (end-start)/60

# 実行
if __name__ == "__main__":
    gsl = (384, 96)
    num_workers = 8
    seed = 0
    results = []
    Nstep = 50000
    bit_width = 2
    group_size = 128
    for model_name in ['deit-s', 'deit-b']:
        # モデルの保存先の設定
        save_dir = os.path.dirname(__file__) + r'/qq_compressed_data/{0}-{1}step-{2}gs-max{3}bit'.format(model_name, Nstep, group_size, bit_width)
        # プロセス開始方法を'spawn'に設定
        set_start_method("spawn", force=True)
        accuracy, q_time = evaluate_models(model_name=model_name, bit_width=bit_width, num_workers=num_workers, save_dir=save_dir, rank_scale=1.5, seed=seed)
        # results.append({'model':model_name, 'bit_width': bit_width, 'q_time':q_time, 'Nstep':Nstep, 'accuracy':accuracy, 'num_workers':num_workers, 'num_workers_per_gpu':1})
        # df = pd.DataFrame(results)
        # df.to_csv('/work2/k-kuroki/quadratic_quantization/nn_compression/cv/test_results/bqq_results.csv')
