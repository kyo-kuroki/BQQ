import torch
from torch.multiprocessing import Process, Queue
from torch.multiprocessing import set_start_method
import os
import sys
sys.path.append('./../../../../BQQ')
sys.path.append('./../utils')
from quantizer import BinaryQuadraticQuantization2 as BQQ2, BinaryQuadraticQuantization as BQQ, BinaryCodingQuantization as BCQ, UniformQuantization as UQ
import queue
from multiprocessing import Process, Queue
import pandas as pd
import time
from build_dataset import get_imagenet
from build_model import get_model
import argparse
from make_bqq_model_from_compressed_data import save_bqq_model
from utils import test_model_accuracy

def argparser():
    parser = argparse.ArgumentParser(description="Quantization Worker Settings")

    # 既存パラメータ
    parser.add_argument("--bit_width", type=int, default=2,
                        help="Quantization bit width (default: 2)")
    parser.add_argument("--model_name", type=str, default="deit-s",
                        help="Model name (default: deit-s)")
    parser.add_argument("--Nstep", type=int, default=5000,
                        help="Number of steps for optimization (default: 5000)")
    parser.add_argument("--main_gpu_id", type=int, default=0,
                        help="Main GPU ID for the process (default: 0)")
    parser.add_argument("--rank_scale", type=float, default=1,
                        help="Rank scale factor for quantization (default: 1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Group size for patch division (default: 32)")
    parser.add_argument("--num_workers_per_gpu", type=int, default=16,
                        help="Number of workers per GPU (default: 16)")  
    parser.add_argument("--save_dir", type=str, default=os.getcwd(),
                        help="Directory to save quantized weights (default: current directory)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")

    # 新規：zeta, eta, Tinit, Tfin
    parser.add_argument("--zeta", type=float, default=4,
                        help="Zeta parameter (default: 4)")
    parser.add_argument("--eta", type=float, default=0.06,
                        help="Eta parameter (learning precision control, default: 0.06)")
    parser.add_argument("--Tinit", type=float, default=0.2,
                        help="Initial temperature (default: 0.2)")
    parser.add_argument("--Tfin", type=float, default=0.005,
                        help="Final temperature (default: 0.005)")

    return parser.parse_args()



# 自作の変換関数（例：正規化）
def custom_weight_transform(weight: torch.Tensor, args, save_path) -> torch.Tensor:
    if len(weight.shape) != 2:
        raise ValueError("The input tensor must be 2-dimensional")
    x = BQQ2(weight, rank_scale=args.rank_scale).bqq_large_matrix_multi_worker(max_patch_size=args.group_size, bit_width=args.bit_width, save_name=save_path, zeta=args.zeta, eta=args.eta, Tinit=args.Tinit, Tfin=args.Tfin, Nstep=args.Nstep, seed=args.seed, workers_per_gpu=args.num_workers_per_gpu, main_gpu_id=args.main_gpu_id)
    return x




def worker(worker_id, param_queue, transform_fn, args):

    while True:
        try:
            name, param = param_queue.get(timeout=1)
        except queue.Empty:
            print(f"Worker {worker_id}: queue empty, exiting.")
            break

        except Exception as e:
            print(f"Worker {worker_id} encountered error: {e}")
            break

        with torch.no_grad():

            # ============================
            # weight in Linear layer
            # ============================
            if name.endswith("weight") and "norm" not in name and "bias" not in name:

                print(f"Quantizing Linear Weight: {name}, shape={param.shape}")

                save_path = os.path.join(args.save_dir, name)
                transformed_param = transform_fn(
                    param.data,
                    save_path=save_path,
                    args=args
                )
                param.data.copy_(transformed_param)

            else:
                print(f"Skipping: {name} ({param.shape})")

            # ============================
            # 保存処理
            # ============================
            save_path = os.path.join(args.save_dir, f"{name}.pt")
            torch.save(param.data.cpu(), save_path)
            print(f"Worker {worker_id} saved: {name}")



# 並列化関数（複数ワーカー）
def transform_model_weights_parallel(model, transform_fn, args):
    os.makedirs(args.save_dir, exist_ok=True)
    param_queue = Queue()
    
    # キューにモデルのパラメータを追加
    for name, param in model.named_parameters():
        if not ('norm' in name or 'bias' in name or 'token' in name or 'pos' in name or 'emb' in name): 
            param_queue.put((name, param.detach().cpu()))
    
    # プロセスを生成
    processes = []
    for worker_id in range(args.num_workers):
        # worker_id = len(processes)  # ワーカーIDを一意に設定

        p = Process(
            target=worker,
            args=(worker_id, param_queue, transform_fn, args)
        )
        processes.append(p)
        p.start()
    
    # プロセスの終了を待つ
    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished")

    

# モデルのテストプロセス
def run_quantization(args):
    model = get_model(args.model_name).cpu()  # GPU に載せない
    torch.cuda.empty_cache()             # 余計なメモリ解放
    start = time.time()
    transform_model_weights_parallel(model, custom_weight_transform, args)
    end = time.time()
    print(f'Quantization Time:{(end-start)/60}[min]')
    save_bqq_model(args.model_name, args.save_dir, args.bit_width, args.group_size, args.Nstep, device=f'cuda:{args.main_gpu_id}')
    trainloader, testloader = get_imagenet(args.model_name)
    device = torch.device(f"cuda:{args.main_gpu_id}") if torch.cuda.is_available() else "cpu"
    imagenet_accuracy = test_model_accuracy(model, testloader, device)
    print(f"Transformed {args.model_name} ImageNet Accuracy: {imagenet_accuracy:.2f}%")
    return imagenet_accuracy, (end-start)/60


def main(args):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    # モデルの保存先の設定
    args.save_dir = f'./bqq_compressed_data/{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    # プロセス開始方法を'spawn'に設定
    set_start_method("spawn", force=True)
    accuracy, q_time = run_quantization(args)
    result = {'model':args.model_name, 'bit_width': args.bit_width, 'q_time':q_time, 'Nstep':args.Nstep, 'accuracy':accuracy, 'num_workers':args.num_workers}
    df = pd.DataFrame([result])
    df.to_csv(f'./results/{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale-{args.bit_width}bit.csv')


# 実行
if __name__ == "__main__":
    args = argparser()
    main(args)