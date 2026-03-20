import argparse
import os
from pathlib import Path
import queue
import time

import pandas as pd
import torch
from torch.multiprocessing import Process, Queue, set_start_method

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
BQQ_ROOT = CV_DIR.parent.parent
UTILS_DIR = CV_DIR / "utils"

for path in (BQQ_ROOT, UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from quantizer import BinaryQuadraticQuantization2 as BQQ2
from build_dataset import get_imagenet
from build_model import get_model
from make_bqq_model_from_compressed_data import save_bqq_model
from utils import test_model_accuracy


def argparser():
    parser = argparse.ArgumentParser(description="Quantization Worker Settings")
    parser.add_argument("--bit_width", type=int, default=2, help="Quantization bit width")
    parser.add_argument("--model_name", type=str, default="deit-s", help="Model name")
    parser.add_argument("--Nstep", type=int, default=5000, help="Number of optimization steps")
    parser.add_argument("--main_gpu_id", type=int, default=0, help="Main GPU ID")
    parser.add_argument("--rank_scale", type=float, default=1.0, help="Rank scale factor")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--group_size", type=int, default=32, help="Group size for patch division")
    parser.add_argument("--num_workers_per_gpu", type=int, default=16, help="Workers per GPU")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save quantized weights")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--data_path", type=str, default=None, help="Path to ImageNet. If omitted, use IMAGENET_DIR or IMAGENET_ROOT.")
    parser.add_argument("--zeta", type=float, default=4.0, help="Zeta parameter")
    parser.add_argument("--eta", type=float, default=0.06, help="Eta parameter")
    parser.add_argument("--Tinit", type=float, default=0.2, help="Initial temperature")
    parser.add_argument("--Tfin", type=float, default=0.005, help="Final temperature")
    return parser.parse_args()


def custom_weight_transform(weight: torch.Tensor, args, save_path) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("The input tensor must be 2-dimensional")
    return BQQ2(weight, rank_scale=args.rank_scale).bqq_large_matrix_multi_worker(
        max_patch_size=args.group_size,
        bit_width=args.bit_width,
        save_name=save_path,
        zeta=args.zeta,
        eta=args.eta,
        Tinit=args.Tinit,
        Tfin=args.Tfin,
        Nstep=args.Nstep,
        seed=args.seed,
        workers_per_gpu=args.num_workers_per_gpu,
        main_gpu_id=args.main_gpu_id,
    )


def worker(worker_id, param_queue, transform_fn, args):
    while True:
        try:
            name, param = param_queue.get(timeout=1)
        except queue.Empty:
            print(f"Worker {worker_id}: queue empty, exiting.")
            break
        except Exception as error:
            print(f"Worker {worker_id} encountered error: {error}")
            break

        with torch.no_grad():
            if name.endswith("weight") and "norm" not in name and "bias" not in name:
                print(f"Quantizing Linear Weight: {name}, shape={param.shape}")
                save_path = os.path.join(args.save_dir, name)
                transformed_param = transform_fn(param.data, save_path=save_path, args=args)
                param.data.copy_(transformed_param)
            else:
                print(f"Skipping: {name} ({param.shape})")

            weight_path = os.path.join(args.save_dir, f"{name}.pt")
            torch.save(param.data.cpu(), weight_path)
            print(f"Worker {worker_id} saved: {name}")


def transform_model_weights_parallel(model, transform_fn, args):
    os.makedirs(args.save_dir, exist_ok=True)
    param_queue = Queue()

    for name, param in model.named_parameters():
        if not ("norm" in name or "bias" in name or "token" in name or "pos" in name or "emb" in name):
            param_queue.put((name, param.detach().cpu()))

    processes = []
    for worker_id in range(args.num_workers):
        process = Process(target=worker, args=(worker_id, param_queue, transform_fn, args))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        print(f"Process {process.pid} has finished")


def run_quantization(args):
    model = get_model(args.model_name).cpu()
    torch.cuda.empty_cache()
    start = time.time()
    transform_model_weights_parallel(model, custom_weight_transform, args)
    end = time.time()
    print(f"Quantization Time:{(end - start) / 60}[min]")

    save_bqq_model(
        args.model_name,
        args.save_dir,
        args.bit_width,
        args.group_size,
        args.Nstep,
        device=f"cuda:{args.main_gpu_id}",
    )
    _, testloader = get_imagenet(args.model_name, data_path=args.data_path)
    device = torch.device(f"cuda:{args.main_gpu_id}") if torch.cuda.is_available() else "cpu"
    imagenet_accuracy = test_model_accuracy(model, testloader, device)
    print(f"Transformed {args.model_name} ImageNet Accuracy: {imagenet_accuracy:.2f}%")
    return imagenet_accuracy, (end - start) / 60


def main(args):
    if args.save_dir is None:
        args.save_dir = str(
            SCRIPT_DIR / "bqq_compressed_data" / f"{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale"
        )

    results_dir = SCRIPT_DIR / "results"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    set_start_method("spawn", force=True)

    accuracy, q_time = run_quantization(args)
    result = {
        "model": args.model_name,
        "bit_width": args.bit_width,
        "q_time": q_time,
        "Nstep": args.Nstep,
        "accuracy": accuracy,
        "num_workers": args.num_workers,
    }
    df = pd.DataFrame([result])
    df.to_csv(results_dir / f"{args.model_name}-{args.Nstep}step-{args.group_size}gs-{args.rank_scale}rankscale-{args.bit_width}bit.csv")


if __name__ == "__main__":
    main(argparser())
