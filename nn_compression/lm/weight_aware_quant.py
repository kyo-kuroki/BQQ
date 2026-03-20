from __future__ import annotations

import argparse
from pathlib import Path
import queue
import re
import time

import torch
from torch.multiprocessing import Process, Queue, set_start_method
from transformers import AutoModelForCausalLM

try:
    from .compressed_data import default_compressed_data_dir, ensure_bqq_root_on_path
except ImportError:
    from compressed_data import default_compressed_data_dir, ensure_bqq_root_on_path


ensure_bqq_root_on_path()

from quantizer import BinaryQuadraticQuantization2 as BQQ2


def parse_args():
    parser = argparse.ArgumentParser(description="Weight-aware BQQ for language models")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--bit_width", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--rank_scale", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--workers_per_gpu", type=int, default=16)
    parser.add_argument("--main_gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_threshold", type=int, default=4)
    parser.add_argument("--save_dir", type=Path, default=None)
    return parser.parse_args()


def is_quantization_target(name: str) -> bool:
    return name.endswith("weight") and all(token not in name for token in ("norm", "bias", "emb"))


def passes_layer_threshold(name: str, layer_threshold: int) -> bool:
    layer_ids = [int(num) for num in re.findall(r"\d+", name)]
    return any(layer_id >= layer_threshold for layer_id in layer_ids)


def quantize_weight(
    weight: torch.Tensor,
    save_prefix: Path,
    *,
    bit_width: int,
    rank_scale: float,
    group_size: int,
    num_steps: int,
    seed: int,
    workers_per_gpu: int,
    main_gpu_id: int,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("The input tensor must be 2-dimensional")

    quantizer = BQQ2(weight, rank_scale=rank_scale)
    return quantizer.bqq_large_matrix_multi_worker(
        max_patch_size=group_size,
        bit_width=bit_width,
        save_name=str(save_prefix),
        zeta=4,
        eta=0.06,
        Tinit=0.2,
        Tfin=0.005,
        Nstep=num_steps,
        seed=seed,
        workers_per_gpu=workers_per_gpu,
        main_gpu_id=main_gpu_id,
    )


def worker(worker_id, param_queue, save_dir, bit_width, rank_scale, group_size, num_steps, seed, workers_per_gpu, main_gpu_id):
    print(f"Worker {worker_id} is starting...")

    while True:
        try:
            name, tensor = param_queue.get(timeout=3)
        except queue.Empty:
            print(f"Worker {worker_id} queue is empty, terminating.")
            break
        except Exception as error:
            print(f"Worker {worker_id} encountered an error: {error}")
            break

        print(f"Worker {worker_id} processing {name}: {tuple(tensor.shape)}")
        save_prefix = save_dir / name
        transformed = quantize_weight(
            tensor,
            save_prefix,
            bit_width=bit_width,
            rank_scale=rank_scale,
            group_size=group_size,
            num_steps=num_steps,
            seed=seed,
            workers_per_gpu=workers_per_gpu,
            main_gpu_id=main_gpu_id,
        )

        torch.save(transformed.cpu(), save_dir / f"{name}.pth")
        print(f"Worker {worker_id} finished {name}")


def transform_model_weights_parallel(
    model,
    save_dir: Path,
    *,
    num_workers: int,
    bit_width: int,
    rank_scale: float,
    group_size: int,
    num_steps: int,
    seed: int,
    workers_per_gpu: int,
    main_gpu_id: int,
    layer_threshold: int,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    param_queue = Queue()

    for name, param in model.named_parameters():
        if not is_quantization_target(name):
            continue
        if not passes_layer_threshold(name, layer_threshold):
            continue
        print(f"Queueing {name}: {tuple(param.shape)}")
        param_queue.put((name, param.detach().cpu()))

    processes = []
    for worker_id in range(num_workers):
        process = Process(
            target=worker,
            args=(
                worker_id,
                param_queue,
                save_dir,
                bit_width,
                rank_scale,
                group_size,
                num_steps,
                seed,
                workers_per_gpu,
                main_gpu_id,
            ),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        print(f"Process {process.pid} has finished")


def evaluate_models(
    model,
    *,
    num_workers=8,
    save_dir: Path,
    bit_width=4,
    rank_scale=1.0,
    group_size=128,
    num_steps=20000,
    seed=0,
    workers_per_gpu=16,
    main_gpu_id=0,
    layer_threshold=0,
):
    print("Start Weight Reparametrization...")
    start = time.time()
    transform_model_weights_parallel(
        model,
        save_dir=save_dir,
        num_workers=num_workers,
        bit_width=bit_width,
        rank_scale=rank_scale,
        group_size=group_size,
        num_steps=num_steps,
        seed=seed,
        workers_per_gpu=workers_per_gpu,
        main_gpu_id=main_gpu_id,
        layer_threshold=layer_threshold,
    )
    elapsed_minutes = (time.time() - start) / 60
    print("Finish Weight Reparametrization")
    print(f"Quantization Time: {elapsed_minutes:.2f} [min]")


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for weight-aware BQQ quantization.")

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = default_compressed_data_dir(args.model_name, args.group_size, args.num_steps)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto").cpu()
    set_start_method("spawn", force=True)
    evaluate_models(
        model=model,
        num_workers=args.num_workers,
        save_dir=Path(save_dir),
        bit_width=args.bit_width,
        rank_scale=args.rank_scale,
        group_size=args.group_size,
        num_steps=args.num_steps,
        seed=args.seed,
        workers_per_gpu=args.workers_per_gpu,
        main_gpu_id=args.main_gpu_id,
        layer_threshold=args.layer_threshold,
    )


if __name__ == "__main__":
    main()
