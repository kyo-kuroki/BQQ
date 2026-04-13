import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm
import pandas as pd
import glob
import argparse
import os
import sys

try:
    from .compressed_data import BQQ_ROOT, default_results_dir
    from .datautils import get_wikitext2_testloader, get_c4_testloader
except ImportError:
    from compressed_data import BQQ_ROOT, default_results_dir
    from datautils import get_wikitext2_testloader, get_c4_testloader

# Ensure bqq_modules is importable (needed for torch.load of BQQ models)
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
import bqq_modules  # noqa: F401


WORKSPACE_ROOT = BQQ_ROOT.parent


def _maybe_add_repo_to_path(repo_name: str, repo_dir: str | None = None) -> None:
    candidates = []
    if repo_dir:
        candidates.append(Path(repo_dir).expanduser())

    env_dir = os.getenv(f"{repo_name.upper()}_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    candidates.append(WORKSPACE_ROOT / repo_name)

    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


def _load_gptq_model(model_path: str, repo_dir: str | None = None):
    try:
        from gptqmodel import GPTQModel
    except ImportError:
        _maybe_add_repo_to_path("GPTQModel", repo_dir=repo_dir)
        from gptqmodel import GPTQModel

    return GPTQModel.load(model_path)


@torch.no_grad()
def compute_ppl_from_testloader(model, testloader, device="cuda"):
    print("Evaluating PPL...")
    model.eval()
    model.to(device)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for inp, tar in tqdm(testloader, desc="Computing PPL"):
        inp, tar = inp.to(device), tar.to(device)
        outputs = model(inp)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tar[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        valid_tokens = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens

    ppl = torch.exp(torch.tensor(total_loss / total_tokens))
    print(f"Perplexity: {ppl.item():.4f}")
    return ppl.item()



def evaluate_zeroshot_downstream_task(args, model):
    print("Evaluating downstream tasks ...")
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    results = simple_evaluate(
        lm,
        tasks=["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "boolq"],
        device=args.device,
    )

    print(results["results"])
    return results["results"]

DEFAULT_DOWNSTREAM_TASK_CONFIGS = (
    {"name": "arc_easy", "task": "arc_easy", "num_fewshot": 0},
    {"name": "arc_challenge", "task": "arc_challenge", "num_fewshot": 0},
    {"name": "hellaswag", "task": "hellaswag", "num_fewshot": 0},
    {"name": "winogrande", "task": "winogrande", "num_fewshot": 0},
    {"name": "piqa", "task": "piqa", "num_fewshot": 0},
    {"name": "boolq", "task": "boolq", "num_fewshot": 0},
    {"name": "race", "task": "race", "num_fewshot": 5},
    {"name": "mmlu", "task": "mmlu", "num_fewshot": 5},
    {"name": "mmlu_pro", "task": "mmlu_pro", "num_fewshot": 5},
    {"name": "gsm8k", "task": "gsm8k", "num_fewshot": 8},
    {"name": "mgsm", "task": "mgsm_direct_en", "num_fewshot": 8},
    {"name": "math", "task": "leaderboard_math_hard", "num_fewshot": 4},
)


def get_default_downstream_task_configs() -> list[dict[str, object]]:
    return [dict(task_config) for task_config in DEFAULT_DOWNSTREAM_TASK_CONFIGS]


def evaluate_downstream_task(
    args,
    model,
    *,
    task_configs: list[dict[str, object]] | None = None,
    lm_cls=None,
    lm_kwargs: dict | None = None,
    simple_evaluate_kwargs: dict | None = None,
    continue_on_error: bool = False,
):
    print("Evaluating downstream tasks ...")

    _patch_lm_eval_transformers_compat()

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tokenizer_name_or_path = getattr(args, "tokenizer_name_or_path", None)
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = getattr(args, "model_name_or_path", None)
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = getattr(args, "model_name", None)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )

    lm_cls = HFLM if lm_cls is None else lm_cls
    lm_kwargs = dict(lm_kwargs or {})
    lm_kwargs.setdefault("batch_size", getattr(args, "batch_size", 1))
    lm = lm_cls(pretrained=model, tokenizer=tokenizer, **lm_kwargs)

    merged_results = {}
    errors: dict[str, str] = {}
    task_configs = task_configs or get_default_downstream_task_configs()
    simple_kwargs = dict(simple_evaluate_kwargs or {})
    simple_kwargs.setdefault("device", args.device)
    simple_kwargs.setdefault("bootstrap_iters", 0)
    simple_kwargs.setdefault("log_samples", False)

    limit = getattr(args, "limit", None)
    if limit is not None:
        simple_kwargs.setdefault("limit", limit)

    for task_config in task_configs:
        logical_name = str(task_config.get("name", task_config["task"]))
        task_name = str(task_config["task"])
        num_fewshot = int(task_config.get("num_fewshot", 0))
        print(f"[downstream] task={task_name} logical_name={logical_name} fewshot={num_fewshot}")
        try:
            res = simple_evaluate(
                lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                **simple_kwargs,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            errors[logical_name] = str(exc)
            print(f"[downstream][ERROR] task={task_name}: {exc}")
            continue

        merged_results.update(res.get("results", {}))
        merged_results.update(res.get("groups", {}))

    if errors:
        merged_results["__errors__"] = errors

    print(merged_results)
    return merged_results

  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="pytorch model path")
    parser.add_argument("--model_name", type=str, required=True, help="transformers model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_len", type=int, default=2048, help="sequence length for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    parser.add_argument("--gptqmodel_dir", type=str, default=None, help="Optional path to the GPTQModel repository")
    parser.add_argument("--eval_downstream", action="store_true", help="Also evaluate downstream tasks (requires lm_eval)")
    parser.add_argument("--eval_c4", action="store_true", help="Also evaluate C4 PPL")
    args = parser.parse_args()


    # evaluate_downstream_task(args, args.model_name, args.model_path)

    print("Loading model:", args.model_name)
    
    if args.model_path is None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", attn_implementation="flash_attention_2")
    else:
        try:
            model = torch.load(args.model_path, weights_only=False)
        except Exception:
            model = _load_gptq_model(args.model_path, repo_dir=args.gptqmodel_dir)

    print("Evaluating PPL...")

    model.eval()
    test_loader = get_wikitext2_testloader(nsamples=None, seed=args.seed, seqlen=args.seq_len, model=args.model_name, batch_size=1)

    ppl = compute_ppl_from_testloader(model, test_loader, device=args.device)
    model_label = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else args.model_name.replace("/", "_")
    print(f"{model_label} WikiText-2 Perplexity: {ppl:.4f}")

    row = {"model": model_label, "wikitext2_ppl": ppl}

    if args.eval_c4:
        print("Evaluating C4 PPL...")
        c4_loader = get_c4_testloader(nsamples=None, seed=args.seed, seqlen=args.seq_len, model=args.model_name, batch_size=1)
        c4_ppl = compute_ppl_from_testloader(model, c4_loader, device=args.device)
        print(f"{model_label} C4 Perplexity: {c4_ppl:.4f}")
        row["c4_ppl"] = c4_ppl

    if args.eval_downstream:
        print("Evaluating Downstream Tasks...")
        dstask_results = evaluate_downstream_task(args, model)
        flat_results = {}
        for task, metrics in dstask_results.items():
            for metric_name, value in metrics.items():
                flat_results[f"{task}_{metric_name}"] = value
        row.update(flat_results)

    df = pd.DataFrame([row])
    results_dir = default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_label}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()



