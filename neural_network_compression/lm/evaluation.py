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
from build_bqq_model import dequantize_bqq_model, load_bqq_as_fp


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


# Task configs: 0-shot commonsense + few-shot reasoning/knowledge
DEFAULT_DOWNSTREAM_TASK_CONFIGS = (
    {"name": "arc_easy",      "task": "arc_easy",               "num_fewshot": 0},
    {"name": "arc_challenge", "task": "arc_challenge",           "num_fewshot": 0},
    {"name": "hellaswag",     "task": "hellaswag",               "num_fewshot": 0},
    {"name": "winogrande",    "task": "winogrande",              "num_fewshot": 0},
    {"name": "piqa",          "task": "piqa",                    "num_fewshot": 0},
    {"name": "boolq",         "task": "boolq",                   "num_fewshot": 0},
    {"name": "race",          "task": "race",                    "num_fewshot": 5},
    {"name": "mmlu",          "task": "mmlu",                    "num_fewshot": 5},
    {"name": "mmlu_pro",      "task": "mmlu_pro",                "num_fewshot": 5},
    {"name": "gsm8k",         "task": "gsm8k",                   "num_fewshot": 8},
    {"name": "mgsm",          "task": "mgsm_direct_en",          "num_fewshot": 8},
    {"name": "math",          "task": "leaderboard_math_hard",   "num_fewshot": 4},
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
    continue_on_error: bool = True,
):
    print("Evaluating downstream tasks ...")
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tokenizer_name_or_path = (
        getattr(args, "tokenizer_name_or_path", None)
        or getattr(args, "model_name_or_path", None)
        or getattr(args, "model_name", None)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True, use_fast=True
    )

    lm_cls = HFLM if lm_cls is None else lm_cls
    lm_kwargs = dict(lm_kwargs or {})
    lm_kwargs.setdefault("batch_size", getattr(args, "batch_size", 1))
    lm = lm_cls(pretrained=model, tokenizer=tokenizer, **lm_kwargs)

    merged_results: dict = {}
    errors: dict[str, str] = {}
    task_configs = task_configs or get_default_downstream_task_configs()
    simple_kwargs = dict(simple_evaluate_kwargs or {})
    simple_kwargs.setdefault("bootstrap_iters", 0)
    simple_kwargs.setdefault("log_samples", False)

    limit = getattr(args, "limit", None)
    if limit is not None:
        simple_kwargs.setdefault("limit", limit)

    for task_config in task_configs:
        logical_name = str(task_config.get("name", task_config["task"]))
        task_name = str(task_config["task"])
        num_fewshot = int(task_config.get("num_fewshot", 0))
        print(f"[downstream] task={task_name} ({num_fewshot}-shot)")
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


def _merge_save_csv(out_path: Path, row: dict) -> None:
    """Merge row into existing CSV (or create new) and save."""
    if out_path.exists():
        existing = pd.read_csv(out_path).to_dict(orient="records")[0]
        existing.update(row)
        row = existing
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


def _task_already_done(out_path: Path, lm_task_name: str) -> bool:
    """Return True if the CSV already contains results for this lm_eval task."""
    if not out_path.exists():
        return False
    try:
        df = pd.read_csv(out_path)
        cols = [c for c in df.columns if c.startswith(f"{lm_task_name}_")]
        return bool(cols) and df[cols[0]].notna().all()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="pytorch model path")
    parser.add_argument("--model_name", type=str, required=True, help="transformers model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_len", type=int, default=2048, help="sequence length for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for lm_eval")
    parser.add_argument("--gptqmodel_dir", type=str, default=None, help="Optional path to the GPTQModel repository")
    parser.add_argument("--eval_downstream", action="store_true", help="Also evaluate downstream tasks (requires lm_eval)")
    parser.add_argument("--downstream_tasks", type=str, default=None,
                        help="Comma-separated subset of task names to run (default: all DEFAULT_DOWNSTREAM_TASK_CONFIGS)")
    parser.add_argument("--eval_c4", action="store_true", help="Also evaluate C4 PPL (full validation set)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a single downstream task by logical name and save immediately. "
                             "Skips PPL. Exits early if result already present in CSV.")
    args = parser.parse_args()

    model_label = (
        os.path.splitext(os.path.basename(args.model_path))[0]
        if args.model_path
        else args.model_name.replace("/", "_")
    )

    results_dir = default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_label}.csv"

    # --task mode: single task, immediate save, skip if already done
    if args.task is not None:
        all_configs = get_default_downstream_task_configs()
        task_config = next((c for c in all_configs if c["name"] == args.task), None)
        if task_config is None:
            valid = [c["name"] for c in all_configs]
            raise ValueError(f"Unknown task '{args.task}'. Valid names: {valid}")

        lm_task_name = str(task_config["task"])
        if _task_already_done(out_path, lm_task_name):
            print(f"Task '{args.task}' already done in {out_path}, skipping.")
            return

        print("Loading model:", args.model_name)
        if args.model_path is None:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name, device_map="auto", attn_implementation="flash_attention_2"
            )
        else:
            try:
                model = load_bqq_as_fp(args.model_path, args.model_name)
            except Exception:
                model = _load_gptq_model(args.model_path, repo_dir=args.gptqmodel_dir)
        model.eval()

        print(f"[task mode] running task={lm_task_name} ({task_config['num_fewshot']}-shot)")
        dstask_results = evaluate_downstream_task(args, model, task_configs=[task_config])
        row: dict = {"model": model_label}
        for task, metrics in dstask_results.items():
            if task == "__errors__":
                continue
            for metric_name, value in metrics.items():
                row[f"{task}_{metric_name}"] = value

        _merge_save_csv(out_path, row)
        return

    print("Loading model:", args.model_name)

    if args.model_path is None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map="auto", attn_implementation="flash_attention_2"
        )
    else:
        try:
            model = load_bqq_as_fp(args.model_path, args.model_name)
        except Exception:
            model = _load_gptq_model(args.model_path, repo_dir=args.gptqmodel_dir)

    model.eval()

    test_loader = get_wikitext2_testloader(
        nsamples=None, seed=args.seed, seqlen=args.seq_len, model=args.model_name, batch_size=1
    )
    ppl = compute_ppl_from_testloader(model, test_loader, device=args.device)
    print(f"{model_label} WikiText-2 Perplexity: {ppl:.4f}")
    row = {"model": model_label, "wikitext2_ppl": ppl}

    if args.eval_c4:
        print("Evaluating C4 PPL...")
        c4_loader = get_c4_testloader(
            nsamples=None, seed=args.seed, seqlen=args.seq_len, model=args.model_name, batch_size=1
        )
        c4_ppl = compute_ppl_from_testloader(model, c4_loader, device=args.device)
        print(f"{model_label} C4 Perplexity: {c4_ppl:.4f}")
        row["c4_ppl"] = c4_ppl

    if args.eval_downstream:
        print("Evaluating Downstream Tasks...")
        task_configs = None
        if args.downstream_tasks:
            names = {t.strip() for t in args.downstream_tasks.split(",")}
            task_configs = [c for c in get_default_downstream_task_configs() if c["name"] in names]
            print(f"  Running subset: {[c['name'] for c in task_configs]}")
        dstask_results = evaluate_downstream_task(args, model, task_configs=task_configs)
        flat_results = {}
        for task, metrics in dstask_results.items():
            if task == "__errors__":
                continue
            for metric_name, value in metrics.items():
                flat_results[f"{task}_{metric_name}"] = value
        row.update(flat_results)

    _merge_save_csv(out_path, row)


if __name__ == "__main__":
    main()
