import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm
import pandas as pd
from lm_eval import evaluator
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import glob
import argparse


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



def evaluate_downstream_task(args, model):
    print("Evaluating downstream tasks ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    lm = HFLM(pretrained=model,
                tokenizer=tokenizer)


    results = simple_evaluate(
        lm,
        tasks=["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "boolq"],
        device=args.device,
    )
    
    print(results["results"])
    return results["results"]
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="pytorch model path")
    parser.add_argument("--model_name", type=str, required=True, help="transformers model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_len", type=int, default=2048, help="sequence length for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    args = parser.parse_args()


    # evaluate_downstream_task(args, args.model_name, args.model_path)

    print("Loading model:", args.model_name)
    
    if args.model_path is None:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", attn_implementation="flash_attention_2")
    else:
        try:
            model = torch.load(args.model_path, weights_only=False)
        except Exception:
            import sys
            sys.path.append("/work2/k-kuroki/GPTQModel")
            from gptqmodel import GPTQModel, QuantizeConfig
            model = GPTQModel.load(args.model_path)

    print("Evaluating PPL...")

    model.eval()
    from datautils import get_wikitext2_testloader
    import pandas as pd
    import os
    test_loader = get_wikitext2_testloader(nsamples=None, seed=args.seed, seqlen=args.seq_len, model=args.model_name, batch_size=1)

    ppl = compute_ppl_from_testloader(model, test_loader, device=args.device)
    print(f"{os.path.basename(args.model_path)} WikiText-2 Perplexity: {ppl:.4f}")

    print("Evaluating Downstream Tasks...")
    dstask_results = evaluate_downstream_task(args, model)

    # ===== make results ======
    # dstask_results は {task_name: {metric_name: value, ...}, ...} なのでフラット化
    flat_results = {}
    for task, metrics in dstask_results.items():
        for metric_name, value in metrics.items():
            flat_results[f"{task}_{metric_name}"] = value

    # 1行の DataFrame にまとめる
    row = {"model": os.path.splitext(os.path.basename(args.model_path))[0], "PPL": ppl}
    row.update(flat_results)
    df = pd.DataFrame([row])  # リストで囲むと1行 DataFrame

    # CSV 保存
    df.to_csv(f"/work2/k-kuroki/BQQLLM/results/{os.path.basename(args.model_path)}.csv", index=False)


if __name__ == "__main__":
    # /work2/k-kuroki/BQQLLM/quantized_model_data/Qwen2.5-0.5B/Qwen2.5-0.5B-2bit-128gs-50000step-calibrated.pth
    main()




