import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm

_PTB_PARQUET_FILES = {
    "train": "https://huggingface.co/datasets/FALcon6/ptb_text_only/resolve/main/penn_treebank/train/0000.parquet",
    "test": "https://huggingface.co/datasets/FALcon6/ptb_text_only/resolve/main/penn_treebank/test/0000.parquet",
    "validation": "https://huggingface.co/datasets/FALcon6/ptb_text_only/resolve/main/penn_treebank/validation/0000.parquet",
}


def _load_ptb_split(split):
    try:
        return load_dataset("ptb_text_only", "penn_treebank", split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        return load_dataset("parquet", data_files={split: _PTB_PARQUET_FILES[split]}, split=split)





def _load_streaming_split(repo_ids, split_candidates):
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]
    last_exc = None
    for repo_id in repo_ids:
        for split in split_candidates:
            try:
                return load_dataset(repo_id, split=split, streaming=True)
            except Exception as exc:
                last_exc = exc
    raise RuntimeError(f"Failed to load any split from repos {repo_ids}: {split_candidates}") from last_exc


def _extract_text_field(example):
    for key in ("text", "raw_content", "content"):
        value = example.get(key) if isinstance(example, dict) else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _build_streaming_loader_from_text_iter(
    dataset_iter,
    model,
    *,
    nsamples=None,
    seed=0,
    seqlen=2048,
    tokenizer=None,
    batch_size=1,
    shuffle=True,
    mask_labels=False,
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    target_chunks = nsamples if nsamples is not None else 128
    target_tokens = max(int(target_chunks), 1) * seqlen
    ids_list = []
    total_tokens = 0

    for example in dataset_iter:
        text = _extract_text_field(example)
        if not text:
            continue
        out = tokenizer(text + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids = out.get("input_ids", [])
        if not ids:
            continue
        tensor_ids = torch.tensor(ids, dtype=torch.long)
        ids_list.append(tensor_ids)
        total_tokens += tensor_ids.numel()
        if total_tokens >= target_tokens:
            break

    if not ids_list:
        raise ValueError("Empty streamed dataset after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    if num_chunks == 0:
        raise ValueError(f"Not enough streamed tokens to build one chunk of length {seqlen}.")

    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is not None and nsamples < num_chunks:
        random.seed(seed)
        indices = random.sample(range(num_chunks), nsamples)
        input_ids = input_ids[indices]

    tars = input_ids.clone()
    if mask_labels:
        tars[:, :-1] = -100

    dataset = TensorDataset(input_ids, tars)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_slimpajama_trainloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1, shuffle=True, mask_labels=False):
    traindata = _load_streaming_split(["DKYoon/SlimPajama-6B", "cerebras/SlimPajama-627B", "MBZUAI-LLM/SlimPajama-627B-DC"], ["train"])
    return _build_streaming_loader_from_text_iter(
        traindata, model, nsamples=nsamples, seed=seed, seqlen=seqlen,
        tokenizer=tokenizer, batch_size=batch_size, shuffle=shuffle, mask_labels=mask_labels,
    )


def get_slimpajama_testloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1):
    evaldata = _load_streaming_split(["DKYoon/SlimPajama-6B", "cerebras/SlimPajama-627B", "MBZUAI-LLM/SlimPajama-627B-DC"], ["validation", "test", "train"])
    return _build_streaming_loader_from_text_iter(
        evaldata, model, nsamples=nsamples, seed=seed, seqlen=seqlen,
        tokenizer=tokenizer, batch_size=batch_size, shuffle=False, mask_labels=False,
    )



def get_wikitext2_trainloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1, shuffle=True, mask_labels=False):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    ids_list = []
    for t in traindata["text"]:
        if not t:
            continue
        out = tokenizer(t + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty WikiText-2 train split after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is None or nsamples >= num_chunks:
        indices = list(range(num_chunks))
    else:
        random.seed(seed)
        indices = random.sample(range(num_chunks), nsamples)

    inps = input_ids[indices]
    tars = inps.clone()
    if mask_labels:
        tars[:, :-1] = -100
    dataset = TensorDataset(inps, tars)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)





def get_wikitext2_testloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1):
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    if tokenizer is None:
        print("Loading tokenizer...", model)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    print("Tokenizing test data (streaming, no overlength warning)...")

    # 1) 行ごとにtokenizeしてinput_idsを足していく（巨大な1回tokenizeを避ける）
    ids_list = []
    for t in testdata["text"]:
        if not t:
            continue
        out = tokenizer(t + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty test split after filtering.")

    input_ids = torch.cat(ids_list, dim=0)  # 1D (T,)

    # 2) seqlenごとに分割
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    # 3) nsamplesだけ抽出
    if nsamples is not None and nsamples < num_chunks:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(num_chunks, generator=g)[:nsamples]
        input_ids = input_ids[indices]

    dataset = TensorDataset(input_ids, input_ids)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_ptb_trainloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1, shuffle=True, mask_labels=False):
    traindata = _load_ptb_split("train")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    ids_list = []
    for s in traindata["sentence"]:
        if not s or not str(s).strip():
            continue
        out = tokenizer(str(s).strip() + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty PTB train split after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is None or nsamples >= num_chunks:
        indices = list(range(num_chunks))
    else:
        random.seed(seed)
        indices = random.sample(range(num_chunks), nsamples)

    inps = input_ids[indices]
    tars = inps.clone()
    if mask_labels:
        tars[:, :-1] = -100
    dataset = TensorDataset(inps, tars)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_ptb_testloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1):
    testdata = _load_ptb_split("test")

    if tokenizer is None:
        print("Loading tokenizer...", model)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    ids_list = []
    for s in testdata["sentence"]:
        if not s or not str(s).strip():
            continue
        out = tokenizer(str(s).strip() + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty PTB test split after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is not None and nsamples < num_chunks:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(num_chunks, generator=g)[:nsamples]
        input_ids = input_ids[indices]

    dataset = TensorDataset(input_ids, input_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_c4_trainloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1, shuffle=True, mask_labels=False):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    # 1件ずつトークナイズして結合（一括 join を避ける）
    ids_list = []
    for t in traindata["text"]:
        if not t or not str(t).strip():
            continue
        out = tokenizer(str(t).strip() + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty C4 train split after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is None or nsamples >= num_chunks:
        indices = list(range(num_chunks))
    else:
        random.seed(seed)
        indices = random.sample(range(num_chunks), nsamples)

    inps = input_ids[indices]
    tars = inps.clone()
    if mask_labels:
        tars[:, :-1] = -100
    dataset = TensorDataset(inps, tars)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_c4_testloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1):
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    if tokenizer is None:
        print("Loading tokenizer...", model)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    ids_list = []
    for t in valdata["text"]:
        if not t or not str(t).strip():
            continue
        out = tokenizer(str(t).strip() + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty C4 validation split after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is not None and nsamples < num_chunks:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(num_chunks, generator=g)[:nsamples]
        input_ids = input_ids[indices]

    dataset = TensorDataset(input_ids, input_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)






def get_redpajama1t_trainloader(model, nsamples=None, seed=0, seqlen=2048, tokenizer=None, batch_size=1, shuffle=True, mask_labels=False):
    # RedPajama-Data-1T-Sample was removed from the Hub; load V2 English sample documents
    # directly via hf_hub_download to avoid the deprecated dataset script.
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset as _load_dataset

    EN_SAMPLE_FILES = [
        f"sample/documents/2023-06/{i:04d}/en_{part}.json.gz"
        for i in range(10)
        for part in ("head", "middle")
    ]

    local_paths = []
    for remote in EN_SAMPLE_FILES:
        try:
            p = hf_hub_download(
                repo_id="togethercomputer/RedPajama-Data-V2",
                filename=remote,
                repo_type="dataset",
            )
            local_paths.append(p)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download RedPajama-Data-V2 file {remote}: {exc}"
            ) from exc

    traindata = _load_dataset("json", data_files=local_paths, split="train")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    ids_list = []
    for t in traindata["raw_content"]:
        if not t or not str(t).strip():
            continue
        out = tokenizer(str(t).strip() + "\n\n", add_special_tokens=False, return_attention_mask=False)
        ids_list.append(torch.tensor(out["input_ids"], dtype=torch.long))

    if len(ids_list) == 0:
        raise ValueError("Empty RedPajama-Data-V2 sample after filtering.")

    input_ids = torch.cat(ids_list, dim=0)
    num_chunks = input_ids.numel() // seqlen
    input_ids = input_ids[: num_chunks * seqlen]
    input_ids = input_ids.view(num_chunks, seqlen)

    if nsamples is None or nsamples >= num_chunks:
        indices = list(range(num_chunks))
    else:
        random.seed(seed)
        indices = random.sample(range(num_chunks), nsamples)

    inps = input_ids[indices]
    tars = inps.clone()
    if mask_labels:
        tars[:, :-1] = -100
    dataset = TensorDataset(inps, tars)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    model="",
    tokenizer=None,
    batch_size=1,
):
    """
    Return (train_loader, test_loader) as DataLoaders for the given dataset name.
    name: 'wikitext2', 'ptb', or 'c4' (substring match).
    """
    if "wikitext2" in name:
        train_loader = get_wikitext2_trainloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size, shuffle=True,
        )
        test_loader = get_wikitext2_testloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size,
        )
        return train_loader, test_loader
    if "ptb" in name:
        train_loader = get_ptb_trainloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size, shuffle=True,
        )
        test_loader = get_ptb_testloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size,
        )
        return train_loader, test_loader
    if "c4" in name:
        train_loader = get_c4_trainloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size, shuffle=True,
        )
        test_loader = get_c4_testloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size,
        )
        return train_loader, test_loader
    if "redpajama1t" in name:
        train_loader = get_redpajama1t_trainloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size, shuffle=True,
        )
        test_loader = get_wikitext2_testloader(
            model, nsamples=nsamples, seed=seed, seqlen=seqlen,
            tokenizer=tokenizer, batch_size=batch_size,
        )
        return train_loader, test_loader
    if "slimpajama" in name:
        try:
            train_loader = get_slimpajama_trainloader(
                model, nsamples=nsamples, seed=seed, seqlen=seqlen,
                tokenizer=tokenizer, batch_size=batch_size, shuffle=True,
            )
            test_loader = get_slimpajama_testloader(
                model, nsamples=nsamples, seed=seed, seqlen=seqlen,
                tokenizer=tokenizer, batch_size=batch_size,
            )
            return train_loader, test_loader
        except Exception as exc:
            print(f"[WARN] SlimPajama loader failed: {exc}")
            print("[WARN] Falling back to C4.")
            train_loader = get_c4_trainloader(
                model, nsamples=nsamples, seed=seed, seqlen=seqlen,
                tokenizer=tokenizer, batch_size=batch_size, shuffle=True,
            )
            test_loader = get_c4_testloader(
                model, nsamples=nsamples, seed=seed, seqlen=seqlen,
                tokenizer=tokenizer, batch_size=batch_size,
            )
            return train_loader, test_loader
    raise ValueError(f"Unknown dataset name: {name!r}. Use 'wikitext2', 'ptb', 'c4', 'redpajama1t', or 'slimpajama'.")
