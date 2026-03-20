import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import os
from tqdm import tqdm, trange
import math

try:
    from .compressed_data import ensure_bqq_root_on_path
    from .datautils import get_wikitext2_trainloader, get_wikitext2_testloader, compute_ppl_from_testloader
    from . import model_loader, binary_quadratic_network
except ImportError:
    from compressed_data import ensure_bqq_root_on_path
    from datautils import get_wikitext2_trainloader, get_wikitext2_testloader, compute_ppl_from_testloader
    import model_loader
    import binary_quadratic_network

ensure_bqq_root_on_path()

import quantizer


import torch
import torch.nn as nn

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to(o, device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    elif hasattr(obj, "_fields"):  # NamedTuple
        return type(obj)(*(recursive_to(getattr(obj, f), device) for f in obj._fields))
    else:
        return obj

class GenericModelParallel(nn.Module):
    def __init__(self, model, gpu_ids=None):
        super().__init__()
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))
        self.devices = [f"cuda:{i}" for i in gpu_ids]
        self.n_gpus = len(self.devices)

        print(f"[GenericModelParallel] Using devices: {self.devices}")

        self.model = model

        # --- Embedding を最初の GPU に固定 ---
        self.embed_tokens = getattr(model, "embed_tokens", None)
        if self.embed_tokens is not None:
            self.embed_tokens = self.embed_tokens.to(self.devices[0])

        # --- 出力 head (lm_head) を最初の GPU に固定 ---
        self.lm_head = getattr(model, "lm_head", None)
        if self.lm_head is not None:
            self.lm_head = self.lm_head.to(self.devices[0])

        # --- メイン層群を分割 ---
        self.layer_groups = self._split_model_layers(model)

    def _split_model_layers(self, model):
        """
        モデル内部を探索し、Sequential/ModuleList 単位で GPU ごとに分割
        Embedding や lm_head は除外する
        """
        layers = []
        for name, module in model.named_children():
            if name in ["embed_tokens", "lm_head"]:
                continue
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                for sub in module:
                    layers.append(sub)
            else:
                layers.append(module)

        n_layers = len(layers)
        per_gpu = (n_layers + self.n_gpus - 1) // self.n_gpus

        groups = []
        for i, dev in enumerate(self.devices):
            start = i * per_gpu
            end = min((i + 1) * per_gpu, n_layers)
            if start >= n_layers:
                break
            group = nn.ModuleList(layers[start:end]).to(dev)
            groups.append(group)
        return groups
    


    def forward(self, input_ids=None, inputs_embeds=None):
        # Embedding
        if inputs_embeds is None:
            assert input_ids is not None
            x = input_ids.to(self.devices[0])
            if self.embed_tokens is not None:
                x = self.embed_tokens(x)
        else:
            x = inputs_embeds.to(self.devices[0])

        # メイン層
        for i, group in enumerate(self.layer_groups):
            dev = self.devices[min(i, len(self.layer_groups) - 1)]
            x = recursive_to(x, dev)
            for layer in group:
                x = layer(x)
                # layer が dict / BaseModelOutput を返した場合は Tensor に変換
                if hasattr(x, "last_hidden_state"):
                    x = x.last_hidden_state
                elif isinstance(x, dict) and "last_hidden_state" in x:
                    x = x["last_hidden_state"]
                elif isinstance(x, (list, tuple)):
                    # まれに tuple で返る場合は先頭を使う
                    x = x[0]

        # lm_head
        if self.lm_head is not None:
            x = recursive_to(x, self.devices[0])
            x = self.lm_head(x)

        # 元の型にラップ
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(logits=x)





def distill_calibration(org_model, quant_model, data_loader, optimizer, epochs, gpu_ids=[0,1,2,3], save_epoch=[2,5,10], model_name=None, bit_width=4, group_size=128, num_step=50000):
    loss_fct = nn.MSELoss()

    # デバイス設定(モデル並列用)
    org_model = GenericModelParallel(org_model, gpu_ids=gpu_ids)
    quant_model = GenericModelParallel(quant_model, gpu_ids=gpu_ids) 
    # org_model = FullModelParallel(org_model, gpu_ids=gpu_ids)
    # quant_model = FullModelParallel(quant_model, gpu_ids=gpu_ids)

    org_model.eval()  # org_modelは固定
    quant_model.train()  # quant_modelを更新

    for epoch in range(epochs):
        for input, target in tqdm(data_loader):

            with torch.no_grad():
                org_logits = org_model(input.to(f'cuda:{gpu_ids[0]}')).logits  # 教師信号 (固定)

            # quant_modelは勾配追跡ON
            quant_logits = quant_model(input.to(f'cuda:{gpu_ids[0]}')).logits

            loss = loss_fct(quant_logits, org_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch:{epoch+1}: loss:{loss}')
        if epoch+1 in save_epoch:
            torch.save(quant_model, os.path.dirname(__file__)+f'/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step-calibrated-{epoch+1}epoch--0.0001lr-2bs.pth')

    return quant_model



# torchrun --nproc_per_node=4 distill_ddp.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def distill_calibration_ddp(org_model, quant_model, data_loader, optimizer, epochs):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    org_model.to(device).eval()
    quant_model = DDP(quant_model.to(device), device_ids=[rank])

    for epoch in range(epochs):
        for input, target in tqdm(data_loader):
            with torch.no_grad():
                org_logits = org_model(input.to(device)).logits

            quant_logits = quant_model(input.to(device)).logits
            loss = nn.functional.mse_loss(quant_logits, org_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()
    return quant_model

import torch
from tqdm import tqdm
import os

@torch.no_grad()
def save_teacher_logits_chunked(
    org_model,
    data_loader,
    save_dir,
    device="cuda:0",
    chunk_size=1000,  # バッチ数単位
    dtype=torch.float16,  # 精度を落として容量削減
):
    os.makedirs(save_dir, exist_ok=True)
    # すでに保存ファイルがあるか確認
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]
    if len(existing_files) > 0:
        print(f"⚠️ 既に教師データが {save_dir} に存在するためスキップします ({len(existing_files)} ファイル)")
        return
    org_model.to(device)
    org_model.eval()

    chunk_inputs, chunk_logits = [], []
    chunk_idx, total_samples = 0, 0

    for i, batch in enumerate(tqdm(data_loader, desc="Saving teacher logits")):
        inputs, _ = batch
        inputs = inputs.to(device)
        logits = org_model(inputs).logits.to(dtype).cpu()

        chunk_inputs.append(inputs.cpu())
        chunk_logits.append(logits)
        total_samples += inputs.size(0)

        # チャンク単位で保存
        if (i + 1) % chunk_size == 0:
            save_path = os.path.join(save_dir, f"teacher_chunk_{chunk_idx:04d}.pt")
            torch.save({
                "inputs": torch.cat(chunk_inputs, dim=0),
                "logits": torch.cat(chunk_logits, dim=0),
            }, save_path)
            print(f"Saved {save_path} ({total_samples} samples so far)")

            chunk_idx += 1
            chunk_inputs.clear()
            chunk_logits.clear()
            torch.cuda.empty_cache()

    # 残りを保存
    if chunk_inputs:
        save_path = os.path.join(save_dir, f"teacher_chunk_{chunk_idx:04d}.pt")
        torch.save({
            "inputs": torch.cat(chunk_inputs, dim=0),
            "logits": torch.cat(chunk_logits, dim=0),
        }, save_path)
        print(f"Saved {save_path} (final chunk, total {total_samples} samples)")

    print("✅ All teacher logits saved.")



import torch
import os
import glob
from torch.utils.data import Dataset

class TeacherLogitsDataset(Dataset):
    def __init__(self, save_dir):
        self.files = sorted(glob.glob(os.path.join(save_dir, "teacher_chunk_*.pt")))
        assert self.files, f"No chunks found in {save_dir}"

        self.chunk_sizes = []
        for f in tqdm(self.files):
            data = torch.load(f, map_location="cpu")
            self.chunk_sizes.append(len(data["inputs"]))
            del data
        self.cumulative_sizes = torch.cumsum(torch.tensor(self.chunk_sizes), dim=0)

    def __len__(self):
        return self.cumulative_sizes[-1].item()

    def _get_chunk(self, chunk_idx):
        if not hasattr(self, "_cache"):
            self._cache = {}
        if chunk_idx not in self._cache:
            self._cache = {chunk_idx: torch.load(self.files[chunk_idx], map_location="cpu")}
        return self._cache[chunk_idx]

    def __getitem__(self, idx):
        chunk_idx = torch.searchsorted(self.cumulative_sizes, torch.tensor(idx), right=True).item()
        prev_cum = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1].item()
        local_idx = idx - prev_cum
        data = self._get_chunk(chunk_idx)
        return data["inputs"][local_idx], data["logits"][local_idx]




# --- 3. 蒸留ループ ---
def distill_from_teacher_dataset(quant_model, dataloader, optimizer, epochs, device="cuda"):
    loss_fct = nn.MSELoss()
    quant_model.to(device)
    quant_model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for inp, teacher_logits in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inp, teacher_logits = inp.to(device), teacher_logits.to(device).float()
            quant_logits = quant_model(inp).logits
            loss = loss_fct(quant_logits, teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | avg_loss: {avg_loss:.4f}")
    return quant_model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--model_path', type=str, required=True, help='Path to the original model')
    parser.add_argument('--bit_width', type=int, choices=[2,3,4], required=True, help='Bit width for quantization')
    parser.add_argument('--group_size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--num_step', type=int, default=50000, help='Number of steps for quantization')
    parser.add_argument('--train_nsamples', type=int, default=None, help='Number of training samples (None for all)')
    parser.add_argument('--test_nsamples', type=int, default=None, help='Number of test samples (None for all)')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--epochs_per_save', type=int, default=2, help='Epochs to save the model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    return args

def ddp_main():
    args = parse_args()
    group_size = args.group_size
    num_step = args.num_step
    seq_len = args.seq_len
    test_nsamples = args.test_nsamples
    train_nsamples = args.train_nsamples
    batch_size = args.batch_size
    vocab_size = 151936  # Qwen系の語彙数（例）
    lr = args.lr
    epochs = args.epochs
    epochs_per_save = args.epochs_per_save
    device = f"cuda:{args.local_rank}"
    model_path = args.model_path
    model_name = os.path.basename(model_path)
    seed = args.seed

    # make distillation data (save logits of the original model)
    train_loader = get_wikitext2_trainloader(nsamples=train_nsamples, seed=seed, seqlen=seq_len, batch_size=batch_size, model=model_path)
    for bit_width in [args.bit_width]:
        # load original model
        o_model = AutoModelForCausalLM.from_pretrained(model_path)
        q_model_path = (
            os.path.dirname(__file__)
            + f"/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step.pth"
        )
        q_model = torch.load(q_model_path, weights_only=False)

        # calibration through DDP
        optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)
        for i in range(epochs // epochs_per_save):
            q_model = distill_calibration_ddp(o_model, q_model, train_loader, optimizer, epochs=epochs_per_save)
            torch.save(q_model.module, os.path.dirname(__file__)+f'/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step-calibrated-{(i+1)*epochs_per_save}epoch.pth')
        del train_loader, o_model
        torch.cuda.empty_cache()
        # check calibrated quantized model PPL
        test_loader = get_wikitext2_testloader(nsamples=test_nsamples, seed=0, seqlen=seq_len, model=model_path)
        ppl = compute_ppl_from_testloader(q_model.module, test_loader, device=device)
        print(f"{epochs}Epochs Calibrated Quantized Model PPL: {ppl:.3f}")

        del testloader, q_model
        torch.cuda.empty_cache()

def from_cache_main():
    args = parse_args()
    group_size = args.group_size
    num_step = args.num_step
    seq_len = args.seq_len
    test_nsamples = args.test_nsamples
    train_nsamples = args.train_nsamples
    batch_size = args.batch_size
    vocab_size = 151936  # Qwen系の語彙数（例）
    lr = args.lr
    epochs = args.epochs
    epochs_per_save = args.epochs_per_save
    device = f"cuda:{args.local_rank}"
    model_path = args.model_path
    model_name = os.path.basename(model_path)
    seed = args.seed

    # load original model
    o_model = AutoModelForCausalLM.from_pretrained(model_path)
    train_loader = get_wikitext2_trainloader(nsamples=train_nsamples, seed=0, seqlen=seq_len, batch_size=batch_size, model=model_path)
    save_teacher_logits_chunked(o_model, train_loader, os.path.dirname(__file__) + f"/distill_data/{model_name}", device=device, chunk_size=10, dtype=torch.float16)

    for bit_width in [args.bit_width]:
        q_model_path = (
            os.path.dirname(__file__)
            + f"/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step.pth"
        )
        q_model = torch.load(q_model_path, weights_only=False)

        # calibration thorough oroginal model logits
        teacher_dataset = TeacherLogitsDataset(os.path.dirname(__file__) + f"/distill_data/{model_name}")
        teacher_dataloader = torch.utils.data.DataLoader(teacher_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)
        for i in range(epochs // epochs_per_save):
            q_model = distill_from_teacher_dataset(q_model, teacher_dataloader, optimizer, epochs=epochs_per_save, device=device)
            torch.save(q_model, os.path.dirname(__file__)+f'/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step-calibrated-{(i+1)*epochs_per_save}epoch.pth')

        # check calibrated quantized model PPL
        testloader = get_wikitext2_testloader(nsamples=test_nsamples, seed=0, seqlen=seq_len, model=model_path)
        ppl = compute_ppl_from_testloader(q_model, testloader, device=device)
        print(f"{epochs}Epochs Calibrated Quantized Model PPL: {ppl:.3f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the original model')
    parser.add_argument(
    '--bit_width',
    type=int,
    choices=[2, 3, 4],
    nargs='+',                    # ← これがポイント！
    required=True,
    help='Bit widths for quantization (e.g., --bit_width 2 3 4)'
    )
    parser.add_argument('--group_size', type=int, default=128, help='Group size for quantization')
    parser.add_argument('--num_step', type=int, default=50000, help='Number of steps for quantization')
    parser.add_argument('--train_nsamples', type=int, default=None, help='Number of training samples (None for all)')
    parser.add_argument('--test_nsamples', type=int, default=None, help='Number of test samples (None for all)')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_epoch', type=int, nargs='+',default=0, help='Epochs to save the model')
    parser.add_argument('--gpu_ids', type=int, nargs='+',default=0, help='GPU IDs to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    group_size = args.group_size
    num_step = args.num_step
    seq_len = args.seq_len
    test_nsamples = args.test_nsamples
    train_nsamples = args.train_nsamples
    batch_size = args.batch_size
    vocab_size = 151936  # Qwen系の語彙数（例）
    lr = args.lr
    epochs = args.epochs
    model_path = args.model_path
    model_name = os.path.basename(model_path)
    seed = args.seed

    # make distillation data (save logits of the original model)
    train_loader = get_wikitext2_trainloader(nsamples=train_nsamples, seed=seed, seqlen=seq_len, batch_size=batch_size, model=model_path)
    for bit_width in args.bit_width:
        # load original model
        o_model = AutoModelForCausalLM.from_pretrained(model_path)
        q_model_path = (
            os.path.dirname(__file__)
            + f"/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step.pth"
        )
        q_model = torch.load(q_model_path, weights_only=False)

        # calibration through DDP
        optimizer = torch.optim.Adam(q_model.parameters(), lr=lr)
        q_model = distill_calibration(o_model, q_model, train_loader, optimizer, epochs=epochs, gpu_ids=args.gpu_ids, save_epoch=args.save_epoch, model_name=model_name, bit_width=bit_width, group_size=group_size, num_step=num_step)
        del train_loader, o_model
        torch.cuda.empty_cache()
        # check calibrated quantized model PPL
        test_loader = get_wikitext2_testloader(nsamples=test_nsamples, seed=0, seqlen=seq_len, model=model_path)
        ppl = compute_ppl_from_testloader(q_model, test_loader, device=f'cuda:{args.gpu_ids[0]}')
        print(f"{epochs}Epochs Calibrated Quantized Model PPL: {ppl:.3f}")

        del test_loader, q_model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # from_cache_main()
    # ddp_main()
    # main()


    model_path = "Qwen/Qwen2.5-1.5B"
    model_name = os.path.basename(model_path)
    group_size = 128
    num_step = 50000
    seq_len = 2048
    test_nsamples = None
    train_nsamples = None
    batch_size = 2
    gpu_id = 1
    for bit_width in [4, 3, 2]:
        # load original model
        o_model = AutoModelForCausalLM.from_pretrained(model_path)
        q_model_path = (
            os.path.dirname(__file__)
            + f"/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step.pth"
        )
        q_model = torch.load(q_model_path, weights_only=False)

        # check calibrated quantized model PPL
        test_loader = get_wikitext2_testloader(nsamples=test_nsamples, seed=0, seqlen=seq_len, model=model_path)
        ppl = compute_ppl_from_testloader(q_model, test_loader, device=f'cuda:{gpu_id}')
        print(f"{10}Epochs Calibrated Quantized Model PPL: {ppl:.3f}")

