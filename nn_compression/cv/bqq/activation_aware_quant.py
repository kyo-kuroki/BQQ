import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from binary_quadratic_network import SymQuantSTE, BQQLinear
import re
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import inspect
import copy
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
CV_DIR = SCRIPT_DIR.parent
BQQ_ROOT = CV_DIR.parent.parent
UTILS_DIR = CV_DIR / "utils"

for path in (BQQ_ROOT, UTILS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from build_dataset import get_imagenet
from build_model import get_model

def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """
    module_name: "model.layers.0.mlp.dense_h_to_4h" みたいなドット区切り
    """
    curr = model
    for name in module_name.split("."):
        if name.isdigit():
            curr = curr[int(name)]
        else:
            curr = getattr(curr, name)
    return curr


def get_parent_and_child_name(model: nn.Module, module_name: str):
    names = module_name.split(".")
    parent = model
    for name in names[:-1]:
        if name.isdigit():
            parent = parent[int(name)]
        else:
            parent = getattr(parent, name)
    child_name = names[-1]
    return parent, child_name


def replace_linear_with_symquant(model: nn.Module, module_name: str, sym_layer: nn.Module):
    parent, child_name = get_parent_and_child_name(model, module_name)
    setattr(parent, child_name, sym_layer)





def collect_linear_io_from_loader(
    model: nn.Module,
    layer_name: str,
    loader,
    device: str = None,
    max_batches: int = None,
):
    layer = get_module_by_name(model, layer_name)

    x_buf = []
    y_buf = []

    def hook_fn(module, input, output):
        x = input[0].detach().cpu()
        y = output.detach().cpu()
        x_buf.append(x.reshape(-1, x.shape[-1]))
        y_buf.append(y.reshape(-1, y.shape[-1]))

    handle = layer.register_forward_hook(hook_fn)
    model.eval()

    is_sharded = getattr(model, "hf_device_map", None) is not None

    if device is not None:
        entry_device = torch.device(device)
    elif is_sharded:
        entry_device = None
    else:
        entry_device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # loader が (inputs, targets) でも inputs 単体でも両対応
            if isinstance(batch, (tuple, list)):
                inp = batch[0]
            elif isinstance(batch, dict):
                # 画像モデルではまず来ないが一応
                inp = batch.get("pixel_values", None) or batch.get("inputs", None)
                if inp is None:
                    raise ValueError(f"Unsupported dict batch keys: {list(batch.keys())}")
            else:
                inp = batch

            if entry_device is not None:
                inp = inp.to(entry_device, non_blocking=True)

            _ = model(inp)

            if max_batches is not None and (i + 1) >= max_batches:
                break

    handle.remove()

    if len(x_buf) == 0:
        raise RuntimeError(
            f"No activations were captured for layer '{layer_name}'. "
            "Check the layer_name and that the layer is executed in forward."
        )

    X = torch.cat(x_buf, dim=0)
    Y = torch.cat(y_buf, dim=0)
    return X, Y






def collect_block_io_from_loader(
    model: nn.Module,
    block_name: str,
    loader,
    device: str = None,
    max_batches: int = None,
):
    block = get_module_by_name(model, block_name)

    inputs = []
    outputs = []

    def hook_fn(module, input, output):
        x = input[0].detach().cpu()
        y = output.detach().cpu()
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        inputs.append(x)
        outputs.append(y)

    handle = block.register_forward_hook(hook_fn)

    model.eval()

    # 分散モデル (device_map="auto") かどうか
    is_sharded = getattr(model, "hf_device_map", None) is not None

    if device is not None:
        entry_device = torch.device(device)
    elif is_sharded:
        # ★ 分散モデルのときは CPU のまま渡して Accelerate に任せる
        entry_device = None
    else:
        # 単一 GPU モデルのときだけ先頭パラメータの device に送る
        entry_device = next(model.parameters()).device

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if entry_device is not None:
                inputs = inputs.to(entry_device)

            _ = model(inputs)

            if max_batches is not None and (i + 1) >= max_batches:
                break

    handle.remove()

    X_block = torch.cat(inputs, dim=0)
    Y_block = torch.cat(outputs, dim=0)
    return X_block, Y_block




def get_block_name_from_layer_name(layer_name: str):
    """
    例:
      'model.layers.0.mlp.down_proj' -> 'model.layers.0'
      'model.decoder.layers.5.self_attn.q_proj' -> 'model.decoder.layers.5'

    マッチしなければ None を返す。
    """
    m = re.search(r"^(.*?layers\.\d+)\.", layer_name)
    if m:
        return m.group(1)
    return None



def train_symquant_block(
    X_block: torch.Tensor,
    Y_block: torch.Tensor,
    block: nn.Module,
    rotary_emb: nn.Module | None,
    seqlen: int,
    epochs: int = 50,
    batch_size: int = 2048,  # トークン数ベースのバッチサイズ
    lr: float = 5e-4,
    device: str | None = None,
    max_grad_norm: float = 0.1,
    cast_block_to_fp32: bool = True,
):
    """
    1つの Transformer Block (Llama/Qwen の DecoderLayer を想定) を、
    ブロック入出力 (X_block, Y_block) に対して
    NMSE || Y_block - block(X_block) ||^2 を最小化するように微調整する。

    device_map="auto" / accelerate のフックと衝突しないように、
    渡された block を deepcopy して「スタンドアロン版 block」を単一デバイス上で学習し、
    学習が終わったら BQQLinear / LayerNorm のパラメータだけ元の block にコピーする。
    """
    # -------------------------------
    # 0. キャリブに使うデバイスを決める
    # -------------------------------
    if device is None:
        first_param = next(block.parameters())
        calib_device = first_param.device
        # CPU なら CUDA があれば cuda:0 に逃がす
        if calib_device.type == "cpu" and torch.cuda.is_available():
            calib_device = torch.device("cuda:0")
    else:
        calib_device = torch.device(device)

    print(f"[INFO] Calibrating cloned block on device={calib_device}")

    # -------------------------------
    # 1. block を deepcopy して単一デバイス上に作る
    # -------------------------------
    orig_block = block  # 後でパラメータを書き戻すために保持
    block = copy.deepcopy(orig_block).to(calib_device)

    # ★ accelerate のフックを完全に剥がす
    # accelerate は forward を new_forward に差し替え、元を _old_forward に退避しているので、
    # それを元に戻す
    for m in block.modules():
        if hasattr(m, "_old_forward"):
            m.forward = m._old_forward
            delattr(m, "_old_forward")

    # dtype 情報
    first_param = next(block.parameters())
    orig_dtype = first_param.dtype

    # --- 必要なら block 本体を fp32 に揃える（デバイスは calib_device のまま） ---
    if cast_block_to_fp32 and orig_dtype != torch.float32:
        for p in block.parameters():
            if p.is_floating_point():
                p.data = p.data.to(dtype=torch.float32)
        print(f"[INFO] Cast cloned block parameters from {orig_dtype} to float32 for calibration.")
        block_dtype = torch.float32
    else:
        block_dtype = orig_dtype

    # RoPE 用モジュールも clone しておく（軽いので問題なし）
    if rotary_emb is not None:
        try:
            rotary_emb = copy.deepcopy(rotary_emb).to(calib_device)
            # 念のため accelerate フックも剥がす
            for m in rotary_emb.modules():
                if hasattr(m, "_old_forward"):
                    m.forward = m._old_forward
                    delattr(m, "_old_forward")
        except Exception:
            rotary_emb = rotary_emb.to(calib_device)

    # -------------------------------
    # 2. X_block / Y_block の整形
    # -------------------------------
    # 入出力は CPU に保持して、ミニバッチだけ calib_device に送る
    X_block = X_block.to("cpu", dtype=torch.float32)
    Y_block = Y_block.to("cpu", dtype=torch.float32)

    N, hidden_dim = X_block.shape
    tokens_per_seq = seqlen

    num_seqs = N // tokens_per_seq
    if num_seqs == 0:
        raise ValueError(
            f"Not enough tokens ({N}) to form even one sequence of length {tokens_per_seq}"
        )

    # 余りは切り捨て
    usable_tokens = num_seqs * tokens_per_seq
    X_block = X_block[:usable_tokens]
    Y_block = Y_block[:usable_tokens]

    # (num_seqs, seqlen, hidden_dim) に reshape
    X_seq = X_block.view(num_seqs, tokens_per_seq, hidden_dim)
    Y_seq = Y_block.view(num_seqs, tokens_per_seq, hidden_dim)

    # batch_size は「トークン数」ベースなので、シーケンス数に変換
    seq_batch_size = max(1, batch_size // tokens_per_seq)

    dataset = TensorDataset(X_seq, Y_seq)
    loader = DataLoader(dataset, batch_size=seq_batch_size, shuffle=True)

    block.train()

    # -------------------------------
    # 3. 学習対象パラメータの抽出
    # -------------------------------
    quant_module_types = (BQQLinear, nn.LayerNorm)  # 必要に応じて増やす

    trainable_params = []
    for m in block.modules():
        if isinstance(m, quant_module_types):
            for p in m.parameters(recurse=False):
                if p.requires_grad:
                    trainable_params.append(p)

    if not trainable_params:
        print("[WARN] No trainable params found in cloned block for BQQLinear/LayerNorm. Skipping.")
        return orig_block

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # block.forward のシグネチャを調べて position_embeddings を渡せるか判定
    block_sig = inspect.signature(block.forward)
    supports_position_embeddings = "position_embeddings" in block_sig.parameters

    # -------------------------------
    # 4. 学習ループ
    # -------------------------------
    for epoch in range(epochs):
        running_loss = 0.0
        total_tokens = 0

        for xb_seq, yb_seq in loader:
            # xb_seq, yb_seq: (B_seq, seqlen, hidden_dim)
            xb_seq = xb_seq.to(calib_device, dtype=torch.float32)
            yb_seq = yb_seq.to(calib_device, dtype=torch.float32)
            bsz, T, H = xb_seq.shape

            # position_ids: (batch, seqlen)
            position_ids = (
                torch.arange(T, device=calib_device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

            # block への入力を dtype 合わせ
            hs_for_block = xb_seq.to(calib_device, dtype=block_dtype)
            pos_ids_block = position_ids  # すでに calib_device

            # --- block への引数を組み立て ---
            block_kwargs = dict(
                hidden_states=hs_for_block,
                attention_mask=None,
                position_ids=pos_ids_block,
                use_cache=False,
                cache_position=None,
            )

            if supports_position_embeddings:
                if rotary_emb is None:
                    raise RuntimeError(
                        "This block expects `position_embeddings`, but `rotary_emb` is None. "
                        "For this LLaMA/Qwen, make sure you pass `base_model.rotary_emb` "
                        "into train_symquant_block()."
                    )
                cos, sin = rotary_emb(hs_for_block, pos_ids_block)
                block_kwargs["position_embeddings"] = (cos, sin)

            # --- forward（AMP 無効） ---
            if calib_device.type == "cuda":
                with torch.cuda.amp.autocast(enabled=False):
                    out = block(**block_kwargs)
            else:
                out = block(**block_kwargs)

            if isinstance(out, tuple):
                out = out[0]

            out = out.to(calib_device, dtype=torch.float32)

            # forward 直後の NaN/Inf チェック
            if not torch.isfinite(out).all():
                print("[WARN] Detected NaN/Inf in block output. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # NMSE: ||y - y_hat||^2 / (||y||^2 / N)
            mse = F.mse_loss(out.view(-1, H), yb_seq.view(-1, H))
            denom = (yb_seq ** 2).mean().detach().clamp_min(1e-6)
            loss = (mse / denom).float()

            if not torch.isfinite(loss):
                print("[WARN] NaN / Inf detected in loss. Skipping this batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 勾配チェック
            bad_grad = False
            for p in trainable_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = True
                    break

            if bad_grad:
                print("[WARN] NaN / Inf detected in gradients. Zeroing grads and skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue

            clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

            running_loss += loss.item() * xb_seq.numel()
            total_tokens += xb_seq.numel()

        avg_loss = running_loss / max(total_tokens, 1)
        print(f"[Block calib] epoch {epoch+1}/{epochs}, loss = {avg_loss:.6f}")

    block.eval()

    # -------------------------------
    # 5. 学習したパラメータを元の block に書き戻す
    # -------------------------------
    for (name_c, m_c), (name_o, m_o) in zip(block.named_modules(), orig_block.named_modules()):
        # 名前がズレてたらスキップ（構造が変わっている可能性）
        if name_c != name_o:
            continue

        if isinstance(m_c, quant_module_types):
            for p_name, p_c in m_c.named_parameters(recurse=False):
                if not hasattr(m_o, p_name):
                    continue
                p_o = getattr(m_o, p_name)
                with torch.no_grad():
                    p_o.data.copy_(p_c.data.to(p_o.device, dtype=p_o.dtype))

    return orig_block




def _best_divisor_leq(n: int, cap: int) -> int:
    """n を割り切る約数のうち、cap 以下で最大のものを返す。"""
    if cap < 1:
        raise ValueError("group_size must be >= 1")
    cap = min(cap, n)
    # 上から順に探すのが最短（n の約数はそう多くない）
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1  # ここには通常来ない

def patchify_2d(x: torch.Tensor, group_size: int):
    """
    x: 2次元テンソル (H, W)
    group_size: パッチの辺長の上限
    返り値: (x_patches, (j, k, m, n))
    - x_patches: 形状 (j, k, m, n)
    - j: 行方向パッチ数, k: 列方向パッチ数
    - m: パッチの高さ(行数), n: パッチの幅(列数)
    """
    if x.dim() != 2:
        raise ValueError("x must be a 2D tensor of shape (H, W)")
    H, W = x.shape
    m = _best_divisor_leq(H, group_size)  # 行側パッチ高（group_size 以下で最大）
    n = _best_divisor_leq(W, group_size)  # 列側パッチ幅（group_size 以下で最大）
    j, k = H // m, W // n

    # (H, W) -> (j, m, k, n) -> (j, k, m, n)
    x_p = x.reshape(j, m, k, n).permute(0, 2, 1, 3).contiguous()
    return x_p, (j, k, m, n)


def init_symquant_from_linear(
    linear: nn.Linear,
    p: int = 2,
    l: int = 16,
    group_size: int = 128,
    quant_bias: bool = True,
    act_bits: int = None,
    device: str = "cuda",
) -> BQQLinear:
    """
    既存 nn.Linear の重み形状を使って、BQQLinear を初期化する。
    ここでは Y, Z をランダム初期化 (std を合わせる程度) にしておく。
    """
    W = linear.weight.data  # (out_features, in_features)
    dtype = W.dtype

    _, (j, k, m, n)= patchify_2d(W, group_size=group_size)

    # 適当なスケールでランダム初期化
    std = W.std().item()
    if std == 0:
        std = 1e-2

    Y_init = torch.randn(p, j, k, m, l, device=device, dtype=dtype) * (std / (l ** 0.5))
    Z_init = torch.randn(p, j, k, l, n, device=device, dtype=dtype) * (std / (l ** 0.5))

    if quant_bias:
        # (p, j, k, 4) で a,b,c,d。とりあえず a=1, b=c=d=0 スタート
        A_init = torch.zeros(p, j, k, 4, device=device, dtype=dtype)
        A_init[..., 0] = 1.0
    else:
        # (p, j, k) で単純スケール
        A_init = torch.ones(p, j, k, device=device, dtype=dtype)

    bias = linear.bias.data.to(device) if linear.bias is not None else None

    sym_layer = BQQLinear(
        Y=Y_init,
        Z=Z_init,
        A=A_init,
        bias=bias,
        act_bits=act_bits,
        quant_bias=quant_bias,
    ).to(device)

    return sym_layer


class PairDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_symquant_layer(
    X: torch.Tensor,
    Y: torch.Tensor,
    sym_layer: BQQLinear,
    epochs: int = 5,
    batch_size: int = 2048,
    lr: float = 1e-3,
    device: str = "cuda",
    max_grad_norm: float = 0.1,
):
    """
    MSE ||Y - f(W', X)||^2 を最小化するように BQQLinear を学習する。
    X, Y は CPU Tensor でも OK。ミニバッチごとに GPU に転送。
    """
    dataset = PairDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    sym_layer.to(device)
    sym_layer.train()

    optimizer = torch.optim.Adam(sym_layer.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    for ep in range(epochs):
        total_loss = 0.0
        total_n = 0
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = sym_layer(xb.float())
            loss = F.mse_loss(pred, yb.float()) / ((yb.float()**2).mean() + 1e-12) # normalized MSE

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(sym_layer.parameters(), max_grad_norm)
            optimizer.step()
            # scheduler.step()
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        print(f"[BQQ calib] epoch {ep+1}/{epochs}  MSE = {total_loss / total_n:.6f}")


    sym_layer.eval()
    return sym_layer




def train_symquant_model_distill(
    teacher_model: nn.Module,
    student_model: nn.Module,
    loader,
    epochs: int = 5,
    lr: float = 5e-5,
    device: str | None = None,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
    loss_type: str = "nmse_logits",  # "nmse_logits" or "kl"
    temperature: float = 1.0,
):
    """
    教師モデル teacher_model の出力に対して、
    生徒モデル student_model（量子化モデル）を蒸留学習する。

    - teacher_model: フル精度モデル（勾配は流さない）
    - student_model: SymQuantLinear 等を含む量子化モデル
    - loader: (input_ids, targets) 形式の DataLoader を想定（targets は使わない）
    - loss_type:
        "nmse_logits" : ロジットの NMSE
        "kl"          : Softmax + KL distillation
    """

    # ---- device_map="auto" 対応の入力 device 決定 ----
    is_sharded = getattr(student_model, "hf_device_map", None) is not None

    if device is not None:
        entry_device = torch.device(device)
    elif is_sharded:
        # 分散モデルなら input_ids は CPU のまま渡して Accelerate に任せる
        entry_device = None
    else:
        # 単一 GPU/CPU モデルのときは student_model の最初の param の device に乗せる
        entry_device = next(student_model.parameters()).device

    # 教師は eval & no_grad
    teacher_model.eval()
    # 生徒は train
    student_model.train()

    # 生徒モデルのみ最適化（量子化パラメータなど）
    optimizer = torch.optim.Adam(
        [p for p in student_model.parameters() if p.requires_grad], lr=lr
    )

    for epoch in range(epochs):
        running_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(loader):
            # loader が (input_ids, targets) で来ることを想定
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch

            if entry_device is not None:
                input_ids = input_ids.to(entry_device)

            # ---- 教師の出力 (勾配なし) ----
            with torch.no_grad():
                teacher_out = teacher_model(input_ids=input_ids)
                # HF の CausalLM を想定
                teacher_logits = teacher_out.logits  # (B, T, V)

            # ---- 生徒の出力 ----
            student_out = student_model(input_ids=input_ids)
            student_logits = student_out.logits  # (B, T, V)

            # ---- ロス計算 ----
            if loss_type == "kl":
                # Knowledge Distillation 損失 (KL)
                T = temperature
                # 教師：確率（soft）
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                # 生徒：log softmax
                student_log_probs = F.log_softmax(student_logits / T, dim=-1)
                loss = F.kl_div(
                    student_log_probs, teacher_probs, reduction="batchmean"
                ) * (T * T)
            else:
                # デフォルト: ロジットの NMSE
                # ||S - T||^2 / (||T||^2 / N) みたいなイメージ
                diff = student_logits - teacher_logits
                mse = F.mse_loss(diff, torch.zeros_like(diff))
                denom = teacher_logits.pow(2).mean() + 1e-12
                loss = mse / denom

            # ---- 逆伝播 & 更新 ----
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()

            # ロギング用（トークン数で重み付け）
            batch_tokens = input_ids.numel()  # B * T
            running_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            if max_batches is not None and (i + 1) >= max_batches:
                break

        avg_loss = running_loss / max(total_tokens, 1)
        print(f"[Model distill] epoch {epoch+1}/{epochs}, loss = {avg_loss:.6f}")

    # 最後は eval に戻しておく
    student_model.eval()
    return student_model


def load_model(model_name, device="auto"):
    # 推奨デフォルト設定
    common_kwargs = dict(torch_dtype="auto", device_map=device)

    # 1. LLaMA 専用ローダー
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
        model = LlamaForCausalLM.from_pretrained(model_name, **common_kwargs)
        return tokenizer, model
    except Exception as e3:
        print(f"[WARN] LLaMA loader failed: {type(e3).__name__} – {e3}")

    # 2. 通常ロード
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        return tokenizer, model
    except Exception as e1:
        print(f"[WARN] Auto model load failed: {type(e1).__name__} – {e1}")

    # 3. trust_remote_code=True で再試行
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            **common_kwargs,
        )
        return tokenizer, model
    except Exception as e2:
        print(
            f"[WARN] Auto model load with trust_remote_code failed: "
            f"{type(e2).__name__} – {e2}"
        )

    # 4. 完全に失敗した場合
    raise RuntimeError(
        f"✗ Failed to load model: {model_name}\n"
        f"Errors:\n"
        f"  • Auto: {e1}\n"
        f"  • Auto + trust_remote_code: {e2}\n"
        f"  • LLaMA: {e3}"
    )



def get_module_paths(model, module_type=nn.Linear):
    """
    model 内のすべての module_type のモジュールに対して、
    ドット区切りのパス名のリストを返す関数。

    ex) ["model.layers.0.mlp.dense_h_to_4h", ...]
    """
    paths = []

    def _search(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # 子が目的のタイプなら追加
            if isinstance(child, module_type):
                paths.append(full_name)

            # 再帰的に探索
            _search(child, full_name)

    _search(model)
    return paths


def build_bqq_model(
    model_name: str,
    quant_state_dict: dict,
    args,
):
    """
    - HF の事前学習済みモデルを CPU にロードして ref_model を作る
    - ref_model 内のすべての nn.Linear を BQQLinear(SymQuantLinear) に差し替える
    - 量子化済み model の state_dict (quant_state_dict) を流し込む
    - 最後に eval_device (cpu / cuda:0 など) に乗せた ref_model を返す

    これで、accelerate/device_map="auto" とは無関係な「単一デバイス BQQ モデル」で
    PPL やサンプル生成ができる。
    """

    print(f"[BQQ-EVAL] Loading fresh ref_model on CPU: {model_name}")
    # ★ ここでは絶対に device="auto" を使わない
    #    CPU or 単一 GPU に丸ごとロードしておく
    _, ref_model = load_model(model_name, device="auto")

    # --- 1. ターゲット Linear 層名リスト（量子化時と同じ exclude 規則を使う） ---
    target_layer_names = get_module_paths(ref_model, nn.Linear)
    exclude = ["model.decoder.project_out", "model.decoder.project_in", "lm_head"]

    target_layer_names = [
        name for name in target_layer_names if not any(ex in name for ex in exclude)
    ]

    print(f"[BQQ-EVAL] Number of Linear layers in ref_model: {len(target_layer_names)}")

    # --- 2. ref_model 側の Linear を BQQLinear(SymQuantLinear) に差し替え ---
    for i, layer_name in enumerate(target_layer_names):
        print(
            f"[BQQ-EVAL] ({i+1}/{len(target_layer_names)}) "
            f"Replacing Linear with BQQ at: {layer_name}"
        )

        orig_linear = get_module_by_name(ref_model, layer_name)
        # ref_model は今 CPU にあるので、とりあえず CPU で初期化
        sym_layer = init_symquant_from_linear(
            orig_linear,
            p=args.p,
            l=args.l,
            group_size=args.group_size,
            quant_bias=args.quant_bias,
            act_bits=args.act_bits,
            device=orig_linear.weight.device,
        )

        # モジュール差し替え
        replace_linear_with_symquant(ref_model, layer_name, sym_layer)

    # --- 3. 量子化済み model の state_dict を ref_model に流し込む ---
    print("[BQQ-EVAL] Loading quantized state_dict into ref_model ...")
    # 念のため CPU に揃える
    quant_state_cpu = {k: v.to("cpu") for k, v in quant_state_dict.items()}

    missing, unexpected = ref_model.load_state_dict(quant_state_cpu, strict=False)
    if missing:
        print("[BQQ-EVAL] Missing keys in ref_model:", missing)
    if unexpected:
        print("[BQQ-EVAL] Unexpected keys from quant_state:", unexpected)

    return ref_model


from binary_quadratic_network import BQQLinearInference
def build_bqq_model_inference(
    model_name: str,
    quant_state_dict: dict,
    args,
    eval_device: str = "cpu",   # "cuda:0" など
):
    """
    推論専用 BQQ モデルの構築:
    - HF の事前学習済みモデルを CPU にロードして ref_model を作る
    - ref_model 内の nn.Linear を BQQLinearInference に直接差し替える
        -> quant_state_dict から Y_fp, Z_fp, A, bias を読み出して構築
    - 残りのパラメータは quant_state_dict から load_state_dict(strict=False) で反映
    - 最後に eval_device に乗せて返す
    """

    print(f"[BQQ-EVAL] Loading fresh ref_model on CPU: {model_name}")
    # ここは device=\"cpu\" 固定の方が安全（必要なら cuda に後で移動）
    _, ref_model = load_model(model_name, device="cpu")

    # --- 1. ターゲット Linear 層名リスト（量子化時と同じ exclude 規則を使う） ---
    target_layer_names = get_module_paths(ref_model, nn.Linear)
    exclude = ["model.decoder.project_out", "model.decoder.project_in", "lm_head"]

    target_layer_names = [
        name for name in target_layer_names if not any(ex in name for ex in exclude)
    ]

    print(f"[BQQ-EVAL] Number of Linear layers in ref_model: {len(target_layer_names)}")

    # --- 2. 各 Linear を BQQLinearInference に置き換え ---
    for i, layer_name in enumerate(target_layer_names):
        print(
            f"[BQQ-EVAL] ({i+1}/{len(target_layer_names)}) "
            f"Replacing Linear with BQQLinearInference at: {layer_name}"
        )

        # 量子化済み state_dict から、このレイヤに対応するパラメータを取り出す
        # BQQ の学習時に BQQLinear の param 名が
        #   {layer_name}.Y_fp, {layer_name}.Z_fp, {layer_name}.A, {layer_name}.bias
        # になっている前提
        prefix = layer_name

        key_Y = prefix + ".Y_fp"
        key_Z = prefix + ".Z_fp"
        key_A = prefix + ".A"
        key_b = prefix + ".bias"

        if key_Y not in quant_state_dict or key_Z not in quant_state_dict or key_A not in quant_state_dict:
            raise KeyError(
                f"[BQQ-EVAL] Quantized tensors for layer '{layer_name}' "
                f"not found in quant_state_dict. Expected keys: {key_Y}, {key_Z}, {key_A}"
            )

        Y_fp = quant_state_dict[key_Y]
        Z_fp = quant_state_dict[key_Z]
        A = quant_state_dict[key_A]
        bias = quant_state_dict.get(key_b, None)

        # 元の Linear（入力・出力次元など確認用に取っておいてもOK）
        orig_linear = get_module_by_name(ref_model, layer_name)

        # BQQLinearInference を構築
        inf_layer = BQQLinearInference.from_quant_tensors(
            Y_fp=Y_fp,
            Z_fp=Z_fp,
            A=A,
            bias=bias,
            act_bits=args.act_bits,
            quant_bias=args.quant_bias,
            device="cpu",            # ref_model は CPU にあるので CPU で作る
            sign_dtype=torch.int8,   # 必要に応じて変更可
            scale_dtype=torch.float16,
        )

        # 念のため in/out_features が一致しているかチェック（debug）
        assert inf_layer.in_features == orig_linear.in_features
        assert inf_layer.out_features == orig_linear.out_features

        # モジュール差し替え（元の replace_linear_with_symquant を再利用）
        replace_linear_with_symquant(ref_model, layer_name, inf_layer)

    # --- 3. 残りのパラメータを quant_state_dict からロード ---
    print("[BQQ-EVAL] Loading remaining parameters from quant_state_dict into ref_model ...")

    quant_state_cpu = {k: v.to("cpu") for k, v in quant_state_dict.items()}

    # strict=False にすることで:
    # - BQQLinearInference 側に存在しない Y_fp, Z_fp, A などは unexpected_keys としてスキップされる
    # - それ以外（embed, ln, lm_head など）はロードされる
    missing, unexpected = ref_model.load_state_dict(quant_state_cpu, strict=False)
    if missing:
        print("[BQQ-EVAL] Missing keys in ref_model (ignored):", missing)
    if unexpected:
        print("[BQQ-EVAL] Unexpected keys from quant_state (ignored):", unexpected)

    # --- 4. 推論モード & 勾配停止 ---
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # --- 5. eval_device に乗せる ---
    ref_model.to(eval_device)
    print(f"[BQQ-EVAL] Done. Inference model is on {eval_device}")

    return ref_model



def main(args):
    import copy

    # --- デバイス決定 ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # --- 1. モデル & トークナイザ読み込み ---
    model_name = args.model_name
    print(f"Loading model (quantization target): {model_name}")
    ref_model = get_model(model_name)  
    if args.model_path is not None:
        model = torch.load(args.model_path, map_location=args.device, weights_only=False)
    else:
        model = copy.deepcopy(ref_model)
    

    # ★ 1-2. 事前学習済み参照モデル（オリジナル）を用意
    print("Creating reference model (frozen pretrained copy)...")
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    # 余裕があれば CPU に退避（GPU メモリ節約）
    try:
        ref_model.to("cpu")
        ref_device_for_io = "cpu"
    except Exception:
        # device_map="auto" 等で .to("cpu") が難しい場合はそのまま使う
        ref_device_for_io = None  # collect_* 側の device 引数で制御
    print("Reference model ready.")

    # --- 2. DataLoader を作る ---
    train_loader, test_loader = get_imagenet(
        args.model_name,
        val_batchsize=args.batch_size,
        calib_batchsize=args.batch_size,
        num_workers=args.num_workers,
        num_traindatas=None,
        data_path=args.data_path,
        seed=0,
    )

    # --- 3. ターゲット Linear 層名リスト ---
    # 参照モデルと量子化モデルで構造は同じなので、どちらから取ってもよいが
    # 「事前学習済みの構造」に揃える意味で ref_model から取っておく
    target_layer_names = get_module_paths(ref_model, nn.Linear)
    exclude = ["norm", "emb", "head"]

    target_layer_names = [
        name for name in target_layer_names if not any(ex in name for ex in exclude)
    ]

    print("Number of Linear layers in the model:", len(target_layer_names))

    # max_batches の扱い
    max_batches = (
        None if args.max_batches is None or args.max_batches <= 0 else args.max_batches
    )

    # ==========================================================
    # 4. キャリブレーション・ループ
    # ==========================================================
    if args.calib_mode == "layer":
        # ------------------------------------------------------
        # レイヤーワイズ：各 Linear ごとに (X, Y) を「参照モデル」から収集し、
        # それに基づいて量子化モデル側の対応レイヤを学習・置換する
        # ------------------------------------------------------
        for i, target_layer_name in enumerate(target_layer_names):
            print(
                f"\n[{i+1}/{len(target_layer_names)}] "
                f"Target Linear layer to quantize: {target_layer_name}"
            )

            # --- 4. (X, Y=WX) を「事前学習済み参照モデル」から収集 ---
            print("Collecting (X, Y) from target linear using train loader (ref_model)...")
            X, Y = collect_linear_io_from_loader(
                model=ref_model,               # ★ 参照モデルから IO を取る
                layer_name=target_layer_name,
                loader=train_loader,
                device="cpu",                  # IO テンソルは CPU に置く
                max_batches=max_batches,
            )

            print("Collected:", X.shape, Y.shape)  # (N, in_features), (N, out_features)

            # --- 5. 参照モデル & 量子化モデルから対象 Linear を取得 ---
            # 参照モデル側：オリジナル重みの供給源
            orig_linear_ref = get_module_by_name(ref_model, target_layer_name)
            # 量子化モデル側：実際に置換するターゲット
            orig_linear_quant = get_module_by_name(model, target_layer_name)

            target_device = orig_linear_ref.weight.device
            target_dtype  = orig_linear_ref.weight.dtype

            if args.model_path is not None:
                # 量子化モデル側の Linear 重みを参照モデル側にコピーしておく
                sym_layer = orig_linear_quant
            else:
                # 参照モデルの Linear から SymQuantLinear を初期化（まず CPU 上で）
                sym_layer = init_symquant_from_linear(
                    orig_linear_ref,              # ★ オリジナル重みをコピー
                    p=args.p,
                    l=args.l,
                    group_size=args.group_size,
                    quant_bias=args.quant_bias,
                    act_bits=args.act_bits,       # act 量子化したければビット数を指定
                    device="cpu",
                )

            # --- 6. MSE ||WX - f(W', X)||^2 で SymQuantLinear を学習 ---
            #     X, Y は「事前学習済みモデルの IO」なので、
            #     全レイヤ一貫してオリジナル挙動に合わせた校正になる
            print("Training SymQuantLinear with collected (X, Y)...")
            sym_layer = train_symquant_layer(
                X,
                Y,
                sym_layer=sym_layer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,                # キャリブ用 GPU
                max_grad_norm=args.max_grad_norm,
            )

            # --- 7. モデル内の Linear を SymQuantLinear に差し替え ---
            # 学習が終わったら、量子化モデル側 Linear と同じ device/dtype に戻してから置換
            sym_layer.to(device=target_device, dtype=target_dtype)
            sym_layer.zero_grad(set_to_none=True)  # 勾配も消しておくと安心
            replace_linear_with_symquant(model, target_layer_name, sym_layer)
            print(f"Replaced {target_layer_name} with SymQuantLinear.")

            model.zero_grad(set_to_none=True)
            del X, Y, sym_layer, orig_linear_ref, orig_linear_quant
            torch.cuda.empty_cache()

    else:
        # ======================================================
        # ブロックワイズ：
        #   1. 各ブロックの入出力 (X_block, Y_block) を「参照モデル」から取得
        #   2. 量子化モデル側のブロック内 Linear を SymQuantLinear に置き換え
        #   3. train_symquant_block で、量子化ブロックを Y_block に合わせて一括で学習
        # ======================================================
        # まず layer_name -> block_name でグルーピング（構造は ref_model / model で同一）
        block_to_layers = {}
        for ln in target_layer_names:
            block_name = get_block_name_from_layer_name(ln)
            if block_name is None:
                # ブロック構造に乗っていない Linear はスキップ
                print(
                    f"[WARN] Could not infer block_name from layer_name: {ln}. "
                    "Skipping in block-wise calibration."
                )
                continue
            block_to_layers.setdefault(block_name, []).append(ln)

        print("Number of blocks for calibration:", len(block_to_layers))
        for bname, layers in block_to_layers.items():
            print(f"  Block: {bname}, #Linear = {len(layers)}")

        total_layers = len(target_layer_names)
        processed_layers = 0

        # --- Qwen2 / LLaMA RoPE 用の rotary_emb を取得（量子化モデル側からでOK） ---
        base_model = getattr(model, "model", model)
        if hasattr(base_model, "rotary_emb"):
            rotary_emb = base_model.rotary_emb
        else:
            rotary_emb = None

        # ブロック名のソート順で処理
        for block_name in sorted(block_to_layers.keys()):
            layer_names = block_to_layers[block_name]
            print(
                f"\n===== Calibrating block: {block_name} "
                f"({len(layer_names)} Linear layers) ====="
            )

            # --- 4-1. ブロック入出力 (X_block, Y_block) を「参照モデル」から収集 ---
            print("Collecting (X_block, Y_block) from block using WikiText2 loader (ref_model)...")
            X_block, Y_block = collect_block_io_from_loader(
                model=ref_model,         # ★ 参照モデルから IO を取る
                block_name=block_name,
                loader=train_loader,
                device="cpu",
                max_batches=max_batches,
            )
            print("Block IO collected:", X_block.shape, Y_block.shape)

            # --- 4-2. このブロック内の Linear を量子化モデル側で SymQuantLinear に置き換え ---
            for ln in layer_names:
                processed_layers += 1
                print(
                    f"  -> Initializing SymQuantLinear for layer "
                    f"[{processed_layers}/{total_layers}]: {ln}"
                )

                # 参照モデル側：オリジナル重み
                orig_linear_ref = get_module_by_name(ref_model, ln)
                # 量子化モデル側：置換ターゲット
                orig_linear_quant = get_module_by_name(model, ln)

                target_device = orig_linear_quant.weight.device
                target_dtype  = orig_linear_quant.weight.dtype

                # まず CPU 上でオリジナル重みから初期化
                sym_layer = init_symquant_from_linear(
                    orig_linear_ref,
                    p=args.p,
                    l=args.l,
                    group_size=args.group_size,
                    quant_bias=args.quant_bias,
                    act_bits=args.act_bits,
                    device="cpu",
                )

                # 対応する shard/device に移してからブロックに埋め込む
                sym_layer.to(device=target_device, dtype=target_dtype)
                replace_linear_with_symquant(model, ln, sym_layer)
                print(f"     Replaced {ln} with SymQuantLinear.")

                del orig_linear_ref, orig_linear_quant, sym_layer

            # --- 4-3. SymQuantLinear に置き換えたブロックを一括で学習 ---
            print(
                "Training SymQuantLinear parameters in this block "
                "to match (X_block -> Y_block) from reference model..."
            )
            block_module = get_module_by_name(model, block_name)  # 量子化モデル側ブロック
            train_symquant_block(
                X_block=X_block,
                Y_block=Y_block,
                block=block_module,
                rotary_emb=rotary_emb,
                seqlen=args.seqlen,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"Finished block-wise calibration for {block_name}.")

            # ★ このブロックの勾配バッファを完全に解放
            model.zero_grad(set_to_none=True)
            del X_block, Y_block, block_module
            torch.cuda.empty_cache()


    del ref_model
    torch.cuda.empty_cache()
    # --- 9. モデルを ./bqq_models に torch.save(model.state_dict(), path) で保存 ---
    save_dir = Path(args.save_dir) if args.save_dir is not None else SCRIPT_DIR / "quantized_bqq_model"
    os.makedirs(save_dir, exist_ok=True)

    # モデル名からファイル名を自動生成
    # 例: Qwen/Qwen3-0.6B → Qwen_Qwen3-0.6B_bqq_2bit_128gs_5e-4lr.pt
    safe_name = args.model_name.replace("/", "_")
    save_name = (
        f"{safe_name}-{args.p}bit-{args.group_size}gs-{args.lr}lr-{args.calib_mode}calib.pth"
    )

    save_path = save_dir / save_name
    print(f"Saving quantized model object to: {save_path}")
    torch.save(model, save_path)
    print("Model saved.")





def attach_nan_detector(block: nn.Module):
    handles = []

    def has_nan(x: torch.Tensor) -> bool:
        # float 以外は一旦スキップ（int/bool なら NaN ありえないので）
        if not x.is_floating_point():
            return False
        # isfinite(x) == False が1つでもあれば NaN/Inf
        return (~torch.isfinite(x)).any().item()

    def make_hook(name):
        def _hook(module, inputs, output):
            bad = False

            # 入力チェック
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor) and has_nan(inp):
                    print(f"[NaN DETECT] in INPUT of {name} ({module.__class__.__name__})")
                    bad = True
                    break

            # 出力チェック
            if isinstance(output, torch.Tensor):
                if has_nan(output):
                    print(f"[NaN DETECT] in OUTPUT of {name} ({module.__class__.__name__})")
                    bad = True
            elif isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor) and has_nan(o):
                        print(f"[NaN DETECT] in OUTPUT of {name} ({module.__class__.__name__})")
                        bad = True
                        break

            if bad:
                print(f"  -> stop after first NaN. Check this module: {name}")
                for h in handles:
                    h.remove()
                raise RuntimeError("NaN detected in module: " + name)

        return _hook

    for name, m in block.named_modules():
        if m is block:
            continue
        h = m.register_forward_hook(make_hook(name))
        handles.append(h)

    return handles




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arguments for evaluation")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name passed to evaluation")
    parser.add_argument("--group_size", type=int, default=32,
                        help="Group size for quantization")
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for fine-tuning")  
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for fine-tuning")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum number of batches to use for calibration")
    parser.add_argument("--p", type=int, default=2,
                        help="Bit width for activations in BQQ")
    parser.add_argument("--l", type=int, default=4,
                        help="Inner dimension for BQQ")
    parser.add_argument("--quant_bias", action='store_true',
                        help="Whether to incorporate bias in BQQ")
    parser.add_argument("--act_bits", type=int, default=None,
                        help="Activation quantization bits (8 means no quantization)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to ImageNet. If omitted, use IMAGENET_DIR or IMAGENET_ROOT.")
    parser.add_argument("--calib_mode", type=str, default="layer",
                        choices=["layer", "block"],)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the quantized model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the model file to be quantized (if not using model_name directly)")
    args = parser.parse_args()
    main(args)
