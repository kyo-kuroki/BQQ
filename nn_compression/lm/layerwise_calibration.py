import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from datautils import get_loaders, get_wikitext2_dataloaders
from parsers import parse_args
from torch.ao.quantization import QuantStub, DeQuantStub
import os
from tqdm import tqdm, trange
import model_loader 
sys.path.append(os.path.dirname(os.path.dirname(__file__)) + '/BQQ')
import binary_quadratic_network
import BQQLLM.layerwise_calibration as layerwise_calibration
import quantizer
import math


def inspect_binary_quadratic(model):
    """
    BinaryQuadratic モジュールの入力・パラメータ・バッファを可視化
    """
    hooks = []
    saved_inputs = {}

    def hook_fn(module, input, output, name):
        # inputはタプル
        saved_inputs[name] = input

    # BinaryQuadratic モジュールを探してhook登録
    for name, module in model.named_modules():
        if module.__class__.__name__ == "BinaryQuadratic":
            hooks.append(module.register_forward_hook(
                lambda m, i, o, name=name: hook_fn(m, i, o, name)
            ))

    return hooks, saved_inputs

def scale_calibration_layer(x, y, z, a, b, c, d, input, device_id=0):
    torch.set_float32_matmul_precision('high')
    # もしゼロで埋めたい場合は:
    d_padded = torch.zeros(a.shape, device=a.device)
    d_padded[0] = d  # 例えば最初だけ埋めるなど用途に応じて

    # 4つのテンソルを結合
    scale_parameters = torch.stack([a, b, c, d_padded], dim=0)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if len(x.shape)!=2:
        raise ValueError(f'Dimention Error: Please Input 2-dim Matrix !! Now x.shape = {x.shape}')
    x = x.to(device).float()
    y = y.to(device).float()
    z = z.to(device).float()
    scale_parameters = scale_parameters.to(device).float()
    scale_parameters_shape = a.shape

    # preprocess for input tensor
    input = input.reshape(-1, input.size(-1)).float() # reshape to 2-dim matrix
    scale_parameters = scale_parameters.to(device).float()
    

    s = ((input.T @ input)/input.shape[0]).to(device) # average covariance matrix
    
    # preventing from overflow
    scale = input.numel() 


    # quantization error function

    def reconstruction(y, z, scale_parameters):        # (p, j, k, m, l) @ (p, j, k, l, n) → (p, j, k, m, n)
        # y.shape, z.shape → (p, j, k, m, l), (p, j, k, l, n)
        bit_width, row_width, col_width, group_row, inter_dimension =  y.shape
        _, _, _, _, group_col =  z.shape

        a, b, c, d = scale_parameters[0], scale_parameters[1], scale_parameters[2], scale_parameters[3].sum(dim=0)
        W_core = torch.matmul(y, z)  # Y @ Z

        # Y.sum over l → (p, j, k, m)
        Y_sum = y.sum(dim=-1, keepdim=True)  # → (p, j, k, m, 1)
        Z_sum = z.sum(dim=-2, keepdim=True)  # → (p, j, k, 1, n)

        # 総和項
        W = a * W_core + b * Y_sum + c * Z_sum 

        # p に沿って加算 → (j, k, m, n)
        W = W.sum(dim=0) + d
        W = W.permute(0, 2, 1, 3).reshape(row_width * group_row, col_width * group_col)

        return W
    
    def error(x, y, z, scale_parameters, s=s):
        q = reconstruction(y, z, scale_parameters)
        return torch.trace((x - q) @ s @ (x - q).T)


    grad = torch.func.grad(error, argnums=3)
    hesse = torch.func.hessian(error, argnums=3)
    


    def optimize_scale(y, z, scale_parameters, s=s):
        # H_scaled = hesse(x, y, z, scale_parameters, s/scale).reshape(v_flat.shape[0], -1)
        # v = grad(x, y, z, scale_parameters)
        # v_flat = v.reshape(-1)
        # v_scaled = v_flat / scale


        # sol = scale_parameters - torch.matmul(torch.linalg.pinv(H_scaled, rcond=1e-16), v_scaled).reshape_as(v).float()
        for i in range(10):
            v = grad(x, y, z, scale_parameters)
            v_flat = v.reshape(-1)
            v_scaled = v_flat / scale
            scale_parameters = scale_parameters - 0.01*v_scaled
        return scale_parameters

    print('before calibration:', error(x, y, z, scale_parameters).item())
    scale_parameters = optimize_scale(y, z, torch.zeros((scale_parameters_shape), device=device), s=s)
    print('after calibration:', error(x, y, z, scale_parameters).item())
    return scale_parameters
    



def scale_calibration(x, y, z, a, b, c, d, input, device_id=0):
    torch.set_float32_matmul_precision('high')
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # --- 前処理 ---
    d_padded = torch.zeros_like(a)
    d_padded[0] = d
    scale_parameters = torch.stack([a, b, c, d_padded], dim=0).to(device)

    p, I, J, _, _ = a.shape
    group_row, group_col = y.shape[-2], z.shape[-1]

    x = x.to(device).float()
    y = y.to(device).float()
    z = z.to(device).float()
    scale_parameters = scale_parameters.to(device).float()

    # --- s を作る ---
    input_flat = input.reshape(-1, input.size(-1)).float().to(device)
    s = ((input_flat.T @ input_flat) / input_flat.shape[0]).to(device)  # (d, d)
    scale = input.numel()

    # --- 再構成関数（ブロック） ---
    def reconstruction_block(y_block, z_block, scale_parameters_block):
        a_b = scale_parameters_block[0]
        b_b = scale_parameters_block[1]
        c_b = scale_parameters_block[2]
        d_b = scale_parameters_block[3].sum(dim=0)

        W_core = torch.matmul(y_block, z_block)
        Y_sum = y_block.sum(dim=-1, keepdim=True)
        Z_sum = z_block.sum(dim=-2, keepdim=True)
        W = a_b * W_core + b_b * Y_sum + c_b * Z_sum
        W = W.sum(dim=0) + d_b
        return W

    # --- ✨ error_block を trace で記述 ---
    def error_block(x_block, y_block, z_block, scale_parameters_block, s_block):
        q_block = reconstruction_block(y_block, z_block, scale_parameters_block)
        diff = x_block - q_block  # (m, n)
        # 二次形式を trace で表現
        val = torch.trace(diff @ s_block @ diff.T)
        return val  # scalar tensor

    def reconstruction(y, z, scale_parameters):
        bit_width, row_width, col_width, group_row, inter_dimension = y.shape
        _, _, _, _, group_col = z.shape

        a_p = scale_parameters[0]
        b_p = scale_parameters[1]
        c_p = scale_parameters[2]
        d_p = scale_parameters[3].sum(dim=0)

        W_core = torch.matmul(y, z)
        Y_sum = y.sum(dim=-1, keepdim=True)
        Z_sum = z.sum(dim=-2, keepdim=True)
        W = a_p * W_core + b_p * Y_sum + c_p * Z_sum
        W = W.sum(dim=0) + d_p
        W = W.permute(0, 2, 1, 3).reshape(row_width * group_row, col_width * group_col)
        return W

    def error(x, y, z, scale_parameters, s=s):
        q = reconstruction(y, z, scale_parameters)
        return torch.trace((x - q) @ s @ (x - q).T)

    # --- ブロック最適化 ---
    def optimize_scale_block(x_block, y_block, z_block, sp_block, s_block):
        params = sp_block.detach().clone().float().to(device)
        params_flat = params.reshape(-1).requires_grad_(True)

        def scalar_loss(p_flat):
            p = p_flat.view_as(params)
            return error_block(x_block, y_block, z_block, p, s_block)

        grad_vec = torch.autograd.functional.jacobian(scalar_loss, params_flat)
        H = torch.autograd.functional.hessian(scalar_loss, params_flat)
        v_scaled = grad_vec / scale

        step = torch.linalg.pinv(H, rcond=1e-12) @ v_scaled

        sol_flat = params_flat - step
        sol = sol_flat.view_as(params)
        return sol.detach().to(device)

    def optimize_scale(x, y, z, scale_parameters, s=s):
        optimized_params = torch.zeros_like(scale_parameters, device=device)
        print("Start per-(i,j) optimization...")

        x_blocks = x.view(I, group_row, J, group_col).permute(0, 2, 1, 3)
        s_blocks = s.view(J, group_col, J, group_col).permute(0, 2, 1, 3)
        print(s_blocks.shape)

        for i in range(I):
            for j in range(J):
                x_block = x_blocks[i, j]
                s_block = s_blocks[j, j]
                sp_block = scale_parameters[:, :, i, j, :, :]
                y_block = y[:, i, j, :, :]
                z_block = z[:, i, j, :, :]
                sol_ij = optimize_scale_block(x_block, y_block, z_block, sp_block, s_block)

                optimized_params[:, :, i, j, :, :] = sol_ij

        return optimized_params

    print('before calibration:', error(x, y, z, scale_parameters, s=s).item())
    scale_parameters = optimize_scale(x, y, z, scale_parameters, s=s)
    print('after calibration:', error(x, y, z, scale_parameters, s=s).item())

    print("Optimization completed.")
    return scale_parameters




def distill_calibration(org_model, quant_model, data_loader, optimizer, epochs, org_device="cuda:0", quant_device="cuda:1"):
    loss_fct = nn.MSELoss()

    org_model.to(org_device)
    quant_model.to(quant_device)
    org_model.eval()  # org_modelは固定
    quant_model.train()  # quant_modelを更新

    for epoch in trange(epochs, desc="Distillation Calibration"):
        for input, target in data_loader:

            with torch.no_grad():
                org_logits = org_model(input.to(org_device)).logits  # 教師信号 (固定)

            # quant_modelは勾配追跡ON
            quant_logits = quant_model(input.to(quant_device)).logits

            loss = loss_fct(quant_logits, org_logits.to(quant_device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return quant_model

@torch.no_grad()
def evaluate_ppl(model, test_loader, device="cuda"):
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0

    for inp, tar in tqdm(test_loader, desc="Evaluating PPL"):
        inp, tar = inp.to(device), tar.to(device)
        outputs = model(inp).logits
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), tar.view(-1))
        total_loss += loss.item() * inp.numel()
        total_tokens += inp.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    print(f"\n📊 Perplexity: {ppl:.3f}")
    return ppl





# if __name__ == "__main__":
#     model_name = "Qwen3-0.6B"
#     bit_width = 4
#     group_size = 128
#     num_step = 50000
#     # ダミー入力を作成してforward（モデルに応じて修正）
#     batch_size = 128
#     seq_len = 16
#     vocab_size = 151936  # Qwen系の語彙数（例）

#     model_path = "Qwen/Qwen3-0.6B"
#     model_weights = AutoModelForCausalLM.from_pretrained(model_path).state_dict()

#     q_model_path = (
#         os.path.dirname(__file__)
#         + f"/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step.pth"
#     )

#     q_model = torch.load(q_model_path, weights_only=False)

#     # hookの設定
#     hooks, saved_inputs = inspect_binary_quadratic(q_model)



#     # 正しい randint 呼び出し
#     dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
#     with torch.no_grad():
#         _ = q_model(dummy_input)

#     # 各 BinaryQuadratic モジュールに対してキャリブレーションを実行
#     for name, module in q_model.named_modules():
#         if module.__class__.__name__ != "BinaryQuadratic":
#             continue

#         print(f"\n=== BinaryQuadratic: {name} ===")

#         if name not in saved_inputs:
#             print("Input not captured")
#             continue

#         inp = saved_inputs[name][0]  # 最初の入力
#         inp = inp.to("cuda").float()  # GPUに転送

#         # パラメータもGPUに転送
#         original_W = model_weights[name+".weight"].to("cuda").float()
#         Y, Z = module.Y.to("cuda").float(), module.Z.to("cuda").float()

#         a, b, c, d = module.a.to("cuda").float(), module.b.to("cuda").float(), module.c.to("cuda").float(), module.d.to("cuda").float()

#         # キャリブレーションを実行
#         scale_parameters = scale_calibration(original_W, Y, Z, a, b, c, d, input=inp, device_id=0)

#         # GPU メモリ解放
#         del inp, original_W, Y, Z, a, b, c, d, scale_parameters
#         torch.cuda.empty_cache()

if __name__ == "__main__":
    model_name = "Qwen3-0.6B"
    bit_width = 4
    group_size = 128
    num_step = 50000
    # ダミー入力を作成してforward（モデルに応じて修正）
    batch_size = 128
    seq_len = 2048
    vocab_size = 151936  # Qwen系の語彙数（例）

    model_path = "Qwen/Qwen3-0.6B"
    o_model = AutoModelForCausalLM.from_pretrained(model_path)

    q_model_path = (
        os.path.dirname(__file__)
        + f"/quantized_model_data/{model_name}/{model_name}-{bit_width}bit-{group_size}gs-{num_step}step.pth"
    )

    q_model = torch.load(q_model_path, weights_only=False)


    optimizer = torch.optim.Adam(q_model.parameters(), lr=1e-4)
    # train_loader, testenc = get_loaders(name="wikitext2", nsamples=batch_size, seqlen=seq_len, model=model_path)
    train_loader, test_loader = get_wikitext2_dataloaders(model=model_path, seqlen=seq_len, batch_size=batch_size, num_workers=4, seed=42)
    calibrated_model = distill_calibration(o_model, q_model, train_loader, optimizer, epochs=10, org_device="cuda:0", quant_device="cuda:1")
    o_ppl = evaluate_ppl(o_model, test_loader, device="cuda:0")
    q_ppl = evaluate_ppl(q_model, test_loader, device="cuda:1")
    cq_ppl = evaluate_ppl(calibrated_model, test_loader, device="cuda:1")
    print(f"Original Model PPL: {o_ppl:.3f}")
    print(f"Quantized Model PPL: {q_ppl:.3f}")
    print(f"Calibrated Quantized Model PPL: {cq_ppl:.3f}")
