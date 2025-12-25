import torch
import torch.nn as nn
import os
from scipy.linalg import hadamard
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Function


class BinaryQuadratic(nn.Module):
    def __init__(self, Y, Z, A, bias=None):
        super().__init__()
        self.bit_width, self.row_width, self.col_width, self.y_row, self.inter_dimension =  Y.shape
        _, _, _, _, self.z_col =  Z.shape

        # バイナリパラメータ
        self.register_buffer("Y", (Y > 0.5))
        self.register_buffer("Z", (Z > 0.5))
        # スケーリング係数：形状 (bit_width, row_width, col_width)
        self.a = nn.Parameter(A[...,0].unsqueeze(-1).unsqueeze(-1).clone())
        self.b = nn.Parameter(A[...,1].unsqueeze(-1).unsqueeze(-1).clone())
        self.c = nn.Parameter(A[...,2].unsqueeze(-1).unsqueeze(-1).clone())
        self.d = nn.Parameter(A[...,3].unsqueeze(-1).unsqueeze(-1).sum(dim=0))  # オフセット項
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, X):
        """
        X: (j*m, k*n)
        Returns: (j*m, ?)
        """
        dtype = X.dtype
        device = self.Y.device
        # (p, j, k, m, l) @ (p, j, k, l, n) → (p, j, k, m, n)
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))  # Y @ Z

        # Y.sum over l → (p, j, k, m)
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)  # → (p, j, k, m, 1)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)  # → (p, j, k, 1, n)

        # 総和項
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum 

        # p に沿って加算 → (j, k, m, n)
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)

        # 出力
        if self.bias is None:
            return X.to(device) @ W.T
        else:
            return  X.to(device) @ W.T + self.bias.type(dtype).to(device)
        

    def get_weight(self, dtype=torch.float32):
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))  # Y @ Z

        # Y.sum over l → (p, j, k, m)
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)  # → (p, j, k, m, 1)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)  # → (p, j, k, 1, n)

        # 総和項
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum 

        # p に沿って加算 → (j, k, m, n)
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)
        return W
    
class HadamardBinaryQuadratic(nn.Module):
    def __init__(self, Y, Z, A, bias=None):
        super().__init__()
        self.bit_width, self.row_width, self.col_width, self.y_row, self.inter_dimension =  Y.shape
        _, _, _, _, self.z_col =  Z.shape

        # バイナリパラメータ
        self.register_buffer("Y", (Y > 0.5))
        self.register_buffer("Z", (Z > 0.5))
        # スケーリング係数：形状 (bit_width, row_width, col_width)
        self.a = nn.Parameter(A[...,0].unsqueeze(-1).unsqueeze(-1).clone())
        self.b = nn.Parameter(A[...,1].unsqueeze(-1).unsqueeze(-1).clone())
        self.c = nn.Parameter(A[...,2].unsqueeze(-1).unsqueeze(-1).clone())
        self.d = nn.Parameter(A[...,3].unsqueeze(-1).unsqueeze(-1).sum(dim=0))  # オフセット項
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, X):
        """
        X: (j*m, k*n)
        Returns: (j*m, ?)
        """
        dtype = X.dtype
        device = self.Y.device
        # (p, j, k, m, l) @ (p, j, k, l, n) → (p, j, k, m, n)
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))  # Y @ Z

        # Y.sum over l → (p, j, k, m)
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)  # → (p, j, k, m, 1)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)  # → (p, j, k, 1, n)

        # 総和項
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum 

        # p に沿って加算 → (j, k, m, n)
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)

        # 出力
        if self.bias is None:
            return X.to(device) @ torch.from_numpy(hadamard(W.shape[1])).float().to(device) @ W.T
        else:
            return  X.to(device) @ torch.from_numpy(hadamard(W.shape[1])).float().to(device) @ W.T + self.bias.type(dtype).to(device)
        

    def get_weight(self, dtype=torch.float32):
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))  # Y @ Z

        # Y.sum over l → (p, j, k, m)
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)  # → (p, j, k, m, 1)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)  # → (p, j, k, 1, n)

        # 総和項
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum 

        # p に沿って加算 → (j, k, m, n)
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)
        return W

def transform_A(A, l):
    # A: (p, r, c, 4)

    A0 = A[..., 0]
    A1 = A[..., 1]
    A2 = A[..., 2]
    A3 = A[..., 3]

    # transform from {0, 1} to {-1, 1} coefficients
    new0 = A0/4 
    new1 = A1/2 + A0/4
    new2 = A2/2 + A0/4 
    new3 = (A0/4 + A1/2 + A2/2)*l + A3

    # (p, r, c, 4) に戻す
    A_new = torch.stack([new0, new1, new2, new3], dim=-1)
    return A_new

def get_matrices(patch_list, bit_width):
    # 特定のビット重みのみ
    row_width = max(patch['patch_row'] for patch in patch_list)+1
    col_width = max(patch['patch_col'] for patch in patch_list)+1
    m, l = patch_list[0]['mat1'].shape
    l, n = patch_list[0]['mat2'].shape
    A = torch.zeros((bit_width, row_width, col_width, 4))
    Y = torch.zeros((bit_width, row_width, col_width, m, l))
    Z = torch.zeros((bit_width, row_width, col_width, l, n))
    for patch in patch_list:
        i, j = patch['patch_row'], patch['patch_col']
        a, y, z, k = patch['coeff'], patch['mat1'], patch['mat2'], patch['bit_idx']

        
        if k >= bit_width:
            continue  # bit_idx が範囲外の場合はスキップ
        else:
            A[k,i,j] = a
            Y[k,i,j] = y
            Z[k,i,j] = z
        
    return A, Y, Z




def replace_linear_with_bqq(model, weights_dir, bit_width, prefix='', device=None, show_tqdm=True):
    iterator = tqdm(model.named_children()) if show_tqdm else model.named_children()
    for name, module in iterator:
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, (nn.Linear)):

            # 重みファイル名にフルネームが入っていると仮定
            weight_list = []
            for file in os.listdir(weights_dir):
                if file.endswith('.pth') and (full_name in file) and ('row' in file):
                    path = os.path.join(weights_dir, file)
                    if device is not None:
                        weight_list += torch.load(path, map_location=device)
                    else:
                        weight_list += torch.load(path, map_location=module.weight.device)
            A, Y, Z = get_matrices(weight_list, bit_width=bit_width)
            bqq = BQQLinear(2*Y-1, 2*Z-1, transform_A(A, l=Y.shape[-1]), bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(module, weights_dir, bit_width, prefix=full_name, show_tqdm=False, device=device)

    return model

def replace_linear_with_hbqq(model, weights_dir, bit_width, prefix=''):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, (nn.Linear)):
            # print(f"Replacing {full_name}: {type(module)}")
            # print('weight shape:', module.weight.shape, 'bias is None ?:', module.bias is None)

            # 重みファイル名にフルネームが入っていると仮定
            weight_list = []
            for file in os.listdir(weights_dir):
                if file.endswith('.pth') and (full_name in file) and ('row' in file):
                    path = os.path.join(weights_dir, file)
                    weight_list += torch.load(path, map_location=module.weight.device)

            A, Y, Z = get_matrices(weight_list, bit_width=bit_width)
            bqq = HadamardBinaryQuadratic(Y, Z, A, bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(module, weights_dir, bit_width, prefix=full_name)

    return model


def replace_weight(model, weights_dir, bit_width):
    for name, param in model.named_parameters():

        if 'head' in name:
            print(f"Skipping {name}")
            continue

        if not 'norm' in name and 'bias' in name:
            print(f"Replacing {name}")
            print('weight shape:', param.shape)

            # 重みファイル名にフルネームが入っていると仮定
            weight_list = []
            for file in os.listdir(weights_dir):
                if file.endswith('.pth') and (name in file) and ('row' in file):
                    path = os.path.join(weights_dir, file)
                    weight_list += torch.load(path, map_location=param.device)

            A, Y, Z = get_matrices(weight_list, bit_width=bit_width)
            bqq = BinaryQuadratic(Y, Z, A, bias=None)
            param.data.copy_(bqq.get_weight())


    return model


def merge_binary_quadratic(diff_layer: BinaryQuadratic, quant_layer: BinaryQuadratic) -> BinaryQuadratic:
    # (p, j, k, m, l) の形状で連結
    merged_Y = torch.cat([quant_layer.Y, diff_layer.Y], dim=0)
    merged_Z = torch.cat([quant_layer.Z, diff_layer.Z], dim=0)

    # (p, j, k, 1, 1) のスケーリング係数を結合
    merged_a = torch.cat([quant_layer.a, diff_layer.a], dim=0)
    merged_b = torch.cat([quant_layer.b, diff_layer.b], dim=0)
    merged_c = torch.cat([quant_layer.c, diff_layer.c], dim=0)

    # d: shape (j, k, m, n) に加算
    merged_d = quant_layer.d + diff_layer.d

    # biasはquant側を使う（diffは通常ゼロまたはNone想定）
    merged_bias = quant_layer.bias

    return BinaryQuadratic(merged_Y, merged_Z, torch.cat([
        merged_a.squeeze(-1).squeeze(-1).unsqueeze(-1),
        merged_b.squeeze(-1).squeeze(-1).unsqueeze(-1),
        merged_c.squeeze(-1).squeeze(-1).unsqueeze(-1),
        merged_d.unsqueeze(-1)  # (j, k, m, n, 1)
    ], dim=-1), bias=merged_bias)



# nビットまでのモデルと、nより大きいビット層のモデル(差分モデル)をマージ
def merge_binaryquadratic_recursive(model_q: nn.Module, model_d: nn.Module, prefix=''):
    for (name_q, module_q), (name_d, module_d) in zip(model_q.named_children(), model_d.named_children()):
        assert name_q == name_d, f"Module name mismatch: {name_q} != {name_d}"
        full_name = f"{prefix}.{name_q}" if prefix else name_q

        if isinstance(module_q, BinaryQuadratic) and isinstance(module_d, BinaryQuadratic):
            merged = merge_binary_quadratic(module_d, module_q)
            setattr(model_q, name_q, merged)
            print(f"Merged BinaryQuadratic at {full_name}")
        else:
            merge_binaryquadratic_recursive(module_q, module_d, prefix=full_name)

    return model_q





class SymQuantSTE(Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: torch.Tensor, num_bits: int):
        if num_bits == 1:
            # 1-bit: s * sign(x)
            s = scale.abs()
            output = s * torch.sgn(input)
        else:
            s = scale.abs().clamp_min(1e-8)
            qmax = 2 ** (num_bits - 1) - 1
            q = torch.clamp(torch.round(input / s), -qmax, qmax)
            output = q * s

        ctx.save_for_backward(input, s)
        ctx.num_bits = num_bits
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, s = ctx.saved_tensors
        num_bits = ctx.num_bits

        if num_bits == 1:
            mask = (input.abs() <= s).to(grad_output.dtype)
            grad_input = grad_output * mask 
        else:
            qmax = 2 ** (num_bits - 1) - 1
            mask = (input.abs() <= qmax * s).to(grad_output.dtype)
            grad_input = grad_output * mask

        return grad_input, None, None


class BQQLinear(nn.Module):
    """
    BQQ 形式の量子化 weight (Y, Z, A) を用いる Linear 層。
    - Y, Z は nn.Parameter (実数) として保持
    - forward で SymQuantSTE(num_bits=1) により 1-bit 量子化
    - A はスケーリング係数 (quant_bias=True: a,b,c,d)
    - 入力アクティベーションも必要なら SymQuantSTE で量子化
    """
    def __init__(
        self,
        Y: torch.Tensor,      # (p, j, k, m, l)   実数初期値
        Z: torch.Tensor,      # (p, j, k, l, n)   実数初期値
        A: torch.Tensor,      # quant_bias=True: (p, j, k, 4) / False: (p, j, k)
        bias: torch.Tensor = None,    # (j*m,) など
        act_bits: int = None,
        quant_bias: bool = True,
    ):
        super().__init__()

        # 実数パラメータとして保持（量子化は forward で）
        self.Y_fp = nn.Parameter(Y.clone().float())  # (p, j, k, m, l)
        self.Z_fp = nn.Parameter(Z.clone().float())  # (p, j, k, l, n)

        self.quant_bias = quant_bias
        if quant_bias:
            # A: (p, j, k, 4)  -> a,b,c,d
            self.A = nn.Parameter(A.clone().float())
        else:
            # A: (p, j, k)     -> 係数のみ
            self.A = nn.Parameter(A.clone().float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone().float())
        else:
            self.bias = None

        self.act_bits = act_bits

        if act_bits is not None:
            self.act_scale = nn.Parameter(torch.tensor(1e-3))  # アクティベーション量子化スケール

        # 形状情報
        p, j, k, m, l = Y.shape
        _, _, _, _, n = Z.shape
        self.p = p
        self.j = j
        self.k = k
        self.m = m
        self.l = l
        self.n = n

        self.in_features = k * n
        self.out_features = j * m
    
    def make_binary_random_matrix(self, dim, seed=0, device="cpu"):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # {-1, +1} の行列
        R = torch.randint(0, 2, (dim, dim), generator=g, device=device, dtype=torch.float32)
        R = R * 2 - 1  # 0→-1, 1→+1
        return R  # (dim, dim)



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (..., in_features = k * n)
        output: (..., out_features = j * m)

        内部計算はすべて fp32 (float32) で行い、
        入力と出力の dtype は元の input.dtype を保つ。
        """
        orig_dtype = input.dtype
        device = self.Y_fp.device

        # ===== 1. アクティベーション量子化 (SymQuantSTE, 任意ビット) =====
        # 内部計算用に fp32 へ
        X = input.to(device=device, dtype=torch.float32) #@ self.make_binary_random_matrix(self.in_features, device=device)  # (..., k*n)

        if self.act_bits is not None:
            X = SymQuantSTE.apply(X, self.act_scale, self.act_bits)

        # ===== 2. Y, Z を 1-bit SymQuantSTE で量子化 =====
        # まず fp32 & 正しい device に揃える
        Y_fp = self.Y_fp.to(device=device, dtype=torch.float32)
        Z_fp = self.Z_fp.to(device=device, dtype=torch.float32)

        # Y: (p, j, k, m, l) -> scale over (m,l)
        Y_scale = Y_fp.abs().mean(dim=(-2, -1), keepdim=True)  # (p, j, k, 1, 1)
        # Z: (p, j, k, l, n) -> scale over (l,n)
        Z_scale = Z_fp.abs().mean(dim=(-2, -1), keepdim=True)  # (p, j, k, 1, 1)

        # 必要ならゼロ割り防止の微小値を入れてもよい
        # Y_scale = Y_scale.clamp_min(1e-8)
        # Z_scale = Z_scale.clamp_min(1e-8)

        Y_q = SymQuantSTE.apply(Y_fp, Y_scale, 1)  # fp32
        Z_q = SymQuantSTE.apply(Z_fp, Z_scale, 1)  # fp32

        # 形状を展開
        p, j, k, m, l = Y_q.shape
        n = Z_q.shape[-1]  # (p, j, k, l, n)

        # ===== 3. 入力 X を (B, k, n) に reshape =====
        orig_shape = X.shape[:-1]                # (...,)
        X_2d = X.reshape(-1, self.in_features)   # (B, k*n)
        B = X_2d.shape[0]
        X_kn = X_2d.view(B, k, n)                # (B, k, n)  fp32

        Yq = Y_q  # すでに fp32 & device 揃っている
        Zq = Z_q

        # ===== 4. メイン項: a * (Y_q @ Z_q) を介した X @ W^T =====
        # T[b,p,j,k,l] = sum_n Z_q[p,j,k,l,n] * X[b,k,n]
        #   -> einsum: "bkn,pjkln->bpjkl"
        T = torch.einsum("bkn,pjkln->bpjkl", X_kn, Zq)  # (B, p, j, k, l), fp32

        # core[b,p,j,k,m] = sum_l Y_q[p,j,k,m,l] * T[b,p,j,k,l]
        #   -> einsum: "pjkml,bpjkl->bpjkm"
        core = torch.einsum("pjkml,bpjkl->bpjkm", Yq, T)  # (B, p, j, k, m), fp32

        if self.quant_bias:
            # A: (p, j, k, 4)  -> a,b,c,d （fp32 & 正しい device）
            A = self.A.to(device=device, dtype=torch.float32)

            # ----- a 項 -----
            a = A[..., 0]                       # (p, j, k)
            a = a.unsqueeze(0).unsqueeze(-1)    # (1, p, j, k, 1)
            core_a = core * a                   # (B, p, j, k, m)
            out1 = core_a.sum(dim=(1, 3))       # sum over p,k -> (B, j, m)

            # ===== 5. b 項: b * Y_sum * sum_n X =====
            # Y_sum[p,j,k,m] = sum_l Y_q[p,j,k,m,l]
            Y_sum = Yq.sum(dim=-1)              # (p, j, k, m)

            b = A[..., 1]                       # (p, j, k)
            # B_coef[j,k,m] = sum_p b[p,j,k] * Y_sum[p,j,k,m]
            B_coef = (b.unsqueeze(-1) * Y_sum).sum(dim=0)  # (j, k, m)

            # Sx[b,k] = sum_n X[b,k,n]
            Sx = X_kn.sum(dim=-1)               # (B, k)

            # out2[b,j,m] = sum_k Sx[b,k] * B_coef[j,k,m]
            out2 = torch.einsum("bk,jkm->bjm", Sx, B_coef)  # (B, j, m)

            # ===== 6. c 項: c * Z_sum * X =====
            # Zs[p,j,k,n] = sum_l Z_q[p,j,k,l,n]
            Zs = Zq.sum(dim=-2)                 # (p, j, k, n)

            # Tz[b,p,j,k] = sum_n Zs[p,j,k,n] * X[b,k,n]
            Tz = torch.einsum("bkn,pjkn->bpjk", X_kn, Zs)   # (B, p, j, k)

            c = A[..., 2]                       # (p, j, k)
            # out3_base[b,j] = sum_{p,k} c[p,j,k] * Tz[b,p,j,k]
            out3_base = (Tz * c.unsqueeze(0)).sum(dim=(1, 3))  # (B, j)
            out3 = out3_base.unsqueeze(-1).expand(-1, -1, m)   # (B, j, m)

            # ===== 7. d 項: d * sum_n X =====
            # 元コード: d = A[...,3].unsqueeze(-1).unsqueeze(-1).sum(dim=0) -> (j,k,1,1)
            d = A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0)  # (j, k, 1, 1)
            D_coef = d[..., 0, 0]                                 # (j, k)

            # base4[b,j] = sum_k D_coef[j,k] * Sx[b,k]
            out4_base = torch.einsum("bk,jk->bj", Sx, D_coef)     # (B, j)
            out4 = out4_base.unsqueeze(-1).expand(-1, -1, m)      # (B, j, m)

            # ===== 8. 4つを合算 =====
            out_bjm = out1 + out2 + out3 + out4   # (B, j, m), fp32

        else:
            # quant_bias=False の場合は a だけ
            # self.A: (p, j, k)
            a = self.A.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, p, j, k, 1)
            core_a = core * a                                # (B, p, j, k, m)
            out_bjm = core_a.sum(dim=(1, 3))                 # (B, j, m)

        # ===== 9. (B, j, m) -> (B, out_features) -> 元shapeへ =====
        out_2d = out_bjm.reshape(B, self.out_features)  # (B, j*m), fp32

        if self.bias is not None:
            # bias も fp32 で足してから最後にまとめて cast
            out_2d = out_2d + self.bias.to(out_2d.device, dtype=torch.float32)

        out = out_2d.view(*orig_shape, self.out_features)  # fp32

        # 戻り値の dtype は元 input に合わせる
        # return out.to(device=input.device, dtype=orig_dtype)
        return out.to(dtype=input.dtype)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BQQLinearInference(nn.Module):
    """
    推論専用 BQQ Linear.
    - Y_sign: {-1, +1} の符号行列 (p, j, k, m, l)
    - Z_sign: {-1, +1} の符号行列 (p, j, k, l, n)
    - Y_scale: (p, j, k, 1, 1)
    - Z_scale: (p, j, k, 1, 1)
    - A: quant_bias=True: (p, j, k, 4) / False: (p, j, k)
    - bias: (j*m,) など
    - act_bits: 入力アクティベーション量子化ビット数（None なら量子化しない）

    学習は想定せず（推論用）。
    """
    def __init__(
        self,
        Y_sign: torch.Tensor,
        Z_sign: torch.Tensor,
        Y_scale: torch.Tensor,
        Z_scale: torch.Tensor,
        A: torch.Tensor,
        bias: torch.Tensor = None,
        act_bits: int = None,
        quant_bias: bool = True,
    ):
        super().__init__()

        # 符号は int8 or float16 で持つ前提
        self.Y_sign = nn.Parameter(Y_sign, requires_grad=False)
        self.Z_sign = nn.Parameter(Z_sign, requires_grad=False)

        # スケール・係数類は fp16/fp32 など
        self.Y_scale = nn.Parameter(Y_scale, requires_grad=False)
        self.Z_scale = nn.Parameter(Z_scale, requires_grad=False)

        self.quant_bias = quant_bias
        self.A = nn.Parameter(A, requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

        self.act_bits = act_bits

        # 形状情報
        p, j, k, m, l = Y_sign.shape
        _, _, _, _, n = Z_sign.shape
        self.p = p
        self.j = j
        self.k = k
        self.m = m
        self.l = l
        self.n = n

        self.in_features = k * n
        self.out_features = j * m

    @classmethod
    def from_trained(cls, layer: "BQQLinear", sign_dtype=torch.int8, scale_dtype=torch.float16):
        """
        学習済み BQQLinear から推論用 BQQLinearInference を生成するヘルパー。
        """
        device = layer.Y_fp.device

        with torch.no_grad():
            # fp32 でスケール計算
            Y_fp = layer.Y_fp.detach().to(torch.float16)
            Z_fp = layer.Z_fp.detach().to(torch.float16)

            # 元の forward と同じ定義
            Y_scale = Y_fp.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # (p,j,k,1,1)
            Z_scale = Z_fp.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # (p,j,k,1,1)

            # 符号だけ保持（±1）
            Y_sign = torch.sign(Y_fp)
            Z_sign = torch.sign(Z_fp)

            if sign_dtype is not None:
                Y_sign = Y_sign.to(sign_dtype)
                Z_sign = Z_sign.to(sign_dtype)

            Y_scale = Y_scale.to(scale_dtype)
            Z_scale = Z_scale.to(scale_dtype)

            A = layer.A.detach().to(scale_dtype).clone()
            bias = layer.bias.detach().to(scale_dtype).clone() if layer.bias is not None else None

            return cls(
                Y_sign=Y_sign.to(device),
                Z_sign=Z_sign.to(device),
                Y_scale=Y_scale.to(device),
                Z_scale=Z_scale.to(device),
                A=A.to(device),
                bias=bias.to(device) if bias is not None else None,
                act_bits=layer.act_bits,
                quant_bias=layer.quant_bias,
            )
        
    @classmethod
    def from_quant_tensors(
        cls,
        Y_fp: torch.Tensor,   # (p,j,k,m,l)
        Z_fp: torch.Tensor,   # (p,j,k,l,n)
        A: torch.Tensor,      # quant_bias=True: (p,j,k,4) / False: (p,j,k)
        bias: torch.Tensor = None,
        act_bits: int = None,
        quant_bias: bool = True,
        sign_dtype=torch.int8,
        scale_dtype=torch.float16,
        device: str | torch.device = "cpu",
    ):
        """
        量子化済み BQQ の実数パラメータ (Y_fp, Z_fp, A, bias) から
        推論専用 BQQLinearInference を構築する。
        """
        device = torch.device(device)

        with torch.no_grad():
            Y_fp = Y_fp.to(device=device, dtype=torch.float16)
            Z_fp = Z_fp.to(device=device, dtype=torch.float16)
            A_fp = A.to(device=device, dtype=torch.float16)
            b_fp = bias.to(device=device, dtype=torch.float16) if bias is not None else None

            # 学習時と同じスケール定義
            Y_scale = Y_fp.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # (p,j,k,1,1)
            Z_scale = Z_fp.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # (p,j,k,1,1)

            # 符号だけ残す（±1）
            Y_sign = torch.sign(Y_fp)
            Z_sign = torch.sign(Z_fp)

            if sign_dtype is not None:
                Y_sign = Y_sign.to(sign_dtype)
                Z_sign = Z_sign.to(sign_dtype)

            Y_scale = Y_scale.to(scale_dtype)
            Z_scale = Z_scale.to(scale_dtype)
            A_q = A_fp.to(scale_dtype)
            b_q = b_fp.to(scale_dtype) if b_fp is not None else None

            layer = cls(
                Y_sign=Y_sign,
                Z_sign=Z_sign,
                Y_scale=Y_scale,
                Z_scale=Z_scale,
                A=A_q,
                bias=b_q,
                act_bits=act_bits,
                quant_bias=quant_bias,
            )
            return layer.to(device)

    def _quantize_activation(self, X: torch.Tensor) -> torch.Tensor:
        """
        推論時のアクティベーション量子化（STE なし、単純に丸めるだけ）
        """
        if self.act_bits is None:
            return X

        x_max = X.max()
        x_min = X.min()

        if self.act_bits == 1:
            act_scale = (x_max - x_min) / 2
            act_scale = act_scale.clamp_min(1e-8)
            return act_scale * torch.sign(X)
        else:
            qmax = 2 ** (self.act_bits - 1) - 1
            act_scale = (x_max - x_min) / (2 * qmax)
            act_scale = act_scale.clamp_min(1e-8)
            q = torch.clamp(torch.round(X / act_scale), -qmax, qmax)
            return q * act_scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (..., in_features = k * n)
        output: (..., out_features = j * m)
        """
        orig_dtype = input.dtype
        device = self.Y_sign.device

        # 内部は fp32 で計算
        X = input.to(device=device, dtype=torch.float16)

        # アクティベーション量子化（必要なら）
        X = self._quantize_activation(X)

        # 1-bit 重みをスケールで戻す（ここだけ連続値）
        Y_sign_f = self.Y_sign.to(dtype=torch.float16)
        Z_sign_f = self.Z_sign.to(dtype=torch.float16)
        Y_scale_f = self.Y_scale.to(dtype=torch.float16)
        Z_scale_f = self.Z_scale.to(dtype=torch.float16)

        Y_q = Y_sign_f * Y_scale_f          # (p,j,k,m,l)
        Z_q = Z_sign_f * Z_scale_f          # (p,j,k,l,n)

        # 形状
        p, j, k, m, l = Y_q.shape
        n = Z_q.shape[-1]

        # ===== 入力 reshape =====
        orig_shape = X.shape[:-1]
        X_2d = X.reshape(-1, self.in_features)  # (B, k*n)
        B = X_2d.shape[0]
        X_kn = X_2d.view(B, k, n)               # (B, k, n)

        # ===== メインの BQQ 計算 =====
        # T[b,p,j,k,l] = sum_n Z_q[p,j,k,l,n] * X[b,k,n]
        T = torch.einsum("bkn,pjkln->bpjkl", X_kn, Z_q)          # (B,p,j,k,l)

        # core[b,p,j,k,m] = sum_l Y_q[p,j,k,m,l] * T[b,p,j,k,l]
        core = torch.einsum("pjkml,bpjkl->bpjkm", Y_q, T)        # (B,p,j,k,m)

        if self.quant_bias:
            A = self.A.to(dtype=torch.float16, device=device)    # (p,j,k,4)

            # --- a 項 ---
            a = A[..., 0]                        # (p,j,k)
            a = a.unsqueeze(0).unsqueeze(-1)     # (1,p,j,k,1)
            core_a = core * a                    # (B,p,j,k,m)
            out1 = core_a.sum(dim=(1, 3))        # (B,j,m)

            # --- b 項 ---
            Y_sum = Y_q.sum(dim=-1)              # (p,j,k,m)
            b = A[..., 1]                        # (p,j,k)
            B_coef = (b.unsqueeze(-1) * Y_sum).sum(dim=0)  # (j,k,m)

            Sx = X_kn.sum(dim=-1)                # (B,k)
            out2 = torch.einsum("bk,jkm->bjm", Sx, B_coef)       # (B,j,m)

            # --- c 項 ---
            Zs = Z_q.sum(dim=-2)                 # (p,j,k,n)
            Tz = torch.einsum("bkn,pjkn->bpjk", X_kn, Zs)        # (B,p,j,k)

            c = A[..., 2]                        # (p,j,k)
            out3_base = (Tz * c.unsqueeze(0)).sum(dim=(1, 3))    # (B,j)
            out3 = out3_base.unsqueeze(-1).expand(-1, -1, m)     # (B,j,m)

            # --- d 項 ---
            d = A[..., 3].unsqueeze(-1).unsqueeze(-1).sum(dim=0)  # (j,k,1,1)
            D_coef = d[..., 0, 0]                                 # (j,k)
            out4_base = torch.einsum("bk,jk->bj", Sx, D_coef)     # (B,j)
            out4 = out4_base.unsqueeze(-1).expand(-1, -1, m)      # (B,j,m)

            out_bjm = out1 + out2 + out3 + out4                   # (B,j,m)

        else:
            # quant_bias=False: a だけ
            a = self.A.to(dtype=torch.float16, device=device).unsqueeze(0).unsqueeze(-1)  # (1,p,j,k,1)
            core_a = core * a
            out_bjm = core_a.sum(dim=(1, 3))       # (B,j,m)

        out_2d = out_bjm.reshape(B, self.out_features)  # (B,j*m)

        if self.bias is not None:
            out_2d = out_2d + self.bias.to(out_2d.device, dtype=torch.float16)

        out = out_2d.view(*orig_shape, self.out_features)

        return out.to(dtype=orig_dtype)
