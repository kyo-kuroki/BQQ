import torch
import torch.nn as nn
import os
from scipy.linalg import hadamard
from tqdm import tqdm


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
            # print(f"Replacing {full_name}: {type(module)}")
            # print('weight shape:', module.weight.shape, 'bias is None ?:', module.bias is None)

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
            bqq = BinaryQuadratic(Y, Z, A, bias=module.bias)
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
