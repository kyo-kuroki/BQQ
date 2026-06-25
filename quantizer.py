import numpy as np
from tqdm import tqdm, trange
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.autograd import Function
import copy
import matplotlib.pyplot as plt
import os
import time
import math
import pynvml
from math import ceil, log2
from scipy.linalg import hadamard
from scipy.fftpack import dct, idct
from itertools import combinations
from PIL import Image
import io


class BinarySTE01(Function):
    """Binary {0,1} quantization with learnable threshold and sigmoid STE."""

    @staticmethod
    def forward(ctx, input, theta, beta):
        centered = input - theta
        beta_clamped = torch.clamp(beta, min=1e-6)
        output = (centered > 0).to(input.dtype)
        ctx.save_for_backward(centered, beta_clamped)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        centered, beta = ctx.saved_tensors
        scaled = beta * centered
        sigma = torch.sigmoid(scaled)
        sigma_grad = sigma * (1.0 - sigma)
        surrogate = beta * sigma_grad
        grad_common = grad_output * surrogate
        grad_input = grad_common
        grad_theta = -grad_common
        grad_beta = torch.sum(grad_output * centered * sigma_grad).reshape_as(beta)
        return grad_input, grad_theta, grad_beta


class _BQQSTERefinementModule(nn.Module):
    def __init__(self, all_decomposed, weight_shape, optimize_factors=True, optimize_coeffs=True, optimize_theta=True, optimize_beta=True):
        super().__init__()
        self.weight_shape = tuple(weight_shape)
        self.entries = []

        sorted_entries = sorted(
            all_decomposed,
            key=lambda e: (e['patch_row'], e['patch_col'], e['bit_idx'])
        )

        for idx, entry in enumerate(sorted_entries):
            y_init = entry['mat1'].float().clone()
            z_init = entry['mat2'].float().clone()
            coeff_init = entry['coeff'].float().clone()
            y_theta_init = torch.tensor(0.5, dtype=torch.float32)
            z_theta_init = torch.tensor(0.5, dtype=torch.float32)
            y_beta_init = torch.tensor(4.0, dtype=torch.float32)
            z_beta_init = torch.tensor(4.0, dtype=torch.float32)

            y_param = nn.Parameter(y_init, requires_grad=optimize_factors)
            z_param = nn.Parameter(z_init, requires_grad=optimize_factors)
            y_theta = nn.Parameter(y_theta_init, requires_grad=optimize_theta)
            z_theta = nn.Parameter(z_theta_init, requires_grad=optimize_theta)
            y_beta = nn.Parameter(y_beta_init, requires_grad=optimize_beta)
            z_beta = nn.Parameter(z_beta_init, requires_grad=optimize_beta)
            coeff_param = nn.Parameter(coeff_init, requires_grad=optimize_coeffs)

            self.register_parameter(f'y_fp_{idx}', y_param)
            self.register_parameter(f'z_fp_{idx}', z_param)
            self.register_parameter(f'y_theta_{idx}', y_theta)
            self.register_parameter(f'z_theta_{idx}', z_theta)
            self.register_parameter(f'y_beta_{idx}', y_beta)
            self.register_parameter(f'z_beta_{idx}', z_beta)
            self.register_parameter(f'coeff_{idx}', coeff_param)

            self.entries.append({
                'row_start': entry['row_start'],
                'row_end': entry['row_end'],
                'col_start': entry['col_start'],
                'col_end': entry['col_end'],
                'patch_row': entry['patch_row'],
                'patch_col': entry['patch_col'],
                'bit_idx': entry['bit_idx'],
                'y_name': f'y_fp_{idx}',
                'z_name': f'z_fp_{idx}',
                'y_theta_name': f'y_theta_{idx}',
                'z_theta_name': f'z_theta_{idx}',
                'y_beta_name': f'y_beta_{idx}',
                'z_beta_name': f'z_beta_{idx}',
                'coeff_name': f'coeff_{idx}',
            })

        self.entries_by_patch_row = {}
        self.row_group_ranges = {}
        self.patch_rows_by_height = {}
        for entry in self.entries:
            patch_row = entry['patch_row']
            self.entries_by_patch_row.setdefault(patch_row, []).append(entry)
            if patch_row not in self.row_group_ranges:
                self.row_group_ranges[patch_row] = (entry['row_start'], entry['row_end'])
        for patch_row, (r0, r1) in self.row_group_ranges.items():
            self.patch_rows_by_height.setdefault(r1 - r0, []).append(patch_row)

    def reconstruct_row_group_batch(self, patch_rows):
        if len(patch_rows) == 0:
            raise ValueError('patch_rows must not be empty')
        first_param = next(self.parameters(), None)
        device = first_param.device if first_param is not None else torch.device('cpu')
        row_blocks = []
        row_ranges = []
        for patch_row in patch_rows:
            r0, r1 = self.row_group_ranges[patch_row]
            block = torch.zeros(r1 - r0, self.weight_shape[1], dtype=torch.float32, device=device)
            for entry in self.entries_by_patch_row[patch_row]:
                y_fp = getattr(self, entry['y_name'])
                z_fp = getattr(self, entry['z_name'])
                y_theta = getattr(self, entry['y_theta_name'])
                z_theta = getattr(self, entry['z_theta_name'])
                y_beta = getattr(self, entry['y_beta_name'])
                z_beta = getattr(self, entry['z_beta_name'])
                coeff = getattr(self, entry['coeff_name'])
                Y_q = BinarySTE01.apply(y_fp, y_theta, y_beta)
                Z_q = BinarySTE01.apply(z_fp, z_theta, z_beta)
                patch = (coeff[0] * (Y_q @ Z_q)
                        + coeff[1] * Y_q.sum(dim=1, keepdim=True)
                        + coeff[2] * Z_q.sum(dim=0, keepdim=True)
                        + coeff[3])
                block[:, entry['col_start']:entry['col_end']] += patch
            row_blocks.append(block)
            row_ranges.append((r0, r1))
        return torch.stack(row_blocks, dim=0), row_ranges

    def reconstruct_weight(self):
        first_param = next(self.parameters(), None)
        device = first_param.device if first_param is not None else torch.device('cpu')
        Wq = torch.zeros(self.weight_shape, dtype=torch.float32, device=device)
        for entry in self.entries:
            y_fp = getattr(self, entry['y_name'])
            z_fp = getattr(self, entry['z_name'])
            y_theta = getattr(self, entry['y_theta_name'])
            z_theta = getattr(self, entry['z_theta_name'])
            y_beta = getattr(self, entry['y_beta_name'])
            z_beta = getattr(self, entry['z_beta_name'])
            coeff = getattr(self, entry['coeff_name'])
            Y_q = BinarySTE01.apply(y_fp, y_theta, y_beta)
            Z_q = BinarySTE01.apply(z_fp, z_theta, z_beta)
            patch = (coeff[0] * (Y_q @ Z_q)
                    + coeff[1] * Y_q.sum(dim=1, keepdim=True)
                    + coeff[2] * Z_q.sum(dim=0, keepdim=True)
                    + coeff[3])
            Wq[entry['row_start']:entry['row_end'], entry['col_start']:entry['col_end']] += patch
        return Wq

    def export_decomposition(self):
        exported = []
        for entry in self.entries:
            y_fp = getattr(self, entry['y_name'])
            z_fp = getattr(self, entry['z_name'])
            y_theta = getattr(self, entry['y_theta_name'])
            z_theta = getattr(self, entry['z_theta_name'])
            coeff = getattr(self, entry['coeff_name'])
            exported.append({
                'patch_row': entry['patch_row'],
                'patch_col': entry['patch_col'],
                'row_start': entry['row_start'],
                'row_end': entry['row_end'],
                'col_start': entry['col_start'],
                'col_end': entry['col_end'],
                'coeff': coeff.detach().cpu().float(),
                'mat1': (y_fp.detach().cpu() > y_theta.detach().cpu()).float(),
                'mat2': (z_fp.detach().cpu() > z_theta.detach().cpu()).float(),
                'bit_idx': entry['bit_idx'],
            })
        return exported




class BinaryQuadraticQuantization():

    def __init__(self, x, rank=None, rank_scale=1, num_stack=1):
        self.rank_scale=rank_scale
        self.num_stack = num_stack
        if isinstance(x, torch.Tensor):
        # GPU上に存在する場合はCPUに移動
            if x.is_cuda:
                x = x.detach().cpu()
                # NumPy配列に変換
            x = x.float().numpy()

        self.x = copy.copy(x)
        self.ndim = self.x.ndim
        self.numel = self.x.size
        if self.ndim == 2:
            self.Nrow, self.Ncol = x.shape
        elif self.ndim ==3:
            self.batch_size, self.Nrow, self.Ncol = x.shape
        else:
            raise ValueError('2次元または3次元のテンソルを入力してください')
        if rank is None:
            self.rank = int(self.rank_scale*self.Ncol*self.Nrow/(self.Nrow + self.Ncol))
        else:
            self.rank = rank
    
        

    def run_bqq(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1, output_type='numpy'):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        n, m = self.x.shape
        # 初期値
        torch.manual_seed(seed)
        yb = torch.rand((n, self.rank), device=device)
        zb = torch.rand((self.rank, m), device=device)
        y = yb - eta * (yb - 0.5)
        z = zb - eta * (zb - 0.5)

        # 入力をGPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        maximum = (x.max() - x.min()) #最大幅(最大-最小)
        x = x/maximum
        # スケーリングファクタの最終要素の１次係数
        coeff = -2 * x.sum()

        # パーツの計算
        yz, y2z2, sigma_y, sigma_y2, sigma_z, sigma_z2 = y @ z, y**2 @ z**2, y.sum(axis=1).unsqueeze(1), (y**2).sum(axis=1).unsqueeze(1), z.sum(axis=0).unsqueeze(0), (z**2).sum(axis=0).unsqueeze(0) 

        # スケーリング係数の最適化
        r0c0, r0c1, r0c2, r0c3, r1c1, r1c2, r1c3, r2c2, r2c3, r3c3 = (yz**2 + yz - y2z2).sum(), ((sigma_y + 1) * yz - y**2 @ z).sum(), ((1 + sigma_z) * yz - y @ z**2).sum(), yz.sum(), (sigma_y**2 + sigma_y - sigma_y2).sum() * m, (sigma_y * sigma_z).sum(), sigma_y.sum() * m, (sigma_z**2 + sigma_z - sigma_z2).sum() * n, sigma_z.sum() * n, n * m
        hesse = 2*torch.tensor([[r0c0, r0c1, r0c2, r0c3],
                    [r0c1, r1c1, r1c2, r1c3],
                    [r0c2, r1c2, r2c2, r2c3],
                    [r0c3, r1c3, r2c3, r3c3]], device=device)
        v = torch.tensor([(-2 * x * yz).sum(), (-2 * x * sigma_y).sum(), (-2 * x * sigma_z).sum(), coeff], device=device)
        
        # スケーリング係数の初期値
        try:
            a = -torch.linalg.solve(hesse, v)
        except RuntimeError as e:
            if "singular" in str(e) or "input is not invertible" in str(e):
                a = -torch.matmul(torch.linalg.pinv(hesse, rcond=1e-15), v)
            else: raise

        for _ in range(Nstep):#tqdm(range(Nstep)):
            # 前進点での計算
            yf = (y + zeta * (y - yb))
            zf = (z + zeta * (z - zb))

            # パーツの計算
            yz, sigma_y, sigma_z = yf @ zf, yf.sum(axis=1).unsqueeze(1), zf.sum(axis=0).unsqueeze(0)
            part = x - (a[3] + a[0] * yz + a[1] * sigma_y + a[2] * sigma_z) ## sigma_zはaxis=1でsumだけどここではaxis=0でsumなことに注意 (yも同様)

            # 平均場エネルギー勾配の計算 (改良版)
            y_energy_grad = (-2 * part @ (a[0] * zf + a[1]).T) + (a[0]**2 + 2*a[0]*a[1]*(1 - 2*yf) + 2*a[0]*a[2]) * (zf.sum(axis=1).unsqueeze(0)) - 2 * (a[0]*a[2] + a[0]**2 * yf) * (zf**2).sum(axis=1).unsqueeze(0) + (a[1]**2) * (1 - 2 * yf) * m
            z_energy_grad = (-2 * (a[0] * yf + a[2]).T @ part) + (a[0]**2 + 2*a[0]*a[1] + 2*a[0]*a[2]*(1 - 2*zf)) * (yf.sum(axis=0).unsqueeze(1)) - 2 * (a[0]**2 * zf + a[0]*a[1]) * (yf**2).sum(axis=0).unsqueeze(1) + (a[2]**2) * (1 - 2 * zf) * n


            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone().detach()
            zb = z.clone().detach()
            y = ya.clone().detach()
            z = za.clone().detach()

            # パーツの計算
            yz, y2z2, sigma_y, sigma_y2, sigma_z, sigma_z2 = y @ z, y**2 @ z**2, y.sum(axis=1).unsqueeze(1), (y**2).sum(axis=1).unsqueeze(1), z.sum(axis=0).unsqueeze(0), (z**2).sum(axis=0).unsqueeze(0) 

            # スケーリング係数の最適化 (改良版)
            r0c0, r0c1, r0c2, r0c3, r1c1, r1c2, r1c3, r2c2, r2c3, r3c3 = (yz**2 + yz - y2z2).sum(), ((sigma_y + 1) * yz - y**2 @ z).sum(), ((1 + sigma_z) * yz - y @ z**2).sum(), yz.sum(), (sigma_y**2 + sigma_y - sigma_y2).sum() * m, (sigma_y * sigma_z).sum(), sigma_y.sum() * m, (sigma_z**2 + sigma_z - sigma_z2).sum() * n, sigma_z.sum() * n, n * m
            hesse = 2*torch.tensor([[r0c0, r0c1, r0c2, r0c3],
                        [r0c1, r1c1, r1c2, r1c3],
                        [r0c2, r1c2, r2c2, r2c3],
                        [r0c3, r1c3, r2c3, r3c3]], device=device)
            v = torch.tensor([(-2 * x * yz).sum(), (-2 * x * sigma_y).sum(), (-2 * x * sigma_z).sum(), coeff], device=device)
            try:
                a = -torch.linalg.solve(hesse, v)
            except RuntimeError as e:
                if "singular" in str(e) or "input is not invertible" in str(e):
                    a = -torch.matmul(torch.linalg.pinv(hesse, rcond=1e-15), v)
                else: raise

            # tempの減少
            temp -= self.delta_temp

        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # パーツの計算
        yz, y2z2, sigma_y, sigma_y2, sigma_z, sigma_z2 = y @ z, y**2 @ z**2, y.sum(axis=1).unsqueeze(1), (y**2).sum(axis=1).unsqueeze(1), z.sum(axis=0).unsqueeze(0), (z**2).sum(axis=0).unsqueeze(0) 
        # スケーリング係数の最適化
        r0c0, r0c1, r0c2, r0c3, r1c1, r1c2, r1c3, r2c2, r2c3, r3c3 = (yz**2 + yz - y2z2).sum(), ((sigma_y + 1) * yz - y**2 @ z).sum(), ((1 + sigma_z) * yz - y @ z**2).sum(), yz.sum(), (sigma_y**2 + sigma_y - sigma_y2).sum() * m, (sigma_y * sigma_z).sum(), sigma_y.sum() * m, (sigma_z**2 + sigma_z - sigma_z2).sum() * n, sigma_z.sum() * n, n * m
        hesse = 2*torch.tensor([[r0c0, r0c1, r0c2, r0c3],
                    [r0c1, r1c1, r1c2, r1c3],
                    [r0c2, r1c2, r2c2, r2c3],
                    [r0c3, r1c3, r2c3, r3c3]], device=device)
        v = torch.tensor([(-2 * x * yz).sum(), (-2 * x * sigma_y).sum(), (-2 * x * sigma_z).sum(), coeff], device=device)
        try:
            a = -torch.linalg.solve(hesse, v)
        except RuntimeError as e:
            if "singular" in str(e) or "input is not invertible" in str(e):
                a = -torch.matmul(torch.linalg.pinv(hesse, rcond=1e-15), v)
            else: raise


        if output_type == 'torch':
            return y, z, maximum*a
        else:
            return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy()
        


        

    def run_bqq_compile(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1, output_type='torch', compile_mode="reduce-overhead", binarize_scaling=False):
        """
        Args:
            binarize_scaling: Trueの場合、ループ内でcompute_aに二値化した値を渡す(V1方式)。
                              Falseの場合、連続値のまま渡す(V2方式、デフォルト)。
        """
        torch.set_float32_matmul_precision('medium')

        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        n, m = self.x.shape

        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)

        torch.manual_seed(seed)
        yb = torch.rand((n, self.rank), device=device)
        zb = torch.rand((self.rank, m), device=device)
        y = yb - eta * (yb - 0.5)
        z = zb - eta * (zb - 0.5)
        
        x = torch.from_numpy(self.x).float().to(device)
        maximum = (x.max() - x.min())
        x = x / maximum
        coeff = -2 * x.sum()
        num_element = torch.tensor(n * m, device=device)


        def compute_a(y, z):

            yz, sigma_y, sigma_z = y @ z, y.sum(dim=1, keepdim=True), z.sum(dim=0, keepdim=True)
            # スケーリング係数の最適化
            r0c0 = (yz**2 + yz - (y**2 @ z**2)).sum()
            r0c1 = ((sigma_y + 1) * yz - y**2 @ z).sum()
            r0c2 = ((1 + sigma_z) * yz - y @ z**2).sum()
            r0c3 = yz.sum()
            r1c1 = (sigma_y**2 + sigma_y - (y**2).sum(axis=1).unsqueeze(1)).sum() * m
            r1c2 = (sigma_y * sigma_z).sum()
            r1c3 = sigma_y.sum() * m
            r2c2 = (sigma_z**2 + sigma_z - (z**2).sum(axis=0).unsqueeze(0)).sum() * n
            r2c3 = sigma_z.sum() * n
            r3c3 = num_element
            
            # hesseとvをインプレースでなく再生成
            hesse = 2 * torch.stack([
                torch.stack([r0c0, r0c1, r0c2, r0c3]),
                torch.stack([r0c1, r1c1, r1c2, r1c3]),
                torch.stack([r0c2, r1c2, r2c2, r2c3]),
                torch.stack([r0c3, r1c3, r2c3, r3c3])
            ])
            
            v = torch.stack([
                (-2 * x * yz).sum(),
                (-2 * x * sigma_y).sum(),
                (-2 * x * sigma_z).sum(),
                coeff
            ])
  
            # return -torch.linalg.solve(hesse, v)
            try:
                # 通常の解法を試みる
                return -torch.linalg.solve(hesse, v)
            except RuntimeError as e:
                # 特異な場合などにエラーが出たら pinv を使って解く
                if "singular" in str(e) or "input is not invertible" in str(e):
                    return -torch.matmul(torch.linalg.pinv(hesse, rcond=1e-15), v)
                else:
                    raise  # その他のエラーは再送出

        a = compute_a(y, z)

        def _loop_body_continuous(y, z, yb, zb, a, temp):
            with torch.no_grad():

                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)
                part = x - (a[3] + a[0] * (yf @ zf) + a[1] * yf.sum(dim=1, keepdim=True) + a[2] * zf.sum(dim=0, keepdim=True))

                y_energy_grad = (-2 * part @ (a[0] * zf + a[1]).T) + (a[0]**2 + 2*a[0]*a[1]*(1 - 2*yf) + 2*a[0]*a[2]) * (zf.sum(axis=1).unsqueeze(0)) - 2 * (a[0]*a[2] + a[0]**2 * yf) * (zf**2).sum(axis=1).unsqueeze(0) + (a[1]**2) * (1 - 2 * yf) * m
                z_energy_grad = (-2 * (a[0] * yf + a[2]).T @ part) + (a[0]**2 + 2*a[0]*a[1] + 2*a[0]*a[2]*(1 - 2*zf)) * (yf.sum(axis=0).unsqueeze(1)) - 2 * (a[0]**2 * zf + a[0]*a[1]) * (yf**2).sum(axis=0).unsqueeze(1) + (a[2]**2) * (1 - 2 * zf) * n

                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
                za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

                a = compute_a(ya, za)
            return ya, za, y, z, a

        def _loop_body_binarize(y, z, yb, zb, a, temp):
            with torch.no_grad():

                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)
                part = x - (a[3] + a[0] * (yf @ zf) + a[1] * yf.sum(dim=1, keepdim=True) + a[2] * zf.sum(dim=0, keepdim=True))

                y_energy_grad = (-2 * part @ (a[0] * zf + a[1]).T) + (a[0]**2 + 2*a[0]*a[1]*(1 - 2*yf) + 2*a[0]*a[2]) * (zf.sum(axis=1).unsqueeze(0)) - 2 * (a[0]*a[2] + a[0]**2 * yf) * (zf**2).sum(axis=1).unsqueeze(0) + (a[1]**2) * (1 - 2 * yf) * m
                z_energy_grad = (-2 * (a[0] * yf + a[2]).T @ part) + (a[0]**2 + 2*a[0]*a[1] + 2*a[0]*a[2]*(1 - 2*zf)) * (yf.sum(axis=0).unsqueeze(1)) - 2 * (a[0]**2 * zf + a[0]*a[1]) * (yf**2).sum(axis=0).unsqueeze(1) + (a[2]**2) * (1 - 2 * zf) * n

                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
                za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

                a = compute_a(torch.where(ya>0.5, 1.0, 0.0), torch.where(za>0.5, 1.0, 0.0))
            return ya, za, y, z, a

        _loop_fn = _loop_body_binarize if binarize_scaling else _loop_body_continuous
        loop_body = torch.compile(_loop_fn, mode=compile_mode)

        # gpuに移動
        temp = torch.tensor(temp, device=device)
        self.delta_temp = torch.tensor(self.delta_temp, device=device)
        for _ in range(Nstep): # trange(Nstep, desc='Decomposing', mininterval=10.0): 
        # for _ in range(Nstep): 
            y = y.detach().clone()
            yb = yb.detach().clone()
            z = z.detach().clone()
            zb = zb.detach().clone()
            a = a.detach().clone()
            y, z, yb, zb, a = loop_body(y, z, yb, zb, a, temp)
            temp -= self.delta_temp

        # 後処理
        y = torch.where(y > 0.5, 1.0, 0.0)
        z = torch.where(z > 0.5, 1.0, 0.0)
        a = compute_a(y, z)

        if output_type == 'torch':
            return y, z, maximum * a
        else:
            return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum * a).detach().cpu().numpy()


    def run_bqq_compile_batched(self, x, rank_scale=1, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=50000, device_id=0, seed=1, compile_mode="reduce-overhead", binarize_scaling=False):
        """
        バッチ版 run_bqq_compile。(B, n, m) のテンソルをバッチ次元で並列に分解する。
        self.x は参照せず、引数 x をそのまま使用する。

        Args:
            x: 入力テンソル (B, n, m)
            rank_scale: ランクスケール
            binarize_scaling: Trueの場合、ループ内でcompute_aに二値化した値を渡す(V1方式)。
                              Falseの場合、連続値のまま渡す(V2方式、デフォルト)。

        Returns:
            y: (B, n, rank), z: (B, rank, m), a: (B, 4) のスケーリング係数 (maximum込み)
        """
        torch.set_float32_matmul_precision('medium')
        torch.manual_seed(seed)

        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        if x.ndim != 3:
            raise ValueError('Dimension Error: Please input 3-dim (B, n, m) tensor')
        B, n, m = x.shape
        x = x.to(device).float()
        maximum = (x.amax(dim=(1, 2)) - x.amin(dim=(1, 2))).view(B, 1, 1)  # (B,1,1)
        x = x / maximum

        coeff = -2 * x.sum(dim=(1, 2))  # (B,)

        delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)

        rank = round(rank_scale * (n * m) / (n + m))

        yb = torch.rand((B, n, rank), device=device)
        zb = torch.rand((B, rank, m), device=device)
        y = yb - eta * (yb - 0.5)
        z = zb - eta * (zb - 0.5)

        def compute_a(y, z):
            # y: (B, n, rank), z: (B, rank, m)
            yz = torch.bmm(y, z)                          # (B, n, m)
            sigma_y = y.sum(dim=2, keepdim=True)           # (B, n, 1)
            sigma_z = z.sum(dim=1, keepdim=True)           # (B, 1, m)

            r0c0 = (yz**2 + yz - torch.bmm(y**2, z**2)).sum(dim=(1, 2))
            r0c1 = ((sigma_y + 1) * yz - torch.bmm(y**2, z)).sum(dim=(1, 2))
            r0c2 = ((1 + sigma_z) * yz - torch.bmm(y, z**2)).sum(dim=(1, 2))
            r0c3 = yz.sum(dim=(1, 2))
            r1c1 = (sigma_y**2 + sigma_y - (y**2).sum(dim=2, keepdim=True)).sum(dim=(1, 2)) * m
            r1c2 = (sigma_y * sigma_z).sum(dim=(1, 2))
            r1c3 = sigma_y.sum(dim=(1, 2)) * m
            r2c2 = (sigma_z**2 + sigma_z - (z**2).sum(dim=1, keepdim=True)).sum(dim=(1, 2)) * n
            r2c3 = sigma_z.sum(dim=(1, 2)) * n
            r3c3 = torch.full((B,), n * m, device=device, dtype=y.dtype)

            hesse = 2 * torch.stack([
                torch.stack([r0c0, r0c1, r0c2, r0c3], dim=1),
                torch.stack([r0c1, r1c1, r1c2, r1c3], dim=1),
                torch.stack([r0c2, r1c2, r2c2, r2c3], dim=1),
                torch.stack([r0c3, r1c3, r2c3, r3c3], dim=1),
            ], dim=1)  # (B, 4, 4)

            v = torch.stack([
                (-2 * x * yz).sum(dim=(1, 2)),
                (-2 * x * sigma_y).sum(dim=(1, 2)),
                (-2 * x * sigma_z).sum(dim=(1, 2)),
                coeff,
            ], dim=1)  # (B, 4)

            try:
                return -torch.linalg.solve(hesse, v)
            except RuntimeError as e:
                if "singular" in str(e) or "input is not invertible" in str(e):
                    return -torch.bmm(torch.linalg.pinv(hesse, rcond=1e-15), v.unsqueeze(-1)).squeeze(-1)
                else:
                    raise

        a = compute_a(y, z)

        def _loop_body_continuous(y, z, yb, zb, a, temp):
            with torch.no_grad():
                a0 = a[:, 0].view(-1, 1, 1)
                a1 = a[:, 1].view(-1, 1, 1)
                a2 = a[:, 2].view(-1, 1, 1)
                a3 = a[:, 3].view(-1, 1, 1)

                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)
                part = x - (a3 + a0 * torch.bmm(yf, zf) + a1 * yf.sum(dim=2, keepdim=True) + a2 * zf.sum(dim=1, keepdim=True))

                zf_sum = zf.sum(dim=2).unsqueeze(1)       # (B, 1, rank)
                zf2_sum = (zf**2).sum(dim=2).unsqueeze(1)  # (B, 1, rank)
                yf_sum = yf.sum(dim=1).unsqueeze(2)        # (B, rank, 1)
                yf2_sum = (yf**2).sum(dim=1).unsqueeze(2)  # (B, rank, 1)

                y_energy_grad = (-2 * torch.bmm(part, (a0 * zf + a1).transpose(1, 2))) + (a0**2 + 2*a0*a1*(1 - 2*yf) + 2*a0*a2) * zf_sum - 2 * (a0*a2 + a0**2 * yf) * zf2_sum + (a1**2) * (1 - 2 * yf) * m
                z_energy_grad = (-2 * torch.bmm((a0 * yf + a2).transpose(1, 2), part)) + (a0**2 + 2*a0*a1 + 2*a0*a2*(1 - 2*zf)) * yf_sum - 2 * (a0**2 * zf + a0*a1) * yf2_sum + (a2**2) * (1 - 2 * zf) * n

                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
                za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

                a = compute_a(ya, za)
            return ya, za, y, z, a

        def _loop_body_binarize(y, z, yb, zb, a, temp):
            with torch.no_grad():
                a0 = a[:, 0].view(-1, 1, 1)
                a1 = a[:, 1].view(-1, 1, 1)
                a2 = a[:, 2].view(-1, 1, 1)
                a3 = a[:, 3].view(-1, 1, 1)

                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)
                part = x - (a3 + a0 * torch.bmm(yf, zf) + a1 * yf.sum(dim=2, keepdim=True) + a2 * zf.sum(dim=1, keepdim=True))

                zf_sum = zf.sum(dim=2).unsqueeze(1)
                zf2_sum = (zf**2).sum(dim=2).unsqueeze(1)
                yf_sum = yf.sum(dim=1).unsqueeze(2)
                yf2_sum = (yf**2).sum(dim=1).unsqueeze(2)

                y_energy_grad = (-2 * torch.bmm(part, (a0 * zf + a1).transpose(1, 2))) + (a0**2 + 2*a0*a1*(1 - 2*yf) + 2*a0*a2) * zf_sum - 2 * (a0*a2 + a0**2 * yf) * zf2_sum + (a1**2) * (1 - 2 * yf) * m
                z_energy_grad = (-2 * torch.bmm((a0 * yf + a2).transpose(1, 2), part)) + (a0**2 + 2*a0*a1 + 2*a0*a2*(1 - 2*zf)) * yf_sum - 2 * (a0**2 * zf + a0*a1) * yf2_sum + (a2**2) * (1 - 2 * zf) * n

                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
                za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

                a = compute_a(torch.where(ya>0.5, 1.0, 0.0), torch.where(za>0.5, 1.0, 0.0))
            return ya, za, y, z, a

        _loop_fn = _loop_body_binarize if binarize_scaling else _loop_body_continuous
        loop_body = torch.compile(_loop_fn, mode=compile_mode)

        temp = torch.tensor(temp, device=device)
        delta_temp = torch.tensor(delta_temp, device=device)
        for _ in range(Nstep):
            y = y.detach().clone()
            yb = yb.detach().clone()
            z = z.detach().clone()
            zb = zb.detach().clone()
            a = a.detach().clone()
            y, z, yb, zb, a = loop_body(y, z, yb, zb, a, temp)
            temp -= delta_temp

        # 後処理
        y = torch.where(y > 0.5, 1.0, 0.0)
        z = torch.where(z > 0.5, 1.0, 0.0)
        a = compute_a(y, z)

        return y, z, maximum.squeeze(-1) * a  # y: (B,n,rank), z: (B,rank,m), a: (B,4)


    def run_multibqq_compile(self, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=50000,
                              device_id=0, seed=1, output_type='torch',
                              compile_mode="reduce-overhead"):
        """
        Joint multi-stack BQQ optimization (torch.compile-accelerated).

        Model:
            W ≈ Σ_n ( a[n,0] · y[n] @ z[n]
                    + a[n,1] · y[n].sum(axis=-1, keepdim=True)
                    + a[n,2] · z[n].sum(axis=-2, keepdim=True)
                    + a[n,3] )
        with y[n] ∈ {0,1}^(Nrow × rank), z[n] ∈ {0,1}^(rank × Ncol),
             a[n] ∈ R^4, n = 0, ..., num_stack - 1.

        The y, z energy gradient uses the binary identity y^k = y, z^k = z
        (same substitution as run_bqq_compile). The scaling coefficients A are
        jointly optimized by Newton step over a 4N × 4N linear system, where
        diagonal 4×4 blocks use the y^k=y substituted formulas and off-diagonal
        blocks use raw inner products between independent stacks.

        Returns:
            y: (num_stack, Nrow, rank) binary
            z: (num_stack, rank, Ncol) binary
            a: (num_stack, 4)  -- already multiplied by `maximum = max(x)-min(x)`
        """
        torch.set_float32_matmul_precision('medium')
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        N = self.num_stack
        n, m = self.Nrow, self.Ncol
        rank = self.rank

        delta_temp = (Tinit - Tfin) / (Nstep - 1)

        torch.manual_seed(seed)
        yb = torch.rand((N, n, rank), device=device)
        zb = torch.rand((N, rank, m), device=device)
        y = yb - eta * (yb - 0.5)
        z = zb - eta * (zb - 0.5)

        x = torch.from_numpy(self.x).float().to(device)
        maximum = (x.max() - x.min())
        x = x / maximum

        W_row_sum = x.sum(dim=1)           # (n,)
        W_col_sum = x.sum(dim=0)           # (m,)
        W_total = x.sum()                  # scalar (0-dim tensor)
        num_element = float(n * m)

        # Precomputed identity & masks (compile-time constants for fixed N)
        diag_mask = torch.eye(N, dtype=torch.bool, device=device)
        damping_eye = 1e-6 * torch.eye(4 * N, device=device, dtype=x.dtype)

        def compute_A(y, z):
            """Solve 4N × 4N linear system for A: (N, 4) (compile-friendly)."""
            yz = torch.bmm(y, z)                                # (N, n, m)
            sigma_y = y.sum(dim=2)                              # (N, n)
            sigma_z = z.sum(dim=1)                              # (N, m)

            y2 = y * y
            z2 = z * z
            y2z = torch.bmm(y2, z)
            yz2_mat = torch.bmm(y, z2)
            y2z2 = torch.bmm(y2, z2)

            # Diagonal entries (within-stack, y^k=y substituted)
            diag_00 = (yz*yz + yz - y2z2).sum(dim=(1, 2))
            diag_01 = ((sigma_y.unsqueeze(2) + 1) * yz - y2z).sum(dim=(1, 2))
            diag_02 = ((1 + sigma_z.unsqueeze(1)) * yz - yz2_mat).sum(dim=(1, 2))
            diag_11 = (sigma_y*sigma_y + sigma_y - y2.sum(dim=2)).sum(dim=1) * m
            diag_22 = (sigma_z*sigma_z + sigma_z - z2.sum(dim=1)).sum(dim=1) * n

            # Cross-stack inner products (no substitution)
            sigma_y_total = sigma_y.sum(dim=1)                  # (N,)
            sigma_z_total = sigma_z.sum(dim=1)                  # (N,)

            F00_cross = torch.einsum('nij,oij->no', yz, yz)
            F01_cross = torch.einsum('ni,oi->no', yz.sum(dim=2), sigma_y)
            F02_cross = torch.einsum('nj,oj->no', yz.sum(dim=1), sigma_z)
            F03_full = yz.sum(dim=(1, 2)).unsqueeze(1).expand(N, N)
            F11_cross = torch.einsum('ni,oi->no', sigma_y, sigma_y) * m
            F12_full = torch.outer(sigma_y_total, sigma_z_total)
            F13_full = (sigma_y_total * m).unsqueeze(1).expand(N, N)
            F22_cross = torch.einsum('nj,oj->no', sigma_z, sigma_z) * n
            F23_full = (sigma_z_total * n).unsqueeze(1).expand(N, N)
            F33_full = torch.full((N, N), num_element, device=device, dtype=y.dtype)

            # Splice diagonals (use torch.where for compile-friendliness)
            F00 = torch.where(diag_mask, diag_00.unsqueeze(0).expand(N, N), F00_cross)
            F01 = torch.where(diag_mask, diag_01.unsqueeze(0).expand(N, N), F01_cross)
            F02 = torch.where(diag_mask, diag_02.unsqueeze(0).expand(N, N), F02_cross)
            F11 = torch.where(diag_mask, diag_11.unsqueeze(0).expand(N, N), F11_cross)
            F22 = torch.where(diag_mask, diag_22.unsqueeze(0).expand(N, N), F22_cross)
            # F03, F12, F13, F23, F33: diagonal already matches substituted form

            # Assemble 4N × 4N Hessian:  H[(n,i),(n',j)] = 2 · <F_i[n], F_j[n']>
            row0 = torch.stack([F00, F01, F02, F03_full], dim=-1)             # (N, N, 4)
            row1 = torch.stack([F01.t(), F11, F12_full, F13_full], dim=-1)
            row2 = torch.stack([F02.t(), F12_full.t(), F22, F23_full], dim=-1)
            row3 = torch.stack([F03_full.t(), F13_full.t(), F23_full.t(), F33_full], dim=-1)
            H4 = torch.stack([row0, row1, row2, row3], dim=-2)                # (N, N, 4, 4)

            H_full = 2 * H4.permute(0, 2, 1, 3).reshape(4 * N, 4 * N)

            # Vector: v[n, i] = -2 · <W, F_i[n]>
            v0 = -2 * (x * yz).sum(dim=(1, 2))
            v1 = -2 * (W_row_sum.unsqueeze(0) * sigma_y).sum(dim=1)
            v2 = -2 * (W_col_sum.unsqueeze(0) * sigma_z).sum(dim=1)
            v3 = (-2 * W_total).expand(N)
            v_full = torch.stack([v0, v1, v2, v3], dim=1).reshape(4 * N)

            # Tikhonov damping handles the null-space (a3 redundancy across stacks)
            scale = H_full.diagonal().abs().max()
            H_reg = H_full + scale * damping_eye
            A_flat = -torch.linalg.solve(H_reg, v_full)
            return A_flat.view(N, 4)

        a = compute_A(y, z)

        def _loop_fn(y, z, yb, zb, a, temp):
            with torch.no_grad():
                a0 = a[:, 0].view(N, 1, 1)
                a1 = a[:, 1].view(N, 1, 1)
                a2 = a[:, 2].view(N, 1, 1)
                a3 = a[:, 3].view(N, 1, 1)

                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)

                yz = torch.bmm(yf, zf)
                sigma_yf = yf.sum(dim=2, keepdim=True)   # (N, n, 1)
                sigma_zf = zf.sum(dim=1, keepdim=True)   # (N, 1, m)

                M_per = a0 * yz + a1 * sigma_yf + a2 * sigma_zf + a3   # (N, n, m)
                total_M = M_per.sum(dim=0)                              # (n, m)
                part = (x - total_M).unsqueeze(0).expand(N, n, m)       # (N, n, m)

                # Y energy gradient (per stack)
                kernel_y = (a0 * zf + a1).transpose(1, 2)               # (N, m, rank)
                linear_y = -2 * torch.bmm(part, kernel_y)               # (N, n, rank)
                zf_sum1 = zf.sum(dim=2).unsqueeze(1)                    # (N, 1, rank)
                zf2_sum1 = (zf*zf).sum(dim=2).unsqueeze(1)
                y_energy_grad = (
                    linear_y
                    + (a0*a0 + 2*a0*a1*(1 - 2*yf) + 2*a0*a2) * zf_sum1
                    - 2 * (a0*a2 + a0*a0 * yf) * zf2_sum1
                    + (a1*a1) * (1 - 2 * yf) * m
                )

                # Z energy gradient
                kernel_z = (a0 * yf + a2).transpose(1, 2)               # (N, rank, n)
                linear_z = -2 * torch.bmm(kernel_z, part)               # (N, rank, m)
                yf_sum0 = yf.sum(dim=1).unsqueeze(2)                    # (N, rank, 1)
                yf2_sum0 = (yf*yf).sum(dim=1).unsqueeze(2)
                z_energy_grad = (
                    linear_z
                    + (a0*a0 + 2*a0*a1 + 2*a0*a2*(1 - 2*zf)) * yf_sum0
                    - 2 * (a0*a0 * zf + a0*a1) * yf2_sum0
                    + (a2*a2) * (1 - 2 * zf) * n
                )

                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(
                    torch.where((y < 0.0) | (y > 1.0),
                                2*y - yb - eta * y_entropy_grad,
                                2*y - yb - eta * (y_energy_grad + y_entropy_grad)),
                    0, 1)
                za = torch.clamp(
                    torch.where((z < 0.0) | (z > 1.0),
                                2*z - zb - eta * z_entropy_grad,
                                2*z - zb - eta * (z_energy_grad + z_entropy_grad)),
                    0, 1)

                a_new = compute_A(ya, za)
            return ya, za, y, z, a_new

        loop_body = torch.compile(_loop_fn, mode=compile_mode)

        temp = torch.tensor(Tinit, device=device)
        delta_temp_t = torch.tensor(delta_temp, device=device)

        for _ in range(Nstep):
            y = y.detach().clone()
            yb = yb.detach().clone()
            z = z.detach().clone()
            zb = zb.detach().clone()
            a = a.detach().clone()
            y, z, yb, zb, a = loop_body(y, z, yb, zb, a, temp)
            temp = temp - delta_temp_t

        y = torch.where(y > 0.5, 1.0, 0.0)
        z = torch.where(z > 0.5, 1.0, 0.0)
        a = compute_A(y, z)

        if output_type == 'torch':
            return y, z, maximum * a
        else:
            return (y.detach().cpu().numpy(),
                    z.detach().cpu().numpy(),
                    (maximum * a).detach().cpu().numpy())


    def run_multibqq_compile_batched(self, x, num_stack=1, rank_scale=1,
                                       zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
                                       Nstep=50000, device_id=0, seed=1,
                                       compile_mode="reduce-overhead"):
        """
        Batched joint multi-stack BQQ.

        Input:
            x: (B, n, m) -- batch of matrices to factorize

        Returns:
            y: (B, num_stack, n, rank)  binary
            z: (B, num_stack, rank, m)  binary
            a: (B, num_stack, 4)        with per-matrix `maximum` pre-applied
        """
        torch.set_float32_matmul_precision('medium')
        torch.manual_seed(seed)
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        if x.ndim != 3:
            raise ValueError('Input must be 3-D (B, n, m)')
        B, n, m = x.shape
        N = num_stack
        rank = round(rank_scale * (n * m) / (n + m))

        x = x.to(device).float()
        maximum = (x.amax(dim=(1, 2)) - x.amin(dim=(1, 2))).view(B, 1, 1)  # (B,1,1)
        x = x / maximum

        delta_temp = (Tinit - Tfin) / (Nstep - 1)

        yb = torch.rand((B, N, n, rank), device=device)
        zb = torch.rand((B, N, rank, m), device=device)
        y = yb - eta * (yb - 0.5)
        z = zb - eta * (zb - 0.5)

        W_row_sum = x.sum(dim=2)           # (B, n)
        W_col_sum = x.sum(dim=1)           # (B, m)
        W_total = x.sum(dim=(1, 2))        # (B,)
        num_element = float(n * m)

        diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)   # (1, N, N)
        damping_eye = (1e-6 * torch.eye(4 * N, device=device, dtype=x.dtype)).unsqueeze(0)  # (1, 4N, 4N)

        def compute_A(y, z):
            """Solve B parallel 4N×4N linear systems; returns A: (B, N, 4)."""
            yz = torch.einsum('bnik,bnkj->bnij', y, z)        # (B, N, n, m)
            sigma_y = y.sum(dim=3)                            # (B, N, n)
            sigma_z = z.sum(dim=2)                            # (B, N, m)

            y2 = y * y
            z2 = z * z
            y2z = torch.einsum('bnik,bnkj->bnij', y2, z)
            yz2_mat = torch.einsum('bnik,bnkj->bnij', y, z2)
            y2z2 = torch.einsum('bnik,bnkj->bnij', y2, z2)

            diag_00 = (yz*yz + yz - y2z2).sum(dim=(2, 3))                                 # (B, N)
            diag_01 = ((sigma_y.unsqueeze(3) + 1) * yz - y2z).sum(dim=(2, 3))
            diag_02 = ((1 + sigma_z.unsqueeze(2)) * yz - yz2_mat).sum(dim=(2, 3))
            diag_11 = (sigma_y*sigma_y + sigma_y - y2.sum(dim=3)).sum(dim=2) * m
            diag_22 = (sigma_z*sigma_z + sigma_z - z2.sum(dim=2)).sum(dim=2) * n

            sigma_y_total = sigma_y.sum(dim=2)        # (B, N)
            sigma_z_total = sigma_z.sum(dim=2)        # (B, N)

            F00_c = torch.einsum('bnij,boij->bno', yz, yz)
            F01_c = torch.einsum('bni,boi->bno', yz.sum(dim=3), sigma_y)
            F02_c = torch.einsum('bnj,boj->bno', yz.sum(dim=2), sigma_z)
            F03_f = yz.sum(dim=(2, 3)).unsqueeze(2).expand(B, N, N)
            F11_c = torch.einsum('bni,boi->bno', sigma_y, sigma_y) * m
            F12_f = torch.einsum('bn,bo->bno', sigma_y_total, sigma_z_total)
            F13_f = (sigma_y_total * m).unsqueeze(2).expand(B, N, N)
            F22_c = torch.einsum('bnj,boj->bno', sigma_z, sigma_z) * n
            F23_f = (sigma_z_total * n).unsqueeze(2).expand(B, N, N)
            F33_f = torch.full((B, N, N), num_element, device=device, dtype=y.dtype)

            F00 = torch.where(diag_mask, diag_00.unsqueeze(2).expand(B, N, N), F00_c)
            F01 = torch.where(diag_mask, diag_01.unsqueeze(2).expand(B, N, N), F01_c)
            F02 = torch.where(diag_mask, diag_02.unsqueeze(2).expand(B, N, N), F02_c)
            F11 = torch.where(diag_mask, diag_11.unsqueeze(2).expand(B, N, N), F11_c)
            F22 = torch.where(diag_mask, diag_22.unsqueeze(2).expand(B, N, N), F22_c)

            row0 = torch.stack([F00, F01, F02, F03_f], dim=-1)
            row1 = torch.stack([F01.transpose(-1, -2), F11, F12_f, F13_f], dim=-1)
            row2 = torch.stack([F02.transpose(-1, -2), F12_f.transpose(-1, -2), F22, F23_f], dim=-1)
            row3 = torch.stack([F03_f.transpose(-1, -2), F13_f.transpose(-1, -2), F23_f.transpose(-1, -2), F33_f], dim=-1)
            H4 = torch.stack([row0, row1, row2, row3], dim=-2)         # (B, N, N, 4, 4)
            H_full = 2 * H4.permute(0, 1, 3, 2, 4).reshape(B, 4 * N, 4 * N)

            v0 = -2 * (x.unsqueeze(1) * yz).sum(dim=(2, 3))            # (B, N)
            v1 = -2 * (W_row_sum.unsqueeze(1) * sigma_y).sum(dim=2)
            v2 = -2 * (W_col_sum.unsqueeze(1) * sigma_z).sum(dim=2)
            v3 = (-2 * W_total).unsqueeze(1).expand(B, N)
            v_full = torch.stack([v0, v1, v2, v3], dim=2).reshape(B, 4 * N)

            scale = H_full.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).view(B, 1, 1)
            H_reg = H_full + scale * damping_eye
            A_flat = -torch.linalg.solve(H_reg, v_full.unsqueeze(-1)).squeeze(-1)
            return A_flat.view(B, N, 4)

        a = compute_A(y, z)

        def _loop_fn(y, z, yb, zb, a, temp):
            with torch.no_grad():
                a0 = a[:, :, 0].view(B, N, 1, 1)
                a1 = a[:, :, 1].view(B, N, 1, 1)
                a2 = a[:, :, 2].view(B, N, 1, 1)
                a3 = a[:, :, 3].view(B, N, 1, 1)

                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)

                yz = torch.einsum('bnik,bnkj->bnij', yf, zf)
                sigma_yf = yf.sum(dim=3, keepdim=True)   # (B, N, n, 1)
                sigma_zf = zf.sum(dim=2, keepdim=True)   # (B, N, 1, m)

                M_per = a0 * yz + a1 * sigma_yf + a2 * sigma_zf + a3   # (B, N, n, m)
                total_M = M_per.sum(dim=1)                              # (B, n, m)
                part = (x - total_M).unsqueeze(1).expand(B, N, n, m)    # (B, N, n, m)

                # Y energy gradient
                kernel_y = (a0 * zf + a1).transpose(-1, -2)             # (B, N, m, rank)
                linear_y = -2 * torch.einsum('bnij,bnjk->bnik', part, kernel_y)

                zf_sum1 = zf.sum(dim=3).unsqueeze(2)                    # (B, N, 1, rank)
                zf2_sum1 = (zf*zf).sum(dim=3).unsqueeze(2)

                y_energy_grad = (
                    linear_y
                    + (a0*a0 + 2*a0*a1*(1 - 2*yf) + 2*a0*a2) * zf_sum1
                    - 2 * (a0*a2 + a0*a0 * yf) * zf2_sum1
                    + (a1*a1) * (1 - 2 * yf) * m
                )

                # Z energy gradient
                kernel_z = (a0 * yf + a2).transpose(-1, -2)             # (B, N, rank, n)
                linear_z = -2 * torch.einsum('bnij,bnjk->bnik', kernel_z, part)

                yf_sum0 = yf.sum(dim=2).unsqueeze(3)                    # (B, N, rank, 1)
                yf2_sum0 = (yf*yf).sum(dim=2).unsqueeze(3)

                z_energy_grad = (
                    linear_z
                    + (a0*a0 + 2*a0*a1 + 2*a0*a2*(1 - 2*zf)) * yf_sum0
                    - 2 * (a0*a0 * zf + a0*a1) * yf2_sum0
                    + (a2*a2) * (1 - 2 * zf) * n
                )

                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(
                    torch.where((y < 0.0) | (y > 1.0),
                                2*y - yb - eta * y_entropy_grad,
                                2*y - yb - eta * (y_energy_grad + y_entropy_grad)),
                    0, 1)
                za = torch.clamp(
                    torch.where((z < 0.0) | (z > 1.0),
                                2*z - zb - eta * z_entropy_grad,
                                2*z - zb - eta * (z_energy_grad + z_entropy_grad)),
                    0, 1)

                a_new = compute_A(ya, za)
            return ya, za, y, z, a_new

        loop_body = torch.compile(_loop_fn, mode=compile_mode)

        temp = torch.tensor(Tinit, device=device)
        delta_temp_t = torch.tensor(delta_temp, device=device)

        for _ in range(Nstep):
            y = y.detach().clone()
            yb = yb.detach().clone()
            z = z.detach().clone()
            zb = zb.detach().clone()
            a = a.detach().clone()
            y, z, yb, zb, a = loop_body(y, z, yb, zb, a, temp)
            temp = temp - delta_temp_t

        y = torch.where(y > 0.5, 1.0, 0.0)
        z = torch.where(z > 0.5, 1.0, 0.0)
        a = compute_A(y, z)

        # maximum: (B, 1, 1); a: (B, N, 4). Broadcast to apply per-matrix scaling.
        return y, z, maximum * a


    def patchify(self, tensor, max_patch_size):
        """
        テンソルをパッチに分割する関数

        Args:
            tensor: 分割するテンソル (バッチxチャンネルx高さx幅)
            max_patch_size: パッチの最大サイズ (int)

        Returns:
            パッチに分割されたテンソル
        """

        height, width = tensor.shape

        # 各次元の最大約数を計算 (256以下)
        
        def get_max_divisor(num, max_value):
            """
            与えられた数値の最大約数を、指定された最大値以下で求める関数

            Args:
                num: 約数を求める数値
                max_value: 最大約数の最大値

            Returns:
                最大約数
            """

            # 最大約数の探索範囲を決定
            limit = max(int(math.sqrt(num)), max_value)

            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        patch_height = get_max_divisor(height, max_patch_size)
        patch_width = get_max_divisor(width, max_patch_size)
        print('Patch Size:({0}x{1})'.format(patch_height, patch_width))

        # unfold関数でパッチに分割 (パッチサイズはpatch_height x patch_width, patch_height x patch_width間隔で切り出す)
        return tensor.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)

    def patchify_3d(self, tensor, max_patch_size=256):
        """
        テンソルをパッチに分割する関数

        Args:
            tensor: 分割するテンソル (バッチxチャンネルx高さx幅)
            max_patch_size: パッチの最大サイズ (int)

        Returns:
            パッチに分割されたテンソル
        """

        channels, height, width = tensor.shape

        # 各次元の最大約数を計算 (256以下)
        
        def get_max_divisor(num, max_value):
            """
            与えられた数値の最大約数を、指定された最大値以下で求める関数

            Args:
                num: 約数を求める数値
                max_value: 最大約数の最大値

            Returns:
                最大約数
            """

            # 最大約数の探索範囲を決定
            limit = max(int(math.sqrt(num)), max_value)

            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        patch_height = get_max_divisor(height, max_patch_size)
        patch_width = get_max_divisor(width, max_patch_size)
        print('Patch Size:({0}x{1})'.format(patch_height, patch_width))

        # unfold関数でパッチに分割 (パッチサイズはpatch_height x patch_width, patch_height x patch_width間隔で切り出す)
        return tensor.unfold(1, patch_height, patch_height).unfold(2, patch_width, patch_width)
    
    def unpatchify(self, patches, original_shape):
        """
        パッチから元のテンソルを復元する関数

        Args:
            patches: パッチに分割されたテンソル (バッチxパッチ高さxパッチ幅xパッチサイズxパッチサイズ)
            original_shape: 元のテンソルの形状 (チャンネルx高さx幅)

        Returns:
            復元されたテンソル
        """
        num_patches_h, num_patches_w, patch_height, patch_width = patches.shape
        height, width = original_shape

        # 元の形状と整合性の確認
        assert height == num_patches_h * patch_height, "Height mismatch between patches and original shape"
        assert width == num_patches_w * patch_width, "Width mismatch between patches and original shape"

        # パッチを再構成
        reconstructed_tensor = torch.zeros((height, width), device=patches.device)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                reconstructed_tensor[
                    i * patch_height: (i + 1) * patch_height,
                    j * patch_width: (j + 1) * patch_width
                ] = patches[i, j, :, :]

        return reconstructed_tensor
    
    def unpatchify_3d(self, patches, original_shape):
        """
        パッチから元のテンソルを復元する関数

        Args:
            patches: パッチに分割されたテンソル (バッチxパッチ高さxパッチ幅xパッチサイズxパッチサイズ)
            original_shape: 元のテンソルの形状 (チャンネルx高さx幅)

        Returns:
            復元されたテンソル
        """
        batch_size, num_patches_h, num_patches_w, patch_height, patch_width = patches.shape
        channels, height, width = original_shape

        # 元の形状と整合性の確認
        assert height == num_patches_h * patch_height, "Height mismatch between patches and original shape"
        assert width == num_patches_w * patch_width, "Width mismatch between patches and original shape"

        # パッチを再構成
        reconstructed_tensor = torch.zeros((channels, height, width), device=patches.device)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                reconstructed_tensor[
                    :, 
                    i * patch_height: (i + 1) * patch_height,
                    j * patch_width: (j + 1) * patch_width
                ] = patches[:, i, j, :, :]

        return reconstructed_tensor

    
    
    def get_least_used_gpu(self, num_gpus):
        # NVML 初期化
        pynvml.nvmlInit()

        min_usage = float('inf')
        best_device = 0

        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # メモリ使用率 (使用量 / 合計)
            usage = mem_info.used / mem_info.total
            # print(f"GPU {i}: {usage * 100:.2f}% 使用中")
            
            # 最も使用率の低いGPUを選択
            if usage < min_usage:
                min_usage = usage
                best_device = i

        # NVML 終了
        pynvml.nvmlShutdown()

        return best_device
    
    def bqq_worker_task(self, queue, result_list, rank_scale_copy, seed, zeta, eta, Tinit, Tfin, Nstep, device_id, bit_width):
        """各ワーカーが処理する行列分解タスク。結果はインメモリで返す。"""
        torch.manual_seed(seed)
        torch.cuda.set_device(device_id)
        while not queue.empty():
            try:
                data = queue.get_nowait()
            except:
                break

            i, j, patch = data['i'], data['j'], data['patch']

            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            patch = patch.to(device)
            original_x = patch.detach().clone()
            update_x = patch.detach().clone()

            decomposed_patches = []

            for bit_idx in range(bit_width):
                decomp_instance = BinaryQuadraticQuantization(x=update_x.clone(), rank_scale=rank_scale_copy)
                y, z, a = decomp_instance.run_bqq_compile(zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type='torch')
                reconst = a[0] * y@z + a[1] * y.sum(axis=1).unsqueeze(1) + a[2] * z.sum(axis=0).unsqueeze(0) + a[3]
                update_x -= reconst

                decomposed_patches.append({
                    'patch_row': i, 'patch_col': j,
                    'coeff': a.cpu(), 'mat1': y.cpu(), 'mat2': z.cpu(),
                    'bit_idx': bit_idx,
                })

            result_list.append({
                'i': i, 'j': j,
                'reconstructed': (original_x - update_x).clone().detach().cpu(),
                'decomposed': decomposed_patches,
            })
            queue.task_done()


    def bqq_large_matrix_multi_worker(self, max_patch_size, bit_width, consolidated_path=None, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=10000, seed=1, workers_per_gpu=8, main_gpu_id=0, use_batch=True, H=None, damping=1e-6, hessian_mode='inter', scale_refine=False, use_multibqq=True, ste_refine_steps=0, ste_refine_lr=1e-3, ste_refine_binary_lr=None, ste_refine_continuous_lr=None, ste_refine_weight_decay=0.0, ste_refine_optimize_factors=True, ste_refine_optimize_coeffs=True, ste_refine_optimize_theta=True, ste_refine_optimize_beta=True, ste_refine_row_group_batch_size=None, ste_refine_log_interval=20):
        """
        大きな行列をパッチに分割し、行列分解を実行して復元。

        Args:
            use_batch: Trueの場合バッチ処理(デフォルト)。Falseならマルチプロセス。
            H: 入力相関行列 X^T X (in_features, in_features)。
            hessian_mode: 'inter' = inter-bit scale refinement,
                          'intra' = intra-bit column compensation (last bit only),
                          'intra-layer' = column-wise N-bit BQQ + compensation,
                          'intra-layer-ste' = intra-layer後にSTE refinementまで実行。
            scale_refine: Trueの場合、最後にinter-bit hessian-aware scale refinementを実行。
            damping: Hessian-awareのTikhonov正則化。
        """
        if H is not None and hessian_mode == 'intra-layer-ste':
            initial_weight = self._intra_layer_hessian_aware_large_matrix_batched(
                max_patch_size, bit_width, H, consolidated_path,
                zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id, damping, scale_refine, use_multibqq)
            if ste_refine_steps > 0:
                refined_weight, _, _ = self.refine_decomposition_with_ste(
                    all_decomposed=consolidated_path,
                    H=H,
                    num_steps=ste_refine_steps,
                    lr=ste_refine_lr,
                    weight_decay=ste_refine_weight_decay,
                    device_id=main_gpu_id,
                    optimize_factors=ste_refine_optimize_factors,
                    factors_lr=ste_refine_binary_lr,
                    continuous_lr=ste_refine_continuous_lr,
                    optimize_coeffs=ste_refine_optimize_coeffs,
                    optimize_theta=ste_refine_optimize_theta,
                    optimize_beta=ste_refine_optimize_beta,
                    row_group_batch_size=ste_refine_row_group_batch_size,
                    consolidated_path=consolidated_path,
                    log_interval=ste_refine_log_interval,
                )
                return refined_weight
            return initial_weight
        elif H is not None and hessian_mode == 'intra-layer':
            return self._intra_layer_hessian_aware_large_matrix_batched(
                max_patch_size, bit_width, H, consolidated_path,
                zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id, damping, scale_refine, use_multibqq)
        elif use_batch:
            return self._large_matrix_batched(
                max_patch_size, bit_width, consolidated_path,
                zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id)
        else:
            return self._large_matrix_multiprocess(
                max_patch_size, bit_width, consolidated_path,
                zeta, eta, Tinit, Tfin, Nstep, seed, workers_per_gpu, main_gpu_id)

    def _large_matrix_multiprocess(self, max_patch_size, bit_width, consolidated_path, zeta, eta, Tinit, Tfin, Nstep, seed, workers_per_gpu, main_gpu_id):
        """マルチプロセスワーカー版"""
        mp.set_start_method("spawn", force=True)
        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.deepcopy(self.x)
        divided_tensor = self.patchify(torch.tensor(x_copy), max_patch_size=max_patch_size)
        num_patches_h, num_patches_w, _, _ = divided_tensor.shape

        all_decomposed = []
        completed = set()
        if consolidated_path and os.path.exists(consolidated_path):
            all_decomposed = torch.load(consolidated_path, weights_only=False, map_location='cpu')
            for p in all_decomposed:
                completed.add((p['patch_row'], p['patch_col']))
            from collections import defaultdict
            by_patch = defaultdict(list)
            for p in all_decomposed:
                by_patch[(p['patch_row'], p['patch_col'])].append(p)
            for (i, j), patches in by_patch.items():
                reconstructed = torch.zeros_like(divided_tensor[i, j], dtype=torch.float32)
                for p in patches:
                    a, y, z = p['coeff'], p['mat1'], p['mat2']
                    reconstructed += a[0] * y @ z + a[1] * y.sum(axis=1).unsqueeze(1) + a[2] * z.sum(axis=0).unsqueeze(0) + a[3]
                divided_tensor[i, j, :, :] = reconstructed
            print(f'Resumed {len(completed)}/{num_patches_h * num_patches_w} patches from {consolidated_path}')

        manager = mp.Manager()
        queue = manager.Queue()
        result_list = manager.list()

        pending = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if (i, j) in completed:
                    continue
                patch = divided_tensor[i, j, :, :]
                queue.put({'i': i, 'j': j, 'patch': patch})
                pending += 1

        if pending > 0:
            num_gpus = torch.cuda.device_count()
            num_workers = min(mp.cpu_count(), num_gpus * workers_per_gpu)
            print(f'Dispatching {pending} patches to {num_workers} workers')
            processes = []
            for worker_id in range(num_workers):
                device_id = (worker_id + main_gpu_id) % num_gpus
                p = mp.Process(target=self.bqq_worker_task, args=(queue, result_list, rank_scale_copy, seed, zeta, eta, Tinit, Tfin, Nstep, device_id, bit_width))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            for data in result_list:
                i, j = data['i'], data['j']
                divided_tensor[i, j, :, :] = data['reconstructed']
                all_decomposed.extend(data['decomposed'])

        if consolidated_path and all_decomposed:
            os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
            torch.save(all_decomposed, consolidated_path)
            print(f'Saved consolidated: {consolidated_path} ({len(all_decomposed)} entries)')

        reconstructed_tensor = self.unpatchify(divided_tensor, x_copy.shape)
        self.x = copy.copy(x_copy)
        return reconstructed_tensor

    def _large_matrix_batched(self, max_patch_size, bit_width, consolidated_path, zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id):
        """
        バッチ処理版。
        - 約数が max_patch_size の半分以上 → 均等分割 (patchifyと同じ)
        - 約数が小さすぎる場合 → max_patch_size で分割 + 余りパッチ
          サイズ別にグループ化してバッチ処理する。
        """
        from collections import defaultdict
        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.deepcopy(self.x)
        original_h, original_w = x_copy.shape

        def get_max_divisor(num, max_value):
            limit = max(int(math.sqrt(num)), max_value)
            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        def compute_patch_ranges(dim_size, max_ps):
            """約数が十分大きければ均等分割、小さければ max_ps + 余りで分割。
            余りが max_ps/2 未満の場合は最後のパッチに統合してアスペクト比を保つ。"""
            divisor = get_max_divisor(dim_size, max_ps)
            if divisor >= max_ps // 2:
                # 均等分割
                n = dim_size // divisor
                return [(i * divisor, (i + 1) * divisor) for i in range(n)]
            else:
                # max_ps チャンク + 余り
                n_full = dim_size // max_ps
                rem = dim_size - n_full * max_ps
                # 余りが小さすぎる場合、最後のフルパッチと統合
                if 0 < rem < max_ps // 2 and n_full > 0:
                    n_full -= 1
                ranges = [(i * max_ps, (i + 1) * max_ps) for i in range(n_full)]
                if n_full * max_ps < dim_size:
                    ranges.append((n_full * max_ps, dim_size))
                return ranges

        h_ranges = compute_patch_ranges(original_h, max_patch_size)
        w_ranges = compute_patch_ranges(original_w, max_patch_size)

        # パッチ仕様の構築 (各パッチの位置とサイズ)
        patch_specs = []
        for i, (r0, r1) in enumerate(h_ranges):
            for j, (c0, c1) in enumerate(w_ranges):
                patch_specs.append({'i': i, 'j': j, 'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1})
        total_patches = len(patch_specs)

        # パッチサイズ情報の表示
        size_counts = defaultdict(int)
        for s in patch_specs:
            size_counts[(s['r1'] - s['r0'], s['c1'] - s['c0'])] += 1
        for (ph, pw), cnt in sorted(size_counts.items(), key=lambda x: -x[1]):
            print(f'Patch Size:({ph}x{pw}), Count: {cnt}')

        x_tensor = torch.tensor(x_copy).float()

        # 復元蓄積 (パッチごとに管理)
        reconst_accum = {}
        for s in patch_specs:
            reconst_accum[(s['i'], s['j'])] = torch.zeros(s['r1'] - s['r0'], s['c1'] - s['c0'])

        # 既存の consolidated ファイルから完了済みパッチを復元
        all_decomposed = []
        completed_bits = {}
        if consolidated_path and os.path.exists(consolidated_path):
            all_decomposed = torch.load(consolidated_path, weights_only=False, map_location='cpu')
            by_patch = defaultdict(list)
            for p in all_decomposed:
                by_patch[(p['patch_row'], p['patch_col'])].append(p)
            for (i, j), patches in by_patch.items():
                if (i, j) not in reconst_accum:
                    continue
                reconstructed = torch.zeros_like(reconst_accum[(i, j)])
                for p in patches:
                    a, y, z = p['coeff'], p['mat1'], p['mat2']
                    reconstructed += a[0] * y @ z + a[1] * y.sum(axis=1).unsqueeze(1) + a[2] * z.sum(axis=0).unsqueeze(0) + a[3]
                reconst_accum[(i, j)] = reconstructed
                completed_bits[(i, j)] = len(patches)
            print(f'Resumed {len(completed_bits)}/{total_patches} patches from {consolidated_path}')

        for bit_idx in range(bit_width):
            # 未完了パッチをサイズ別にグループ化
            size_groups = defaultdict(list)
            for s in patch_specs:
                key = (s['i'], s['j'])
                if completed_bits.get(key, 0) > bit_idx:
                    continue
                ph, pw = s['r1'] - s['r0'], s['c1'] - s['c0']
                original = x_tensor[s['r0']:s['r1'], s['c0']:s['c1']]
                residual = original - reconst_accum[key]
                size_groups[(ph, pw)].append((s, residual))

            # サイズグループごとにバッチ処理
            for (ph, pw), group in size_groups.items():
                specs = [g[0] for g in group]
                residuals = [g[1] for g in group]
                x_batch = torch.stack(residuals)
                print(f'Bit {bit_idx}: processing {len(residuals)} patches of ({ph}x{pw})')

                y_b, z_b, a_b = self.run_bqq_compile_batched(
                    x_batch, rank_scale=rank_scale_copy,
                    zeta=zeta, eta=eta, Tinit=Tinit, Tfin=Tfin,
                    Nstep=Nstep, device_id=main_gpu_id, seed=seed
                )

                a0 = a_b[:, 0].view(-1, 1, 1)
                a1 = a_b[:, 1].view(-1, 1, 1)
                a2 = a_b[:, 2].view(-1, 1, 1)
                a3 = a_b[:, 3].view(-1, 1, 1)
                reconst_batch = (a0 * torch.bmm(y_b, z_b) + a1 * y_b.sum(dim=2, keepdim=True) + a2 * z_b.sum(dim=1, keepdim=True) + a3).cpu()

                for b, s in enumerate(specs):
                    key = (s['i'], s['j'])
                    reconst_accum[key] += reconst_batch[b]
                    all_decomposed.append({
                        'patch_row': s['i'], 'patch_col': s['j'],
                        'row_start': s['r0'], 'row_end': s['r1'],
                        'col_start': s['c0'], 'col_end': s['c1'],
                        'coeff': a_b[b].cpu(), 'mat1': y_b[b].cpu(), 'mat2': z_b[b].cpu(),
                        'bit_idx': bit_idx,
                    })

        # consolidated ファイルに一括保存
        if consolidated_path and all_decomposed:
            os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
            torch.save(all_decomposed, consolidated_path)
            print(f'Saved consolidated: {consolidated_path} ({len(all_decomposed)} entries)')

        # 全パッチを元の行列に配置
        reconstructed_tensor = torch.zeros(original_h, original_w)
        for s in patch_specs:
            reconstructed_tensor[s['r0']:s['r1'], s['c0']:s['c1']] = reconst_accum[(s['i'], s['j'])]

        self.x = copy.copy(x_copy)
        return reconstructed_tensor

    # ------------------------------------------------------------------
    # Hessian-aware scale refinement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cholesky_safe(C, damping=1e-6):
        """Cholesky with adaptive damping."""
        mean_diag = C.diagonal().mean().item()
        for scale in [damping, 1e-4, 1e-2, 1e-1]:
            try:
                return torch.linalg.cholesky(
                    C + scale * mean_diag * torch.eye(C.shape[0], device=C.device, dtype=C.dtype))
            except torch.linalg.LinAlgError:
                continue
        return None


    @staticmethod
    def _resolve_hessian_matrix(H, in_features):
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H)
        H = H.float()
        if H.ndim != 2:
            raise ValueError(f'H must be 2D, got shape {tuple(H.shape)}')
        if H.shape != (in_features, in_features):
            raise ValueError(
                f'H shape {tuple(H.shape)} is incompatible with in_features={in_features}'
            )
        return H

    def refine_decomposition_with_ste(
        self,
        all_decomposed,
        H,
        num_steps=1000,
        lr=1e-3,
        weight_decay=0.0,
        device_id=0,
        optimize_factors=True,
        optimize_coeffs=True,
        optimize_theta=True,
        optimize_beta=True,
        factors_lr=None,
        continuous_lr=None,
        row_group_batch_size=None,
        consolidated_path=None,
        log_interval=20,
    ):
        """
        Refine a saved BQQ decomposition by minimizing
        tr((W - W') H (W - W')^T) with AdamW and a straight-through estimator
        over binary Y/Z factors, where H = X X^T.

        Args:
            all_decomposed: list of patch dicts or path to a torch-saved list.
            H: Hessian / activation covariance matrix with shape
               (in_features, in_features).
        Returns:
            refined_weight, refined_decomposition, history
        """
        if isinstance(all_decomposed, (str, os.PathLike)):
            all_decomposed = torch.load(all_decomposed, map_location='cpu')
        all_decomposed = copy.deepcopy(all_decomposed)

        if len(all_decomposed) == 0:
            raise ValueError('all_decomposed must not be empty')

        W_target = torch.as_tensor(copy.deepcopy(self.x)).float()
        if W_target.ndim != 2:
            raise ValueError(f'self.x must be 2D, got shape {tuple(W_target.shape)}')

        H_mat = self._resolve_hessian_matrix(H, W_target.shape[1])

        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        module = _BQQSTERefinementModule(
            all_decomposed,
            weight_shape=W_target.shape,
            optimize_factors=optimize_factors,
            optimize_coeffs=optimize_coeffs,
            optimize_theta=optimize_theta,
            optimize_beta=optimize_beta,
        ).to(device)
        W_target = W_target.to(device)
        H_mat = H_mat.to(device)

        named_params = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
        if len(named_params) == 0:
            raise ValueError('No trainable parameters selected for refinement')

        binary_lr = lr if factors_lr is None else factors_lr
        continuous_lr_value = lr if continuous_lr is None else continuous_lr
        binary_params = [p for name, p in named_params if name.startswith('y_fp_') or name.startswith('z_fp_')]
        continuous_params = [p for name, p in named_params if not (name.startswith('y_fp_') or name.startswith('z_fp_'))]
        param_groups = []
        if continuous_params:
            param_groups.append({'params': continuous_params, 'lr': continuous_lr_value, 'weight_decay': weight_decay})
        if binary_params:
            param_groups.append({'params': binary_params, 'lr': binary_lr, 'weight_decay': weight_decay})

        optimizer = torch.optim.AdamW(param_groups)
        history = []
        best_loss = float('inf')
        best_step = -1
        best_state_dict = None

        grouped_patch_rows = []
        for patch_height in sorted(module.patch_rows_by_height):
            patch_rows = module.patch_rows_by_height[patch_height]
            batch_size = len(patch_rows) if not row_group_batch_size or row_group_batch_size <= 0 else row_group_batch_size
            for start in range(0, len(patch_rows), batch_size):
                grouped_patch_rows.append(patch_rows[start:start + batch_size])

        for step in range(num_steps):
            for patch_rows in grouped_patch_rows:
                optimizer.zero_grad(set_to_none=True)
                Wq_batch, row_ranges = module.reconstruct_row_group_batch(patch_rows)
                target_batch = torch.stack([W_target[r0:r1, :] for r0, r1 in row_ranges], dim=0)
                diff_batch = target_batch - Wq_batch
                loss = torch.sum((diff_batch @ H_mat) * diff_batch) / diff_batch.numel()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                Wq_eval = module.reconstruct_weight()
                diff_eval = W_target - Wq_eval
                loss_value = (torch.sum((diff_eval @ H_mat) * diff_eval) / diff_eval.numel()).detach().cpu().item()
            history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step + 1
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }

            if log_interval and (step == 0 or (step + 1) % log_interval == 0 or step + 1 == num_steps):
                print(f'STE refine step {step + 1}/{num_steps}: loss={loss_value:.6e}, best={best_loss:.6e}')

        if best_state_dict is not None:
            module.load_state_dict({
                key: value.to(device)
                for key, value in best_state_dict.items()
            })
            print(f'STE refine restored best step {best_step}/{num_steps}: best_loss={best_loss:.6e}')

        refined_weight = module.reconstruct_weight().detach().cpu()
        refined_decomposition = module.export_decomposition()

        if consolidated_path:
            os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
            torch.save(refined_decomposition, consolidated_path)
            print(f'Saved refined consolidated: {consolidated_path} ({len(refined_decomposition)} entries)')

        return refined_weight, refined_decomposition, history

    def optimize_decomposition_from_scratch_with_ste(
        self,
        max_patch_size,
        bit_width,
        H,
        num_steps=200,
        lr=1e-3,
        weight_decay=0.0,
        device_id=0,
        optimize_factors=True,
        optimize_coeffs=True,
        optimize_theta=True,
        optimize_beta=True,
        factors_lr=None,
        continuous_lr=None,
        row_group_batch_size=None,
        consolidated_path=None,
        log_interval=20,
        seed=0,
    ):
        """
        Optimize a BQQ decomposition from scratch with the same STE objective
        used in refine_decomposition_with_ste.
        """
        W_target = torch.as_tensor(copy.deepcopy(self.x)).float()
        if W_target.ndim != 2:
            raise ValueError(f'self.x must be 2D, got shape {tuple(W_target.shape)}')

        def get_max_divisor(num, max_value):
            limit = max(int(math.sqrt(num)), max_value)
            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        def compute_patch_ranges(dim_size, max_ps):
            divisor = get_max_divisor(dim_size, max_ps)
            if divisor >= max_ps // 2:
                n = dim_size // divisor
                return [(i * divisor, (i + 1) * divisor) for i in range(n)]
            n_full = dim_size // max_ps
            rem = dim_size - n_full * max_ps
            if 0 < rem < max_ps // 2 and n_full > 0:
                n_full -= 1
            ranges = [(i * max_ps, (i + 1) * max_ps) for i in range(n_full)]
            if n_full * max_ps < dim_size:
                ranges.append((n_full * max_ps, dim_size))
            return ranges

        h_ranges = compute_patch_ranges(W_target.shape[0], max_patch_size)
        w_ranges = compute_patch_ranges(W_target.shape[1], max_patch_size)

        generator = torch.Generator().manual_seed(seed)
        all_decomposed = []
        for patch_row, (r0, r1) in enumerate(h_ranges):
            ph = r1 - r0
            for patch_col, (c0, c1) in enumerate(w_ranges):
                pw = c1 - c0
                patch = W_target[r0:r1, c0:c1]
                patch_mean = patch.mean().item()
                patch_scale = patch.std().item()
                rank = max(1, int(round(self.rank_scale * ph * pw / max(ph + pw, 1))))
                coeff_base = torch.tensor([
                    patch_scale / max(bit_width, 1),
                    0.0,
                    0.0,
                    patch_mean / max(bit_width, 1),
                ], dtype=torch.float32)
                for bit_idx in range(bit_width):
                    all_decomposed.append({
                        'patch_row': patch_row,
                        'patch_col': patch_col,
                        'row_start': r0,
                        'row_end': r1,
                        'col_start': c0,
                        'col_end': c1,
                        'coeff': coeff_base.clone(),
                        'mat1': torch.rand(ph, rank, generator=generator),
                        'mat2': torch.rand(rank, pw, generator=generator),
                        'bit_idx': bit_idx,
                    })

        return self.refine_decomposition_with_ste(
            all_decomposed=all_decomposed,
            H=H,
            num_steps=num_steps,
            lr=lr,
            weight_decay=weight_decay,
            device_id=device_id,
            optimize_factors=optimize_factors,
            optimize_coeffs=optimize_coeffs,
            optimize_theta=optimize_theta,
            optimize_beta=optimize_beta,
            factors_lr=factors_lr,
            continuous_lr=continuous_lr,
            row_group_batch_size=row_group_batch_size,
            consolidated_path=consolidated_path,
            log_interval=log_interval,
        )




    def _intra_layer_hessian_aware_large_matrix_batched(
        self, max_patch_size, bit_width, H,
        consolidated_path, zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id,
        damping=1e-6, scale_refine=True, use_multibqq=True,
    ):
        """
        Column-wise Hessian-aware BQQ: process column groups sequentially,
        each with full N-bit BQQ decomposition followed by GPTQ-style compensation.
        If scale_refine=True, finishes with inter-bit Hessian-aware scale refinement.

        For each column group j (left to right):
          1. N-bit BQQ decompose W_work[:, c0:c1] (all bits, all row groups)
          2. Compute error E_j = W_work[:, c0:c1] - Wq[:, c0:c1]
          3. Compensate remaining columns: W_work[:, c1:] += E_j @ H_12 @ H_22^{-1}

        Unlike _intra_bit which processes bit-by-bit then column-by-column,
        this processes column-by-column with all bits at once.

        If use_multibqq=True, each column group is optimized jointly with
        run_multibqq_compile_batched(..., num_stack=bit_width) instead of
        applying run_bqq_compile_batched sequentially per bit.
        """
        from collections import defaultdict
        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.deepcopy(self.x)
        original_h, original_w = x_copy.shape
        dtype = torch.float32

        def get_max_divisor(num, max_value):
            limit = max(int(math.sqrt(num)), max_value)
            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        def compute_patch_ranges(dim_size, max_ps):
            divisor = get_max_divisor(dim_size, max_ps)
            if divisor >= max_ps // 2:
                n = dim_size // divisor
                return [(i * divisor, (i + 1) * divisor) for i in range(n)]
            else:
                n_full = dim_size // max_ps
                rem = dim_size - n_full * max_ps
                if 0 < rem < max_ps // 2 and n_full > 0:
                    n_full -= 1
                ranges = [(i * max_ps, (i + 1) * max_ps) for i in range(n_full)]
                if n_full * max_ps < dim_size:
                    ranges.append((n_full * max_ps, dim_size))
                return ranges

        h_ranges = compute_patch_ranges(original_h, max_patch_size)
        w_ranges = compute_patch_ranges(original_w, max_patch_size)

        x_tensor = torch.tensor(x_copy).float()
        H = H.to(dtype=torch.float32, device=x_tensor.device)
        num_col_groups = len(w_ranges)

        # GPTQ-style: precompute damped H_inv once to guarantee finite compensation.
        # Per-group solve of H[c1:,c1:]^{-1} is numerically unstable for ill-conditioned H.
        COMP_DAMPING = 0.01  # 1% relative damping, same default as GPTQ paper
        _L = self._cholesky_safe(H, COMP_DAMPING)
        if _L is not None:
            H_inv = torch.cholesky_inverse(_L)
            print(f'Precomputed H_inv for compensation ({list(H_inv.shape)}, damping={COMP_DAMPING})')
        else:
            H_inv = None
            print('WARNING: Cholesky failed, compensation disabled')

        all_decomposed = []
        Wq = torch.zeros(original_h, original_w, dtype=dtype)
        W_work = x_tensor.clone()

        for j, (c0, c1) in enumerate(w_ranges):
            pw = c1 - c0

            # N-bit BQQ on this column group (all bits, batched across row groups)
            if use_multibqq:
                col_patches = []
                for i, (r0, r1) in enumerate(h_ranges):
                    col_patches.append(W_work[r0:r1, c0:c1].clone())

                x_batch = torch.stack(col_patches)
                print(f'Col {j}/{num_col_groups}: jointly decomposing {len(col_patches)} '
                      f'patches of ({col_patches[0].shape[0]}x{pw}) with {bit_width} stacks')

                y_mb, z_mb, a_mb = self.run_multibqq_compile_batched(
                    x_batch, num_stack=bit_width, rank_scale=rank_scale_copy,
                    zeta=zeta, eta=eta, Tinit=Tinit, Tfin=Tfin,
                    Nstep=Nstep, device_id=main_gpu_id, seed=seed
                )

                for b_idx, (r0, r1) in enumerate(h_ranges):
                    for bit_idx in range(bit_width):
                        yb = y_mb[b_idx, bit_idx].cpu()
                        zb = z_mb[b_idx, bit_idx].cpu()
                        coeff = a_mb[b_idx, bit_idx].cpu()
                        bit_reconst = (coeff[0] * yb @ zb
                                      + coeff[1] * yb.sum(dim=1, keepdim=True)
                                      + coeff[2] * zb.sum(dim=0, keepdim=True)
                                      + coeff[3])
                        Wq[r0:r1, c0:c1] += bit_reconst

                        all_decomposed.append({
                            'patch_row': b_idx, 'patch_col': j,
                            'row_start': r0, 'row_end': r1,
                            'col_start': c0, 'col_end': c1,
                            'coeff': coeff, 'mat1': yb, 'mat2': zb,
                            'bit_idx': bit_idx,
                        })
            else:
                # Residual decomposition: bit by bit on W_work[:, c0:c1]
                col_residual = W_work[:, c0:c1].clone()

                for bit_idx in range(bit_width):
                    col_patches = []
                    for i, (r0, r1) in enumerate(h_ranges):
                        col_patches.append(col_residual[r0:r1, :].clone())

                    x_batch = torch.stack(col_patches)
                    print(f'Col {j}/{num_col_groups}, bit {bit_idx}: '
                          f'decomposing {len(col_patches)} patches of ({col_patches[0].shape[0]}x{pw})')

                    y_b, z_b, a_b = self.run_bqq_compile_batched(
                        x_batch, rank_scale=rank_scale_copy,
                        zeta=zeta, eta=eta, Tinit=Tinit, Tfin=Tfin,
                        Nstep=Nstep, device_id=main_gpu_id, seed=seed
                    )

                    for b_idx, (r0, r1) in enumerate(h_ranges):
                        yb, zb, coeff = y_b[b_idx].cpu(), z_b[b_idx].cpu(), a_b[b_idx].cpu()
                        bit_reconst = (coeff[0] * yb @ zb
                                      + coeff[1] * yb.sum(dim=1, keepdim=True)
                                      + coeff[2] * zb.sum(dim=0, keepdim=True)
                                      + coeff[3])
                        Wq[r0:r1, c0:c1] += bit_reconst
                        col_residual[r0:r1, :] -= bit_reconst

                        all_decomposed.append({
                            'patch_row': b_idx, 'patch_col': j,
                            'row_start': r0, 'row_end': r1,
                            'col_start': c0, 'col_end': c1,
                            'coeff': coeff, 'mat1': yb, 'mat2': zb,
                            'bit_idx': bit_idx,
                        })

            # Compensation: GPTQ-style update using precomputed H_inv.
            # Correct formula (per GPTQ paper): err = E_j / diag(H_inv[c0:c1,c0:c1]),
            # then W_work[:, c1:] -= err @ H_inv[c0:c1, c1:]
            if H_inv is not None and c1 < original_w:
                E_j = W_work[:, c0:c1] - Wq[:, c0:c1]
                d = H_inv[c0:c1, c0:c1].diagonal().clamp(min=1e-8)
                E_normalized = E_j / d[None, :]
                update = E_normalized @ H_inv[c0:c1, c1:]
                if torch.isfinite(update).all():
                    W_work[:, c1:] -= update
                    print(f'  Compensated remaining {original_w - c1} columns')
                else:
                    print(f'  WARNING: compensation update non-finite, skipping')

        # --- Optional: inter-bit Hessian-aware scale refinement ---
        if scale_refine:
            from collections import defaultdict as _dd
            print(f'Scale refine: {len(h_ranges)} row-groups, {bit_width} bits')

            S = self._cholesky_safe(H, damping)
            if S is None:
                print('  WARNING: Cholesky failed, skipping scale refine')
            else:
                S_j_list = [S[c0:c1, :] for c0, c1 in w_ranges]
                col_sum_S_list = [sj.sum(dim=0, keepdim=True) for sj in S_j_list]

                binary_by_patch = _dd(list)
                entries_by_patch = _dd(list)
                for p in sorted(all_decomposed, key=lambda pp: (pp['patch_row'], pp['patch_col'], pp['bit_idx'])):
                    key = (p['patch_row'], p['patch_col'])
                    binary_by_patch[key].append((p['mat1'], p['mat2']))
                    entries_by_patch[key].append(p)

                n_params = num_col_groups * (3 * bit_width + 1)
                PtP_list, Ptr_list = [], []
                ones_col_ph = {}

                for i, (r0, r1) in enumerate(h_ranges):
                    ph = r1 - r0
                    R_S = x_tensor[r0:r1, :].to(dtype=dtype) @ S
                    if ph not in ones_col_ph:
                        ones_col_ph[ph] = torch.ones(ph, 1, dtype=dtype)
                    ones_col = ones_col_ph[ph]

                    G_cols = []
                    for j in range(num_col_groups):
                        for b_idx in range(bit_width):
                            Y_b, Z_b = binary_by_patch[(i, j)][b_idx]
                            Y_b, Z_b = Y_b.to(dtype=dtype), Z_b.to(dtype=dtype)
                            G_cols.extend([
                                ((Y_b @ Z_b) @ S_j_list[j]).reshape(-1),
                                (Y_b.sum(-1, keepdim=True) @ col_sum_S_list[j]).reshape(-1),
                                (ones_col @ (Z_b.sum(-2, keepdim=True) @ S_j_list[j])).reshape(-1),
                            ])
                        G_cols.append((ones_col @ col_sum_S_list[j]).reshape(-1))

                    Phi = torch.stack(G_cols, dim=1)
                    PtP_list.append(Phi.T @ Phi)
                    Ptr_list.append(Phi.T @ R_S.reshape(-1, 1))

                PtP_batch = torch.stack(PtP_list)
                Ptr_batch = torch.stack(Ptr_list)
                mean_diag = PtP_batch.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)
                eye = torch.eye(n_params, dtype=dtype).unsqueeze(0)

                theta_batch = None
                for reg in [1e-6, 1e-4, 1e-2, 1e-1]:
                    try:
                        sol = torch.linalg.solve(PtP_batch + reg * mean_diag * eye, Ptr_batch)
                        if sol.isfinite().all():
                            theta_batch = sol.squeeze(-1)
                            break
                    except Exception:
                        continue

                if theta_batch is not None:
                    Wq = torch.zeros(original_h, original_w, dtype=dtype)
                    for i, (r0, r1) in enumerate(h_ranges):
                        theta = theta_batch[i]
                        p2 = 0
                        for j, (c0, c1) in enumerate(w_ranges):
                            patch_entries = entries_by_patch[(i, j)]
                            for b_idx in range(bit_width):
                                a_v, b_v, c_v = theta[p2].item(), theta[p2+1].item(), theta[p2+2].item()
                                p2 += 3
                                Y_b, Z_b = binary_by_patch[(i, j)][b_idx]
                                patch_entries[b_idx]['coeff'] = torch.tensor([a_v, b_v, c_v, 0.0], dtype=dtype)
                                Wq[r0:r1, c0:c1] += (a_v * Y_b.float() @ Z_b.float()
                                    + b_v * Y_b.float().sum(1, keepdim=True)
                                    + c_v * Z_b.float().sum(0, keepdim=True))
                            d_v = theta[p2].item(); p2 += 1
                            patch_entries[0]['coeff'][3] = d_v
                            Wq[r0:r1, c0:c1] += d_v
                    print(f'  Scale refine done')
                else:
                    print('  WARNING: scale refine solve failed')

        # Save
        if consolidated_path and all_decomposed:
            os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
            torch.save(all_decomposed, consolidated_path)
            print(f'Saved consolidated: {consolidated_path} ({len(all_decomposed)} entries)')

        self.x = copy.copy(x_copy)
        return Wq


class BinaryMatrixFactorization():
    def __init__(self):
        pass

    def diag_zero_3d(self, y):
        # 各スライスごとに対角成分を0にする
        batch_size, _, _ = y.size()
        diag = torch.eye(y.shape[1], dtype=y.dtype, device=y.device).unsqueeze(0).expand(batch_size, -1, -1)
        return y * (1 - diag)

    def gradients_3d(self, x, y, z, a):
        
        common = (a**2).view(x.shape[0], 1, 1) - 2*a.view(x.shape[0], 1, 1)*x
        
        # バッチ次元で計算
        y_grad = torch.matmul(common, z.transpose(1, 2)) + 2*(a**2).view(x.shape[0], 1, 1) * torch.matmul(y, self.diag_zero_3d(torch.matmul(z, z.transpose(1, 2))))
        z_grad = torch.matmul(y.transpose(1, 2), common) + 2*(a**2).view(x.shape[0], 1, 1) * torch.matmul(self.diag_zero_3d(torch.matmul(y.transpose(1, 2), y)), z)

        return y_grad, z_grad

    

    def run_binary_multi(self, x, rank_scale, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1, compile_mode="reduce-overhead"):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        torch.manual_seed(seed)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        batch_size, n, m = x.shape
        self.rank = int(round(rank_scale * ((n * m) / (n + m))))  # rank_scaleはスケーリング係数
        self.delta_temp = torch.tensor((Tinit - Tfin) / (Nstep - 1), device=device)  
        temp = torch.tensor(Tinit, device=device)

        
        # 入力をGPUに転送
        x = x.to(device)
        maximum = x.max(dim=2)[0].max(dim=1)[0] - x.min(dim=2)[0].min(dim=1)[0]  # 最大幅(最大-最小)
        x = x/maximum.view(x.shape[0], 1, 1)
        
        
        yb = torch.rand((batch_size, n, self.rank), device=device)
        zb = torch.rand((batch_size, self.rank, m), device=device)
        
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb
        matrix = (y.transpose(1, 2)@y)*(z@z.transpose(1, 2))
        up = torch.triu(matrix, diagonal=1)
        down = torch.tril(matrix, diagonal=-1)
        a = ((x * (y @ z)).sum(dim=(1, 2))) / ((y @ z).sum(dim=(1, 2)) + up.sum(dim=(1, 2)) + down.sum(dim=(1, 2)))

        @torch.compile(mode=compile_mode)
        def loop_body(y, z, yb, zb, a, temp):
            with torch.no_grad():
                # 更新計算
                yf = y + zeta * (y - yb)
                zf = z + zeta * (z - zb)

                y_energy_grad, z_energy_grad = self.gradients_3d(x, yf, zf, a)

                # yとzの更新
                y_entropy_grad = temp * (y - 0.5)
                z_entropy_grad = temp * (z - 0.5)

                ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
                za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

                # aの更新
                matrix = ((ya.transpose(1, 2)@ya)*(za@za.transpose(1, 2)))

                up = torch.triu(matrix, diagonal=1) 
                # down = torch.tril(matrix, diagonal=-1) # up.sum()=down.sum()だから省略
                common = ya @ za

                a = ((x * common).sum(dim=(1, 2))) / (common.sum(dim=(1, 2)) + 2*up.sum(dim=(1,2)))

            return ya, za, y, z, a

        for _ in tqdm(range(Nstep)):
            y = y.detach().clone()
            yb = yb.detach().clone()
            z = z.detach().clone()
            zb = zb.detach().clone()
            a = a.detach().clone()
            y, z, yb, zb, a = loop_body(y, z, yb, zb, a, temp)
            temp -= self.delta_temp
            

        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # aの更新
        matrix = ((y.transpose(1, 2)@y)*(z@z.transpose(1, 2)))
        up = torch.triu(matrix, diagonal=1) 
        common = y @ z
        a = ((x * common).sum(dim=(1, 2))) / (common.sum(dim=(1, 2)) + 2*up.sum(dim=(1,2)))
        

        # print('Final Energy', H, self.energy_binary_multi(x, y, z, a))
        return y.detach(), z.detach(), (maximum*a).view(x.shape[0], 1, 1).detach()
   





class BinaryCodingQuantization():
    def __init__(self):
        pass
    

    @torch.inference_mode()
    def run_bcq(self, w, bit_width, Nstep=15, group_size=-1, transpose=False, exponent=0.0, clipping=1.0, pruning=0.0, use_bst=True):
        qbits = bit_width
        w_org = copy.deepcopy(w)
        w = w.flatten().unsqueeze(0)
        rounds = Nstep
        '''
        Post-training Weighted Quantization (BCQ format)
        https://openreview.net/pdf?id=2Id6XxTjz7c

        rounds == 0: greedy algorithm
        rounds == 1: refined greedy algorithm
        rounds >= 2: alternating algorithm

        :param w: a weight tensor of layer
        :param qbits: number of quantization bits for the `w`
        :param rounds: number of iterations for refining both alpha and B
        :param group_size: number of weights in which a scaling factor can be shared
        :param transpose: if `transpose` is True, `w` is a transposed when using this method.
        :param exponent: the exponent term of weighted factor.
                        if `exponent` is zero, this method is exactly the same as conventional BCQ method.
        :param clipping: the clipping importance term(0 <= clipping <= 1) of weighted factor.
        :param pruning: the pruning ratio(0 <= pruning <= 1) of weighted factor.
        :param use_bst: if `use_bst` is True(default), the binary matrix is calculated using BST algorithm.
                        if `use_bst` is False, the binary matrix is calculated with greedy algorithm.
        '''
        w_ = w.clone().float()
        w_ = w_.cuda()

        if transpose:
            assert len(w_.shape) == 2, f'Check your weight shape {w_.shape}'
            w_ = w_.transpose(1, 0).contiguous()
        
        orig_shape = w_.shape
        group_size = group_size if group_size > 0 else orig_shape[-1]
        w_ = w_.view([-1, group_size])
    
        # init weighted
        w_abs = w_.abs()
        ws, _ = w_abs.view(-1).sort()
        wf = torch.ones(w_.shape, dtype=torch.float32, device=w.device)
        if pruning > 0.0:
            wf = wf * (w_ != 0.0)
        if exponent > 0.0 or clipping < 1.0:
            wf = w_abs / w_abs.max()
        # weighted factor for C
        if clipping < 1.0:
            c_th = ws[int(ws.size(0) * clipping)].item()
            wf = wf * w_abs.max() / c_th
            wf[wf > 1.0] = 1.0
        # weighted factor for E
        if exponent > 0.0:
            wf = wf ** exponent
        # weighted factor for P
        if pruning > 0.0:
            p_th = ws[int(ws.shape[0] * pruning)].item()
            wf[w_abs <= p_th] = 0.0
            w_[w_abs <= p_th] = 0.0

        wf = wf.to(w_.device)
        # greedy & alternating algo.
        ret, B, alpha = self.greedy_mean_torch(w_, n_bits=qbits, wf=wf)
        if rounds > 0 and qbits > 1:
            # for _ in range(rounds):
            for _ in range(rounds):
                ret, B, alpha = self.refine_mean_torch(w_, ret, B, alpha, wf=wf, use_bst=use_bst)

        ret = ret.view(orig_shape) 
        if transpose:
            ret = ret.transpose(1, 0).contiguous()

        del w_
        
        B = B.reshape([orig_shape[0], orig_shape[1] // group_size, group_size, qbits])
        alpha = alpha.reshape([orig_shape[0], orig_shape[1] // group_size, qbits])
        m, n = w_org.shape
        B = B.reshape(m, n, qbits)
        alpha = alpha.squeeze(0).squeeze(0)
        ret = ret.reshape_as(w_org)

        return ret, B, alpha

    def greedy_mean_torch(self, w, n_bits=1, wf=None):
        B = torch.zeros(w.shape + (n_bits,), device=w.device)
        Alpha = torch.zeros(w.shape[0], n_bits, device=w.device)
    
        r, w_hat = w.clone(), 0.
        for i in range(n_bits):
            b = r.sign()
            
            if wf is not None:
                a1sum = torch.sum(wf, dim=1)
                alpha = (r.abs()*wf).sum(dim=1) / torch.sum(wf, dim=1)
                alpha[torch.isnan(alpha)] = 0.
                alpha = alpha.view(alpha.shape[0], 1)
            else:
                alpha = r.abs().mean(dim=1, keepdim=True)
            
            r -= b * alpha
            w_hat += b * alpha
            B[:,:,i] = b
            Alpha[:,i] = alpha.view(-1)
        

        return w_hat, B, Alpha

    def refine_mean_torch(self, w, w_hat, B, Alpha, wf=None, use_bst=True):
        w = w.float()
        d1, d2 = w.shape
        with torch.no_grad():
            n_bits = B.shape[-1]
            Bt = B.transpose(1, 2)
            if wf is not None:
                Bt = Bt * wf.unsqueeze(1)
            B_cov = Bt.bmm(B)
            Btw = Bt.bmm(w.unsqueeze(-1)).view(d1, n_bits)

            Alpha_new = self.batch_cg_torch(B_cov, Btw, x=Alpha)
            Alpha_new, _ = Alpha_new.abs().sort(descending=True)

            if use_bst == False:
                r = w.clone()
                B_new = torch.zeros_like(B)
                for i in range(n_bits):
                    B_new[:, :, i] = r.sign()
                    r -= B_new[:, :, i] * Alpha_new[:, i].view([-1, 1])
                del r
            else:
                B_new = self.find_B_torch(w, Alpha_new)
                B_new = B_new * (wf != 0.0).unsqueeze(-1)
            w_hat_new = torch.einsum('ijl,il->ij', (B_new, Alpha_new))

        return w_hat_new, B_new, Alpha_new

    def list_binary_vecs(self, n):
        ListBinaryVecs = {0 : [[]]}
        for m in range(1, n+1):
            ListBinaryVecs[m] = [[1.] + l for l in ListBinaryVecs[m-1]] + [[-1.] + l for l in ListBinaryVecs[m-1]]
        return ListBinaryVecs

    def find_B_torch(self, w, Alpha):
        '''Find optimal quantization assignment via binary search (torch)'''
        n_bits = Alpha.shape[-1]

        ListBinaryVecs = self.list_binary_vecs(n_bits)
        bin_mat = torch.from_numpy(np.vstack(ListBinaryVecs[n_bits]).astype(np.float32)).to(w.device)

        d1, d2 = w.shape
        row_inds = torch.arange(d1, dtype=torch.long).view(d1, 1).repeat([1, d2]).view(-1)
        # w is d1xd2, Alpha is d1xk, v is d1x2^k
        v = Alpha.mm(bin_mat.t())
        v_sorted, inds = torch.sort(v)
        # Binary search to find nearest neighbor
        w_flat = w.view([-1])
        Left = torch.zeros(d1*d2, dtype=torch.long, device=w.device)
        Right = torch.ones(d1*d2, dtype=torch.long, device=w.device) * (2 ** n_bits - 1)
        for i in range(n_bits):
            Mid_Left = torch.div(Left + Right - 1, 2, rounding_mode='trunc')
            Mid_Right = Mid_Left + 1
            mid_vals = (v_sorted[row_inds, Mid_Left] + v_sorted[row_inds, Mid_Right]) / 2
            inds_left = (w_flat < mid_vals)
            Right[inds_left] = Mid_Left[inds_left]
            Left[~inds_left] = Mid_Right[~inds_left]
        assignment_inds = inds[row_inds, Left].view(d1, d2)
        return bin_mat[assignment_inds, :]

    def batch_cg_torch(self, A, b, x=None):
        '''Batch conjugate gradient for solving Ax = b'''
        d1, k, _ = A.shape
        # Initialize
        x = x.clone().view(d1, k, 1)
        b = b.view(d1, k, 1)
        r = b - A.bmm(x)
        rtr_new = r.transpose(1, 2).bmm(r)
        p = r.clone()
        # Perform batch CG
        for i in range(k):
            rtr = rtr_new
            Ap = A.bmm(p)
            alpha = rtr / (p.transpose(1, 2).bmm(Ap) + 1e-6)
            x += alpha * p
            r -= alpha * Ap
            rtr_new = r.transpose(1, 2).bmm(r)
            beta = rtr_new / (rtr + 1e-6)
            p = r + beta * p
        return x.view(d1, k)

    def patchify(self, tensor, max_patch_size=256):
        """
        テンソルをパッチに分割する関数

        Args:
            tensor: 分割するテンソル (バッチxチャンネルx高さx幅)
            max_patch_size: パッチの最大サイズ (int)

        Returns:
            パッチに分割されたテンソル
        """

        height, width = tensor.shape

        # 各次元の最大約数を計算 (256以下)
        
        def get_max_divisor(num, max_value):
            """
            与えられた数値の最大約数を、指定された最大値以下で求める関数

            Args:
                num: 約数を求める数値
                max_value: 最大約数の最大値

            Returns:
                最大約数
            """

            # 最大約数の探索範囲を決定
            limit = max(int(math.sqrt(num)), max_value)

            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        patch_height = get_max_divisor(height, max_patch_size)
        patch_width = get_max_divisor(width, max_patch_size)
        print('Patch Size:({0}x{1})'.format(patch_height, patch_width))

        # unfold関数でパッチに分割 (パッチサイズはpatch_height x patch_width, patch_height x patch_width間隔で切り出す)
        return tensor.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)
    

    
    def unpatchify(self, patches, original_shape):
        """
        パッチから元のテンソルを復元する関数

        Args:
            patches: パッチに分割されたテンソル (バッチxパッチ高さxパッチ幅xパッチサイズxパッチサイズ)
            original_shape: 元のテンソルの形状 (チャンネルx高さx幅)

        Returns:
            復元されたテンソル
        """
        num_patches_h, num_patches_w, patch_height, patch_width = patches.shape
        height, width = original_shape

        # 元の形状と整合性の確認
        assert height == num_patches_h * patch_height, "Height mismatch between patches and original shape"
        assert width == num_patches_w * patch_width, "Width mismatch between patches and original shape"

        # パッチを再構成
        reconstructed_tensor = torch.zeros((height, width), device=patches.device)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                reconstructed_tensor[
                    i * patch_height: (i + 1) * patch_height,
                    j * patch_width: (j + 1) * patch_width
                ] = patches[i, j, :, :]

        return reconstructed_tensor
    
    def bcq_large_matrix(self, w, max_patch_size, bit_width, Nstep=50, save_name=None):
        """
        大きな行列をパッチに分け、それぞれのパッチで行列分解を行い、復元する関数

        Args:
            max_patch_size: パッチの最大サイズ
            zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type: 行列分解に関するパラメータ

        Returns:
            元の形状に復元されたテンソル
        
        注意：入力のテンソルは３次元(batch_num, row_num, colum_num)になっていないといけないので、２次元の場合は変換してから入力すること
        """
        if save_name is None:
            save = False
        else: save = True

        

        # テンソルをパッチに分割
        divided_tensor = self.patchify(w, max_patch_size=max_patch_size)

        # パッチサイズと数を取得
        num_patches_h, num_patches_w, patch_height, patch_width = divided_tensor.shape


        # 各パッチで行列分解を実行
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = divided_tensor[i, j, :, :]

                # 分解結果を保存するリスト
                decomposed_patches = []

                reconst, B, alpha = self.run_bcq(patch, bit_width, Nstep=Nstep, group_size=-1, transpose=False, exponent=0.0, clipping=1.0, pruning=0.0, use_bst=True)
                if save:
                    data = {'patch_row':i, 'patch_col':j, 'coeff':alpha, 'binary_matrix':B, 'bit_width':bit_width}
                    decomposed_patches.append(data)
                    torch.save(decomposed_patches, (save_name + f'_row{i}_col{j}.pth'))
                
                # 分解結果を保存
                divided_tensor[i, j, :, :] = (reconst).clone().detach()

        # パッチから元の形状に復元
        reconstructed_tensor = self.unpatchify(divided_tensor, w.shape)

        return reconstructed_tensor
    
class UniformQuantization():
    def __init__(self):
        pass

    def run_uq(self, matrix, n_bits, device=torch.device("cpu")):
        matrix = matrix.to(device)
        num_levels = 2**n_bits
        """
        行列を量子化し、最適な量子化を選ぶ関数。

        Parameters:
            matrix (np.ndarray): 量子化する行列
            num_levels (int): 量子化のビット深度（例えば256なら8ビット量子化）

        Returns:
            quantized_matrix (np.ndarray): 最適な量子化結果
            best_scale (float): 最適なスケール
            min_error (float): 最小誤差 (RMSE)
        """
        # 標準偏差と範囲を計算
        mean = (matrix).mean()
        min = (matrix).min()
        max = (matrix).max()
        
        # 量子化スケールを標準偏差の倍率としていくつか試す
        min_error = float('inf')
        best_scale = None
        quantized_matrix = None
        
        for range_max in (torch.linspace(mean, max, 100)):
            for range_min in torch.linspace(min, mean, 100):
                range_min = range_min.to(device)
                range_max = range_max.to(device)
                # 行列を量子化
                quantized = torch.clamp(matrix, range_min, range_max)  # 範囲外をクリップ
                if range_max == range_min:buffer=1e-8
                else:buffer=0
                quantized = torch.round(
                    (quantized - range_min) / (range_max - range_min + buffer) * (num_levels - 1)
                ) / (num_levels - 1) * (range_max - range_min) + range_min
            
                # MSEを計算
                error = (((matrix - quantized) ** 2).mean())
                
                # 最適スケールを更新
                if error < min_error:
                    min_error = error
                    best_scale = (range_min, range_max)
                    quantized_matrix = quantized

        return quantized_matrix
    
    def channel_wise_uq(self, tensor, n_bits):
        matrix_list = []
        for i in range(tensor.shape[0]):
            matrix_list.append(self.run_uq(tensor[i], n_bits))
        return (torch.stack(matrix_list, axis=0))
    
    def patchify(self, tensor, max_patch_size=256):
        """
        テンソルをパッチに分割する関数

        Args:
            tensor: 分割するテンソル (バッチxチャンネルx高さx幅)
            max_patch_size: パッチの最大サイズ (int)

        Returns:
            パッチに分割されたテンソル
        """

        height, width = tensor.shape

        # 各次元の最大約数を計算 (256以下)
        
        def get_max_divisor(num, max_value):
            """
            与えられた数値の最大約数を、指定された最大値以下で求める関数

            Args:
                num: 約数を求める数値
                max_value: 最大約数の最大値

            Returns:
                最大約数
            """

            # 最大約数の探索範囲を決定
            limit = max(int(math.sqrt(num)), max_value)

            for i in range(limit, 0, -1):
                if num % i == 0 and i <= max_value:
                    return i
            return 1

        patch_height = get_max_divisor(height, max_patch_size)
        patch_width = get_max_divisor(width, max_patch_size)
        print('Patch Size:({0}x{1})'.format(patch_height, patch_width))

        # unfold関数でパッチに分割 (パッチサイズはpatch_height x patch_width, patch_height x patch_width間隔で切り出す)
        return tensor.unfold(0, patch_height, patch_height).unfold(1, patch_width, patch_width)
    

    
    def unpatchify(self, patches, original_shape):
        """
        パッチから元のテンソルを復元する関数

        Args:
            patches: パッチに分割されたテンソル (バッチxパッチ高さxパッチ幅xパッチサイズxパッチサイズ)
            original_shape: 元のテンソルの形状 (チャンネルx高さx幅)

        Returns:
            復元されたテンソル
        """
        num_patches_h, num_patches_w, patch_height, patch_width = patches.shape
        height, width = original_shape

        # 元の形状と整合性の確認
        assert height == num_patches_h * patch_height, "Height mismatch between patches and original shape"
        assert width == num_patches_w * patch_width, "Width mismatch between patches and original shape"

        # パッチを再構成
        reconstructed_tensor = torch.zeros((height, width), device=patches.device)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                reconstructed_tensor[
                    i * patch_height: (i + 1) * patch_height,
                    j * patch_width: (j + 1) * patch_width
                ] = patches[i, j, :, :]

        return reconstructed_tensor
    
    def uq_large_matrix(self, w, max_patch_size, bit_width, save_name=None, device=torch.device("cpu")):
        """
        大きな行列をパッチに分け、それぞれのパッチで行列分解を行い、復元する関数

        Args:
            max_patch_size: パッチの最大サイズ
            zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type: 行列分解に関するパラメータ

        Returns:
            元の形状に復元されたテンソル
        
        注意：入力のテンソルは３次元(batch_num, row_num, colum_num)になっていないといけないので、２次元の場合は変換してから入力すること
        """
        if save_name is None:
            save = False
        else: save = True

        

        # テンソルをパッチに分割
        divided_tensor = self.patchify(w, max_patch_size=max_patch_size)

        # パッチサイズと数を取得
        num_patches_h, num_patches_w, patch_height, patch_width = divided_tensor.shape


        # 各パッチで行列分解を実行
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = divided_tensor[i, j, :, :]

                # 分解結果を保存するリスト
                decomposed_patches = []

                reconst = self.run_uq(patch, bit_width, device)
                if save:
                    data = {'patch_row':i, 'patch_col':j, 'q_matrix':reconst.to('cpu'), 'bit_width':bit_width}
                    decomposed_patches.append(data)
                    torch.save(decomposed_patches, (save_name + f'_row{i}_col{j}.pth'))
                
                # 分解結果を保存
                divided_tensor[i, j, :, :] = (reconst).clone().detach()

        # パッチから元の形状に復元
        reconstructed_tensor = self.unpatchify(divided_tensor, w.shape)

        return reconstructed_tensor
    






class LatticeVectorQuantization:
    def __init__(self):
        pass



    def generate_e8_root(self):
        e8_vectors = []

        # --- Type A: ±0.5 with even number of - signs ---
        signs = np.array(np.meshgrid(*[[0.5, -0.5]] * 8)).T.reshape(-1, 8)
        even_signs = signs[np.sum(signs == -0.5, axis=1) % 2 == 0]
        e8_vectors.append(even_signs)

        # --- Type B: ±1 at 2 positions, rest 0 ---
        for i, j in combinations(range(8), 2):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    vec = np.zeros(8)
                    vec[i] = s1
                    vec[j] = s2
                    e8_vectors.append(vec)

        return np.vstack(e8_vectors)
    
    def expand_codebook_with_scaling(self, codebook, x, num_scales=17):
        """
        - codebook: [240, 8] テンソル
        - x: 入力データ行列 [N, 8]
        - num_scales: 係数の数（デフォルト17）
        """
        # 1. 入力データのノルム最大値を計算
        max_norm = torch.norm(x, dim=1).max()

        # 2. 係数を0〜max_normの間で等間隔に分割（ただし0は除く）
        scale_factors = torch.linspace(0, max_norm, steps=num_scales + 1)[1:]  # 長さ17, index 1~17

        # 3. スケーリングされたコードブックを生成（240 × 17 = 4080個）
        expanded_codebook = torch.cat([
            scale * codebook for scale in scale_factors
        ], dim=0)  # shape: [4080, 8]

        return expanded_codebook
    
    def run_e8_lvq(self, x, n_bits, scale_bits=2):
        original_shape = x.shape
        x_flat = x.reshape(-1)

        # パディング（nの倍数にする）
        total_elements = x_flat.numel()
        remainder = total_elements % 8
        if remainder != 0:
            pad_size = 8 - remainder
            x_flat = torch.cat([x_flat, torch.zeros(pad_size, device=x.device, dtype=x.dtype)])

        D = x_flat.reshape(-1, 8).float()

        C = torch.tensor(self.generate_e8_root()).float() # make codebook

        """
        C: (240, 8) コードブック（コードベクトルの集合）
        D: (n, 8)   データ行列（元のベクトル）

        Returns:
            D_hat: (n, 8) 復元された行列
        """
        totalD = torch.zeros_like(D)
        for bit in range(n_bits):
            # Normalize C and D for cosine similarity
            C_norm = torch.nn.functional.normalize(C, dim=1)  # (240, 8)
            D_norm = torch.nn.functional.normalize(D, dim=1)  # (n, 8)

            # (n, 240) の cosine 類似度行列を計算
            similarity = D_norm @ C_norm.T

            # 各D[i]に対して最も類似度の高いC[j]のインデックス
            indices = torch.argmax(similarity, dim=1)  # (n,)

            # 対応するコードベクトルを取り出す
            selected_codes = C[indices]  # (n, 8)

            # スカラー係数 α を最小二乗で計算（α = (x・c) / (c・c)）
            numerators = torch.sum(D * selected_codes, dim=1)        # (n,)
            denominators = torch.sum(selected_codes ** 2, dim=1)     # (n,)
            alpha = numerators / denominators                        # (n,)
            alpha = UniformQuantization().run_uq(alpha, n_bits=scale_bits)

            # αを (n,1) に reshape して selected_codes にかける
            D_hat = alpha.unsqueeze(1) * selected_codes  # (n, 8)
            D = D - D_hat
            totalD += D_hat

        totalD = totalD.reshape(-1)[:original_shape.numel()]

        return totalD.reshape(original_shape)
    
    def calc_memory_size(self, x, n_bits, scale_bits=2):
        x_flat = x.reshape(-1)

        # パディング（nの倍数にする）
        total_elements = x_flat.numel()
        remainder = total_elements % 8
        if remainder != 0:
            pad_size = 8 - remainder
            x_flat = torch.cat([x_flat, torch.zeros(pad_size, device=x.device, dtype=x.dtype)])

        D = x_flat.reshape(-1, 8)
        num_data, _ = D.shape
        memory = (num_data * (8 + scale_bits) + 32) * n_bits # id:8bit, scales bits:scale_bits, scale of scales: 32bit
        return memory / 8 # byte
    

    def run_scaled_e8_lvq(self, x, n_bits, num_scales=273):
        """
        Args:
            x: 任意shapeのテンソル（例: [B, D]）
            expanded_codebook: [4080, 8] のテンソル（スケーリングされたE8コードブック）

        Returns:
            totalD: x と同じ shape の量子化復元テンソル
            indices: 各ブロックに対応するコードブックの index（[num_blocks]）
        """
        original_shape = x.shape
        x_flat = x.reshape(-1)

        # パディングして8の倍数に
        total_elements = x_flat.numel()
        remainder = total_elements % 8
        if remainder != 0:
            pad_size = 8 - remainder
            x_flat = torch.cat([x_flat, torch.zeros(pad_size, device=x.device, dtype=x.dtype)])

        # (N, 8) に変換
        D = x_flat.reshape(-1, 8).float()

        # コードブック (M, 8)
        codebook = self.generate_e8_root()
        expanded_codebook = self.expand_codebook_with_scaling(codebook=codebook, x=D, num_scales=num_scales)
        C = expanded_codebook.to(D.device).float()

        totalD = torch.zeros_like(D)

        for bit in range(n_bits):
            # L2距離（二乗誤差）をバッチで計算
            # D: (N, 8), C: (M, 8) → dist^2[i,j] = ||D[i] - C[j]||^2
            D2 = (D ** 2).sum(dim=1, keepdim=True)      # (N, 1)
            C2 = (C ** 2).sum(dim=1).unsqueeze(0)       # (1, M)
            DC = D @ C.T                                 # (N, M)
            dist2 = D2 - 2 * DC + C2                     # (N, M)

            # 最も距離が小さいセントロイドを選ぶ
            indices = torch.argmin(dist2, dim=1)         # (N,)
            selected_codes = C[indices]                  # (N, 8)

            totalD += selected_codes
            D -= selected_codes

        # 復元された量子化テンソル
        # totalD = selected_codes.reshape(-1)[:x_flat.numel()]
        totalD = totalD[:total_elements]             # パディング除去

        return totalD.reshape(original_shape)
    

    def calc_scaled_memory_size(self, x, n_bits, num_scales):
        x_flat = x.reshape(-1)

        # パディング（nの倍数にする）
        total_elements = x_flat.numel()
        remainder = total_elements % 8
        if remainder != 0:
            pad_size = 8 - remainder
            x_flat = torch.cat([x_flat, torch.zeros(pad_size, device=x.device, dtype=x.dtype)])

        D = x_flat.reshape(-1, 8)
        num_data, _ = D.shape
        memory = (num_data * (math.ceil(math.log2(240*num_scales)))) * n_bits # id:8bit, scale:32bit
        return memory / 8 # byte
    















class TransformQuantization:
    def __init__(self):
        self.uq = UniformQuantization()

    def run_hq(self, matrix, n_bits, n_reshape=None):
        """
        アダマール変換 + 量子化 + 復元 を行う。

        Parameters:
        - matrix: ndarray of shape (n_samples, n_features)
        - n_bits: int, 量子化ビット数

        Returns:
        - reconstructed_matrix: 復元後の行列（元の空間に近い）
        """
        matrix = np.array(matrix)
        if n_reshape is None:
            n_reshape = matrix.shape[-1]
        original_shape = matrix.shape
        matrix = matrix.reshape((-1, n_reshape))
        n_samples, n_features = matrix.shape

        # 列数が2のべき乗でなければパディング
        target_dim = 2 ** int(np.ceil(np.log2(n_features)))
        pad_width = target_dim - n_features
        if pad_width > 0:
            matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')

        # アダマール行列
        H = hadamard(matrix.shape[1])

        # アダマール変換（列方向）
        transformed = matrix @ H.T

        # 量子化
        quantized = UniformQuantization().run_uq(torch.tensor(transformed), n_bits)

        # 逆アダマール変換
        inverse_transformed = quantized.float() @ torch.tensor(H).float() / H.shape[0]  # Hadamard is self-inverse up to scaling

        # パディングを戻す
        if pad_width > 0:
            inverse_transformed = inverse_transformed[:, :n_features]


        return inverse_transformed.reshape(original_shape)

    def calc_memory_size(self, matrix, n_bits, n_reshape=None):
        matrix = np.array(matrix)
        if n_reshape is None:
            n_reshape = matrix.shape[-1]
        matrix = matrix.reshape(-1, n_reshape)
        n_samples, n_features = matrix.shape
        target_dim = 2 ** int(np.ceil(np.log2(n_features)))
        pad_width = target_dim - n_features
        if pad_width > 0:
            n_features = target_dim  # パディング後の列数を使う

        data_bytes = n_samples * n_features * n_bits / 8
        param_bytes = 4 + 4  # scale と bias をfloat32で2つ分

        total_bytes = data_bytes + param_bytes
        return total_bytes

    def next_power_of_two(self, x):
        return 1 << (x - 1).bit_length()



    def run_ht_compress(self, X, remaining_ratio, n_bits=32):
        """
        アダマール変換を右から（列方向のみ）かけて圧縮する。
        - X: 実数行列（NumPy）
        - remaining_ratio: 保持する列数の割合（0〜1）
        - n_bits: 量子化ビット数（32でfloat圧縮、他はスカラー量子化）
        """
        original_shape = X.shape
        m, n = original_shape

        # 列数（右）を2のべき乗にパディング
        n_pad = self.next_power_of_two(n)
        X_padded = np.zeros((m, n_pad))
        X_padded[:, :n] = X

        # アダマール行列（列方向のみ）
        Hn = hadamard(n_pad)

        # 右から変換
        Y = X_padded @ Hn

        # 列方向だけをカット
        k_col = min(n, math.ceil(n * remaining_ratio))

        if n_bits == 32:
            Y_compress = Y[:, :k_col]
        else:
            Y_compress = UniformQuantization().run_uq(torch.tensor(Y[:, :k_col]), n_bits=n_bits).numpy()

        # マスクして埋め戻す
        Y_masked = np.zeros_like(Y)
        Y_masked[:, :k_col] = Y_compress

        # 逆変換（右から）
        X_recon_padded = Y_masked @ Hn / n_pad

        # 元のサイズに戻す
        X_recon = X_recon_padded[:, :n]

        # メモリサイズ計算
        if n_bits == 32:
            memory_size = m * k_col * n_bits / 8  # bytes
        else:
            memory_size = m * k_col * n_bits / 8 + 4 + 4  # bias + scale

        return X_recon, memory_size

    
    



    def run_dct_compress(self, X, remaining_ratio, n_bits=32):
        m, n = X.shape

        # DCT-II（2次元 DCT）
        def dct2(a):
            if isinstance(a, torch.Tensor):
                a = a.cpu().numpy()
            return dct(dct(a.T, norm='ortho').T, norm='ortho')

        # 逆 DCT（IDCT-II）
        def idct2(a):
            if isinstance(a, torch.Tensor):
                a = a.cpu().numpy()
            return idct(idct(a.T, norm='ortho').T, norm='ortho')

        # DCT変換（パディングなし）
        Y = dct2(X)

        # 残す比率に基づいて行・列の数を直接決定
        scale = math.sqrt(remaining_ratio)
        k_row = min(m, math.ceil(m * scale))
        k_col = min(n, math.ceil(n * scale))

        # マスク処理（必要な部分だけ残す）
        if n_bits == 32:
            Y_compress = Y[:k_row, :k_col]
        else:
            Y_compress = UniformQuantization().run_uq(torch.tensor(Y[:k_row, :k_col]), n_bits=n_bits).numpy()

        Y_masked = np.zeros_like(Y)
        Y_masked[:k_row, :k_col] = Y_compress


        # 逆DCT（パディングなし）
        X_recon = idct2(Y_masked)

        # メモリサイズの見積もり（量子化された場合は補正）
        if n_bits == 32:
            memory_size = k_col * k_row * n_bits / 8  # byte
        else:
            memory_size = k_col * k_row * n_bits / 8 + 4 + 4  # byte (bias and scale)

        return X_recon, memory_size
    


class JPEG():
    def __init__(self):
        pass
    



    def run_jpeg_compress(self, X: np.ndarray, n_bits=4):
        """
        NumPyの実数行列 X（0〜1正規化前提）をJPEGに圧縮し、
        再度復元したNumPy配列を返す。
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        num_elements = X.shape[0] * X.shape[1]
        target_bytes = n_bits * num_elements / 8
        # 0〜255にスケーリングしてuint8に変換
        X8 = UniformQuantization().run_uq(X, n_bits=8).detach().numpy()
        bias = X8.min()
        scale = X8.max() - X8.min()
        X_clipped = (X8 - bias)/scale
        X_uint8 = (X_clipped * 255).astype(np.uint8)

        # グレースケール画像に変換
        img = Image.fromarray(X_uint8, mode='L')  # 'L' = 8bitグレースケール


        # JPEG品質を調整しながら圧縮
        best_quality = 100
        best_size = float('inf')
        for quality in range(100, 0, -1):  # 高品質から試す
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            size = len(buffer.getvalue())
            if size <= target_bytes:
                best_quality = quality
                best_size = size
                break

        # 圧縮したJPEGを復元
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=best_quality)
        buffer.seek(0)
        img_recon = Image.open(buffer)
        X_recon = np.array(img_recon).astype(np.float32) / 255.0
        X_recon = X_recon * scale + bias

        return X_recon, best_size + 4 + 4  # +bias, +scale (同様のオーバーヘッド)

    








class VectorQuantization():
    def __init__(self):
        pass        

    def run_vq(self, matrix, num_centroid, centroid_bits=32):
        """
        ベクトル量子化を行い、復元行列を返す。

        Parameters:
        - matrix: ndarray of shape (n_samples, n_features), 各行が量子化対象のベクトル
        - num_centroid: int, セントロイド（クラスタ）の数

        Returns:
        - reconstructed_matrix: ndarray of shape (n_samples, n_features)
        """
        # 入力をnumpy配列に変換（念のため）
        matrix = np.array(matrix)

        # KMeansクラスタリングを実行（量子化）
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_centroid, random_state=0, n_init='auto')
        kmeans.fit(matrix)

        # 各ベクトルが割り当てられたセントロイドのインデックスを取得
        labels = kmeans.predict(matrix)

        # セントロイドから復元行列を作成
        if centroid_bits == 32:
            centroids = kmeans.cluster_centers_
        else:
            centroids = UniformQuantization().run_uq(torch.tensor(kmeans.cluster_centers_), centroid_bits)
        reconstructed_matrix = centroids[labels]

        return reconstructed_matrix
    

    def calc_memory_size(self, matrix, num_centroid, centroid_bits=32):
        num_row, num_col = matrix.shape

        # インデックスに必要なビット数 → バイト換算（切り上げ）
        bits_per_index = ceil(log2(num_centroid))
        bytes_per_index = ceil(bits_per_index / 8)
        idx_memory = num_row * bytes_per_index  # 単位: バイト

        # セントロイド部分（float32 = 4バイト）
        if centroid_bits == 32:
            centroid_memory = num_centroid * num_col * centroid_bits/8   # 単位: バイト
        else: centroid_memory = num_centroid * num_col * centroid_bits/8 + 32/8

        return idx_memory + centroid_memory


# Backward compatibility alias (V1 class was merged into V2)
