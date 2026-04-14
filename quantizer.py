import numpy as np
from tqdm import tqdm, trange
import torch
import torch.multiprocessing as mp
import torch.nn as nn
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



class BinaryQuadraticQuantization():

    def __init__(self, x, rank=None, rank_scale=1):
        self.rank_scale=rank_scale
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


    def bqq_large_matrix_multi_worker(self, max_patch_size, bit_width, consolidated_path=None, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=10000, seed=1, workers_per_gpu=8, main_gpu_id=0, use_batch=True, H=None, damping=1e-6):
        """
        大きな行列をパッチに分割し、行列分解を実行して復元。
        結果は consolidated_path に1ファイルで保存する。
        既存の consolidated_path があれば完了済みパッチをスキップして再開する。

        Args:
            use_batch: Trueの場合バッチ処理(デフォルト)。Falseならマルチプロセス。
            H: 入力相関行列 X^T X (in_features, in_features)。
               指定時はHessian-awareスケール最適化をbit毎に実行。
            damping: Hessian-awareのTikhonov正則化。
        """
        if H is not None:
            return self._hessian_aware_large_matrix_batched(
                max_patch_size, bit_width, H, consolidated_path,
                zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id, damping)
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


    # ------------------------------------------------------------------
    # Hessian-aware large matrix batched
    # ------------------------------------------------------------------

    def _hessian_aware_large_matrix_batched(
        self, max_patch_size, bit_width, H,
        consolidated_path, zeta, eta, Tinit, Tfin, Nstep, seed, main_gpu_id,
        damping=1e-6,
    ):
        """
        Hessian-aware BQQ quantization.

        Same patching and batched binary decomposition as _large_matrix_batched,
        but after each bit, refines ALL scale coefficients (bits 0..current)
        to minimise  tr((W - Wq) H (W - Wq)^T).

        The refinement is done per-patch independently (row-group structure
        makes patches on different rows fully independent).

        Args:
            H: Input correlation matrix (in_features, in_features), i.e. X^T X.
               Must be on CPU; will be sliced per column-group.
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

        patch_specs = []
        for i, (r0, r1) in enumerate(h_ranges):
            for j, (c0, c1) in enumerate(w_ranges):
                patch_specs.append({'i': i, 'j': j, 'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1})
        total_patches = len(patch_specs)

        size_counts = defaultdict(int)
        for s in patch_specs:
            size_counts[(s['r1'] - s['r0'], s['c1'] - s['c0'])] += 1
        for (ph, pw), cnt in sorted(size_counts.items(), key=lambda x: -x[1]):
            print(f'Patch Size:({ph}x{pw}), Count: {cnt}')

        x_tensor = torch.tensor(x_copy).float()
        H = H.float()  # ensure float32

        # Per-patch binary data: binary_data[(i,j)] = [(Y_0, Z_0), (Y_1, Z_1), ...]
        binary_data = defaultdict(list)
        # Per-patch current coefficients: coeffs_data[(i,j)] = [coeff_0, coeff_1, ...]
        coeffs_data = defaultdict(list)
        # Reconstruction accumulator
        reconst_accum = {}
        for s in patch_specs:
            reconst_accum[(s['i'], s['j'])] = torch.zeros(s['r1'] - s['r0'], s['c1'] - s['c0'])

        for bit_idx in range(bit_width):
            # --- 1. Compute residuals and run batched BQQ ---
            size_groups = defaultdict(list)
            for s in patch_specs:
                key = (s['i'], s['j'])
                ph, pw = s['r1'] - s['r0'], s['c1'] - s['c0']
                original = x_tensor[s['r0']:s['r1'], s['c0']:s['c1']]
                residual = original - reconst_accum[key]
                size_groups[(ph, pw)].append((s, residual))

            for (ph, pw), group in size_groups.items():
                specs = [g[0] for g in group]
                residuals = [g[1] for g in group]
                x_batch = torch.stack(residuals)
                print(f'Bit {bit_idx}: decomposing {len(residuals)} patches of ({ph}x{pw})')

                y_b, z_b, a_b = self.run_bqq_compile_batched(
                    x_batch, rank_scale=rank_scale_copy,
                    zeta=zeta, eta=eta, Tinit=Tinit, Tfin=Tfin,
                    Nstep=Nstep, device_id=main_gpu_id, seed=seed
                )

                for b, s in enumerate(specs):
                    key = (s['i'], s['j'])
                    binary_data[key].append((y_b[b].cpu(), z_b[b].cpu()))
                    coeffs_data[key].append(a_b[b].cpu())

            # --- 2. Hessian-aware scale refinement per row-group (batched solve) ---
            num_row_groups = len(h_ranges)
            num_col_groups = len(w_ranges)
            current_bits = bit_idx + 1
            n_params = num_col_groups * (3 * current_bits + 1)
            print(f'Bit {bit_idx}: refining scales for {num_row_groups} row-groups '
                  f'({n_params} params each, batched solve)')

            # Cholesky of full H (shared across all rows)
            S = self._cholesky_safe(H.float(), damping)
            if S is None:
                print('  WARNING: Cholesky failed on H, skipping refinement')
            else:
                dtype = torch.float32
                ones_col_ph = {}  # cached per patch height

                # Precompute S_j blocks and their column sums
                S_j_list = []
                col_sum_S_list = []
                for c0, c1 in w_ranges:
                    S_j = S[c0:c1, :]            # (pw, full_w)
                    S_j_list.append(S_j)
                    col_sum_S_list.append(S_j.sum(dim=0, keepdim=True))  # (1, full_w)

                # Build PtP and Ptr for each row, then batch solve
                PtP_list = []
                Ptr_list = []
                row_valid = []

                for i, (r0, r1) in enumerate(h_ranges):
                    ph = r1 - r0
                    full_w = original_w
                    W_row = x_tensor[r0:r1, :].to(dtype=dtype)
                    R_S = W_row @ S  # (ph, full_w)

                    if ph not in ones_col_ph:
                        ones_col_ph[ph] = torch.ones(ph, 1, dtype=dtype)
                    ones_col = ones_col_ph[ph]

                    # Build Phi columns
                    G_cols = []
                    for j in range(num_col_groups):
                        S_j = S_j_list[j]
                        col_sum_S = col_sum_S_list[j]

                        bits = binary_data[(i, j)]
                        for b_idx in range(current_bits):
                            Y_b, Z_b = bits[b_idx]
                            Y_b = Y_b.to(dtype=dtype)
                            Z_b = Z_b.to(dtype=dtype)

                            G_a = (Y_b @ Z_b) @ S_j
                            G_b = Y_b.sum(dim=-1, keepdim=True) @ col_sum_S
                            G_c = ones_col @ (Z_b.sum(dim=-2, keepdim=True) @ S_j)
                            G_cols.extend([G_a.reshape(-1), G_b.reshape(-1), G_c.reshape(-1)])

                        G_d = ones_col @ col_sum_S_list[j]
                        G_cols.append(G_d.reshape(-1))

                    Phi = torch.stack(G_cols, dim=1)  # (ph*full_w, n_params)
                    rhs = R_S.reshape(-1, 1)

                    PtP_list.append(Phi.T @ Phi)
                    Ptr_list.append(Phi.T @ rhs)
                    row_valid.append(i)

                if PtP_list:
                    # Batch solve: (num_rows, n_params, n_params) @ (num_rows, n_params, 1)
                    PtP_batch = torch.stack(PtP_list)    # (R, P, P)
                    Ptr_batch = torch.stack(Ptr_list)     # (R, P, 1)
                    mean_diag = PtP_batch.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # (R,1,1)
                    eye = torch.eye(n_params, dtype=dtype).unsqueeze(0)

                    theta_batch = None
                    for reg in [1e-6, 1e-4, 1e-2, 1e-1]:
                        try:
                            A = PtP_batch + reg * mean_diag * eye
                            sol = torch.linalg.solve(A, Ptr_batch)  # (R, P, 1)
                            if sol.isfinite().all():
                                theta_batch = sol.squeeze(-1)  # (R, P)
                                break
                        except Exception:
                            continue

                    if theta_batch is not None:
                        # Unpack theta into per-patch coefficients
                        for row_idx, i in enumerate(row_valid):
                            theta = theta_batch[row_idx]
                            p = 0
                            for j in range(num_col_groups):
                                coeffs = []
                                for b_idx in range(current_bits):
                                    a_val = theta[p].item(); p += 1
                                    b_val = theta[p].item(); p += 1
                                    c_val = theta[p].item(); p += 1
                                    coeffs.append(torch.tensor([a_val, b_val, c_val, 0.0]))
                                d_val = theta[p].item(); p += 1
                                coeffs[0][3] = d_val  # assign d to bit 0
                                coeffs_data[(i, j)] = coeffs
                        print(f'  Refined {len(row_valid)} row-groups')
                    else:
                        print('  WARNING: batched solve failed')

            # --- 3. Recompute reconst_accum from updated coefficients ---
            for s in patch_specs:
                key = (s['i'], s['j'])
                accum = torch.zeros(s['r1'] - s['r0'], s['c1'] - s['c0'])
                for (Y_b, Z_b), coeff in zip(binary_data[key], coeffs_data[key]):
                    a0, a1, a2, a3 = coeff[0], coeff[1], coeff[2], coeff[3]
                    accum += a0 * Y_b @ Z_b + a1 * Y_b.sum(dim=1, keepdim=True) \
                           + a2 * Z_b.sum(dim=0, keepdim=True) + a3
                reconst_accum[key] = accum

        # --- Build all_decomposed for saving ---
        all_decomposed = []
        for s in patch_specs:
            key = (s['i'], s['j'])
            for bit_idx, ((Y_b, Z_b), coeff) in enumerate(
                    zip(binary_data[key], coeffs_data[key])):
                all_decomposed.append({
                    'patch_row': s['i'], 'patch_col': s['j'],
                    'row_start': s['r0'], 'row_end': s['r1'],
                    'col_start': s['c0'], 'col_end': s['c1'],
                    'coeff': coeff, 'mat1': Y_b, 'mat2': Z_b,
                    'bit_idx': bit_idx,
                })

        if consolidated_path and all_decomposed:
            os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
            torch.save(all_decomposed, consolidated_path)
            print(f'Saved consolidated: {consolidated_path} ({len(all_decomposed)} entries)')

        reconstructed_tensor = torch.zeros(original_h, original_w)
        for s in patch_specs:
            reconstructed_tensor[s['r0']:s['r1'], s['c0']:s['c1']] = reconst_accum[(s['i'], s['j'])]

        self.x = copy.copy(x_copy)
        return reconstructed_tensor


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
