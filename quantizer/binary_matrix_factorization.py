import numpy as np
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import os
import time
import math


class BinaryMatrixFactorization():
    def __init__(self, x, rank=None, rank_scale=2, sparse=False):
        self.rank_scale=rank_scale
        if isinstance(x, torch.Tensor):
        # GPU上に存在する場合はCPUに移動
            if x.is_cuda:
                x = x.detach().cpu()
                # NumPy配列に変換
            x = x.numpy()

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
        if sparse:
            self.nan_indices = torch.isnan(torch.Tensor(self.x)).nonzero(as_tuple=False)
            self.nan_mask = torch.isnan(torch.Tensor(self.x))


    def energy_binary(self, x, y, z, a):
        matrix = (y.T@y)*(z@z.T)
        # 対角成分を0にするマスクを作成
        up = torch.triu(matrix, diagonal=1)
        down = torch.tril(matrix, diagonal=-1)
        return torch.sum(x**2) + torch.sum((a**2 - 2*a*x) * (y@z)) + a**2 * (up.sum() + down.sum())
    
    def sparse_energy_binary(self, x, y, z, a):
        common = (y @ z)
        return self.sparse_matrix(x**2 + (a**2 - 2*a*x) * common + a**2 * (common**2 - (y**2 @ z**2))).sum()
    
    def sparse_energy_affin(self, x, y, z, a, b):
        common = (y @ z)
        return self.sparse_matrix(x**2 + (a**2 - 2*a*x) * common + a**2 * (common**2 - (y**2 @ z**2)) + b**2 - 2*b * (x - a*common)).sum()

    def energy_binary_multi(self, x, y, z, a):
        matrix = (y.transpose(1, 2)@y)*(z@z.transpose(1, 2))
        up = torch.triu(matrix, diagonal=1)
        down = torch.tril(matrix, diagonal=-1)
        return torch.sum(x**2, dim=(1, 2)) + torch.sum((a.view(x.shape[0], 1, 1)**2 - 2*a.view(x.shape[0], 1, 1)*x) * (y@z), dim=(1,2)) \
        + a**2 * (up.sum(dim=(1,2)) + down.sum(dim=(1,2)))
    
    def diag_zero(self, y):
        return y*(1-torch.eye(y.size(0), dtype=y.dtype, device=y.device))
    
    def gradients(self, x, y, z, a):
        common = a**2 - 2*a*x 
        y_grad = common @ z.T + 2*a**2 * (y @ self.diag_zero(z @ z.T)) 
        z_grad = y.T @ common + 2*a**2 * (self.diag_zero(y.T @ y) @ z) 
        return y_grad, z_grad
    
    def affin_gradients(self, x, y, z, a, b):
        '''
        x: original matrix
        y: first binary matrix
        z: second binary matrix
        a: scaling factor
        b: bias
        '''
        common = a**2 - 2*a*x 
        y_grad = common @ z.T + 2*a**2 * (y @ self.diag_zero(z @ z.T)) + 2*a*b*z.sum(axis=1)
        z_grad = y.T @ common + 2*a**2 * (self.diag_zero(y.T @ y) @ z) + 2*a*b*y.sum(axis=0).unsqueeze(-1)
        # common2 = y@z
        # y_grad = (common + 2 * (a**2) * common2) @ z.T - 2*(a**2) * y * (z**2).sum(axis=1) + 2*a*b*z.sum(axis=1)
        # z_grad = y.T @ (common + 2 * (a**2) * common2) - 2*(a**2) * z * (y**2).sum(axis=0).unsqueeze(-1) + 2*a*b*y.sum(axis=0).unsqueeze(-1)
        return y_grad, z_grad
    
    def sparse_gradients(self, x, y, z, a):
        common = self.sparse_matrix(a**2 - 2*a*x).to_sparse()
        common_2 = self.sparse_matrix(2*a**2 * (y @ z)).to_sparse()
        not_nan_mask = torch.logical_not(self.nan_mask).float().to_sparse()
        y_grad = torch.sparse.mm(common, z.T) + torch.sparse.mm(common_2, z.T) - 2*a**2 * y * (torch.sparse.mm(not_nan_mask ,(z**2).T))
        z_grad = (torch.sparse.mm(common.T, y)).T + torch.sparse.mm(common_2.T, y).T - 2*a**2 * (torch.sparse.mm(not_nan_mask.T, y**2)).T * z
        return y_grad, z_grad 

    def sparse_matrix(self, x):
        # NaNの位置を0に置き換えます 
        return x.masked_fill(self.nan_mask, 0.0)
    
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
    
    def sopq_energy(self, x, y, z, a):
        a_dash = a.unsqueeze(-1).unsqueeze(-1)
        return torch.sum((x - torch.sum(a_dash*y@z, axis=0))**2) + torch.sum((a_dash**2)*(y@z - (y**2)@(z**2)))
    
    # sum-of-and quantization
    def soaq_energy(self, x, y, z, a):
        yz = y@z
        y_z = (1-y)@z
        yz_ = y@(1-z)
        y_z_ = (1-y)@(1-z)
        reconst = a[0]*yz + a[1]*y_z + a[2]*yz_ + a[3]*y_z_
        binarize_term = ((a[0])**2)*(yz - (y**2)@(z**2)) + ((a[1])**2)*(y_z - ((1-y)**2)@(z**2)) + ((a[2])**2)*(yz_ - (y**2)@((1-z)**2)) + ((a[3])**2)*(y_z_ - ((1-y)**2)@((1-z)**2))
        # reconst = (a[0]*y + a[1]*(1-y))@z + (a[2]*y+a[3]*(1-y))@(1-z) 
        # binarize_term = ((a[0])**2)*(y@z - (y**2)@(z**2)) + ((a[1])**2)*((1-y)@z - ((1-y)**2)@(z**2)) + ((a[2])**2)*(y@(1-z) - (y**2)@((1-z)**2)) + ((a[3])**2)*((1-y)@(1-z) - ((1-y)**2)@((1-z)**2))
        return torch.sum((x - reconst)**2 + binarize_term) 
    
    # quadratic quantization
    def qq_energy(self, x, y, z, a):
        yz = y@z
        reconst = a[0]*yz + a[1]*y.sum(axis=1) + a[2]*z.sum(axis=0) + a[3]
        binarize_term = ((a[0])**2)*(yz - (y**2)@(z**2)) + ((a[1])**2)*((y*(1-y)).sum(axis=1)) + ((a[2])**2)*((z*(1-z)).sum(axis=0)) #+ 2*a[0]*a[1]*(yz - (y**2)@z) + 2*a[0]*a[2]*(yz - y@(z**2))
        return torch.sum((x - reconst)**2 + binarize_term) 
    
    # sum-of-sum quantization
    def sum_quantization_energy(self, x, y, z, a):
        reconst = a*(y + z)
        return torch.sum((x - reconst)**2)
    
    # recurrent binary quantization
    def rbq_energy(self, x, y, z, a, n, m):
        return torch.sum((x - a*torch.linalg.matrix_power(y@y.T, n)@y@z@torch.linalg.matrix_power(z.T@z, m))**2)

    def channelwise_hadamard_sum(self, A):
        n, l, m = A.shape
        B = torch.zeros((n, n), dtype=A.dtype, device=A.device)
        
        for i in range(n):
            B[i, i] = A[i].sum() # チャネルiの行列の要素和
            
            for j in range(i, n):
                hadamard_sum = (A[i] * A[j]).sum()  # アダマール積の要素和
                B[i, j] = hadamard_sum
                B[j, i] = hadamard_sum  # 対称性を利用

        return B
    
    
    def a_hessian_and_grad_for_sq(self, x, y, z, a):
        y = y.clone().detach().requires_grad_()
        z = z.clone().detach().requires_grad_()

        # 関数の出力を計算
        energy = self.sum_quantization_energy(x, y, z, a)

        # 1階微分（勾配）
        grad_y, grad_z = torch.autograd.grad(energy, (y, z), create_graph=True)

        # 2階微分（ヘッセ行列）
        hessian_y = torch.autograd.functional.hessian(lambda y: self.sum_quantization_energy(x, y, z, a), y)
        hessian_z = torch.autograd.functional.hessian(lambda z: self.sum_quantization_energy(x, y, z, a), z)

        hessian_y, hessian_z = hessian_y.squeeze(1).squeeze(-1), hessian_z.squeeze(-2).squeeze(0)

        return grad_y, hessian_y, grad_z, hessian_z
    
    def a_hessian_and_grad_for_sopq(self, x, y, z, a):
        a = a.clone().detach().requires_grad_()
        # 関数の出力を計算
        energy = self.sopq_energy(x, y, z, a)
        
        # aに対する1階微分（勾配）
        grad = torch.autograd.grad(energy, a, create_graph=True)[0]
        
        # aに対するヘッセ行列（2階微分）
        hessian = []
        for i in range(a.shape[0]):
            # 2階微分を計算
            second_derivative = torch.autograd.grad(grad[i], a, retain_graph=True)[0]
            hessian.append(second_derivative)
        
        # ヘッセ行列をスタックして返す
        hessian_matrix = torch.stack(hessian)
        
        # 勾配とヘッセ行列を一緒に返す
        return grad, hessian_matrix
    
    def a_hessian_and_grad_for_soaq(self, x, y, z, a):
        a = a.clone().detach().requires_grad_()
        # 関数の出力を計算
        energy = self.soaq_energy(x, y, z, a)
        
        # aに対する1階微分（勾配）
        grad = torch.autograd.grad(energy, a, create_graph=True)[0]
        
        # aに対するヘッセ行列（2階微分）
        hessian = []
        for i in range(a.shape[0]):
            # 2階微分を計算
            second_derivative = torch.autograd.grad(grad[i], a, retain_graph=True)[0]
            hessian.append(second_derivative)
        
        # ヘッセ行列をスタックして返す
        hessian_matrix = torch.stack(hessian)
        
        # 勾配とヘッセ行列を一緒に返す
        return grad, hessian_matrix
    
    def a_hessian_and_grad(self, function, x, y, z, a):
        a = a.clone().detach().requires_grad_()
        # 関数の出力を計算
        energy = function(x, y, z, a)
        
        # aに対する1階微分（勾配）
        grad = torch.autograd.grad(energy, a, create_graph=True)[0]
        
        # aに対するヘッセ行列（2階微分）
        hessian = []
        for i in range(a.shape[0]):
            # 2階微分を計算
            second_derivative = torch.autograd.grad(grad[i], a, retain_graph=True)[0]
            hessian.append(second_derivative)
        
        # ヘッセ行列をスタックして返す
        hessian_matrix = torch.stack(hessian)
        
        # 勾配とヘッセ行列を一緒に返す
        return grad, hessian_matrix
    
    def augment_matrix_with_diagonal(self, B):
        n = B.shape[0]
        diag_elements = torch.diag(B)  # Bの対角成分を取得

        # 新しい (n+1, n+1) サイズの行列をゼロで初期化
        B_augmented = torch.zeros((n + 1, n + 1), dtype=B.dtype, device=B.device)

        # 元の行列 B を埋め込む
        B_augmented[:n, :n] = B

        # n+1列目に対角成分の列ベクトルを追加
        B_augmented[:n, n] = diag_elements

        # n+1行目に対角成分の行ベクトルを追加
        B_augmented[n, :n] = diag_elements

        # (n+1, n+1) 成分に 要素数N を設定
        B_augmented[n, n] = self.numel

        return B_augmented

    

    def run_binary(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        y = rng.random((self.Nrow, self.rank))
        z = rng.random((self.rank, self.Ncol))

        
        # 入力をGPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        # 注意！！！実行時はxを正規化しているけどself.xはそのまま！（self.xはnumpyです！）
        maximum = x.max() # 要素の最大が0の行列はエラー出ます！！
        x = x/maximum
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb
        matrix = (y.T@y)*(z@z.T)
        up = torch.triu(matrix, diagonal=1)
        down = torch.tril(matrix, diagonal=-1)
        a = (torch.sum(x * (y @ z))) / (torch.sum(y @ z) + up.sum() + down.sum())

        # 描画用のリスト
        H_list = []
        a_list = []
        for _ in tqdm(range(Nstep)):
            # 更新計算
            yf = y + zeta * (y - yb)
            zf = z + zeta * (z - zb)

            yf.requires_grad_(True)
            zf.requires_grad_(True)
            a.requires_grad_(True)
            H = self.energy_binary(x, yf, zf, a)
            H.backward()
            y_energy_grad = yf.grad
            z_energy_grad = zf.grad

            
            
            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone()
            zb = z.clone()
            y = ya.clone()
            z = za.clone()  

            # aの更新
            matrix = (y.T@y)*(z@z.T)
            up = torch.triu(matrix, diagonal=1)
            down = torch.tril(matrix, diagonal=-1)
            a = (torch.sum(x * (y @ z))) / (torch.sum(y @ z) + up.sum() + down.sum())
          
            
            # tempの減少
            temp -= self.delta_temp
            

            # エネルギーの計算
            H = self.energy_binary(x, y, z, a)

            # リスト追加
            H_list.append(H.item())
            a_list.append(a.item())

        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        H = torch.sum((x - a*(y @ z))**2)
        print('Final Energy', H)
        print('Confirm', self.energy_binary(x, y, z, a))
        return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy(), H_list, a_list
    
    def run_binary_light(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        y = rng.random((self.Nrow, self.rank))
        z = rng.random((self.rank, self.Ncol))

        
        # 入力をGPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        maximum = x.max()
        x = x/maximum
        
       
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb
        matrix = (y.T@y)*(z@z.T)
        up_sum = torch.triu(matrix, diagonal=1).sum()
        yz = y @ z
        a = (torch.sum(torch.nan_to_num(x) * (yz))) / (torch.sum(yz) + 2*up_sum)

        for _ in tqdm(range(Nstep)):
            # 更新計算
            yf = y + zeta * (y - yb)
            zf = z + zeta * (z - zb)

            y_energy_grad, z_energy_grad = self.gradients(x, yf, zf, a)
             

            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone()
            zb = z.clone()
            y = ya.clone()
            z = za.clone()  

            # aの更新
            matrix = (y.T@y)*(z@z.T)
            up_sum = torch.triu(matrix, diagonal=1).sum()
            yz = y @ z
            a = (torch.sum(torch.nan_to_num(x) * (yz))) / (torch.sum(yz) + 2*up_sum)        
            
            # tempの減少
            temp -= self.delta_temp
            
        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # aの更新
        matrix = (y.T@y)*(z@z.T)
        up_sum = torch.triu(matrix, diagonal=1).sum()
        yz = y @ z
        a = (torch.sum(torch.nan_to_num(x) * (yz))) / (torch.sum(yz) + 2*up_sum)
        H = torch.sum((x - a*(yz))**2)
        print('Final Energy', H)
        print('Confirm', self.energy_binary(x, y, z, a))
        return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy()
    
    def run_affin(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        y = rng.random((self.Nrow, self.rank))
        z = rng.random((self.rank, self.Ncol))

        
        # 入力をGPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        maximum = (x - x.min()).max()
        x = x/maximum
        N_element = torch.numel(x)
       
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb

        yz = y @ z
        b = x.min()
        matrix = (y.T@y)*(z@z.T)
        up_sum = torch.triu(matrix, diagonal=1).sum()
        yz = y @ z
        a = (torch.sum(torch.nan_to_num(x-b) * (yz))) / (torch.sum(yz) + 2*up_sum)  

        for _ in tqdm(range(Nstep)):
            # 更新計算
            yf = y + zeta * (y - yb)
            zf = z + zeta * (z - zb)

            y_energy_grad, z_energy_grad = self.affin_gradients(x, yf, zf, a, b)

            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone()
            zb = z.clone()
            y = ya.clone()
            z = za.clone()  

            # a, bの更新
            yz = y @ z
            b = torch.sum(x - a * yz) / N_element
            matrix = (y.T@y)*(z@z.T)
            up_sum = torch.triu(matrix, diagonal=1).sum()
            a = (torch.sum(torch.nan_to_num(x-b) * (yz))) / (torch.sum(yz) + 2*up_sum)  
            
            # tempの減少
            temp -= self.delta_temp
            
        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # aの更新
        yz = y @ z
        b = torch.sum(x - a * yz) / N_element
        matrix = (y.T@y)*(z@z.T)
        up_sum = torch.triu(matrix, diagonal=1).sum()
        a = (torch.sum(torch.nan_to_num(x-b) * (yz))) / (torch.sum(yz) + 2*up_sum)  
       
        print('Final Energy', torch.sum((x - a*(y @ z)-b)**2))
        return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy(), (maximum*b).detach().cpu().numpy()
    
    def run_binary_sparse(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPUデバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        y = rng.random((self.Nrow, self.rank))
        z = rng.random((self.rank, self.Ncol))
        
        # GPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        self.nan_mask = self.nan_mask.to(device)
        maximum = self.sparse_matrix(x).max()
        x = (x/maximum)
       
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb
        # aの更新
        matrix = (y.T@y)*(z@z.T)
        up_sum = torch.triu(matrix, diagonal=1).sum()
        yz = y @ z
        a = (torch.sum(torch.nan_to_num(x) * (yz))) / (torch.sum(yz) + 2*up_sum)

        for _ in tqdm(range(Nstep)):
            # 更新計算
            yf = (y + zeta * (y - yb)).requires_grad_(True)
            zf = (z + zeta * (z - zb)).requires_grad_(True)

            error = self.sparse_energy_binary(x.masked_fill(self.nan_mask, 0.0), yf, zf, a)
            error.backward()
            y_energy_grad = yf.grad
            z_energy_grad = zf.grad

            # y_energy_grad, z_energy_grad = self.sparse_gradients(x, yf, zf, a)

            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone()
            zb = z.clone()
            y = ya.clone()
            z = za.clone()  

            

            # aの更新
            yz = y @ z
            a = (torch.sum(self.sparse_matrix(x) * (yz))) / (self.sparse_matrix(yz).sum() + self.sparse_matrix(yz**2 - (y**2 @ z**2)).sum())

            # gradの初期化
            yf.grad.zero_()
            zf.grad.zero_()
          
            # tempの減少
            temp -= self.delta_temp
            
        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # aの更新
        yz = y @ z
        a = (torch.sum(self.sparse_matrix(x) * (yz))) / (self.sparse_matrix(yz).sum() + self.sparse_matrix(yz**2 - (y**2 @ z**2)).sum())

        H = torch.sum((x - a*(y @ z))**2)
        print('Final Energy', H)
        print('Confirm', self.energy_binary(x, y, z, a))
        return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy()
    

    def run_affin_sparse(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPUデバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        y = rng.random((self.Nrow, self.rank))
        z = rng.random((self.rank, self.Ncol))
        
        # GPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        self.nan_mask = self.nan_mask.to(device)
        maximum = (self.sparse_matrix(x) - self.sparse_matrix(x).min()).max()
        x = (x/maximum)
        N_element = self.nan_mask.sum()
       
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb
        # a, bの更新
        b = self.sparse_matrix(x).min()
        yz = y @ z
        a = (torch.sum(self.sparse_matrix(x-b) * (yz))) / (self.sparse_matrix(yz).sum() + self.sparse_matrix(yz**2 - (y**2 @ z**2)).sum())

        for _ in tqdm(range(Nstep)):
            # 更新計算
            yf = (y + zeta * (y - yb)).requires_grad_(True)
            zf = (z + zeta * (z - zb)).requires_grad_(True)

            error = self.sparse_energy_binary(x.masked_fill(self.nan_mask, 0.0), yf, zf, a)
            error.backward()
            y_energy_grad = yf.grad
            z_energy_grad = zf.grad

            # y_energy_grad, z_energy_grad = self.sparse_gradients(x, yf, zf, a)

            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone()
            zb = z.clone()
            y = ya.clone()
            z = za.clone()  

            # a, bの更新
            yz = y @ z
            b = torch.sum(self.sparse_matrix(x - a * yz)) / N_element
            a = (torch.sum(self.sparse_matrix(x-b) * (yz))) / (self.sparse_matrix(yz).sum() + self.sparse_matrix(yz**2 - (y**2 @ z**2)).sum())
            

            # gradの初期化
            yf.grad.zero_()
            zf.grad.zero_()
          
            # tempの減少
            temp -= self.delta_temp
            
        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)

        # a, bの更新
        yz = y @ z
        b = torch.sum(self.sparse_matrix(x - a * yz)) / N_element
        a = (torch.sum(self.sparse_matrix(x-b) * (yz))) / (self.sparse_matrix(yz).sum() + self.sparse_matrix(yz**2 - (y**2 @ z**2)).sum())

        return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy(), (maximum*b).detach().cpu().numpy()
    
    def run_binary_multi(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        batch_size, n, m = self.x.shape
        y = rng.random((batch_size, n, self.rank))
        z = rng.random((batch_size, self.rank, m))

        
        # 入力をGPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        # sigma = torch.sqrt((x**2).sum(dim=(1,2)))
        # x /= sigma.view(x.shape[0], 1, 1)
        maximum = x.max(dim=2)[0].max(dim=1)[0]
        x = x/maximum.view(x.shape[0], 1, 1)
        
        
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        
        # 初期化
        y = yb - eta * yb
        z = zb - eta * zb
        matrix = (y.transpose(1, 2)@y)*(z@z.transpose(1, 2))
        up = torch.triu(matrix, diagonal=1)
        down = torch.tril(matrix, diagonal=-1)
        a = ((x * (y @ z)).sum(dim=(1, 2))) / ((y @ z).sum(dim=(1, 2)) + up.sum(dim=(1, 2)) + down.sum(dim=(1, 2)))

        for _ in tqdm(range(Nstep)):
            # 更新計算
            yf = y + zeta * (y - yb)
            zf = z + zeta * (z - zb)

            y_energy_grad, z_energy_grad = self.gradients_3d(x, yf, zf, a)

            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # 前の状態を保持
            yb = y.clone()
            zb = z.clone()
            y = ya.clone()
            z = za.clone()  

            # aの更新
            matrix = ((y.transpose(1, 2)@y)*(z@z.transpose(1, 2)))

            up = torch.triu(matrix, diagonal=1) 
            # down = torch.tril(matrix, diagonal=-1) # up.sum()=down.sum()だから省略
            common = y @ z

            a = ((x * common).sum(dim=(1, 2))) / (common.sum(dim=(1, 2)) + 2*up.sum(dim=(1,2)))
          
            
            # tempの減少
            temp -= self.delta_temp
            

        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # aの更新
        matrix = ((y.transpose(1, 2)@y)*(z@z.transpose(1, 2)))
        up = torch.triu(matrix, diagonal=1) 
        common = y @ z
        a = ((x * common).sum(dim=(1, 2))) / (common.sum(dim=(1, 2)) + 2*up.sum(dim=(1,2)))
        # NaNを0に置き換え
        nannum_y = torch.isnan(y).sum().item()
        nannum_z = torch.isnan(z).sum().item()
        if (nannum_y > 0):
            y = torch.nan_to_num(y, nan=0.0)
            print('yにnanが{}個含まれていました'.format(nannum_y))
        if (nannum_z > 0):
            z = torch.nan_to_num(z, nan=0.0)
            print('zにnanが{}個含まれていました'.format(nannum_z))
        

        # print('Final Energy', H, self.energy_binary_multi(x, y, z, a))
        return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).view(x.shape[0], 1, 1).detach().cpu().numpy()
    
    
    
    def run_soaq(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1, output_type='numpy'):
        self.delta_temp = (Tinit - Tfin) / (Nstep - 1)
        temp = copy.copy(Tinit)
        # GPU デバイスを指定
        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(seed=seed)
        n, m = self.x.shape
        y = rng.random((n, self.rank))
        z = rng.random((self.rank, m))

        
        # 入力をGPUに転送
        x = torch.from_numpy(self.x).float().to(device)
        maximum = (x - x.min()).max()
        x = x/maximum
        
        
        yb = torch.from_numpy(y).float().to(device)
        zb = torch.from_numpy(z).float().to(device)
        
        
        # 初期化
        y = yb - eta * (yb - 0.5)
        z = zb - eta * (zb - 0.5)

        # aの初期値
        a = torch.randn(4, device=device)
        a_grad, a_hesse = self.a_hessian_and_grad_for_soaq(x=x, y=y, z=z, a=a)
        a = a - a_grad @ torch.linalg.inv(a_hesse)

        for _ in tqdm(range(Nstep)):
            # 前進点での計算
            yf = (y + zeta * (y - yb)).detach().requires_grad_()
            zf = (z + zeta * (z - zb)).detach().requires_grad_()
            a = a.detach().requires_grad_()
            

            # 勾配の計算
            energy = self.soaq_energy(x, yf, zf, a)
            energy.backward()

            y_energy_grad, z_energy_grad = yf.grad, zf.grad

            # yとzの更新
            y_entropy_grad = temp * (y - 0.5)
            z_entropy_grad = temp * (z - 0.5)

            ya = torch.clamp(torch.where((y<0.0) | (y>1.0), 2*y - yb - eta * y_entropy_grad, 2*y - yb  - eta * (y_energy_grad + y_entropy_grad)), 0, 1)
            za = torch.clamp(torch.where((z<0.0) | (z>1.0), 2*z - zb - eta * z_entropy_grad, 2*z - zb - eta * (z_energy_grad + z_entropy_grad)), 0, 1)

            # aの最適化
            a_grad, a_hesse = self.a_hessian_and_grad_for_soaq(x=x, y=ya, z=za, a=a)
            a = a - a_grad @ torch.linalg.inv(a_hesse)

            # 前の状態を保持
            yb = y.clone().detach()
            zb = z.clone().detach()
            y = ya.clone().detach()
            z = za.clone().detach()

            yf.grad.zero_()
            zf.grad.zero_()
          
            
            # tempの減少
            temp -= self.delta_temp

        y = torch.where(y>0.5, 1.0, 0.0)
        z = torch.where(z>0.5, 1.0, 0.0)
        # aの最適化
        a_grad, a_hesse = self.a_hessian_and_grad_for_soaq(x=x, y=y, z=z, a=a)
        a = a - a_grad @ torch.linalg.inv(a_hesse)

        nannum_y = torch.isnan(y).sum().item()
        nannum_z = torch.isnan(z).sum().item()
        if (nannum_y > 0):
            y = torch.nan_to_num(y, nan=0.0)
            print('yにnanが{}個含まれていました'.format(nannum_y))
        if (nannum_z > 0):
            z = torch.nan_to_num(z, nan=0.0)
            print('zにnanが{}個含まれていました'.format(nannum_z))
        
        if output_type == 'torch':
            return y, z, maximum*a
        else:
            return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy()
        

    def run_qq(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1, output_type='numpy'):
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
        a = - torch.linalg.solve(hesse, v)

        for _ in tqdm(range(Nstep)):
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
            a = - torch.linalg.solve(hesse, v)
    
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
        a = - torch.linalg.solve(hesse, v)


        if output_type == 'torch':
            return y, z, maximum*a
        else:
            return y.detach().cpu().numpy(), z.detach().cpu().numpy(), (maximum*a).detach().cpu().numpy()
        


    def run_qq_compile(self, zeta, eta, Tinit, Tfin, Nstep, device_id=0, seed=1, output_type='numpy'):

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

        # パーツの計算
        yz, y2z2, sigma_y, sigma_y2, sigma_z, sigma_z2 = y @ z, y**2 @ z**2, y.sum(axis=1).unsqueeze(1), (y**2).sum(axis=1).unsqueeze(1), z.sum(axis=0).unsqueeze(0), (z**2).sum(axis=0).unsqueeze(0) 

        # スケーリング係数と一次係数の初期値
        r0c0, r0c1, r0c2, r0c3, r1c1, r1c2, r1c3, r2c2, r2c3, r3c3 = (yz**2 + yz - y2z2).sum(), ((sigma_y + 1) * yz - y**2 @ z).sum(), ((1 + sigma_z) * yz - y @ z**2).sum(), yz.sum(), (sigma_y**2 + sigma_y - sigma_y2).sum() * m, (sigma_y * sigma_z).sum(), sigma_y.sum() * m, (sigma_z**2 + sigma_z - sigma_z2).sum() * n, sigma_z.sum() * n, num_element
        hesse = 2*torch.tensor([[r0c0, r0c1, r0c2, r0c3],
                    [r0c1, r1c1, r1c2, r1c3],
                    [r0c2, r1c2, r2c2, r2c3],
                    [r0c3, r1c3, r2c3, r3c3]], device=device)
        v = torch.tensor([(-2 * x * yz).sum(), (-2 * x * sigma_y).sum(), (-2 * x * sigma_z).sum(), coeff], device=device)
        
        # スケーリング係数の初期値 
        a = - torch.linalg.solve(hesse, v)

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
  
            return -torch.linalg.solve(hesse, v)

        a = compute_a(y, z)

        @torch.compile(mode="max-autotune", backend="inductor")
        def loop_body(y, z, yb, zb, a, temp):
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

        for _ in tqdm(range(Nstep)):
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

        
    

    def decompose_affin(self, zeta=15, eta=0.01, Tinit=0.5, Tfin=0, Nstep=None, device_id=0, seed=1):
        if Nstep is None:
            Nstep = self.rank * (self.Nrow + self.Ncol)
        if self.x.min() < 0:
            # 非負にする
            b = self.x.min()
            self.x -= b
        else:
            b = 0
        y, z, a = self.run_binary_light(zeta, eta, Tinit, Tfin, Nstep, device_id, seed)

        # self.xのバイアスを戻しておく
        self.x += b

        return y, z, a, b
    
    
    
    def decompose_affin_multi(self, zeta=15, eta=0.01, Tinit=0.5, Tfin=0, Nstep=None, device_id=0, seed=1, output_type='torch'):
        if Nstep is None:
            Nstep = 5*self.rank * (self.Nrow + self.Ncol)
        if self.x.min() < 0:
            # 非負にする(注意：チャネルごとにバイアスを求めているわけではなく、全チャネル共通のバイアスなのでスカラー値です)
            b = self.x.min()
            self.x -= b
        else:
            b = 0

        device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

        y, z, a = self.run_binary_multi(zeta, eta, Tinit, Tfin, Nstep, device_id, seed)
        # xの値を戻しておく
        self.x += b

        if output_type == 'torch':
            return torch.from_numpy(y).float().to(device), torch.from_numpy(z).float().to(device), torch.from_numpy(a).float().to(device), b
        elif output_type == 'numpy':
            return y, z, a, b
        else: raise ValueError('output_typeにtorchかnumpyを指定してください')



    def patchify(self, tensor, max_patch_size=256):
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




    def decompose_large_matrix(self, max_patch_size, zeta=0.5, eta=0.1, Tinit=0.5, Tfin=0, Nstep=None, device_id=0, seed=1, output_type='torch'):
        """
        大きな行列をパッチに分け、それぞれのパッチで行列分解を行い、復元する関数

        Args:
            max_patch_size: パッチの最大サイズ
            zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type: 行列分解に関するパラメータ

        Returns:
            元の形状に復元されたテンソル
        """
        # if self.x.min() < 0:
        #     b = self.x.min()
        # else:
        #     b = 0
        b = self.x.min()

        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.copy(self.x)
        self.x -= b
        biased_x = copy.copy(self.x)

        # テンソルをパッチに分割
        divided_tensor = self.patchify(torch.tensor(biased_x), max_patch_size=max_patch_size)
        # パッチから復元できているか確認
        # print(torch.tensor(biased_x) == self.unpatchify(divided_tensor, x_copy.shape))

        # パッチサイズと数を取得
        _, num_patches_h, num_patches_w, patch_height, patch_width = divided_tensor.shape

        # 分解結果を保存するリスト
        decomposed_patches = []

        # 各パッチで行列分解を実行
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = divided_tensor[:, i, j, :, :]
                self.x = copy.copy(patch)
                self.__init__(x=copy.copy(patch), rank_scale=rank_scale_copy)
                
                # 行列分解の実行
                y, z, a, _ = self.decompose_affin_multi(zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type)  # _は必ず0
                data = {'patch_row':i, 'patch_col':j, 'coeff':a, 'mat1':y, 'mat2':z, 'bias':b}
                decomposed_patches.append(data)
                
                # 分解結果を保存
                divided_tensor[:, i, j, :, :] = copy.copy(a * (y @ z) + b)

        # パッチから元の形状に復元
        reconstructed_tensor = self.unpatchify(divided_tensor, x_copy.shape)

        # 元のテンソルに戻す
        self.x = copy.copy(x_copy)
        return reconstructed_tensor, decomposed_patches
    


    def decompose_affin_large_matrix(self, max_patch_size, zeta=0.5, eta=0.1, Tinit=0.5, Tfin=0.001, Nstep=None, device_id=0, seed=1, output_type='torch'):
        """
        大きな行列をパッチに分け、それぞれのパッチで行列分解を行い、復元する関数

        Args:
            max_patch_size: パッチの最大サイズ
            zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type: 行列分解に関するパラメータ

        Returns:
            元の形状に復元されたテンソル
        """

        b = self.x.min()

        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.copy(self.x)
        self.x -= b
        biased_x = copy.copy(self.x)

        # テンソルをパッチに分割
        divided_tensor = self.patchify(torch.tensor(biased_x), max_patch_size=max_patch_size)
        # パッチから復元できているか確認
        # print(torch.tensor(biased_x) == self.unpatchify(divided_tensor, x_copy.shape))

        # パッチサイズと数を取得
        _, num_patches_h, num_patches_w, patch_height, patch_width = divided_tensor.shape

        # 分解結果を保存するリスト
        decomposed_patches = []

        # 各パッチで行列分解を実行
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = divided_tensor[:, i, j, :, :]
                self.x = copy.copy(patch)
                self.__init__(x=copy.copy(patch), rank_scale=rank_scale_copy)
                
                # 行列分解の実行
                y, z, a, b = self.run_affin(zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type)  # _は必ず0
                data = {'patch_row':i, 'patch_col':j, 'coeff':a, 'mat1':y, 'mat2':z, 'bias':b}
                decomposed_patches.append(data)
                
                # 分解結果を保存
                divided_tensor[:, i, j, :, :] = copy.copy(a * (y @ z) + b)

        # パッチから元の形状に復元
        reconstructed_tensor = self.unpatchify(divided_tensor, x_copy.shape)

        # 元のテンソルに戻す
        self.x = copy.copy(x_copy)
        return reconstructed_tensor, decomposed_patches
    

    def qq_large_matrix(self, max_patch_size, bit_width, save_name=None, zeta=0.5, eta=0.1, Tinit=0.5, Tfin=0.001, Nstep=10000, device_id=0, seed=1):
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

        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.copy(self.x)

        # テンソルをパッチに分割
        divided_tensor = self.patchify(torch.tensor(x_copy), max_patch_size=max_patch_size)

        # パッチサイズと数を取得
        _, num_patches_h, num_patches_w, patch_height, patch_width = divided_tensor.shape


        # 各パッチで行列分解を実行
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = divided_tensor[:, i, j, :, :]
                original_x = copy.copy(patch.squeeze(0))
                update_x = copy.copy(patch.squeeze(0))

                # 分解結果を保存するリスト
                decomposed_patches = []

                for bit_idx in range(bit_width):
                    self.__init__(x=copy.copy(update_x), rank_scale=rank_scale_copy)
                    # 行列分解の実行
                    y, z, a = self.run_soaq(zeta, eta, Tinit, Tfin, Nstep, device_id, seed)  # _は必ず0
                    reconst = (a[0]*y + a[1]*(1-y))@z + (a[2]*y+a[3]*(1-y))@(1-z)
                    update_x = update_x - torch.tensor(reconst)
                    if save:
                        data = {'patch_row':i, 'patch_col':j, 'coeff':a, 'mat1':y, 'mat2':z, 'bit_idx':bit_idx}
                        decomposed_patches.append(data)

                if save:
                    torch.save(decomposed_patches, (save_name + f'_row{i}_col{j}.pth'))
                
                # 分解結果を保存
                divided_tensor[:, i, j, :, :] = (original_x-update_x).clone().detach()

        # パッチから元の形状に復元
        reconstructed_tensor = self.unpatchify(divided_tensor, x_copy.shape)

        # 元のテンソルに戻す
        self.x = copy.copy(x_copy)
        return reconstructed_tensor



    def qq_worker_task(self, queue, result_list, rank_scale_copy, seed, zeta, eta, Tinit, Tfin, Nstep, device_id, bit_width, save_name=None):
        """ 各ワーカーが処理する行列分解タスク """
        if save_name is None:
            save = False
        else: save = True
        while not queue.empty():
            try:
                data = queue.get_nowait()
            except:
                break

            i, j, patch = data['i'], data['j'], data['patch']
            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            patch = patch.to(device)
            original_x = patch.squeeze(0).clone()
            update_x = patch.squeeze(0).clone()

            decomposed_patches = []
            for bit_idx in range(bit_width):
                decomp_instance = BinaryMatrixFactorization(x=update_x.clone(), rank_scale=rank_scale_copy)
                y, z, a = decomp_instance.run_soaq(zeta, eta, Tinit, Tfin, Nstep, device_id, seed, output_type='torch')
                reconst = (a[0]*y + a[1]*(1-y))@z + (a[2]*y+a[3]*(1-y))@(1-z)
                update_x -= reconst

                if save:
                    data = {'patch_row': i, 'patch_col': j, 'coeff': a, 'mat1': y, 'mat2': z, 'bit_idx': bit_idx}
                    decomposed_patches.append(data)

            if save:
                torch.save(decomposed_patches, f"{save_name}_row{i}_col{j}.pth")
            result_list.append({'i': i, 'j': j, 'reconstructed': (original_x - update_x).clone().detach().cpu()})
            queue.task_done()

    def qq_large_matrix_multi_worker(self, max_patch_size, bit_width, save_name=None, zeta=0.5, eta=0.1, Tinit=0.5, Tfin=0.001, Nstep=10000, seed=1, workers_per_gpu=2):
        """
        大きな行列をパッチに分割し、並列に行列分解を実行して復元
        """
        mp.set_start_method("spawn", force=True)
        rank_scale_copy = copy.copy(self.rank_scale)
        x_copy = copy.copy(self.x)
        divided_tensor = self.patchify(torch.tensor(x_copy), max_patch_size=max_patch_size)
        _, num_patches_h, num_patches_w, _, _ = divided_tensor.shape

        manager = mp.Manager()
        queue = manager.Queue()
        result_list = manager.list()

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = divided_tensor[:, i, j, :, :]
                queue.put({'i': i, 'j': j, 'patch': patch})
        
        num_gpus = torch.cuda.device_count()
        num_workers = min(mp.cpu_count(), num_gpus * workers_per_gpu)
        processes = []
        
        for worker_id in range(num_workers):
            device_id = worker_id % num_gpus
            # print('worker{0} is processing on the device{1}'.format(worker_id, device_id))
            p = mp.Process(target=self.qq_worker_task, args=(queue, result_list, rank_scale_copy, seed, zeta, eta, Tinit, Tfin, Nstep, device_id, bit_width, save_name))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        for data in result_list:
            i, j = data['i'], data['j']
            divided_tensor[:, i, j, :, :] = data['reconstructed']
        
        reconstructed_tensor = self.unpatchify(divided_tensor, x_copy.shape)
        self.x = copy.copy(x_copy)
        
        return reconstructed_tensor
