import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from quantizer import (
    BinaryQuadraticQuantization as BQQ,
    BinaryQuadraticQuantization as BQQ2,
    BinaryCodingQuantization as BCQ,
    UniformQuantization as UQ,
    VectorQuantization as VQ,
    LatticeVectorQuantization as LVQ,
    JPEG,
)
from lplr.compressors import direct_svd_quant, lplr, lplr_svd, iterative_lplr



# MSEの計算
def MSE(matrix1, matrix2):
    # NumPy 配列なら torch.Tensor に変換
    if isinstance(matrix1, np.ndarray):
        matrix1 = torch.from_numpy(matrix1.astype(np.float32))
    if isinstance(matrix2, np.ndarray):
        matrix2 = torch.from_numpy(matrix2.astype(np.float32))
    
    # torch.Tensor 同士で計算
    return torch.mean((matrix1 - matrix2) ** 2)






def load_matrix_data():
    matrix_list = []
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # ディレクトリ内のすべての .pt ファイルを取得
    pt_files = glob.glob(os.path.join(base_dir, '*.pt'))
    
    # 各ファイルをロードしてリストに追加
    for file_path in pt_files:
        matrix = torch.load(file_path)
        matrix_list.append(matrix)
        print(f"[INFO] Loaded: {os.path.basename(file_path)}")

    return matrix_list


    
def normalize_zero_mean_unit_var(matrix):
    mean = matrix.mean()
    std = matrix.std()  # 母標準偏差（分散 N 分の 1）
    return (matrix - mean) / std

def svd_matrix(matrix, rank):
    U, s, VT = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank], s[:rank], VT[:rank, :]

import numpy as np

def crop_to_max_power_of_two(matrices):
    """
    各行列を、1辺が2のべき乗になるように中央から切り出す。
    サイズは可能な限り最大のべき乗を選ぶ。
    
    Parameters:
        matrices: list of 2D numpy arrays
    
    Returns:
        cropped_matrices: list of 2D numpy arrays (正方行列)
    """
    cropped_matrices = []

    for mat in matrices:
        m, n = mat.shape
        # 使える最大の2のべき乗サイズを決定（正方形）
        side = 2 ** int(np.floor(np.log2(min(m, n))))

        # 中央から切り出し位置を計算
        start_row = (m - side) // 2
        start_col = (n - side) // 2

        cropped = mat[start_row:start_row + side, start_col:start_col + side]
        cropped_matrices.append(cropped)

    return cropped_matrices





def plot_tradeoff(matrix_list, bitwidth_list=[1, 2, 3, 4], Nstep=50000, device_id=1):
    num_matrices = len(matrix_list)
    fig = plt.figure(figsize=(8 * (num_matrices+1), 36))  # 4行×N列
    n_row = 4
    records = []

    for col, matrix in tqdm(enumerate(matrix_list)):
        matrix = normalize_zero_mean_unit_var(matrix)
        ax1 = fig.add_subplot(n_row, num_matrices, col + 1)
        
        # --- 1. トレードオフ曲線 ---
        q_e_list, q_m_list = [], []
        bcq_e_list, bcq_m_list = [], []
        qq_e_list, qq_m_list = [], []
        vq_e_list, vq_m_list = [], []
        vq4_e_list, vq4_m_list = [], []
        vq8_e_list, vq8_m_list = [], []
        lq_e_list, lq_m_list = [], []
        lplr_e_list, lplr_m_list = [], []
        jpeg_e_list, jpeg_m_list = [], []
        stack = torch.zeros_like(matrix)

        # SVD
        print(f'running svd...')
        svd_e_list = []
        svd_m_list = []
        for r in torch.linspace(bitwidth_list[0]*matrix.numel()/(32*(matrix.shape[0]+matrix.shape[1])), bitwidth_list[-1]*matrix.numel()/(32*(matrix.shape[0]+matrix.shape[1])), 5):
            r = int(r)
            U, s, V = svd_matrix(matrix, rank=r)
            approx_tensor = U@(torch.diag(s)@V)
            error = MSE(matrix, approx_tensor)
            svd_m_list.append((32 * (U.shape[0] * r + r * V.shape[1]) + 32*4)/8000)  # KB単位
            svd_e_list.append(error)
        ax1.plot(svd_m_list, svd_e_list, label=f'SVD', marker='p', color='purple', linewidth=8, markersize=20)



        # SVD + Quantization
        for bit in [4, 8]:
            print(f'running LPLR {bit}-bit...')
            svd_e_list = []
            svd_m_list = []
            for r in torch.linspace(bitwidth_list[0]*matrix.numel()/(bit*(matrix.shape[0]+matrix.shape[1])), bitwidth_list[-1]*matrix.numel()/(bit*(matrix.shape[0]+matrix.shape[1])), 5):
                r = int(r)
                # U, s, V = svd_matrix(matrix, rank=r)
                # Uq, Vq = UQ().run_uq(U, bit), UQ().run_uq(torch.diag(s)@V, bit)
                # approx_tensor = Uq@Vq
                approx_tensor = lplr_svd(X=matrix, r=r, B1=bit, B2=bit, normalize_and_shift=True)
                error = MSE(matrix, approx_tensor)
                svd_m_list.append((bit * (U.shape[0] * r + r * V.shape[1]) + 32*4)/8000)
                svd_e_list.append(error)
                break
            # if bit==8:
            #     ax1.plot(svd_m_list, svd_e_list, label=f'SVD + {bit}-bit UQ', marker='p', color='indigo', linewidth=8, markersize=20)
            # elif bit==4: ax1.plot(svd_m_list, svd_e_list, label=f'SVD + {bit}-bit UQ', marker='p', color='orchid', linewidth=8, markersize=20)
            # elif bit==2: ax1.plot(svd_m_list, svd_e_list, label=f'SVD + {bit}-bit UQ', marker='p', color='violet', linewidth=8, markersize=20)
            # elif bit==1: ax1.plot(svd_m_list, svd_e_list, label=f'SVD + {bit}-bit UQ', marker='p', color='mediumvioletred', linewidth=8, markersize=20)
            if bit==8:
                ax1.plot(svd_m_list, svd_e_list, label=f'LPLR {bit}-bit ', marker='p', color='indigo', linewidth=8, markersize=20)
            elif bit==4: ax1.plot(svd_m_list, svd_e_list, label=f'LPLR {bit}-bit', marker='p', color='orchid', linewidth=8, markersize=20)
            elif bit==2: ax1.plot(svd_m_list, svd_e_list, label=f'LPLR {bit}-bit', marker='p', color='violet', linewidth=8, markersize=20)
            elif bit==1: ax1.plot(svd_m_list, svd_e_list, label=f'LPLR {bit}-bit', marker='p', color='mediumvioletred', linewidth=8, markersize=20)


        for bw in (bitwidth_list):
            # UQ
            print('running uq...')
            quantized = UQ().run_uq(matrix, bw)
            q_e_list.append(MSE(quantized, matrix))
            q_m_list.append((bw * matrix.numel() + 32 * 2) / 8000)

            # BCQ
            print('running bcq...')
            quantized, _, _ = BCQ().run_bcq(matrix, bw)
            bcq_e_list.append(MSE(quantized.cpu(), matrix))
            bcq_m_list.append((bw * matrix.numel() + 32 * bw) / 8000)

            # BQQ
            print('running bqq...')
            # y, z, a = BQQ().run_bqq_compile(matrix - stack, rank_scale=1, zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, device_id=device_id, seed=1)
            BQQ2_instance = BQQ2(x=matrix - stack, rank_scale=1)
            y, z, a = BQQ2_instance.run_bqq_compile(zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, device_id=device_id, seed=1)
            reconst = a[0] * y @ z + a[1] * y.sum(1, keepdim=True) + a[2] * z.sum(0, keepdim=True) + a[3]
            stack += reconst.cpu()
            qq_e_list.append(MSE(matrix, stack))
            qq_m_list.append(((y.numel() + z.numel()) * bw + 32 * (3 * bw + 1)) / 8000)

            # VQ
            print('running vq...')
            num_centroid = int(matrix.shape[0] * (matrix.shape[1] * bw - 4)/(32*matrix.shape[1]))
            quantized = VQ().run_vq(matrix, num_centroid=num_centroid)
            vq_e_list.append(MSE(matrix, quantized))
            vq_m_list.append(VQ().calc_memory_size(matrix, num_centroid)/1000)

            # VQ + 4bit UQ
            print('running 4vq...')
            num_centroid = int(matrix.shape[0] * (matrix.shape[1] * bw - 4)/(4*matrix.shape[1]))
            quantized = VQ().run_vq(matrix, num_centroid=num_centroid, centroid_bits=4)
            vq4_e_list.append(MSE(matrix, quantized))
            vq4_m_list.append(VQ().calc_memory_size(matrix, num_centroid, centroid_bits=4)/1000)

            # VQ + 8bit UQ
            print('running 8vq...')
            num_centroid = int(matrix.shape[0] * (matrix.shape[1] * bw - 4)/(8*matrix.shape[1]))
            quantized = VQ().run_vq(matrix, num_centroid=num_centroid, centroid_bits=8)
            vq8_e_list.append(MSE(matrix, quantized))
            vq8_m_list.append(VQ().calc_memory_size(matrix, num_centroid, centroid_bits=8)/1000)
            

            # LVQ
            print('running lvq...')
            quantized = LVQ().run_e8_lvq(matrix, n_bits=bw, scale_bits=1)
            lq_e_list.append(MSE(matrix, quantized))
            lq_m_list.append(LVQ().calc_memory_size(quantized, n_bits=bw, scale_bits=1)/1000)

            # LPLR
            print('running lplr...')
            r = int(matrix.numel()/((matrix.shape[0]+matrix.shape[1])))
            quantized = lplr_svd(matrix, r=r , B1=bw, B2=bw, normalize_and_shift=True)
            lplr_e_list.append(MSE(matrix, quantized))
            lplr_m_list.append((bw * r * (matrix.shape[0]+matrix.shape[1]) + 32*4)/8000)


            # JPEG
            print('running jpeg...')
            quantized, memory_size = JPEG().run_jpeg_compress(matrix, n_bits=bw)
            jpeg_e_list.append(MSE(matrix, quantized))
            jpeg_m_list.append(memory_size/1000)


        dic = ['DeiT','ImageNet','Random', 'Distance', 'SIFT']

        ax1.plot(lplr_m_list, lplr_e_list, label='LPLR', marker='x', color='magenta', linewidth=8, markersize=20)
        ax1.plot(q_m_list, q_e_list, label='UQ', marker='s', color='black', linewidth=8, markersize=20)
        ax1.plot(bcq_m_list, bcq_e_list, label='BCQ', marker='^', color='blue', linewidth=8, markersize=20)
        ax1.plot(vq_m_list, vq_e_list, label='VQ', marker='<', color='orange', linewidth=8, markersize=20)
        ax1.plot(vq4_m_list, vq4_e_list, label='VQ + 4-bit UQ', marker='<', color='coral', linewidth=8, markersize=20)
        ax1.plot(vq8_m_list, vq8_e_list, label='VQ + 8-bit UQ', marker='<', color='saddlebrown', linewidth=8, markersize=20)
        ax1.plot(lq_m_list, lq_e_list, label='LVQ', marker='D', color='green', linewidth=8, markersize=20)
        ax1.plot(jpeg_m_list, jpeg_e_list, label='8-bit UQ + JPEG', marker='+', color='darkturquoise', linewidth=8, markersize=20)
        ax1.plot(qq_m_list, qq_e_list, label='BQQ', marker='o', color='r', linewidth=8, markersize=20)
        ax1.set_xlabel('Memory Size [KB]', fontsize=45, labelpad=20)
        ax1.set_ylabel('MSE', fontsize=45, labelpad=20)


        ax1.set_title(dic[col], fontsize=75, pad=40)
        if len(matrix_list) == col+1:
            ax1.legend(fontsize=40, loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.grid(True)

        ax1.tick_params(axis='x', labelsize=35)  # x軸目盛りのフォントサイズ
        ax1.tick_params(axis='y', labelsize=35)  # y軸目盛りのフォントサイズ

        # --- 2. ヒストグラム ---
        ax2 = fig.add_subplot(n_row, num_matrices, num_matrices + col + 1)
        ax2.hist(matrix.cpu().numpy().ravel(), bins=100, color='navy', alpha=0.7)
        
        # ax2.set_title(f'Matrix {col+1}: Histogram', fontsize=20)
        ax2.set_title(' ', fontsize=60)
        ax2.set_xlabel('Value', fontsize=45, labelpad=20)
        ax2.set_ylabel('Frequency', fontsize=45, labelpad=20)

        ax2.tick_params(axis='x', labelsize=35)  # x軸目盛りのフォントサイズ
        ax2.tick_params(axis='y', labelsize=35)  # y軸目盛りのフォントサイズ

        # ---3. 特異値ヒストグラム ---
        ax3 = fig.add_subplot(n_row, num_matrices, 2 * num_matrices + col + 1)
        # singular value
        U, s, V = svd_matrix(matrix, rank=max(matrix.shape))
        ax3.hist(s.cpu().numpy().ravel(), bins=100, color='firebrick', alpha=0.7)
        ax3.set_xlabel('Singular Value', fontsize=45, labelpad=20)
        ax3.set_ylabel('Frequency', fontsize=45, labelpad=20)
        # ax3.set_xscale('log')
        ax3.tick_params(axis='x', labelsize=35)  # x軸目盛りのフォントサイズ
        ax3.tick_params(axis='y', labelsize=35)  # y軸目盛りのフォントサイズ

        # --- 4. 3Dプロット ---
        ax4 = fig.add_subplot(n_row, num_matrices, 3 * num_matrices + col + 1, projection='3d')

        data = matrix.cpu().numpy()
        n, m = data.shape
        X, Y = np.meshgrid(np.arange(m), np.arange(n))
        ax4.plot_surface(X, Y, data, cmap='viridis')
        ax4.set_xlabel("Col Index", fontsize=45, labelpad=40)
        ax4.set_ylabel("Row Index", fontsize=45, labelpad=40)
        ax4.set_zlabel("Value", fontsize=45, labelpad=40)
        ax4.tick_params(axis='x', labelsize=35)  # x軸目盛りのフォントサイズ
        ax4.tick_params(axis='y', labelsize=35)  # y軸目盛りのフォントサイズ
        plt.tight_layout()

        ################## dataframe の作成 #####################

        # カテゴリ名（タイトル用） → 適切に col を定義しておくこと
        dic = ['DeiT', 'ImageNet','Random','Distance', 'SIFT']

        # プロットに使った全データセットを整理
        plot_data = [
            ('UQ', q_m_list, q_e_list),
            ('BCQ', bcq_m_list, bcq_e_list),
            ('VQ', vq_m_list, vq_e_list),
            ('VQ + 4-bit UQ', vq4_m_list, vq4_e_list),
            ('VQ + 8-bit UQ', vq8_m_list, vq8_e_list),
            ('LVQ', lq_m_list, lq_e_list),
            ('8-bit UQ + JPEG', jpeg_m_list, jpeg_e_list),
            ('BQQ', qq_m_list, qq_e_list),
        ]

        # レコード化
        for method, m_list, e_list in plot_data:
            for m, e in zip(m_list, e_list):
                records.append({
                    'Method': method,
                    'Memory': m,
                    'MSE': e.item(),
                    'Category': dic[col]
                })

        # DataFrame化してCSV出力
        df = pd.DataFrame(records)

        print("CSV出力完了: tradeoff_results.csv")


    return fig, df




if __name__ == '__main__':
    matrix_list = load_matrix_data()
    fig, df = plot_tradeoff(matrix_list, bitwidth_list=[1, 2, 3, 4], Nstep=50000, device_id=0)
    df.to_csv(os.path.dirname(__file__)+'/results/tradeoff.csv', index=False)
    fig.savefig(os.path.dirname(__file__)+'/results/tradeoff.pdf')
