import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from quantizer import BinaryQuadraticQuantization as BQQ


def MSE(matrix1, matrix2):
    return ((matrix1 - matrix2)**2).mean()



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




def plot_tradeoff(matrix_list, bitwidth_list=[1, 2, 3, 4], rank_scale_list=[0.25, 0.5, 1, 1.5, 2], Nstep=50000, device_id=1):
    num_matrices = len(matrix_list)
    fig = plt.figure(figsize=(8 * (num_matrices+1), 28))  # 3行×N列
    records = []
    for col, matrix in tqdm(enumerate(matrix_list)):
        matrix = normalize_zero_mean_unit_var(matrix)
        ax1 = fig.add_subplot(3, num_matrices, col + 1)
        
        # --- 1. トレードオフ曲線 ---
        bqq_lists = {}
        for n in rank_scale_list:
            bqq_lists[f"bqq{n}_e_list"] = []
            bqq_lists[f"bqq{n}_m_list"] = []


        for rank_scale in rank_scale_list:
            stack = torch.zeros_like(matrix)
            for bw in range(1, int(bitwidth_list[-1]/rank_scale)+1):

                # BQQ
                print(f'running bqq...{bw} stacks')
                instance = BQQ(matrix - stack, rank_scale=rank_scale)
                y, z, a = instance.run_bqq_compile(zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005, Nstep=Nstep, device_id=device_id, seed=1, output_type='torch')
                reconst = a[0] * y @ z + a[1] * y.sum(1, keepdim=True) + a[2] * z.sum(0, keepdim=True) + a[3]
                stack += reconst.cpu()
                bqq_lists[f'bqq{rank_scale}_e_list'].append(MSE(matrix, stack))
                bqq_lists[f'bqq{rank_scale}_m_list'].append(((y.numel() + z.numel()) * bw + 32 * (3 * bw + 1)) / 8)
            

        dic = ['DeiT', 'ImageNet', 'Random', 'Distance', 'SIFT']

        # bqq_lists のキーをペアで処理
        for n in rank_scale_list:
            e_key = f"bqq{n}_e_list"
            m_key = f"bqq{n}_m_list"
            
            e_list = bqq_lists.get(e_key, [])
            m_list = bqq_lists.get(m_key, [])
            
            # 要素数が合っている前提
            for e, m in zip(e_list, m_list):
                records.append({'instance':dic[col], 'rank_scale': n, 'memory_size': m, 'mse': e.item()})
        # データフレームに変換
        df = pd.DataFrame(records)

        # CSVに保存
        df.to_csv(os.path.dirname(__file__)+'/results/bqq_effect_of_dimension.csv', index=False)



        # カラーマップから色を取得（暖色系の 'OrRd' を使用）
        cmap = plt.get_cmap('OrRd')
        colors = [cmap(i) for i in np.linspace(0.3, 1.0, len(rank_scale_list))]  # 0.3〜1.0の範囲で色を取得

        # プロット
        for n, color in zip(rank_scale_list, colors):
            ax1.plot(bqq_lists[f'bqq{n}_m_list'], bqq_lists[f'bqq{n}_e_list'],
                    label=f'BQQ-{n}', marker='o', color=color, linewidth=8, markersize=20)
            
        ax1.set_xlabel('Memory Size [Byte]', fontsize=30)
        ax1.set_ylabel('MSE', fontsize=30)


        ax1.set_title(dic[col], fontsize=50)
        if len(matrix_list) == col+1:
            ax1.legend(fontsize=40, loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax1.grid(True)

        # --- 2. ヒストグラム ---
        ax2 = fig.add_subplot(3, num_matrices, num_matrices + col + 1)
        ax2.hist(matrix.cpu().numpy().ravel(), bins=100, color='navy', alpha=0.7)
        ax2.set_title(' ', fontsize=40)
        ax2.set_xlabel('Value', fontsize=30)
        ax2.set_ylabel('Frequency', fontsize=30)

        # --- 3. 3Dプロット ---
        ax3 = fig.add_subplot(3, num_matrices, 2 * num_matrices + col + 1, projection='3d')
        data = matrix.cpu().numpy()
        n, m = data.shape
        X, Y = np.meshgrid(np.arange(m), np.arange(n))
        ax3.plot_surface(X, Y, data, cmap='viridis')
        ax3.set_xlabel("Col Index", fontsize=30)
        ax3.set_ylabel("Row Index", fontsize=30)
        ax3.set_zlabel("Value", fontsize=30)

    plt.tight_layout()
    return fig, df






if __name__ == '__main__':
    matrix_list = load_matrix_data()
    fig, df = plot_tradeoff(matrix_list, bitwidth_list=[2, 3, 4], Nstep=10000, device_id=0)
    fig.savefig(os.path.dirname(__file__)+'/results/dimension_effect.pdf')
    df.to_csv(os.path.dirname(__file__)+'/results/dimension_effect.csv')
