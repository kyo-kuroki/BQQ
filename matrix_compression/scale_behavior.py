import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from quantizer import BinaryQuadraticQuantization2 as BQQ


def MSE(matrix1, matrix2):
    return ((matrix1 - matrix2)**2).mean()

def svd_matrix(matrix, rank):
    U, s, VT = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :rank], s[:rank], VT[:rank, :]



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



def plot_scaling_coefficients(matrix, bitwidth_list=[1, 2, 3, 4], 
                              Nstep=50000, device_id=1, title='Matrix', save_dir='./results'):
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)

    matrix = normalize_zero_mean_unit_var(matrix)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    records = []

    rank_scale = 1
    a0_list, a1_list, a2_list, a3_list = [], [], [], []

    stack = torch.zeros_like(matrix)
    max_bw = int(bitwidth_list[-1])

    for bw in range(1, max_bw + 1):
        print(f"Running BQQ: bw={bw}, rank_scale={rank_scale}")
        instance = BQQ(matrix - stack, rank_scale=rank_scale)
        y, z, a = instance.run_bqq_compile(
            zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
            Nstep=Nstep, device_id=device_id, seed=1, output_type='torch'
        )

        a0_list.append(a[0].item())
        a1_list.append(a[1].item())
        a2_list.append(a[2].item())
        a3_list.append(a[3].item()/y.shape[1])

        # Update residual
        reconst = a[0] * y @ z + a[1] * y.sum(1, keepdim=True) + a[2] * z.sum(0, keepdim=True) + a[3]
        stack += reconst.cpu()

        records.append({
            'bitwidth': bw,
            'a0': a[0].item(),
            'a1': a[1].item(),
            'a2': a[2].item(),
            'a3': a[3].item()/y.shape[1]
        })


    x_vals = range(1, max_bw + 1)
    ax.plot(x_vals, a0_list, label='$r_i$', color='red', marker='o', linewidth=3)
    ax.plot(x_vals, a1_list, label='$s_i$', color='blue', marker='o',  linewidth=3)
    ax.plot(x_vals, a2_list, label='$t_i$', color='green', marker='o', linewidth=3)
    # ax.plot(x_vals, a3_list, label='$u_i$', color='orange', marker='o', linewidth=3)

    ax.set_xlabel("Greedy Iteration Step (bitwidth)", fontsize=16)
    ax.set_ylabel("Scaling Coefficient Value", fontsize=16)
    ax.set_title(f"{title}: Scaling Coefficient Transitions", fontsize=20)
    ax.grid(True)
    ax.legend(fontsize=14)

    plt.tight_layout()

    df = pd.DataFrame(records)
    return fig, df






if __name__ == '__main__':
    matrix_list = load_matrix_data()
    matrix = matrix_list[0] # select matrix you want to analyze
    fig, df = plot_scaling_coefficients(matrix, bitwidth_list=[1, 2, 3, 4, 5, 6], Nstep=10000, device_id=0)
    fig.savefig(os.path.dirname(__file__)+'/results/scale_behavior.pdf')
    df.to_csv(os.path.dirname(__file__)+'/results/scale_behavior.csv')

