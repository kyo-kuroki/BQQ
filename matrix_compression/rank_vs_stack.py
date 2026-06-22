import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from quantizer import BinaryQuadraticQuantization as BQQ


def reconstruct_stack(y, z, a):
    a0 = a[:, 0].view(-1, 1, 1)
    a1 = a[:, 1].view(-1, 1, 1)
    a2 = a[:, 2].view(-1, 1, 1)
    a3 = a[:, 3].view(-1, 1, 1)
    return (a0 * torch.bmm(y, z)
            + a1 * y.sum(dim=2, keepdim=True)
            + a2 * z.sum(dim=1, keepdim=True)
            + a3).sum(dim=0)


def run(n=64, m=64, n_trials=5, Nstep=100000, device_id=0, save_dir=None):
    save_dir = save_dir or os.path.join(os.path.dirname(__file__), 'results', 'figs')
    os.makedirs(save_dir, exist_ok=True)

    configs = [
        ('N=1, rs=2',   1, 2.0),
        ('N=2, rs=1',   2, 1.0),
        ('N=4, rs=0.5', 4, 0.5),
    ]

    Ws = [torch.randn(n, m, generator=torch.Generator().manual_seed(t))
          for t in range(n_trials)]

    records = []
    for label, N, rs in configs:
        print(f'=== {label} (Nstack={N}, rank_scale={rs}) ===')
        for t, W in enumerate(Ws):
            inst = BQQ(W, rank_scale=rs, num_stack=N)
            rank = inst.rank
            y, z, a = inst.run_multibqq_compile(
                zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
                Nstep=Nstep, device_id=device_id, seed=1,
            )
            M = reconstruct_stack(y, z, a)
            mse = ((W.cuda(device_id) - M) ** 2).mean().item()
            yz_bits = N * rank * (n + m)
            coef_bits = N * 4 * 16
            total_bits = yz_bits + coef_bits
            records.append({
                'label': label, 'N': N, 'rank_scale': rs, 'rank': rank,
                'trial': t, 'mse': mse,
                'yz_bits': yz_bits, 'coef_bits': coef_bits, 'total_bits': total_bits,
            })
            print(f'  [trial {t}] rank={rank} MSE = {mse:.6f}')

    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, f'rank_vs_stack_{n}x{m}.csv')
    df.to_csv(csv_path, index=False)

    # Summary
    agg = df.groupby(['label', 'N', 'rank_scale', 'rank', 'total_bits'])['mse'].agg(['mean', 'std']).reset_index()
    print('\n--- Summary ---')
    print(agg.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [r['label'] for r in agg.to_dict('records')]
    means = agg['mean'].values
    stds = agg['std'].values
    colors = ['#4C72B0', '#C44E52', '#DD8452']

    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='black', linewidth=1.2, error_kw={'elinewidth': 1.5})
    for b, mu, row in zip(bars, means, agg.to_dict('records')):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f'{mu:.4f}\n({row["total_bits"]} bits)',
                ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('MSE', fontsize=14)
    ax.set_title(f'BQQ at fixed bit budget (~2 bits/element): rank vs stack tradeoff\n'
                 f'Gaussian {n}×{m}, {n_trials} trials, Nstep={Nstep}', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, f'rank_vs_stack_{n}x{m}.pdf')
    fig.savefig(pdf_path)
    print(f'\nSaved: {pdf_path}')
    print(f'Saved: {csv_path}')
    return fig, df


if __name__ == '__main__':
    run(n=64, m=64, n_trials=5, Nstep=100000, device_id=0)
