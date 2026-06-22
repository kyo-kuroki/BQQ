import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from quantizer import BinaryQuadraticQuantization as BQQ


def reconstruct_bqq_stack(y, z, a):
    """y: (N, n, rank), z: (N, rank, m), a: (N, 4) -> reconstruct (n, m)."""
    a0 = a[:, 0].view(-1, 1, 1)
    a1 = a[:, 1].view(-1, 1, 1)
    a2 = a[:, 2].view(-1, 1, 1)
    a3 = a[:, 3].view(-1, 1, 1)
    return (a0 * torch.bmm(y, z)
            + a1 * y.sum(dim=2, keepdim=True)
            + a2 * z.sum(dim=1, keepdim=True)
            + a3).sum(dim=0)


def run(n=64, m=64, n_trials=5, N_max=5, Nstep_base=10000,
        device_id=0, save_dir=None):
    save_dir = save_dir or os.path.join(os.path.dirname(__file__), 'results', 'figs')
    os.makedirs(save_dir, exist_ok=True)

    Ws = [torch.randn(n, m, generator=torch.Generator().manual_seed(t))
          for t in range(n_trials)]

    records = []
    for N in range(1, N_max + 1):
        Nstep = N * Nstep_base
        print(f'=== N={N}, Nstep={Nstep} ===')

        for t, W in enumerate(Ws):
            W_dev = W.cuda(device_id)

            # --- Joint multi-BQQ ---
            inst = BQQ(W, rank_scale=1, num_stack=N)
            y, z, a = inst.run_multibqq_compile(
                zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
                Nstep=Nstep, device_id=device_id, seed=1,
            )
            M_joint = reconstruct_bqq_stack(y, z, a)
            mse_joint = ((W_dev - M_joint) ** 2).mean().item()

            # --- Greedy residual stacking (1-bit BQQ × N times) ---
            residual = W.clone()
            M_greedy = torch.zeros_like(W)
            for b in range(N):
                inst_g = BQQ(residual, rank_scale=1)
                y_g, z_g, a_g = inst_g.run_bqq_compile(
                    zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
                    Nstep=Nstep_base, device_id=device_id, seed=b + 1,
                    output_type='torch',
                )
                recon_g = (a_g[0] * (y_g @ z_g)
                           + a_g[1] * y_g.sum(1, keepdim=True)
                           + a_g[2] * z_g.sum(0, keepdim=True)
                           + a_g[3]).cpu()
                M_greedy += recon_g
                residual = W - M_greedy
            mse_greedy = ((W - M_greedy) ** 2).mean().item()

            records.append({
                'N': N, 'Nstep_joint': Nstep, 'Nstep_per_bit_greedy': Nstep_base,
                'trial': t,
                'mse_joint': mse_joint,
                'mse_greedy': mse_greedy,
            })
            print(f'  [trial {t}] joint={mse_joint:.6f}  greedy={mse_greedy:.6f}')

    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, f'multibqq_vs_greedy_{n}x{m}.csv')
    df.to_csv(csv_path, index=False)

    # Aggregate
    agg = df.groupby('N').agg(
        joint_mean=('mse_joint', 'mean'), joint_std=('mse_joint', 'std'),
        greedy_mean=('mse_greedy', 'mean'), greedy_std=('mse_greedy', 'std'),
    ).reset_index()
    print('\n--- Summary ---')
    print(agg.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6.5))
    Ns = agg['N'].values
    ax.errorbar(Ns, agg['mse_greedy'], yerr=agg['greedy_std'],
                marker='o', markersize=10, color='#4C72B0', linewidth=2,
                capsize=5, label='Greedy residual stacking (1-bit BQQ × N)')
    ax.errorbar(Ns, agg['mse_joint'], yerr=agg['joint_std'],
                marker='s', markersize=10, color='#C44E52', linewidth=2,
                capsize=5, label='Joint multi-BQQ (run_multibqq_compile)')
    ax.set_yscale('log')
    ax.set_xticks(Ns)
    ax.set_xlabel('Bit budget N (number of stacks)', fontsize=14)
    ax.set_ylabel('MSE (log scale)', fontsize=14)
    ax.set_title(f'Joint vs greedy multi-bit BQQ on Gaussian {n}×{m}\n'
                 f'(N trials = {n_trials}, equal compute: Nstep_joint = N·{Nstep_base})',
                 fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()

    pdf_path = os.path.join(save_dir, f'multibqq_vs_greedy_{n}x{m}.pdf')
    fig.savefig(pdf_path)
    print(f'\nSaved: {pdf_path}')
    print(f'Saved: {csv_path}')
    return fig, df


if __name__ == '__main__':
    run(n=64, m=64, n_trials=5, N_max=5, Nstep_base=10000, device_id=0)
