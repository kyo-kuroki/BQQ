import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats

from quantizer import BinaryQuadraticQuantization as BQQ


def main(n=512, m=512, Nstep=100000, device_id=0):
    save_dir = os.path.join(os.path.dirname(__file__), 'results', 'figs')
    os.makedirs(save_dir, exist_ok=True)

    # ---- Single Gaussian matrix ----
    torch.manual_seed(0)
    W = torch.randn(n, m)

    # ---- Scalar 1-bit residual ----
    alpha = W.abs().mean()
    W_scalar = alpha * torch.sign(W)
    R_scalar = (W - W_scalar).numpy().ravel()

    # ---- BQQ 1-bit residual ----
    inst = BQQ(W, rank_scale=1)
    y, z, a = inst.run_bqq_compile(
        zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
        Nstep=Nstep, device_id=device_id, seed=1, output_type='torch',
    )
    W_bqq = (a[0] * (y @ z)
             + a[1] * y.sum(1, keepdim=True)
             + a[2] * z.sum(0, keepdim=True)
             + a[3]).cpu()
    R_bqq = (W - W_bqq).numpy().ravel()

    # ---- Stats ----
    def report(name, r):
        mu, sd = r.mean(), r.std()
        kurt = stats.kurtosis(r)         # Fisher (Normal=0)
        skew = stats.skew(r)
        return mu, sd, kurt, skew

    print(f'{"":15s} {"mean":>9s} {"std":>9s} {"kurt":>9s} {"skew":>9s}')
    for name, r in [('original W', W.numpy().ravel()),
                    ('R_scalar', R_scalar),
                    ('R_BQQ', R_bqq)]:
        mu, sd, kurt, skew = report(name, r)
        print(f'{name:15s} {mu:9.4f} {sd:9.4f} {kurt:9.4f} {skew:9.4f}')

    # ---- Plot: 1 row x 3 cols: hist (scalar), hist (bqq), QQ plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Hist scalar residual ---
    ax = axes[0]
    ax.hist(R_scalar, bins=100, density=True, alpha=0.65, color='#4C72B0',
            edgecolor='black', linewidth=0.3, label='R_scalar')
    sd_s = R_scalar.std()
    xs = np.linspace(R_scalar.min(), R_scalar.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, 0, sd_s), 'r-', linewidth=2,
            label=f'N(0, {sd_s**2:.3f})')
    ax.set_title(f'Scalar 1-bit residual\nkurtosis = {stats.kurtosis(R_scalar):.3f}',
                 fontsize=14)
    ax.set_xlabel('residual value', fontsize=12)
    ax.set_ylabel('density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Hist BQQ residual ---
    ax = axes[1]
    ax.hist(R_bqq, bins=100, density=True, alpha=0.65, color='#C44E52',
            edgecolor='black', linewidth=0.3, label='R_BQQ')
    sd_b = R_bqq.std()
    xs = np.linspace(R_bqq.min(), R_bqq.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, 0, sd_b), 'r-', linewidth=2,
            label=f'N(0, {sd_b**2:.3f})')
    ax.set_title(f'BQQ 1-bit residual\nkurtosis = {stats.kurtosis(R_bqq):.3f}',
                 fontsize=14)
    ax.set_xlabel('residual value', fontsize=12)
    ax.set_ylabel('density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- QQ plot (both overlaid against N(0,1) after standardization) ---
    ax = axes[2]
    for r, name, color in [(R_scalar, 'R_scalar', '#4C72B0'),
                           (R_bqq, 'R_BQQ', '#C44E52')]:
        q_emp = np.sort((r - r.mean()) / r.std())
        q_theo = stats.norm.ppf((np.arange(1, len(q_emp) + 1) - 0.5) / len(q_emp))
        ax.plot(q_theo, q_emp, '.', markersize=2, color=color, alpha=0.5,
                label=name)
    lim = max(abs(q_theo).max(), 5)
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1.2,
            label='y=x (Gaussian)')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_xlabel('Theoretical quantile (standard normal)', fontsize=12)
    ax.set_ylabel('Empirical quantile (standardized)', fontsize=12)
    ax.set_title('QQ plot vs standard normal', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle(f'1-bit quantization residual distribution: Gaussian {n}×{m}',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    pdf_path = os.path.join(save_dir, f'residual_distribution_{n}x{m}.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'\nSaved: {pdf_path}')
    return fig


if __name__ == '__main__':
    main(n=512, m=512, Nstep=100000, device_id=0)
