import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from quantizer import BinaryQuadraticQuantization as BQQ


def scalar_1bit_quantize(W):
    """
    1-bit scalar quantization (closed-form optimum), per matrix.

    W_hat = alpha * sign(W),   alpha* = argmin_alpha ||W - alpha * sign(W)||_F^2
                              = mean(|W|)

    f(alpha) = ||W||^2 - 2*alpha*sum(|W|) + alpha^2 * N is quadratic in alpha,
    so Newton's method converges in one step to the closed form.

    Accepts W of shape (n, m) or (B, n, m); returns (W_hat, alpha) with the
    leading batch dim preserved.
    """
    if W.ndim == 2:
        alpha = W.abs().mean()
    else:
        alpha = W.abs().mean(dim=(1, 2), keepdim=True)  # (B,1,1)
    return alpha * torch.sign(W), alpha


def bqq_1bit_batched(W_batch, rank_scale, Nstep, device_id, seed):
    """
    1-bit BQQ on a (B, n, m) batch in a single GPU call.

    rank_scale=1   -> rank = nm/(n+m)   (binary storage = nm bits, 1 bit/element)
    rank_scale<1   -> reduced rank, smaller bit budget
    """
    inst = BQQ(torch.zeros(W_batch.shape[1], W_batch.shape[2]), rank_scale=rank_scale)
    y, z, a = inst.run_bqq_compile_batched(
        W_batch, rank_scale=rank_scale,
        zeta=4, eta=0.06, Tinit=0.2, Tfin=0.005,
        Nstep=Nstep, device_id=device_id, seed=seed,
    )
    # y: (B, n, rank), z: (B, rank, m), a: (B, 4)
    a0 = a[:, 0].view(-1, 1, 1)
    a1 = a[:, 1].view(-1, 1, 1)
    a2 = a[:, 2].view(-1, 1, 1)
    a3 = a[:, 3].view(-1, 1, 1)
    reconst = (
        a0 * torch.bmm(y, z)
        + a1 * y.sum(dim=2, keepdim=True)
        + a2 * z.sum(dim=1, keepdim=True)
        + a3
    )
    rank = y.shape[2]
    return reconst.cpu(), rank


def run(n=512, m=512, n_trials=5, Nstep=100000, device_id=0, std=1.0, save_dir=None):
    save_dir = save_dir or os.path.join(os.path.dirname(__file__), 'results', 'figs')
    os.makedirs(save_dir, exist_ok=True)

    # --- Build batched Gaussian random matrices (one per trial), N(0, std^2) ---
    Ws = std * torch.stack([torch.randn(n, m, generator=torch.Generator().manual_seed(t))
                            for t in range(n_trials)], dim=0)  # (B, n, m)

    # --- Scalar 1-bit (closed form), batched ---
    Ws_scalar, alpha = scalar_1bit_quantize(Ws)
    mse_scalar = ((Ws - Ws_scalar) ** 2).mean(dim=(1, 2)).numpy()

    # --- BQQ 1-bit with rank = nm/(n+m) (rank_scale = 1) ---
    base_rank = round(n * m / (n + m))
    print(f'Running batched BQQ (rank={base_rank}): B={n_trials}, n={n}, m={m}, Nstep={Nstep}...')
    Ws_bqq, rank_full = bqq_1bit_batched(Ws, rank_scale=1.0,
                                          Nstep=Nstep, device_id=device_id, seed=1)
    mse_bqq = ((Ws - Ws_bqq) ** 2).mean(dim=(1, 2)).numpy()

    # --- BQQ 1-bit with rank reduced by 1 (offsets the 4-coefficient overhead) ---
    target_rank = base_rank - 1
    rank_scale_reduced = target_rank / base_rank
    print(f'Running batched BQQ (rank={target_rank}, rank_scale={rank_scale_reduced:.6f})...')
    Ws_bqq_m1, rank_reduced = bqq_1bit_batched(Ws, rank_scale=rank_scale_reduced,
                                                Nstep=Nstep, device_id=device_id, seed=1)
    mse_bqq_m1 = ((Ws - Ws_bqq_m1) ** 2).mean(dim=(1, 2)).numpy()
    assert rank_reduced == target_rank, f'unexpected rank {rank_reduced} != {target_rank}'

    # --- Bit-budget bookkeeping (scaling coefficients stored as fp16) ---
    coef_bits = 16
    bits_scalar = n * m + coef_bits                              # sign + 1 fp16
    bits_bqq_full = rank_full * (n + m) + 4 * coef_bits          # y,z + 4 fp16
    bits_bqq_m1 = rank_reduced * (n + m) + 4 * coef_bits

    records = []
    for t in range(n_trials):
        records.append({
            'trial': t, 'n': n, 'm': m,
            'mse_scalar': float(mse_scalar[t]),
            'mse_bqq_full': float(mse_bqq[t]),
            'mse_bqq_rank_minus1': float(mse_bqq_m1[t]),
            'alpha_scalar': float(alpha.view(-1)[t]),
            'bits_scalar': bits_scalar,
            'bits_bqq_full': bits_bqq_full,
            'bits_bqq_rank_minus1': bits_bqq_m1,
        })
        print(f'[trial {t}] scalar={mse_scalar[t]:.6f}  '
              f'BQQ(rank={rank_full})={mse_bqq[t]:.6f}  '
              f'BQQ(rank={rank_reduced})={mse_bqq_m1[t]:.6f}')
    df = pd.DataFrame(records)

    theo_scalar = (std ** 2) * (1.0 - 2.0 / np.pi)  # W ~ N(0,std^2): MSE = std^2 * (1 - 2/pi)

    means = [mse_scalar.mean(), mse_bqq.mean(), mse_bqq_m1.mean()]
    stds = [mse_scalar.std(ddof=1), mse_bqq.std(ddof=1), mse_bqq_m1.std(ddof=1)]
    labels = [
        f'Scalar 1-bit\n(α·sign(W))\n{bits_scalar:,} bits',
        f'BQQ 1-bit\nrank={rank_full}\n{bits_bqq_full:,} bits',
        f'BQQ 1-bit\nrank={rank_reduced} (−1)\n{bits_bqq_m1:,} bits',
    ]
    colors = ['#4C72B0', '#C44E52', '#DD8452']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors,
                  edgecolor='black', linewidth=1.2, error_kw={'elinewidth': 1.5})
    for b, mu in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f'{mu:.4f}', ha='center', va='bottom', fontsize=14)

    theo_label = (f'Scalar theoretical: σ²(1−2/π) = {theo_scalar:.4f}'
                  if std != 1.0 else f'Scalar theoretical: 1 − 2/π = {theo_scalar:.4f}')
    ax.axhline(theo_scalar, color='black', linestyle='--', linewidth=2.0,
               label=theo_label)
    ax.set_ylabel('MSE', fontsize=16)
    ax.set_title(f'1-bit quantization MSE: Gaussian {n}×{m}, σ={std}\n'
                 f'({n_trials} trials, mean ± std, BQQ Nstep={Nstep})', fontsize=13)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    suffix = f'{n}x{m}' if std == 1.0 else f'{n}x{m}_std{std:g}'
    pdf_path = os.path.join(save_dir, f'bqq_vs_scalar_1bit_{suffix}.pdf')
    csv_path = os.path.join(save_dir, f'bqq_vs_scalar_1bit_{suffix}.csv')
    fig.savefig(pdf_path)
    df.to_csv(csv_path, index=False)
    print(f'\nSaved figure : {pdf_path}')
    print(f'Saved CSV    : {csv_path}')
    print(f'\nMean MSE : scalar              = {means[0]:.6f} ± {stds[0]:.6f}  ({bits_scalar:,} bits)')
    print(f'Mean MSE : BQQ rank={rank_full}        = {means[1]:.6f} ± {stds[1]:.6f}  ({bits_bqq_full:,} bits)')
    print(f'Mean MSE : BQQ rank={rank_reduced} (−1) = {means[2]:.6f} ± {stds[2]:.6f}  ({bits_bqq_m1:,} bits)')
    print(f'Theory   : scalar              = {theo_scalar:.6f}  (1 − 2/π for W~N(0,1))')
    return fig, df


if __name__ == '__main__':
    run(n=512, m=512, n_trials=5, Nstep=100000, device_id=0)
