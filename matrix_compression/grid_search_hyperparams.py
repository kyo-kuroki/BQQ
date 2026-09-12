"""Parallel (batched) hyperparameter grid search for matrix-weighted BQQ.

Sweeps the annealing hyperparameters (zeta / eta / Tinit / Tfin / seed) of the
matrix-weighted multi-stack BQQ optimizer *in one batched anneal* by tiling the
target weight across the grid.  Ranks combos by the true Hessian-weighted output
error  tr((W - Wq) H (W - Wq)^T).

The weight W (out x in) and column metric M = H (in x in) come from the same
problem builders as comparison_of_optimization.py (synthetic / deit / llama), so
you can grid-search on a real DeiT/LLaMA layer (optionally cropped) or synthetic.

Examples
--------
# Synthetic 64x64, sweep zeta x eta:
python grid_search_hyperparams.py --model-family synthetic \
    --synthetic-in-features 64 --synthetic-out-features 64 \
    --zeta 1.5,2,3 --eta 0.05,0.1 --nstep 10000

# Real DeiT blocks.0.attn.proj cropped to 64x64:
HF_HUB_OFFLINE=1 python grid_search_hyperparams.py --model-family deit \
    --crop 64 --zeta 1.5,2,3 --eta 0.05,0.1,0.2 --tfin 0.005,0.001 --nstep 50000
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import torch

import comparison_of_optimization as co
from quantizer import BinaryQuadraticQuantization as BQQ


def _floats(s):
    return [float(v) for v in str(s).split(',') if v != '']


def _ints(s):
    return [int(v) for v in str(s).split(',') if v != '']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # problem
    ap.add_argument('--model-family', default='synthetic', choices=['synthetic', 'deit', 'llama'])
    ap.add_argument('--timm-model', default='deit_small_patch16_224')
    ap.add_argument('--module-name', default='blocks.0.attn.proj')
    ap.add_argument('--llama-model-name', default='meta-llama/Llama-2-7b-hf')
    ap.add_argument('--llama-nsamples', type=int, default=16)
    ap.add_argument('--llama-seqlen', type=int, default=512)
    ap.add_argument('--llama-dataset', default='wikitext2')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--synthetic-in-features', type=int, default=64)
    ap.add_argument('--synthetic-out-features', type=int, default=64)
    ap.add_argument('--synthetic-samples', type=int, default=2048)
    ap.add_argument('--crop', type=int, default=0,
                    help='If >0, crop the (real) W to a crop x crop top-left block '
                         '(and the matching input channels).')
    ap.add_argument('--crop-row', type=int, default=0)
    ap.add_argument('--crop-col', type=int, default=0)
    # bqq fixed knobs
    ap.add_argument('--bits', type=int, default=2, help='num_stack (bit width).')
    ap.add_argument('--rank-scale', type=float, default=1.0)
    ap.add_argument('--nstep', type=int, default=50000)
    ap.add_argument('--device-id', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0, help='seed for problem construction')
    # grid axes (comma-separated); omitted axes use the optimizer default
    ap.add_argument('--zeta', type=str, default='2.0')
    ap.add_argument('--eta', type=str, default='0.1')
    ap.add_argument('--tinit', type=str, default='0.1')
    ap.add_argument('--tfin', type=str, default='0.005')
    ap.add_argument('--grid-seed', type=str, default='1',
                    help='annealing init seeds to sweep (comma-separated).')
    ap.add_argument('--compile-mode', default='reduce-overhead')
    ap.add_argument('--output-csv', type=str,
                    default=str(Path(__file__).with_name('results') / 'grid_search_hyperparams.csv'))
    args = ap.parse_args()

    W, X, source = co.build_problem(
        seed=args.seed, batch_size=args.batch_size, image_size=args.image_size,
        module_name=args.module_name, force_synthetic=(args.model_family == 'synthetic'),
        synthetic_in_features=args.synthetic_in_features,
        synthetic_out_features=args.synthetic_out_features,
        synthetic_samples=args.synthetic_samples,
        model_family=args.model_family, timm_model=args.timm_model,
        llama_model_name=args.llama_model_name, llama_nsamples=args.llama_nsamples,
        llama_seqlen=args.llama_seqlen, llama_dataset=args.llama_dataset,
        device_id=args.device_id)

    if args.crop and args.crop > 0:
        c = args.crop
        r0, c0 = args.crop_row, args.crop_col
        W = W[r0:r0 + c, c0:c0 + c].contiguous()
        X = X[c0:c0 + c, :].contiguous()

    H = (X @ X.T).float()
    print(f'Source: {source}')
    print(f'W {tuple(W.shape)}, X {tuple(X.shape)}, H {tuple(H.shape)}')

    grid = {}
    if args.zeta:
        grid['zeta'] = _floats(args.zeta)
    if args.eta:
        grid['eta'] = _floats(args.eta)
    if args.tinit:
        grid['Tinit'] = _floats(args.tinit)
    if args.tfin:
        grid['Tfin'] = _floats(args.tfin)
    if args.grid_seed:
        grid['seed'] = _ints(args.grid_seed)

    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f'Grid: { {k: v for k, v in grid.items()} }  ->  {n_combos} combos')

    q = BQQ(x=W, rank_scale=args.rank_scale)
    results, best = q.grid_search_hyperparams(
        W, H, grid, num_stack=args.bits, rank_scale=args.rank_scale,
        Nstep=args.nstep, device_id=args.device_id, compile_mode=args.compile_mode,
        verbose=True)

    df = pd.DataFrame([{k: r[k] for k in ('base_idx', 'combo_idx', 'zeta', 'eta',
                                          'Tinit', 'Tfin', 'seed', 'error')}
                       for r in results])
    df['source'] = source
    df['weight_shape'] = str(tuple(W.shape))
    df['bits'] = args.bits
    df['nstep'] = args.nstep

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print('\n=== grid (sorted by error within each base) ===')
    print(df.to_string(index=False))
    b0 = best[0]
    default_row = df[(df.base_idx == 0)].iloc[-1]  # worst for reference
    print(f'\nBest for base 0: error={b0["error"]:.6e} at '
          f'zeta={b0["zeta"]}, eta={b0["eta"]}, Tinit={b0["Tinit"]}, '
          f'Tfin={b0["Tfin"]}, seed={b0["seed"]}')
    print(f'  (worst in grid: error={default_row["error"]:.6e})  '
          f'best/worst = {b0["error"]/default_row["error"]:.4f}')
    print(f'Saved results to: {out}')


if __name__ == '__main__':
    main()
