"""Budget-preserving progressive bit-width allocation.

Shared by:
  * blockwise_quant.py -- per-LAYER allocation inside one (independently
    quantized) transformer block: earlier Linears in the block get fewer bits
    (their quantization error is absorbed by the later Linears, which are still
    full precision / re-tuned), later Linears get more.
  * e2e_quantize.py -- per-BLOCK allocation across the whole model, quantized
    front-to-back with KL fine-tuning between blocks.

Continuous bit-width
--------------------
BQQ storage per element is
    bits/elt = bit_width * rank * (n + m) / (n * m),   rank = round(rs * nm/(n+m))
             ~= bit_width * rank_scale                 (continuous, tiling-free)
so we keep the integer ``bit_width`` fixed and set ``rank_scale = eff_bits/bit_width``
to realise any continuous effective bit-width.

Allocation (budget-preserving, param-weighted)
----------------------------------------------
For items i = 0..N-1 in quantization order with param counts P_i (P = sum P_i):
    f_i   = (sum_{j>i} P_j) / P        # downstream fraction, decreasing in i
    fbar  = sum_i P_i f_i / P          # param-weighted mean
    eff_i = base + strength * (fbar - f_i)
For any ``strength`` this gives  sum_i P_i eff_i = base * P  exactly, i.e. the
total memory equals a uniform run. Values are then clamped to [lo, hi] and the
clamped mass is redistributed over the still-free items so the budget is re-hit
(best effort when the bounds are too tight to allow it).
"""

from typing import List, Optional, Sequence


def redistribute_bounded(vals: Sequence[float], weights: Sequence[float],
                         target_total: float, lo: float, hi: float,
                         iters: int = 200) -> List[float]:
    """Clamp ``vals`` to [lo, hi] while keeping sum(weights * vals) == target_total.

    Free (unclamped) items absorb the deficit uniformly (a constant shift, which
    preserves their relative order); anything pushed past a bound is clamped on
    the next pass. Best effort if the bounds make the budget infeasible.
    """
    v = [min(hi, max(lo, float(x))) for x in vals]
    n = len(v)
    for _ in range(iters):
        free = [not (v[i] <= lo or v[i] >= hi) for i in range(n)]
        cur = sum(weights[i] * v[i] for i in range(n))
        deficit = target_total - cur
        free_w = sum(weights[i] for i in range(n) if free[i])
        if free_w <= 0.0 or abs(deficit) < 1e-9:
            break
        delta = deficit / free_w
        moved = False
        for i in range(n):
            if free[i]:
                nv = min(hi, max(lo, v[i] + delta))
                if nv != v[i]:
                    moved = True
                v[i] = nv
        if not moved:
            break
    return v


def allocate_progressive(counts: Sequence[float], base: float, strength: float,
                         lo: float, hi: float) -> List[float]:
    """Return per-item values, increasing with depth, preserving sum(counts*v)==base*sum(counts)."""
    counts = [float(c) for c in counts]
    n = len(counts)
    total = float(sum(counts))
    if n == 0 or total <= 0.0:
        raise ValueError('allocate_progressive requires non-empty positive counts')

    downstream = [0.0] * n
    acc = 0.0
    for i in range(n - 1, -1, -1):
        downstream[i] = acc
        acc += counts[i]
    frac = [downstream[i] / total for i in range(n)]
    fbar = sum(counts[i] * frac[i] for i in range(n)) / total

    vals = [base + strength * (fbar - frac[i]) for i in range(n)]
    vals = redistribute_bounded(vals, counts, target_total=base * total, lo=lo, hi=hi)
    return vals


def build_bit_plan(keys: Sequence, counts: Sequence[float], *, bit_width: int,
                   base_rank_scale: float, strength: float, min_bits: float,
                   max_bits: float) -> dict:
    """Compute an effective-bit / rank_scale schedule over ordered ``keys``.

    Returns a dict with 'base_eff_bits', 'total_params', 'realized_avg_bits',
    and 'items': {key: {eff_bits, rank_scale, params, downstream_frac}} plus an
    ordered 'order' list of keys.
    """
    keys = list(keys)
    counts = [float(c) for c in counts]
    if len(keys) != len(counts):
        raise ValueError('keys and counts must have equal length')
    total = float(sum(counts))
    base_eff = float(bit_width) * float(base_rank_scale)
    eff = allocate_progressive(counts, base_eff, strength, min_bits, max_bits)

    downstream = [0.0] * len(keys)
    acc = 0.0
    for i in range(len(keys) - 1, -1, -1):
        downstream[i] = acc
        acc += counts[i]

    items = {}
    for i, k in enumerate(keys):
        items[k] = {
            'eff_bits': eff[i],
            'rank_scale': eff[i] / float(bit_width),
            'params': counts[i],
            'downstream_frac': downstream[i] / total if total > 0 else 0.0,
        }
    realized = sum(counts[i] * eff[i] for i in range(len(keys)))
    return {
        'bit_width': int(bit_width),
        'base_rank_scale': float(base_rank_scale),
        'strength': float(strength),
        'min_bits': float(min_bits),
        'max_bits': float(max_bits),
        'base_eff_bits': base_eff,
        'total_params': total,
        'realized_avg_bits': realized / total if total > 0 else 0.0,
        'order': keys,
        'items': items,
    }


def format_bit_plan_table(plan: dict, *, title: str = 'Progressive bit plan',
                          key_header: str = 'target', key_width: int = 28,
                          highlight_key=None) -> str:
    lines = [
        f"{title}: {len(plan['order'])} items, base={plan['base_eff_bits']:.4f} bits/param, "
        f"realized_avg={plan['realized_avg_bits']:.4f} bits/param "
        f"(bit_width={plan['bit_width']}, strength={plan['strength']:g}, "
        f"clamp=[{plan['min_bits']:g}, {plan['max_bits']:g}])",
        f"  {key_header:<{key_width}} {'params':>12} {'down_frac':>9} "
        f"{'eff_bits':>9} {'rank_scale':>10}",
    ]
    for k in plan['order']:
        info = plan['items'][k]
        mark = ' <--' if (highlight_key is not None and k == highlight_key) else ''
        ks = str(k)
        if len(ks) > key_width:
            ks = ks[:key_width - 1] + '…'
        lines.append(
            f"  {ks:<{key_width}} {int(info['params']):>12,} {info['downstream_frac']:>9.4f} "
            f"{info['eff_bits']:>9.4f} {info['rank_scale']:>10.4f}{mark}")
    return '\n'.join(lines)
