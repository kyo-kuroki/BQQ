"""
CV-specific BQQ utilities.
Core classes are in nn_compression/src/binary_quadratic_network.py.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm

# Re-export all shared classes/functions
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from bqq_modules import (  # noqa: F401
    BinaryQuadratic,
    BQQLinear,
    BQQLinearInference,
    SymQuantSTE,
    get_matrices,
    transform_A,
    merge_binary_quadratic,
    merge_binaryquadratic_recursive,
)


# ---------------------------------------------------------------------------
# CV-specific helpers (filesystem-based patch loading)
# ---------------------------------------------------------------------------

def _load_patches_from_dir(weights_dir, full_name, device=None):
    """Load patch files matching a layer name from a directory."""
    weight_list = []
    for file in os.listdir(weights_dir):
        if file.endswith('.pth') and (full_name in file) and ('row' in file):
            path = os.path.join(weights_dir, file)
            weight_list += torch.load(path, map_location=device)
    return weight_list


def replace_linear_with_bqq(model, weights_dir, bit_width, prefix='', device=None, show_tqdm=True):
    """Replace Linear layers with BQQLinear ({-1,+1} representation)."""
    iterator = tqdm(model.named_children()) if show_tqdm else model.named_children()
    for name, module in iterator:
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, nn.Linear):
            map_loc = device if device is not None else module.weight.device
            weight_list = _load_patches_from_dir(weights_dir, full_name, device=map_loc)
            A, Y, Z = get_matrices(weight_list, bit_width=bit_width)
            bqq = BQQLinear(2 * Y - 1, 2 * Z - 1, transform_A(A, l=Y.shape[-1]), bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(module, weights_dir, bit_width, prefix=full_name, show_tqdm=False, device=device)

    return model


