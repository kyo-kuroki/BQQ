"""
LM-specific BQQ utilities.
Core classes are in nn_compression/src/binary_quadratic_network.py.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from .compressed_data import build_consolidated_index, build_patch_index, load_layer_patches
except ImportError:
    from compressed_data import build_consolidated_index, build_patch_index, load_layer_patches

# Re-export all shared classes/functions
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from bqq_modules import (  # noqa: F401
    BinaryQuadratic,
    HadamardBinaryQuadratic,
    get_matrices,
    merge_binary_quadratic,
    merge_binaryquadratic_recursive,
)


# ---------------------------------------------------------------------------
# LM-specific helpers (uses compressed_data module for patch loading)
# ---------------------------------------------------------------------------

def _load_layer_matrices(layer_name, patch_index, bit_width, map_location):
    patch_list = load_layer_patches(layer_name, patch_index, map_location=map_location)
    if not patch_list:
        return None
    return get_matrices(patch_list, bit_width=bit_width)


def replace_linear_with_bqq(model, weights_dir, bit_width, prefix='', device=None, show_tqdm=True, patch_index=None):
    if patch_index is None:
        patch_index = build_consolidated_index(weights_dir) or build_patch_index(weights_dir)

    iterator = tqdm(model.named_children()) if show_tqdm else model.named_children()
    for name, module in iterator:
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, nn.Linear):
            weight_key = f"{full_name}.weight"
            matrices = _load_layer_matrices(
                weight_key, patch_index, bit_width,
                map_location=device if device is not None else module.weight.device,
            )
            if matrices is None:
                print(f"  [WARN] No patches for {weight_key}, keeping original Linear")
                continue

            A, Y, Z = matrices
            bqq = BinaryQuadratic(Y, Z, A, bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_bqq(
                module, weights_dir, bit_width,
                prefix=full_name, show_tqdm=False, device=device, patch_index=patch_index,
            )

    return model


def replace_linear_with_hbqq(model, weights_dir, bit_width, prefix='', patch_index=None):
    if patch_index is None:
        patch_index = build_consolidated_index(weights_dir) or build_patch_index(weights_dir)

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if 'head' in full_name:
            print(f"Skipping {full_name}")
            continue

        if isinstance(module, nn.Linear):
            weight_key = f"{full_name}.weight"
            matrices = _load_layer_matrices(weight_key, patch_index, bit_width, map_location=module.weight.device)
            if matrices is None:
                continue

            A, Y, Z = matrices
            bqq = HadamardBinaryQuadratic(Y, Z, A, bias=module.bias)
            setattr(model, name, bqq)
        else:
            replace_linear_with_hbqq(module, weights_dir, bit_width, prefix=full_name, patch_index=patch_index)

    return model


def replace_weight(model, weights_dir, bit_width):
    patch_index = build_consolidated_index(weights_dir) or build_patch_index(weights_dir)

    for name, param in model.named_parameters():
        if 'head' in name:
            print(f"Skipping {name}")
            continue

        if 'norm' not in name and 'bias' in name:
            print(f"Replacing {name}")
            print('weight shape:', param.shape)

            matrices = _load_layer_matrices(name, patch_index, bit_width, map_location=param.device)
            if matrices is None:
                continue

            A, Y, Z = matrices
            bqq = BinaryQuadratic(Y, Z, A, bias=None)
            param.data.copy_(bqq.get_weight())

    return model
