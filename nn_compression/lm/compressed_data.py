from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
from typing import DefaultDict

import torch


LM_DIR = Path(__file__).resolve().parent
NN_COMPRESSION_DIR = LM_DIR.parent
BQQ_ROOT = NN_COMPRESSION_DIR.parent


def ensure_bqq_root_on_path() -> None:
    root = str(BQQ_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def model_basename(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def default_compressed_data_dir(model_name: str, group_size: int, num_steps: int) -> Path:
    return LM_DIR / "bqq_compressed_data" / f"{model_basename(model_name)}-{group_size}gs-{num_steps}step"


def default_quantized_model_dir(model_name: str) -> Path:
    return LM_DIR / "quantized_model_data" / model_basename(model_name)


def default_results_dir() -> Path:
    return LM_DIR / "results"


def build_patch_index(weights_dir: str | Path) -> dict[str, list[Path]]:
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        raise FileNotFoundError(f"Compressed data directory does not exist: {weights_path}")

    index: DefaultDict[str, list[Path]] = defaultdict(list)
    for path in sorted(weights_path.glob("*.pth")):
        if "_row" not in path.stem:
            continue
        layer_name = path.stem.rsplit("_row", 1)[0]
        index[layer_name].append(path)

    return dict(index)


def load_layer_patches(
    layer_name: str,
    patch_index: dict[str, list[Path]],
    map_location: str | torch.device = "cpu",
) -> list[dict]:
    patch_list: list[dict] = []
    for path in patch_index.get(layer_name, []):
        patch_list.extend(torch.load(path, weights_only=False, map_location=map_location))
    return patch_list


def aggregate_matrices(patch_list: list[dict], output_shape: tuple[int, ...], bit_width: int) -> torch.Tensor:
    if not patch_list:
        raise ValueError("patch_list must not be empty")

    dtype = patch_list[0]["coeff"].dtype
    device = patch_list[0]["coeff"].device
    result_matrix = torch.zeros(output_shape, dtype=dtype, device=device)

    for patch in patch_list:
        i, j = patch["patch_row"], patch["patch_col"]
        a, y, z, bit_idx = patch["coeff"], patch["mat1"], patch["mat2"], patch["bit_idx"]

        if bit_idx >= bit_width:
            continue

        patch_result = a[0] * y @ z
        patch_result += a[1] * y.sum(axis=1, keepdim=True)
        patch_result += a[2] * z.sum(axis=0, keepdim=True)
        patch_result += a[3]

        row_start, row_end = i * patch_result.shape[0], (i + 1) * patch_result.shape[0]
        col_start, col_end = j * patch_result.shape[1], (j + 1) * patch_result.shape[1]
        result_matrix[row_start:row_end, col_start:col_end] += patch_result

    return result_matrix


def get_bqq_matrices(patch_list: list[dict], bit_width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not patch_list:
        raise ValueError("patch_list must not be empty")

    row_width = max(patch["patch_row"] for patch in patch_list) + 1
    col_width = max(patch["patch_col"] for patch in patch_list) + 1
    m, l = patch_list[0]["mat1"].shape
    _, n = patch_list[0]["mat2"].shape
    coeff_dtype = patch_list[0]["coeff"].dtype
    matrix_dtype = patch_list[0]["mat1"].dtype

    A = torch.zeros((bit_width, row_width, col_width, 4), dtype=coeff_dtype)
    Y = torch.zeros((bit_width, row_width, col_width, m, l), dtype=matrix_dtype)
    Z = torch.zeros((bit_width, row_width, col_width, l, n), dtype=matrix_dtype)

    for patch in patch_list:
        i, j = patch["patch_row"], patch["patch_col"]
        a, y, z, bit_idx = patch["coeff"], patch["mat1"], patch["mat2"], patch["bit_idx"]

        if bit_idx >= bit_width:
            continue

        A[bit_idx, i, j] = a
        Y[bit_idx, i, j] = y
        Z[bit_idx, i, j] = z

    return A, Y, Z
