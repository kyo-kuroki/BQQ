from __future__ import annotations

import math
import os
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
    # os.scandir は glob("*.pth") + sorted() より大幅に高速（数百万ファイル対応）
    for entry in os.scandir(weights_path):
        if not entry.name.endswith(".pth") or "_row" not in entry.name:
            continue
        stem = entry.name[:-4]  # remove .pth
        layer_name = stem.rsplit("_row", 1)[0]
        index[layer_name].append(Path(entry.path))

    # パッチの行・列順にソート
    for key in index:
        index[key].sort()

    return dict(index)


def consolidated_path(weights_dir: Path, layer_name: str) -> Path:
    """統合パッチファイルのパスを返す。"""
    return weights_dir / "_consolidated" / f"{layer_name}.pth"


def get_max_divisor(num: int, max_value: int) -> int:
    if max_value <= 0:
        raise ValueError('group_size must be a positive integer')
    limit = max(int(math.sqrt(num)), max_value)
    for candidate in range(limit, 0, -1):
        if num % candidate == 0 and candidate <= max_value:
            return candidate
    return 1


def get_patch_layout(shape: list[int] | tuple[int, int], group_size: int) -> dict[str, int]:
    if len(shape) != 2:
        raise ValueError(f'Only 2D tensors are supported for patch layout, got shape={tuple(shape)}')
    height, width = int(shape[0]), int(shape[1])
    patch_height = get_max_divisor(height, group_size)
    patch_width = get_max_divisor(width, group_size)
    return {
        'height': height,
        'width': width,
        'patch_height': patch_height,
        'patch_width': patch_width,
        'num_patch_rows': height // patch_height,
        'num_patch_cols': width // patch_width,
    }


def consolidate_target_patches(
    weights_dir: str | Path,
    target_name: str,
    *,
    shape: tuple[int, int],
    group_size: int,
) -> Path:
    """単一ターゲットのパッチファイルを統合して _consolidated/ に保存する。

    パッチレイアウト (shape, group_size) から (i, j) を列挙し、対応する
    _row{i}_col{j}.pth をロードして 1 ファイルにまとめる。
    ディレクトリ全体のスキャンが不要なため、数百万ファイルあっても高速。
    """
    weights_dir = Path(weights_dir)
    out_dir = weights_dir / "_consolidated"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{target_name}.pth"

    if out_path.exists():
        return out_path

    layout = get_patch_layout(shape, group_size)
    patch_list: list[dict] = []
    for i in range(layout["num_patch_rows"]):
        for j in range(layout["num_patch_cols"]):
            path = weights_dir / f"{target_name}_row{i}_col{j}.pth"
            if path.exists():
                patch_list.extend(torch.load(path, weights_only=False, map_location="cpu"))

    if patch_list:
        torch.save(patch_list, out_path)

    return out_path


def consolidate_all_patches(weights_dir: str | Path) -> None:
    """全ターゲットのパッチファイルを統合して _consolidated/ に保存する。

    既存の統合ファイルはスキップ。これにより replace_linear_with_bqq が
    数百万回の torch.load → ターゲット数回に高速化される。
    """
    weights_dir = Path(weights_dir)
    out_dir = weights_dir / "_consolidated"
    out_dir.mkdir(exist_ok=True)

    index = build_patch_index(weights_dir)
    total = len(index)
    for i, (layer_name, patch_paths) in enumerate(index.items()):
        out_path = out_dir / f"{layer_name}.pth"
        if out_path.exists():
            continue
        patch_list: list[dict] = []
        for path in patch_paths:
            patch_list.extend(torch.load(path, weights_only=False, map_location="cpu"))
        torch.save(patch_list, out_path)
        if (i + 1) % 20 == 0:
            print(f"  Consolidated {i + 1}/{total} targets")
    print(f"Consolidation done: {total} targets in {out_dir}")


def build_consolidated_index(weights_dir: str | Path) -> dict[str, Path]:
    """_consolidated/ ディレクトリからターゲット名 → consolidated ファイルパスの辞書を構築する。"""
    consolidated_dir = Path(weights_dir) / "_consolidated"
    if not consolidated_dir.exists():
        return {}
    index: dict[str, Path] = {}
    for entry in os.scandir(consolidated_dir):
        if entry.name.endswith(".pth"):
            layer_name = entry.name[:-4]  # remove .pth
            index[layer_name] = Path(entry.path)
    return index


def load_layer_patches(
    layer_name: str,
    patch_index: dict[str, list[Path] | Path],
    map_location: str | torch.device = "cpu",
) -> list[dict]:
    entry = patch_index.get(layer_name)
    if entry is None:
        return []

    # consolidated index の場合: 値が Path（単一ファイル）
    if isinstance(entry, Path):
        if entry.exists():
            return torch.load(entry, weights_only=False, map_location=map_location)
        return []

    # 旧 patch index の場合: 値が list[Path]
    paths = entry
    if paths:
        cpath = consolidated_path(paths[0].parent, layer_name)
        if cpath.exists():
            return torch.load(cpath, weights_only=False, map_location=map_location)

    # フォールバック: 個別パッチファイルからロード
    patch_list: list[dict] = []
    for path in paths:
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
