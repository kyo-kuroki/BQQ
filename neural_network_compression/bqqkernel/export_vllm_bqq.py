"""Export a packed BQQ checkpoint into a vLLM-oriented weight directory.

This is the first bridge step for vLLM integration.  It keeps the original HF
module names and writes:

  - config.json / tokenizer files from the base HF model
  - bqq_config.json with per-layer BQQ shapes
  - quantization_config.json advertising quant_method="bqq"
  - model.safetensors containing normal model tensors and packed BQQ tensors

The companion `bqqkernel.vllm_quantization` module consumes the metadata and
registers a custom vLLM LinearMethod for matching layer prefixes.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TORCHAUDIO", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCHAUDIO", "1")

if "torchaudio" not in sys.modules:
    torchaudio_stub = types.ModuleType("torchaudio")
    torchaudio_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
    sys.modules["torchaudio"] = torchaudio_stub

import dill
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer


def _add_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _bqq_layer_metadata(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    _add_repo_to_path()
    from neural_network_compression.bqqkernel.bqq_modules import PackedBinaryQuadratic

    layers: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, PackedBinaryQuadratic):
            continue
        layers[name] = {
            "bit_width": int(module.bit_width),
            "row_width": int(module.row_width),
            "col_width": int(module.col_width),
            "y_row": int(module.y_row),
            "z_col": int(module.z_col),
            "inter_dimension": int(module.inter_dimension),
            "k8": int(module._k8),
            "input_size": int(module.col_width * module.z_col),
            "output_size": int(module.row_width * module.y_row),
            "has_bias": module.bias is not None,
            "tensor_names": {
                "Y_packed": f"{name}.Y_packed",
                "Z_packed": f"{name}.Z_packed",
                "Y_sum_i16": f"{name}.Y_sum_i16",
                "Z_sum_i16": f"{name}.Z_sum_i16",
                "a": f"{name}.a",
                "b": f"{name}.b",
                "c": f"{name}.c",
                "d": f"{name}.d",
                "bias": f"{name}.bias" if module.bias is not None else None,
            },
        }
    return layers


_BQQ_TENSOR_SUFFIXES = (
    "Y_packed",
    "Z_packed",
    "Y_sum_i16",
    "Z_sum_i16",
    "a",
    "b",
    "c",
    "d",
    "bias",
)


_FUSED_SUFFIXES = {
    ".self_attn.qkv_proj": [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"],
    ".mlp.gate_up_proj": [".mlp.gate_proj", ".mlp.up_proj"],
    ".linear_attn.in_proj_qkvz": [
        ".linear_attn.in_proj_qkv",
        ".linear_attn.in_proj_z",
    ],
    ".linear_attn.in_proj_ba": [
        ".linear_attn.in_proj_b",
        ".linear_attn.in_proj_a",
    ],
}


def _replace_suffix(name: str, old_suffix: str, new_suffix: str) -> str | None:
    if name.endswith(old_suffix):
        return name[: -len(old_suffix)] + new_suffix
    return None


def _make_fused_metadata(layers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    consumed: set[str] = set()

    for fused_suffix, part_suffixes in _FUSED_SUFFIXES.items():
        for layer_name in list(layers):
            part0 = _replace_suffix(layer_name, part_suffixes[0], fused_suffix)
            if part0 is None:
                continue
            fused_name = part0
            part_names = [
                _replace_suffix(fused_name, fused_suffix, part_suffix)
                for part_suffix in part_suffixes
            ]
            if any(part_name not in layers for part_name in part_names):
                continue
            part_metas = [layers[part_name] for part_name in part_names if part_name is not None]
            input_sizes = {meta["input_size"] for meta in part_metas}
            if len(input_sizes) != 1:
                raise ValueError(f"Fused BQQ parts have different input sizes: {fused_name}")
            fused[fused_name] = {
                "fused": True,
                "parts": part_names,
                "input_size": part_metas[0]["input_size"],
                "output_size": sum(int(meta["output_size"]) for meta in part_metas),
                "has_bias": any(bool(meta["has_bias"]) for meta in part_metas),
            }
            consumed.update(part_names)  # type: ignore[arg-type]

    for name, meta in layers.items():
        if name in consumed:
            continue
        fused[name] = dict(meta)
        fused[name]["fused"] = False

    for name, meta in fused.items():
        if meta.get("fused"):
            for part_idx, part_name in enumerate(meta["parts"]):
                part_meta = layers[part_name]
                tensor_names = {}
                for suffix in _BQQ_TENSOR_SUFFIXES:
                    if suffix == "bias" and not part_meta["has_bias"]:
                        tensor_names[suffix] = None
                    else:
                        tensor_names[suffix] = f"{name}.p{part_idx}_{suffix}"
                part_meta = dict(part_meta)
                part_meta["tensor_names"] = tensor_names
                meta.setdefault("part_metadata", []).append(part_meta)
        else:
            tensor_names = {}
            for suffix in _BQQ_TENSOR_SUFFIXES:
                if suffix == "bias" and not meta["has_bias"]:
                    tensor_names[suffix] = None
                else:
                    tensor_names[suffix] = f"{name}.{suffix}"
            meta["tensor_names"] = tensor_names

    return fused


def _export_state_dict_for_vllm(
    model: torch.nn.Module,
    layers: dict[str, dict[str, Any]],
    vllm_layers: dict[str, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    bqq_prefixes = tuple(f"{name}." for name in layers)
    tensors: dict[str, torch.Tensor] = {}

    for name, tensor in state.items():
        if name.startswith(bqq_prefixes):
            continue
        tensors[name] = tensor.detach().cpu().contiguous()

    for vllm_name, meta in vllm_layers.items():
        if meta.get("fused"):
            for part_idx, part_name in enumerate(meta["parts"]):
                part_meta = layers[part_name]
                for suffix in _BQQ_TENSOR_SUFFIXES:
                    if suffix == "bias" and not part_meta["has_bias"]:
                        continue
                    src = f"{part_name}.{suffix}"
                    dst = f"{vllm_name}.p{part_idx}_{suffix}"
                    tensors[dst] = state[src].detach().cpu().contiguous()
        else:
            for suffix in _BQQ_TENSOR_SUFFIXES:
                if suffix == "bias" and not meta["has_bias"]:
                    continue
                src = f"{vllm_name}.{suffix}"
                dst = f"{vllm_name}.{suffix}"
                tensors[dst] = state[src].detach().cpu().contiguous()
    _break_shared_storage_aliases(tensors)
    return tensors


def _break_shared_storage_aliases(tensors: dict[str, torch.Tensor]) -> None:
    """safetensors cannot save multiple tensor entries backed by one storage."""
    seen: dict[tuple[int, int], str] = {}
    for name, tensor in list(tensors.items()):
        if tensor.device.type != "cpu":
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key in seen:
            tensors[name] = tensor.clone()
        else:
            seen[key] = name


def _unquantized_linear_names(model: torch.nn.Module) -> list[str]:
    _add_repo_to_path()
    from neural_network_compression.bqqkernel.bqq_modules import PackedBinaryQuadratic

    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.append(name)
        elif isinstance(module, PackedBinaryQuadratic):
            continue
    return names


def _copy_or_save_hf_files(model_name: str, output_dir: Path, hf_source_dir: Path | None) -> None:
    if hf_source_dir is not None and (hf_source_dir / "config.json").exists():
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ):
            src = hf_source_dir / name
            if src.exists():
                shutil.copy2(src, output_dir / name)
        return

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.quantization_config = {"quant_method": "bqq"}
    config.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    tokenizer.save_pretrained(output_dir)


def _inject_bqq_quantization_config(
    output_dir: Path,
    layers: dict[str, dict[str, Any]],
    modules_to_not_convert: list[str],
) -> None:
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["quantization_config"] = {
        "quant_method": "bqq",
        "format": "packed_binary_quadratic",
        "version": 1,
        "layers": layers,
        "modules_to_not_convert": modules_to_not_convert,
    }
    rope_parameters = config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        rope_parameters.pop("mrope_section", None)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def export_bqq_for_vllm(
    *,
    model_path: Path,
    model_name: str,
    output_dir: Path,
    hf_source_dir: Path | None = None,
    copy_source_checkpoint: bool = False,
    metadata_only: bool = False,
) -> Path:
    _add_repo_to_path()
    import neural_network_compression.bqqkernel.bqq_modules as _bqq_modules  # noqa: F401

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading packed BQQ model from {model_path}")
    model = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=dill,
    )
    if not hasattr(model, "state_dict"):
        raise TypeError(f"{model_path} did not deserialize to a torch.nn.Module")

    layers = _bqq_layer_metadata(model)
    if not layers:
        raise ValueError("No PackedBinaryQuadratic layers found. Pack the model before exporting.")
    vllm_layers = _make_fused_metadata(layers)

    print(f"Found {len(layers)} packed BQQ layers")
    print(f"Exporting {len(vllm_layers)} vLLM BQQ linear layers")
    if hf_source_dir is None and (model_path.parent / "config.json").exists():
        hf_source_dir = model_path.parent
    _copy_or_save_hf_files(model_name, output_dir, hf_source_dir)
    modules_to_not_convert = _unquantized_linear_names(model)
    _inject_bqq_quantization_config(output_dir, vllm_layers, modules_to_not_convert)

    bqq_config = {
        "format": "packed_binary_quadratic",
        "version": 1,
        "base_model": model_name,
        "source_checkpoint": str(model_path),
        "hf_layers": layers,
        "layers": vllm_layers,
        "modules_to_not_convert": modules_to_not_convert,
    }
    with open(output_dir / "bqq_config.json", "w") as f:
        json.dump(bqq_config, f, indent=2)

    quantization_config = {
        "quant_method": "bqq",
        "format": "packed_binary_quadratic",
        "version": 1,
        "bqq_config": "bqq_config.json",
    }
    with open(output_dir / "quantization_config.json", "w") as f:
        json.dump(quantization_config, f, indent=2)

    if metadata_only:
        print("Skipping model.safetensors because --metadata-only was set")
    else:
        print(f"Writing tensors to model.safetensors")
        tensors = _export_state_dict_for_vllm(model, layers, vllm_layers)
        save_file(tensors, output_dir / "model.safetensors", metadata={"format": "pt"})

    if copy_source_checkpoint and not metadata_only:
        shutil.copy2(model_path, output_dir / model_path.name)

    print(f"Saved vLLM BQQ export to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hf-source-dir", type=Path, default=None,
                        help="Directory with config/tokenizer artifacts to copy. "
                             "Defaults to the checkpoint parent when config.json exists.")
    parser.add_argument("--copy-source-checkpoint", action="store_true")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Write configs/metadata only; skip large model.safetensors")
    args = parser.parse_args()

    export_bqq_for_vllm(
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        hf_source_dir=args.hf_source_dir,
        copy_source_checkpoint=args.copy_source_checkpoint,
        metadata_only=args.metadata_only,
    )


if __name__ == "__main__":
    main()
