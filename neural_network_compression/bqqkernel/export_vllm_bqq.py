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
from safetensors.torch import save_model
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

    print(f"Found {len(layers)} packed BQQ layers")
    if hf_source_dir is None and (model_path.parent / "config.json").exists():
        hf_source_dir = model_path.parent
    _copy_or_save_hf_files(model_name, output_dir, hf_source_dir)

    bqq_config = {
        "format": "packed_binary_quadratic",
        "version": 1,
        "base_model": model_name,
        "source_checkpoint": str(model_path),
        "layers": layers,
        "modules_to_not_convert": _unquantized_linear_names(model),
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
        save_model(model, str(output_dir / "model.safetensors"), metadata={"format": "pt"})

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
