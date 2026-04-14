"""
Export a BQQ-quantized model to HuggingFace-compatible format.

The exported model can be loaded with:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("path/to/export", trust_remote_code=True)

The export directory contains:
    config.json          - base config + bqq_metadata (layer shapes)
    model.safetensors    - all parameters (Y, Z as buffers; a, b, c, d as params)
    modeling_bqq.py      - BinaryQuadratic module + auto-registered model class
    tokenizer files      - copied from base model
"""

import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from bqq_modules import BinaryQuadratic


def _collect_bqq_metadata(model):
    """Collect shape metadata for all BinaryQuadratic layers in the model."""
    bqq_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, BinaryQuadratic):
            bqq_layers[name] = {
                'bit_width': module.bit_width,
                'row_width': module.row_width,
                'col_width': module.col_width,
                'y_row': module.y_row,
                'inter_dimension': module.inter_dimension,
                'z_col': module.z_col,
                'has_bias': module.bias is not None,
            }
    return bqq_layers


# Template for the modeling file that ships with the exported model.
# This is written to output_dir/modeling_bqq.py and loaded via trust_remote_code.
_MODELING_TEMPLATE = r'''"""
BQQ (Binary Quadratic Quantization) model — auto-loaded by trust_remote_code=True.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


class BinaryQuadratic(nn.Module):
    """BQQ layer: Y, Z are binary buffers; a, b, c, d are learnable scale factors."""

    def __init__(self, bit_width, row_width, col_width, y_row, inter_dimension, z_col, has_bias=False):
        super().__init__()
        self.bit_width = bit_width
        self.row_width = row_width
        self.col_width = col_width
        self.y_row = y_row
        self.inter_dimension = inter_dimension
        self.z_col = z_col

        self.register_buffer("Y", torch.zeros(bit_width, row_width, col_width, y_row, inter_dimension, dtype=torch.bool))
        self.register_buffer("Z", torch.zeros(bit_width, row_width, col_width, inter_dimension, z_col, dtype=torch.bool))
        self.a = nn.Parameter(torch.zeros(bit_width, row_width, col_width, 1, 1))
        self.b = nn.Parameter(torch.zeros(bit_width, row_width, col_width, 1, 1))
        self.c = nn.Parameter(torch.zeros(bit_width, row_width, col_width, 1, 1))
        self.d = nn.Parameter(torch.zeros(row_width, col_width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(row_width * y_row)) if has_bias else None

    def forward(self, X):
        dtype = X.dtype
        device = self.Y.device
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)
        if self.bias is None:
            return X.to(device) @ W.T
        else:
            return X.to(device) @ W.T + self.bias.type(dtype).to(device)

    def get_weight(self, dtype=torch.float32):
        W_core = torch.matmul(self.Y.type(dtype), self.Z.type(dtype))
        Y_sum = self.Y.sum(dim=-1, keepdim=True).type(dtype)
        Z_sum = self.Z.sum(dim=-2, keepdim=True).type(dtype)
        W = self.a.type(dtype) * W_core + self.b.type(dtype) * Y_sum + self.c.type(dtype) * Z_sum
        W = W.sum(dim=0) + self.d.type(dtype)
        W = W.permute(0, 2, 1, 3).reshape(self.row_width * self.y_row, self.col_width * self.z_col)
        return W


class BQQConfig(PretrainedConfig):
    model_type = "bqq"

    def __init__(self, base_model_name="", base_config=None, bqq_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.base_config = base_config or {}
        self.bqq_layers = bqq_layers or {}


def _set_module(model, name, module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


class BQQModelForCausalLM(PreTrainedModel):
    config_class = BQQConfig

    def __init__(self, config):
        super().__init__(config)

        # Rebuild base model
        base_config = AutoConfig.from_pretrained(config.base_model_name)
        self.inner_model = AutoModelForCausalLM.from_config(base_config)

        # Replace Linear layers with empty BinaryQuadratic shells
        for name, meta in config.bqq_layers.items():
            bqq = BinaryQuadratic(
                bit_width=meta["bit_width"],
                row_width=meta["row_width"],
                col_width=meta["col_width"],
                y_row=meta["y_row"],
                inter_dimension=meta["inter_dimension"],
                z_col=meta["z_col"],
                has_bias=meta.get("has_bias", False),
            )
            _set_module(self.inner_model, name, bqq)

    def forward(self, **kwargs):
        return self.inner_model(**kwargs)

    @property
    def device(self):
        return next(self.parameters()).device


# Register so AutoModelForCausalLM can find us
BQQConfig.register_for_auto_class()
BQQModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")
'''


def default_hf_output_dir(base_model_name, bqq_layers):
    """Generate default output path: hf_models/{ModelName}-{N}bit-bqq"""
    model_basename = base_model_name.rstrip("/").split("/")[-1]
    if bqq_layers:
        bit_width = next(iter(bqq_layers.values()))['bit_width']
    else:
        bit_width = 0
    return Path("hf_models") / f"{model_basename}-{bit_width}bit-bqq"


def export_for_hf(bqq_model, base_model_name, output_dir=None, save_tokenizer=True):
    """
    Export a BQQ model to HuggingFace-compatible directory.

    Args:
        bqq_model: Model with BinaryQuadratic modules (from build_bqq_model or blockwise_quant)
        base_model_name: Original HuggingFace model name (e.g. "Qwen/Qwen3-2B")
        output_dir: Directory to save. If None, defaults to bqq/{ModelName}-{N}bit
        save_tokenizer: Whether to copy tokenizer files from base model
    """
    bqq_layers_meta = _collect_bqq_metadata(bqq_model)
    if output_dir is None:
        output_dir = default_hf_output_dir(base_model_name, bqq_layers_meta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. BQQ layer metadata (already collected for default path)
    bqq_layers = bqq_layers_meta
    print(f"Found {len(bqq_layers)} BinaryQuadratic layers")

    # 2. Build config
    base_config = bqq_model.config.to_dict() if hasattr(bqq_model, 'config') else {}
    config = {
        "model_type": "bqq",
        "auto_map": {
            "AutoConfig": "modeling_bqq.BQQConfig",
            "AutoModelForCausalLM": "modeling_bqq.BQQModelForCausalLM",
        },
        "base_model_name": base_model_name,
        "base_config": base_config,
        "bqq_layers": bqq_layers,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # 3. Save state_dict
    state_dict = bqq_model.state_dict()
    # Prefix keys with "inner_model." to match BQQModelForCausalLM structure
    prefixed = {"inner_model." + k: v for k, v in state_dict.items()}

    try:
        from safetensors.torch import save_file
        save_file(prefixed, output_dir / "model.safetensors")
        print(f"Saved model.safetensors ({len(prefixed)} tensors)")
    except ImportError:
        torch.save(prefixed, output_dir / "pytorch_model.bin")
        print(f"Saved pytorch_model.bin ({len(prefixed)} tensors)")

    # 4. Write modeling_bqq.py
    with open(output_dir / "modeling_bqq.py", "w") as f:
        f.write(_MODELING_TEMPLATE)
    print("Wrote modeling_bqq.py")

    # 5. Copy tokenizer
    if save_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.save_pretrained(output_dir)
            print("Saved tokenizer")
        except Exception as e:
            print(f"Warning: could not save tokenizer: {e}")

    print(f"\nExported to {output_dir}")
    print(f"Load with: AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")
