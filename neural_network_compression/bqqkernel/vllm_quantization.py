"""Experimental vLLM quantization bridge for packed BQQ layers.

This module is intended to be copied or imported into a vLLM installation and
registered as a custom quantization method named ``bqq``.  It consumes the
``bqq_config.json`` emitted by ``bqqkernel/export_vllm_bqq.py``.

Current scope:
  - one GPU / no tensor-parallel sharding
  - vLLM fused Linear prefixes backed by multiple HF BQQ tensors
  - PackedBinaryQuadratic CUDA forward for decode/prefill
"""

from __future__ import annotations

import json
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Any

import torch


def _install_text_only_torchcodec_stub() -> None:
    """Avoid torchcodec import failures in text-only vLLM processes.

    vLLM 0.25 imports multimodal video helpers while resolving quantization
    configs.  On this cluster torchcodec is installed but its binary
    dependencies are incompatible with the active PyTorch/CUDA stack.  BQQ text
    models do not need video decoding, so a minimal stub is safer than failing
    before quantization registration.
    """
    if "torchcodec" in sys.modules:
        return
    tc = types.ModuleType("torchcodec")
    tc.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
    dec = types.ModuleType("torchcodec.decoders")
    dec.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)

    class VideoDecoder:  # pragma: no cover - only used as an import placeholder
        pass

    dec.VideoDecoder = VideoDecoder
    tc.decoders = dec
    sys.modules["torchcodec"] = tc
    sys.modules["torchcodec.decoders"] = dec


_install_text_only_torchcodec_stub()


def _require_vllm():
    try:
        from vllm.model_executor.layers.linear import (
            LinearBase,
            LinearMethodBase,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
        from vllm.model_executor.utils import set_weight_attrs
    except ImportError as exc:
        raise ImportError(
            "vLLM is required to use bqqkernel.vllm_quantization. "
            "Install vLLM, then register BQQConfig as a quantization method."
        ) from exc
    return LinearBase, LinearMethodBase, QuantizationConfig, UnquantizedLinearMethod, set_weight_attrs


def _load_bqq_config_from_dict(config: dict[str, Any]) -> dict[str, Any]:
    cfg_path = config.get("bqq_config")
    if cfg_path is None:
        return config
    path = Path(cfg_path)
    if not path.exists():
        # vLLM may pass only the contents of quantization_config.json.  In that
        # case the caller should resolve this relative to the model directory
        # and pass the loaded bqq_config dict directly.
        return config
    with open(path) as f:
        loaded = json.load(f)
    merged = dict(config)
    merged.update(loaded)
    return merged


class BQQConfig(_require_vllm()[2]):  # type: ignore[misc]
    """vLLM QuantizationConfig for packed BinaryQuadratic layers."""

    def __init__(
        self,
        layers: dict[str, dict[str, Any]],
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return f"BQQConfig(num_layers={len(self.layers)})"

    def get_name(self):
        return "bqq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json", "bqq_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BQQConfig":
        config = _load_bqq_config_from_dict(config)
        layers = config.get("layers")
        if not isinstance(layers, dict) or not layers:
            raise ValueError("BQQ quantization config requires a non-empty 'layers' mapping")
        modules_to_not_convert = config.get("modules_to_not_convert", [])
        return cls(layers=layers, modules_to_not_convert=modules_to_not_convert)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        LinearBase, _LinearMethodBase, _QuantizationConfig, UnquantizedLinearMethod, _set_weight_attrs = _require_vllm()
        if not isinstance(layer, LinearBase):
            return None
        if prefix in self.modules_to_not_convert:
            return UnquantizedLinearMethod()
        meta = self.layers.get(prefix)
        if meta is None:
            return None
        return BQQLinearMethod(meta)


class BQQLinearMethod(_require_vllm()[1]):  # type: ignore[misc]
    """vLLM LinearMethod that delegates to PackedBinaryQuadratic.forward."""

    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.tensor_names = meta.get("tensor_names", {})

    @staticmethod
    def _register_bqq_params(
        layer: torch.nn.Module,
        meta: dict[str, Any],
        params_dtype: torch.dtype,
        set_weight_attrs,
        weight_loader,
        *,
        prefix: str = "",
    ) -> None:
        bit_width = int(meta["bit_width"])
        row_width = int(meta["row_width"])
        col_width = int(meta["col_width"])
        y_row = int(meta["y_row"])
        z_col = int(meta["z_col"])
        k8 = int(meta["k8"])

        def register_param(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
            param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(prefix + name, param)
            set_weight_attrs(param, {"weight_loader": weight_loader})

        register_param("Y_packed", (bit_width, row_width, col_width, y_row, k8), torch.uint8)
        register_param("Z_packed", (bit_width, row_width, col_width, z_col, k8), torch.uint8)
        register_param("Y_sum_i16", (bit_width, row_width, col_width, y_row, 1), torch.int16)
        register_param("Z_sum_i16", (bit_width, row_width, col_width, 1, z_col), torch.int16)
        register_param("a", (bit_width, row_width, col_width, 1, 1), params_dtype)
        register_param("b", (bit_width, row_width, col_width, 1, 1), params_dtype)
        register_param("c", (bit_width, row_width, col_width, 1, 1), params_dtype)
        register_param("d", (row_width, col_width, 1, 1), params_dtype)

    @staticmethod
    def _build_runtime(layer: torch.nn.Module, meta: dict[str, Any], *, prefix: str = ""):
        from neural_network_compression.bqqkernel.bqq_modules import PackedBinaryQuadratic

        def param(name: str) -> torch.Tensor:
            return getattr(layer, prefix + name).data

        runtime = PackedBinaryQuadratic(
            Y_packed=param("Y_packed"),
            Z_packed=param("Z_packed"),
            a=param("a"),
            b=param("b"),
            c=param("c"),
            d=param("d"),
            Y_sum_i16=param("Y_sum_i16"),
            Z_sum_i16=param("Z_sum_i16"),
            inter_dimension=int(meta["inter_dimension"]),
            y_row=int(meta["y_row"]),
            z_col=int(meta["z_col"]),
            bias=None,
        )
        runtime.eval()
        runtime.to(device=param("Y_packed").device)
        return runtime

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        expected_in = int(self.meta["input_size"])
        expected_out = int(self.meta["output_size"])
        if input_size_per_partition != expected_in or sum(output_partition_sizes) != expected_out:
            raise ValueError(
                "BQQ layer shape mismatch for prefix "
                f"{getattr(layer, 'prefix', '<unknown>')}: "
                f"vLLM saw in={input_size_per_partition}, out={sum(output_partition_sizes)}; "
                f"BQQ export has in={expected_in}, out={expected_out}. "
                "Tensor parallel sharding is not supported by this BQQ bridge."
            )

        _LinearBase, _LinearMethodBase, _QuantizationConfig, _UnquantizedLinearMethod, set_weight_attrs = _require_vllm()
        weight_loader = extra_weight_attrs.get("weight_loader")

        if self.meta.get("fused"):
            part_metas = self.meta.get("part_metadata", [])
            part_sizes = [int(part_meta["output_size"]) for part_meta in part_metas]
            if output_partition_sizes != part_sizes and sum(output_partition_sizes) != sum(part_sizes):
                raise ValueError(
                    "BQQ fused layer output partition mismatch for prefix "
                    f"{getattr(layer, 'prefix', '<unknown>')}: "
                    f"vLLM partitions={output_partition_sizes}; BQQ parts={part_sizes}."
                )
            for part_idx, part_meta in enumerate(part_metas):
                self._register_bqq_params(
                    layer,
                    part_meta,
                    params_dtype,
                    set_weight_attrs,
                    weight_loader,
                    prefix=f"p{part_idx}_",
                )
        else:
            self._register_bqq_params(
                layer,
                self.meta,
                params_dtype,
                set_weight_attrs,
                weight_loader,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.meta.get("fused"):
            layer.bqq_runtimes = [
                self._build_runtime(layer, part_meta, prefix=f"p{part_idx}_")
                for part_idx, part_meta in enumerate(self.meta.get("part_metadata", []))
            ]
        else:
            layer.bqq_runtime = self._build_runtime(layer, self.meta)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runtime = getattr(layer, "bqq_runtime", None)
        runtimes = getattr(layer, "bqq_runtimes", None)
        if self.meta.get("fused"):
            if runtimes is None:
                self.process_weights_after_loading(layer)
                runtimes = layer.bqq_runtimes
            out = torch.cat([runtime(x) for runtime in runtimes], dim=-1)
        else:
            if runtime is None:
                self.process_weights_after_loading(layer)
                runtime = layer.bqq_runtime
            out = runtime(x)
        if bias is not None:
            out = out + bias
        return out


def register_bqq_quantization():
    """Register ``quant_method='bqq'`` with vLLM's quantization registry."""
    from vllm.model_executor.layers.quantization import register_quantization_config

    return register_quantization_config("bqq")(BQQConfig)


# Importing this module inside a vLLM process is enough to make
# `quant_method: "bqq"` discoverable.
BQQConfig = register_bqq_quantization()
