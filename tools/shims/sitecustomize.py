"""Process-wide import shims for local BQQ/vLLM experiments.

vLLM imports multimodal video utilities while inspecting text-only Qwen3.5
models.  The current environment has an incompatible torchcodec install, so
subprocess inspection fails before the text model is loaded.  Keeping this shim
on PYTHONPATH makes those subprocesses see a minimal torchcodec placeholder.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types


def _install_text_only_torchcodec_stub() -> None:
    if "torchcodec" in sys.modules:
        return

    torchcodec = types.ModuleType("torchcodec")
    torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)

    decoders = types.ModuleType("torchcodec.decoders")
    decoders.__spec__ = importlib.machinery.ModuleSpec(
        "torchcodec.decoders", loader=None
    )

    class VideoDecoder:
        pass

    decoders.VideoDecoder = VideoDecoder
    torchcodec.decoders = decoders
    sys.modules["torchcodec"] = torchcodec
    sys.modules["torchcodec.decoders"] = decoders


_install_text_only_torchcodec_stub()


_ORIGINAL_IMPORT_MODULE = importlib.import_module


def _import_module_with_flash_attn_fallback(name: str, package: str | None = None):
    if name == "flash_attn.ops.triton.rotary":
        raise ModuleNotFoundError(
            "flash_attn rotary is disabled by BQQ sitecustomize because the "
            "installed flash-attn extension is ABI-incompatible with PyTorch"
        )
    return _ORIGINAL_IMPORT_MODULE(name, package)


importlib.import_module = _import_module_with_flash_attn_fallback
