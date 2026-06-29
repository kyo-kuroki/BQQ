"""Vision model loading and transformer-block resolution for CV BQQ.

Parallels lm/src/model_loader.py (load_causal_lm + decoder-layer helpers), but
for timm vision transformers (ViT / DeiT / Swin).

Block resolution:
  - ViT / DeiT: blocks live in ``model.blocks`` -> prefix ``blocks.{i}``
  - Swin:       blocks live in ``model.layers[s].blocks`` -> prefix
                ``layers.{s}.blocks.{b}`` (flattened in stage/block order)

Per-target and cache-first quantization walk ``named_parameters`` and do not need
the block concept, so they work for every model. Per-block parallel mode targets
ViT/DeiT primarily; Swin is supported best-effort via the flattened block list.
"""

from __future__ import annotations

import torch.nn as nn

import timm


MODEL_CHOICES = ['deit-s', 'deit-b', 'vit-s', 'vit-b', 'swin-t', 'swin-s']

_TIMM_NAMES = {
    'swin-s': 'swin_small_patch4_window7_224',
    'swin-t': 'swin_tiny_patch4_window7_224',
    'deit-b': 'deit_base_patch16_224',
    'deit-s': 'deit_small_patch16_224',
    'vit-s': 'vit_small_patch16_224',
    'vit-b': 'vit_base_patch16_224',
}


def get_vision_model(model_abbreviation: str) -> nn.Module:
    if model_abbreviation not in _TIMM_NAMES:
        raise ValueError(
            f"Unknown model '{model_abbreviation}'. Valid choices: {MODEL_CHOICES}")
    return timm.create_model(_TIMM_NAMES[model_abbreviation], pretrained=True)


# Backward-compatible alias (old code imported `get_model`).
get_model = get_vision_model


# ---------------------------------------------------------------------------
# Transformer-block resolution
# ---------------------------------------------------------------------------

def resolve_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return an ordered list of (prefix, block_module) for the transformer blocks.

    Handles the flat ViT/DeiT layout (``model.blocks``) and the hierarchical Swin
    layout (``model.layers[s].blocks``).
    """
    # ViT / DeiT: a single flat container of blocks.
    blocks = getattr(model, 'blocks', None)
    if isinstance(blocks, (nn.ModuleList, nn.Sequential)) and len(blocks) > 0:
        return [(f'blocks.{i}', blk) for i, blk in enumerate(blocks)]

    # Swin: stages (``layers``), each with its own ``blocks``.
    layers = getattr(model, 'layers', None)
    if isinstance(layers, (nn.ModuleList, nn.Sequential)) and len(layers) > 0:
        result: list[tuple[str, nn.Module]] = []
        for s, stage in enumerate(layers):
            stage_blocks = getattr(stage, 'blocks', None)
            if isinstance(stage_blocks, (nn.ModuleList, nn.Sequential)):
                for b, blk in enumerate(stage_blocks):
                    result.append((f'layers.{s}.blocks.{b}', blk))
        if result:
            return result

    raise ValueError('Could not resolve transformer blocks for this model architecture')


def get_num_blocks(model: nn.Module) -> int:
    return len(resolve_blocks(model))


def get_block(model: nn.Module, block_idx: int) -> nn.Module:
    return resolve_blocks(model)[block_idx][1]


def get_block_prefix(model: nn.Module, block_idx: int) -> str:
    return resolve_blocks(model)[block_idx][0]


def _get_by_dotted_attr(root: nn.Module, dotted: str):
    cur = root
    for part in dotted.split('.'):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def set_block(model: nn.Module, block_idx: int, block: nn.Module) -> None:
    prefix = get_block_prefix(model, block_idx)
    parts = prefix.split('.')
    parent = _get_by_dotted_attr(model, '.'.join(parts[:-1])) if len(parts) > 1 else model
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = block
    else:
        setattr(parent, last, block)
