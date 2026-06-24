from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    from .compressed_data import aggregate_matrices, build_patch_index, load_layer_patches
except ImportError:
    from compressed_data import aggregate_matrices, build_patch_index, load_layer_patches


def load_causal_lm(model_name: str, *, device_map=None):
    def skip(*args, **kwargs):
        del args, kwargs

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    kwargs = {"dtype": "auto"}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.seqlen = getattr(model, "seqlen", 2048)
    return model


def get_llama(model_name: str):
    return load_causal_lm(model_name)

_DECODER_LAYER_CONTAINER_CANDIDATES = (
    "model.layers",
    "layers",
    "transformer.h",
    "gpt_neox.layers",
)


def _get_by_dotted_attr(root: nn.Module, dotted: str):
    cur = root
    for part in dotted.split('.'):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_decoder_layers(model: nn.Module) -> tuple[str, nn.ModuleList]:
    for dotted in _DECODER_LAYER_CONTAINER_CANDIDATES:
        module = _get_by_dotted_attr(model, dotted)
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return dotted, module

    fallback = None
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        if not all(isinstance(child, nn.Module) for child in module):
            continue
        score = 0
        lname = name.lower()
        if name.endswith('layers'):
            score += 4
        if name.endswith('h'):
            score += 3
        if 'layer' in lname:
            score += 2
        first = module[0]
        first_child_names = [child_name for child_name, _ in first.named_modules()]
        joined = ' '.join(first_child_names)
        if 'self_attn' in joined or 'mlp' in joined or 'linear_attn' in joined:
            score += 3
        if fallback is None or score > fallback[0]:
            fallback = (score, name, module)

    if fallback is not None:
        return fallback[1], fallback[2]

    raise ValueError('Could not resolve transformer decoder layers for this model architecture')


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    return resolve_decoder_layers(model)[1]


def get_decoder_layer(model: nn.Module, block_idx: int) -> nn.Module:
    return get_decoder_layers(model)[block_idx]


def set_decoder_layer(model: nn.Module, block_idx: int, block: nn.Module) -> None:
    get_decoder_layers(model)[block_idx] = block


def get_decoder_num_layers(model: nn.Module) -> int:
    return len(get_decoder_layers(model))


def get_decoder_block_prefix(model: nn.Module, block_idx: int) -> str:
    layer_path, _ = resolve_decoder_layers(model)
    return f"{layer_path}.{block_idx}"


def _should_restore_parameter(
    name: str,
    *,
    weight_qtz: bool,
    emb_qtz: bool,
    head_qtz: bool,
) -> bool:
    is_weight = "weight" in name and "norm" not in name and "head" not in name and "emb" not in name
    is_head = "head" in name and "norm" not in name
    is_embedding = "emb" in name and "norm" not in name
    return (weight_qtz and is_weight) or (head_qtz and is_head) or (emb_qtz and is_embedding)


def get_quantized_model(model, weights_dir, bit_width, weight_qtz=True, emb_qtz=True, head_qtz=True):
    patch_index = build_patch_index(weights_dir)

    for name, param in tqdm(model.named_parameters(), desc="Restoring quantized weights"):
        if not _should_restore_parameter(
            name,
            weight_qtz=weight_qtz,
            emb_qtz=emb_qtz,
            head_qtz=head_qtz,
        ):
            continue

        patch_list = load_layer_patches(name, patch_index, map_location="cpu")
        if not patch_list:
            continue

        original_shape = param.shape
        conversion_shape = param.reshape(param.shape[0], -1).shape
        reconst = aggregate_matrices(patch_list, conversion_shape, bit_width=bit_width)
        param.data.copy_(reconst.reshape(original_shape).to(device=param.device, dtype=param.dtype))

    return model
