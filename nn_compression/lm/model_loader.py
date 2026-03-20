from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    from .compressed_data import aggregate_matrices, build_patch_index, load_layer_patches
except ImportError:
    from compressed_data import aggregate_matrices, build_patch_index, load_layer_patches


def load_causal_lm(model_name: str):
    def skip(*args, **kwargs):
        del args, kwargs

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.seqlen = getattr(model, "seqlen", 2048)
    return model


def get_llama(model_name: str):
    return load_causal_lm(model_name)


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
