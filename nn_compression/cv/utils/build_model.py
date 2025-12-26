import timm
import torchvision.models as models
import torch
from transformers import SwinForImageClassification

def get_model(model_abbreviation):
    if model_abbreviation == 'swin-s':
        model = timm.create_model("swin_small_patch4_window7_224", pretrained=True)

    if model_abbreviation == 'swin-t':
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)

    if model_abbreviation == 'deit-b':
        model = timm.create_model("deit_base_patch16_224", pretrained=True)

    if model_abbreviation == 'deit-s':
        model = timm.create_model("deit_small_patch16_224", pretrained=True)

    if model_abbreviation == 'vit-s':
        model = timm.create_model("vit_small_patch16_224", pretrained=True)
    
    if model_abbreviation == 'vit-b':
        model = timm.create_model("vit_base_patch16_224", pretrained=True)

    return model