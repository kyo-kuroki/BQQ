import timm
import torchvision.models as models
import torch
from transformers import SwinForImageClassification

def get_model(model_abbreviation):
    if model_abbreviation == 'swin-s':
        model_name = "microsoft/swin-small-patch4-window7-224"
        model = SwinForImageClassification.from_pretrained(model_name)
        # model = timm.create_model("swin_small_patch4_window7_224", pretrained=True)

    if model_abbreviation == 'swin-t':
        model_name = "microsoft/swin-tiny-patch4-window7-224"
        model = SwinForImageClassification.from_pretrained(model_name)
        # model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)

    if model_abbreviation == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        checkpoint_path = "/work/k-kuroki/resnet18_imagenet.pth.tar"  # 実際のパスに変更
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    if model_abbreviation == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        checkpoint_path = "/work/k-kuroki/resnet50_imagenet.pth.tar"  # 実際のパスに変更
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    if model_abbreviation == 'deit-b':
        model = timm.create_model("deit_base_patch16_224", pretrained=True)

    if model_abbreviation == 'deit-s':
        model = timm.create_model("deit_small_patch16_224", pretrained=True)

    if model_abbreviation == 'vit-s':
        model = timm.create_model("vit_small_patch16_224", pretrained=True)
    
    if model_abbreviation == 'vit-b':
        model = timm.create_model("vit_base_patch16_224", pretrained=True)

    return model