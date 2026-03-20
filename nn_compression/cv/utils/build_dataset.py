import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import math
from torch.utils.data import DataLoader
import random


def _resolve_imagenet_path(data_path=None):
    if data_path:
        return os.fspath(Path(data_path).expanduser())

    for env_var in ("IMAGENET_DIR", "IMAGENET_ROOT", "ILSVRC2012_DIR"):
        env_value = os.getenv(env_var)
        if env_value:
            return os.fspath(Path(env_value).expanduser())

    raise ValueError(
        "ImageNet path is not set. Pass data_path or set IMAGENET_DIR, IMAGENET_ROOT, or ILSVRC2012_DIR."
    )


def get_imagenet(model_name, val_batchsize=32, calib_batchsize=32, num_workers=16, num_traindatas=None, data_path=None, seed=0):
    data_path = _resolve_imagenet_path(data_path)

    if "deit" in model_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif 'vit' in model_name:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif 'swin' in model_name:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    # Data
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    val_dataset = datasets.ImageFolder(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = val_batchsize,
        shuffle=False,
        num_workers = num_workers,
        pin_memory=True,
    )

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    if num_traindatas is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        indices = torch.randperm(len(train_dataset), generator=g)[:num_traindatas]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=calib_batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return train_loader, val_loader

def calibsample_from_trainloader(train_loader, n, seed=0):
    random.seed(seed)
    all_inputs, all_targets = [], []
    for inputs, targets in train_loader:
        all_inputs.append(inputs)
        all_targets.append(targets)
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    idx = torch.randperm(len(all_inputs))[:n]
    return all_inputs[idx], all_targets[idx]

def build_transform(input_size=224, interpolation="bicubic",
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):
    def _pil_interp(method):
        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            return Image.BILINEAR
    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size, interpolation=ip
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

# CIFAR-10/100用データローダー
def get_cifar_dataloader(dataset_name, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # CIFARの画像をモデルに合うサイズにリサイズ
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Use 'CIFAR10' or 'CIFAR100'.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
