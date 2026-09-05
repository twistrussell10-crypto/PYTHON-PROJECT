"""模型定义和可独立加载的 checkpoint。"""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large

from .data import WEIGHTS


def build_model(pretrained=True):
    model = mobilenet_v3_large(weights=WEIGHTS if pretrained else None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 37)
    return model


def select_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请使用 --device cpu")
    return torch.device(requested)


def load_checkpoint(path, device="cpu"):
    if not Path(path).is_file():
        raise FileNotFoundError(f"模型不存在：{path}。请先运行训练命令。")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_large" or len(checkpoint.get("classes", [])) != 37:
        raise ValueError("模型格式或类别数量不匹配")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
