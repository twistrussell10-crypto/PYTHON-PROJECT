"""模型定义和可独立加载的 checkpoint。"""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large

from .data import WEIGHTS


def build_model(pretrained=True):
    """创建用于 37 个宠物品种分类的 MobileNetV3-Large。

    参数:
        pretrained: 为 True 时加载 ImageNet 预训练权重，训练时通常使用 True；
            读取本项目 checkpoint 时使用 False，避免重复下载权重。
    返回:
        最后一层输出维度已经改为 37 的 PyTorch 模型。
    """
    # 原模型最后一层输出 1000 个 ImageNet 类别；这里保留前面的特征提取网络。
    model = mobilenet_v3_large(weights=WEIGHTS if pretrained else None)
    # in_features 读取原层输入维度，确保替换后的线性层能与前一层正确连接。
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 37)
    return model


def select_device(requested="auto"):
    """把用户指定的设备字符串转换成 torch.device，并检查 CUDA 可用性。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请使用 --device cpu")
    return torch.device(requested)


def load_checkpoint(path, device="cpu"):
    """从磁盘恢复模型和实验元数据。

    checkpoint 不只保存神经网络参数，还保存类别顺序。预测时必须沿用这个顺序，
    否则模型输出编号会被错误地解释成另一个品种。
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"模型不存在：{path}。请先运行训练命令。")
    # 先映射到 CPU，随后再统一移动到目标设备，兼容无显卡的演示电脑。
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "mobilenet_v3_large" or len(checkpoint.get("classes", [])) != 37:
        raise ValueError("模型格式或类别数量不匹配")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint
