"""使用官方划分；验证集只从 trainval 中分层抽取，避免测试集泄漏。"""
from pathlib import Path
import random

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Large_Weights

# 统一保存预训练权重枚举，模型构建和验证预处理必须使用同一版本。
WEIGHTS = MobileNet_V3_Large_Weights.IMAGENET1K_V2


def prepare(root):
    """下载并准备 Oxford-IIIT Pet 的官方 trainval 和 test 数据。"""
    for split in ("trainval", "test"):
        dataset = OxfordIIITPet(str(root), split=split, download=True)
        print(f"{split}: {len(dataset)} images, {len(dataset.classes)} breeds", flush=True)


def records(root, split):
    """读取官方标注文件，返回每张图片的名称、标签、物种和本地路径。

    官方标签从 1 开始，而 PyTorch 的分类标签从 0 开始，因此这里统一减 1。
    """
    base = Path(root) / "oxford-iiit-pet"
    annotation = base / "annotations" / f"{split}.txt"
    if not annotation.exists():
        raise FileNotFoundError("数据尚未准备，请先运行 python -m pet_classifier prepare")
    rows = []
    for line in annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        # 每行依次为图片名、品种编号、物种编号和头部框编号；最后一项本项目不用。
        name, label, species, _ = line.split()
        rows.append({"name": name, "label": int(label) - 1,
                     "species": "cat" if int(species) == 1 else "dog",
                     "path": str(base / "images" / f"{name}.jpg")})
    return rows


def class_metadata(rows):
    """按照模型标签编号生成 37 项类别信息，并验证编号连续完整。"""
    by_id = {}
    for row in rows:
        # 文件名最后的下划线部分是图片序号，例如 samoyed_12 -> samoyed。
        name = row["name"].rsplit("_", 1)[0]
        by_id[row["label"]] = {"name": name, "species": row["species"]}
    if sorted(by_id) != list(range(37)):
        raise ValueError("需要完整的 37 个类别，标签必须连续。")
    return [by_id[i] for i in range(37)]


def stratified_split(rows, val_fraction=0.2, seed=42):
    """按类别分层拆分训练集与验证集。

    输入 rows 为样本字典列表；输出 (train, val)。固定 seed 可复现实验，逐类
    拆分可避免随机划分后某个稀有类别完全没有验证样本。
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction 必须在 0 和 1 之间")
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    train, val = [], []
    # 按标签排序使遍历顺序固定，再在每一类内部独立随机打乱。
    for label in sorted(groups):
        group = groups[label][:]
        if len(group) < 2:
            raise ValueError("每类至少需要两张图片才能划分训练集和验证集")
        rng.shuffle(group)
        n_val = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def image_transform(training=False):
    """返回训练或推理阶段的图像预处理流水线。

    训练阶段加入随机增强以降低过拟合；验证、测试和预测阶段直接使用预训练
    权重附带的确定性变换，从而保证三者输入处理完全一致。
    """
    if not training:
        return WEIGHTS.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class PetDataset(Dataset):
    """把样本记录转换成 PyTorch 可迭代的数据集。"""
    def __init__(self, rows, training=False):
        """保存样本记录，并提前建立当前阶段使用的图像变换。"""
        self.rows = rows
        self.transform = image_transform(training)

    def __len__(self):
        """返回数据集中图片数量，供 DataLoader 计算批次数量。"""
        return len(self.rows)

    def __getitem__(self, index):
        """读取第 index 张图片，返回形状 [3,224,224] 的张量和整数标签。"""
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, row["label"]
