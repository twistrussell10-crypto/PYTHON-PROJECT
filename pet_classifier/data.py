"""使用官方划分；验证集只从 trainval 中分层抽取，避免测试集泄漏。"""
from pathlib import Path
import random

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Large_Weights

WEIGHTS = MobileNet_V3_Large_Weights.IMAGENET1K_V2


def prepare(root):
    for split in ("trainval", "test"):
        dataset = OxfordIIITPet(str(root), split=split, download=True)
        print(f"{split}: {len(dataset)} images, {len(dataset.classes)} breeds", flush=True)


def records(root, split):
    base = Path(root) / "oxford-iiit-pet"
    annotation = base / "annotations" / f"{split}.txt"
    if not annotation.exists():
        raise FileNotFoundError("数据尚未准备，请先运行 python -m pet_classifier prepare")
    rows = []
    for line in annotation.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, label, species, _ = line.split()
        rows.append({"name": name, "label": int(label) - 1,
                     "species": "cat" if int(species) == 1 else "dog",
                     "path": str(base / "images" / f"{name}.jpg")})
    return rows


def class_metadata(rows):
    by_id = {}
    for row in rows:
        name = row["name"].rsplit("_", 1)[0]
        by_id[row["label"]] = {"name": name, "species": row["species"]}
    if sorted(by_id) != list(range(37)):
        raise ValueError("需要完整的 37 个类别，标签必须连续。")
    return [by_id[i] for i in range(37)]


def stratified_split(rows, val_fraction=0.2, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction 必须在 0 和 1 之间")
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    train, val = [], []
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
    def __init__(self, rows, training=False):
        self.rows = rows
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, row["label"]
