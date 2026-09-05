"""输出完整测试集指标、混淆矩阵、逐图预测和误分类示例。"""
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import records, class_metadata
from .engine import make_loader, write_json
from .model import load_checkpoint, select_device


def evaluate(args):
    torch.set_num_threads(4)
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    rows = records(args.data, "test")
    if class_metadata(rows) != checkpoint["classes"]:
        raise ValueError("测试集类别顺序与模型不一致")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    loader = make_loader(rows, False, args.batch_size, args.workers, device)
    all_probabilities = []
    with torch.inference_mode():
        for images, _ in loader:
            all_probabilities.append(model(images.to(device)).softmax(1).cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    targets = np.array([r["label"] for r in rows])
    predictions = probabilities.argmax(1)
    top5 = np.argsort(probabilities, axis=1)[:, -5:]
    classes = checkpoint["classes"]
    names = [c["name"] for c in classes]
    report = classification_report(targets, predictions, labels=list(range(37)),
                                   target_names=names, output_dict=True, zero_division=0)
    species = np.array([c["species"] for c in classes])
    metrics = {"split": "official test", "samples": len(rows),
               "top1_accuracy": float(accuracy_score(targets, predictions)),
               "top5_accuracy": float((top5 == targets[:, None]).any(1).mean()),
               "macro_f1": float(f1_score(targets, predictions, average="macro")),
               "species_accuracy": float((species[targets] == species[predictions]).mean()),
               "checkpoint_epoch": checkpoint["epoch"],
               "validation_accuracy": checkpoint["val_accuracy"],
               "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
    write_json(out / "metrics.json", metrics)
    write_json(out / "classification_report.json", report)
    matrix = confusion_matrix(targets, predictions, labels=list(range(37)))
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true / predicted"] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + row.tolist())
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "true_breed", "predicted_breed", "probability", "correct"])
        for row, target, pred, probs in zip(rows, targets, predictions, probabilities):
            writer.writerow([row["name"], names[target], names[pred], float(probs[pred]), bool(target == pred)])
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(37), yticks=range(37), xticklabels=names, yticklabels=names,
           xlabel="Predicted breed", ylabel="True breed", title="Oxford-IIIT Pet | Official test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    errors = np.flatnonzero(targets != predictions)
    errors = sorted(errors, key=lambda i: float(probabilities[i, predictions[i]]), reverse=True)[:12]
    if errors:
        fig, axes = plt.subplots(3, 4, figsize=(14, 11))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, errors):
            with Image.open(rows[index]["path"]) as image:
                ax.imshow(ImageOps.exif_transpose(image).convert("RGB"))
            ax.set_title(f"True: {names[targets[index]]}\nPred: {names[predictions[index]]}\n"
                         f"Score: {probabilities[index, predictions[index]]:.1%}", fontsize=8)
        fig.suptitle("Highest-confidence mistakes (official test set)")
        fig.tight_layout()
        fig.savefig(out / "errors.png", dpi=140)
        plt.close(fig)
    history_path = Path(args.checkpoint).parent / "history.csv"
    if history_path.exists():
        with history_path.open(encoding="utf-8") as file:
            history = list(csv.DictReader(file))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [int(r["epoch"]) for r in history]
        for ax, metric in zip(axes, ("loss", "accuracy")):
            for split in ("train", "val"):
                ax.plot(epochs, [float(r[f"{split}_{metric}"]) for r in history], marker="o", label=split)
            ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"Training / validation {metric}")
            ax.legend()
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "training_curves.png", dpi=160)
        plt.close(fig)
    summary = f"""# Oxford-IIIT Pet 实验报告

## 任务与方法
识别 37 种猫犬品种。使用 ImageNet 预训练的 MobileNetV3-Large，先训练分类头，再微调全模型。
官方 trainval 内按类别分层划分训练/验证集；只根据验证集准确率选择最佳 checkpoint。
测试数据不参与梯度更新或最佳轮次选择。

## 本次实测结果
| 项目 | 数值 |
|---|---:|
| 官方测试图片数 | {metrics['samples']} |
| Top-1 准确率 | {metrics['top1_accuracy']:.2%} |
| Top-5 准确率 | {metrics['top5_accuracy']:.2%} |
| Macro F1 | {metrics['macro_f1']:.4f} |
| 猫犬物种准确率（取预测品种所属物种） | {metrics['species_accuracy']:.2%} |
| 最佳模型轮次 | {metrics['checkpoint_epoch']} |
| 最佳验证准确率 | {metrics['validation_accuracy']:.2%} |

## 结果文件
- `training_curves.png`：训练/验证损失与准确率。
- `confusion_matrix.png`、`confusion_matrix.csv`：真实类别为行、预测类别为列。
- `classification_report.json`：各品种 precision、recall、F1 和 support。
- `predictions.csv`：每张官方测试图的预测，可复核。
- `errors.png`：置信度最高的最多 12 张误分类图片（有错误时生成）。
- `metrics.json`：原始指标与 checkpoint SHA-256，用于对应模型和评估结果。

## 局限与后续实验
这是单次固定随机种子的实验，不能代表所有拍摄环境。输出分数是 softmax 分数，未做概率校准。
模型只会在已知 37 个品种中选择；不识别未知品种、混血或非宠物图片，也不具备可靠的拒识能力。
可在后续学习中研究相似品种混淆、概率校准、更强主干和多种子对比；新方案应使用验证集选择。

## 来源
- [Oxford 官方数据集与许可](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Torchvision MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)
"""
    (out / "REPORT.md").write_text(summary, encoding="utf-8")
    print(metrics, flush=True)
    print(f"Report: {out / 'REPORT.md'}", flush=True)
