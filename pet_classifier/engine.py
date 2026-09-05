"""两阶段迁移学习：先训练分类头，再以较小学习率微调整个网络。"""
import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PetDataset, records, class_metadata, stratified_split
from .model import build_model, select_device


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(rows, training, batch_size, workers, device, seed=42):
    return DataLoader(PetDataset(rows, training), batch_size=batch_size, shuffle=training,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def run_epoch(model, loader, device, optimizer=None, scaler=None, frozen=False):
    training = optimizer is not None
    model.train(training)
    if training and frozen:
        # 冻结 backbone 时同时固定 BatchNorm 的统计量。
        model.features.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return {"loss": total_loss / count, "accuracy": correct / count}


def train(args):
    if args.epochs < 1 or not 0 <= args.freeze_epochs <= args.epochs:
        raise ValueError("epochs 至少为 1，freeze_epochs 必须在 0 和 epochs 之间")
    if args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("batch_size、lr 必须为正数，workers 不能为负数")
    out = Path(args.output)
    if (out / "best.pt").exists():
        raise FileExistsError("输出目录已有模型；请用 --output 指定新的目录，保留原实验。")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = select_device(args.device)
    all_rows = records(args.data, "trainval")
    train_rows, val_rows = stratified_split(all_rows, args.val_fraction, args.seed)
    classes = class_metadata(all_rows)
    config = vars(args).copy()
    config.update(device_used=str(device), torch_version=str(torch.__version__),
                  started_at=datetime.now(timezone.utc).isoformat(),
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  train_count=len(train_rows), val_count=len(val_rows),
                  architecture="mobilenet_v3_large", pretrained_weights="IMAGENET1K_V2")
    provenance = Path(args.data) / "oxford-iiit-pet" / "provenance.json"
    if provenance.exists():
        config["data_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
    write_json(out / "config.json", config)
    write_json(out / "split.json", {"train": [r["name"] for r in train_rows],
                                     "validation": [r["name"] for r in val_rows]})
    write_json(out / "classes.json", classes)
    train_loader = make_loader(train_rows, True, args.batch_size, args.workers, device, args.seed)
    val_loader = make_loader(val_rows, False, args.batch_size, args.workers, device, args.seed)
    model = build_model().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history, optimizer, scheduler = -1.0, [], None, None
    print(f"Device: {device}; train={len(train_rows)}, validation={len(val_rows)}", flush=True)
    for epoch in range(args.epochs):
        start = time.perf_counter()
        frozen = epoch < args.freeze_epochs
        if epoch == 0 or epoch == args.freeze_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad = not frozen
            if frozen:
                groups = [{"params": model.classifier.parameters(), "lr": args.lr}]
            else:
                groups = [{"params": model.features.parameters(), "lr": args.lr * 0.1},
                          {"params": model.classifier.parameters(), "lr": args.lr * 0.3}]
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            remaining = args.freeze_epochs if frozen else args.epochs - epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, remaining))
        tr = run_epoch(model, train_loader, device, optimizer, scaler, frozen)
        val = run_epoch(model, val_loader, device)
        row = {"epoch": epoch + 1, "stage": "head" if frozen else "finetune",
               "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
               "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - start}
        history.append(row)
        if val["accuracy"] > best:
            best = val["accuracy"]
            checkpoint = {"architecture": "mobilenet_v3_large", "model_state": model.state_dict(),
                          "classes": classes, "epoch": epoch + 1, "val_accuracy": best,
                          "preprocess": "MobileNet_V3_Large_Weights.IMAGENET1K_V2", "config": config}
            torch.save(checkpoint, out / "best.pt.tmp")
            (out / "best.pt.tmp").replace(out / "best.pt")
        scheduler.step()
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch+1}/{args.epochs} [{row['stage']}] "
              f"train={tr['accuracy']:.2%} val={val['accuracy']:.2%} "
              f"loss={val['loss']:.4f} time={row['seconds']:.1f}s best={best:.2%}", flush=True)
    print(f"Training finished: {out / 'best.pt'}", flush=True)
