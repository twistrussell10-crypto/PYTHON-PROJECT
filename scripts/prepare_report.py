"""从已保存的实测结果准备汇报数据，不训练模型或重新选择最佳轮次。"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pet_classifier.data import records
from pet_classifier.labels import breed_zh
from pet_classifier.predict import Predictor


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    run = ROOT / "outputs" / "baseline"
    report = run / "evaluation"
    build = ROOT / ".report-build"
    build.mkdir(exist_ok=True)
    metrics = read_json(report / "metrics.json")
    digest = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    assert digest == metrics["checkpoint_sha256"], "模型与评估报告不一致"
    with (report / "predictions.csv").open(encoding="utf-8", newline="") as source:
        predictions = list(csv.DictReader(source))
    correct = sum(row["correct"] == "True" for row in predictions)
    assert len(predictions) == metrics["samples"]
    assert abs(correct / len(predictions) - metrics["top1_accuracy"]) < 1e-12
    errors = [row for row in predictions if row["correct"] == "False"]
    confusion = Counter((row["true_breed"], row["predicted_breed"]) for row in errors)
    with (run / "history.csv").open(encoding="utf-8", newline="") as source:
        history = [{key: (value if key == "stage" else float(value)) for key, value in row.items()}
                   for row in csv.DictReader(source)]
    split = read_json(run / "split.json")
    trainval, test = records(ROOT / "data", "trainval"), records(ROOT / "data", "test")
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == {r["name"] for r in trainval}
    assert not {r["name"] for r in trainval} & {r["name"] for r in test}
    examples_dir = ROOT / "examples"
    manifest = {}
    for filename, original in [("Abyssinian.jpg", "Abyssinian_1"), ("Basset_Hound.jpg", "basset_hound_1")]:
        match = next(r for r in trainval + test if r["name"].lower() == original.lower())
        manifest[filename] = {"original": match["name"], "breed": match["name"].rsplit("_", 1)[0],
                              "split": "test" if match in test else "trainval"}
    demos = []
    predictor = Predictor(run / "best.pt", "cpu")
    for breed in ("beagle", "Bengal", "samoyed", "Sphynx"):
        row = sorted([r for r in test if r["name"].rsplit("_", 1)[0] == breed], key=lambda r:r["name"])[0]
        filename = f"demo_{breed}.jpg"
        shutil.copy2(row["path"], examples_dir / filename)
        manifest[filename] = {"original": row["name"], "breed": breed, "split": "test"}
        demos.append(dict(manifest[filename], file=str(examples_dir / filename),
                          predictions=predictor.predict_file(examples_dir / filename)))
    (examples_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    classes = read_json(run / "classes.json")
    evidence = {"metrics": metrics, "correct": correct, "incorrect": len(errors), "history": history,
        "config": read_json(run / "config.json"), "parameters": sum(p.numel() for p in predictor.model.parameters()),
        "model_bytes": (run / "best.pt").stat().st_size, "classes": classes,
        "train_count": len(split["train"]), "val_count": len(split["validation"]),
        "species_counts": dict(Counter(c["species"] for c in classes)),
        "training_seconds": sum(row["seconds"] for row in history), "demos": demos,
        "confusions": [{"true": a, "predicted": b, "true_zh": breed_zh(a), "predicted_zh": breed_zh(b), "count": n}
                       for (a,b),n in confusion.most_common(6)],
        "errors": sorted(errors, key=lambda r:float(r["probability"]), reverse=True)[:3],
        "breed_zh": {c["name"]:breed_zh(c["name"]) for c in classes}}
    (build / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key:evidence[key] for key in ["correct","incorrect","parameters","training_seconds","confusions"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
