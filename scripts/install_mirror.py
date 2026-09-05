"""将镜像图片按文件名还原；只采用官方 trainval/test 标注，不使用镜像划分。"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import requests
from torchvision.datasets.utils import extract_archive

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "oxford-iiit-pet"
ANNOTATION_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
ANNOTATION_MD5 = "95a8c909bbe2e81eed6a22bccdf3f68f"


def install_annotations():
    BASE.mkdir(parents=True, exist_ok=True)
    archive = BASE / "annotations.tar.gz"
    if not archive.exists() or hashlib.md5(archive.read_bytes()).hexdigest() != ANNOTATION_MD5:
        temporary = archive.with_suffix(".partial")
        with requests.get(ANNOTATION_URL, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != ANNOTATION_MD5:
            raise RuntimeError("官方标注 MD5 校验失败")
        temporary.replace(archive)
    extract_archive(str(archive), str(BASE))
    print("Official annotations downloaded and MD5 verified", flush=True)


def main(annotations_only=False):
    install_annotations()
    if annotations_only:
        return
    extracted = ROOT / "data" / "mirror-download" / "extracted"
    sources = sorted(extracted.rglob("*.jpg"))
    if not sources or len({p.name for p in sources}) != len(sources):
        raise ValueError("镜像未解压，或图片文件名重复")
    (BASE / "images").mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, BASE / "images" / path.name)
    # 验证官方划分需要的每张图都存在，并保存图片内容指纹。
    sets = {}
    for split in ("trainval", "test"):
        names = [line.split()[0] for line in (BASE / "annotations" / f"{split}.txt").read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
        sets[split] = set(names)
        missing = [name for name in names if not (BASE / "images" / f"{name}.jpg").is_file()]
        if missing:
            raise ValueError(f"缺少官方 {split} 图片：{missing[:5]}")
    if sets["trainval"] & sets["test"]:
        raise ValueError("官方训练和测试划分存在重复")
    digest = hashlib.sha256()
    for name in sorted(sets["trainval"] | sets["test"]):
        digest.update(name.encode("utf-8"))
        digest.update((BASE / "images" / f"{name}.jpg").read_bytes())
    provenance = {"image_source": "https://zenodo.org/records/8067751",
                  "mirror_zip_md5": hashlib.md5((ROOT / "data" / "mirror-download" / "pets.zip").read_bytes()).hexdigest(),
                  "annotations_source": ANNOTATION_URL, "annotations_md5": ANNOTATION_MD5,
                  "splits": {key: len(value) for key, value in sets.items()},
                  "images_sha256_ordered_by_name": digest.hexdigest(),
                  "note": "镜像图片已还原；仅使用 Oxford 官方 trainval/test 划分。"}
    (BASE / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(provenance, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-only", action="store_true")
    main(parser.parse_args().annotations_only)
