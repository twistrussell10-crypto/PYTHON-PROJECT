"""可选：从 Zenodo 的同一数据集镜像下载，支持分块重试与完整 MD5 校验。

用法：python scripts/download_mirror.py
只获取并解压经过校验的 ZIP；标准下载入口仍是 python -m pet_classifier prepare。
"""
import concurrent.futures
import hashlib
import math
import os
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "mirror-download"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata = requests.get("https://zenodo.org/api/records/8067751", timeout=60)
    metadata.raise_for_status()
    file = metadata.json()["files"][0]
    url, total = file["links"]["self"], file["size"]
    checksum = file["checksum"].removeprefix("md5:")
    size = 32 * 1024 * 1024

    def download_part(index):
        start, end = index * size, min(total, (index + 1) * size) - 1
        part = CACHE / f"part-{index:03}.bin"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(4):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise RuntimeError("镜像未返回所请求的精确字节范围")
                    with part.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                if part.stat().st_size != end - start + 1:
                    raise RuntimeError("下载分块不完整")
                print(f"Part {index + 1}/{math.ceil(total / size)} complete", flush=True)
                return part
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    archive = CACHE / "pets.zip"
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(download_part, range(math.ceil(total / size))))
    digest = hashlib.md5()
    with archive.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
    if digest.hexdigest() != checksum:
        raise RuntimeError("镜像 ZIP 的 MD5 校验失败，请重新下载缓存分块")
    print(f"MD5 verified: {checksum}", flush=True)
    # ZIP 中的路径必须位于独立缓存目录内部，禁止路径穿越。
    destination = (CACHE / "extracted").resolve()
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if os.path.commonpath([str(destination), str(target)]) != str(destination):
                raise ValueError("ZIP 包含越界路径")
        zipped.extractall(destination)
    print(f"Extracted mirror to {destination}", flush=True)
    from install_mirror import main as install_mirror
    install_mirror()


if __name__ == "__main__":
    main()
