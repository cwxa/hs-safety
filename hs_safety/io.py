"""持久化、日志和可复现性基础工具。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def hash_object(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    """先写临时文件，再原子替换，避免中断留下半个清单。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")
    os.replace(temp, path)


def get_logger(name="hs_safety"):
    return logging.getLogger(name)


def configure_logging(directory="logs", command="run"):
    Path(directory).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(Path(directory) / f"{command}.log", encoding="utf-8")],
        force=True,
    )


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # GRU 的 CPU 小矩阵使用固定线程，降低运行间时延波动。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timed(logger, checkpoint, **parameters):
    start = time.perf_counter()
    logger.info("START checkpoint=%s parameters=%s", checkpoint, parameters)
    try:
        yield
    except Exception:
        logger.exception("FAILED checkpoint=%s elapsed_seconds=%.3f", checkpoint, time.perf_counter() - start)
        raise
    logger.info("END checkpoint=%s elapsed_seconds=%.3f", checkpoint, time.perf_counter() - start)
