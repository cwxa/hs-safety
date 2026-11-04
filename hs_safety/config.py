"""读取配置并尽早验证会影响实验含义的参数。"""
from pathlib import Path

import yaml

from .io import hash_object


def experiment_signature(config):
    """训练、推理及报告共用的实验身份，避免跨阶段重复实现。"""
    selected = {key: config.get(key, {}) for key in ("model", "features", "training", "data")}
    selected["implementation"] = "causal-probes-v1-adamw"
    return hash_object(selected)


def load_config(path="configs/main.yaml"):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a mapping")
    base = int(config["features"]["base_stride"])
    stride = int(config["features"]["stride"])
    if base <= 0 or stride <= 0 or stride % base:
        raise ValueError("Detection stride must be a positive multiple of base_stride")
    if not 0 < float(config["evaluation"]["fpr"]) < 1:
        raise ValueError("Target FPR must lie strictly between zero and one")
    if int(config["features"]["pca_components"]) < 1:
        raise ValueError("PCA components must be positive")
    layers = config["model"]["layers"]
    if sorted(set(layers)) != layers or layers[0] < 1:
        raise ValueError("Layers must be sorted unique one-based block indices")
    return config


def ensure_directories(config):
    for path in config["paths"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    Path(config["model"]["cache_dir"]).mkdir(parents=True, exist_ok=True)
