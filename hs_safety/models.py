"""轻量安全探针；所有前缀预测目标均为完整回复的最终安全标签。"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    """固定各实验的可见信息与特征配置，避免比较时暗中改变输入。"""

    family: str
    dynamic: bool = True
    final_layer_only: bool = False
    stride: int | None = None


MODEL_SPECS = {
    "prompt_lr": ModelSpec("prompt", dynamic=False),
    "text_lr": ModelSpec("text", dynamic=False),
    "static_lr": ModelSpec("static", dynamic=False),
    "mean_lr": ModelSpec("mean", dynamic=False),
    "gru": ModelSpec("gru"),
    "gru_no_dynamics": ModelSpec("gru", dynamic=False),
    "gru_last_layer": ModelSpec("gru", final_layer_only=True),
    "mlp": ModelSpec("mlp"),
    "gru_stride4": ModelSpec("gru", stride=4),
    "gru_stride16": ModelSpec("gru", stride=16),
}
DEFAULT_MODELS = tuple(MODEL_SPECS)


def set_seed(seed: int, threads: int = 2) -> None:
    """固定随机源并限制线程；不同硬件的浮点舍入仍可能略有不同。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)


def prefix_indices(length: int, maximum: int) -> np.ndarray:
    """每条回复等距选取有限监督点，包含终点，不使长回复主导损失。"""

    if length < 1:
        raise ValueError("A feature sequence must contain at least one checkpoint.")
    return np.unique(np.linspace(0, length - 1, min(length, max(1, maximum))).round().astype(np.int64))


def cumulative_mean(sequence: np.ndarray) -> np.ndarray:
    """因果累计均值：第 t 行只汇总前 t 个检查点。"""

    return (np.cumsum(sequence, axis=0, dtype=np.float64) / np.arange(1, len(sequence) + 1)[:, None]).astype(np.float32)


def text_vectorizer(max_features: int = 20000) -> FeatureUnion:
    """英文词及字符 n-gram；词表与 IDF 仅由训练前缀拟合。"""

    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, sublinear_tf=True)),
        ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=max_features, sublinear_tf=True)),
    ])


class SequenceGRU(nn.Module):
    """单向 GRU；右侧 padding 不会影响任何有效前缀的预测。"""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(values)
        return self.head(states).squeeze(-1)


class PrefixMLP(nn.Module):
    """只观察当前检查点动态特征的非线性对照，不额外累积序列记忆。"""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def build_network(family: str, input_size: int, hidden_size: int) -> nn.Module:
    if family == "gru":
        return SequenceGRU(input_size, hidden_size)
    if family == "mlp":
        return PrefixMLP(input_size, hidden_size)
    raise ValueError(f"Unknown neural model family: {family}")
