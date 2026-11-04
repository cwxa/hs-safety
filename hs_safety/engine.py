"""从冻结隐状态缓存训练轻量探针、校准阈值并输出可审计结果。"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.nn.utils.rnn import pad_sequence

from .config import experiment_signature
from .features import load_feature_split
from .io import save_json as _atomic_save_json
from .io import write_jsonl as _atomic_write_jsonl
from .metrics import (
    aggregate_metrics,
    calibrate_threshold,
    flatten_metrics,
    paired_group_bootstrap,
    response_metrics,
)
from .models import (
    DEFAULT_MODELS,
    MODEL_SPECS,
    build_network,
    cumulative_mean,
    prefix_indices,
    set_seed,
    text_vectorizer,
)

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "dev", "calibration", "test", "external")


def _signature(config: dict) -> str:
    return experiment_signature(config)


def _save_json(path: Path, data: dict) -> None:
    _atomic_save_json(path, data)


def _feature_fingerprint(config: dict):
    path = Path(config["paths"].get("features", "artifacts/features")) / "train" / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"signature": manifest.get("signature"), "projection_hash": manifest.get("projection_hash")}


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict]) -> None:
    _atomic_write_jsonl(path, rows)


def _model_dir(config: dict, name: str, seed: int) -> Path:
    return Path(config["paths"].get("models", "artifacts/models")) / name / f"seed_{seed}"


def _prediction_path(config: dict, name: str, seed: int, split: str) -> Path:
    return Path(config["paths"].get("results", "results")) / "predictions" / name / f"seed_{seed}" / f"{split}.jsonl"


def _load_split(config: dict, split: str, name: str):
    spec = MODEL_SPECS[name]
    return load_feature_split(config, split, stride=spec.stride,
                              final_layer_only=spec.final_layer_only, dynamic=spec.dynamic)


def _transformed_sequences(sequences: list[np.ndarray], name: str) -> list[np.ndarray]:
    return [cumulative_mean(values) for values in sequences] if MODEL_SPECS[name].family == "mean" else sequences


def _training_arrays(records: list[dict], sequences: list[np.ndarray], max_prefixes: int):
    values, labels, weights = [], [], []
    for record, sequence in zip(records, sequences):
        chosen = prefix_indices(len(sequence), max_prefixes)
        values.append(sequence[chosen])
        labels.extend([int(record["label"])] * len(chosen))
        weights.extend([1.0 / len(chosen)] * len(chosen))
    return np.concatenate(values), np.asarray(labels), np.asarray(weights)


def _tokenizer(config: dict):
    from transformers import AutoTokenizer
    model = config["model"]
    local = Path(model.get("local_path", "artifacts/backbones/qwen2.5-0.5b"))
    source = str(local) if local.exists() else model["id"]
    return AutoTokenizer.from_pretrained(source, cache_dir=model.get("cache_dir", ".cache/huggingface"), local_files_only=True)


def _texts(records: list[dict], positions: list[np.ndarray], family: str, tokenizer=None, maximum: int | None = None):
    texts, labels, weights = [], [], []
    for record, current_positions in zip(records, positions):
        chosen = prefix_indices(len(current_positions), maximum) if maximum else np.arange(len(current_positions))
        for index in chosen:
            prompt = str(record["prompt"])
            if family == "text":
                visible = tokenizer.decode(record["response_ids"][:int(current_positions[index])], skip_special_tokens=False)
                prompt = prompt + "\nAssistant: " + visible
            texts.append(prompt)
            labels.append(int(record["label"]))
            weights.append(1.0 / len(chosen))
    return texts, np.asarray(labels), np.asarray(weights)


def _device(config: dict) -> torch.device:
    requested = str(config.get("training", {}).get("device", "cpu"))
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def _scale_sequences(scaler: StandardScaler, sequences: list[np.ndarray]) -> list[np.ndarray]:
    return [scaler.transform(sequence).astype(np.float32) for sequence in sequences]


def _neural_scores(network, sequences: list[np.ndarray], device: torch.device, batch_size: int) -> list[np.ndarray]:
    network.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            padded = pad_sequence([torch.from_numpy(values) for values in batch], batch_first=True).to(device)
            probabilities = torch.sigmoid(network(padded)).cpu().numpy()
            output.extend(probabilities[i, :len(values)].astype(np.float64) for i, values in enumerate(batch))
    return output


def _fit_neural(config: dict, name: str, seed: int, train, dev, directory: Path) -> dict:
    settings = config.get("training", {})
    train_records, train_sequences, _ = train
    dev_records, dev_sequences, _ = dev
    # 标准化也只拟合训练检查点，按回答等权，避免长回复主导预处理。
    sampled, _, weights = _training_arrays(train_records, train_sequences, int(settings.get("max_prefixes", 8)))
    scaler = StandardScaler().fit(sampled, sample_weight=weights)
    train_values = _scale_sequences(scaler, train_sequences)
    dev_values = _scale_sequences(scaler, dev_sequences)
    device = _device(config)
    input_size, hidden_size = train_values[0].shape[1], int(settings.get("hidden_size", 64))
    network = build_network(MODEL_SPECS[name].family, input_size, hidden_size).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(settings.get("lr", 1e-3)), weight_decay=float(settings.get("weight_decay", 1e-4)))
    labels = np.asarray([int(record["label"]) for record in train_records])
    positive = int(labels.sum())
    if positive == 0 or positive == len(labels):
        raise ValueError("Training requires both safe and unsafe final-response labels.")
    positive_weight = torch.tensor((len(labels) - positive) / positive, dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")
    batch_size = int(settings.get("batch_size", 32))
    maximum = int(settings.get("max_prefixes", 8))
    best_score, best_state, patience = -float("inf"), None, 0
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(int(settings.get("epochs", 20))):
        started = time.perf_counter()
        network.train()
        order = rng.permutation(len(train_values))
        loss_total, count = 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            current = [train_values[i] for i in indices]
            padded = pad_sequence([torch.from_numpy(values) for values in current], batch_first=True).to(device)
            logits = network(padded)
            losses = []
            for row_index, response_index in enumerate(indices):
                supervised = torch.as_tensor(prefix_indices(len(current[row_index]), maximum), device=device)
                predictions = logits[row_index, supervised]
                targets = torch.full_like(predictions, float(labels[response_index]))
                losses.append(criterion(predictions, targets).mean())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += float(loss.item()) * len(indices)
            count += len(indices)
        dev_scores = _neural_scores(network, dev_values, device, batch_size)
        dev_labels = np.asarray([int(record["label"]) for record in dev_records])
        auc = float(roc_auc_score(dev_labels, [max(values) for values in dev_scores])) if len(np.unique(dev_labels)) == 2 else 0.5
        epoch_info = {"epoch": epoch + 1, "loss": loss_total / count, "dev_maxscore_roc_auc": auc, "elapsed_seconds": time.perf_counter() - started}
        history.append(epoch_info)
        LOGGER.info("Training checkpoint model=%s seed=%d epoch=%d loss=%.6f dev_maxscore_auc=%.6f elapsed_seconds=%.3f", name, seed, epoch + 1, epoch_info["loss"], auc, epoch_info["elapsed_seconds"])
        if auc > best_score + float(settings.get("min_delta", 1e-4)):
            best_score = auc
            best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(settings.get("patience", 3)):
            break
    torch.save(best_state, directory / "network.pt")
    joblib.dump(scaler, directory / "preprocessor.joblib")
    return {"input_size": input_size, "hidden_size": hidden_size, "device": str(device), "best_dev_maxscore_roc_auc": best_score, "history": history}


def _fit_linear(config: dict, name: str, seed: int, train, directory: Path) -> dict:
    settings = config.get("training", {})
    records, sequences, positions = train
    family = MODEL_SPECS[name].family
    maximum = int(settings.get("max_prefixes", 8))
    if family in ("prompt", "text"):
        tokenizer = _tokenizer(config) if family == "text" else None
        texts, labels, weights = _texts(records, positions, family, tokenizer, maximum)
        preprocessor = text_vectorizer(int(settings.get("tfidf_max_features", 20000)))
        values = preprocessor.fit_transform(texts)
    else:
        sequences = _transformed_sequences(sequences, name)
        values, labels, weights = _training_arrays(records, sequences, maximum)
        preprocessor = StandardScaler().fit(values, sample_weight=weights)
        values = preprocessor.transform(values)
    if len(np.unique(labels)) < 2:
        raise ValueError("Training requires both safe and unsafe final-response labels.")
    # 类别权重以回答计数确定；prefix 权重之和对每条回答均为一。
    response_labels = np.asarray([int(record["label"]) for record in records])
    n_positive = int(response_labels.sum())
    class_weight = {0: len(records) / (2 * (len(records) - n_positive)), 1: len(records) / (2 * n_positive)}
    estimator = LogisticRegression(C=float(settings.get("lr_C", 1.0)), max_iter=int(settings.get("lr_max_iter", 500)),
                                   solver="liblinear", class_weight=class_weight, random_state=seed)
    with threadpool_limits(limits=1):
        estimator.fit(values, labels, sample_weight=weights)
    joblib.dump(preprocessor, directory / "preprocessor.joblib")
    joblib.dump(estimator, directory / "estimator.joblib")
    return {"input_size": values.shape[1], "n_supervised_prefixes": len(labels), "max_iter": estimator.max_iter,
            "n_iter": estimator.n_iter_.tolist(), "class_weight": class_weight}


def _predict(config: dict, name: str, seed: int, split_data) -> list[dict]:
    started = time.perf_counter()
    directory = _model_dir(config, name, seed)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["config_signature"] != _signature(config):
        raise ValueError(f"Model configuration signature mismatch: {directory}")
    if metadata.get("feature_cache_signature") != _feature_fingerprint(config):
        raise ValueError(f"Model training feature-cache signature mismatch: {directory}")
    records, sequences, positions = split_data
    if not records:
        return []
    family = MODEL_SPECS[name].family
    preprocessor = joblib.load(directory / "preprocessor.joblib")
    if family in ("gru", "mlp"):
        device = _device(config)
        network = build_network(family, metadata["input_size"], metadata["hidden_size"]).to(device)
        network.load_state_dict(torch.load(directory / "network.pt", map_location=device, weights_only=True))
        scores = _neural_scores(network, _scale_sequences(preprocessor, sequences), device, int(config.get("training", {}).get("batch_size", 32)))
    else:
        estimator = joblib.load(directory / "estimator.joblib")
        if family in ("prompt", "text"):
            tokenizer = _tokenizer(config) if family == "text" else None
            texts, _, _ = _texts(records, positions, family, tokenizer)
            with threadpool_limits(limits=1):
                flat_scores = estimator.predict_proba(preprocessor.transform(texts))[:, 1]
            boundaries = np.cumsum([0] + [len(values) for values in positions])
            scores = [flat_scores[boundaries[i]:boundaries[i + 1]] for i in range(len(records))]
        else:
            sequences = _transformed_sequences(sequences, name)
            with threadpool_limits(limits=1):
                scores = [estimator.predict_proba(preprocessor.transform(values))[:, 1] for values in sequences]
    rows = []
    for record, current_positions, current_scores in zip(records, positions, scores):
        rows.append({"id": record["id"], "group": record["group"], "label": int(record["label"]),
                     "response_len": int(record["response_len"]), "categories": record.get("categories", []),
                     "positions": np.asarray(current_positions, dtype=np.int64).tolist(),
                     "scores": np.asarray(current_scores, dtype=np.float64).tolist()})
    LOGGER.info("Prediction completed model=%s seed=%d responses=%d checkpoints=%d mode=offline_batch_replay elapsed_seconds=%.3f", name, seed, len(rows), sum(len(row["scores"]) for row in rows), time.perf_counter() - started)
    return rows


def _benchmark_neural_detector(config: dict, name: str, seed: int, split_data) -> dict:
    """实测 batch=1 因果状态推进；不把特征缓存回放速度冒充生成速度。"""

    directory = _model_dir(config, name, seed)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    device = _device(config)
    network = build_network(MODEL_SPECS[name].family, metadata["input_size"], metadata["hidden_size"]).to(device)
    network.load_state_dict(torch.load(directory / "network.pt", map_location=device, weights_only=True))
    network.eval()
    scaler = joblib.load(directory / "preprocessor.joblib")
    maximum = int(config.get("evaluation", {}).get("latency_responses", 16))
    sequences = split_data[1][:max(1, maximum)]
    if not sequences:
        return {}
    elapsed, online, checkpoint_count = [], [], 0
    with torch.inference_mode():
        # 预热模型与设备；每条回复的循环状态仍从空状态重新开始。
        warmup = torch.zeros((1, 1, metadata["input_size"]), device=device)
        for _ in range(5):
            network(warmup)
        for sequence in sequences:
            hidden, current_scores = None, []
            for vector in sequence:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                scaled = scaler.transform(vector[None, :]).astype(np.float32)
                values = torch.from_numpy(scaled).unsqueeze(0).to(device)
                if MODEL_SPECS[name].family == "gru":
                    state, hidden = network.gru(values, hidden)
                    score = torch.sigmoid(network.head(state)).item()
                else:
                    score = torch.sigmoid(network(values)).item()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter() - started)
                current_scores.append(score)
                checkpoint_count += 1
            online.append(np.asarray(current_scores))
    offline = _neural_scores(network, _scale_sequences(scaler, sequences), device, batch_size=1)
    error = max(float(np.max(np.abs(left - right))) for left, right in zip(online, offline))
    if error > 1e-4:
        raise ValueError(f"Online/offline detector discrepancy exceeds tolerance: model={name} error={error}")
    return {"mode": "cached_feature_batch1_streaming_detector", "device": str(device),
            "n_responses": len(sequences), "n_checkpoints": checkpoint_count,
            "checkpoint_ms_p50": float(np.median(elapsed) * 1000),
            "checkpoint_ms_p95": float(np.quantile(elapsed, 0.95) * 1000),
            "checkpoint_ms_mean": float(np.mean(elapsed) * 1000),
            "online_offline_max_abs_score_error": error,
            "includes": "StandardScaler, host/device input transfer, causal detector forward and scalar readback",
            "excludes": "backbone inference, tokenizer, PCA and dynamic feature extraction; no generation throughput claim"}


class StreamingDetector:
    """同训练预处理与阈值共享的在线检查点接口；每次新回复必须 reset。"""

    def __init__(self, config: dict, model_name: str = "gru", seed: int = 42):
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        self.family = MODEL_SPECS[model_name].family
        if self.family in ("prompt", "text"):
            raise ValueError("Text baselines require observed text and do not support the numeric feature API.")
        self.model_name = model_name
        self.seed = int(seed)
        torch.set_num_threads(max(1, int(config.get("training", {}).get("num_threads", config.get("training", {}).get("threads", 2)))))
        self.device = _device(config)
        directory = _model_dir(config, model_name, self.seed)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        signature = _signature(config)
        if self.metadata["config_signature"] != signature or calibration["config_signature"] != signature:
            raise ValueError("Streaming detector model/calibration configuration signature mismatch.")
        current_fingerprint = _feature_fingerprint(config)
        if current_fingerprint is not None and self.metadata.get("feature_cache_signature") != current_fingerprint:
            raise ValueError("Streaming detector training feature-cache signature mismatch.")
        self.threshold = float(calibration["threshold"])
        self.input_size = int(self.metadata["input_size"])
        self.preprocessor = joblib.load(directory / "preprocessor.joblib")
        self.network = None
        self.estimator = None
        if self.family in ("gru", "mlp"):
            self.network = build_network(self.family, self.input_size, int(self.metadata["hidden_size"])).to(self.device)
            self.network.load_state_dict(torch.load(directory / "network.pt", map_location=self.device, weights_only=True))
            self.network.eval()
        else:
            self.estimator = joblib.load(directory / "estimator.joblib")
        self.reset()

    def reset(self) -> None:
        self.hidden = None
        self.checkpoints_seen = 0
        self.feature_sum = None
        self.alerted = False

    def update(self, feature) -> dict:
        """输入一个仅依赖当前前缀的数值向量，返回分数与本次触发状态。"""

        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.input_size or not np.isfinite(vector).all():
            raise ValueError(f"Expected {self.input_size} finite feature values, received shape {vector.shape}.")
        self.checkpoints_seen += 1
        if self.family == "mean":
            self.feature_sum = vector.astype(np.float64) if self.feature_sum is None else self.feature_sum + vector
            vector = (self.feature_sum / self.checkpoints_seen).astype(np.float32)
        scaled = self.preprocessor.transform(vector[None, :]).astype(np.float32)
        if self.network is not None:
            with torch.inference_mode():
                values = torch.from_numpy(scaled).unsqueeze(0).to(self.device)
                if self.family == "gru":
                    state, self.hidden = self.network.gru(values, self.hidden)
                    score = float(torch.sigmoid(self.network.head(state)).item())
                else:
                    score = float(torch.sigmoid(self.network(values)).item())
        else:
            with threadpool_limits(limits=1):
                score = float(self.estimator.predict_proba(scaled)[0, 1])
        alert = score > self.threshold
        self.alerted = self.alerted or alert
        return {"score": score, "alert": alert}


def train_all(config: dict, models=None, seeds=None) -> dict:
    """训练默认主基线与消融；每次完整落盘后输出各分区分数轨迹。"""

    names = list(models or config.get("training", {}).get("models", DEFAULT_MODELS))
    chosen_seeds = [int(seed) for seed in (seeds or config.get("training", {}).get("seeds", [13, 42, 2026]))]
    summary = {}
    for name in names:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {name}. Available models: {', '.join(MODEL_SPECS)}")
        started = time.perf_counter()
        train, dev = _load_split(config, "train", name), _load_split(config, "dev", name)
        if not train[0] or not dev[0]:
            raise ValueError("Training and development splits must be nonempty.")
        for seed in chosen_seeds:
            seed_started = time.perf_counter()
            set_seed(seed, int(config.get("training", {}).get("num_threads", config.get("training", {}).get("threads", 2))))
            directory = _model_dir(config, name, seed)
            directory.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Training started model=%s seed=%d n_train=%d n_dev=%d input_size=%d", name, seed, len(train[0]), len(dev[0]), train[1][0].shape[1])
            metadata = {"model": name, "seed": seed, "family": MODEL_SPECS[name].family,
                        "config_signature": _signature(config), "feature_configuration": vars(MODEL_SPECS[name]),
                        "feature_cache_signature": _feature_fingerprint(config),
                        "feature_path": str(Path(config["paths"].get("features", "artifacts/features")).resolve()),
                        "preprocessor_path": str((directory / "preprocessor.joblib").resolve()),
                        "target": "Y_final", "supervision": "response-balanced weak prefix supervision",
                        "optimizer": "AdamW" if MODEL_SPECS[name].family in ("gru", "mlp") else "liblinear",
                        "n_train": len(train[0]), "n_dev": len(dev[0])}
            fit = _fit_neural(config, name, seed, train, dev, directory) if MODEL_SPECS[name].family in ("gru", "mlp") else _fit_linear(config, name, seed, train, directory)
            metadata.update(fit)
            metadata["train_seconds"] = time.perf_counter() - seed_started
            metadata["model_path"] = str((directory / ("network.pt" if MODEL_SPECS[name].family in ("gru", "mlp") else "estimator.joblib")).resolve())
            metadata["training_data_signature"] = hashlib.sha256(json.dumps([(record["id"], record["group"], int(record["label"])) for record in train[0]], sort_keys=True).encode("utf-8")).hexdigest()
            _save_json(directory / "metadata.json", metadata)
            for split in SPLITS:
                try:
                    values = train if split == "train" else dev if split == "dev" else _load_split(config, split, name)
                except FileNotFoundError:
                    if split == "external":
                        LOGGER.info("Optional external split unavailable model=%s seed=%d", name, seed)
                        continue
                    raise
                if values[0]:
                    rows = _predict(config, name, seed, values)
                    _write_rows(_prediction_path(config, name, seed, split), rows)
            LOGGER.info("Training completed model=%s seed=%d directory=%s", name, seed, directory)
        summary[name] = {"seeds": chosen_seeds, "elapsed_seconds": time.perf_counter() - started}
    _save_json(Path(config["paths"].get("results", "results")) / "training_summary.json", summary)
    return summary


def _available(config: dict):
    root = Path(config["paths"].get("models", "artifacts/models"))
    found = []
    for metadata_path in sorted(root.glob("*/seed_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_signature") == _signature(config):
            if metadata.get("feature_cache_signature") != _feature_fingerprint(config):
                raise ValueError(f"Model training feature-cache signature mismatch: {metadata_path}")
            found.append((metadata["model"], int(metadata["seed"]), metadata_path.parent))
    if not found:
        raise FileNotFoundError(f"No trained models matching this configuration were found in {root}.")
    return found


def _ensure_predictions(config: dict, name: str, seed: int, split: str) -> list[dict]:
    path = _prediction_path(config, name, seed, split)
    if path.exists():
        return _read_rows(path)
    rows = _predict(config, name, seed, _load_split(config, split, name))
    _write_rows(path, rows)
    return rows


def calibrate_all(config: dict) -> dict:
    summary = {}
    target = float(config.get("evaluation", {}).get("fpr", 0.05))
    for name, seed, directory in _available(config):
        rows = _ensure_predictions(config, name, seed, "calibration")
        calibration = calibrate_threshold(rows, target)
        calibration.update({"model": name, "seed": seed, "config_signature": _signature(config), "split": "calibration"})
        _save_json(directory / "calibration.json", calibration)
        summary.setdefault(name, {})[str(seed)] = calibration
        LOGGER.info("Calibration completed model=%s seed=%d threshold=%.8f n_safe=%d sequence_fpr=%.6f target_fpr=%.6f", name, seed, calibration["threshold"], calibration["n_safe"], calibration["empirical_sequence_fpr"], target)
    _save_json(Path(config["paths"].get("results", "results")) / "calibration_summary.json", summary)
    return summary


def evaluate_all(config: dict) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(max(1, int(config.get("training", {}).get("num_threads", config.get("training", {}).get("threads", 2)))))
    settings = config.get("evaluation", {})
    budgets = [int(value) for value in settings.get("budgets", [32, 64, 128, 256])]
    output = {"schema_version": 1, "task": "final_response_risk_prediction", "config_signature": _signature(config),
              "target_fpr": float(settings.get("fpr", 0.05)), "models": {}, "paired_comparisons": [],
              "notes": ["All prefix targets are final-response labels, not token-level harmfulness labels.",
                        "Thresholds are fixed using calibration safe-response maximum scores and strict > comparison.",
                        "Detection positions are conditional on detecting unsafe final responses; they do not measure harmful onset.",
                        "Bootstrap samples prompt groups jointly across models and seeds; thresholds are not refitted."]}
    trajectories = {}
    csv_rows = []
    for name, seed, directory in _available(config):
        calibration_path = directory / "calibration.json"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration is required before evaluation: {calibration_path}")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["config_signature"] != _signature(config):
            raise ValueError(f"Calibration configuration signature mismatch: {calibration_path}")
        threshold = float(calibration["threshold"])
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        entry = {"threshold": threshold, "calibration": calibration, "splits": {},
                 "train_seconds": metadata.get("train_seconds"),
                 "training_epochs": len(metadata.get("history", [])) if "history" in metadata else None}
        for split in ("dev", "calibration", "test", "external"):
            try:
                rows = _ensure_predictions(config, name, seed, split)
            except FileNotFoundError:
                if split == "external":
                    continue
                raise
            if not rows:
                continue
            metrics = response_metrics(rows, threshold, budgets)
            entry["splits"][split] = metrics
            trajectories.setdefault(split, {}).setdefault(name, {})[seed] = (rows, threshold)
            csv_rows.extend({"model": name, "seed": seed, "split": split, "metric": metric, "value": value} for metric, value in flatten_metrics(metrics).items())
        if MODEL_SPECS[name].family in ("gru", "mlp") and bool(settings.get("benchmark_detector", True)):
            entry["detector_latency"] = _benchmark_neural_detector(config, name, seed, _load_split(config, "test", name))
        output["models"].setdefault(name, {"seeds": {}, "aggregate": {}})["seeds"][str(seed)] = entry
    for name, values in output["models"].items():
        all_splits = sorted({split for entry in values["seeds"].values() for split in entry["splits"]})
        for split in all_splits:
            values["aggregate"][split] = aggregate_metrics([entry["splits"][split] for entry in values["seeds"].values() if split in entry["splits"]])
    for split in ("test", "external"):
        candidates = trajectories.get(split, {})
        if "gru" not in candidates:
            continue
        for reference in sorted(set(candidates) - {"gru"}):
            with threadpool_limits(limits=1):
                comparisons = paired_group_bootstrap(candidates[reference], candidates["gru"], budgets,
                    int(settings.get("bootstrap", 1000)), int(settings.get("bootstrap_seed", 2026)))
            for comparison in comparisons:
                comparison.update({"split": split, "reference": reference, "challenger": "gru"})
            output["paired_comparisons"].extend(comparisons)
    output["timing"] = {"evaluation_elapsed_seconds": time.perf_counter() - started}
    root = Path(config["paths"].get("results", "results"))
    _save_json(root / "metrics.json", output)
    with (root / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "seed", "split", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    LOGGER.info("Evaluation completed models=%d paired_comparisons=%d elapsed_seconds=%.3f output=%s", len(output["models"]), len(output["paired_comparisons"]), output["timing"]["evaluation_elapsed_seconds"], root / "metrics.json")
    return output
