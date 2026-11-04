"""训练集限定的增量PCA，以及离线/在线共用的因果动态特征。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import checkpoint_positions
from .extraction import _save_array, iter_raw_sequences, raw_signature, validate_raw_manifest
from .io import get_logger, hash_file, hash_object, load_json, read_jsonl, save_json, timed


def projection_path(config):
    return Path(config["paths"]["preprocessing"]) / "projection.npz"


def uniform_indices(length, maximum=8):
    if length < 1:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, min(length, maximum)).round().astype(np.int64))


def project_hidden(hidden, projection):
    """固定训练统计量逐位置变换，不依赖当前回答的未来或总长度。"""
    hidden = np.asarray(hidden, dtype=np.float32)
    result = np.einsum("...lh,lch->...lc", hidden - projection["mean"], projection["components"], optimize=True)
    return (result / projection["scale"]).astype(np.float32)


def causal_dynamic_features(projected, dynamic=True, final_layer_only=False):
    """变化量定义于投影空间；第一检查点差分置零，避免伪造起始前史。"""
    projected = np.asarray(projected, dtype=np.float32)
    if projected.ndim != 3:
        raise ValueError("Projected hidden states must have shape checkpoints x layers x components")
    if final_layer_only:
        projected = projected[:, -1:, :]
    flat = projected.reshape(len(projected), -1)
    if not dynamic:
        return flat.copy()
    previous = np.concatenate([projected[:1], projected[:-1]], axis=0)
    delta = projected - previous
    norm_now = np.linalg.norm(projected, axis=-1)
    norm_old = np.linalg.norm(previous, axis=-1)
    cosine = 1.0 - np.sum(projected * previous, axis=-1) / np.maximum(norm_now * norm_old, 1e-8)
    norm_difference = norm_now - norm_old
    cosine[0], norm_difference[0] = 0.0, 0.0
    values = np.concatenate([flat, delta.reshape(len(projected), -1), cosine, norm_difference], axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite dynamic feature encountered")
    return values.astype(np.float32)


class CausalFeatureBuilder:
    """在线每次只接受一个已闭合检查点，与离线特征构造完全相同。"""

    def __init__(self, projection, dynamic=True, final_layer_only=False):
        self.projection = projection
        self.dynamic = dynamic
        self.final_layer_only = final_layer_only
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, hidden):
        # 与落盘FP16投影使用同一量化路径，消除离线/在线精度协议差异。
        current = project_hidden(np.asarray(hidden, dtype=np.float16)[None], self.projection)[0].astype(np.float16).astype(np.float32)
        pair = current[None] if self.previous is None else np.stack([self.previous, current])
        feature = causal_dynamic_features(pair, self.dynamic, self.final_layer_only)[-1]
        self.previous = current
        return feature


def load_projection(config):
    with np.load(projection_path(config), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def fit_projection(config):
    from sklearn.decomposition import IncrementalPCA
    from threadpoolctl import threadpool_limits

    logger = get_logger("features")
    raw_dir = Path(config["paths"]["raw_features"]) / "train"
    raw_manifest = validate_raw_manifest(raw_dir, verify_files=False)
    if not raw_manifest.get("complete"):
        raise ValueError("Training raw cache is incomplete")
    signature = hash_object({"raw_train": raw_manifest["signature"],
                             "components": config["features"]["pca_components"],
                             "fit_prefixes": config["features"]["pca_fit_prefixes"]})
    destination = projection_path(config)
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata["signature"] != signature or metadata["sha256"] != hash_file(destination):
            raise ValueError("Projection cache signature mismatch")
        logger.info("Verified existing train-only projection components=%d", config["features"]["pca_components"])
        return metadata
    components = int(config["features"]["pca_components"])
    batch_size = int(config["features"]["pca_batch_size"])
    n_layers = len(config["model"]["layers"])
    estimators = [IncrementalPCA(n_components=components, batch_size=batch_size) for _ in range(n_layers)]
    pending, count, fitted = [], 0, 0
    with timed(logger, "fit_train_only_projection", components=components):
        with threadpool_limits(limits=config["training"].get("num_threads", 4)):
            for _, hidden, _ in iter_raw_sequences(config, "train"):
                chosen = np.asarray(hidden[uniform_indices(len(hidden), config["features"]["pca_fit_prefixes"])], dtype=np.float32)
                pending.append(chosen)
                count += len(chosen)
                if count >= batch_size:
                    block = np.concatenate(pending)
                    for layer, estimator in enumerate(estimators):
                        estimator.partial_fit(block[:, layer])
                    fitted += len(block)
                    pending, count = [], 0
            if count >= components:
                block = np.concatenate(pending)
                for layer, estimator in enumerate(estimators):
                    estimator.partial_fit(block[:, layer])
                fitted += len(block)
            elif fitted == 0:
                raise ValueError(f"PCA needs at least {components} training checkpoints; got {count}")
        projection = {"components": np.stack([est.components_ for est in estimators]).astype(np.float32),
                      "mean": np.stack([est.mean_ for est in estimators]).astype(np.float32),
                      "scale": np.sqrt(np.maximum(np.stack([est.explained_variance_ for est in estimators]), 1e-8)).astype(np.float32)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez(stream, **projection)
        metadata = {"schema_version": 1, "signature": signature, "sha256": hash_file(destination),
                    "fit_split": "train", "training_cache_signature": raw_manifest["signature"],
                    "n_fit_checkpoints": fitted, "unused_tail_checkpoints": count if count < components else 0,
                    "components_per_layer": components, "layers": config["model"]["layers"],
                    "whiten": True,
                    "explained_variance_ratio": [float(est.explained_variance_ratio_.sum()) for est in estimators]}
        save_json(metadata_path, metadata)
    return metadata


def transform_feature_caches(config, splits=None):
    logger = get_logger("features")
    projection = load_projection(config)
    projection_hash = hash_file(projection_path(config))
    for split in splits or ["train", "dev", "calibration", "test", "external"]:
        raw_dir = Path(config["paths"]["raw_features"]) / split
        if not (raw_dir / "manifest.json").exists():
            continue
        raw_manifest = validate_raw_manifest(raw_dir, verify_files=False)
        if not raw_manifest.get("complete"):
            raise ValueError(f"Cannot project incomplete split={split}")
        signature = hash_object({"raw_signature": raw_manifest["signature"], "projection": projection_hash})
        destination = Path(config["paths"]["features"]) / split
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "manifest.json"
        if path.exists():
            manifest = load_json(path)
            if manifest["signature"] != signature:
                raise ValueError(f"Projected feature signature mismatch split={split}")
            if manifest.get("complete"):
                if all((destination / filename).exists() and hash_file(destination / filename) == digest
                       for shard in manifest["shards"] for filename, digest in shard["files"].items()):
                    logger.info("Verified existing projected split=%s", split)
                    continue
                raise ValueError(f"Corrupt projected cache split={split}")
        else:
            manifest = {"schema_version": 1, "signature": signature, "raw_signature": raw_manifest["signature"],
                        "projection_hash": projection_hash, "layers": config["model"]["layers"],
                        "base_stride": config["features"]["base_stride"], "shards": [], "complete": False}
        committed = {shard["prefix"] for shard in manifest["shards"]}
        for shard in raw_manifest["shards"]:
            prefix = shard["prefix"]
            if prefix in committed:
                continue
            raw = np.load(raw_dir / f"{prefix}.values.npy", mmap_mode="r", allow_pickle=False)
            values = project_hidden(raw, projection).astype(np.float16)
            payload = {f"{prefix}.values.npy": values,
                       f"{prefix}.positions.npy": np.load(raw_dir / f"{prefix}.positions.npy", allow_pickle=False),
                       f"{prefix}.offsets.npy": np.load(raw_dir / f"{prefix}.offsets.npy", allow_pickle=False)}
            hashes = {}
            for filename, array in payload.items():
                _save_array(destination / filename, array)
                hashes[filename] = hash_file(destination / filename)
            manifest["shards"].append({"prefix": prefix, "ids": shard["ids"], "files": hashes})
            save_json(path, manifest)
        manifest["complete"] = True
        save_json(path, manifest)
        logger.info("Projected cache completed split=%s shards=%d", split, len(manifest["shards"]))


def build_features(config):
    fit_projection(config)
    transform_feature_caches(config)


def load_feature_split(config, split, stride=None, final_layer_only=False, dynamic=True):
    records_path = Path(config["paths"]["processed"]) / f"{split}.jsonl"
    if not records_path.exists():
        return [], [], []
    records = read_jsonl(records_path)
    if not records:
        return [], [], []
    destination = Path(config["paths"]["features"]) / split
    manifest = load_json(destination / "manifest.json")
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete projected cache split={split}")
    if manifest["raw_signature"] != raw_signature(config, records):
        raise ValueError(f"Processed token inputs do not match the cached representation split={split}")
    audit_path = Path(config["paths"]["processed"]) / "audit.json"
    if audit_path.exists():
        expected_records = load_json(audit_path).get("records_hash", {}).get(split)
        if expected_records is not None and expected_records != hash_object(records):
            raise ValueError(f"Processed labels, groups or text differ from the preparation audit split={split}")
    if manifest["projection_hash"] != hash_file(projection_path(config)):
        raise ValueError("Projection does not match feature cache")
    stride = int(stride or config["features"]["stride"])
    if stride % manifest["base_stride"]:
        raise ValueError("Requested stride must be a multiple of the cached base stride")
    lookup, sequences, positions = {}, [], []
    for shard in manifest["shards"]:
        prefix = shard["prefix"]
        values = np.load(destination / f"{prefix}.values.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(destination / f"{prefix}.offsets.npy", allow_pickle=False)
        token_positions = np.load(destination / f"{prefix}.positions.npy", allow_pickle=False)
        for index, record_id in enumerate(shard["ids"]):
            left, right = offsets[index:index + 2]
            lookup[record_id] = (np.asarray(values[left:right], dtype=np.float32), token_positions[left:right])
    if set(lookup) != {r["id"] for r in records}:
        raise ValueError(f"Record/cache identity mismatch split={split}")
    for record in records:
        projected, cached_positions = lookup[record["id"]]
        requested = checkpoint_positions(record["response_len"], stride)
        indices = np.searchsorted(cached_positions, requested)
        if np.any(indices >= len(cached_positions)) or not np.array_equal(cached_positions[indices], requested):
            raise ValueError("Requested token checkpoints are missing from the cache")
        sequences.append(causal_dynamic_features(projected[indices], dynamic, final_layer_only))
        positions.append(requested)
    return records, sequences, positions
