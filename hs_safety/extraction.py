"""冻结因果模型的分片隐状态提取与可校验断点恢复。"""
from __future__ import annotations

import gc
import os
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from .data import checkpoint_positions
from .io import get_logger, hash_file, hash_object, load_json, read_jsonl, save_json, timed


def resolve_device(config):
    import torch

    device = config["model"].get("device", "auto")
    return "cuda:0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)


def load_backbone(config):
    import torch
    from transformers import AutoModelForCausalLM

    options = config["model"]
    local = Path(options.get("local_path", ""))
    location = str(local) if (local / "config.json").exists() else options["id"]
    device = resolve_device(config)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        location, local_files_only=True, cache_dir=options.get("cache_dir"),
        torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map={"": device}, attn_implementation="sdpa",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if max(options["layers"]) > model.config.num_hidden_layers:
        raise ValueError("Selected layer exceeds backbone depth")
    get_logger("extraction").info("Backbone loaded model=%s device=%s dtype=%s", location, device, dtype)
    return model


def hidden_states_at_tokens(model, prompt_ids, response_ids, layers, positions, device=None):
    """一次因果前向只取已消费位置；调用decoder避免构建全部词表logits。"""
    import torch

    device = device or next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, output_hidden_states=True, return_dict=True)
        indices = torch.as_tensor(np.asarray(positions) + len(prompt_ids) - 1, dtype=torch.long, device=device)
        selected = torch.stack([output.hidden_states[layer][0].index_select(0, indices) for layer in layers], dim=1)
        result = selected.to(dtype=torch.float16, device="cpu").numpy()
    return result


def _save_array(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def raw_signature(config, records):
    return hash_object({"schema_version": 1, "model": config["model"],
                        "base_stride": config["features"]["base_stride"],
                        "records": [[r["id"], r["prompt_ids"], r["response_ids"]] for r in records]})


def validate_raw_manifest(directory, expected_signature=None, verify_files=True):
    directory = Path(directory)
    manifest = load_json(directory / "manifest.json")
    if expected_signature is not None and manifest["signature"] != expected_signature:
        raise ValueError("Raw cache signature mismatch; use a separate output directory")
    all_ids = []
    for shard in manifest["shards"]:
        for name, digest in shard["files"].items():
            path = directory / name
            if not path.exists() or (verify_files and hash_file(path) != digest):
                raise ValueError(f"Incomplete or corrupt cache shard: {path}")
        prefix = shard["prefix"]
        values = np.load(directory / f"{prefix}.values.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(directory / f"{prefix}.offsets.npy", allow_pickle=False)
        positions = np.load(directory / f"{prefix}.positions.npy", allow_pickle=False)
        if (values.ndim != 3 or len(offsets) != len(shard["ids"]) + 1 or offsets[0] != 0
                or offsets[-1] != len(values) or len(positions) != len(values)
                or np.any(np.diff(offsets) <= 0)):
            raise ValueError(f"Invalid raw cache offsets or dimensions: {directory / prefix}")
        for left, right in pairwise(offsets):
            if positions[left] < 1 or np.any(np.diff(positions[left:right]) <= 0):
                raise ValueError("Invalid raw cache checkpoint ordering")
        all_ids.extend(shard["ids"])
    if len(all_ids) != len(set(all_ids)) or all_ids != manifest["record_ids"][:len(all_ids)]:
        raise ValueError("Raw cache record IDs are duplicated or out of order")
    if manifest.get("complete") and all_ids != manifest["record_ids"]:
        raise ValueError("Completed raw cache does not cover all records")
    return manifest


def iter_raw_sequences(config, split):
    directory = Path(config["paths"]["raw_features"]) / split
    manifest = validate_raw_manifest(directory, verify_files=False)
    for shard in manifest["shards"]:
        name = shard["prefix"]
        values = np.load(directory / f"{name}.values.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(directory / f"{name}.offsets.npy", allow_pickle=False)
        positions = np.load(directory / f"{name}.positions.npy", allow_pickle=False)
        for index, record_id in enumerate(shard["ids"]):
            left, right = offsets[index:index + 2]
            yield record_id, np.asarray(values[left:right]), positions[left:right]


def extract_features(config, splits=None, limit=None):
    import torch

    logger = get_logger("extraction")
    processed = Path(config["paths"]["processed"])
    splits = splits or ["train", "dev", "calibration", "test", "external"]
    pending = []
    for split in splits:
        path = processed / f"{split}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path)
        if limit is not None:
            records = records[:limit]
        destination = Path(config["paths"]["raw_features"]) / split
        destination.mkdir(parents=True, exist_ok=True)
        signature = raw_signature(config, records)
        if (destination / "manifest.json").exists():
            manifest = validate_raw_manifest(destination, signature)
            if manifest.get("complete"):
                logger.info("Verified existing raw split=%s samples=%d", split, len(records))
                continue
        else:
            manifest = {"schema_version": 1, "signature": signature, "split": split,
                        "model": config["model"], "base_stride": config["features"]["base_stride"],
                        "record_ids": [r["id"] for r in records], "shards": [], "complete": False}
        pending.append((split, records, destination, manifest))
    if not pending:
        return
    model = load_backbone(config)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_total = time.perf_counter()
    total_tokens = 0
    for split, records, destination, manifest in pending:
        completed = sum(len(shard["ids"]) for shard in manifest["shards"])
        shard_size = int(config["features"]["shard_size"])
        with timed(logger, "extract_split", split=split, total=len(records), resumed=completed):
            for start in range(completed, len(records), shard_size):
                arrays, token_positions, offsets, ids = [], [], [0], []
                for record in records[start:start + shard_size]:
                    positions = checkpoint_positions(record["response_len"], config["features"]["base_stride"])
                    hidden = hidden_states_at_tokens(model, record["prompt_ids"], record["response_ids"],
                                                     config["model"]["layers"], positions, device)
                    arrays.append(hidden)
                    token_positions.append(positions)
                    offsets.append(offsets[-1] + len(positions))
                    ids.append(record["id"])
                    total_tokens += len(record["prompt_ids"]) + record["response_len"]
                prefix = f"shard_{len(manifest['shards']):05d}"
                payload = {f"{prefix}.values.npy": np.concatenate(arrays),
                           f"{prefix}.positions.npy": np.concatenate(token_positions),
                           f"{prefix}.offsets.npy": np.asarray(offsets, dtype=np.int64)}
                hashes = {}
                for name, array in payload.items():
                    _save_array(destination / name, array)
                    hashes[name] = hash_file(destination / name)
                manifest["shards"].append({"prefix": prefix, "ids": ids, "files": hashes})
                save_json(destination / "manifest.json", manifest)
                logger.info("Shard committed split=%s completed=%d/%d elapsed_seconds=%.2f", split,
                            start + len(ids), len(records), time.perf_counter() - start_total)
            manifest["complete"] = True
            save_json(destination / "manifest.json", manifest)
    elapsed = time.perf_counter() - start_total
    save_json(Path(config["paths"]["results"]) / "extraction_timing.json",
              {"elapsed_seconds": elapsed, "tokens": total_tokens,
               "tokens_per_second": total_tokens / max(elapsed, 1e-9),
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0,
               "device": str(device), "model": config["model"]["id"]})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_causal_alignment(config, records, max_samples=3):
    """验证后缀隔离和KV位置对齐；输出真实误差而非假定完全相同。"""
    import torch
    from torch.nn import functional

    model = load_backbone(config)
    device, layers = next(model.parameters()).device, config["model"]["layers"]
    observations = []
    for record in records[:max_samples]:
        response = record["response_ids"][:32]
        positions = np.arange(1, len(response) + 1, dtype=np.int32)
        full = hidden_states_at_tokens(model, record["prompt_ids"], response, layers, positions)
        with torch.inference_mode():
            prompt = torch.tensor([record["prompt_ids"]], device=device)
            out = model.model(input_ids=prompt, use_cache=True, output_hidden_states=True)
            cache = out.past_key_values
            incremental = []
            for token in response:
                out = model.model(input_ids=torch.tensor([[token]], device=device), past_key_values=cache,
                                  use_cache=True, output_hidden_states=True)
                cache = out.past_key_values
                incremental.append(torch.stack([out.hidden_states[layer][0, -1] for layer in layers]).float().cpu())
        incremental = torch.stack(incremental)
        complete = torch.from_numpy(full).float()
        cosine = functional.cosine_similarity(complete.reshape(-1, full.shape[-1]),
                                              incremental.reshape(-1, full.shape[-1]), dim=-1)
        prefix_length = min(8, len(response))
        prefix = hidden_states_at_tokens(model, record["prompt_ids"], response[:prefix_length], layers,
                                         positions[:prefix_length])
        observations.append({"id": record["id"], "minimum_cosine_similarity": float(cosine.min()),
                             "mean_absolute_difference": float((complete - incremental).abs().mean()),
                             "prefix_max_abs_difference": float(np.max(np.abs(full[:prefix_length].astype("float32")
                                                                                    - prefix.astype("float32"))))})
    result = {"samples": observations, "passed": all(r["minimum_cosine_similarity"] >= 0.999 for r in observations)}
    save_json(Path(config["paths"]["results"]) / "causal_alignment.json", result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not result["passed"]:
        raise AssertionError(f"Causal alignment failed: {result}")
    return result
