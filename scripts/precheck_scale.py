"""1.5B资源预检。默认只检查CPU元数据；显式 --run-gpu 才加载GPU模型。

本脚本不重新清洗样本、不保存PCA/探针训练结果。计时中的临时统计量均会丢弃。
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hs_safety.config import load_config
from hs_safety.io import (
    configure_logging,
    get_logger,
    hash_file,
    hash_object,
    load_json,
    read_jsonl,
    save_json,
)

SPLITS = ("train", "dev", "calibration", "test", "external")


def inspect_shared_data(config, main_config):
    """只读复用相同样本与token边界，输出每个分区的内容散列。"""

    shared = Path(config["paths"]["processed"]).resolve()
    if shared != Path(main_config["paths"]["processed"]).resolve():
        raise ValueError("Scale validation must reuse the exact main processed directory.")
    for key in ("raw_features", "features", "preprocessing", "models", "results", "logs"):
        if Path(config["paths"][key]).resolve() == Path(main_config["paths"][key]).resolve():
            raise ValueError(f"Scale output must be isolated from the main experiment: {key}")
    rows, audit = {}, {}
    for split in SPLITS:
        path = shared / f"{split}.jsonl"
        if not path.exists():
            continue
        rows[split] = read_jsonl(path)
        audit[split] = {"path": str(path), "sha256": hash_file(path), "n_responses": len(rows[split]),
                        "tokens": sum(len(row["prompt_ids"]) + len(row["response_ids"]) for row in rows[split])}
    if len(rows.get("train", [])) < int(config["scale"]["precheck_samples"]):
        raise ValueError("Training split has fewer responses than the requested resource sample.")
    train = sorted(rows["train"], key=lambda row: (len(row["prompt_ids"]) + row["response_len"], row["id"]))
    indices = np.linspace(0, len(train) - 1, int(config["scale"]["precheck_samples"])).round().astype(int)
    samples = [("train", train[index]) for index in indices]
    # 内存压力覆盖全部分区的最长样本；该样本若来自测试，仅用于前向资源测量。
    longest = max(((split, row) for split, values in rows.items() for row in values),
                  key=lambda item: len(item[1]["prompt_ids"]) + item[1]["response_len"])
    if len(longest[1]["prompt_ids"]) + longest[1]["response_len"] > len(samples[-1][1]["prompt_ids"]) + samples[-1][1]["response_len"]:
        samples[-1] = longest
    return rows, samples, audit


def inspect_tokenizers(config, main_config, samples):
    from transformers import AutoTokenizer

    main = Path(main_config["model"]["local_path"])
    scale = Path(config["model"]["local_path"])
    files = {}
    for name in ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"):
        left, right = hash_file(main / name), hash_file(scale / name)
        files[name] = {"main_sha256": left, "scale_sha256": right, "identical": left == right}
    main_tokenizer = AutoTokenizer.from_pretrained(main, local_files_only=True)
    scale_tokenizer = AutoTokenizer.from_pretrained(scale, local_files_only=True)
    main_vocab = hash_object(main_tokenizer.get_vocab())
    scale_vocab = hash_object(scale_tokenizer.get_vocab())
    template_matches = main_tokenizer.chat_template == scale_tokenizer.chat_template
    mismatches = []
    for split, record in samples:
        prompt_ids = scale_tokenizer.apply_chat_template(
            [{"role": "system", "content": config["model"]["system_prompt"]},
             {"role": "user", "content": record["prompt"]}], tokenize=True, add_generation_prompt=True)
        response_ids = scale_tokenizer.encode(record["response"], add_special_tokens=False)
        if prompt_ids != record["prompt_ids"] or response_ids != record["response_ids"]:
            mismatches.append({"split": split, "id": record["id"]})
    required_files_match = all(files[name]["identical"] for name in ("vocab.json", "merges.txt", "tokenizer.json"))
    passed = main_vocab == scale_vocab and template_matches and required_files_match and not mismatches
    result = {"passed": passed, "files": files, "main_canonical_vocab_sha256": main_vocab,
              "scale_canonical_vocab_sha256": scale_vocab, "vocab_size": len(scale_tokenizer),
              "chat_template_identical": template_matches, "sample_token_id_mismatches": mismatches,
              "n_retokenized_for_verification_only": len(samples),
              "processed_data_rewritten": False}
    if not passed:
        raise ValueError(f"Tokenizer equivalence failed: {result}")
    return result


def benchmark_cpu_stages(config, all_rows, measured):
    """只用预检中的训练样本测量临时PCA与探针；任何临时权重均不保存。"""

    import torch
    from sklearn.decomposition import IncrementalPCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from torch.nn.utils.rnn import pad_sequence

    from hs_safety.features import causal_dynamic_features, project_hidden
    from hs_safety.models import SequenceGRU, prefix_indices

    settings = config["training"]
    threads = int(settings.get("num_threads", 4))
    torch.set_num_threads(threads)
    torch.manual_seed(int(settings["seeds"][0]))
    train_samples = [item for item in measured if item["split"] == "train"]
    if not train_samples:
        raise ValueError("CPU resource estimation needs training examples in the measured sample.")
    component_count = int(config["features"]["pca_components"])
    batch_size = int(config["features"]["pca_batch_size"])
    chosen = np.concatenate([item["hidden"][prefix_indices(len(item["hidden"]), config["features"]["pca_fit_prefixes"])] for item in train_samples]).astype(np.float32)
    repeat = max(1, math.ceil(batch_size / len(chosen)))
    block = np.tile(chosen, (repeat, 1, 1))[:batch_size]
    estimators = [IncrementalPCA(n_components=component_count, batch_size=batch_size) for _ in config["model"]["layers"]]
    with threadpool_limits(limits=threads):
        first_started = time.perf_counter()
        for layer, estimator in enumerate(estimators):
            estimator.partial_fit(block[:, layer])
        first_fit_seconds = time.perf_counter() - first_started
        started = time.perf_counter()
        for layer, estimator in enumerate(estimators):
            estimator.partial_fit(block[:, layer])
        steady_fit_seconds = time.perf_counter() - started
        projection = {"components": np.stack([value.components_ for value in estimators]).astype(np.float32),
                      "mean": np.stack([value.mean_ for value in estimators]).astype(np.float32),
                      "scale": np.sqrt(np.maximum(np.stack([value.explained_variance_ for value in estimators]), 1e-8)).astype(np.float32)}
        hidden = np.concatenate([item["hidden"] for item in train_samples]).astype(np.float32)
        started = time.perf_counter()
        project_hidden(hidden, projection)
        projection_seconds = time.perf_counter() - started
    sequences, static_sequences, labels = [], [], []
    for item in train_samples:
        positions = item["positions"]
        mask = (positions % int(config["features"]["stride"]) == 0) | (positions == item["record"]["response_len"])
        projected = project_hidden(item["hidden"][mask], projection).astype(np.float16).astype(np.float32)
        sequences.append(causal_dynamic_features(projected))
        static_sequences.append(causal_dynamic_features(projected, dynamic=False))
        labels.append(int(item["record"]["label"]))
    scaler = StandardScaler().fit(np.concatenate(sequences))
    neural_batch_size = int(settings["batch_size"])
    batch_indices = np.linspace(0, len(sequences) - 1, neural_batch_size).round().astype(int)
    # 覆盖长度分位数且包含最长训练序列，避免用短序列低估padding后的训练成本。
    current = [sequences[index] for index in batch_indices]
    values = pad_sequence([torch.from_numpy(scaler.transform(sequence).astype(np.float32)) for sequence in current], batch_first=True)
    targets = torch.tensor([labels[index] for index in batch_indices], dtype=torch.float32)
    network = SequenceGRU(values.shape[-1], int(settings["hidden_size"]))
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    criterion = torch.nn.BCEWithLogitsLoss()

    def step():
        logits = network(values)
        losses = [criterion(logits[index, prefix_indices(len(sequence), int(settings["max_prefixes"]))],
                            targets[index].expand(len(prefix_indices(len(sequence), int(settings["max_prefixes"])))))
                  for index, sequence in enumerate(current)]
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

    for _ in range(2):
        step()
    step_times = []
    for _ in range(3):
        started = time.perf_counter()
        step()
        step_times.append(time.perf_counter() - started)
    prefix_values, prefix_labels, prefix_weights = [], [], []
    for sequence, label in zip(static_sequences, labels):
        selected = sequence[prefix_indices(len(sequence), int(settings["max_prefixes"]))]
        prefix_values.append(selected)
        prefix_labels.extend([label] * len(selected))
        prefix_weights.extend([1 / len(selected)] * len(selected))
    matrix = np.concatenate(prefix_values)
    if len(set(prefix_labels)) == 2:
        with threadpool_limits(limits=1):
            started = time.perf_counter()
            transformed = StandardScaler().fit_transform(matrix)
            LogisticRegression(solver="liblinear", max_iter=500, class_weight="balanced").fit(transformed, prefix_labels, sample_weight=prefix_weights)
            linear_fit_seconds = time.perf_counter() - started
    else:
        linear_fit_seconds = None
    n_train = len(all_rows["train"])
    n_epochs, n_seeds = int(settings["epochs"]), len(settings["seeds"])
    n_fit = sum(min(math.ceil(row["response_len"] / config["features"]["base_stride"]), config["features"]["pca_fit_prefixes"]) for row in all_rows["train"])
    n_checkpoints = sum(math.ceil(row["response_len"] / config["features"]["base_stride"]) for rows in all_rows.values() for row in rows)
    pca_estimate = first_fit_seconds + max(0, math.ceil(n_fit / batch_size) - 1) * steady_fit_seconds
    project_estimate = projection_seconds / len(hidden) * n_checkpoints
    # 按完整20轮上限估计，不假设早停。另加开发/测试前向的保守余量。
    neural_estimate = float(np.median(step_times)) * math.ceil(n_train / neural_batch_size) * n_epochs * n_seeds * 1.5
    linear_estimate = max(30.0, linear_fit_seconds / len(matrix) * n_fit) * n_seeds if linear_fit_seconds is not None else 1800.0
    return {"pca_benchmark_first_fit_seconds": first_fit_seconds, "pca_benchmark_steady_batch_seconds": steady_fit_seconds,
            "pca_benchmark_batch_rows": len(block), "pca_benchmark_training_rows_available": len(chosen),
            "pca_benchmark_repeated_rows": repeat > 1, "pca_estimated_seconds": pca_estimate,
            "projection_benchmark_seconds": projection_seconds, "projection_benchmark_checkpoints": len(hidden),
            "projection_estimated_seconds": project_estimate, "gru_cpu_training_step_seconds": float(np.median(step_times)),
            "gru_estimated_seconds_all_seeds_max_epochs": neural_estimate,
            "static_lr_benchmark_seconds": linear_fit_seconds, "static_lr_estimated_seconds_all_seeds": linear_estimate,
            "threads": threads, "probe_training_device": "cpu", "temporary_parameters_saved": False,
            "total_cpu_stage_estimated_seconds": pca_estimate + project_estimate + neural_estimate + linear_estimate}


def measure_gpu(config, all_rows, samples, report, output):
    import torch

    from hs_safety.data import checkpoint_positions
    from hs_safety.extraction import hidden_states_at_tokens, load_backbone

    logger = get_logger("precheck_scale")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the scale GPU precheck cannot pass on CPU fallback.")
    free, total = torch.cuda.mem_get_info()
    report["gpu"] = {"device": torch.cuda.get_device_name(), "total_gib": total / 2**30, "free_gib_before_load": free / 2**30}
    if free / 2**30 < float(config["scale"]["min_free_gpu_gib_before_load"]):
        report.update({"status": "resource_skip", "passed": False,
                       "skip_reason": "Free GPU memory is below the pre-load floor; no model was loaded and no retry was attempted."})
        save_json(output, report)
        return report
    model = None
    measured, observations = [], []
    try:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model = load_backbone(config)
        torch.cuda.synchronize()
        report["gpu"]["model_load_seconds"] = time.perf_counter() - started
        for _, record in (samples[0], samples[len(samples) // 2], samples[-1]):
            positions = checkpoint_positions(record["response_len"], config["features"]["base_stride"])
            hidden_states_at_tokens(model, record["prompt_ids"], record["response_ids"], config["model"]["layers"], positions)
        minimum_free = torch.cuda.mem_get_info()[0]
        for split, record in samples:
            positions = checkpoint_positions(record["response_len"], config["features"]["base_stride"])
            torch.cuda.synchronize()
            started = time.perf_counter()
            hidden = hidden_states_at_tokens(model, record["prompt_ids"], record["response_ids"], config["model"]["layers"], positions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
            observed = {"id": record["id"], "split": split, "prompt_tokens": len(record["prompt_ids"]),
                        "response_tokens": record["response_len"], "total_tokens": len(record["prompt_ids"]) + record["response_len"],
                        "n_checkpoints": len(positions), "elapsed_seconds": elapsed}
            observations.append(observed)
            measured.append({"split": split, "record": record, "hidden": hidden, "positions": positions})
            logger.info("Scale resource sample=%d/%d split=%s tokens=%d elapsed_seconds=%.6f", len(observations), len(samples), split, observed["total_tokens"], elapsed)
        report["gpu"].update({"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                               "peak_device_used_gib_at_checkpoint": (total - minimum_free) / 2**30})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report["samples"] = observations
    report["status"] = "gpu_measurement_complete_cpu_estimation_pending"
    save_json(output, report)
    cpu = benchmark_cpu_stages(config, all_rows, measured)
    report["cpu_stage_estimate"] = cpu
    lengths = np.asarray([item["total_tokens"] for item in observations])
    times = np.asarray([item["elapsed_seconds"] for item in observations])
    order = np.argsort(lengths)
    # 八个长度桶的中位数减轻单次计时噪声，再对真实全量长度做插值求和。
    buckets = np.array_split(order, min(8, len(order)))
    centers = np.asarray([np.median(lengths[indices]) for indices in buckets])
    medians = np.maximum.accumulate([np.median(times[indices]) for indices in buckets])
    centers = np.append(centers, lengths.max())
    medians = np.append(medians, max(medians[-1], float(times[lengths == lengths.max()].max())))
    all_lengths = np.asarray([len(row["prompt_ids"]) + row["response_len"] for rows in all_rows.values() for row in rows])
    extraction_seconds = float(np.interp(all_lengths, centers, medians).sum())
    n_checkpoints = sum(math.ceil(row["response_len"] / config["features"]["base_stride"]) for rows in all_rows.values() for row in rows)
    hidden_size = int(load_json(Path(config["model"]["local_path"]) / "config.json")["hidden_size"])
    raw_bytes = n_checkpoints * len(config["model"]["layers"]) * hidden_size * 2
    projected_bytes = n_checkpoints * len(config["model"]["layers"]) * config["features"]["pca_components"] * 2
    io_seconds = (raw_bytes + projected_bytes) / (50 * 1024**2)
    safety = float(config["scale"]["estimate_safety_factor"])
    total_seconds = report["gpu"]["model_load_seconds"] + safety * (extraction_seconds + io_seconds + cpu["total_cpu_stage_estimated_seconds"])
    peak = max(report["gpu"]["peak_reserved_gib"], report["gpu"]["peak_device_used_gib_at_checkpoint"])
    report["estimate"] = {"all_split_extraction_seconds_before_safety_factor": extraction_seconds,
                          "cache_write_seconds_at_50_mib_per_second": io_seconds,
                          "raw_cache_bytes": raw_bytes, "projected_cache_bytes": projected_bytes,
                          "safety_factor": safety, "conservative_total_pipeline_hours": total_seconds / 3600,
                          "conservative_gpu_extraction_hours": safety * extraction_seconds / 3600,
                          "epochs_assumed": config["training"]["epochs"], "n_seeds": len(config["training"]["seeds"]),
                          "note": "Resource estimate only; elapsed training and extraction measurements remain authoritative."}
    report["checks"] = {"tokenizers_equivalent": report["tokenizer_equivalence"]["passed"],
                        "peak_gpu_below_limit": peak < float(config["scale"]["max_peak_gpu_gib"]),
                        "all_stage_estimate_below_budget": total_seconds / 3600 < float(config["scale"]["max_pipeline_hours"])}
    report["passed"] = all(report["checks"].values())
    report["status"] = "passed" if report["passed"] else "failed_resource_budget"
    return report


def main():
    parser = argparse.ArgumentParser(description="Auditable scale-model resource precheck; CPU-only unless --run-gpu")
    parser.add_argument("--config", default="configs/scale.yaml")
    parser.add_argument("--run-gpu", action="store_true", help="Load 1.5B only after the main GPU extraction has completed")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config["paths"]["logs"], "precheck_scale")
    logger = get_logger("precheck_scale")
    output = Path(config["paths"]["results"]) / "resource_precheck.json"
    report = {"schema_version": 1, "model": config["model"], "config_sha256": hash_file(args.config),
              "passed": False, "status": "initializing", "gpu_requested": args.run_gpu,
              "limits": config["scale"], "processed_data_modified": False}
    try:
        main_config = load_config(config["scale"]["main_config"])
        all_rows, samples, data_audit = inspect_shared_data(config, main_config)
        report["shared_data"] = data_audit
        report["tokenizer_equivalence"] = inspect_tokenizers(config, main_config, samples)
        report["sample_selection"] = {"method": "training length quantiles; replace longest with all-split maximum if needed",
                                      "n": len(samples), "ids": [row["id"] for _, row in samples],
                                      "test_samples_used_for_training_estimate": False}
        manifest = load_json(Path(config["paths"]["results"]) / "model_download_manifest.json")
        if manifest["revision"] != config["model"]["revision"]:
            raise ValueError("Downloaded model revision differs from the scale configuration.")
        report["model_weights_sha256"] = hash_file(Path(config["model"]["local_path"]) / "model.safetensors")
        if report["model_weights_sha256"] != manifest["files"]["model.safetensors"]["sha256"]:
            raise ValueError("Scale model weight checksum mismatch.")
        if args.run_gpu:
            report = measure_gpu(config, all_rows, samples, report, output)
        else:
            report["status"] = "static_checks_passed_awaiting_gpu_measurement"
        save_json(output, report)
        logger.info("Scale precheck completed status=%s passed=%s output=%s", report["status"], report["passed"], output)
        if args.run_gpu and not report["passed"]:
            raise SystemExit(2)
    except Exception as error:
        report.update({"status": "failed", "passed": False, "error_type": type(error).__name__, "error": str(error)})
        save_json(output, report)
        logger.exception("Scale precheck failed output=%s", output)
        raise


if __name__ == "__main__":
    main()
