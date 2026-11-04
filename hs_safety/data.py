"""真实公开回答的清洗、分组与长度过滤，绝不截断后继承整段标签。"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .io import get_logger, hash_file, hash_object, read_jsonl, save_json, timed, write_jsonl


def normalize_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def prompt_group(prompt):
    return hashlib.sha256(normalize_text(prompt).encode("utf-8")).hexdigest()


def checkpoint_positions(length, stride):
    """位置从1开始；末尾不足一个间隔时也必须监测。"""
    if length < 0 or stride <= 0:
        raise ValueError("Length must be nonnegative and stride must be positive")
    if length == 0:
        return np.empty(0, dtype=np.int32)
    positions = list(range(stride, length + 1, stride))
    if not positions or positions[-1] != length:
        positions.append(length)
    return np.asarray(positions, dtype=np.int32)


def load_tokenizer(config):
    from transformers import AutoTokenizer

    model = config["model"]
    local = Path(model.get("local_path", ""))
    location = str(local) if (local / "tokenizer_config.json").exists() else model["id"]
    return AutoTokenizer.from_pretrained(location, cache_dir=model.get("cache_dir"), local_files_only=True)


def _read_beaver(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_external(path):
    import pyarrow as pa
    from pyarrow import ipc

    source = pa.memory_map(str(path), "r")
    try:
        table = ipc.open_stream(source).read_all()
    except pa.ArrowInvalid:
        source.seek(0)
        table = ipc.open_file(source).read_all()
    return table.to_pylist()


def original_annotation(row, source):
    """原始标签单独存档；不将原始类别或评审信息加入检测器输入。"""
    field = "is_safe" if source == "beavertails" else "label"
    value = row[field]
    return {"original_label_field": field, "original_label": value,
            "original_categories": row.get("category", {})}


def export_source_annotations(config):
    raw, processed = Path(config["paths"]["raw"]), Path(config["paths"]["processed"])
    index = {}
    sources = [("beavertails", "30k_train", raw / "beavertails/round0/30k/train.jsonl.gz"),
               ("beavertails", "30k_test", raw / "beavertails/round0/30k/test.jsonl.gz"),
               ("streamguard", "test", raw / "streamguard/wildguard/qwen3_8b/test/data-00000-of-00001.arrow")]
    for source, source_split, path in sources:
        rows = _read_beaver(path) if source == "beavertails" else _read_external(path)
        for row in rows:
            if source != "beavertails" and row.get("label") not in (0, 1, False, True):
                continue
            pair = hash_object([normalize_text(row.get("prompt", "")), normalize_text(row.get("response", ""))])
            key = (source, source_split, f"{source}-{pair[:24]}")
            index.setdefault(key, original_annotation(row, source))
    annotations = []
    for split in ("train", "dev", "calibration", "test", "external"):
        for record in read_jsonl(processed / f"{split}.jsonl"):
            annotation = index[(record["source"], record["source_split"], record["id"])]
            label = int(not annotation["original_label"]) if record["source"] == "beavertails" else int(annotation["original_label"])
            if label != record["label"]:
                raise ValueError(f"Original label mapping disagrees with prepared label id={record['id']}")
            annotations.append({"id": record["id"], "split": split, "source": record["source"],
                                "source_split": record["source_split"], **annotation})
    path = processed / "source_annotations.jsonl"
    write_jsonl(path, annotations)
    save_json(processed / "source_annotations.manifest.json",
              {"records": len(annotations), "sha256": hash_file(path),
               "prepared_audit_sha256": hash_file(processed / "audit.json"),
               "note": "Source metadata only; detector records and learned artifacts are unchanged."})
    get_logger("data").info("Original annotations archived: records=%d path=%s", len(annotations), path)


def _tokenize_records(raw_rows, source, tokenizer, config, source_split):
    logger = get_logger("data")
    accepted, rejected = [], Counter()
    total_labels, retained_labels, total_categories, retained_categories = Counter(), Counter(), Counter(), Counter()
    seen_pairs = set()
    for row in raw_rows:
        prompt, response = str(row.get("prompt", "")), str(row.get("response", ""))
        if source == "beavertails":
            label = int(not bool(row["is_safe"]))
            cats = [key for key, value in row.get("category", {}).items() if value]
        else:
            value = row.get("label")
            if value not in (0, 1, False, True):
                rejected["missing_or_invalid_label"] += 1
                continue
            label, cats = int(value), []
        total_labels[str(label)] += 1
        total_categories.update(cats)
        if not prompt.strip() or not response.strip():
            rejected["empty_text"] += 1
            continue
        # 明确拼接prompt IDs和response IDs，使离线回放与在线消费具有相同边界。
        pids = tokenizer.apply_chat_template(
            [{"role": "system", "content": config["model"]["system_prompt"]},
             {"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        )
        rids = tokenizer.encode(response, add_special_tokens=False)
        if len(pids) > config["data"]["max_prompt_tokens"]:
            rejected["prompt_over_limit"] += 1
            continue
        if not rids or len(rids) > config["data"]["max_response_tokens"]:
            rejected["response_over_limit_or_empty"] += 1
            continue
        pair = hash_object([normalize_text(prompt), normalize_text(response)])
        if pair in seen_pairs:
            rejected["duplicate_qa"] += 1
            continue
        seen_pairs.add(pair)
        accepted.append({"id": f"{source}-{pair[:24]}", "group": prompt_group(prompt),
                         "source": source, "source_split": source_split, "prompt": prompt,
                         "response": response, "label": label, "categories": cats,
                         "prompt_ids": pids, "response_ids": rids, "response_len": len(rids)})
        retained_labels[str(label)] += 1
        retained_categories.update(cats)
    audit = {"source": source, "source_split": source_split, "input": len(raw_rows), "accepted": len(accepted),
             "rejected": dict(rejected), "input_labels": dict(total_labels), "retained_labels": dict(retained_labels),
             "input_categories": dict(total_categories), "retained_categories": dict(retained_categories)}
    logger.info("Data filtering audit=%s", audit)
    return accepted, audit


def grouped_split(records, train_size, dev_size, calibration_size, seed=42):
    """按prompt组划分；目标数量允许因完整组约束出现小幅偏差。"""
    groups = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    output = {"train": [], "dev": [], "calibration": []}
    # 首先保证独立开发集与校准集，再使用剩余组构造训练集。
    cursor = 0
    for split, target in [("dev", dev_size), ("calibration", calibration_size), ("train", train_size)]:
        while cursor < len(keys) and len(output[split]) < target:
            output[split].extend(groups[keys[cursor]])
            cursor += 1
    for name, rows in output.items():
        for record in rows:
            record["split"] = name
    return output


def _find_near_overlap(candidates, references, threshold):
    """字符TF-IDF近邻审计；只用于划分清理，不读取模型特征或测试标签。"""
    if not candidates or not references:
        return set(), []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    unique_candidates = {r["group"]: normalize_text(r["prompt"]) for r in candidates}
    unique_references = {r["group"]: normalize_text(r["prompt"]) for r in references}
    keys, texts = list(unique_candidates), list(unique_candidates.values())
    ref_keys, ref_texts = list(unique_references), list(unique_references.values())
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                                 max_features=50000, dtype=np.float32)
    matrix = vectorizer.fit_transform(ref_texts + texts)
    model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=1).fit(matrix[:len(ref_texts)])
    overlaps, examples = set(), []
    for start in range(0, len(texts), 128):
        distances, indices = model.kneighbors(matrix[len(ref_texts) + start:len(ref_texts) + start + 128])
        for offset, (distance, index) in enumerate(zip(distances[:, 0], indices[:, 0])):
            if 1.0 - float(distance) >= threshold:
                overlaps.add(keys[start + offset])
                if len(examples) < 50:
                    examples.append({"candidate_group": keys[start + offset], "reference_group": ref_keys[index],
                                     "similarity": 1.0 - float(distance)})
    return overlaps, examples


def prepare_data(config):
    logger = get_logger("data")
    raw, processed = Path(config["paths"]["raw"]), Path(config["paths"]["processed"])
    tokenizer = load_tokenizer(config)
    all_rows, audits = {}, []
    sources = [("pool", "beavertails", raw / "beavertails/round0/30k/train.jsonl.gz", "30k_train"),
               ("test", "beavertails", raw / "beavertails/round0/30k/test.jsonl.gz", "30k_test"),
               ("external", "streamguard", raw / "streamguard/wildguard/qwen3_8b/test/data-00000-of-00001.arrow", "test")]
    with timed(logger, "prepare_data"):
        for split, source, path, original_split in sources:
            if not path.exists():
                raise FileNotFoundError(f"Missing raw asset: {path}. Run scripts/download_assets.py first")
            rows = _read_beaver(path) if source == "beavertails" else _read_external(path)
            all_rows[split], audit = _tokenize_records(rows, source, tokenizer, config, original_split)
            audits.append(audit)
        # 测试集具有保留优先级。外部测试与主测试的重复记录只从外部移除。
        main_test = all_rows["test"]
        test_groups = {r["group"] for r in main_test}
        test_responses = {normalize_text(r["response"]) for r in main_test}
        external = [r for r in all_rows["external"] if r["group"] not in test_groups
                    and normalize_text(r["response"]) not in test_responses]
        reference = main_test + external
        reference_groups = {r["group"] for r in reference}
        reference_responses = {normalize_text(r["response"]) for r in reference}
        pool_before = len(all_rows["pool"])
        pool = [r for r in all_rows["pool"] if r["group"] not in reference_groups
                and normalize_text(r["response"]) not in reference_responses]
        near_groups, near_examples = _find_near_overlap(pool, reference, config["data"]["near_duplicate_threshold"])
        pool = [r for r in pool if r["group"] not in near_groups]
        splits = grouped_split(pool, config["data"]["train_size"], config["data"]["dev_size"],
                               config["data"]["calibration_size"], config["data"]["split_seed"])
        splits.update(test=main_test, external=external)
        # 不同训练分区之间的相同回答模板也不能跨界，优先保留calibration、dev。
        seen_response = set()
        template_removed = {}
        for split in ["calibration", "dev", "train"]:
            rows = splits[split]
            retained = [r for r in rows if normalize_text(r["response"]) not in seen_response]
            template_removed[split] = len(rows) - len(retained)
            seen_response.update(normalize_text(r["response"]) for r in retained)
            splits[split] = retained
        for split, rows in splits.items():
            for row in rows:
                row["split"] = split
            write_jsonl(processed / f"{split}.jsonl", rows)
            logger.info("Prepared split=%s samples=%d labels=%s groups=%d", split, len(rows),
                        dict(Counter(r["label"] for r in rows)), len({r["group"] for r in rows}))
        # 分组互斥是硬约束，失败即终止而不输出可训练状态。
        names = list(splits)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                if {r["group"] for r in splits[left]} & {r["group"] for r in splits[right]}:
                    raise ValueError(f"Prompt leakage detected: {left} vs {right}")
        report = {"schema_version": 1, "config_hash": hash_object(config["data"]), "sources": audits,
                  "split_counts": {k: len(v) for k, v in splits.items()},
                  "split_labels": {k: dict(Counter(str(r["label"]) for r in v)) for k, v in splits.items()},
                  "pool_before_overlap_filter": pool_before, "pool_after_overlap_filter": len(pool),
                  "near_prompt_groups_removed": len(near_groups), "near_prompt_examples": near_examples,
                  "response_templates_removed": template_removed, "tokenizer_model": config["model"],
                  "records_hash": {k: hash_object(v) for k, v in splits.items()}}
        save_json(processed / "audit.json", report)
        export_source_annotations(config)
    return report
