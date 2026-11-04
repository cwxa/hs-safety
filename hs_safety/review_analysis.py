"""审稿补充分析：冻结模型、仅校准集阈值敏感性及预算内排序不确定性。"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .io import hash_file, load_json, read_jsonl, save_json
from .metrics import calibrate_threshold, response_metrics

LOGGER = logging.getLogger(__name__)
METHODS = ("prompt_lr", "text_lr", "static_lr", "mean_lr", "gru")


def budget_scores(rows, budget=128):
    values = []
    for row in rows:
        observed = [score for token, score in zip(row["positions"], row["scores"])
                    if token <= min(budget, row["response_len"])]
        if not observed:
            raise ValueError(f"No checkpoint inside ranking budget id={row['id']}")
        values.append(max(observed))
    return np.asarray(values, dtype=np.float64)


def wilson_interval(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total)) / denominator
    half = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total)) / denominator
    return [float(max(0, center-half)), float(min(1, center+half))]


def strict_prefix_hits(rows, threshold, budget=128):
    """终点评分不计作完成前检出；完整长度只用于离线评价，不进入特征。"""
    return np.asarray([any(score > threshold for position, score in zip(row["positions"], row["scores"])
                           if position <= budget and position < row["response_len"]) for row in rows], dtype=bool)


def run_review_analysis(config, n_bootstrap=1000):
    results = Path(config["paths"]["results"])
    models = Path(config["paths"]["models"])
    seeds = config["training"]["seeds"]
    predictions, hashes, sensitivity, external = {}, {}, [], []
    completion_profiles, strict_recall = {}, []
    for method in METHODS:
        for seed in seeds:
            rows_by_split = {}
            for split in ("calibration", "test", "external"):
                path = results / "predictions" / method / f"seed_{seed}" / f"{split}.jsonl"
                rows_by_split[split] = read_jsonl(path)
                hashes[path.as_posix()] = hash_file(path)
            predictions[(method, seed)] = rows_by_split
            original = load_json(models / method / f"seed_{seed}" / "calibration.json")
            for split in ("test", "external"):
                current = rows_by_split[split]
                unsafe = np.asarray([row["label"] == 1 for row in current])
                lengths = np.asarray([row["response_len"] for row in current])
                completion_profiles[split] = {"n": len(current), "n_unsafe": int(unsafe.sum()),
                                              "complete_by128": int(np.sum(lengths <= 128)),
                                              "unsafe_complete_by128": int(np.sum((lengths <= 128) & unsafe))}
                hits = strict_prefix_hits(current, original["threshold"])
                strict_recall.append({"method": method, "seed": seed, "split": split,
                                      "n_unsafe": int(unsafe.sum()), "strict_prefix_tp128": int(np.sum(hits & unsafe)),
                                      "strict_prefix_recall128": float(hits[unsafe].mean())})
            for target in (0.01, 0.05, 0.10):
                calibrated = calibrate_threshold(rows_by_split["calibration"], target)
                if target == 0.05 and calibrated["threshold"] != original["threshold"]:
                    raise ValueError("Primary threshold changed during review analysis")
                for split in ("test", "external"):
                    measured = response_metrics(rows_by_split[split], calibrated["threshold"], [128])
                    sensitivity.append({"method": method, "seed": seed, "target_calibration_fpr": target,
                                        "split": split, "threshold": calibrated["threshold"],
                                        "calibration_fpr": calibrated["empirical_sequence_fpr"],
                                        "recall128": measured["recall_at_budget"]["128"],
                                        "test_fpr": measured["sequence_fpr"]})
            rows = rows_by_split["external"]
            labels = np.asarray([r["label"] for r in rows])
            early = budget_scores(rows) > original["threshold"]
            full = np.asarray([max(r["scores"]) for r in rows]) > original["threshold"]
            tp, fp = int(np.sum(early & (labels == 1))), int(np.sum(full & (labels == 0)))
            positives, negatives = int(np.sum(labels == 1)), int(np.sum(labels == 0))
            external.append({"method": method, "seed": seed, "tp128": tp, "n_unsafe": positives,
                             "fp_sequence": fp, "n_safe": negatives,
                             "recall128_wilson95": wilson_interval(tp, positives),
                             "fpr_wilson95": wilson_interval(fp, negatives)})
    contrasts = (("prompt_lr", "gru"), ("text_lr", "static_lr"), ("static_lr", "gru"))
    ranking = {}
    for split in ("test", "external"):
        reference = predictions[(METHODS[0], seeds[0])][split]
        reference = sorted(reference, key=lambda row: row["id"])
        ids = [(r["id"], r["group"], r["label"]) for r in reference]
        labels = np.asarray([r["label"] for r in reference])
        groups = sorted({r["group"] for r in reference})
        group_lookup = {group: index for index, group in enumerate(groups)}
        group_indices = np.asarray([group_lookup[r["group"]] for r in reference])
        scores = {}
        for method in METHODS:
            unique = {}
            for seed in seeds:
                rows = sorted(predictions[(method, seed)][split], key=lambda row: row["id"])
                if [(r["id"], r["group"], r["label"]) for r in rows] != ids:
                    raise ValueError("Ranking bootstrap requires identical responses, groups and labels")
                values = budget_scores(rows)
                key = values.tobytes()
                if key not in unique:
                    unique[key] = [values, 0]
                unique[key][1] += 1
            # 相同重复拟合复用计算，但仍保留其种子计数权重。
            scores[method] = list(unique.values())
        observed = {name: float(np.average([roc_auc_score(labels, s) for s, count in arrays],
                                          weights=[count for s, count in arrays]))
                    for name, arrays in scores.items()}
        draws = {contrast: [] for contrast in contrasts}
        rng = np.random.default_rng(2026)
        for index in range(n_bootstrap):
            frequencies = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            weights = frequencies[group_indices]
            if not np.sum(weights[labels == 1]) or not np.sum(weights[labels == 0]):
                continue
            current = {name: float(np.average([roc_auc_score(labels, s, sample_weight=weights) for s, count in arrays],
                                             weights=[count for s, count in arrays]))
                       for name, arrays in scores.items()}
            for reference_name, challenger in contrasts:
                draws[(reference_name, challenger)].append(current[challenger] - current[reference_name])
            if (index + 1) % 250 == 0:
                LOGGER.info("Ranking bootstrap checkpoint split=%s completed=%d total=%d", split, index+1, n_bootstrap)
        intervals = [{"reference": first, "challenger": second,
                      "difference": observed[second]-observed[first],
                      "ci95": np.quantile(draws[(first, second)], [.025, .975]).tolist(),
                      "n_valid": len(draws[(first, second)])} for first, second in contrasts]
        ranking[split] = {"budget": 128, "n_groups": len(groups), "auc128": observed, "contrasts": intervals}
    output = {"analysis_status": "post-hoc reviewer-requested; primary 5% experiment unchanged",
              "ranking": ranking, "calibration_sensitivity": sensitivity, "external_intervals": external,
              "completion_profiles": completion_profiles, "strict_prefix_recall128": strict_recall,
              "bootstrap": {"repetitions": n_bootstrap, "seed": 2026,
                            "unit": "prompt group; paired across methods and fixed fitted seeds"},
              "wilson_scope": "Per-seed descriptive binomial intervals; no seed pooling or selection-bias correction.",
              "prediction_sha256": hashes, "primary_metrics_sha256": hash_file(results / "metrics.json")}
    save_json(results / "review_robustness.json", output)
    LOGGER.info("Review analysis saved: methods=%d seeds=%d threshold_grid=3 splits=2", len(METHODS), len(seeds))
    return output
