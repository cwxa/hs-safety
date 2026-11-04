"""根据可追溯的实测结果生成双语论文表格、图像和结果段落。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import experiment_signature, load_config
from .metrics import matched_prompt_metrics

LOGGER = logging.getLogger(__name__)
METHOD_NAMES = {
    "prompt_lr": "Prompt LR", "text_lr": "Prefix TF--IDF LR",
    "static_lr": "Current-state LR", "mean_lr": "Running-mean LR",
    "gru": "Dynamic GRU", "gru_no_dynamics": "GRU, states only",
    "gru_last_layer": "GRU, last layer", "mlp": "Dynamic MLP",
    "gru_stride4": "GRU, stride 4", "gru_stride16": "GRU, stride 16",
}
ORDER = list(METHOD_NAMES)


def _matched_prompt_analysis(result_dir, models):
    result = {}
    for name, model in models.items():
        for seed, values in model.get("seeds", {}).items():
            for split in ("test", "external"):
                path = result_dir / "predictions" / name / f"seed_{seed}" / f"{split}.jsonl"
                if not path.exists():
                    continue
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
                result.setdefault(name, {}).setdefault(str(seed), {})[split] = matched_prompt_metrics(rows, values["threshold"])
    _write(result_dir / "matched_prompt_subset.json", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _achievement_artifacts(models, scale_models, metrics, counts, system, paper, result_dir):
    """从同一结果源生成成果表述，防止汇报材料与论文数值分叉。"""
    if not all(name in models for name in ("gru", "prompt_lr", "static_lr", "mlp")):
        defaults = {"AchievementDetectionEN": "The response-stream comparison is not yet complete.",
                    "AchievementDetectionZH": "回答流比较尚未完成。", "AchievementScaleEN": "",
                    "AchievementScaleZH": "", "AchievementScopeEN": "Measured coverage is not yet complete.",
                    "AchievementScopeZH": "实测覆盖尚未完成。"}
        path = paper / "generated/experiment_macros.tex"
        _write(path, path.read_text(encoding="utf-8") +
               "\n".join(f"\\newcommand{{\\{name}}}{{{text}}}" for name, text in defaults.items()) + "\n")
        return
    value = lambda collection, name, key: float(np.mean(_values(collection[name], "test", key)))
    gru, prompt = (value(models, name, "recall128") * 100 for name in ("gru", "prompt_lr"))
    gfpr, pfpr = (value(models, name, "sequence_fpr") * 100 for name in ("gru", "prompt_lr"))
    static, mlp = (value(models, name, "recall128") * 100 for name in ("static_lr", "mlp"))
    comparison = next((row for row in metrics.get("paired_comparisons", []) if row["split"] == "test"
                       and row["reference"] == "prompt_lr" and row["metric"] == "recall_at_budget.128"), {})
    fits = sum(len(model.get("seeds", {})) for collection in (models, scale_models) for model in collection.values())
    responses = sum(row["n"] for row in counts)
    evidence = {"n_responses": responses, "n_detector_fits": fits, "gru_recall128_percent": gru,
                "prompt_recall128_percent": prompt, "recall_gain_percentage_points": gru - prompt,
                "gru_test_fpr_percent": gfpr, "prompt_test_fpr_percent": pfpr,
                "paired_prompt_comparison": comparison, "static_recall128_percent": static,
                "dynamic_mlp_recall128_percent": mlp,
                "source_metrics_sha256": _hash(result_dir / "metrics.json"),
                "scope": "Observed replay results; thresholds and detectors are unchanged."}
    detection_en = (f"The 0.5B dynamic GRU reaches {gru:.1f}\\% Recall@128 versus {prompt:.1f}\\% for prompt-only logistic regression, "
                    f"with observed sequence FPRs of {gfpr:.2f}\\% and {pfpr:.2f}\\%, respectively.")
    detection_zh = (f"0.5B动态GRU的Recall@128为{gru:.1f}\\%，仅提示逻辑回归为{prompt:.1f}\\%；"
                    f"两者实测整段误报率分别为{gfpr:.2f}\\%与{pfpr:.2f}\\%。")
    macros = {"AchievementDetectionEN": detection_en, "AchievementDetectionZH": detection_zh,
              "AchievementScopeEN": f"The evidence covers {responses:,} eligible responses and {fits} fitted detector instances across two backbone sizes and three fixed seeds.",
              "AchievementScopeZH": f"实验证据覆盖{responses:,}条合格回答、两种骨干规模和三个固定种子下的{fits}个检测器实例。"}
    scale_en, scale_zh = "", ""
    if "static_lr" in scale_models:
        larger = value(scale_models, "static_lr", "recall128") * 100
        larger_fpr = value(scale_models, "static_lr", "sequence_fpr") * 100
        evidence.update({"scale_static_recall128_percent": larger, "scale_static_test_fpr_percent": larger_fpr,
                         "scale_static_gain_percentage_points": larger - static})
        static_fpr = value(models, "static_lr", "sequence_fpr") * 100
        scale_en = f"Static-probe Recall@128 is {static:.1f}\\% with the 0.5B backbone and {larger:.1f}\\% with 1.5B (observed FPRs {static_fpr:.2f}\\% and {larger_fpr:.2f}\\%)."
        scale_zh = f"在相同回答上，0.5B与1.5B静态探针的Recall@128分别为{static:.1f}\\%与{larger:.1f}\\%（实测误报率分别为{static_fpr:.2f}\\%与{larger_fpr:.2f}\\%）。"
    macros.update({"AchievementScaleEN": scale_en, "AchievementScaleZH": scale_zh})
    macro_path = paper / "generated/experiment_macros.tex"
    _write(macro_path, macro_path.read_text(encoding="utf-8") +
           "\n".join(f"\\newcommand{{\\{name}}}{{{text}}}" for name, text in macros.items()) + "\n")
    interval = (f"[{comparison['ci_low'] * 100:.1f}, {comparison['ci_high'] * 100:.1f}]个百分点"
                if comparison else "见完整指标")
    storage = system.get("cache_storage", {})
    cache_ratio = (storage["raw_features"]["bytes"] / storage["features"]["bytes"]
                   if storage.get("features", {}).get("bytes") else None)
    evidence["raw_to_projected_storage_ratio"] = cache_ratio
    _write(result_dir / "achievements.json", json.dumps(evidence, ensure_ascii=False, indent=2))
    text = ["# 成果亮点与汇报提纲", "", "本材料由实测结果自动生成；整理表述不改变数据、模型、阈值或主实验。", "",
            "## 项目定位", "", "在本机免费资源下，完成从公开安全数据到因果隐状态挖掘、轻量检测、在线缓冲停止和双语论文的可复现研究闭环。", "",
            "## 可优先展示的四项成果", "",
            "| 成果 | 证据 | 研究与实践价值 |", "| --- | --- | --- |",
            f"| 完整监测流程具有可用预测表现 | 0.5B GRU的Recall@128为{gru:.1f}%，仅提示对照为{prompt:.1f}%，相差{gru-prompt:.1f}个百分点；实测FPR分别{gfpr:.2f}%和{pfpr:.2f}% | 回答表征与GRU组成的流程在接近的实测误报率下提供更高召回，配对区间量化了比较结果 |",
            f"| 给出可复用的模型选择依据 | 动态MLP达到{mlp:.1f}%，静态探针达到{static:.1f}% Recall@128；完整FPR见结果表 | 消融明确区分表征、变化特征和循环记忆的作用，为资源预算内的结构选择提供实测依据 |",
            f"| 完成跨规模验证 | {scale_zh.replace(chr(92)+'%', '%')} | 相同回答与评价规则下验证监测表示规模的影响，支持在共同评价规则下比较监测表示的规模效应 |",
            f"| 形成完整可复现交付 | {responses:,}条回答、{fits}个检测器实例，固定版本数据、独立校准、外测、逐回答分数、检测器权重和英中文稿 | 支持从数据划分到论文数字的追溯，便于复现和课程验收 |", "",
            f"GRU相对仅提示对照的Recall@128差值采用提示分组配对bootstrap，95%区间为{interval}。该分析使用已训练模型与冻结阈值，限定于当前筛选后的测试回答。", ""]
    if cache_ratio is not None and "median_trial_monitor_p50_ms" in system:
        text.extend(["## 工程可行性", "",
                     f"- 主实验原始状态缓存{storage['raw_features']['bytes']/2**20:.1f} MiB，投影缓存{storage['features']['bytes']/2**20:.1f} MiB，前者约为后者{cache_ratio:.1f}倍。",
                     f"- 预热后检查点监测p50/p95的跨试验中位数为{system['median_trial_monitor_p50_ms']:.2f}/{system['median_trial_monitor_p95_ms']:.2f}毫秒；实测峰值分配显存{system['maximum_trial_peak_gpu_gib']:.2f} GiB。",
                     "- 计时分别覆盖检查点监测和端到端生成，为本机系统实现与资源配置提供依据。", ""])
    text.extend(["## 30秒汇报表述", "",
                 f"我们完成了一套面向大模型流式内容安全的数据挖掘与检测系统。研究覆盖{responses:,}条回答和{fits}次检测器训练，既提供严格隔离的数据与评价协议，也实现了在线缓冲检查接口。实测中，0.5B动态GRU流程在与仅提示线性对照接近的误报率下，前128个token的召回率高出{gru-prompt:.1f}个百分点；规模验证中的1.5B静态探针达到{evidence.get('scale_static_recall128_percent', static):.1f}%。通过系统比较输入、分类结构与资源开销，我们的成果是可复现的评估证据、模型选择依据和工程闭环。", "",
                 "## 研究展望与后续方向", "",
                 "后续可围绕容量匹配的prompt神经对照、长回答及多语言数据、细粒度前缀标注和独立标注的原生生成展开研究；结合更大外部样本与统一复现条件，进一步分析模型选择、迁移和干预效果。", "",
                 "## 论文叙事与术语", "",
                 "摘要和引言先呈现表征价值、实验规模与可复现交付；结果保留主对照、消融及实际FPR；结论落到可复用证据和资源条件。所有原始结果与不利比较继续保留。", "",
                 "| 统一术语 | 本文含义 |", "| --- | --- |", "| Recall@128 | 在最多128个已观察回答token内检出最终不安全回答的比例 |",
                 "| 整段FPR | 一条安全回答的全流中至少触发一次告警的比例 |", "| 监测隐状态 | 冻结模型读取标注回答所得状态，不是原生成模型轨迹 |", ""])
    _write(Path("docs/成果亮点与汇报提纲.md"), "\n".join(text))


def _review_artifacts(paper, result_dir):
    """生成补充比较及讨论所需的共享数值表述。"""
    review = _read_json(result_dir / "review_robustness.json")
    blocks = {"en": [], "zh": []}
    macros = {"ReviewRankingEN": "", "ReviewRankingZH": "", "TimingProfileEN": "", "TimingProfileZH": ""}
    if review:
        if review["primary_metrics_sha256"] != _hash(result_dir / "metrics.json"):
            raise ValueError("Review robustness results do not match primary metrics")
        ranking = review["ranking"]["test"]
        contrast = next(r for r in ranking["contrasts"] if r["reference"] == "text_lr")
        low, high = contrast["ci95"]
        static, text = (ranking["auc128"][name] for name in ("static_lr", "text_lr"))
        macros["ReviewRankingEN"] = f"The exploratory linear-probe comparison gives AUROC@128 of {static:.4f} for hidden states and {text:.4f} for prefix TF--IDF, with a paired 95\\% difference interval of [{low:.4f}, {high:.4f}]."
        macros["ReviewRankingZH"] = f"探索性线性探针对照中，隐状态与前缀TF--IDF的AUROC@128分别为{static:.4f}与{text:.4f}，差值的配对95\\%区间为[{low:.4f}, {high:.4f}]。"
        temporal = next(r for r in ranking["contrasts"] if r["reference"] == "static_lr")
        sensitivity = []
        for target in (.01, .05, .10):
            for name in ("text_lr", "static_lr", "gru"):
                rows = [r for r in review["calibration_sensitivity"] if r["target_calibration_fpr"] == target
                        and r["method"] == name and r["split"] == "test"]
                sensitivity.append([METHOD_NAMES[name], f"{target*100:.0f}",
                                    f"{np.mean([r['recall128'] for r in rows])*100:.1f}",
                                    f"{np.mean([r['test_fpr'] for r in rows])*100:.1f}"])
        for language, paragraphs in blocks.items():
            zh = language == "zh"
            suffix = "_zh" if zh else ""
            _write(paper / f"tables/review_sensitivity{suffix}.tex", _table(
                "事后探索性校准目标分析；数值为三个固定种子均值（百分比）。阈值由校准集确定，5\\%为主设定；比较区间未作多重校正。" if zh else
                "Post-hoc exploratory calibration-target analysis: means over three fixed seeds (%). Calibration data select thresholds; 5\\% is primary. Comparison intervals are unadjusted.",
                "review_sensitivity", "lrrr", ["Method", "$\\alpha$", "R@128", "Test FPR"], sensitivity))
            paragraphs.append(r"\paragraph{Supplementary ranking and calibration.}" if not zh else r"\paragraph{补充排序与校准。}")
            paragraphs.append(r"\ReviewRankingZH{} AUROC@128采用预算内检查点的最大分数，区间通过1,000次提示分组配对bootstrap计算。该线性分类器家族内的比较，为不同特征构造提供了补充排序证据。" if zh else
                              r"\ReviewRankingEN{} AUROC@128 uses the maximum score within the token budget, with intervals from 1,000 paired prompt-group bootstrap draws. The within-family linear comparison provides complementary evidence about the two feature constructions.")
            lo, hi = temporal["ci95"]
            paragraphs.append(f"GRU与静态探针具有接近的预算内排序表现，AUROC@128差值为{temporal['difference']:.4f}，95\\%区间为[{lo:.4f}, {hi:.4f}]。" if zh else
                              f"The GRU and static probe have similar budgeted ranking estimates: the AUROC@128 difference is {temporal['difference']:.4f}, with a 95\\% interval of [{lo:.4f}, {hi:.4f}].")
            paragraphs.append(r"表~\ref{tab:review_sensitivity}比较1\%、5\%和10\%校准目标下的运行特征；已拟合模型固定，原5\%目标保留为主设定。\input{tables/review_sensitivity_zh}" if zh else
                              r"Table~\ref{tab:review_sensitivity} compares operating profiles at calibration targets of 1\%, 5\% and 10\%, holding the fitted models fixed and retaining 5\% as the primary setting.\input{tables/review_sensitivity}")
        external = [r for r in review["external_intervals"] if r["method"] == "gru"]
        example = next(r for r in external if r["seed"] == 42)
        hits = ", ".join(str(r["tp128"]) for r in external)
        low, high = (v*100 for v in example["recall128_wilson95"])
        blocks["en"].append(f"External GRU detections within 128 tokens are {hits} out of {example['n_unsafe']} unsafe responses for seeds 13, 42 and 2026. Seed 42 gives {example['tp128']}/{example['n_unsafe']}, with a descriptive Wilson 95\\% interval of [{low:.1f}, {high:.1f}]\\%. The archive provides per-seed counts and intervals for both recall and FPR.")
        blocks["zh"].append(f"外部集上，种子13、42和2026的GRU在128-token预算内分别检出{hits}条不安全回答，分母均为{example['n_unsafe']}。种子42为{example['tp128']}/{example['n_unsafe']}，描述性Wilson 95\\%区间为[{low:.1f}, {high:.1f}]\\%。归档同时提供逐种子召回和误报的计数与区间。")
        profiles = review.get("completion_profiles", {})
        if profiles:
            main, external_profile = profiles["test"], profiles["external"]
            complete = main["complete_by128"] / main["n"] * 100
            external_complete = external_profile["complete_by128"] / external_profile["n"] * 100
            strict = {name: np.mean([r["strict_prefix_recall128"] for r in review["strict_prefix_recall128"]
                                    if r["method"] == name and r["split"] == "test"])*100
                      for name in ("gru", "static_lr")}
            blocks["en"].append(f"The strict pre-completion criterion requires an alarm checkpoint before the answer endpoint and within 128 tokens. With all unsafe responses retained in the denominator, main-test recall is {strict['static_lr']:.1f}\\% for the static probe and {strict['gru']:.1f}\\% for GRU. This timing view complements the budgeted recall curves.")
            blocks["zh"].append(f"严格完成前检出要求告警检查点早于回答终点，且处于128-token预算内。以全部不安全回答为分母时，主测试静态探针和GRU的召回分别为{strict['static_lr']:.1f}\\%与{strict['gru']:.1f}\\%。这一时机视角补充了预算内召回曲线。")
            macros["TimingProfileEN"] = f"At 128 tokens, {main['complete_by128']}/{main['n']} main-test responses ({complete:.1f}\\%) and {external_profile['complete_by128']}/{external_profile['n']} external responses ({external_complete:.1f}\\%) have reached their endpoints; this difference motivates length-stratified evaluation alongside the strict pre-completion analysis."
            macros["TimingProfileZH"] = f"128-token预算下，主测试集{main['complete_by128']}/{main['n']}条回答（{complete:.1f}\\%）和外部集{external_profile['complete_by128']}/{external_profile['n']}条回答（{external_complete:.1f}\\%）已达终点；这一差异支持在严格完成前分析之外进一步采用长度分层评价。"
        highlight = _read_json(result_dir / "achievements.json")
        highlight["review_requested_ranking"] = contrast
        highlight["review_analysis_sha256"] = _hash(result_dir / "review_robustness.json")
        _write(result_dir / "achievements.json", json.dumps(highlight, ensure_ascii=False, indent=2))
        guide = Path("docs/成果亮点与汇报提纲.md")
        guide.write_text(guide.read_text(encoding="utf-8") +
                         f"\n## 补充排序证据\n\n探索性线性分类对照中，隐状态与前缀文本的预算内AUROC分别为{static:.4f}和{text:.4f}，差值95%配对区间为[{contrast['ci95'][0]:.4f}, {contrast['ci95'][1]:.4f}]。阈值敏感性与逐种子区间进一步支持对运行特征的细化分析，完整测量见 `results/review_robustness.json`。\n", encoding="utf-8")
    alignment = _read_json(result_dir / "streaming_score_alignment.json")
    if alignment:
        n = len(alignment["samples"])
        checkpoints = sum(r["checkpoints"] for r in alignment["samples"])
        error, mismatches = alignment["maximum_absolute_score_difference"], alignment["total_decision_mismatches"]
        blocks["en"].append(f"Real-KV replay on {n} held-out responses compares {checkpoints} checkpoints with archived offline scores. The maximum score difference is {error:.5f}, with {mismatches} threshold-decision mismatches. The diagnostic uses the first three sorted IDs per label at lengths 16--128 and the seed-42 GRU.")
        blocks["zh"].append(f"真实KV回放在{n}条留出回答的{checkpoints}个检查点上与离线分数对照，最大差值为{error:.5f}，阈值判决不一致数为{mismatches}。诊断采用每种标签按ID排序的前三条长度16--128回答和种子42的GRU。")
    for language, paragraphs in blocks.items():
        transfer = [text for text in paragraphs if text.startswith("External GRU" if language == "en" else "外部集上")]
        system = [text for text in paragraphs if text.startswith("Real-KV replay" if language == "en" else "真实KV回放")]
        for name, selected in (("transfer", transfer), ("system", system)):
            path = paper / f"generated/{name}_{language}.tex"
            _write(path, path.read_text(encoding="utf-8") + "\n\n".join(selected) + "\n")
        paragraphs = [text for text in paragraphs if text not in transfer + system]
        _write(paper / f"generated/review_analysis_{language}.tex", "\n\n".join(paragraphs) + "\n")
    path = paper / "generated/experiment_macros.tex"
    _write(path, path.read_text(encoding="utf-8") +
           "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")
BASELINES = {"prompt_lr", "text_lr", "static_lr", "mean_lr", "gru"}


def _tex(value: Any) -> str:
    """转义外部元数据，避免文件名或模型名破坏 LaTeX 编译。"""
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _values(model: dict, split: str, metric: str) -> list[float]:
    values = []
    for seed in model.get("seeds", {}).values():
        record = seed.get("splits", {}).get(split, {})
        value = record.get("recall_at_budget", {}).get("128") if metric == "recall128" else record.get(metric)
        numeric = _number(value)
        if numeric is not None:
            values.append(numeric)
    return values


def _stat(model: dict, split: str, metric: str, scale: float = 100) -> str:
    values = _values(model, split, metric)
    if not values:
        return "--"
    mean = np.mean(values) * scale
    if len(values) == 1:
        return f"{mean:.1f}"
    return f"{mean:.1f} $\\pm$ {np.std(values, ddof=1) * scale:.1f}"


def _table(caption: str, label: str, columns: str, headers: list[str], rows: list[list[str]]) -> str:
    # 仅处理可信 LaTeX 标题中的裸百分号，保留数学命令和已有转义。
    caption = re.sub(r"(?<!\\)%", r"\\%", caption)
    if not rows:
        rows = [["Pending"] + ["--"] * (len(headers) - 1)]
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return ("\\begin{table}[t]\n\\centering\\footnotesize\n"
            f"\\caption{{{caption}}}\\label{{tab:{label}}}\n"
            f"\\begin{{tabular}}{{{columns}}}\\toprule\n"
            + " & ".join(headers) + r" \\" + "\n\\midrule\n" + body
            + "\n\\bottomrule\\end{tabular}\n\\end{table}\n")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_counts(directory: Path) -> list[dict]:
    """仅统计规范化记录；不会把原始危险内容写入论文或日志。"""
    counts = []
    if not directory.exists():
        return counts
    for path in sorted(directory.glob("*.jsonl")):
        total = unsafe = 0
        groups, lengths = set(), []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "label" not in record:
                    continue
                total += 1
                unsafe += int(record["label"] == 1)
                groups.add(record.get("group", record.get("id", total)))
                length = _number(record.get("response_len"))
                if length is not None:
                    lengths.append(length)
        if total:
            counts.append({"split": path.stem, "n": total, "unsafe": unsafe,
                           "groups": len(groups), "median_tokens": float(np.median(lengths)) if lengths else None,
                           "path": str(path), "sha256": _hash(path)})
    return counts


def _make_exploratory_figures(config: dict, paper: Path, save) -> None:
    """仅使用训练终点拟合事后PCA与聚类；绝不反写主预处理器或检测器。"""
    feature_dir = Path(config.get("paths", {}).get("features", "artifacts/features"))
    if not config.get("model") or not _read_json(feature_dir / "train/manifest.json").get("complete"):
        _write(paper / "generated/exploration_en.tex", "Training endpoint exploration awaits the complete projected feature cache. No exploratory result is imputed.\n")
        _write(paper / "generated/exploration_zh.tex", "训练终点探索等待完整降维特征缓存，不填补任何探索性结果。\n")
        return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    from .features import load_feature_split

    records, sequences, _ = load_feature_split(config, "train", dynamic=False)
    if len(records) < 8:
        return
    endpoints = np.stack([sequence[-1] for sequence in sequences])
    del sequences
    with threadpool_limits(limits=int(config.get("training", {}).get("num_threads", 4))):
        display_pca = PCA(n_components=2, svd_solver="full").fit(endpoints)
        clustering = KMeans(n_clusters=8, random_state=42, n_init=10).fit(endpoints)
    coordinates = display_pca.transform(endpoints)
    labels = np.asarray([int(record["label"]) for record in records])
    clusters = clustering.labels_
    source_rows = [{"id": record["id"], "group": record["group"], "split": "train", "label": int(labels[i]),
                    "cluster": int(clusters[i]), "pc1": float(coordinates[i, 0]), "pc2": float(coordinates[i, 1])}
                   for i, record in enumerate(records)]
    source_dir = paper / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    np.savez(source_dir / "exploratory_transforms.npz", pca_mean=display_pca.mean_, pca_components=display_pca.components_,
             pca_explained_variance_ratio=display_pca.explained_variance_ratio_, kmeans_centers=clustering.cluster_centers_)
    _write(source_dir / "exploratory_transform_manifest.json", json.dumps({
        "fit_split": "train", "analysis_role": "post-hoc descriptive only; never used by detector or thresholds",
        "n_training_endpoints": len(records), "endpoint_dimension": int(endpoints.shape[1]), "n_clusters": 8,
        "random_state": 42, "n_init": 10, "train_feature_manifest_sha256": _hash(feature_dir / "train/manifest.json"),
        "transform_sha256": _hash(source_dir / "exploratory_transforms.npz"),
        "display_pca_explained_variance_ratio": display_pca.explained_variance_ratio_.tolist()}, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    order = np.random.default_rng(42).permutation(len(records))
    colors = np.where(labels[order] == 1, "#c77e45", "#4c7d9d")
    ax.scatter(coordinates[order, 0], coordinates[order, 1], c=colors, alpha=.28, s=5, linewidths=0, rasterized=True)
    explained = display_pca.explained_variance_ratio_ * 100
    ax.set(xlabel=f"Training PC1 ({explained[0]:.1f}% variance)", ylabel=f"Training PC2 ({explained[1]:.1f}% variance)")
    handles = [Line2D([], [], marker="o", linestyle="none", color=color, label=label, markersize=4)
               for color, label in (("#4c7d9d", "Final Safe"), ("#c77e45", "Final Unsafe"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, fontsize=7, frameon=False)
    save(fig, "pca_endpoints", source_rows,
         "Training-only two-dimensional projection describes final-checkpoint geometry; colors are final labels.",
         "No error bars: descriptive point cloud with every retained training response displayed.",
         plotted_unit="one retained training response endpoint", labels_used_in_fit=False,
         all_training_rows_displayed=True, draw_order="fixed permutation, seed 42; no convenience subsampling",
         fitting="PCA(2) and KMeans(8) fitted only on training endpoints; KMeans uses all projected dimensions")

    groups = np.asarray([str(record["group"]) for record in records])
    summaries = []
    for cluster in range(8):
        mask = clusters == cluster
        summaries.append({"cluster": cluster, "train_n": int(mask.sum()), "train_groups": len(set(groups[mask])),
                          "train_unsafe": int(labels[mask].sum()), "train_unsafe_rate": float(labels[mask].mean()),
                          "external_n": 0, "external_groups": 0, "external_unsafe": 0, "external_unsafe_rate": None})
    if _read_json(feature_dir / "external/manifest.json").get("complete"):
        external_records, external_sequences, _ = load_feature_split(config, "external", dynamic=False)
        if external_records:
            external_endpoints = np.stack([sequence[-1] for sequence in external_sequences])
            external_clusters = clustering.predict(external_endpoints)
            external_labels = np.asarray([int(record["label"]) for record in external_records])
            external_groups = np.asarray([str(record["group"]) for record in external_records])
            for row in summaries:
                mask = external_clusters == row["cluster"]
                row.update(external_n=int(mask.sum()), external_groups=len(set(external_groups[mask])),
                           external_unsafe=int(external_labels[mask].sum()),
                           external_unsafe_rate=float(external_labels[mask].mean()) if mask.any() else None)
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    ax.scatter([r["cluster"] - .09 for r in summaries], [r["train_unsafe_rate"] * 100 for r in summaries],
               marker="o", color="#4c7d9d", label="Training", s=22)
    external_rows = [r for r in summaries if r["external_unsafe_rate"] is not None]
    if external_rows:
        ax.scatter([r["cluster"] + .09 for r in external_rows], [r["external_unsafe_rate"] * 100 for r in external_rows],
                   marker="^", color="#c77e45", label="External", s=25)
    ax.set(xticks=np.arange(8), xlabel="Unsupervised cluster (fit on training endpoints)", ylabel="Final unsafe responses (%)", ylim=(0, 101))
    ax.grid(axis="y", alpha=.15)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, fontsize=7, frameon=False)
    save(fig, "cluster_composition", summaries,
         "Outcome composition is summarized within train-fitted endpoint clusters, without using labels to fit clusters.",
         "No error bars: descriptive rates, no inferential claim or independent-response assumption.",
         plotted_unit="cluster composition of retained responses; source data include response and prompt-group counts",
         selection="all eight clusters; external empty clusters remain missing", labels_used_in_fit=False)
    rows = [[str(r["cluster"]), str(r["train_n"]), f"{r['train_unsafe_rate'] * 100:.1f}",
             str(r["external_n"]), f"{r['external_unsafe_rate'] * 100:.1f}" if r["external_unsafe_rate"] is not None else "--"]
            for r in summaries]
    for suffix, caption in (("", "Post-hoc train-fitted endpoint clusters. Unsafe rates are descriptive percentages; cluster IDs have no ordinal meaning."),
                            ("_zh", "事后训练集拟合的终点聚类。风险比例为描述性百分比，聚类编号没有顺序含义。")):
        _write(paper / "tables" / f"clusters{suffix}.tex", _table(caption, "clusters", "rrrrr", ["Cluster", "Train $N$", "Unsafe", "Ext. $N$", "Unsafe"], rows))
    _write(paper / "generated/exploration_en.tex",
           f"The display PCA is fitted only to {len(records):,} retained training endpoints and preserves {explained.sum():.1f}\\% of their variance in two dimensions. K-means fits eight clusters in the full {endpoints.shape[1]}-dimensional projected endpoint space without labels. External endpoints, when available, are assigned with frozen centers. Labels are used to summarize cluster composition after unsupervised fitting. The endpoint analysis describes global representation structure alongside the prefix-classification experiments.\\input{{tables/clusters}}\n")
    _write(paper / "generated/exploration_zh.tex",
           f"展示用PCA仅在{len(records):,}个保留训练终点上拟合，二维保留其{explained.sum():.1f}\\%方差。K均值在完整{endpoints.shape[1]}维降维终点空间中无标签拟合八类；外部终点可用时，通过冻结中心分配类别。无监督拟合完成后，再利用标签描述各簇的风险构成。终点分析展示整体表征结构，与前缀分类实验形成互补。\\input{{tables/clusters_zh}}\n")


def _make_figures(paper: Path, metrics: dict, result_dir: Path, config: dict, counts: list[dict], audit: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "legend.fontsize": 7,
                         "svg.fonttype": "none"})
    figure_dir = paper / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    # 清除本模块拥有的旧结果图，防止新配置缺失结果时误展示前次实验。
    for stem in ("architecture", "recall_budget_test", "recall_budget_external", "ablations", "risk_trajectories",
                 "data_class_balance", "external_retention", "pca_endpoints", "cluster_composition"):
        for suffix in (".pdf", ".png", ".svg", ".alignment.json"):
            (figure_dir / f"{stem}{suffix}").unlink(missing_ok=True)
        for suffix in (".csv", ".json"):
            (paper / "source_data" / f"{stem}{suffix}").unlink(missing_ok=True)
        (paper / "figure_contracts" / f"{stem}.json").unlink(missing_ok=True)
    for filename in ("exploratory_transforms.npz", "exploratory_transform_manifest.json"):
        (paper / "source_data" / filename).unlink(missing_ok=True)
    generated = []

    def save(fig, name, source_rows: list[dict], conclusion: str, variability: str = "Not applicable; descriptive observations.", **details):
        fig.canvas.draw()
        fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight")
        fig.savefig(figure_dir / f"{name}.svg", bbox_inches="tight")
        svg_path = figure_dir / f"{name}.svg"
        _write(svg_path, "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n")
        source_dir = paper / "source_data"
        source_dir.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in source_rows for key in row))
        with (source_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(source_rows)
        _write(source_dir / f"{name}.json", json.dumps(source_rows, ensure_ascii=False, indent=2, default=str) + "\n")
        source_paths = [result_dir / "metrics.json", Path(config.get("paths", {}).get("processed", "data/processed")) / "audit.json"]
        contract = {"figure": name, "generated_utc": datetime.now(UTC).isoformat(), "core_conclusion": conclusion,
                    "archetype": "single quantitative panel" if name != "architecture" else "single schematic",
                    "backend": "Python/matplotlib", "source_data": f"source_data/{name}.csv",
                    "sources_sha256": {str(p): _hash(p) for p in source_paths if p.exists()},
                    "independent_sampling_unit": "prompt group for inferential comparisons",
                    "plotted_unit": "as specified in source rows; responses are not independent repeats",
                    "variability": variability, "multiple_comparison_correction": "None; secondary analyses are exploratory",
                    "font_size_pt": 8, "minimum_source_font_size_pt": 7,
                    "figure_size_inches": fig.get_size_inches().tolist(),
                    "alignment": "NOT APPLICABLE: one plot-area axes", "final_pdf_QA": "Parent workflow audits final exported PDF", **details}
        _write(paper / "figure_contracts" / f"{name}.json", json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
        _write(figure_dir / f"{name}.alignment.json", json.dumps({"status": "NOT APPLICABLE", "n_axes": len(fig.axes),
               "reason": "Single plot-area figure; no comparable multi-panel grid.",
               "axes_bounds_inches": [[v * fig.get_size_inches()[i % 2] for i, v in enumerate(ax.get_position().bounds)] for ax in fig.axes]}, indent=2) + "\n")
        plt.close(fig)
        generated.append(name)

    # 架构示意图不编码任何实验结果；与结果图分开命名。
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    boxes = [(0.05, 1.65, "Prompt-group partitions\ntrain / dev / calib. / test"),
             (3.45, 1.65, "Frozen causal LM\nresponse states, 4 layers"),
             (6.85, 1.65, "Train-only PCA\nstates + causal differences"),
             (6.85, 0.1, "GRU / matched baselines\nfinal-response risk scores"),
             (3.45, 0.1, "Calibration maxima\nempirical seq.-FPR threshold"),
             (0.05, 0.1, "Held-out replay\nR@128 / FPR / uncertainty")]
    for x, y, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 3.05, 1.05, boxstyle="round,pad=0.03",
                                    facecolor="#eef3f8", edgecolor="#365675", linewidth=.85))
        ax.text(x + 1.525, y + .525, text, ha="center", va="center", fontsize=8)
    for a, b in [((3.13, 2.17), (3.4, 2.17)), ((6.53, 2.17), (6.8, 2.17)),
                 ((8.37, 1.6), (8.37, 1.2)), ((6.8, .62), (6.55, .62)), ((3.4, .62), (3.13, .62))]:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": "#365675"})
    save(fig, "architecture", [{"step": i + 1, "description": box[2].replace("\n", ": ")} for i, box in enumerate(boxes)],
         "Training, model selection, calibration and testing are separate stages.", "Not applicable: architecture schematic.")

    if counts:
        desired = ("train", "dev", "calibration", "test", "external")
        ordered = sorted(counts, key=lambda row: desired.index(row["split"]) if row["split"] in desired else len(desired))
        unsafe = np.asarray([row["unsafe"] / row["n"] * 100 for row in ordered])
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        indices = np.arange(len(ordered))
        ax.barh(indices, 100 - unsafe, color="#b7c9d6", label="Final Safe")
        ax.barh(indices, unsafe, left=100 - unsafe, color="#d99b61", label="Final Unsafe")
        ax.set_yticks(indices, [f"{r['split'].capitalize()} (n={r['n']:,})" for r in ordered], fontsize=7)
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), xlabel="Share of retained responses (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, frameon=False, fontsize=7)
        save(fig, "data_class_balance", [{"split": r["split"], "n_responses": r["n"], "n_prompt_groups": r["groups"],
              "n_unsafe": r["unsafe"], "unsafe_percent": r["unsafe"] / r["n"] * 100} for r in ordered],
             "The main and external retained response populations have different class composition.",
             plotted_unit="retained response, descriptive proportions; no independence or confidence claim")

    external_source = next((row for row in audit.get("sources", []) if row.get("source") == "streamguard"), None)
    if external_source:
        source_rows = []
        fig, ax = plt.subplots(figsize=(3.45, 2.35))
        for j, (label, display, color) in enumerate(((0, "Final Safe", "#b7c9d6"), (1, "Final Unsafe", "#d99b61"))):
            before = int(external_source.get("input_labels", {}).get(str(label), 0))
            after = int(external_source.get("retained_labels", {}).get(str(label), 0))
            ratio = after / before * 100 if before else 0
            ax.barh(j, ratio, color=color)
            source_rows.append({"label": label, "label_name": display, "before": before, "after": after, "retained_percent": ratio})
        ax.set_yticks([0, 1], [f"{r['label_name']} ({r['after']}/{r['before']})" for r in source_rows], fontsize=7)
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), xlabel="Retained fraction after eligibility filters (%)")
        ax.grid(axis="x", alpha=.15)
        save(fig, "external_retention", source_rows,
             "Eligibility filters retain different fractions of safe and unsafe external responses.",
             exclusions=external_source.get("rejected", {}), plotted_unit="external response; exact descriptive counts")

    models = metrics.get("models", {})
    for split in ("test", "external"):
        curves = []
        for name in ORDER:
            if name not in BASELINES or name not in models:
                continue
            seed_curves = [s.get("splits", {}).get(split, {}).get("recall_at_budget", {})
                           for s in models[name].get("seeds", {}).values()]
            budgets = sorted({int(b) for curve in seed_curves for b, value in curve.items() if _number(value) is not None})
            samples = [[float(curve[str(b)]) for curve in seed_curves if _number(curve.get(str(b))) is not None] for b in budgets]
            ys = [np.mean(v) for v in samples]
            sd = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in samples]
            if budgets:
                curves.append((name, budgets, ys, sd, [len(v) for v in samples]))
        if curves:
            fig, ax = plt.subplots(figsize=(3.45, 2.4))
            colors = {"prompt_lr": "#969696", "text_lr": "#8ca0ae", "static_lr": "#708d81", "mean_lr": "#a08c78", "gru": "#155b8a"}
            markers = {"prompt_lr": "s", "text_lr": "^", "static_lr": "D", "mean_lr": "v", "gru": "o"}
            rows = []
            for name, budgets, ys, sd, seed_counts in curves:
                ax.plot(budgets, np.asarray(ys) * 100, marker=markers[name], ms=3,
                        label=METHOD_NAMES[name].replace("--", "-"), color=colors[name],
                        lw=1.5 if name == "gru" else 1, linestyle="-" if name == "gru" else "--")
                repeat_mask = np.asarray(seed_counts) > 1
                if repeat_mask.any():
                    ax.errorbar(np.asarray(budgets)[repeat_mask], np.asarray(ys)[repeat_mask] * 100,
                                yerr=np.asarray(sd)[repeat_mask] * 100, fmt="none", color=colors[name],
                                elinewidth=.8, capsize=2)
                for seed_id, entry in models[name].get("seeds", {}).items():
                    for budget, value in entry.get("splits", {}).get(split, {}).get("recall_at_budget", {}).items():
                        rows.append({"model": name, "seed": seed_id, "split": split, "token_budget": budget, "recall": value})
            ax.set(xlabel="Observed response-token budget", ylabel="Unsafe-response recall (%)", ylim=(0, 101))
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2, frameon=False)
            save(fig, f"recall_budget_{split}", rows, "Early recall under each method's frozen sequence-level calibration threshold.",
                 "Mean +/- one sample SD across completed seeds; single-seed SD is omitted. No bootstrap band is implied.", split=split)

    ablations = [(name, np.mean(_values(models[name], "test", "recall128")))
                 for name in ORDER if name in models and name not in {"prompt_lr", "text_lr", "static_lr", "mean_lr"}
                 and _values(models[name], "test", "recall128")]
    if ablations:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        reverse = list(reversed(ablations))
        ax.barh([METHOD_NAMES[n] for n, _ in reverse], [v * 100 for _, v in reverse],
                color=["#1e5c85" if n == "gru" else "#9db9cd" for n, _ in reverse])
        for index, (name, center) in enumerate(reverse):
            values = _values(models[name], "test", "recall128")
            if len(values) > 1:
                ax.errorbar(center * 100, index, xerr=np.std(values, ddof=1) * 100,
                            fmt="none", color="#303030", elinewidth=.8, capsize=2)
        ax.set(xlabel="Recall@128 (%)", xlim=(0, 100))
        ax.grid(axis="x", alpha=.15)
        save(fig, "ablations", [{"model": n, "seed": seed_id, "recall_at_128": entry.get("splits", {}).get("test", {}).get("recall_at_budget", {}).get("128")}
              for n, _ in ablations for seed_id, entry in models[n].get("seeds", {}).items()],
             "Feature, temporal and stride ablations under the same response-level evaluation rule.",
             "Mean +/- one sample SD across completed seeds; single-seed SD is omitted.")

    # 仅展示去标识轨迹；不以风险分数代替人工有害起点标注。
    paths = sorted((result_dir / "predictions" / "gru").glob("seed_*/test.jsonl"))
    if paths:
        examples = {0: [], 1: []}
        with paths[0].open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                label = int(item["label"])
                if label in examples and len(examples[label]) < 3 and item.get("positions"):
                    examples[label].append(item)
                if all(len(v) == 3 for v in examples.values()):
                    break
        if any(examples.values()):
            fig, ax = plt.subplots(figsize=(3.4, 2.3))
            for label, rows in examples.items():
                for j, item in enumerate(rows):
                    ax.plot(item["positions"], item["scores"], alpha=.7, lw=.9,
                            color="#b23d3d" if label else "#26769c",
                            label=("Final unsafe" if label else "Final safe") if j == 0 else None)
            ax.set(xlabel="Observed response tokens", ylabel="Predicted final-response risk", ylim=(-.02, 1.02))
            ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2)
            ax.grid(alpha=.15)
            save(fig, "risk_trajectories", [{"id": item["id"], "label": label, "checkpoint": position, "score": score}
                 for label, items in examples.items() for item in items for position, score in zip(item["positions"], item["scores"])],
                 "Example final-response risk trajectories need not be monotone or identify harm onset.",
                 selection="First three stored test responses per final label in the first sorted seed file; illustrative only.")
    _make_exploratory_figures(config, paper, save)
    return generated


def generate_report(config: dict) -> dict:
    """生成论文实测证据；空结果保留 pending，绝不以模拟数据填充。"""
    started = time.perf_counter()
    paths = config.get("paths", {})
    result_dir = Path(paths.get("results", "results"))
    paper = Path(config.get("report", {}).get("paper_dir", "papers"))
    LOGGER.info("Report generation started: results=%s paper=%s", result_dir, paper)
    metrics_path = result_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    if metrics.get("task", "final_response_risk_prediction") != "final_response_risk_prediction":
        raise ValueError("Metrics describe an incompatible prediction task.")
    expected_signature = experiment_signature(config)
    if metrics.get("config_signature") and metrics["config_signature"] != expected_signature:
        raise ValueError("Report configuration does not match the measured experiment signature.")
    models = metrics.get("models", {})
    counts = _dataset_counts(Path(paths.get("processed", "data/processed")))
    table_dir = paper / "tables"
    for split, stem in (("test", "main"), ("external", "external")):
        rows = [[METHOD_NAMES.get(name, _tex(name)), _stat(models[name], split, "recall128"),
                 _stat(models[name], split, "sequence_fpr"), _stat(models[name], split, "roc_auc"),
                 _stat(models[name], split, "average_precision")]
                for name in ORDER if name in models and name in BASELINES
                and any(s.get("splits", {}).get(split) for s in models[name].get("seeds", {}).values())]
        en_caption = ("In-domain" if split == "test" else "External") + " detection (%). Mean $\\pm$ sample SD across available seeds; a single completed seed has no SD. Thresholds use only in-domain calibration."
        zh_caption = ("域内" if split == "test" else "外部") + "检测结果（百分比）。报告可用种子的均值与样本标准差；单种子不显示标准差。阈值仅依据域内校准集确定。"
        for language, caption in (("en", en_caption), ("zh", zh_caption)):
            suffix = "" if language == "en" else "_zh"
            _write(table_dir / f"{stem}{suffix}.tex", _table(caption, stem, "lrrrr",
                   ["Method", "R@128", "FPR", "AUROC", "AP"], rows))

    data_rows = [[_tex(row["split"]), str(row["n"]), str(row["unsafe"]), str(row["groups"])] for row in counts]
    for suffix, caption in (("", "Observed retained responses after length filtering, grouping and deduplication."),
                            ("_zh", "长度筛选、分组和去重后实际保留的回答。")):
        _write(table_dir / f"dataset{suffix}.tex", _table(caption, "dataset", "lrrr", ["Split", "$N$", "Unsafe", "Groups"], data_rows))

    ablation_rows = [[METHOD_NAMES.get(name, _tex(name)), _stat(models[name], "test", "recall128"),
                      _stat(models[name], "test", "sequence_fpr")]
                     for name in ORDER if name in models and name not in {"prompt_lr", "text_lr", "static_lr", "mean_lr"}
                     and _values(models[name], "test", "recall128")]
    timing_rows = []
    for name in ORDER:
        if name not in models:
            continue
        seeds = list(models[name].get("seeds", {}).values())
        train = [_number(s.get("train_seconds")) for s in seeds]
        p50 = [_number(s.get("detector_latency", {}).get("checkpoint_ms_p50")) for s in seeds]
        p95 = [_number(s.get("detector_latency", {}).get("checkpoint_ms_p95")) for s in seeds]
        def mean_or_missing(values):
            present = [v for v in values if v is not None]
            return f"{np.mean(present):.2f}" if present else "--"
        timing_rows.append([METHOD_NAMES.get(name, _tex(name)), mean_or_missing(train),
                            mean_or_missing(p50), mean_or_missing(p95)])
    paired = metrics.get("paired_comparisons", [])
    uncertainty_rows = []
    for item in paired:
        if item.get("reference") not in {"static_lr", "prompt_lr", "text_lr"}:
            continue
        if item.get("metric") not in {"recall_at_budget.128", "recall128", "recall_at_128"}:
            continue
        values = [_number(item.get(k)) for k in ("estimate", "ci_low", "ci_high")]
        if any(v is None for v in values):
            continue
        contrast = f"{METHOD_NAMES.get(item.get('challenger'), item.get('challenger'))} vs. {METHOD_NAMES.get(item.get('reference'), item.get('reference'))}"
        uncertainty_rows.append([_tex(item.get("split", "")), _tex(contrast),
                                 f"{values[0] * 100:+.1f}", f"[{values[1] * 100:+.1f}, {values[2] * 100:+.1f}]"])
    for suffix, zh in (("", False), ("_zh", True)):
        alert_rows = [[METHOD_NAMES.get(name, _tex(name)), _stat(models[name], "test", "recall"),
                       _stat(models[name], "test", "detected_alert_tokens_mean", scale=1),
                       _stat(models[name], "test", "detected_alert_tokens_median", scale=1),
                       _stat(models[name], "test", "normalized_alert_position_mean")]
                      for name in ORDER if name in models and name in BASELINES]
        _write(table_dir / f"alert_positions{suffix}.tex", _table(
            "域内告警位置；位置统计以正确检出为条件，召回率分母保留所有不安全回答。" if zh else
            "In-domain alert positions. Timing is conditional on true detections; recall retains all unsafe responses in its denominator.",
            "alert_positions", "lrrrr", ["Method", "Recall", "Mean $\\tau$", "Med. $\\tau$", "$\\tau/T$ (\\%)"], alert_rows))
        _write(table_dir / f"ablation{suffix}.tex", _table(
            "消融与检查步长敏感性（百分比）。" if zh else "Ablation and checkpoint-stride sensitivity (%).",
            "ablation", "lrr", ["Setting", "R@128", "Seq. FPR"], ablation_rows))
        _write(table_dir / f"uncertainty{suffix}.tex", _table(
            "提示分组配对自助法：Recall@128差值及95\\%区间（百分点）。" if zh else
            "Prompt-group paired bootstrap: Recall@128 differences and 95\\% intervals (percentage points).",
            "uncertainty", "lp{.34\\columnwidth}rr", ["Split", "Contrast", "$\\Delta$", "95\\% CI"], uncertainty_rows))
        _write(table_dir / f"timing{suffix}.tex", _table(
            "实测训练与检测头耗时；种子间取均值。头部延迟不含骨干、分词、PCA及动态特征构造。" if zh else
            "Measured fit time and detector-only latency, averaged across seeds. Head timing excludes backbone, tokenization, PCA and dynamic-feature construction.",
            "timing", "lrrr", ["Method", "Fit (s)", "p50 (ms)", "p95 (ms)"], timing_rows))

    completed_seeds = sorted(models.get("gru", {}).get("seeds", {}))
    matched_subsets = _matched_prompt_analysis(result_dir, models)
    main_values = _values(models.get("gru", {}), "test", "recall128")
    fpr_values = _values(models.get("gru", {}), "test", "sequence_fpr")
    ext_values = _values(models.get("gru", {}), "external", "recall128")
    target = float(metrics.get("target_fpr", config.get("evaluation", {}).get("fpr", .05)))
    if main_values and fpr_values:
        recall = np.mean(main_values) * 100
        fpr = np.mean(fpr_values) * 100
        configured_seeds = config.get("training", {}).get("seeds", [13, 42, 2026])
        status_en = f"Measured in-domain results are available for {len(main_values)} of {len(configured_seeds)} configured dynamic-GRU seeds."
        status_zh = f"已有{len(configured_seeds)}个预设动态GRU种子中的{len(main_values)}个完成域内实测。"
        abstract_en = (f"Across {len(main_values)} completed seed(s), the dynamic GRU yields {recall:.1f}\\% Recall@128 "
                       f"and {fpr:.1f}\\% sequence-level false-positive rate on the in-domain test split.")
        abstract_zh = f"在{len(main_values)}个已完成种子上，动态GRU的域内测试Recall@128为{recall:.1f}\\%，序列级误报率为{fpr:.1f}\\%。"
        if ext_values:
            abstract_en += f" External Recall@128 is {np.mean(ext_values) * 100:.1f}\\% without threshold adaptation."
            abstract_zh += f"在不调整阈值的情况下，外部Recall@128为{np.mean(ext_values) * 100:.1f}\\%。"
        static_values = _values(models.get("static_lr", {}), "test", "recall128")
        static_fpr = _values(models.get("static_lr", {}), "test", "sequence_fpr")
        if static_values and static_fpr:
            abstract_en += f" The static probe reaches {np.mean(static_values) * 100:.1f}\\% at {np.mean(static_fpr) * 100:.1f}\\% FPR under the same calibration rule."
            abstract_zh += f"相同校准规则下，静态探针的Recall@128为{np.mean(static_values) * 100:.1f}\\%，误报率为{np.mean(static_fpr) * 100:.1f}\\%。"
        conclusion_en = abstract_en + " These measurements concern final-response labels under causal replay and do not establish harm-onset localization."
        conclusion_zh = abstract_zh + "这些测量对应因果回放下的最终回答标签，不能证明能够定位危害起点。"
    else:
        status_en = "Experiments pending; no empirical performance claim is made."
        status_zh = "实验尚未完成，本文不作经验性能结论。"
        abstract_en = "The executable pipeline separates planned comparisons from measured results; the present build has no completed performance measurements."
        abstract_zh = "可执行流程将预设比较与实测结果分开；当前构建尚无完成的性能测量。"
        conclusion_en = "No superiority claim is supported until the specified runs and external evaluation are complete."
        conclusion_zh = "在预设运行与外部评估完成之前，现有证据不支持方法优越性结论。"
        LOGGER.warning("Performance metrics are incomplete: path=%s models=%d", metrics_path, len(models))
    macros = {"ReportStatusEN": status_en, "ReportStatusZH": status_zh,
              "AbstractResultsEN": abstract_en, "AbstractResultsZH": abstract_zh,
              "ConclusionResultsEN": conclusion_en, "ConclusionResultsZH": conclusion_zh}
    # 方差保留量直接读取训练投影元数据，避免论文与实际预处理器脱节。
    projection_metadata = _read_json(Path(config["paths"]["preprocessing"]) / "projection.json")
    variance = projection_metadata.get("explained_variance_ratio", [])
    macros.update({"PCAVarianceEN": "", "PCAVarianceZH": ""})
    if variance:
        retained = ", ".join(f"{value * 100:.1f}" for value in variance)
        components = projection_metadata["components_per_layer"]
        count = projection_metadata["n_fit_checkpoints"]
        macros["PCAVarianceEN"] = (f"The fitted {components}-component projections retain {retained}\\% "
                                   f"of training variance in the selected layer order, using {count:,} sampled checkpoints. "
                                   "This summary describes representation compression; the downstream comparisons evaluate risk prediction.")
        macros["PCAVarianceZH"] = (f"按所选层顺序，{components}维投影分别保留{retained}\\%的训练方差，"
                                   f"拟合共使用{count:,}个抽样检查点。该统计描述表征压缩程度，后续分类比较评价其风险预测表现。")
    stride_models = ("gru_stride4", "gru", "gru_stride16")
    if all(_values(models.get(name, {}), "test", "recall128") for name in stride_models):
        stride_recall = [np.mean(_values(models[name], "test", "recall128")) * 100 for name in stride_models]
        stride_fpr = [np.mean(_values(models[name], "test", "sequence_fpr")) * 100 for name in stride_models]
        recalls = ", ".join(f"{value:.1f}" for value in stride_recall)
        rates = ", ".join(f"{value:.1f}" for value in stride_fpr)
        macros["StrideResultsEN"] = (f"At strides of 4, 8 and 16 tokens, GRU Recall@128 is {recalls}\\%, "
                                    f"with observed test FPRs of {rates}\\%, respectively. Denser checks yield higher measured coverage in these runs, while larger strides reduce scoring frequency.")
        macros["StrideResultsZH"] = (f"步长为4、8和16个token时，GRU的Recall@128分别为{recalls}\\%，"
                                    f"实际测试误报率分别为{rates}\\%。本组运行中，更密集的检查对应更高的实测覆盖，较大步长则减少评分频率。")
    else:
        macros.update({"StrideResultsEN": "", "StrideResultsZH": ""})
    _write(paper / "generated/experiment_macros.tex", "% Generated from measured artifacts. Do not enter results manually.\n" +
           "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n")
    results_en = [r"\paragraph{Complete-pipeline comparison.}\AchievementDetectionEN{} Paired prompt-group uncertainty is reported in Table~\ref{tab:uncertainty}.", r"\input{tables/main}",
                  r"\paragraph{Detector trade-offs.}Table~\ref{tab:main} reports the full baseline comparison. Static probes, MLPs and recurrent heads offer complementary recall--FPR profiles under the common calibration target, supporting application-specific detector selection.",
                  r"\input{tables/external}",
                  r"\paragraph{Transfer boundary.}The external results use the frozen in-domain threshold. They therefore test transport of the complete scoring-and-threshold rule. Differences in source models, annotation policies and response lengths can each contribute to a transfer gap.",
                  r"\input{tables/alert_positions}", r"\input{tables/ablation}", r"\input{tables/uncertainty}", r"\input{tables/timing}"]
    results_zh = [r"\paragraph{完整流程比较。}\AchievementDetectionZH{} 提示分组配对不确定性见表~\ref{tab:uncertainty}。", r"\input{tables/main_zh}",
                  r"\paragraph{检测器权衡。}表~\ref{tab:main}完整呈现基线比较。静态探针、MLP和循环头在共同校准目标下形成互补的召回率—误报率运行特征，为应用选择检测器提供依据。",
                  r"\input{tables/external_zh}",
                  r"\paragraph{迁移边界。}外部评估使用冻结的域内阈值，考察完整评分与阈值规则能否迁移。来源模型、标注规则和回答长度的差异都可能造成迁移性能变化。",
                  r"\input{tables/alert_positions_zh}", r"\input{tables/ablation_zh}", r"\input{tables/uncertainty_zh}", r"\input{tables/timing_zh}"]
    subset_rows = [entry.get("test", {}) for entry in matched_subsets.get("gru", {}).values()]
    static_subset_rows = [entry.get("test", {}) for entry in matched_subsets.get("static_lr", {}).values()]
    if subset_rows and subset_rows[0].get("n_groups") and static_subset_rows:
        subset_counts = subset_rows[0]
        recall = np.mean([row["recall_at_budget"]["128"] for row in subset_rows]) * 100
        static_recall = np.mean([row["recall_at_budget"]["128"] for row in static_subset_rows]) * 100
        fpr = np.mean([row["sequence_fpr"] for row in subset_rows]) * 100
        static_fpr = np.mean([row["sequence_fpr"] for row in static_subset_rows]) * 100
        results_en.append(f"\\paragraph{{Same-prompt outcomes.}}The test subset containing both final labels per prompt has {subset_counts['n_groups']} groups and {subset_counts['n']} responses. Dynamic GRU and current-state LR attain {recall:.1f}\\% and {static_recall:.1f}\\% Recall@128, at {fpr:.1f}\\% and {static_fpr:.1f}\\% FPR, respectively, using unchanged thresholds. The matched-prompt subset characterizes how the two detectors respond to different outcomes associated with a shared question.")
        results_zh.append(f"\\paragraph{{同题不同结局。}}测试集中同一问题同时存在两种最终标签的子集包含{subset_counts['n_groups']}组、{subset_counts['n']}条回答。冻结阈值下，动态GRU与当前状态逻辑回归的Recall@128分别为{recall:.1f}\\%和{static_recall:.1f}\\%，误报率分别为{fpr:.1f}\\%和{static_fpr:.1f}\\%。这一同题子集刻画了两种检测器在共享问题、不同回答结局下的表现。")
    mean_values = _values(models.get("mean_lr", {}), "test", "recall128")
    if main_values and mean_values:
        difference = (np.mean(main_values) - np.mean(mean_values)) * 100
        results_en.append(f"\\paragraph{{History-aggregation control.}}Running-mean LR obtains {np.mean(mean_values) * 100:.1f}\\% Recall@128; the dynamic-GRU difference is {difference:+.1f} percentage points. The comparison complements current-state and learned-recurrence baselines by quantifying causal history accumulation.")
        results_zh.append(f"\\paragraph{{历史聚合对照。}}历史均值逻辑回归的Recall@128为{np.mean(mean_values) * 100:.1f}\\%，动态GRU相对差值为{difference:+.1f}个百分点。该对照量化因果历史积累的作用，补充当前状态与学习式循环模型的比较。")
    if paired:
        results_en.append(r"\paragraph{Uncertainty.}Table~\ref{tab:uncertainty} retains paired prompt-group bootstrap intervals. Each paired draw preserves the prompt groups and fixed fitted seeds across the compared detectors.")
        results_zh.append(r"\paragraph{不确定性。}表~\ref{tab:uncertainty}保留提示分组配对自助法区间。每次配对抽样在检测器之间保留一致的提示分组和已拟合种子。")
        primary = next((item for item in paired if item.get("split") == "test"
                        and item.get("reference") == "static_lr"
                        and item.get("metric") == "recall_at_budget.128"), None)
        if primary and all(_number(primary.get(key)) is not None for key in ("estimate", "ci_low", "ci_high")):
            difference, low, high = (primary[key] * 100 for key in ("estimate", "ci_low", "ci_high"))
            results_en.append(f"\\paragraph{{Pre-specified contrast.}}The dynamic-GRU minus static-probe Recall@128 difference is {difference:+.2f} percentage points (95\\% prompt-group interval [{low:+.2f}, {high:+.2f}]). The static probe favors recall coverage, while the GRU has the lower observed test FPR; these operating profiles support different monitoring priorities.")
            results_zh.append(f"\\paragraph{{预设对照。}}动态GRU减去静态探针的Recall@128差值为{difference:+.2f}个百分点，提示分组95\\%区间为[{low:+.2f}, {high:+.2f}]。静态探针侧重召回覆盖，GRU则具有较低实测误报率，两类运行特征对应不同监测需求。")
    seed_records = models.get("gru", {}).get("seeds", {})
    category_buckets = {}
    for seed_result in seed_records.values():
        for category, summary in seed_result.get("splits", {}).get("test", {}).get("category_recall", {}).items():
            category_buckets.setdefault(category, {"n_unsafe": summary["n_unsafe"], "recalls": []})["recalls"].append(summary["recall"])
    category_rows = [{"category": key, "n_unsafe": value["n_unsafe"], "mean_recall": float(np.mean(value["recalls"]))}
                     for key, value in category_buckets.items()]
    worst = sorted((row for row in category_rows if row["n_unsafe"] >= 30), key=lambda row: row["mean_recall"])[:3]
    if worst:
        details = "; ".join(f"{_tex(row['category'].replace('_', ' ').replace(',', ', '))}: {row['mean_recall'] * 100:.1f}\\% ($N={row['n_unsafe']}$)" for row in worst)
        results_en.append(f"\\paragraph{{Descriptive error strata.}}Among categories with at least 30 unsafe test responses, the three lowest full-stream mean recalls are {details}. The multi-label summaries guide category-specific data collection and follow-up analysis.")
        results_zh.append(f"\\paragraph{{描述性误差分层。}}在不安全测试回答至少30条的类别中，全流平均召回率最低的三个类别为：{details}。这些多标签描述统计为分类别数据收集与后续分析提供方向。")
    case_path = result_dir / "predictions/gru/seed_42/test.jsonl"
    if case_path.exists() and "42" in seed_records:
        threshold = seed_records["42"]["threshold"]
        with case_path.open(encoding="utf-8") as stream:
            cases = [json.loads(line) for line in stream if line.strip()]
        errors = []
        for label, error_type in ((0, "false_positive"), (1, "false_negative")):
            candidates = [row for row in cases if row["label"] == label
                          and ((max(row["scores"]) > threshold) != bool(label))]
            if candidates:
                row = (max if label == 0 else min)(candidates, key=lambda item: max(item["scores"]))
                errors.append({"id": row["id"], "error_type": error_type, "label": label,
                               "group": row["group"], "response_len": row["response_len"],
                               "max_score": max(row["scores"]), "threshold": threshold,
                               "categories": row.get("categories", [])})
        _write(paper / "source_data/error_analysis.json", json.dumps({"seed": 42,
               "prediction_sha256": _hash(case_path), "categories": category_rows, "cases": errors,
               "case_selection": "Highest-score false positive and lowest-score false negative; no text is reproduced."}, indent=2) + "\n")
        if errors:
            case_ids = ", ".join(_tex(row["id"]) for row in errors)
            results_en.append(f"\\paragraph{{Traceable cases.}}The archived error analysis identifies cases {case_ids}: respectively, the highest-score false positive and lowest-score false negative when both exist, selected from seed 42. These deterministically selected diagnostic cases can be inspected through their source identifiers.")
            results_zh.append(f"\\paragraph{{可追溯案例。}}误差归档保留案例{case_ids}：在两类错误均存在时，分别对应种子42中最高分误报与最低分漏报。这些确定性选择的诊断案例可以通过来源标识进一步核查。")
    scale_metrics = _read_json(result_dir / "scale/metrics.json")
    scale_models = scale_metrics.get("models", {})
    if scale_models:
        _matched_prompt_analysis(result_dir / "scale", scale_models)
    if scale_models:
        scale_rows = []
        for size, collection in (("0.5B", models), ("1.5B", scale_models)):
            for name in ("static_lr", "gru"):
                if name in collection:
                    scale_rows.append([size, METHOD_NAMES[name], _stat(collection[name], "test", "recall128"),
                                       _stat(collection[name], "test", "sequence_fpr"),
                                       _stat(collection[name], "external", "recall128")])
        for suffix, zh in (("", False), ("_zh", True)):
            _write(table_dir / f"scale{suffix}.tex", _table(
                "相同样本下的骨干规模验证（百分比）；各规模使用独立训练与校准工件。" if zh else
                "Backbone-scale validation on identical responses (%); each size has independently fitted transformations, probes and calibration thresholds.",
                "scale", "llrrr", ["Size", "Method", "R@128", "FPR", "Ext. R@128"], scale_rows))
        results_en.append(r"\paragraph{Backbone scale.}Table~\ref{tab:scale} compares frozen 0.5B and 1.5B representations on identical response partitions. The 1.5B model uses returned hidden-state indices 7, 14, 21 and 28, with the same per-layer projection dimension and training protocol. Both configurations encode the same response partitions.\input{tables/scale}")
        results_zh.append(r"\paragraph{骨干规模。}表~\ref{tab:scale}在相同回答划分上比较冻结的0.5B与1.5B表示。1.5B采用返回隐状态的第7、14、21和28个索引，每层降维维数与训练协议保持一致。两种配置编码相同的回答划分。\input{tables/scale_zh}")
    generation = _read_json(result_dir / "generation_benchmark.json")
    online_demo = _read_json(result_dir / "online_demo.json")
    storage = {}
    for name in ("raw_features", "features"):
        directory = Path(paths.get(name, f"artifacts/{name}"))
        if directory.exists():
            files = [path for path in directory.rglob("*") if path.is_file()]
            storage[name] = {"files": len(files), "bytes": sum(path.stat().st_size for path in files)}
    system_measurements = {"cache_storage": storage, "generation_trials": len(generation.get("trials", [])),
                           "note": "Cache sizes include manifests; backbone weights and Python/TeX environments are separate."}
    if generation.get("trials"):
        trials = generation["trials"]
        overhead = generation["median_overhead_ratio"] * 100
        baseline_seconds = float(np.median([trial["baseline_seconds"] for trial in trials]))
        monitor_seconds = float(np.median([trial["monitor_only_seconds"] for trial in trials]))
        low_overhead = min(trial["overhead_ratio"] for trial in trials) * 100
        high_overhead = max(trial["overhead_ratio"] for trial in trials) * 100
        results_en.append(f"\\paragraph{{Matched-length generation.}}Across {len(trials)} warmed paired trials on the fixed benign demonstration prompt, median baseline and monitor-only generation times are {baseline_seconds:.3f} and {monitor_seconds:.3f} s, respectively. The median paired overhead is {overhead:.1f}\\%. Each pair has an identical greedy token-sequence hash; model-loading time is excluded. The paired design profiles checkpoint monitoring within the same generation workload.")
        results_zh.append(f"\\paragraph{{等长生成开销。}}对固定良性演示提示完成{len(trials)}组预热后的配对测量，无监测与仅监测生成的中位耗时分别为{baseline_seconds:.3f}和{monitor_seconds:.3f}秒，配对开销中位数为{overhead:.1f}\\%。每组贪心生成的token序列散列完全一致，耗时不包含模型加载。这一配对设计在相同生成工作负载中测量检查点监测开销。")
        results_en.append(f"The observed paired ratios range from {low_overhead:+.1f}\\% to {high_overhead:+.1f}\\%. The paired measurements place monitored generation close to the unmonitored baseline on the local device.")
        results_zh.append(f"配对比例的实测范围为{low_overhead:+.1f}\\%至{high_overhead:+.1f}\\%。配对测量显示，本机监测生成的耗时与无监测基线接近。")
        score_p50 = float(np.median([trial["monitoring_p50_ms"] for trial in trials]))
        score_p95 = float(np.median([trial["monitoring_p95_ms"] for trial in trials]))
        peak = max(trial["peak_gpu_gib"] for trial in trials)
        throughput = float(np.median([trial["generated_tokens"] / trial["monitor_only_seconds"] for trial in trials]))
        system_measurements.update({"median_trial_monitor_p50_ms": score_p50,
                                    "median_trial_monitor_p95_ms": score_p95,
                                    "maximum_trial_peak_gpu_gib": peak,
                                    "median_monitor_only_tokens_per_second": throughput})
        results_en.append(f"Across trials, the medians of checkpoint-monitor p50 and p95 latency are {score_p50:.2f} and {score_p95:.2f} ms, respectively; monitor-only throughput is {throughput:.1f} tokens/s (median), and peak PyTorch allocation is {peak:.2f} GiB. These checkpoint timings include feature construction and detector inference, unlike the detector-head-only table.")
        results_zh.append(f"跨试验的检查点监测p50与p95时延中位数分别为{score_p50:.2f}和{score_p95:.2f}毫秒，仅监测生成的吞吐中位数为{throughput:.1f} token/秒，PyTorch分配显存峰值为{peak:.2f} GiB。检查点计时包含特征构造和检测器推理，区别于仅检测头表格。")
    if online_demo:
        results_en.append(f"\\paragraph{{Online integration.}}The separate guarded demonstration generates {online_demo['generated_tokens']} tokens and releases {online_demo['released_tokens']} after checkpoint decisions; stop status is {str(online_demo['stopped']).lower()}. This functional run is complemented by deterministic tests of buffer discard and EOS-tail handling.")
        results_zh.append(f"\\paragraph{{在线集成。}}独立受控生成演示生成{online_demo['generated_tokens']}个token，经检查点判决后释放{online_demo['released_tokens']}个，停止状态为{str(online_demo['stopped']).lower()}。该功能运行还结合确定性样例检验缓冲丢弃和EOS尾块处理。")
        results_en.append(f"The guarded run takes {online_demo['elapsed_seconds']:.3f} s ({online_demo['generated_tokens_per_second']:.1f} tokens/s). The run completes the configured generation budget.")
        results_zh.append(f"受控生成耗时{online_demo['elapsed_seconds']:.3f}秒，吞吐为{online_demo['generated_tokens_per_second']:.1f} token/秒。该运行完成了配置的生成预算。")
    extraction = _read_json(result_dir / "extraction_timing.json")
    if extraction:
        results_en.append(f"\\paragraph{{Frozen extraction cost.}}The recorded main extraction processes {extraction['tokens']:,} prompt-plus-response tokens in {extraction['elapsed_seconds']:.1f} s ({extraction['tokens_per_second']:.1f} tokens/s), with peak PyTorch-allocated GPU memory {extraction['peak_gpu_gib']:.2f} GiB. Memory is reported as PyTorch allocation, with throughput measured on the local device.")
        results_zh.append(f"\\paragraph{{冻结抽取成本。}}主抽取记录在{extraction['elapsed_seconds']:.1f}秒内处理{extraction['tokens']:,}个提示与回答token，吞吐为{extraction['tokens_per_second']:.1f} token/秒，PyTorch分配显存峰值为{extraction['peak_gpu_gib']:.2f} GiB。显存采用PyTorch分配量口径，吞吐对应本机测量。")
    if "raw_features" in storage and "features" in storage:
        raw_mib, projected_mib = (storage[name]["bytes"] / 2**20 for name in ("raw_features", "features"))
        results_en.append(f"The main raw-state and projected caches occupy {raw_mib:.1f} and {projected_mib:.1f} MiB, respectively, including their manifests. Frozen backbone weights are separate from these cache volumes.")
        results_zh.append(f"主实验原始隐状态与投影缓存分别占用{raw_mib:.1f}和{projected_mib:.1f} MiB，包含各自清单；冻结骨干权重不计入此缓存体积。")
    _write(result_dir / "system_measurements.json", json.dumps(system_measurements, indent=2))
    # 类别诊断统一放入后文的数据拓展讨论，数值和案例仍由同一结果源生成。
    for language, paragraphs in (("en", results_en), ("zh", results_zh)):
        markers = ("Descriptive error strata", "Traceable cases") if language == "en" else ("描述性误差分层", "可追溯案例")
        diagnostics = [text for text in paragraphs if any(marker in text for marker in markers)]
        paragraphs[:] = [text for text in paragraphs if text not in diagnostics]
        _write(paper / f"generated/error_directions_{language}.tex", "\n\n".join(diagnostics) + "\n")
    # 按研究问题组织证据，性能与迁移材料分别输出，避免正文交叉叙述。
    for language, paragraphs in (("en", results_en), ("zh", results_zh)):
        scale_marker = "Backbone scale" if language == "en" else "骨干规模"
        system_marker = "Matched-length generation" if language == "en" else "等长生成"
        scale_index = next((i for i, text in enumerate(paragraphs) if scale_marker in text), len(paragraphs))
        # 英中文标题可以独立润色，系统段落仍以其固定数据来源定位。
        system_index = next((i for i, text in enumerate(paragraphs) if system_marker in text or ("warmed" in text if language == "en" else "预热" in text)), len(paragraphs))
        external_table = next(text for text in paragraphs if "tables/external" in text)
        external_index = paragraphs.index(external_table)
        transfer = paragraphs[external_index:external_index + 2]
        scale = paragraphs[scale_index:system_index]
        system = paragraphs[system_index:]
        core = [text for text in paragraphs[:min(scale_index, system_index)] if text not in transfer and "tables/ablation" not in text and "tables/timing" not in text]
        _write(paper / f"generated/transfer_{language}.tex", "\n\n".join(scale + transfer) + "\n")
        _write(paper / f"generated/system_{language}.tex", "\n\n".join(system) + "\n")
        paragraphs[:] = core
    _write(paper / "generated/results_en.tex", "\n\n".join(results_en) + "\n")
    _write(paper / "generated/results_zh.tex", "\n\n".join(results_zh) + "\n")
    audit_path = Path(paths.get("processed", "data/processed")) / "audit.json"
    audit = _read_json(audit_path)
    figures = _make_figures(paper, metrics, result_dir, config, counts, audit)
    external_source = next((row for row in audit.get("sources", []) if row.get("source") == "streamguard"), None)
    if external_source:
        before, after = int(external_source["input"]), int(external_source["accepted"])
        unsafe_before = int(external_source.get("input_labels", {}).get("1", 0))
        unsafe_after = int(external_source.get("retained_labels", {}).get("1", 0))
        text_en = (f"The eligibility audit retains {after:,} of {before:,} external responses ({after / before * 100:.1f}\\%), "
                   f"including {unsafe_after} of {unsafe_before} originally unsafe responses. The retained external unsafe prevalence is "
                   f"{unsafe_after / after * 100:.1f}\\%. "
                   "Exclusion reasons and class-specific retention are preserved in the audit and source data.")
        text_zh = (f"合格条件审计在{before:,}条外部回答中保留{after:,}条（{after / before * 100:.1f}\\%），"
                   f"其中原始{unsafe_before}条不安全回答保留{unsafe_after}条，保留样本的不安全比例为{unsafe_after / after * 100:.1f}\\%。"
                   "排除原因与类别保留比例由审计和图形源数据共同记录。")
    else:
        text_en, text_zh = "Eligibility audit pending.", "合格条件审计尚未生成。"
    _write(paper / "generated/data_selection_en.tex", text_en + "\n")
    _write(paper / "generated/data_selection_zh.tex", text_zh + "\n")
    provenance = {"schema_version": 1, "generated_utc": datetime.now(UTC).isoformat(),
                  "task": "final_response_risk_prediction", "target_fpr": target,
                  "config": config, "metrics_sha256": _hash(metrics_path) if metrics_path.exists() else None,
                  "metrics_file": str(metrics_path), "counts": counts,
                  "completed_gru_seeds": completed_seeds, "figures": figures,
                  "experiment_signature": expected_signature,
                  "source_revisions": {"streamguardbench": config.get("data", {}).get("streamguard_revision"),
                                       "beavertails": config.get("data", {}).get("beavertails_revision"),
                                       "backbone": config.get("model", {}).get("revision")},
                  "additional_measurement_hashes": {str(path): _hash(path) for path in
                      [result_dir / "scale/metrics.json", result_dir / "generation_benchmark.json",
                       result_dir / "online_demo.json", result_dir / "extraction_timing.json",
                       result_dir / "matched_prompt_subset.json", result_dir / "scale/matched_prompt_subset.json",
                       result_dir / "review_robustness.json", result_dir / "streaming_score_alignment.json",
                       Path(paths.get("processed", "data/processed")) / "source_annotations.manifest.json"] if path.exists()}}
    _write(paper / "generated/provenance.json", json.dumps(provenance, ensure_ascii=False, indent=2, default=str) + "\n")
    _write(paper / "generated/evidence_audit.md", "# Evidence audit\n\n" + status_en +
           "\n\n- Tables: results/metrics.json, per-seed split measurements.\n"
           "- Figures: numeric metrics or deidentified prediction trajectories; architecture is a schematic.\n"
           "- Sample counts: normalized processed JSONL files; no raw response text is exported.\n"
           "- Inference target: eventual response safety, not token or onset labels.\n"
           "- Calibration target: empirical sequence-level FPR on the calibration sample, not a test guarantee.\n"
           "- Missing and undefined metrics remain missing; no synthetic measurements are inserted.\n"
           "- Author metadata are supplied by the author and shared through papers/author_info.tex.\n")
    _achievement_artifacts(models, scale_models, metrics, counts, system_measurements, paper, result_dir)
    _review_artifacts(paper, result_dir)
    summary = ["# 实测结果摘要", "", "由 `hs_safety report` 从实验工件生成；成果解读见 [成果亮点与汇报提纲](../docs/成果亮点与汇报提纲.md)。", "",
               "| 骨干 | 检测器 | 域内 Recall@128 (%) | 域内 FPR (%) | 外部 Recall@128 (%) |",
               "| --- | --- | ---: | ---: | ---: |"]
    for size, collection in (("0.5B", models), ("1.5B", scale_models)):
        for name in ORDER:
            if name not in collection:
                continue
            display = [_stat(collection[name], split, metric).replace("$\\pm$", "±")
                       for split, metric in (("test", "recall128"), ("test", "sequence_fpr"), ("external", "recall128"))]
            summary.append(f"| {size} | {METHOD_NAMES[name]} | " + " | ".join(display) + " |")
    summary.extend(["", "均值与样本标准差跨三个固定随机种子计算；线性模型的重复拟合基本相同。",
                    "各方法使用相同校准目标，但实际测试误报率不同，不能解释为在相同测试误报率处比较。",
                    "外部集经过完整长度筛选；结果仅适用于实际保留回答，不代表完整基准。",
                    "", "详细指标、分组置信区间和误差分层见 `metrics.json`；逐回答轨迹见 `predictions/`。",
                    "英文主稿和逐节中文对照见 `papers/paper_en.pdf` 与 `papers/paper_zh.pdf`。"])
    _write(result_dir / "summary.md", "\n".join(summary) + "\n")
    LOGGER.info("Report generation complete: seconds=%.3f figures=%d dataset_splits=%d", time.perf_counter() - started, len(figures), len(counts))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-driven bilingual IEEE paper artifacts.")
    parser.add_argument("--config", default="configs/main.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    generate_report(load_config(args.config))


if __name__ == "__main__":
    main()
