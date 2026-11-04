"""回答级校准、预算指标与配对重采样测试。"""

from copy import deepcopy

import pytest

from hs_safety.metrics import (
    calibrate_threshold,
    matched_prompt_metrics,
    paired_group_bootstrap,
    response_metrics,
)


def row(identifier, label, scores, positions=None, group=None):
    positions = positions or list(range(1, len(scores) + 1))
    return {"id": identifier, "group": group or identifier, "label": label,
            "scores": scores, "positions": positions, "response_len": positions[-1]}


def test_same_prompt_subset_keeps_both_outcomes_and_all_responses_without_recalibration():
    rows = [row("a", 0, [0.9], group="mixed"), row("b", 0, [0.1], group="mixed"),
            row("c", 1, [0.8], group="mixed"), row("d", 1, [0.1], group="single")]
    metrics = matched_prompt_metrics(rows, threshold=0.7)
    assert metrics["n_groups"] == 1
    assert metrics["n"] == 3
    assert metrics["recall_at_budget"]["128"] == 1
    assert metrics["sequence_fpr"] == 0.5
    assert matched_prompt_metrics([rows[-1]], 0.7)["n"] == 0


def test_calibration_uses_safe_response_maxima_and_strict_ties():
    rows = [row("s1", 0, [0.1]), row("s2", 0, [0.2]),
            row("s3", 0, [0.1, 0.9]), row("s4", 0, [0.9, 0.1]),
            row("u", 1, [999.0])]
    calibration = calibrate_threshold(rows, target_fpr=0.25)
    assert calibration["threshold"] == pytest.approx(0.9)
    assert calibration["comparison"] == ">"
    assert calibration["n_safe"] == 4
    assert calibration["allowed_false_alarms"] == 1
    assert calibration["empirical_sequence_fpr"] == 0.0
    assert response_metrics(rows[:4], calibration["threshold"])["sequence_fpr"] == 0.0


def test_zero_fpr_and_missing_safe_calibration_boundaries():
    calibration = calibrate_threshold([row("s", 0, [0.1, 0.8])], 0)
    assert calibration["threshold"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="safe"):
        calibrate_threshold([row("u", 1, [0.5])])
    for target in (-0.1, 1.0):
        with pytest.raises(ValueError, match="FPR"):
            calibrate_threshold([row("s", 0, [0.5])], target)


def test_response_metrics_do_not_backfill_later_alarms_into_earlier_budgets():
    rows = [row("late", 1, [0.2, 0.9], [4, 10]),
            row("miss", 1, [0.1, 0.3], [4, 6]),
            row("safe", 0, [0.9, 0.1], [4, 9])]
    metrics = response_metrics(rows, threshold=0.5, budgets=[4, 8, 10])
    assert metrics["confusion"] == {"tp": 1, "fp": 1, "tn": 0, "fn": 1}
    assert metrics["recall"] == 0.5
    assert metrics["sequence_fpr"] == 1.0
    assert metrics["recall_at_budget"] == {"4": 0.0, "8": 0.0, "10": 0.5}
    assert metrics["detected_alert_tokens_mean"] == 10
    assert metrics["normalized_alert_position_mean"] == 1.0


def test_budget_before_first_checkpoint_has_no_alarm():
    metrics = response_metrics([row("answer", 1, [0.9, 0.2], [8, 12])], 0.5, [4, 8])
    assert metrics["recall_at_budget"] == {"4": 0.0, "8": 1.0}


@pytest.mark.parametrize("invalid", [
    {"scores": [0.1, float("nan")]}, {"positions": [2, 2]},
    {"positions": [0, 4]}, {"response_len": 8}, {"label": 2},
])
def test_malformed_trajectories_fail(invalid):
    value = row("answer", 1, [0.1, 0.8], [2, 4])
    value.update(invalid)
    with pytest.raises(ValueError):
        response_metrics([value], 0.5)


def test_empty_and_one_class_evaluation_are_explicit():
    assert response_metrics([], 0.5) == {"n": 0}
    metrics = response_metrics([row("safe", 0, [0.1])], 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["recall"] is None
    assert metrics["sequence_fpr"] == 0


def test_paired_bootstrap_keeps_groups_and_seeds_aligned():
    rows = [row("a", 0, [0.1], group="g1"), row("b", 1, [0.8], group="g1"),
            row("c", 0, [0.2], group="g2"), row("d", 1, [0.9], group="g2")]
    reference = {13: (rows, 0.5), 42: (deepcopy(rows), 0.5)}
    estimates = paired_group_bootstrap(reference, deepcopy(reference), [1], n_bootstrap=20, seed=6)
    assert estimates
    for estimate in estimates:
        assert estimate["estimate"] == 0.0
        assert estimate["ci_low"] == 0.0
        assert estimate["ci_high"] == 0.0
        assert estimate["n_groups"] == 2
        assert estimate["n_seeds"] == 2
    changed = deepcopy(reference)
    changed[13][0][0]["group"] = "different-group"
    with pytest.raises(ValueError, match="identical"):
        paired_group_bootstrap(reference, changed, [1], n_bootstrap=2)
