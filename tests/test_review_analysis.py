"""补充排序分析不得使用预算后的分数；小样本区间保留零事件不确定性。"""
import numpy as np

from hs_safety.review_analysis import budget_scores, strict_prefix_hits, wilson_interval


def test_ranking_budget_excludes_future_scores_and_uses_true_short_endpoint():
    rows = [{"id": "long", "positions": [8, 128, 136], "scores": [.1, .3, .99], "response_len": 136},
            {"id": "short", "positions": [8, 12], "scores": [.2, .7], "response_len": 12}]
    np.testing.assert_allclose(budget_scores(rows, 128), [.3, .7])


def test_wilson_zero_events_does_not_claim_zero_population_error():
    low, high = wilson_interval(0, 54)
    assert low == 0
    assert 0.06 < high < 0.07
    low, high = wilson_interval(54, 54)
    assert 0.93 < low < 0.94
    assert high == 1
    assert wilson_interval(0, 0) == [None, None]


def test_strict_prefix_detection_excludes_answer_endpoint():
    rows = [{"positions": [8, 128], "scores": [.1, .99], "response_len": 128},
            {"positions": [8, 12], "scores": [.8, .7], "response_len": 12},
            {"positions": [1], "scores": [.9], "response_len": 1}]
    np.testing.assert_array_equal(strict_prefix_hits(rows, .5), [False, True, False])
