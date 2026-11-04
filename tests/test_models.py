"""回答等权和单向时序模型的行为测试。"""

import numpy as np
import pytest
import torch

from hs_safety.engine import _texts, _training_arrays
from hs_safety.models import SequenceGRU, cumulative_mean, prefix_indices


def test_each_response_has_unit_prefix_weight_and_includes_endpoint():
    records = [{"label": 0}, {"label": 1}]
    sequences = [np.zeros((2, 3), dtype=np.float32), np.ones((9, 3), dtype=np.float32)]
    values, labels, weights = _training_arrays(records, sequences, max_prefixes=4)
    assert values.shape == (6, 3)
    assert weights[labels == 0].sum() == pytest.approx(1.0)
    assert weights[labels == 1].sum() == pytest.approx(1.0)
    chosen = prefix_indices(9, 4)
    assert chosen[0] == 0 and chosen[-1] == 8


def test_cumulative_mean_uses_only_observed_checkpoints():
    original = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    changed = original.copy()
    changed[2] = [1000, 2000]
    np.testing.assert_array_equal(cumulative_mean(original)[:2], cumulative_mean(changed)[:2])
    np.testing.assert_allclose(cumulative_mean(original), [[1, 2], [2, 3], [3, 4]])


def test_gru_predictions_ignore_future_values_and_right_padding():
    torch.manual_seed(11)
    network = SequenceGRU(input_size=3, hidden_size=5).eval()
    original = torch.randn(1, 7, 3)
    changed = original.clone()
    changed[:, 4:] = torch.randn(1, 3, 3) * 100
    with torch.inference_mode():
        scores = network(original)
        changed_scores = network(changed)
        prefix_scores = network(original[:, :4])
    torch.testing.assert_close(scores[:, :4], changed_scores[:, :4])
    torch.testing.assert_close(scores[:, :4], prefix_scores)


def test_text_baseline_decodes_only_the_visible_response_ids():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(str(value) for value in ids)

    records = [{"prompt": "Question", "response_ids": [10, 20, 30, 40], "label": 1}]
    texts, labels, weights = _texts(records, [np.asarray([2, 4])], "text", Tokenizer())
    assert texts[0] == "Question\nAssistant: 10 20"
    assert texts[1] == "Question\nAssistant: 10 20 30 40"
    np.testing.assert_array_equal(labels, [1, 1])
    assert weights.sum() == pytest.approx(1.0)
