"""在线门控及 EOS 前残留块必须先检查后释放。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hs_safety import demo
from hs_safety.demo import ReleaseBuffer


def test_pending_tokens_are_invisible_until_the_decision():
    buffer = ReleaseBuffer()
    buffer.append(11)
    buffer.append(12)
    assert buffer.pending == [11, 12]
    assert buffer.released == []
    assert buffer.decide(False) == {"released": [11, 12], "discarded": []}
    assert buffer.pending == []
    assert buffer.released == [11, 12]


def test_alarm_discards_only_the_unreleased_block_and_is_terminal():
    buffer = ReleaseBuffer()
    buffer.append(11)
    buffer.decide(False)
    buffer.append(12)
    buffer.append(13)
    assert buffer.decide(True) == {"released": [], "discarded": [12, 13]}
    assert buffer.released == [11]
    assert buffer.pending == []
    assert buffer.stopped is True
    with pytest.raises(RuntimeError):
        buffer.append(14)
    with pytest.raises(RuntimeError):
        buffer.decide(False)


@pytest.mark.parametrize("alert", [False, True])
@pytest.mark.parametrize("benchmark", [False, True])
@pytest.mark.parametrize("monitor", [False, True])
def test_generation_checks_partial_buffer_before_eos_release(tmp_path, monkeypatch, alert, benchmark, monitor):
    """只生成三个内容 token；stride=8 时也必须在 EOS 处分支检查。"""
    consumed, detector_calls = [], []

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.generation_config = SimpleNamespace(eos_token_id=2)

        def forward(self, input_ids, **kwargs):
            tokens = input_ids[0].tolist()
            consumed.append(tokens)
            following = {7: 11, 11: 12, 12: 13, 13: 2}[tokens[-1]]
            logits = torch.full((1, 1, 20), -10.0)
            logits[0, 0, following] = 10.0
            hidden = tuple(torch.full((1, 1, 2), float(tokens[-1] + index)) for index in range(3))
            return SimpleNamespace(logits=logits, hidden_states=hidden, past_key_values=object())

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 7]

        def decode(self, ids, **kwargs):
            return " ".join(str(value) for value in ids)

    class FakeDetector:
        threshold = 0.5

        def __init__(self, *args, **kwargs):
            pass

        def update(self, feature):
            detector_calls.append(feature.copy())
            return {"alert": alert, "score": 0.9 if alert else 0.1}

    projection = {"components": np.repeat(np.eye(2)[None], 2, axis=0),
                  "mean": np.zeros((2, 2)), "scale": np.ones((2, 2))}
    monkeypatch.setattr(demo, "load_backbone", lambda config: FakeBackbone())
    monkeypatch.setattr(demo, "load_tokenizer", lambda config: FakeTokenizer())
    monkeypatch.setattr(demo, "load_projection", lambda config: projection)
    monkeypatch.setattr(demo, "StreamingDetector", FakeDetector)
    config = {"model": {"id": "test-backbone", "system_prompt": "Test", "layers": [1, 2]},
              "features": {"stride": 8}, "data": {"max_prompt_tokens": 8},
              "paths": {"results": str(tmp_path)},
              "demo": {"model_name": "gru", "seed": 42, "max_new_tokens": 10, "prompt": "Test prompt"}}
    result = demo.run_demo(config, benchmark=benchmark, monitor=monitor)
    should_stop = alert and monitor and not benchmark
    assert consumed == [[1, 7], [11], [12], [13]]
    assert len(detector_calls) == (1 if monitor else 0)
    assert result["generated_tokens"] == 3
    assert result["stopped"] is should_stop
    assert result["released_tokens"] == (0 if should_stop else 3)
    assert result["released_text"] == ("" if should_stop else "11 12 13")
    assert result["events"][0]["position"] == 3
    assert result["events"][0]["discarded_tokens"] == (3 if should_stop else 0)
