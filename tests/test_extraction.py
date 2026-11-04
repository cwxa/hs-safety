"""微型真实模型的消费位置、后缀隔离与 KV 增量对齐。"""

import numpy as np
import torch

from hs_safety.extraction import hidden_states_at_tokens


def test_hidden_state_positions_match_consumed_response_tokens(tiny_causal_model):
    prompt, response, positions = [1, 7, 8], [11, 12, 13, 14], np.asarray([1, 4])
    actual = hidden_states_at_tokens(tiny_causal_model, prompt, response, [2, 4], positions)
    with torch.inference_mode():
        output = tiny_causal_model.model(
            input_ids=torch.tensor([prompt + response]), use_cache=False, output_hidden_states=True)
    expected = torch.stack([output.hidden_states[layer][0, [3, 6]] for layer in [2, 4]], dim=1)
    np.testing.assert_array_equal(actual, expected.half().numpy())


def test_future_response_tokens_do_not_change_earlier_hidden_states(tiny_causal_model):
    prompt, first, second = [1, 7], [11, 12, 13, 14, 15], [11, 12, 13, 81, 82]
    positions = np.asarray([1, 2, 3])
    original = hidden_states_at_tokens(tiny_causal_model, prompt, first, [2, 4], positions)
    changed = hidden_states_at_tokens(tiny_causal_model, prompt, second, [2, 4], positions)
    prefix = hidden_states_at_tokens(tiny_causal_model, prompt, first[:3], [2, 4], positions)
    np.testing.assert_allclose(original, changed, atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(original, prefix, atol=2e-3, rtol=2e-3)


def test_kv_incremental_states_align_with_full_causal_forward(tiny_causal_model):
    prompt, response, layers = [1, 7, 8], [11, 12, 13, 14, 15], [2, 4]
    full = hidden_states_at_tokens(tiny_causal_model, prompt, response, layers, np.arange(1, 6))
    incremental = []
    with torch.inference_mode():
        output = tiny_causal_model.model(input_ids=torch.tensor([prompt]), use_cache=True)
        cache = output.past_key_values
        for token in response:
            output = tiny_causal_model.model(
                input_ids=torch.tensor([[token]]), past_key_values=cache,
                use_cache=True, output_hidden_states=True)
            cache = output.past_key_values
            incremental.append(torch.stack([output.hidden_states[layer][0, -1] for layer in layers]))
    expected = torch.stack(incremental).half().numpy()
    np.testing.assert_allclose(full, expected, atol=2e-3, rtol=2e-3)
