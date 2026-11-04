"""分片提交、恢复与损坏检测，不读取真实模型或数据。"""

import numpy as np
import pytest
import torch

from hs_safety import extraction
from hs_safety.io import hash_file, load_json, write_jsonl


def test_interrupted_extraction_resumes_only_uncommitted_responses(tmp_path, monkeypatch):
    config = {"paths": {"processed": str(tmp_path / "processed"),
                         "raw_features": str(tmp_path / "raw"), "results": str(tmp_path / "results")},
              "model": {"id": "local-test-backbone", "layers": [1, 2]},
              "features": {"base_stride": 4, "shard_size": 1}}
    records = [{"id": f"r{i}", "prompt_ids": [1, 2], "response_ids": [10 + i] * length,
                "response_len": length} for i, length in enumerate([6, 9, 10])]
    write_jsonl(tmp_path / "processed" / "train.jsonl", records)
    monkeypatch.setattr(extraction, "load_backbone", lambda current: torch.nn.Linear(1, 1))
    calls = []

    def interrupted(model, prompt, response, layers, positions, device):
        calls.append(response[0])
        if len(calls) == 2:
            raise RuntimeError("Simulated extraction interruption")
        return np.full((len(positions), len(layers), 3), response[0], dtype=np.float16)

    monkeypatch.setattr(extraction, "hidden_states_at_tokens", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        extraction.extract_features(config, splits=["train"])
    directory = tmp_path / "raw" / "train"
    partial = load_json(directory / "manifest.json")
    assert partial["complete"] is False
    assert len(partial["shards"]) == 1
    original_hash = hash_file(directory / "shard_00000.values.npy")
    resumed_calls = []

    def resumed(model, prompt, response, layers, positions, device):
        resumed_calls.append(response[0])
        return np.full((len(positions), len(layers), 3), response[0], dtype=np.float16)

    monkeypatch.setattr(extraction, "hidden_states_at_tokens", resumed)
    extraction.extract_features(config, splits=["train"])
    assert resumed_calls == [11, 12]
    assert hash_file(directory / "shard_00000.values.npy") == original_hash
    final = extraction.validate_raw_manifest(directory)
    assert final["complete"] is True
    assert [record_id for record_id, _, _ in extraction.iter_raw_sequences(config, "train")] == ["r0", "r1", "r2"]
    extraction.extract_features(config, splits=["train"])
    assert resumed_calls == [11, 12]
    with pytest.raises(ValueError, match="signature"):
        extraction.validate_raw_manifest(directory, expected_signature="different")
    (directory / "shard_00000.values.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt|Corrupt|Incomplete"):
        extraction.validate_raw_manifest(directory)
