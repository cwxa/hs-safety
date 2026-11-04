"""离线/在线共用特征的数值定义和无未来泄漏测试。"""

import numpy as np
import pytest

from hs_safety import features
from hs_safety.features import CausalFeatureBuilder, causal_dynamic_features, project_hidden
from hs_safety.io import hash_file, save_json, write_jsonl


def identity_projection(layers=2, dimensions=3):
    return {"components": np.repeat(np.eye(dimensions)[None], layers, axis=0).astype(np.float32),
            "mean": np.zeros((layers, dimensions), dtype=np.float32),
            "scale": np.ones((layers, dimensions), dtype=np.float32)}


def test_dynamic_features_have_explicit_first_step_and_drift_values():
    projected = np.asarray([[[3.0, 0.0]], [[0.0, 4.0]]], dtype=np.float32)
    actual = causal_dynamic_features(projected)
    np.testing.assert_allclose(actual[0], [3, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(actual[1], [0, 4, -3, 4, 1, 1])


def test_changing_future_states_cannot_change_earlier_features():
    rng = np.random.default_rng(42)
    original = rng.normal(size=(9, 4, 3)).astype(np.float32)
    changed = original.copy()
    changed[5:] = rng.normal(size=changed[5:].shape) * 100
    np.testing.assert_array_equal(
        causal_dynamic_features(original)[:5], causal_dynamic_features(changed)[:5])


@pytest.mark.parametrize(("dynamic", "last_only"), [(False, False), (True, False), (True, True)])
def test_online_feature_builder_matches_offline_and_resets(dynamic, last_only):
    projection = identity_projection()
    hidden = np.random.default_rng(7).normal(size=(5, 2, 3)).astype(np.float32)
    builder = CausalFeatureBuilder(projection, dynamic=dynamic, final_layer_only=last_only)
    online = np.stack([builder.update(value) for value in hidden])
    # 离线路径先保存原始 FP16，再保存投影 FP16；在线应经过相同的量化边界。
    cached = project_hidden(hidden.astype(np.float16), projection).astype(np.float16).astype(np.float32)
    expected = causal_dynamic_features(cached, dynamic, last_only)
    np.testing.assert_allclose(online, expected, atol=1e-6)
    builder.reset()
    np.testing.assert_allclose(builder.update(hidden[0]), expected[0], atol=1e-6)


def test_projection_uses_fixed_training_center_components_and_scale():
    projection = identity_projection(layers=1, dimensions=2)
    projection["mean"][0] = [1, 2]
    projection["scale"][0] = [2, 4]
    hidden = np.asarray([[[5, 10]], [[9, 18]]], dtype=np.float32)
    np.testing.assert_allclose(project_hidden(hidden, projection), [[[2, 2]], [[4, 4]]])


def test_invalid_or_nonfinite_dynamic_input_fails():
    with pytest.raises(ValueError, match="shape"):
        causal_dynamic_features(np.ones((3, 2)))
    with pytest.raises(ValueError, match="Nonfinite"):
        causal_dynamic_features(np.full((2, 1, 2), np.nan))


def test_pca_fits_only_train_and_limits_each_response_contribution(tmp_path, monkeypatch):
    config = {"paths": {"raw_features": str(tmp_path / "raw"), "preprocessing": str(tmp_path / "pre")},
              "features": {"pca_components": 2, "pca_fit_prefixes": 2, "pca_batch_size": 4},
              "model": {"layers": [1, 2]}, "training": {"num_threads": 1}}
    rng = np.random.default_rng(3)
    sequences = [rng.normal(size=(5, 2, 4)).astype(np.float32) + i for i in range(4)]
    requested = []

    def raw_sequences(current_config, split):
        requested.append(split)
        assert split == "train"
        for index, hidden in enumerate(sequences):
            yield str(index), hidden, np.arange(1, 6)

    monkeypatch.setattr(features, "validate_raw_manifest",
                        lambda *args, **kwargs: {"complete": True, "signature": "train-only-fixture"})
    monkeypatch.setattr(features, "iter_raw_sequences", raw_sequences)
    metadata = features.fit_projection(config)
    assert requested == ["train"]
    assert metadata["fit_split"] == "train"
    assert metadata["n_fit_checkpoints"] == 8
    expected_mean = np.concatenate([sequence[[0, 4]] for sequence in sequences]).mean(axis=0)
    np.testing.assert_allclose(features.load_projection(config)["mean"], expected_mean, atol=1e-6)


def projected_cache(tmp_path):
    config = {"paths": {"processed": str(tmp_path / "processed"), "features": str(tmp_path / "features"),
                         "preprocessing": str(tmp_path / "preprocessing")},
              "model": {"id": "tiny-backbone"}, "features": {"stride": 8, "base_stride": 4}}
    projection_file = features.projection_path(config)
    projection_file.parent.mkdir(parents=True)
    np.savez(projection_file, **identity_projection(layers=1, dimensions=2))
    record = {"id": "answer-1", "response_len": 10, "label": 1,
              "prompt_ids": [1], "response_ids": list(range(2, 12))}
    write_jsonl(tmp_path / "processed" / "test.jsonl", [record])
    directory = tmp_path / "features" / "test"
    directory.mkdir(parents=True)
    values = np.asarray([[[1, 0]], [[3, 0]], [[6, 0]]], dtype=np.float16)
    np.save(directory / "shard_00000.values.npy", values, allow_pickle=False)
    np.save(directory / "shard_00000.positions.npy", [4, 8, 10], allow_pickle=False)
    np.save(directory / "shard_00000.offsets.npy", [0, 3], allow_pickle=False)
    save_json(directory / "manifest.json", {
        "complete": True, "projection_hash": hash_file(projection_file), "base_stride": 4,
        "raw_signature": features.raw_signature(config, [record]),
        "shards": [{"prefix": "shard_00000", "ids": [record["id"]]}],
    })
    return config, directory, values


def test_stride_selection_uses_token_positions_and_recomputes_deltas(tmp_path):
    config, _, values = projected_cache(tmp_path)
    records, sequences, positions = features.load_feature_split(config, "test", stride=8)
    assert records[0]["id"] == "answer-1"
    np.testing.assert_array_equal(positions[0], [8, 10])
    expected = causal_dynamic_features(values[[1, 2]].astype(np.float32))
    np.testing.assert_allclose(sequences[0], expected)
    np.testing.assert_allclose(sequences[0][1, 2:4], [3, 0])


def test_cache_projection_identity_and_stride_are_enforced(tmp_path):
    config, _, _ = projected_cache(tmp_path)
    with pytest.raises(ValueError, match="multiple"):
        features.load_feature_split(config, "test", stride=6)
    np.savez(features.projection_path(config), altered=np.ones(2))
    with pytest.raises(ValueError, match="Projection"):
        features.load_feature_split(config, "test")


def test_token_change_cannot_reuse_a_cache_under_the_same_record_id(tmp_path):
    config, _, _ = projected_cache(tmp_path)
    from hs_safety.io import read_jsonl
    path = tmp_path / "processed/test.jsonl"
    records = read_jsonl(path)
    records[0]["response_ids"][0] = 95
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="token inputs"):
        features.load_feature_split(config, "test")


def test_label_change_must_match_the_preparation_audit(tmp_path):
    config, _, _ = projected_cache(tmp_path)
    from hs_safety.io import hash_object, read_jsonl
    path = tmp_path / "processed/test.jsonl"
    records = read_jsonl(path)
    save_json(tmp_path / "processed/audit.json", {"records_hash": {"test": hash_object(records)}})
    records[0]["label"] = 0
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="preparation audit"):
        features.load_feature_split(config, "test")
