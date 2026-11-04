"""数据分组、长度过滤和 token 检查点的协议测试。"""

from copy import deepcopy

import numpy as np
import pytest

from hs_safety.data import (
    _tokenize_records,
    checkpoint_positions,
    grouped_split,
    normalize_text,
    prompt_group,
)


@pytest.mark.parametrize(
    ("length", "stride", "expected"),
    [(0, 4, []), (1, 4, [1]), (3, 4, [3]), (4, 4, [4]),
     (8, 4, [4, 8]), (10, 4, [4, 8, 10]), (10, 8, [8, 10]), (10, 16, [10])],
)
def test_checkpoints_include_the_final_consumed_token(length, stride, expected):
    np.testing.assert_array_equal(checkpoint_positions(length, stride), expected)


@pytest.mark.parametrize(("length", "stride"), [(-1, 4), (4, 0), (4, -1)])
def test_invalid_checkpoint_arguments_fail(length, stride):
    with pytest.raises(ValueError):
        checkpoint_positions(length, stride)


def test_normalized_prompts_have_the_same_group():
    assert normalize_text("  ＡＢＣ\nStraße  ") == "abc strasse"
    assert prompt_group("  ＡＢＣ\nStraße  ") == prompt_group("abc STRASSE")
    assert prompt_group("abc STRASSE") != prompt_group("abc strasse extra")


def test_grouped_split_is_deterministic_and_never_splits_a_prompt():
    records = [{"id": f"g{group}-{answer}", "group": f"g{group}"}
               for group in range(8) for answer in range(3)]
    first = grouped_split(deepcopy(records), 7, 4, 4, seed=19)
    second = grouped_split(deepcopy(records), 7, 4, 4, seed=19)
    assert first == second
    groups = {name: {row["group"] for row in rows} for name, rows in first.items()}
    for left, left_groups in groups.items():
        for right, right_groups in groups.items():
            if left != right:
                assert left_groups.isdisjoint(right_groups)
    for name, rows in first.items():
        assert all(row["split"] == name for row in rows)
        assert len(rows) % 3 == 0
        for group in groups[name]:
            assert sum(row["group"] == group for row in rows) == 3


class WordTokenizer:
    """用词数控制长度边界，不依赖真实 tokenizer 文件。"""

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2] + list(range(10, 10 + len(messages[-1]["content"].split())))

    def encode(self, text, add_special_tokens=False):
        return list(range(20, 20 + len(text.split())))


def test_overlength_response_is_rejected_instead_of_truncated():
    config = {"model": {"system_prompt": "Test system"},
              "data": {"max_prompt_tokens": 6, "max_response_tokens": 3}}
    rows = [
        {"prompt": "question one", "response": "short answer", "is_safe": True},
        {"prompt": "question two", "response": "one two three four", "is_safe": False},
        {"prompt": "QUESTION   ONE", "response": "SHORT ANSWER", "is_safe": True},
    ]
    accepted, audit = _tokenize_records(rows, "beavertails", WordTokenizer(), config, "30k_train")
    assert len(accepted) == 1
    assert accepted[0]["response"] == "short answer"
    assert accepted[0]["response_len"] == len(accepted[0]["response_ids"]) == 2
    assert accepted[0]["label"] == 0
    assert audit["rejected"]["response_over_limit_or_empty"] == 1
    assert audit["rejected"]["duplicate_qa"] == 1


def test_external_labels_are_not_silently_imputed():
    config = {"model": {"system_prompt": "Test system"},
              "data": {"max_prompt_tokens": 8, "max_response_tokens": 8}}
    rows = [
        {"prompt": "question one", "response": "answer one", "label": 1},
        {"prompt": "question two", "response": "answer two"},
        {"prompt": "question three", "response": "answer three", "label": "0"},
    ]
    accepted, audit = _tokenize_records(rows, "external", WordTokenizer(), config, "test")
    assert [row["label"] for row in accepted] == [1]
    assert audit["rejected"]["missing_or_invalid_label"] == 2


def test_original_annotation_archive_preserves_native_labels_without_editing_detector_records(tmp_path, monkeypatch):
    from hs_safety import data
    from hs_safety.io import hash_file, read_jsonl, save_json, write_jsonl

    config = {"model": {"system_prompt": "Test system"},
              "data": {"max_prompt_tokens": 8, "max_response_tokens": 8},
              "paths": {"raw": str(tmp_path / "raw"), "processed": str(tmp_path / "processed")}}
    beaver = [{"prompt": "a question", "response": "a reply", "is_safe": True,
               "category": {"risk": False}}]
    external = [{"prompt": "another question", "response": "another reply", "label": 1}]
    train, _ = _tokenize_records(beaver, "beavertails", WordTokenizer(), config, "30k_train")
    test, _ = _tokenize_records(external, "streamguard", WordTokenizer(), config, "test")
    processed = tmp_path / "processed"
    for split, rows in {"train": train, "dev": [], "calibration": [], "test": [], "external": test}.items():
        write_jsonl(processed / f"{split}.jsonl", rows)
    save_json(processed / "audit.json", {})
    before = hash_file(processed / "train.jsonl")
    monkeypatch.setattr(data, "_read_beaver", lambda path: beaver if path.name == "train.jsonl.gz" else [])
    monkeypatch.setattr(data, "_read_external", lambda path: external)
    data.export_source_annotations(config)
    annotations = read_jsonl(processed / "source_annotations.jsonl")
    assert annotations[0]["original_label"] is True
    assert annotations[0]["original_label_field"] == "is_safe"
    assert annotations[0]["original_categories"] == {"risk": False}
    assert annotations[1]["original_label"] == 1
    assert annotations[1]["original_label_field"] == "label"
    assert hash_file(processed / "train.jsonl") == before
