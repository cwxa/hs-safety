"""公开导出必须隔离研究数据与身份字段，不修改课程源文件。"""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.package_public import export_project, public_text, selected_file, verify_manifest


@pytest.mark.parametrize("name", [
    "data/processed/train.jsonl", "results/predictions/gru/seed_42/test.jsonl",
    "artifacts/models/gru.pt", "logs/train.log", "reviews/input/paper.pdf",
    ".env", "papers/qa/paper_en_01.png", "papers/source_data/exploratory_transforms.npz",
])
def test_private_artifacts_never_selected(name):
    assert not selected_file(Path(name))


@pytest.mark.parametrize("name", ["hs_safety/features.py", "results/metrics.json", "papers/paper_en.tex"])
def test_public_source_and_aggregate_evidence_selected(name):
    assert selected_file(Path(name))


def test_author_redaction_preserves_name_and_affiliation():
    identifier = str(10**9 + 7)
    original = (r"\newcommand{\AuthorNameZH}{陈纬昕}" + "\n" +
                r"\newcommand{\AuthorInstitutionZH}{南京邮电大学}" + "\n" +
                rf"\newcommand{{\AuthorStudentID}}{{{identifier}}}")
    transformed = public_text(Path("papers/author_info.tex"), original)
    assert identifier not in transformed
    assert "陈纬昕" in transformed and "南京邮电大学" in transformed
    assert identifier in original


def test_public_paper_has_no_student_line_or_subject():
    source = (r"\author{\IEEEauthorblockA{\AuthorInstitutionEN\\Student ID: \AuthorStudentID}}" +
              r"\hypersetup{pdfauthor={\AuthorNameEN},pdfsubject={Student ID: \AuthorStudentID}}")
    transformed = public_text(Path("papers/paper_en.tex"), source)
    assert "Student ID" not in transformed
    assert r"pdfauthor={\AuthorNameEN}" in transformed


def test_export_refuses_existing_target_without_changing_it(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        export_project(tmp_path / "source", target)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_manifest_detects_changes_after_review(tmp_path):
    source = tmp_path / "README.md"
    payload = b"reviewed"
    source.write_bytes(payload)
    manifest = {"files": {"README.md": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(tmp_path) == 1
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from manifest"):
        verify_manifest(tmp_path)


def test_manifest_rejects_paths_outside_public_tree(tmp_path):
    (tmp_path / "PUBLIC_MANIFEST.json").write_text(
        json.dumps({"files": {"../README.md": {"bytes": 0, "sha256": ""}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid public manifest entry"):
        verify_manifest(tmp_path)
