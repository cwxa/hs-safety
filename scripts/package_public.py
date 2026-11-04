"""白名单导出公开项目，隔离课程隐私、本地缓存和原始数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)
ROOT_FILES = {
    "README.md", "README.zh-CN.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff",
    "CHANGELOG.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
    "pyproject.toml", "requirements.lock", ".gitignore", ".gitattributes",
}
SOURCE_DIRS = {"hs_safety", "tests", "scripts", "configs", "docs", ".github", ".githooks"}
RESULT_FILES = {
    "metrics.json", "metrics.csv", "summary.md", "system_measurements.json",
    "review_robustness.json", "scale/metrics.json", "scale/metrics.csv",
}
PAPER_SUFFIXES = {".tex", ".bib", ".md", ".json", ".csv", ".svg", ".pdf"}
EXCLUDED_PARTS = {"__pycache__", "qa", ".pytest_cache", ".ruff_cache", ".git"}
MANIFEST_NAME = "PUBLIC_MANIFEST.json"


def selected_file(relative: Path) -> bool:
    """仅选取明确定义的源码和聚合证据，不根据当前Git追踪状态放宽范围。"""
    if EXCLUDED_PARTS.intersection(relative.parts):
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    if relative.parts[0] in SOURCE_DIRS:
        return relative.suffix in {".py", ".md", ".yaml", ".yml", ".ps1", ".json"} or (
            relative.parts[0] == ".githooks" and relative.name == "pre-commit")
    if relative.parts[0] == "results":
        return Path(*relative.parts[1:]).as_posix() in RESULT_FILES
    if relative.parts[0] == "papers":
        return relative.suffix in PAPER_SUFFIXES
    return False


def public_text(relative: Path, text: str) -> str:
    """保留作者署名，移除课程学号及PDF中的学号元数据。"""
    if relative.as_posix() == "papers/author_info.tex":
        text = re.sub(r"(\\newcommand\{\\AuthorStudentID\}\{)[^}]*\}", r"\g<1>}", text)
    if relative.as_posix() in {"papers/paper_en.tex", "papers/paper_zh.tex"}:
        text = text.replace(r"\\Student ID: \AuthorStudentID", "")
        text = text.replace(r"\\学号：\AuthorStudentID", "")
        text = text.replace(r",pdfsubject={Student ID: \AuthorStudentID}", "")
    return text


def export_project(root: Path, destination: Path) -> int:
    if destination.exists():
        raise ValueError(f"Export destination already exists: {destination}")
    # 禁止把导出写入任何源码目录，避免后续工作把隔离树误当成项目实现。
    if destination == root or root.is_relative_to(destination):
        raise ValueError("Export destination must not contain the research workspace")
    if destination.is_relative_to(root) and destination.relative_to(root).parts[0] not in {"deliverables", ".cache"}:
        raise ValueError("An in-workspace export must be under deliverables or .cache")
    selected = []
    for name in ROOT_FILES:
        selected.append(root / name)
    for directory in SOURCE_DIRS | {"papers"}:
        selected.extend(path for path in (root / directory).rglob("*") if path.is_file())
    selected.extend(root / "results" / name for name in RESULT_FILES)
    selected = sorted({path for path in selected if selected_file(path.relative_to(root))})
    # 源码可重建公开PDF；绝不复制含学号的课程最终稿。
    selected = [path for path in selected if path.relative_to(root).as_posix() not in {
        "papers/paper_en.pdf", "papers/paper_zh.pdf"}]
    missing = [str(path.relative_to(root)) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Public export inputs are missing: {missing}")
    destination.mkdir(parents=True)
    for path in selected:
        relative = path.relative_to(root)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".pdf"}:
            shutil.copyfile(path, output)
        else:
            # 与仓库.gitattributes保持一致，确保Windows导出与新克隆的哈希相同。
            output.write_text(public_text(relative, path.read_text(encoding="utf-8-sig")),
                              encoding="utf-8", newline="\n")
    LOGGER.info("Public export complete files=%d destination=%s", len(selected), destination)
    return len(selected)


def finalize_project(root: Path, destination: Path) -> dict:
    """检查实际公开树并记录摘要；生成构建中间文件不进入提交清单。"""
    import fitz

    identifier = re.search(r"\\newcommand\{\\AuthorStudentID\}\{([^}]*)\}",
                           (root / "papers/author_info.tex").read_text(encoding="utf-8"))
    private_id = identifier.group(1) if identifier else ""
    secrets = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|"
                         r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    personal_path = re.compile(r"[A-Za-z]:[\\/](?:Users[\\/]|!cwx[\\/])", re.IGNORECASE)
    files = {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(destination)
        if not selected_file(relative):
            continue
        if path.suffix == ".pdf":
            with fitz.open(path) as document:
                text = "\n".join(page.get_text() for page in document) + json.dumps(document.metadata)
        else:
            text = path.read_text(encoding="utf-8-sig")
        if secrets.search(text) or personal_path.search(text) or (private_id and private_id in text):
            raise ValueError(f"Private content detected in public file: {relative.as_posix()}")
        if re.search(r"(?:Student ID|学号)[^\n]{0,12}\d{6,}", text):
            raise ValueError(f"Student identifier detected: {relative.as_posix()}")
        payload = path.read_bytes()
        files[relative.as_posix()] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    for required in ("papers/paper_en.pdf", "papers/paper_zh.pdf", "LICENSE", "CITATION.cff"):
        if required not in files:
            raise ValueError(f"Public release is incomplete: {required}")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    manifest = {"schema_version": 1, "research_source_commit": revision.stdout.strip(),
                "privacy_policy": "Name and affiliation retained; student ID and private research artifacts omitted.",
                "files": files}
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
    LOGGER.info("Public manifest verified files=%d bytes=%d", len(files), sum(row["bytes"] for row in files.values()))
    return manifest


def verify_manifest(destination: Path) -> int:
    """按公开清单核验字节，拒绝越界或不在白名单内的条目。"""
    manifest = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        relative = Path(name)
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()) or not selected_file(relative):
            raise ValueError(f"Invalid public manifest entry: {name}")
        payload = path.read_bytes()
        if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError(f"Public file differs from manifest: {name}")
    LOGGER.info("Public manifest hashes passed files=%d", len(manifest["files"]))
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = args.root.resolve()
    destination = args.output.resolve()
    if args.verify:
        verify_manifest(destination)
    elif args.finalize:
        finalize_project(root, destination)
    else:
        export_project(root, destination)


if __name__ == "__main__":
    main()
