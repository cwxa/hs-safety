"""打包源码、实测证据和轻量权重，逐文件记录 SHA-256 并核对 ZIP CRC。"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(root: Path) -> list[Path]:
    sources = ["hs_safety", "tests", "scripts", "configs", "docs", "reviews", ".github", ".githooks",
               "papers", "results", "data/processed", "artifacts/models", "artifacts/preprocessing",
               "artifacts/scale/models", "artifacts/scale/preprocessing"]
    excluded_parts = {"__pycache__", "qa"}
    excluded_suffixes = {".pyc", ".aux", ".bbl", ".blg", ".log", ".out", ".xdv", ".tmp"}
    files = {path for path in root.iterdir() if path.is_file()
             and (path.suffix in {".md", ".toml", ".lock"} or path.name in {"LICENSE", ".gitignore"})}
    for source in sources:
        for path in (root / source).rglob("*"):
            if (path.is_file() and not excluded_parts.intersection(path.relative_to(root).parts)
                    and path.suffix not in excluded_suffixes):
                files.add(path)
    # 保留上游声明与训练日志；大骨干、原始隐状态、环境与下载缓存由复现入口重建。
    files.update(path for path in (root / "data/raw").rglob("README.md"))
    files.add(root / "data/raw/download_manifest.json")
    files.update(path for path in (root / "logs").rglob("*.log"))
    return sorted(files)


def verify_evidence(root: Path) -> dict:
    tests = ET.parse(root / "results/test_report.xml").getroot()
    suites = list(tests.iter("testsuite"))
    totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if not totals["tests"] or totals["failures"] or totals["errors"]:
        raise ValueError(f"Test evidence is incomplete: {totals}")
    paper_audit = json.loads((root / "papers/qa/paper_audit.json").read_text(encoding="utf-8"))
    if any(paper["problems"] for paper in paper_audit):
        raise ValueError("PDF automatic verification contains unresolved problems")
    for paper in paper_audit:
        if sha256(root / paper["pdf"]) != paper["sha256"]:
            raise ValueError("PDF changed after verification; rerun verify_papers.py")
    figures, fonts = {}, {}
    for path in sorted((root / "papers/figure_contracts").glob("*.collision-audit.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit["verdict"] != "PASS":
            raise ValueError(f"Figure collision verification failed: {path}")
        figures[path.name] = audit["summary"]
        font_path = path.with_name(path.name.replace(".collision-audit", ".font-audit"))
        font = json.loads(font_path.read_text(encoding="utf-8"))
        if not font["auditable"] or font["below_minimum_count"]:
            raise ValueError(f"Figure font verification failed: {font_path}")
        fonts[font_path.name] = font["minimum_found_pt"]
    if len(figures) != len(list((root / "papers/figures").glob("*.pdf"))):
        raise ValueError("Figure verification coverage is incomplete")
    return {"tests": totals, "paper_pages": {Path(paper["pdf"]).name: paper["page_count"]
                                             for paper in paper_audit},
            "pdf_automatic_problems": 0, "figure_collision_audits": figures,
            "figure_minimum_font_pt": fonts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("deliverables/hs_safety_research.zip"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    quality = verify_evidence(root)
    files = collect_files(root)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    manifest = {"created_utc": datetime.now(UTC).isoformat(), "source_commit": revision.stdout.strip(),
                "quality": quality, "files": {path.relative_to(root).as_posix(): {
                    "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files},
                "omitted": ["large frozen backbones", "raw/derived hidden-state caches", "Python environment",
                            "TeX binaries and package cache", "raw upstream dataset containers"]}
    readme = """# Research delivery

Start with README.md and docs/reproducibility.md. Both final IEEEtran PDFs and
their complete LaTeX sources are in papers/. Author metadata are shared through
papers/author_info.tex. Results include all 36 completed model fits, held-out
calibration, external evaluation and per-response score trajectories. Fitted
detectors and PCA parameters are in artifacts/; prepared records and upstream
dataset notices are in data/. Original project code is MIT; dataset/model terms
remain separate (BeaverTails CC BY-NC 4.0; StreamGuardBench Apache-2.0).

The archive omits multi-GB backbones and feature caches. Download fixed upstream
revisions with scripts/download_assets.py and scripts/download_scale.py; rebuild
features with the documented extract commands. Prepared data already exists:
do not prepare or run the scale configuration. Rebuild papers using XeLaTeX plus
BibTeX or the documented portable Tectonic tool. Current final PDFs are included.

DELIVERY_MANIFEST.json lists SHA-256 hashes for all payload files. The manifest
and this note are metadata and do not hash themselves. The companion .sha256
file verifies the whole archive. All files remain local; this is not a public
dataset or model release.
"""
    logger.info("Packaging started: files=%s output=%s", len(files), output)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("DELIVERY_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("DELIVERY_README.md", readme)
    with ZipFile(output) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise ValueError(f"Archive CRC verification failed: {bad_file}")
    checksum = sha256(output)
    output.with_suffix(".sha256").write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Packaging complete: files=%s bytes=%s sha256=%s seconds=%.3f",
                len(files), output.stat().st_size, checksum, time.perf_counter() - started)


if __name__ == "__main__":
    main()
