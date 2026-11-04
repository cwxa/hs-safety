"""下载固定版本的公开模型和研究数据；不上传本地数据。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from huggingface_hub import hf_hub_download, snapshot_download

    from hs_safety.config import ensure_directories, load_config
    from hs_safety.io import configure_logging, get_logger, save_json, timed

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_directories(config)
    configure_logging(config["paths"]["logs"], "download")
    logger = get_logger("download")
    os.environ["HF_HOME"] = str(Path(config["model"]["cache_dir"]).resolve())
    manifest = {"model": config["model"], "datasets": {}}
    if not args.skip_model:
        with timed(logger, "download_model", model=config["model"]["id"], revision=config["model"]["revision"]):
            snapshot_download(
                config["model"]["id"], revision=config["model"]["revision"],
                local_dir=config["model"]["local_path"], token=False, max_workers=2,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"],
            )
    raw = Path(config["paths"]["raw"])
    resources = [
        ("PKU-Alignment/BeaverTails", config["data"]["beavertails_revision"],
         ["round0/30k/train.jsonl.gz", "round0/30k/test.jsonl.gz", "README.md"], raw / "beavertails"),
        ("Alibaba-AAIG/StreamGuardBench", config["data"]["streamguard_revision"],
         ["wildguard/qwen3_8b/test/data-00000-of-00001.arrow", "README.md"], raw / "streamguard"),
    ]
    for repo, revision, files, destination in resources:
        with timed(logger, "download_dataset", repo=repo, revision=revision):
            for filename in files:
                hf_hub_download(repo, filename=filename, repo_type="dataset", revision=revision,
                                local_dir=destination, token=False)
        manifest["datasets"][repo] = {"revision": revision, "files": files}
    save_json(raw / "download_manifest.json", manifest)
    logger.info("Asset download completed manifest=%s", raw / "download_manifest.json")


if __name__ == "__main__":
    main()
