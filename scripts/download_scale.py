"""下载固定提交的免费公开1.5B模型；仅写规模验证资源目录。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from huggingface_hub import HfApi, snapshot_download

    from hs_safety.config import load_config
    from hs_safety.io import configure_logging, get_logger, hash_file, save_json, timed

    parser = argparse.ArgumentParser(description="Download the pinned public Qwen 1.5B scale backbone")
    parser.add_argument("--config", default="configs/scale.yaml")
    parser.add_argument("--verify-only", action="store_true", help="Verify local assets without network access")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config["paths"]["logs"], "download_scale")
    logger = get_logger("download_scale")
    options = config["model"]
    revision = options["revision"]
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Scale backbone revision must be a full immutable commit SHA.")
    os.environ["HF_HOME"] = str(Path(options["cache_dir"]).resolve())
    destination = Path(options["local_path"])
    if not args.verify_only:
        with timed(logger, "download_pinned_scale_model", model=options["id"], revision=revision):
            remote = HfApi(token=False).model_info(options["id"], revision=revision)
            if remote.sha != revision:
                raise ValueError("Official model metadata does not match the pinned revision.")
            snapshot_download(options["id"], revision=revision, local_dir=destination,
                              token=False, max_workers=2,
                              allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"])
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "LICENSE"]
    missing = [filename for filename in required if not (destination / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing scale assets: {missing}")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": hash_file(path)}
             for path in sorted(destination.iterdir()) if path.is_file() and not path.name.startswith(".")}
    manifest = {"schema_version": 1, "model_id": options["id"], "revision": revision,
                "official_url": f"https://huggingface.co/{options['id']}/tree/{revision}",
                "local_path": str(destination.resolve()), "verification_only": args.verify_only,
                "files": files, "download_scope": "public pretrained model only; no local data uploaded"}
    output = Path(config["paths"]["results"]) / "model_download_manifest.json"
    save_json(output, manifest)
    logger.info("Scale assets verified revision=%s files=%d bytes=%d manifest=%s", revision, len(files), sum(item["bytes"] for item in files.values()), output)


if __name__ == "__main__":
    main()
