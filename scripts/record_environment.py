"""从实测运行环境生成依赖闭包和资源指纹，避免手工维护重复版本清单。"""
from __future__ import annotations

import platform
import sys
from collections import deque
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import tomllib

    from hs_safety.io import hash_file, save_json

    settings = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    roots = settings["project"]["dependencies"] + settings["project"]["optional-dependencies"]["dev"]
    # torchvision用于修复本机预装环境的视觉依赖冲突，源码仍只执行文本任务。
    roots.append("torchvision")
    pending = deque(Requirement(value) for value in roots)
    visited, packages = set(), {}
    while pending:
        requirement = pending.popleft()
        key = canonicalize_name(requirement.name)
        extras = tuple(sorted(requirement.extras))
        if (key, extras) in visited:
            continue
        visited.add((key, extras))
        distribution = metadata.distribution(requirement.name)
        actual = distribution.version
        if requirement.specifier and actual not in requirement.specifier:
            raise ValueError(f"Installed dependency violates requirement: {requirement} actual={actual}")
        packages[key] = actual
        for child in distribution.requires or []:
            parsed = Requirement(child)
            if parsed.marker is None or any(parsed.marker.evaluate({"extra": extra}) for extra in (extras or ("",))):
                pending.append(parsed)
    header = ("# Generated from the tested Windows / Python environment by scripts/record_environment.py.\n"
              "# CUDA packages are obtained from the official PyTorch wheel index.\n"
              "--extra-index-url https://download.pytorch.org/whl/cu126\n")
    Path("requirements.lock").write_text(header + "\n".join(f"{key}=={value}" for key, value in sorted(packages.items())) + "\n", encoding="utf-8")
    assets = {}
    for path in [Path("artifacts/backbones/qwen2.5-0.5b/model.safetensors"),
                 Path(".tools/tectonic/tectonic.exe")]:
        if path.exists():
            assets[str(path)] = {"bytes": path.stat().st_size, "sha256": hash_file(path)}
    save_json("results/environment_packages.json", {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "environment": "Project virtualenv; existing CUDA torch reused through system-site-packages",
        "tex_engine": "Tectonic 0.17.0", "asset_fingerprints": assets,
    })
    print(f"Recorded dependency closure: {len(packages)} packages")


if __name__ == "__main__":
    main()
