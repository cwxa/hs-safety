"""核对公开文档的相对文件链接，不联网检查外部站点。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def check_links(root: Path) -> list[str]:
    files = [root / "README.md", root / "README.zh-CN.md", root / "papers/README.md"]
    files.extend((root / "docs").glob("*.md"))
    errors = []
    for path in files:
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            target = match.group(1).strip().strip("<>")
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            destination = path.parent / unquote(parts.path)
            if not destination.exists():
                errors.append(f"{path.relative_to(root).as_posix()}: missing link {target}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check_links(args.root)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Documentation relative links passed")


if __name__ == "__main__":
    main()
