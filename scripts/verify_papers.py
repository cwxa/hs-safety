"""检查最终 PDF 的文本、页面边界与引用，并生成逐页审阅图。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw


def verify_paper(path: Path, output: Path) -> dict:
    document = fitz.open(path)
    problems, pages, thumbnails = [], [], []
    for index, page in enumerate(document):
        text = page.get_text()
        for marker in ("[?]", "??", "\ufffd", "Pending", "PENDING"):
            if marker in text:
                problems.append({"page": index + 1, "unexpected_marker": marker})
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    # 容许字体包围盒的亚像素舍入，但拒绝文字越过纸面边界。
                    box = fitz.Rect(span["bbox"])
                    if not (page.rect + (-1, -1, 1, 1)).contains(box):
                        problems.append({"page": index + 1, "out_of_page_text": span["text"]})
        preview = output / f"{path.stem}_{index + 1:02d}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False).save(preview)
        with Image.open(preview) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((510, 690))
            thumbnails.append(thumb)
        pages.append({"page": index + 1, "characters": len(text), "preview": preview.as_posix()})
    # 全部页面同时展示，供人工逐页核对分栏、图表裁切和异常留白。
    columns, cell_width, cell_height = 2, 530, 725
    sheet = Image.new("RGB", (columns * cell_width, ((len(pages) + 1) // 2) * cell_height), "#d9dde3")
    draw = ImageDraw.Draw(sheet)
    for index, thumbnail in enumerate(thumbnails):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        draw.text((x + 12, y + 6), f"{path.stem} / page {index + 1}", fill="black")
        sheet.paste(thumbnail, (x + 10, y + 26))
    sheet_path = output / f"{path.stem}_contact.png"
    sheet.save(sheet_path)
    return {"pdf": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_count": len(document), "pages": pages,
            "problems": problems, "contact_sheet": sheet_path.as_posix()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", default="papers")
    parser.add_argument("--output", default="papers/qa")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = [verify_paper(Path(args.papers) / f"paper_{language}.pdf", output)
               for language in ("en", "zh")]
    for report in reports:
        print(f"PDF checked: file={report['pdf']} pages={report['page_count']} problems={len(report['problems'])}")
    (output / "paper_audit.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    if any(report["problems"] for report in reports):
        raise SystemExit("PDF verification failed; inspect paper_audit.json")


if __name__ == "__main__":
    main()
