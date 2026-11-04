"""全量扫描最终英文 PDF 的技能词表，保留上下文供人工判断。"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

TERMS = ["innovative", "pioneering", "revolutionary paradigm", "transformative framework", "superior",
         "surpass", "excel", "remarkable", "unprecedented", "achieves SOTA", "breakthrough performance",
         "general-purpose", "is capable of", "notably", "yet", "yielding", "at its essence", "encompass",
         "differentiate", "reveal", "underscore", "exhibit superior capability", "exceed", "pave the way for",
         "highlight the potential of", "profound challenges", "stems from", "rigid", "impede"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["before", "after", "polished"], required=True)
    args = parser.parse_args()
    pdf = Path("papers/paper_en.pdf")
    document = fitz.open(pdf)
    findings = []
    for number, page in enumerate(document, 1):
        text = re.sub(r"-\s*\n", "", page.get_text())
        text = re.sub(r"\s+", " ", text)
        for term in [*TERMS, "—"]:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                findings.append({"page": number, "term": term,
                                 "context": text[max(0, match.start()-75):match.end()+85]})
    output = {"pdf": pdf.as_posix(), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
              "scope": "Every page of the rendered English PDF, including captions and bibliography.",
              "pages_scanned": len(document), "match_policy": "Case-insensitive substring; contextual review required.",
              "findings": findings, "term_counts": {term: sum(r["term"] == term for r in findings)
                                                       for term in [*TERMS, "—"]}}
    path = Path("reviews") / f"language_scan_{args.stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Language scan completed: pages={len(document)} occurrences={len(findings)} output={path}")


if __name__ == "__main__":
    main()
