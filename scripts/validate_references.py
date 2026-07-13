from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.md"
FORMATTED_MD = ROOT / "paper" / "Survey_Multimodal_Intelligence_ISSA_25SF51115.md"
BIB = ROOT / "paper" / "references.bib"


KEY_PATTERN = re.compile(r"\[@([A-Za-z0-9:_-]+)\]")
NUMBERED_PATTERN = re.compile(r"\[(\d+)\]")
BIBKEY_PATTERN = re.compile(r"@\w+\{([^,]+),")


def ordered_source_citations(text: str) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for key in KEY_PATTERN.findall(text):
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def main() -> None:
    paper_text = PAPER.read_text(encoding="utf-8")
    bib_text = BIB.read_text(encoding="utf-8")

    source_keys = ordered_source_citations(paper_text)
    bib_keys = set(BIBKEY_PATTERN.findall(bib_text))
    cited_keys = set(source_keys)

    if FORMATTED_MD.exists():
        formatted_text = FORMATTED_MD.read_text(encoding="utf-8")
        numbered_citations = {int(num) for num in NUMBERED_PATTERN.findall(formatted_text)}
        out_of_range = sorted(num for num in numbered_citations if num < 1 or num > len(source_keys))
        if out_of_range:
            print("Numbered citations outside generated reference range:")
            for num in out_of_range:
                print(f"  - [{num}]")
            raise SystemExit(1)

    missing = sorted(cited_keys - bib_keys)

    print(f"Citation keys used in paper: {len(cited_keys)}")
    print(f"Bibliography entries: {len(bib_keys)}")

    if missing:
        print("Missing bibliography entries:")
        for key in missing:
            print(f"  - {key}")
        raise SystemExit(1)

    print("Reference validation passed.")


if __name__ == "__main__":
    main()
