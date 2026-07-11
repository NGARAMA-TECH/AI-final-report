from __future__ import annotations

from collections import Counter
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "chinaxiv-202509.00064.pdf"
TARGET = ROOT / "paper" / "paper_chinaxiv_final_with_bottom_line.pdf"
OUT = ROOT / "docs" / "chinaxiv_sample_comparison.md"


def summarize_pdf(path: Path, pages: int = 3) -> dict[str, object]:
    doc = fitz.open(path)
    page = doc[0]
    sizes = []
    lefts = []
    rights = []
    tops = []
    bottoms = []
    font_names = Counter()
    for page_index in range(min(pages, doc.page_count)):
        p = doc[page_index]
        width, height = p.rect.width, p.rect.height
        spans = []
        for block in p.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        spans.append(span)
        if not spans:
            continue
        left = min(span["bbox"][0] for span in spans)
        right = width - max(span["bbox"][2] for span in spans)
        top = min(span["bbox"][1] for span in spans)
        bottom = height - max(span["bbox"][3] for span in spans)
        lefts.append(left)
        rights.append(right)
        tops.append(top)
        bottoms.append(bottom)
        for span in spans:
            sizes.append(round(span["size"], 1))
            font_names[span.get("font", "unknown")] += 1
    return {
        "pages": doc.page_count,
        "page_width_pt": round(page.rect.width, 1),
        "page_height_pt": round(page.rect.height, 1),
        "left_margin_pt": round(sum(lefts) / len(lefts), 1),
        "right_margin_pt": round(sum(rights) / len(rights), 1),
        "top_text_pt": round(sum(tops) / len(tops), 1),
        "bottom_text_pt": round(sum(bottoms) / len(bottoms), 1),
        "common_font_sizes": Counter(sizes).most_common(6),
        "common_fonts": font_names.most_common(5),
    }


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target PDF not found: {TARGET}")
    sample = summarize_pdf(SAMPLE)
    target = summarize_pdf(TARGET)
    lines = [
        "# ChinaXiv Sample Layout Comparison",
        "",
        "Sample file: `chinaxiv-202509.00064.pdf`",
        "",
        "Target file: `paper/paper_chinaxiv_final_with_bottom_line.pdf`",
        "",
        "## Measured Layout",
        "",
        "| Metric | Sample | Target |",
        "| --- | ---: | ---: |",
    ]
    for key in [
        "pages",
        "page_width_pt",
        "page_height_pt",
        "left_margin_pt",
        "right_margin_pt",
        "top_text_pt",
        "bottom_text_pt",
    ]:
        lines.append(f"| {key} | {sample[key]} | {target[key]} |")
    lines += [
        "",
        "## Font Observations",
        "",
        f"- Sample common font sizes: `{sample['common_font_sizes']}`",
        f"- Target common font sizes: `{target['common_font_sizes']}`",
        f"- Sample common fonts: `{sample['common_fonts']}`",
        f"- Target common fonts: `{target['common_fonts']}`",
        "",
        "## Formatting Decision",
        "",
        "The target manuscript uses Letter page size, narrow centered text width, 10 pt body text, 12 pt section headings, a bottom footer rule on every page, academic figure/table captions, numbered citations, and numbered references to match the observable ChinaXiv sample style as closely as practical in Word format.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
