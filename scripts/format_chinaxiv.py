from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[1]
PAPER_MD = ROOT / "paper" / "paper.md"
BIB = ROOT / "paper" / "references.bib"
MODEL_TABLE = ROOT / "paper" / "tables" / "model_taxonomy.csv"
BENCHMARK_TABLE = ROOT / "paper" / "tables" / "benchmark_taxonomy.csv"
OUT_MD = ROOT / "paper" / "paper_chinaxiv.md"
OUT_DOCX = ROOT / "paper" / "paper_chinaxiv.docx"


CITATION_RE = re.compile(r"\[@([A-Za-z0-9:_-]+)\]")
BIB_ENTRY_RE = re.compile(r"@(\w+)\{([^,]+),([\s\S]*?)\n\}", re.MULTILINE)
BIB_FIELD_RE = re.compile(r"\n\s*(\w+)\s*=\s*\{([\s\S]*?)\},?", re.MULTILINE)


def parse_bib(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for entry_type, key, body in BIB_ENTRY_RE.findall(text):
        fields = {"ENTRYTYPE": entry_type}
        for field, value in BIB_FIELD_RE.findall(body):
            fields[field.lower()] = " ".join(value.split())
        entries[key] = fields
    return entries


def citation_order(text: str) -> OrderedDict[str, int]:
    order: OrderedDict[str, int] = OrderedDict()
    for key in CITATION_RE.findall(text):
        if key not in order:
            order[key] = len(order) + 1
    return order


def replace_citations(text: str, order: OrderedDict[str, int]) -> str:
    def repl(match: re.Match[str]) -> str:
        return f"[{order[match.group(1)]}]"

    return CITATION_RE.sub(repl, text)


def split_authors(author_field: str) -> list[str]:
    if not author_field:
        return []
    return [a.strip() for a in author_field.split(" and ")]


def normalize_author(author: str) -> str:
    author = author.strip("{}")
    if "," in author:
        last, first = [part.strip() for part in author.split(",", 1)]
        return f"{last} {first}"
    return author


def format_authors(author_field: str) -> str:
    authors = [normalize_author(a) for a in split_authors(author_field)]
    if not authors:
        return ""
    if len(authors) > 3:
        return ", ".join(authors[:3]) + ", et al"
    return ", ".join(authors)


def format_reference(index: int, entry: dict[str, str]) -> str:
    authors = format_authors(entry.get("author", ""))
    title = entry.get("title", "").rstrip(".")
    year = entry.get("year", "")
    journal = entry.get("journal", "")
    booktitle = entry.get("booktitle", "")
    institution = entry.get("institution", "")
    url = entry.get("url", "")
    entry_type = entry.get("ENTRYTYPE", "").lower()
    author_part = authors.rstrip(".")

    if entry_type == "inproceedings":
        source = booktitle
        return f"[{index}] {author_part}. {title}[C]. {source}, {year}."
    if entry_type == "article":
        source = journal
        return f"[{index}] {author_part}. {title}[J]. {source}, {year}."
    if entry_type == "techreport":
        suffix = f" {url}" if url else ""
        return f"[{index}] {author_part}. {title}[R]. {institution}, {year}.{suffix}"
    return f"[{index}] {author_part}. {title}. {year}."


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_to_markdown(path: Path) -> str:
    rows = read_csv(path)
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def build_chinaxiv_markdown() -> tuple[str, list[str], list[str]]:
    original = PAPER_MD.read_text(encoding="utf-8")
    bib = parse_bib(BIB)
    order = citation_order(original)
    text = replace_citations(original, order)

    text = text.replace("## References\n\nThe bibliography is maintained in `references.bib`.\n", "").rstrip()

    model_table = "Table 1. Representative MLLM architecture taxonomy.\n\n" + csv_to_markdown(MODEL_TABLE)
    benchmark_table = "Table 2. Representative MLLM benchmark taxonomy.\n\n" + csv_to_markdown(BENCHMARK_TABLE)

    text = text.replace(
        "## 6. Training Paradigms",
        model_table + "\n\n## 6. Training Paradigms",
        1,
    )
    text = text.replace(
        "### 7.6 Comparative Analysis Without Fabricated Scores",
        benchmark_table + "\n\n### 7.6 Comparative Analysis Without Fabricated Scores",
        1,
    )

    references = [format_reference(number, bib[key]) for key, number in order.items()]
    text += "\n\n## References\n\n" + "\n\n".join(references) + "\n"
    return text, references, list(order.keys())


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    para = cell.paragraphs[0]
    run = para.add_run(text.replace("_", " "))
    run.bold = bold
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    run.font.size = Pt(8)


def add_csv_table(doc: Document, caption: str, path: Path) -> None:
    para = doc.add_paragraph(caption)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in para.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(9)
        run.bold = True

    rows = read_csv(path)
    headers = list(rows[0].keys())
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        set_cell_text(table.rows[0].cells[idx], header, bold=True)
        set_cell_shading(table.rows[0].cells[idx], "D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for idx, header in enumerate(headers):
            set_cell_text(cells[idx], row[header])


def set_normal_run(run, size: int = 10, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    run.font.size = Pt(size)
    run.bold = bold


def add_paragraph(doc: Document, text: str, style: str | None = None, align=None) -> None:
    para = doc.add_paragraph(style=style)
    if align is not None:
        para.alignment = align
    para.paragraph_format.first_line_indent = Cm(0.74) if style is None else None
    para.paragraph_format.line_spacing = 1.15
    run = para.add_run(text)
    set_normal_run(run)


def add_heading(doc: Document, text: str, level: int) -> None:
    para = doc.add_paragraph()
    if level == 1:
        para.paragraph_format.space_before = Pt(12)
        para.paragraph_format.space_after = Pt(6)
        size = 13
    else:
        para.paragraph_format.space_before = Pt(8)
        para.paragraph_format.space_after = Pt(4)
        size = 11
    run = para.add_run(text)
    set_normal_run(run, size=size, bold=True)


def create_docx(china_md: str, references: list[str]) -> None:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.18)
    section.right_margin = Cm(3.18)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    styles["Normal"].font.size = Pt(10)

    lines = china_md.splitlines()
    title = lines[0].lstrip("# ").strip()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    set_normal_run(r, size=16, bold=True)

    metadata = [
        "ISSA RASHID ISSA",
        "School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China",
        "Corresponding author: ISSA RASHID ISSA, 25sf51115@stu.hit.edu.cn",
        "ChinaXiv preprint style manuscript",
    ]
    for item in metadata:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(item)
        set_normal_run(r, size=10)

    doc.add_paragraph()

    in_references = False
    in_code_block = False
    skip_until_after_header = False
    idx = 1
    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()
        idx += 1

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith("Authors:") or stripped.startswith("Affiliation:") or stripped.startswith("Corresponding author:"):
            continue
        if stripped.startswith("Table 1. Representative"):
            add_csv_table(doc, stripped, MODEL_TABLE)
            skip_until_after_header = True
            continue
        if stripped.startswith("Table 2. Representative"):
            add_csv_table(doc, stripped, BENCHMARK_TABLE)
            skip_until_after_header = True
            continue
        if skip_until_after_header:
            if stripped.startswith("## ") or stripped.startswith("### "):
                skip_until_after_header = False
            else:
                continue
        if not stripped:
            continue
        if stripped == "## References":
            in_references = True
            add_heading(doc, "References", 1)
            continue
        if in_references:
            add_paragraph(doc, stripped)
            continue
        if stripped.startswith("## "):
            add_heading(doc, stripped[3:], 1)
            continue
        if stripped.startswith("### "):
            add_heading(doc, stripped[4:], 2)
            continue
        if stripped.startswith("- "):
            para = doc.add_paragraph(style="List Bullet")
            run = para.add_run(stripped[2:])
            set_normal_run(run)
            continue
        if re.match(r"^\d+\.\s+", stripped):
            para = doc.add_paragraph(style="List Number")
            run = para.add_run(re.sub(r"^\d+\.\s+", "", stripped))
            set_normal_run(run)
            continue
        add_paragraph(doc, stripped)

    doc.save(OUT_DOCX)


def main() -> None:
    china_md, references, _keys = build_chinaxiv_markdown()
    OUT_MD.write_text(china_md, encoding="utf-8")
    create_docx(china_md, references)
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_DOCX}")


if __name__ == "__main__":
    main()
