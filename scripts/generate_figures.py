from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE = ROOT / "paper" / "tables" / "model_taxonomy.csv"
BENCHMARK_TABLE = ROOT / "paper" / "tables" / "benchmark_taxonomy.csv"
FIG_DIR = ROOT / "paper" / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_model_timeline() -> None:
    rows = read_rows(MODEL_TABLE)
    counts = Counter(row["year"] for row in rows)
    years = sorted(counts)
    values = [counts[year] for year in years]

    plt.figure(figsize=(6.2, 3.0))
    bars = plt.bar(years, values, color="#6f6f6f", edgecolor="black", linewidth=0.6)
    plt.xlabel("Publication year", fontsize=9)
    plt.ylabel("Representative models", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(range(0, max(values) + 2), fontsize=8)
    plt.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_timeline_models.png", dpi=300)
    plt.close()


def save_benchmark_timeline() -> None:
    rows = read_rows(BENCHMARK_TABLE)
    counts = Counter(row["year"] for row in rows)
    years = sorted(counts)
    values = [counts[year] for year in years]

    plt.figure(figsize=(6.2, 3.0))
    plt.plot(years, values, marker="o", color="black", linewidth=1.2)
    plt.fill_between(years, values, color="#bdbdbd", alpha=0.45)
    plt.xlabel("Publication year", fontsize=9)
    plt.ylabel("Representative benchmarks", fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(range(0, max(values) + 2), fontsize=8)
    plt.grid(axis="both", linestyle="--", linewidth=0.4, alpha=0.6)
    for year, value in zip(years, values):
        plt.text(year, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_timeline_benchmarks.png", dpi=300)
    plt.close()


def add_box(ax, xy, text: str, width=1.8, height=0.55, face="#eeeeee") -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=0.8,
        edgecolor="black",
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def add_arrow(ax, start, end) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=10, linewidth=0.9, color="black")
    ax.add_patch(arrow)


def save_architecture_diagram() -> None:
    plt.figure(figsize=(6.2, 3.0))
    ax = plt.gca()
    ax.axis("off")
    add_box(ax, (0.1, 1.65), "Image / document / video frame", 1.7, 0.55, "#f2f2f2")
    add_box(ax, (2.15, 1.65), "Vision encoder", 1.25, 0.55, "#d9eaf7")
    add_box(ax, (3.75, 1.65), "Connector\nprojection / Q-Former /\ncross-attention", 1.55, 0.75, "#e2f0d9")
    add_box(ax, (5.65, 1.65), "LLM backbone", 1.25, 0.55, "#fff2cc")
    add_box(ax, (7.2, 1.65), "Grounded answer", 1.25, 0.55, "#fce4d6")
    for start, end in [((1.8, 1.93), (2.15, 1.93)), ((3.4, 1.93), (3.75, 1.93)), ((5.3, 1.93), (5.65, 1.93)), ((6.9, 1.93), (7.2, 1.93))]:
        add_arrow(ax, start, end)
    ax.text(4.25, 0.75, "Core risk: visual evidence can be lost or distorted before language reasoning.", ha="center", fontsize=8)
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0.4, 2.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_architecture.png", dpi=300)
    plt.close()


def save_connector_diagram() -> None:
    plt.figure(figsize=(6.2, 3.0))
    ax = plt.gca()
    ax.axis("off")
    add_box(ax, (0.2, 2.0), "Linear projection", 1.4, 0.5, "#f2f2f2")
    add_box(ax, (2.2, 2.0), "Low cost", 1.0, 0.5, "#e2f0d9")
    add_box(ax, (3.8, 2.0), "Detail bottleneck", 1.3, 0.5, "#fce4d6")
    add_box(ax, (0.2, 1.2), "Q-Former / queries", 1.4, 0.5, "#f2f2f2")
    add_box(ax, (2.2, 1.2), "Selective compression", 1.2, 0.5, "#e2f0d9")
    add_box(ax, (3.8, 1.2), "May discard evidence", 1.3, 0.5, "#fce4d6")
    add_box(ax, (0.2, 0.4), "Cross-attention", 1.4, 0.5, "#f2f2f2")
    add_box(ax, (2.2, 0.4), "Flexible evidence access", 1.2, 0.5, "#e2f0d9")
    add_box(ax, (3.8, 0.4), "Higher compute", 1.3, 0.5, "#fce4d6")
    for y in [2.25, 1.45, 0.65]:
        add_arrow(ax, (1.6, y), (2.2, y))
        add_arrow(ax, (3.4, y), (3.8, y))
    ax.text(3.1, 2.75, "Connector choices trade off cost, evidence preservation, and interpretability.", ha="center", fontsize=9)
    ax.set_xlim(0, 5.4)
    ax.set_ylim(0.1, 3.0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_connector_tradeoffs.png", dpi=300)
    plt.close()


def save_benchmark_taxonomy_diagram() -> None:
    plt.figure(figsize=(6.2, 3.0))
    ax = plt.gca()
    ax.axis("off")
    add_box(ax, (2.35, 2.2), "MLLM evaluation", 1.5, 0.5, "#d9eaf7")
    branches = [
        ((0.2, 1.35), "Perception\nVQA / GQA"),
        ((1.65, 0.65), "OCR & documents\nTextVQA / DocVQA"),
        ((3.15, 0.65), "Reasoning\nScienceQA / MathVista"),
        ((4.65, 1.35), "Hallucination\nPOPE / HallusionBench"),
    ]
    for xy, label in branches:
        add_box(ax, xy, label, 1.25, 0.65, "#f2f2f2")
        add_arrow(ax, (3.1, 2.2), (xy[0] + 0.63, xy[1] + 0.65))
    ax.text(3.1, 0.15, "No single benchmark captures perception, reasoning, grounding, and safety simultaneously.", ha="center", fontsize=8)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(0, 2.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_benchmark_taxonomy.png", dpi=300)
    plt.close()


def save_grounding_workflow_diagram() -> None:
    plt.figure(figsize=(6.2, 3.0))
    ax = plt.gca()
    ax.axis("off")
    labels = [
        ("User question", 0.1),
        ("Visual evidence\nregions / OCR", 1.75),
        ("Grounding check", 3.45),
        ("Reasoning", 5.0),
        ("Answer with\nuncertainty", 6.4),
    ]
    for label, x in labels:
        add_box(ax, (x, 1.45), label, 1.1, 0.65, "#f2f2f2")
    for x1, x2 in [(1.2, 1.75), (2.85, 3.45), (4.55, 5.0), (6.1, 6.4)]:
        add_arrow(ax, (x1, 1.78), (x2, 1.78))
    ax.text(3.8, 0.65, "Recommended behavior: cite visual evidence or state that evidence is insufficient.", ha="center", fontsize=8)
    ax.set_xlim(0, 7.7)
    ax.set_ylim(0.25, 2.7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure7_evidence_grounded.png", dpi=300)
    plt.close()


def save_hallucination_diagram() -> None:
    plt.figure(figsize=(6.2, 3.0))
    ax = plt.gca()
    ax.axis("off")
    add_box(ax, (0.25, 1.8), "Image evidence:\nbowl + cup\n(no knife)", 1.35, 0.75, "#e2f0d9")
    add_box(ax, (2.25, 1.8), "Language prior:\nkitchens often\ncontain knives", 1.35, 0.75, "#fff2cc")
    add_box(ax, (4.25, 1.8), "Hallucinated answer:\n'a knife is visible'", 1.45, 0.75, "#fce4d6")
    add_arrow(ax, (1.6, 2.18), (2.25, 2.18))
    add_arrow(ax, (3.6, 2.18), (4.25, 2.18))
    add_box(ax, (2.0, 0.55), "Preferred answer:\n'I do not see a knife;\nvisible items are...'", 2.2, 0.85, "#d9eaf7")
    ax.text(3.0, 0.2, "Hallucination occurs when plausible context overrides absent visual evidence.", ha="center", fontsize=8)
    ax.set_xlim(0, 6.1)
    ax.set_ylim(0.05, 2.85)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_hallucination.png", dpi=300)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_model_timeline()
    save_benchmark_timeline()
    save_architecture_diagram()
    save_connector_diagram()
    save_benchmark_taxonomy_diagram()
    save_grounding_workflow_diagram()
    save_hallucination_diagram()
    print(f"Wrote {FIG_DIR / 'figure1_architecture.png'}")
    print(f"Wrote {FIG_DIR / 'figure2_connector_tradeoffs.png'}")
    print(f"Wrote {FIG_DIR / 'figure3_timeline_models.png'}")
    print(f"Wrote {FIG_DIR / 'figure4_benchmark_taxonomy.png'}")
    print(f"Wrote {FIG_DIR / 'figure5_timeline_benchmarks.png'}")
    print(f"Wrote {FIG_DIR / 'figure6_hallucination.png'}")
    print(f"Wrote {FIG_DIR / 'figure7_evidence_grounded.png'}")


if __name__ == "__main__":
    main()
