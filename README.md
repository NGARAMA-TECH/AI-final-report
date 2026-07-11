# AI Final Report Reproducibility Package

Research topic: A Comprehensive Survey of Multimodal Large Language Models: Architectures, Training Paradigms, Benchmark Evaluation, and Future Directions

Author: ISSA ISSA RASHID  
Student ID: 25SF51115  
Email: 25sf51115@stu.hit.edu.cn  
Course: Advanced Artificial Intelligence (2026)  
University: Harbin Institute of Technology, Shenzhen  
Paper type: ChinaXiv survey paper  

## Scope

This repository contains a reproducible survey-paper package for a course final report on multimodal large language models (MLLMs). It does not claim new model training results. The included scripts support reproducible generation and validation of survey tables from curated metadata.

## Important Reproducibility Statement

No benchmark numbers are fabricated in this package. The paper discusses architectures, training paradigms, datasets, and evaluation practices qualitatively. Any future quantitative table must be populated only from executed experiments or directly verified source papers.

The intended public repository URL is:

https://github.com/NGARAMA-TECH/AI-final-report

At the time this local package was prepared, the public GitHub repository existed but was empty. Upload these local files before claiming that the public repository contains the reproducibility materials.

## Repository Structure

```text
AI-final-report/
|-- README.md
|-- requirements.txt
|-- environment.yml
|-- configs/
|   |-- literature_review.yaml
|-- datasets/
|   |-- README.md
|   |-- raw/
|   |-- processed/
|-- docs/
|   |-- notes.md
|   |-- timeline.md
|-- paper/
|   |-- paper.md
|   |-- references.bib
|   |-- figures/
|   |-- tables/
|       |-- benchmark_taxonomy.csv
|       |-- model_taxonomy.csv
|-- scripts/
|   |-- generate_tables.py
|   |-- validate_references.py
|-- src/
|   |-- preprocessing/
|   |-- training/
|   |-- evaluation/
|   |-- inference/
|   |-- utils/
|   |-- visualization/
|-- results/
|   |-- figures/
|   |-- tables/
|   |-- predictions/
```

## Quick Start

Create an environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Validate citation keys used in the paper:

```bash
python scripts/validate_references.py
```

Regenerate survey tables:

```bash
python scripts/generate_tables.py
```

The generated outputs are written to `results/tables/`.

## Final Paper Files

The ChinaXiv-style submission files are in `paper/`:

- `paper_chinaxiv.md`: formatted source manuscript with numbered citations and references
- `paper_chinaxiv.docx`: Word submission document
- `paper_chinaxiv_final_with_bottom_line.pdf`: final PDF submission document with footer rule
- `CHINAXIV_FORMAT_CHECKLIST.md`: formatting and integrity checklist
- `figures/model_timeline.png` and `figures/benchmark_timeline.png`: reproducible academic charts

The layout comparison against the provided ChinaXiv sample is in `docs/chinaxiv_sample_comparison.md`.

The working draft remains available as `paper/paper.md`.

The paper currently has a 222-word abstract and 42 numbered references in the formatted manuscript.

## Citation Policy

Use only genuine papers, technical reports, datasets, and benchmark descriptions. Add new references to `paper/references.bib` before citing them in `paper/paper.md`.
