# Project Documents

The project documents are written in LaTeX (`latex/`) and compiled to PDF (`pdf/`).
**Edit the `.tex` files. The PDFs are build outputs.**

| # | Document | PDF | Source | Answers |
|---|---|---|---|---|
| 00 | Problem Statement | [pdf](pdf/00_PROBLEM_STATEMENT.pdf) | [tex](latex/00_PROBLEM_STATEMENT.tex) | Why this exists, and why the current pipeline cannot get there |
| 01 | PRD | [pdf](pdf/01_PRD.pdf) | [tex](latex/01_PRD.tex) | Who it is for, what it must do, how success is measured |
| 02 | TRD | [pdf](pdf/02_TRD.pdf) | [tex](latex/02_TRD.tex) | Model choice (and why not a blend of two models), pipelines, API, evaluation |
| 03 | UI/UX Design | [pdf](pdf/03_UI_UX_DESIGN.pdf) | [tex](latex/03_UI_UX_DESIGN.tex) | Screens, keyboard-first review, design tokens |
| 04 | App Flow | [pdf](pdf/04_APP_FLOW.pdf) | [tex](latex/04_APP_FLOW.tex) | End-to-end flows, state machines, failure handling |
| 05 | Backend Schema | [pdf](pdf/05_BACKEND_SCHEMA.pdf) | [tex](latex/05_BACKEND_SCHEMA.tex) | SQLite DDL, invariants, file store, export formats, migration |
| 06 | Implementation Plan | [pdf](pdf/06_IMPLEMENTATION_PLAN.pdf) | [tex](latex/06_IMPLEMENTATION_PLAN.tex) | Phases, tasks, acceptance criteria, experiments |
| – | Style Guide | [pdf](pdf/STYLE_GUIDE.pdf) | [tex](latex/STYLE_GUIDE.tex) | Italic = Kiswahili, hybrid words, numbers, markers |

## Building

Requires TeX Live with XeLaTeX (`texlive-xetex`, `texlive-latex-extra`, `texlive-pictures`) and the
FreeSerif and DejaVu fonts (`fonts-freefont-otf`, `fonts-dejavu-core`).

```bash
cd docs/latex
make          # builds every document into docs/pdf/
make clean    # removes docs/latex/build/
```

Layout:

```text
docs/latex/
  kctdocs.sty      # shared style: fonts, earth-tone colours, headers, title block, tables, listings, TikZ styles
  0X_*.tex         # one file per document
  figures/*.tex    # TikZ diagrams (architecture, flows, state machine, ER overview, Gantt)
  Makefile
```

[PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md) describes the original VAD-chunk workflow and is kept for history. The documents above supersede it.
