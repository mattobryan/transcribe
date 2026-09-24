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
| 07 | Kaggle Runbook | [pdf](pdf/07_KAGGLE_RUNBOOK.pdf) | [tex](latex/07_KAGGLE_RUNBOOK.tex) | Step-by-step unattended pilot run on a Kaggle GPU |
| – | Style Guide | [pdf](pdf/STYLE_GUIDE.pdf) | [tex](latex/STYLE_GUIDE.tex) | Italic = Kiswahili, hybrid words, numbers, markers |

## Versions and change logs

Every document shows its version and time (EAT) in the title block and ends with a **Change log**:
a version table, then one entry per changed section with what it said before, what it says now,
and links to the previous text (on GitHub at the exact commit and lines, and as an archived PDF).
Sections that changed carry a "Changed in vX.Y" note under their heading.

Previous versions are kept in `pdf/archive/v1.0/` and `pdf/archive/v1.1/`.

When you change a document:

1. Before editing, copy the current PDFs into `pdf/archive/v<current>/`.
2. Edit the `.tex`, bump the Version row, and add `\kctchanged{<new>}` under each changed heading.
3. Add a row to the document's `kctversions` table and one `\kctchange{...}` entry per changed section.
   Use `\kctsrc{<previous commit>}{<file>.tex}{<first line>}{<last line>}{...}` for the previous-text link.
4. `make`, then commit the `.tex` and PDFs together.

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
