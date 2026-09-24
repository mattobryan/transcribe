# Build all project documents to ../pdf with XeLaTeX.
# Requires: TeX Live (xetex, latex-extra, pictures), fonts FreeSerif + DejaVu.
DOCS := 00_PROBLEM_STATEMENT 01_PRD 02_TRD 03_UI_UX_DESIGN 04_APP_FLOW \
        05_BACKEND_SCHEMA 06_IMPLEMENTATION_PLAN 07_KAGGLE_RUNBOOK STYLE_GUIDE
OUTDIR := ../pdf

all: $(DOCS:%=$(OUTDIR)/%.pdf)

$(OUTDIR)/%.pdf: %.tex kctdocs.sty $(wildcard figures/*.tex)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build $<
	mkdir -p $(OUTDIR) && cp build/$*.pdf $@

clean:
	rm -rf build

.PHONY: all clean
