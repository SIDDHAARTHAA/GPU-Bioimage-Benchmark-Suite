#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX_DIR="$ROOT_DIR/MiniProjectTemplate"

cd "$TEX_DIR"

pdflatex -interaction=nonstopmode MiniProjectTemplate.tex
bibtex MiniProjectTemplate
pdflatex -interaction=nonstopmode MiniProjectTemplate.tex
pdflatex -interaction=nonstopmode MiniProjectTemplate.tex

echo "Report generated: $TEX_DIR/MiniProjectTemplate.pdf"
