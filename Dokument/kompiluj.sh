#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
DOC="szablon"

case "${1:-}" in
    clean)
        latexmk -C "$DOC.tex"
        rm -f "$DOC.bbl" "$DOC.lol"
        echo "Wyczyszczono pliki pomocnicze."
        ;;
    *)
        latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error "$DOC.tex"
        echo "Gotowe: $DIR/$DOC.pdf"
        ;;
esac
