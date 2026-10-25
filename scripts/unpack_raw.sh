#!/usr/bin/env bash
# Unpack the chunks Bhavika shares into data/raw/{seq,ann}/.
#   bash scripts/unpack_raw.sh  [dir holding the BRepVLM folders]      (default: data/BRepVLM)
# STEP files (.7z) are not needed for training; only for STEP-based evaluation later.
set -euo pipefail
SRC=${1:-data/BRepVLM}
OUT=${RAW:-data/raw}
mkdir -p "$OUT/seq" "$OUT/ann"
while IFS= read -r f; do
  case "$f" in
    *Sequences*|*/seq_*) echo "seq <- $f"; tar -xzf "$f" -C "$OUT/seq" --exclude='*.quantization_diagnostics.json' ;;
    *Text_Annotations*)  echo "ann <- $f"; tar -xzf "$f" -C "$OUT/ann" ;;
    *) echo "skip  $f" ;;
  esac
done < <(find "$SRC" -name '*.tar.gz' | sort)
echo "sequences: $(find "$OUT/seq" -name '*.seq.txt' | wc -l)   annotated models: $(find "$OUT/ann" -name '*_global_shape.json' | wc -l)"
