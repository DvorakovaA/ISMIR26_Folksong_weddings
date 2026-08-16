#!/bin/bash
# Run pairwise feature selection for all language pairs.
# By running feature_select_pipeline.py script with parameters
#
# ./all_feat_select.sh probabilities.parquet labels.csv label_col output_dir

set -euo pipefail

PARQUET="${1:?}"
LABELS="${2:?}"
LABEL_COL="${3:?}"
OUT_DIR="${4:?}"

LANGUAGES=(cs ko uk nl et)

mkdir -p "$OUT_DIR"

for ((i = 0; i < ${#LANGUAGES[@]}; i++)); do
    for ((j = i + 1; j < ${#LANGUAGES[@]}; j++)); do
        A="${LANGUAGES[$i]}"
        B="${LANGUAGES[$j]}"
        OUT="${OUT_DIR}/${A}_vs_${B}.txt"

        echo "Computing ${A} vs ${B}"
        python3 feature_select_pipeline.py "$PARQUET" \
            --labels "$LABELS" \
            --label-col "$LABEL_COL" \
            --class-a "$A" \
            --class-b "$B" \
            --output "$OUT"
    done
done

echo "Done. Results in ${OUT_DIR}/"