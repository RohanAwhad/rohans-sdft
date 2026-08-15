#!/bin/bash
# Poll for eval result file. Usage: bash poll_eval.sh <run_name> <epoch>
# Example: bash poll_eval.sh run_18 epoch_3
set -e

RUN="${1:?Usage: bash poll_eval.sh <run_name> <epoch>}"
EPOCH="${2:?Usage: bash poll_eval.sh <run_name> <epoch>}"
TARGET="/home/rohan/1_Projects/maas-knowledge-eval/eval_results/analyze_deepresearch/${RUN}/${EPOCH}/run_1.json"

echo "Polling for: ${TARGET}"
while true; do
    if [ -f "$TARGET" ]; then
        echo "$(date '+%H:%M:%S') FOUND: ${TARGET}"
        python3 -c "import json; d=json.load(open('${TARGET}')); print(f'accuracy: {d[\"summary\"][\"accuracy\"]:.4f}')"
        exit 0
    fi
    echo "$(date '+%H:%M:%S') not found, waiting 30s..."
    sleep 30
done
