#!/bin/bash
# Resume eval loop: skips checkpoints that already have eval_results.jsonl,
# continues past per-checkpoint failures (no set -e) logging them clearly.
# Run from the eval dir (cwd = where eval_maas_sdft.py + eval_results/ live).
export CLOUD_ML_REGION=${CLOUD_ML_REGION:-us-east5}
export ANTHROPIC_VERTEX_PROJECT_ID=${ANTHROPIC_VERTEX_PROJECT_ID:-itpc-gcp-ai-eng-claude}

VENV=${VENV:-./.venv}
TEST=${TEST:-/home/rohan/1_Projects/sdft_knowledge_ingestion_experiment/data/test_maas_sdft.jsonl}
BASE=${BASE:-./patched_ckpts}
STEPS=${STEPS:-"400 800 1200 1600 2000 2400 2800 3200 3600 4000"}
KINDS=${KINDS:-"sft osft"}
OUT_DIR=${OUT_DIR:-eval_results}

for step in $STEPS; do
  for kind in $KINDS; do
    out=${OUT_DIR}/${kind}/samples_${step}
    if [ -s "$out/eval_results.jsonl" ]; then
      echo "=== SKIP ${kind} samples_${step} (already done) ==="
      continue
    fi
    echo "=== ${kind} samples_${step} ==="
    CUDA_VISIBLE_DEVICES=0 $VENV/bin/python eval_maas_sdft.py \
      --model $BASE/${kind}/samples_${step} \
      --test_jsonl $TEST \
      --output_dir "$out"
    if [ $? -ne 0 ]; then
      echo "FAILED: ${kind} samples_${step}"
    fi
  done
done

echo "ALL EVALS DONE"
