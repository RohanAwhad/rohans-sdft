#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .logprob_venv/bin/activate

# Run e2e test
GPU_TRAINER=${GPU_TRAINER:-2} GPU_SERVER=${GPU_SERVER:-3} python test_e2e.py
