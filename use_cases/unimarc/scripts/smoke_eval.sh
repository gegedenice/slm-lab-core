#!/usr/bin/env bash
set -euo pipefail

USE_CASE="unimarc"
BASE="{{cookiecutter.default_model_id}}" # This will need to be loaded from config
EVAL_DATA="data/eval/heldout.jsonl" # Relative to use-case dir

echo "[prep] building tiny split from HF (if configured)…"
make prep USE_CASE=$USE_CASE || true

echo "[train] quick LoRA run…"
python -m cli.finetune run $USE_CASE

echo "[eval] comparing baseline vs tuned…"
# Note: The evaluate script would also need to be adapted to be use-case aware
# For now, this is a placeholder
echo "python -m cli.evaluate run $BASE use_cases/$USE_CASE/runs/adapter use_cases/$USE_CASE/$EVAL_DATA"
