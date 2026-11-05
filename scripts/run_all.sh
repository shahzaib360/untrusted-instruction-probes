#!/usr/bin/env bash
# Reproduce the full pipeline for one model.
#   scripts/run_all.sh [HF_MODEL_ID] [RUN_NAME]
# Defaults reproduce the main results (Qwen2.5-1.5B-Instruct, ~25 min on 4 CPU cores).
set -euo pipefail
MODEL=${1:-Qwen/Qwen2.5-1.5B-Instruct}
NAME=${2:-qwen2.5-1.5b}

python -m uip.build_dataset --out data/prompts.jsonl
python -m uip.generate --model "$MODEL" --out-dir "runs/$NAME" --batch-size 32
python -m uip.verify_activations --run-dir "runs/$NAME" --model "$MODEL"
python -m uip.text_features --run-dir "runs/$NAME"
python -m uip.evaluate --run-dir "runs/$NAME" --out "results/$NAME"
python -m uip.plots --results "results/$NAME"
python -m uip.report --results "results/$NAME"
