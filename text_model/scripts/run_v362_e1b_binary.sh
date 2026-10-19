#!/usr/bin/env bash
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
DATA="$ROOT/data/processed/v362_e1b_binary_60prefix_40replay"
OUT="$ROOT/runs/v362_e1b_binary_60prefix_40replay_s202_conservative"
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
INIT="$ROOT/runs/repro_v361b_b320_head96/best_model.pt"
MODEL="${BASE_MODEL:-$ROOT/models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B}"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing E1-B output: $OUT" >&2
  exit 2
fi
mkdir -p "$DATA/source_inputs"

"$PY" text_model/scripts/data/build_v362_strict_prefix_inputs.py \
  --splits train \
  --output-dir "$DATA/source_inputs"

"$PY" text_model/scripts/data/build_v362_e1b_mixed_train.py \
  --prefix-jsonl "$DATA/source_inputs/strict_prefix_train.jsonl" \
  --replay-jsonl "$ROOT/data/processed/v361_longdoc_text_route_20260824/train_v361_longdoc_6500.jsonl" \
  --output-jsonl "$DATA/train_e1b_binary_6500.jsonl" \
  --total-rows 6500 \
  --prefix-fraction 0.60 \
  --seed 202

CUDA_VISIBLE_DEVICES=2 "$PY" text_model/scripts/training/train_v362_e1b_binary.py \
  --train-jsonl "$DATA/train_e1b_binary_6500.jsonl" \
  --prefix-dev-jsonl "$ROOT/runs/v362_e0_binary/inputs/strict_prefix_dev.jsonl" \
  --fulltext-dev-jsonl "$ROOT/data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl" \
  --model "$MODEL" \
  --v341-checkpoint "$V341" \
  --init-longdoc-checkpoint "$INIT" \
  --output "$OUT" \
  --route B \
  --input-mode head_context_enhanced_chunks \
  --chunk-length 320 \
  --overlap 64 \
  --head-context-length 96 \
  --max-chunks 12 \
  --encoder-micro-batch-size 12 \
  --batch-size 1 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --patience 1 \
  --freeze-backbone-epochs 1 \
  --top-layers 4 \
  --encoder-learning-rate 0 \
  --task-learning-rate 2e-6 \
  --fulltext-accuracy-floor 0.99 \
  --prefix-fpr-ceiling 0.05 \
  --seed 202 \
  --amp \
  --progress-every 200

echo "E1-B finished at $(date --iso-8601=seconds): $OUT"
