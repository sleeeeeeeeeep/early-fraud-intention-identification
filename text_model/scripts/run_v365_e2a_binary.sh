#!/usr/bin/env bash
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
TRAIN="$ROOT/data/processed/v365_e2a_te_prefix_60new_40replay_s202/train_e2a_binary_6500.jsonl"
OUT="$ROOT/runs/v365_e2a_binary_te60_replay40_s202"
MODEL="${BASE_MODEL:-$ROOT/models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B}"
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
M0="$ROOT/runs/repro_v361b_b320_head96/best_model.pt"
LAUNCH_LOG="$ROOT/runs/v365_e2a_binary_te60_replay40_s202.training.log"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite E2-A output: $OUT" >&2
  exit 2
fi
for required in "$TRAIN" "$MODEL/config.json" "$V341" "$M0"; do
  test -e "$required" || { echo "Missing required file: $required" >&2; exit 2; }
done

CUDA_VISIBLE_DEVICES=2 "$PY" text_model/scripts/training/train_v365_e2a_binary.py \
  --train-jsonl "$TRAIN" \
  --model "$MODEL" \
  --v341-checkpoint "$V341" \
  --init-longdoc-checkpoint "$M0" \
  --output "$OUT" \
  --route B \
  --input-mode head_context_enhanced_chunks \
  --chunk-length 320 \
  --overlap 64 \
  --head-context-length 96 \
  --max-chunks 12 \
  --encoder-micro-batch-size 12 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --freeze-backbone-epochs 1 \
  --top-layers 4 \
  --encoder-learning-rate 0 \
  --task-learning-rate 2e-6 \
  --weight-decay 0.01 \
  --warmup-ratio 0.05 \
  --gradient-clip-norm 1.0 \
  --seed 202 \
  --amp \
  --progress-every 200 \
  > "$LAUNCH_LOG" 2>&1

mkdir -p "$OUT/logs"
cp "$LAUNCH_LOG" "$OUT/logs/training.log"
printf '0\n' > "$OUT/logs/training_exit_code.txt"
echo "E2-A finished at $(date --iso-8601=seconds): $OUT"
