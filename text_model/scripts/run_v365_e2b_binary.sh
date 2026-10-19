#!/usr/bin/env bash
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
E2A_DATA="$ROOT/data/processed/v365_e2a_te_prefix_60new_40replay_s202"
E2A_UNIQUE="$E2A_DATA/v365_train_te_prefix_unique_1200.jsonl"
E2A_MIXTURE="$E2A_DATA/train_e2a_binary_6500.jsonl"
E2A_MANIFEST="$E2A_DATA/manifest.json"
V365_TRAIN="$ROOT/data/processed/v365_2000_final_20260923/data/v365_calls_train_final.jsonl"
DATA_OUT="$ROOT/data/processed/v365_e2b_boundary_balanced_60new_40replay_s202"
TRAIN="$DATA_OUT/train_e2b_binary_6500.jsonl"
OUT="$ROOT/runs/v365_e2b_boundary_balanced_s202"
META="$ROOT/runs/v365_e2b_boundary_balanced_s202.launch_metadata"
MODEL="${BASE_MODEL:-$ROOT/models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B}"
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
M0="$ROOT/runs/repro_v361b_b320_head96/best_model.pt"
PREREG_DIR="$ROOT/text_model/experiments/v365_e2b"
PREREG_MD="$PREREG_DIR/E2B_PREREGISTRATION.md"
PREREG_JSON="$PREREG_DIR/e2b_preregistration.json"
BUILD_CODE="$ROOT/text_model/scripts/data/build_v365_e2b_boundary_mixture.py"
TRAIN_CODE="$ROOT/text_model/scripts/training/train_v365_e2b_binary.py"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"

for target in "$DATA_OUT" "$OUT" "$META"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite E2-B artifact: $target" >&2
    exit 2
  fi
done
for required in \
  "$E2A_UNIQUE" "$E2A_MIXTURE" "$E2A_MANIFEST" "$V365_TRAIN" \
  "$MODEL/config.json" "$V341" "$M0" "$PREREG_MD" "$PREREG_JSON" \
  "$BUILD_CODE" "$TRAIN_CODE"; do
  test -e "$required" || {
    echo "Missing required E2-B input: $required" >&2
    exit 2
  }
  case "${required,,}" in
    *test*|*restricted*)
      echo "Refusing Test/restricted input: $required" >&2
      exit 3
      ;;
  esac
done

mkdir -p "$META"
date --iso-8601=seconds > "$META/started_at.txt"
df -h "$ROOT/runs" > "$META/disk_before.txt"
sha256sum "$PREREG_MD" "$PREREG_JSON" \
  > "$META/preregistration_sha256.txt"
sha256sum "$BUILD_CODE" "$TRAIN_CODE" \
  > "$META/code_sha256.txt"
sha256sum "$E2A_UNIQUE" "$E2A_MIXTURE" "$E2A_MANIFEST" "$V365_TRAIN" "$M0" \
  > "$META/source_input_sha256.txt"

set +e
"$PY" "$BUILD_CODE" \
  --e2a-unique "$E2A_UNIQUE" \
  --e2a-mixture "$E2A_MIXTURE" \
  --e2a-manifest "$E2A_MANIFEST" \
  --v365-train "$V365_TRAIN" \
  --tokenizer "$MODEL" \
  --output-dir "$DATA_OUT" \
  --total-rows 6500 \
  --new-prefix-fraction 0.60 \
  --seed 202 \
  2>&1 | tee "$META/data_build.log"
BUILD_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$BUILD_RC" > "$META/data_build_exit_code.txt"
if [[ "$BUILD_RC" -ne 0 ]]; then
  echo "E2-B data construction failed with exit code $BUILD_RC; logs kept at $META" >&2
  exit "$BUILD_RC"
fi

find "$DATA_OUT" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$META/data_output_sha256.txt"

set +e
CUDA_VISIBLE_DEVICES=2 "$PY" "$TRAIN_CODE" \
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
  --boundary-margin 0.5 \
  --boundary-rank-weight 0.20 \
  --pre-alarm-cap-weight 0.10 \
  --seed 202 \
  --amp \
  --progress-every 200 \
  2>&1 | tee "$META/training.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$TRAIN_RC" > "$META/training_exit_code.txt"
if [[ "$TRAIN_RC" -ne 0 ]]; then
  echo "E2-B training failed with exit code $TRAIN_RC; logs kept at $META" >&2
  exit "$TRAIN_RC"
fi

sha256sum "$OUT/best_model.pt" "$OUT/manifest.json" "$OUT/metrics.json" \
  > "$META/training_output_sha256.txt"
date --iso-8601=seconds > "$META/finished_at.txt"
df -h "$ROOT/runs" > "$META/disk_after.txt"
mv "$META" "$OUT/logs"

echo "E2-B data construction and training finished at $(date --iso-8601=seconds): $OUT"
