#!/usr/bin/env bash
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
OUT="$ROOT/runs/v365_e2a_dev_compare_20261011"
BASELINE="$ROOT/runs/v365_dev_baseline_compare_20261010"
FULL="$BASELINE/inputs/v365_dev_fulltext_label_blind.jsonl"
PREFIX="$BASELINE/inputs/v365_dev_cumulative_prefix_label_blind.jsonl"
LONG_DEV="$ROOT/data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl"
BASE_MODEL=models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
M0="$ROOT/runs/repro_v361b_b320_head96/best_model.pt"
M1="$ROOT/runs/v362_e1b_binary_60prefix_40replay_s202_conservative/best_model.pt"
E2="$ROOT/runs/v365_e2a_binary_te60_replay40_s202/best_model.pt"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite E2-A evaluation output: $OUT" >&2
  exit 2
fi
for required in "$FULL" "$PREFIX" "$LONG_DEV" "$BASE_MODEL/config.json" "$V341" "$M0" "$M1" "$E2"; do
  test -e "$required" || { echo "Missing required file: $required" >&2; exit 2; }
  case "${required,,}" in
    *test*|*restricted*) echo "Refusing Test/restricted input: $required" >&2; exit 3 ;;
  esac
done
mkdir -p "$OUT/logs" "$OUT/raw_inference/e2a/fulltext" \
  "$OUT/raw_inference/e2a/cumulative_prefix"

sha256sum "$FULL" "$PREFIX" "$LONG_DEV" "$M0" "$M1" "$E2" \
  > "$OUT/logs/input_checkpoint_sha256.txt"
sha256sum text_model/inference/run_longdoc_inference.py \
  > "$OUT/logs/inference_code_sha256.txt"

infer() {
  local gpu="$1"
  local input="$2"
  local checkpoint="$3"
  local destination="$4"
  local log="$5"
  mkdir -p "$destination"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m text_model.inference.run_longdoc_inference \
    --input-jsonl "$input" \
    --model "$BASE_MODEL" \
    --v341-checkpoint "$V341" \
    --longdoc-checkpoint "$checkpoint" \
    --route B \
    --output-dir "$destination" \
    --input-mode head_context_enhanced_chunks \
    --chunk-length 320 \
    --overlap 64 \
    --head-context-length 96 \
    --encoder-micro-batch-size 12 \
    --evidence-components attention,top1,top2,last,mean \
    --batch-size 2 \
    > "$log" 2>&1
}

# The fixed V365 label-blind inputs were already evaluated for M0 and M1 by the
# same unchanged inference program.  Only E2-A needs new inference here.
infer 3 "$FULL" "$E2" "$OUT/raw_inference/e2a/fulltext" \
  "$OUT/logs/e2a_v365_fulltext.log"
infer 3 "$PREFIX" "$E2" "$OUT/raw_inference/e2a/cumulative_prefix" \
  "$OUT/logs/e2a_v365_cumulative_prefix.log"

# Re-run all three on the original long-text Dev in one batch for the strictest
# binary-only comparison.  Each process receives a dedicated GPU.
infer 0 "$LONG_DEV" "$M0" "$OUT/raw_inference/m0/long_dev" \
  "$OUT/logs/m0_long_dev.log" &
PID_M0=$!
infer 1 "$LONG_DEV" "$M1" "$OUT/raw_inference/m1/long_dev" \
  "$OUT/logs/m1_long_dev.log" &
PID_M1=$!
infer 3 "$LONG_DEV" "$E2" "$OUT/raw_inference/e2a/long_dev" \
  "$OUT/logs/e2a_long_dev.log" &
PID_E2=$!
wait "$PID_M0"
wait "$PID_M1"
wait "$PID_E2"

"$PY" text_model/scripts/evaluation/report_v365_e2a_compare.py \
  --dev-source "$ROOT/data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl" \
  --prefix-input "$PREFIX" \
  --baseline-root "$BASELINE" \
  --e2-full-pred "$OUT/raw_inference/e2a/fulltext/predictions.jsonl" \
  --e2-full-metrics "$OUT/raw_inference/e2a/fulltext/metrics.json" \
  --e2-prefix-pred "$OUT/raw_inference/e2a/cumulative_prefix/predictions.jsonl" \
  --e2-prefix-metrics "$OUT/raw_inference/e2a/cumulative_prefix/metrics.json" \
  --long-dev-source "$LONG_DEV" \
  --m0-long-pred "$OUT/raw_inference/m0/long_dev/predictions.jsonl" \
  --m0-long-metrics "$OUT/raw_inference/m0/long_dev/metrics.json" \
  --m1-long-pred "$OUT/raw_inference/m1/long_dev/predictions.jsonl" \
  --m1-long-metrics "$OUT/raw_inference/m1/long_dev/metrics.json" \
  --e2-long-pred "$OUT/raw_inference/e2a/long_dev/predictions.jsonl" \
  --e2-long-metrics "$OUT/raw_inference/e2a/long_dev/metrics.json" \
  --m0-checkpoint "$M0" \
  --m1-checkpoint "$M1" \
  --e2-checkpoint "$E2" \
  --output-dir "$OUT" \
  > "$OUT/logs/report.log" 2>&1

printf '0\n' > "$OUT/logs/evaluation_exit_code.txt"
echo "E2-A Dev comparison finished at $(date --iso-8601=seconds): $OUT"
