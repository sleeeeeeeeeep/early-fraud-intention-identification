#!/usr/bin/env bash
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
OUT="$ROOT/runs/v365_e2b_dev_compare_20261012"
E2A_ROOT="$ROOT/runs/v365_e2a_dev_compare_20261011"
BASELINE="$ROOT/runs/v365_dev_baseline_compare_20261010"
DEV_SOURCE="$ROOT/data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl"
FULL="$BASELINE/inputs/v365_dev_fulltext_label_blind.jsonl"
PREFIX="$BASELINE/inputs/v365_dev_cumulative_prefix_label_blind.jsonl"
LONG_DEV="$ROOT/data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl"
BASE_MODEL=models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
M0="$ROOT/runs/repro_v361b_b320_head96/best_model.pt"
M1="$ROOT/runs/v362_e1b_binary_60prefix_40replay_s202_conservative/best_model.pt"
E2A="$ROOT/runs/v365_e2a_binary_te60_replay40_s202/best_model.pt"
E2B="$ROOT/runs/v365_e2b_boundary_balanced_s202/best_model.pt"
PREREG_JSON="$ROOT/text_model/experiments/v365_e2b/e2b_preregistration.json"
INFERENCE_CODE="$ROOT/text_model/inference/run_longdoc_inference.py"
REPORT_CODE="$ROOT/text_model/scripts/evaluation/report_v365_e2b_compare.py"
BASELINE_REPORT_CODE="$ROOT/text_model/scripts/evaluation/report_v365_dev_baseline.py"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite E2-B evaluation output: $OUT" >&2
  exit 2
fi
for required in \
  "$DEV_SOURCE" "$FULL" "$PREFIX" "$LONG_DEV" "$BASE_MODEL/config.json" "$V341" \
  "$M0" "$M1" "$E2A" "$E2B" "$PREREG_JSON" "$INFERENCE_CODE" "$REPORT_CODE" \
  "$BASELINE_REPORT_CODE" "$E2A_ROOT/run_manifest.json" \
  "$E2A_ROOT/reports/model_comparison.json"; do
  test -e "$required" || {
    echo "Missing required E2-B evaluation input: $required" >&2
    exit 2
  }
  case "${required,,}" in
    *test*|*restricted*)
      echo "Refusing Test/restricted input: $required" >&2
      exit 3
      ;;
  esac
done
test -d "$E2A_ROOT/scored" || {
  echo "Missing audited E2-A scored directory: $E2A_ROOT/scored" >&2
  exit 2
}
case "${OUT,,}" in
  *test*|*restricted*)
    echo "Refusing Test/restricted output: $OUT" >&2
    exit 3
    ;;
esac

mkdir -p \
  "$OUT/logs" \
  "$OUT/raw_inference/e2b/fulltext" \
  "$OUT/raw_inference/e2b/cumulative_prefix" \
  "$OUT/raw_inference/e2b/long_dev"
date --iso-8601=seconds > "$OUT/logs/started_at.txt"
sha256sum \
  "$DEV_SOURCE" "$FULL" "$PREFIX" "$LONG_DEV" \
  "$M0" "$M1" "$E2A" "$E2B" "$PREREG_JSON" "$E2A_ROOT/run_manifest.json" \
  > "$OUT/logs/input_checkpoint_preregistration_sha256.txt"
sha256sum "$INFERENCE_CODE" "$REPORT_CODE" "$BASELINE_REPORT_CODE" \
  > "$OUT/logs/evaluation_code_sha256.txt"
find "$E2A_ROOT/scored" "$E2A_ROOT/reports" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT/logs/reused_e2a_artifacts_sha256.txt"

infer() {
  local input="$1"
  local checkpoint="$2"
  local destination="$3"
  local phase="$4"
  local log="$OUT/logs/${phase}.log"
  local rc_file="$OUT/logs/${phase}_exit_code.txt"

  set +e
  CUDA_VISIBLE_DEVICES=3 "$PY" -m text_model.inference.run_longdoc_inference \
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
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$rc_file"
  return "$rc"
}

# M0/M1/E2-A use the already audited results in E2A_ROOT.  Only E2-B receives
# new inference here, and every new process is fixed to physical GPU 3.
infer "$FULL" "$E2B" "$OUT/raw_inference/e2b/fulltext" \
  e2b_v365_fulltext || {
    rc=$?
    echo "E2-B V365 full-text inference failed; output retained at $OUT" >&2
    exit "$rc"
  }
infer "$PREFIX" "$E2B" "$OUT/raw_inference/e2b/cumulative_prefix" \
  e2b_v365_cumulative_prefix || {
    rc=$?
    echo "E2-B V365 prefix inference failed; output retained at $OUT" >&2
    exit "$rc"
  }
infer "$LONG_DEV" "$E2B" "$OUT/raw_inference/e2b/long_dev" \
  e2b_v361_long_dev || {
    rc=$?
    echo "E2-B V361 long-Dev inference failed; output retained at $OUT" >&2
    exit "$rc"
  }

sha256sum \
  "$OUT/raw_inference/e2b/fulltext/predictions.jsonl" \
  "$OUT/raw_inference/e2b/fulltext/metrics.json" \
  "$OUT/raw_inference/e2b/cumulative_prefix/predictions.jsonl" \
  "$OUT/raw_inference/e2b/cumulative_prefix/metrics.json" \
  "$OUT/raw_inference/e2b/long_dev/predictions.jsonl" \
  "$OUT/raw_inference/e2b/long_dev/metrics.json" \
  > "$OUT/logs/e2b_raw_inference_sha256.txt"

set +e
"$PY" "$REPORT_CODE" \
  --dev-source "$DEV_SOURCE" \
  --full-input "$FULL" \
  --prefix-input "$PREFIX" \
  --e2a-root "$E2A_ROOT" \
  --e2b-full-pred "$OUT/raw_inference/e2b/fulltext/predictions.jsonl" \
  --e2b-full-metrics "$OUT/raw_inference/e2b/fulltext/metrics.json" \
  --e2b-prefix-pred "$OUT/raw_inference/e2b/cumulative_prefix/predictions.jsonl" \
  --e2b-prefix-metrics "$OUT/raw_inference/e2b/cumulative_prefix/metrics.json" \
  --long-dev-source "$LONG_DEV" \
  --e2b-long-pred "$OUT/raw_inference/e2b/long_dev/predictions.jsonl" \
  --e2b-long-metrics "$OUT/raw_inference/e2b/long_dev/metrics.json" \
  --m0-checkpoint "$M0" \
  --m1-checkpoint "$M1" \
  --e2a-checkpoint "$E2A" \
  --e2b-checkpoint "$E2B" \
  --preregistration "$PREREG_JSON" \
  --inference-code "$INFERENCE_CODE" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/logs/report.log"
REPORT_RC=${PIPESTATUS[0]}
set -e
printf '%s\n' "$REPORT_RC" > "$OUT/logs/report_exit_code.txt"
if [[ "$REPORT_RC" -ne 0 ]]; then
  echo "E2-B comparison report failed with exit code $REPORT_RC; output retained at $OUT" >&2
  exit "$REPORT_RC"
fi

find "$OUT/reports" "$OUT/scored" "$OUT/raw_inference" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT/logs/evaluation_output_sha256.txt"
printf '0\n' > "$OUT/logs/evaluation_exit_code.txt"
date --iso-8601=seconds > "$OUT/logs/finished_at.txt"

echo "E2-B fixed Dev comparison finished at $(date --iso-8601=seconds): $OUT"
