#!/usr/bin/env bash
# Approved report-only resume. Never calls inference or training.
set -euo pipefail
ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
CODE="$ROOT/text_model/scripts/e2d"
WORK="$ROOT/runs/v365_e2d_workflow_20261014"
GATE="$ROOT/runs/v365_e2d_temporal_gate_s202"
EVAL="$ROOT/runs/v365_e2d_dev_compare_20261016_reportfix"
AUDIT="$ROOT/text_model/experiments/v365_e2d/audit/20261016_report_schema_fix"
PREREG="$ROOT/text_model/experiments/v365_e2d/e2d_preregistration.json"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/text_model:$CODE"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -e "$EVAL" || -e "$AUDIT/resume_started_at.txt" ]]; then
  echo "Refusing to overwrite report-only resume artifacts" >&2
  exit 2
fi
date --iso-8601=seconds > "$AUDIT/resume_started_at.txt"
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$AUDIT/resume_exit_code.txt"
  date --iso-8601=seconds > "$AUDIT/resume_finished_at.txt"
  echo "E2-D report-only resume exit=$rc"
}
trap finish EXIT
# The sole changed original code file is the approved report-schema repair.
awk '$2 !~ /\/report_e2d_dev_compare.py$/ {print}' \
  "$WORK/logs/frozen_code_prereg_sha256.txt" > "$AUDIT/unchanged_code_prereg.sha256"
sha256sum -c "$AUDIT/unchanged_code_prereg.sha256"
sha256sum -c "$WORK/logs/gate_sealed_before_dev_sha256.txt"
sha256sum -c "$WORK/logs/train_predictions_sha256.txt"
sha256sum "$CODE/report_e2d_dev_compare.py" "$CODE/resume_e2d_dev_report.sh" \
  > "$AUDIT/repaired_report_code.sha256"
timeout --signal=TERM --kill-after=10s 10m "$PY" "$CODE/report_e2d_dev_compare.py" \
  --dev-source "$ROOT/data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl" \
  --prefix-input "$ROOT/runs/v365_dev_baseline_compare_20261010/inputs/v365_dev_cumulative_prefix_label_blind.jsonl" \
  --long-dev-source "$ROOT/data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl" \
  --e2a-root "$ROOT/runs/v365_e2a_dev_compare_20261011" \
  --e2b-root "$ROOT/runs/v365_e2b_dev_compare_20261012" \
  --e2c-root "$ROOT/runs/v365_e2c_dev_compare_20261013" \
  --gate-checkpoint "$GATE/temporal_gate.pt" --preregistration "$PREREG" \
  --output-dir "$EVAL"
sha256sum -c "$AUDIT/unchanged_code_prereg.sha256"
sha256sum -c "$WORK/logs/gate_sealed_before_dev_sha256.txt"
sha256sum -c "$AUDIT/repaired_report_code.sha256"
find "$EVAL" -type f -print0 | sort -z | xargs -0 sha256sum > "$AUDIT/resumed_dev_output.sha256"
echo "E2-D completed with report-schema repair. Training was not repeated. Test remains locked."
