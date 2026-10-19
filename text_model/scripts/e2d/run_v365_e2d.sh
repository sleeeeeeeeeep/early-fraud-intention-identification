#!/usr/bin/env bash
set -euo pipefail
ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
CODE="$ROOT/text_model/scripts/e2d"
PREREG="$ROOT/text_model/experiments/v365_e2d/e2d_preregistration.json"
DATA="$ROOT/data/processed/v365_e2d_gate_train_s202"
WORK="$ROOT/runs/v365_e2d_workflow_20261014"
GATE="$ROOT/runs/v365_e2d_temporal_gate_s202"
EVAL="$ROOT/runs/v365_e2d_dev_compare_20261014"
M1="$ROOT/runs/v362_e1b_binary_60prefix_40replay_s202_conservative/best_model.pt"
BASE="${BASE_MODEL:-$ROOT/models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B}"
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/text_model:$CODE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
for target in "$WORK" "$GATE" "$EVAL"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite E2-D output: $target" >&2
    exit 2
  fi
done
mkdir -p "$WORK/logs" "$WORK/frozen_m1_train_predictions"
date --iso-8601=seconds > "$WORK/logs/started_at.txt"
finish() {
  local rc=$?
  printf '%s\n' "$rc" > "$WORK/logs/workflow_exit_code.txt"
  date --iso-8601=seconds > "$WORK/logs/finished_at.txt"
  echo "E2-D workflow exit=$rc"
}
trap finish EXIT
sha256sum "$CODE"/*.py "$CODE/run_v365_e2d.sh" "$PREREG" \
  > "$WORK/logs/frozen_code_prereg_sha256.txt"
phase() {
  local name="$1"
  shift
  date --iso-8601=seconds > "$WORK/logs/${name}_started_at.txt"
  set +e
  "$@" 2>&1 | tee "$WORK/logs/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$WORK/logs/${name}_exit_code.txt"
  date --iso-8601=seconds > "$WORK/logs/${name}_finished_at.txt"
  return "$rc"
}
if [[ ! -d "$DATA" ]]; then
  phase build_inputs "$PY" "$CODE/build_e2d_train_inputs.py" \
    --v365-train "$ROOT/data/processed/v365_2000_final_20260923/data/v365_calls_train_final.jsonl" \
    --output-dir "$DATA"
fi
phase input_preflight "$PY" -c \
  'import json,sys; from pathlib import Path; from e2d_gate_common import sha256_file; p=Path(sys.argv[1]); m=json.load(open(p/"input_manifest.json")); assert m["validation"]["pairs"]==4633; assert m["validation"]["unique_prefix_endpoints"]==8258; assert m["safety"]["dev_files_read"]==0 and m["safety"]["test_or_restricted_files_read"]==0; assert all(sha256_file(Path(x["path"]))==x["sha256"] for x in m["outputs"].values()); print("Train-only input integrity PASS")' "$DATA"
phase checkpoint_integrity "$PY" -c \
  'import sys; from e2d_gate_common import sha256_file,M1_SHA256; h=sha256_file(sys.argv[1]); print(h,sys.argv[1],flush=True); assert h==M1_SHA256' "$M1"
phase frozen_m1_features env CUDA_VISIBLE_DEVICES=0 "$PY" "$CODE/run_e2d_frozen_inference.py" \
  --input-jsonl "$DATA/train_prefix_inputs_label_blind.jsonl" \
  --model "$BASE" --v341-checkpoint "$V341" --longdoc-checkpoint "$M1" \
  --route B --output-dir "$WORK/frozen_m1_train_predictions" \
  --input-mode head_context_enhanced_chunks --chunk-length 320 --overlap 64 \
  --head-context-length 96 --encoder-micro-batch-size 12 \
  --evidence-components attention,top1,top2,last,mean --batch-size 2
sha256sum "$WORK/frozen_m1_train_predictions"/*.json* > "$WORK/logs/train_predictions_sha256.txt"
phase gate_training "$PY" "$CODE/train_e2d_gate.py" \
  --data-dir "$DATA" --prediction-dir "$WORK/frozen_m1_train_predictions" \
  --preregistration "$PREREG" --output-dir "$GATE"
sha256sum "$GATE/temporal_gate.pt" "$GATE/training_manifest.json" > "$WORK/logs/gate_sealed_before_dev_sha256.txt"
phase dev_comparison "$PY" "$CODE/report_e2d_dev_compare.py" \
  --dev-source "$ROOT/data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl" \
  --prefix-input "$ROOT/runs/v365_dev_baseline_compare_20261010/inputs/v365_dev_cumulative_prefix_label_blind.jsonl" \
  --long-dev-source "$ROOT/data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl" \
  --e2a-root "$ROOT/runs/v365_e2a_dev_compare_20261011" \
  --e2b-root "$ROOT/runs/v365_e2b_dev_compare_20261012" \
  --e2c-root "$ROOT/runs/v365_e2c_dev_compare_20261013" \
  --gate-checkpoint "$GATE/temporal_gate.pt" --preregistration "$PREREG" --output-dir "$EVAL"
find "$EVAL" -type f -print0 | sort -z | xargs -0 sha256sum > "$WORK/logs/dev_output_sha256.txt"
sha256sum -c "$WORK/logs/frozen_code_prereg_sha256.txt"
sha256sum -c "$WORK/logs/gate_sealed_before_dev_sha256.txt"
echo "E2-D complete. Test remains locked."
