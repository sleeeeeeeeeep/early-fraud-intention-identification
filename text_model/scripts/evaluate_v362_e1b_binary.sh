#!/usr/bin/env bash
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
RUN="$ROOT/runs/v362_e1b_binary_60prefix_40replay_s202_conservative"
CKPT="$RUN/best_model.pt"
INPUTS="$ROOT/runs/v362_e0_binary/inputs"
SOURCE="$ROOT/data/processed/v361_longdoc_text_route_20260824"
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
MODEL="${BASE_MODEL:-$ROOT/models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B}"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"
test -f "$CKPT"
mkdir -p "$RUN/eval_logs"
for view in fulltext_dev fulltext_stress strict_prefix_dev strict_prefix_stress cumulative_prefix_dev cumulative_prefix_stress; do
  if [[ -e "$RUN/predictions/$view" ]]; then
    echo "Refusing to overwrite evaluation view: $RUN/predictions/$view" >&2
    exit 2
  fi
done

run_one() {
  local view="$1" input="$2" gpu="$3"
  echo "[$(date --iso-8601=seconds)] START $view gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" text_model/inference/run_longdoc_inference.py \
    --input-jsonl "$input" \
    --model "$MODEL" \
    --v341-checkpoint "$V341" \
    --longdoc-checkpoint "$CKPT" \
    --route B \
    --output-dir "$RUN/predictions/$view" \
    --input-mode head_context_enhanced_chunks \
    --chunk-length 320 \
    --overlap 64 \
    --head-context-length 96 \
    --encoder-micro-batch-size 12 \
    --evidence-components attention,top1,top2,last,mean \
    --batch-size 2 \
    > "$RUN/eval_logs/${view}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] DONE  $view"
}

run_one fulltext_dev "$SOURCE/long_dev_v361_240.jsonl" 0 & p0=$!
run_one fulltext_stress "$SOURCE/long_stress_v361_120.jsonl" 1 & p1=$!
run_one strict_prefix_dev "$INPUTS/strict_prefix_dev.jsonl" 2 & p2=$!
run_one strict_prefix_stress "$INPUTS/strict_prefix_stress.jsonl" 3 & p3=$!
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"

run_one cumulative_prefix_dev "$INPUTS/cumulative_prefix_dev.jsonl" 2 & p2=$!
run_one cumulative_prefix_stress "$INPUTS/cumulative_prefix_stress.jsonl" 3 & p3=$!
wait "$p2"; wait "$p3"

"$PY" text_model/scripts/data/report_v362_binary_e0.py \
  --run-dir "$RUN" \
  --report-name E1B_BINARY \
  --schema-stage e1b \
  --checkpoint "$CKPT"

echo "E1-B evaluation complete: $RUN"
