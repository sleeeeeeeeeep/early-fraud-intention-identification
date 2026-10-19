#!/usr/bin/env bash
# Final V362 E0: dedicated binary detection head only.
set -euo pipefail

ROOT="${TELEFRAUD_ROOT:-$(pwd)}"
PY="${TELEFRAUD_PYTHON:-python}"
OUT="$ROOT/runs/v362_e0_binary"
V341="${V341_CHECKPOINT:-$ROOT/models/v341/best_model/pytorch_model.bin}"
CKPT="$ROOT/runs/repro_v361b_b320_head96/best_model.pt"
MODEL="${BASE_MODEL:-$ROOT/models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B}"
SOURCE="$ROOT/data/processed/v361_longdoc_text_route_20260824"

cd "$ROOT"
export PYTHONPATH="$ROOT/text_model"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing final E0 directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT/inputs" "$OUT/predictions" "$OUT/logs" "$OUT/reports"

"$PY" text_model/scripts/data/build_v362_strict_prefix_inputs.py \
  --output-dir "$OUT/inputs" | tee "$OUT/logs/build_strict_prefix_inputs.log"
"$PY" text_model/scripts/data/build_v362_cumulative_prefix_inputs.py \
  --output-dir "$OUT/inputs" | tee "$OUT/logs/build_cumulative_prefix_inputs.log"

{
  echo "task=v362_e0_binary_head"
  echo "created_at=$(date --iso-8601=seconds)"
  echo "python=$PY"
  "$PY" --version
  echo "checkpoint=$CKPT"
  echo "checkpoint_sha256=$(sha256sum "$CKPT" | awk '{print $1}')"
  echo "v341_checkpoint=$V341"
  echo "v341_checkpoint_sha256=$(sha256sum "$V341" | awk '{print $1}')"
  echo "score=p_binary[1]=softmax(binary_logits)[1]"
  echo "route=B"
  echo "input_mode=head_context_enhanced_chunks"
  echo "chunk_length=320"
  echo "overlap=64"
  echo "head_context_length=96"
  echo "batch_size=2"
} > "$OUT/PROVENANCE.txt"

run_one() {
  local view="$1"
  local input="$2"
  local gpu="$3"
  local output="$OUT/predictions/$view"
  echo "[$(date --iso-8601=seconds)] START $view gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" text_model/inference/run_longdoc_inference.py \
    --input-jsonl "$input" \
    --model "$MODEL" \
    --v341-checkpoint "$V341" \
    --longdoc-checkpoint "$CKPT" \
    --route B \
    --output-dir "$output" \
    --input-mode head_context_enhanced_chunks \
    --chunk-length 320 \
    --overlap 64 \
    --head-context-length 96 \
    --encoder-micro-batch-size 12 \
    --evidence-components attention,top1,top2,last,mean \
    --batch-size 2 \
    > "$OUT/logs/${view}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] DONE  $view"
}

# Four endpoint views.  Each process owns one GPU.
run_one fulltext_dev "$SOURCE/long_dev_v361_240.jsonl" 0 & p0=$!
run_one fulltext_stress "$SOURCE/long_stress_v361_120.jsonl" 1 & p1=$!
run_one strict_prefix_dev "$OUT/inputs/strict_prefix_dev.jsonl" 2 & p2=$!
run_one strict_prefix_stress "$OUT/inputs/strict_prefix_stress.jsonl" 3 & p3=$!
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"

# Cumulative-prefix time curves.  Dev and stress are independent.
run_one cumulative_prefix_dev "$OUT/inputs/cumulative_prefix_dev.jsonl" 2 & p2=$!
run_one cumulative_prefix_stress "$OUT/inputs/cumulative_prefix_stress.jsonl" 3 & p3=$!
wait "$p2"; wait "$p3"

"$PY" text_model/scripts/data/report_v362_binary_e0.py \
  --run-dir "$OUT" | tee "$OUT/logs/report_binary_e0.log"

echo "[$(date --iso-8601=seconds)] V362 binary E0 complete: $OUT"
