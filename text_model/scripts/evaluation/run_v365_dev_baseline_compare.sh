#!/usr/bin/env bash
set -euo pipefail

PROJECT="${TELEFRAUD_ROOT:-$(pwd)}"
PYTHON="${TELEFRAUD_PYTHON:-python}"
OUT="$PROJECT/runs/v365_dev_baseline_compare_20261010"
DEV="$PROJECT/data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl"
BASE_MODEL="models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B"
V341="${V341_CHECKPOINT:-$PROJECT/models/v341/best_model/pytorch_model.bin}"
M0="$PROJECT/runs/repro_v361b_b320_head96/best_model.pt"
M1="$PROJECT/runs/v362_e1b_binary_60prefix_40replay_s202_conservative/best_model.pt"

cd "$PROJECT"
mkdir -p "$OUT/inputs" "$OUT/logs" "$OUT/raw_inference"

for required in "$DEV" "$BASE_MODEL/config.json" "$V341" "$M0" "$M1"; do
  test -e "$required" || { echo "Missing required file: $required" >&2; exit 2; }
done

stat -c '%n %s %Y' "$M0" "$M1" > "$OUT/logs/checkpoint_stat_before.txt"
df -h "$PROJECT" > "$OUT/logs/disk_before.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$OUT/logs/gpu_before.txt"

"$PYTHON" scripts/evaluation/build_v365_dev_eval_inputs.py \
  --source "$DEV" \
  --output-dir "$OUT/inputs" \
  > "$OUT/logs/build_inputs.log" 2>&1

FULL="$OUT/inputs/v365_dev_fulltext_label_blind.jsonl"
PREFIX="$OUT/inputs/v365_dev_cumulative_prefix_label_blind.jsonl"

run_model() {
  local gpu="$1"
  local model_id="$2"
  local checkpoint="$3"
  local model_out="$OUT/raw_inference/$model_id"
  mkdir -p "$model_out/fulltext" "$model_out/cumulative_prefix"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m text_model.inference.run_longdoc_inference \
    --input-jsonl "$FULL" \
    --model "$BASE_MODEL" \
    --v341-checkpoint "$V341" \
    --longdoc-checkpoint "$checkpoint" \
    --route B \
    --output-dir "$model_out/fulltext" \
    --input-mode head_context_enhanced_chunks \
    --chunk-length 320 \
    --overlap 64 \
    --head-context-length 96 \
    --encoder-micro-batch-size 12 \
    --evidence-components attention,top1,top2,last,mean \
    --batch-size 2 \
    > "$OUT/logs/${model_id}_fulltext.log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m text_model.inference.run_longdoc_inference \
    --input-jsonl "$PREFIX" \
    --model "$BASE_MODEL" \
    --v341-checkpoint "$V341" \
    --longdoc-checkpoint "$checkpoint" \
    --route B \
    --output-dir "$model_out/cumulative_prefix" \
    --input-mode head_context_enhanced_chunks \
    --chunk-length 320 \
    --overlap 64 \
    --head-context-length 96 \
    --encoder-micro-batch-size 12 \
    --evidence-components attention,top1,top2,last,mean \
    --batch-size 2 \
    > "$OUT/logs/${model_id}_cumulative_prefix.log" 2>&1
}

# One model per GPU.  Each model evaluates the same full and prefix files with
# the same code and hyperparameters; only the long-document checkpoint differs.
run_model 0 m0_v361 "$M0" &
PID_M0=$!
run_model 1 m1_e1b "$M1" &
PID_M1=$!
wait "$PID_M0"
wait "$PID_M1"

"$PYTHON" scripts/evaluation/report_v365_dev_baseline.py \
  --dev-source "$DEV" \
  --prefix-input "$PREFIX" \
  --m0-full-pred "$OUT/raw_inference/m0_v361/fulltext/predictions.jsonl" \
  --m0-full-metrics "$OUT/raw_inference/m0_v361/fulltext/metrics.json" \
  --m0-prefix-pred "$OUT/raw_inference/m0_v361/cumulative_prefix/predictions.jsonl" \
  --m0-prefix-metrics "$OUT/raw_inference/m0_v361/cumulative_prefix/metrics.json" \
  --m1-full-pred "$OUT/raw_inference/m1_e1b/fulltext/predictions.jsonl" \
  --m1-full-metrics "$OUT/raw_inference/m1_e1b/fulltext/metrics.json" \
  --m1-prefix-pred "$OUT/raw_inference/m1_e1b/cumulative_prefix/predictions.jsonl" \
  --m1-prefix-metrics "$OUT/raw_inference/m1_e1b/cumulative_prefix/metrics.json" \
  --m0-checkpoint "$M0" \
  --m1-checkpoint "$M1" \
  --output-dir "$OUT" \
  > "$OUT/logs/report.log" 2>&1

stat -c '%n %s %Y' "$M0" "$M1" > "$OUT/logs/checkpoint_stat_after.txt"
cmp "$OUT/logs/checkpoint_stat_before.txt" "$OUT/logs/checkpoint_stat_after.txt"

if find "$OUT" -type f \( -name '*.pt' -o -name '*.bin' -o -name '*.safetensors' \) -print -quit | grep -q .; then
  echo "Unexpected model-weight file under output directory" >&2
  exit 3
fi

df -h "$PROJECT" > "$OUT/logs/disk_after.txt"
echo "DONE: $OUT"
