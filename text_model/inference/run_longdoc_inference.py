#!/usr/bin/env python3
"""长文本路线与固定窗口聚合对照的推理入口。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from data.longdoc_chunker import ChunkConfig, HEAD_MODE, PLAIN_MODE
    from data.longdoc_collator import LongDocCollator
    from inference.train_longdoc import RowsDataset, child_mapping, read_jsonl, write_jsonl
    from models.fixed_logit_aggregator import SUPPORTED_FIXED_POOLING, aggregate_window_logits
    from models.longdoc_multitask_classifier import build_longdoc_model
    from training.longdoc_metrics import classification_metrics
except ImportError:
    from text_model.data.longdoc_chunker import ChunkConfig, HEAD_MODE, PLAIN_MODE
    from text_model.data.longdoc_collator import LongDocCollator
    from text_model.inference.train_longdoc import RowsDataset, child_mapping, read_jsonl, write_jsonl
    from text_model.models.fixed_logit_aggregator import SUPPORTED_FIXED_POOLING, aggregate_window_logits
    from text_model.models.longdoc_multitask_classifier import build_longdoc_model
    from text_model.training.longdoc_metrics import classification_metrics


FIXED_CONTROL_MODES = ("max", "top2_mean", "logsumexp")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "label_13" not in rows[0]:
        raise ValueError(f"invalid evaluation CSV: {path}")
    for index, row in enumerate(rows):
        row["sample_id"] = str(row.get("sample_id") or row.get("id") or index)
    return rows


def _load_longdoc_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(state, strict=True)
    return {
        "checkpoint": str(path),
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--v341-checkpoint", type=Path, required=True)
    parser.add_argument("--longdoc-checkpoint", type=Path)
    parser.add_argument("--route", choices=("A", "B"), default="B")
    parser.add_argument(
        "--fixed-pooling",
        choices=sorted(SUPPORTED_FIXED_POOLING | {"all"}),
        help="固定窗口聚合方式；all 会复用一次窗口编码并输出三种工程对照。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=(PLAIN_MODE, HEAD_MODE), default=PLAIN_MODE)
    parser.add_argument("--chunk-length", type=int, choices=(256, 320, 384), default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--head-context-length", type=int, choices=(48, 64, 96), default=64)
    parser.add_argument("--encoder-micro-batch-size", type=int, default=12)
    parser.add_argument(
        "--evidence-components", default="attention,top1,top2,last,mean"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-final-test-b", action="store_true")
    return parser.parse_args()


def resolve_fixed_pooling_modes(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if value == "all":
        return FIXED_CONTROL_MODES
    if value not in SUPPORTED_FIXED_POOLING:
        raise ValueError(f"unsupported fixed pooling mode: {value}")
    return (value,)


def summarize_predictions(
    predictions: list[dict[str, Any]],
    latencies_ms: list[float],
    *,
    chunk_length: int,
    peak_gpu_memory_mb: float,
    wall_time_seconds: float,
) -> dict[str, Any]:
    # V362 E0 only evaluates the dedicated two-class detection head.  The
    # 13-class joint output is retained in each prediction record strictly as
    # auxiliary/debug information; it must not be used as the fraud score.
    labels = [int(int(row["label_13"]) != 0) for row in predictions]
    predicted = [int(row["prediction_binary"]) for row in predictions]
    metrics = classification_metrics(labels, predicted, 2)
    metrics["task"] = "binary_fraud_detection"
    metrics["probability_field"] = "p_binary[1]"
    metrics["coverage_min"] = min(
        (float(row["full_text_coverage"]) for row in predictions), default=0.0
    )
    ordered_latency = sorted(latencies_ms)
    metrics["latency_ms_mean"] = (
        statistics.fmean(latencies_ms) if latencies_ms else 0.0
    )
    metrics["latency_ms_p50"] = (
        ordered_latency[round((len(ordered_latency) - 1) * 0.50)]
        if ordered_latency
        else 0.0
    )
    metrics["latency_ms_p95"] = (
        ordered_latency[round((len(ordered_latency) - 1) * 0.95)]
        if ordered_latency
        else 0.0
    )
    metrics["peak_gpu_memory_mb"] = peak_gpu_memory_mb
    metrics["wall_time_seconds"] = wall_time_seconds
    metrics["throughput_samples_per_second"] = (
        len(predictions) / wall_time_seconds if wall_time_seconds > 0 else 0.0
    )
    return metrics


@torch.no_grad()
def main() -> int:
    args = parse_args()
    fixed_modes = resolve_fixed_pooling_modes(args.fixed_pooling)
    lowered = args.input_jsonl.name.lower().replace("-", "_")
    is_test_b = "test_b" in lowered or "testb" in lowered
    if is_test_b and not args.allow_final_test_b:
        raise SystemExit("Test-B requires --allow-final-test-b after configuration freeze")
    rows = read_rows(args.input_jsonl)
    evidence_components = tuple(
        part.strip() for part in args.evidence_components.split(",") if part.strip()
    )
    from transformers import AutoTokenizer

    tokenizer_source = args.v341_checkpoint.parent
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    config = ChunkConfig(
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        input_mode=args.input_mode,
        head_context_length=args.head_context_length,
    )
    collator = LongDocCollator(tokenizer, config, child_code_to_index=child_mapping())
    loader = DataLoader(
        RowsDataset(rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    model = build_longdoc_model(
        args.model,
        route=args.route,
        encoder_micro_batch_size=args.encoder_micro_batch_size,
        evidence_components=evidence_components,
        v341_checkpoint=args.v341_checkpoint,
    )
    checkpoint_report = None
    if args.longdoc_checkpoint is not None:
        checkpoint_report = _load_longdoc_checkpoint(model, args.longdoc_checkpoint)
    if not fixed_modes and args.longdoc_checkpoint is None:
        raise SystemExit("route inference requires --longdoc-checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    output_modes = fixed_modes or (f"route_{args.route.lower()}",)
    predictions_by_mode: dict[str, list[dict[str, Any]]] = {
        mode: [] for mode in output_modes
    }
    latencies_ms: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_started = time.perf_counter()
    for batch in loader:
        tensor_batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if not fixed_modes:
            outputs = model(
                input_ids=tensor_batch["input_ids"],
                attention_mask=tensor_batch["attention_mask"],
                token_type_ids=tensor_batch["token_type_ids"],
                chunk_mask=tensor_batch["chunk_mask"],
                chunk_positions=tensor_batch["chunk_positions"],
                global_length_features=tensor_batch["global_length_features"],
                hard_last=True,
            )
            logits_by_mode = {output_modes[0]: outputs.joint_logits.float()}
            binary_logits_by_mode = {
                output_modes[0]: outputs.binary_logits.float()
            }
            attention = outputs.evidence["attention_weights"].float()
            risks = outputs.risk_scores.float()
            top_indices = outputs.evidence["top1_indices"]
            top_evidence_indices = outputs.evidence["top2_indices"]
            last_indices = outputs.evidence["last_indices"]
        else:
            encoded = model.chunk_encoder(
                input_ids=tensor_batch["input_ids"],
                attention_mask=tensor_batch["attention_mask"],
                token_type_ids=tensor_batch["token_type_ids"],
                chunk_mask=tensor_batch["chunk_mask"],
                chunk_positions=tensor_batch["chunk_positions"],
            )
            shape = encoded["base_chunk_repr"].shape
            flat = encoded["base_chunk_repr"].reshape(-1, shape[-1])
            window_heads = model.logits_from_hidden(flat)
            window_logits = window_heads.joint_logits.reshape(shape[0], shape[1], 13)
            window_binary_logits = window_heads.binary_logits.reshape(
                shape[0], shape[1], 2
            )
            logits_by_mode = {
                mode: aggregate_window_logits(
                    window_logits, tensor_batch["chunk_mask"], mode
                )
                for mode in fixed_modes
            }
            binary_logits_by_mode = {
                mode: aggregate_window_logits(
                    window_binary_logits, tensor_batch["chunk_mask"], mode
                )
                for mode in fixed_modes
            }
            normal_probability = window_binary_logits.softmax(dim=-1)[..., 0]
            risks = (1.0 - normal_probability) * tensor_batch["chunk_mask"].float()
            attention = risks / risks.sum(dim=1, keepdim=True).clamp_min(1e-8)
            top_indices = risks.argmax(dim=1)
            top_evidence_indices = risks.masked_fill(
                ~tensor_batch["chunk_mask"], -torch.inf
            ).topk(k=min(2, risks.shape[1]), dim=1).indices
            last_indices = torch.full_like(top_indices, -1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.extend(
            [elapsed_ms / max(len(batch["sample_id"]), 1)] * len(batch["sample_id"])
        )

        masks = tensor_batch["chunk_mask"].detach().cpu()
        for mode, call_logits in logits_by_mode.items():
            probs = call_logits.softmax(dim=-1).detach().cpu()
            preds = call_logits.argmax(dim=-1).detach().cpu()
            binary_logits = binary_logits_by_mode[mode]
            binary_probs = binary_logits.softmax(dim=-1).detach().cpu()
            binary_preds = binary_logits.argmax(dim=-1).detach().cpu()
            for index, metadata in enumerate(batch["metadata"]):
                count = int(masks[index].sum())
                record = dict(metadata)
                record.update(
                    {
                        "prediction_13": int(preds[index]),
                        "p_label_13": probs[index].tolist(),
                        "prediction_binary": int(binary_preds[index]),
                        "p_binary": binary_probs[index].tolist(),
                        "chunk_attention_weights": attention[index, :count]
                        .detach()
                        .cpu()
                        .tolist(),
                        "chunk_risk_scores": risks[index, :count]
                        .detach()
                        .cpu()
                        .tolist(),
                        "top_evidence_chunk": int(
                            top_indices[index].detach().cpu()
                        ),
                        "top_evidence_chunks": [
                            int(value)
                            for value in top_evidence_indices[index]
                            .detach()
                            .cpu()
                            .tolist()[:count]
                        ],
                        "last_high_risk_chunk": int(
                            last_indices[index].detach().cpu()
                        ),
                        "full_text_coverage": float(metadata["coverage_ratio"]),
                        "inference_mode": mode,
                    }
                )
                predictions_by_mode[mode].append(record)

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    wall_time_seconds = time.perf_counter() - run_started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_mode = {
        mode: summarize_predictions(
            predictions,
            latencies_ms,
            chunk_length=args.chunk_length,
            peak_gpu_memory_mb=peak_gpu_memory_mb,
            wall_time_seconds=wall_time_seconds,
        )
        for mode, predictions in predictions_by_mode.items()
    }
    common = {"checkpoint": checkpoint_report, "config": vars(args)}
    if len(output_modes) == 1:
        mode = output_modes[0]
        write_jsonl(
            args.output_dir / "predictions.jsonl", predictions_by_mode[mode]
        )
        payload: dict[str, Any] = {"metrics": metrics_by_mode[mode], **common}
    else:
        for mode in output_modes:
            mode_dir = args.output_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            write_jsonl(mode_dir / "predictions.jsonl", predictions_by_mode[mode])
            (mode_dir / "metrics.json").write_text(
                json.dumps(
                    {"metrics": metrics_by_mode[mode], "mode": mode, **common},
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        payload = {"modes": metrics_by_mode, **common}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
