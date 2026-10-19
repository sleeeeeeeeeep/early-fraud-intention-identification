#!/usr/bin/env python3
"""Train V362 E1-B with strict-prefix supervision plus full-text replay.

Only the dedicated binary detection head is optimized/evaluated.  The subtype,
child, and joint-13 heads are loaded for checkpoint compatibility but frozen and
excluded from the loss and checkpoint-selection policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from data.longdoc_chunker import ChunkConfig, HEAD_MODE, PLAIN_MODE
    from data.longdoc_collator import LongDocCollator
    from inference.train_longdoc import apply_backbone_train_mode, child_mapping, set_backbone_stage
    from models.longdoc_multitask_classifier import build_longdoc_model
except ImportError:
    from text_model.data.longdoc_chunker import ChunkConfig, HEAD_MODE, PLAIN_MODE
    from text_model.data.longdoc_collator import LongDocCollator
    from text_model.inference.train_longdoc import apply_backbone_train_mode, child_mapping, set_backbone_stage
    from text_model.models.longdoc_multitask_classifier import build_longdoc_model


class RowsDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def weighted_binary_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    per_row = F.cross_entropy(logits.float(), targets.long(), reduction="none")
    weights = weights.float()
    return (per_row * weights).sum() / weights.sum().clamp_min(1e-8)


def safe_div(a: int | float, b: int | float) -> float | None:
    return None if b == 0 else float(a) / float(b)


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        prediction = int(score >= threshold)
        if label and prediction:
            tp += 1
        elif label:
            fn += 1
        elif prediction:
            fp += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = 0.0 if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": threshold,
        "samples": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_fpr": safe_div(fp, fp + tn),
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank = ((index + 1) + end) / 2.0
        rank_sum += rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in sorted(zip(scores, labels), reverse=True):
        tp += int(label == 1)
        fp += int(label == 0)
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    records: list[dict[str, Any]] = []
    losses: list[float] = []
    started = time.perf_counter()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                chunk_mask=batch["chunk_mask"],
                chunk_positions=batch["chunk_positions"],
                global_length_features=batch["global_length_features"],
                hard_last=True,
            )
            loss = weighted_binary_loss(outputs.binary_logits, batch["binary_target"], batch["binary_weight"])
        probabilities = outputs.binary_logits.float().softmax(dim=-1).detach().cpu()
        weights = batch["binary_weight"].detach().cpu()
        targets = batch["binary_target"].detach().cpu()
        losses.append(float(loss.detach().cpu()))
        for index, metadata in enumerate(raw_batch["metadata"]):
            counted = float(weights[index]) > 0.0
            record = dict(metadata)
            record.update(
                {
                    "binary_target": int(targets[index]),
                    "binary_weight": float(weights[index]),
                    "prediction_binary": int(probabilities[index].argmax()),
                    "p_binary": probabilities[index].tolist(),
                    "counted_in_binary_metrics": counted,
                }
            )
            records.append(record)
            if counted:
                labels.append(int(targets[index]))
                scores.append(float(probabilities[index, 1]))
    metrics = binary_metrics(labels, scores)
    metrics.update(
        {
            "loss": statistics.fmean(losses) if losses else 0.0,
            "roc_auc": roc_auc(labels, scores),
            "pr_auc": pr_auc(labels, scores),
            "excluded_masked_rows": len(records) - len(labels),
            "wall_time_seconds": time.perf_counter() - started,
        }
    )
    return metrics, records


def load_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"checkpoint mismatch: {incompatible}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, metrics: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": dict(metrics),
            "e1b_config": dict(config),
        },
        path,
    )


def trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable_parameters": trainable, "total_parameters": total}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--prefix-dev-jsonl", type=Path, required=True)
    parser.add_argument("--fulltext-dev-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--v341-checkpoint", type=Path, required=True)
    parser.add_argument("--init-longdoc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("A", "B"), default="B")
    parser.add_argument("--input-mode", choices=(PLAIN_MODE, HEAD_MODE), default=HEAD_MODE)
    parser.add_argument("--chunk-length", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--head-context-length", type=int, default=96)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--encoder-micro-batch-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-6)
    parser.add_argument("--task-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--fulltext-accuracy-floor", type=float, default=0.99)
    parser.add_argument("--prefix-fpr-ceiling", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predictions").mkdir()
    train_rows = read_jsonl(args.train_jsonl)
    prefix_dev_rows = read_jsonl(args.prefix_dev_jsonl)
    fulltext_dev_rows = read_jsonl(args.fulltext_dev_jsonl)
    if any(float(row.get("binary_loss_mask", 1.0)) <= 0.0 for row in train_rows):
        raise ValueError("E1-B physical mixture must not contain masked/ambiguous training rows")

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer_source = args.v341_checkpoint.parent if (args.v341_checkpoint.parent / "tokenizer.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    chunk_config = ChunkConfig(
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        input_mode=args.input_mode,
        head_context_length=args.head_context_length,
        max_chunks_per_call=args.max_chunks,
    )
    collator = LongDocCollator(tokenizer, chunk_config, child_code_to_index=child_mapping())
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        RowsDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )
    prefix_dev_loader = DataLoader(RowsDataset(prefix_dev_rows), batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    fulltext_dev_loader = DataLoader(RowsDataset(fulltext_dev_rows), batch_size=args.eval_batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    model = build_longdoc_model(
        args.model,
        route=args.route,
        encoder_micro_batch_size=args.encoder_micro_batch_size,
        evidence_components=("attention", "top1", "top2", "last", "mean"),
        v341_checkpoint=args.v341_checkpoint,
    )
    init_report = load_checkpoint(model, args.init_longdoc_checkpoint)
    for parameter in model.subtype_classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in model.official_subcategory_classifier.parameters():
        parameter.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    amp_dtype = torch.float16 if args.amp and device.type == "cuda" else None
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype is not None)

    backbone_ids = {id(parameter) for parameter in model.chunk_encoder.backbone.parameters()}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    task_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.encoder_learning_rate},
            {"params": task_parameters, "lr": args.task_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=total_optimizer_steps,
    )

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    stale = 0

    baseline_prefix, baseline_prefix_records = evaluate(model, prefix_dev_loader, device, amp_dtype)
    baseline_fulltext, baseline_fulltext_records = evaluate(model, fulltext_dev_loader, device, amp_dtype)
    write_jsonl(args.output / "predictions" / "epoch_0_prefix_dev.jsonl", baseline_prefix_records)
    write_jsonl(args.output / "predictions" / "epoch_0_fulltext_dev.jsonl", baseline_fulltext_records)
    baseline = {"prefix_dev": baseline_prefix, "fulltext_dev": baseline_fulltext}
    print("[epoch 0] prefix", json.dumps(baseline_prefix, ensure_ascii=False), flush=True)
    print("[epoch 0] fulltext", json.dumps(baseline_fulltext, ensure_ascii=False), flush=True)

    for epoch in range(1, args.epochs + 1):
        stage = set_backbone_stage(
            model.chunk_encoder.backbone,
            epoch,
            frozen_epochs=args.freeze_backbone_epochs,
            top_layer_epochs=args.epochs,
            top_layers=args.top_layers,
        )
        model.train()
        apply_backbone_train_mode(model.chunk_encoder.backbone, stage)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_started = time.perf_counter()
        for step, raw_batch in enumerate(train_loader, 1):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    chunk_mask=batch["chunk_mask"],
                    chunk_positions=batch["chunk_positions"],
                    global_length_features=batch["global_length_features"],
                )
                loss = weighted_binary_loss(outputs.binary_logits, batch["binary_target"], batch["binary_weight"])
                scaled = loss / args.gradient_accumulation_steps
            scaler.scale(scaled).backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())
            if step % args.progress_every == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_started
                print(
                    f"[epoch {epoch}/{args.epochs}] step {step}/{len(train_loader)} "
                    f"loss={running_loss/step:.6f} elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

        prefix_metrics, prefix_records = evaluate(model, prefix_dev_loader, device, amp_dtype)
        fulltext_metrics, fulltext_records = evaluate(model, fulltext_dev_loader, device, amp_dtype)
        write_jsonl(args.output / "predictions" / f"epoch_{epoch}_prefix_dev.jsonl", prefix_records)
        write_jsonl(args.output / "predictions" / f"epoch_{epoch}_fulltext_dev.jsonl", fulltext_records)
        guard_pass = (
            float(fulltext_metrics["accuracy"] or 0.0) >= args.fulltext_accuracy_floor
            and float(prefix_metrics["normal_fpr"] or 0.0) <= args.prefix_fpr_ceiling
        )
        selection_score = (
            float(prefix_metrics["recall"] or 0.0) + 0.05 * float(prefix_metrics["f1"] or 0.0)
            if guard_pass
            else -10.0 + float(fulltext_metrics["accuracy"] or 0.0) - float(prefix_metrics["normal_fpr"] or 0.0)
        )
        epoch_record = {
            "epoch": epoch,
            "backbone_stage": stage,
            "train_loss": running_loss / len(train_loader),
            "train_minutes": (time.perf_counter() - epoch_started) / 60.0,
            "trainable": trainable_summary(model),
            "prefix_dev": prefix_metrics,
            "fulltext_dev": fulltext_metrics,
            "selection_guard_pass": guard_pass,
            "selection_score": selection_score,
        }
        history.append(epoch_record)
        print(f"[epoch {epoch}] metrics {json.dumps(epoch_record, ensure_ascii=False)}", flush=True)
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            stale = 0
            save_checkpoint(args.output / "best_model.pt", model, epoch, epoch_record, config)
            write_jsonl(args.output / "predictions" / "best_prefix_dev.jsonl", prefix_records)
            write_jsonl(args.output / "predictions" / "best_fulltext_dev.jsonl", fulltext_records)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop after epoch {epoch}", flush=True)
                break

    manifest = {
        "schema_version": "v362.e1b.binary_training.v1",
        "task": "binary fraud detection only",
        "score": "p_binary[1]",
        "train_jsonl": str(args.train_jsonl),
        "train_sha256": sha256(args.train_jsonl),
        "train_rows": len(train_rows),
        "prefix_dev_rows": len(prefix_dev_rows),
        "fulltext_dev_rows": len(fulltext_dev_rows),
        "init_checkpoint": init_report,
        "config": config,
        "baseline_epoch_0": baseline,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "selection_policy": {
            "guard": f"fulltext accuracy >= {args.fulltext_accuracy_floor} and prefix normal FPR <= {args.prefix_fpr_ceiling}",
            "objective": "maximize prefix recall + 0.05 * prefix F1",
            "stress_used": False,
        },
        "history": history,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps({"status": "done", "best_epoch": best_epoch, "history": history}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"E1-B complete: best_epoch={best_epoch} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
