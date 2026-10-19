#!/usr/bin/env python3
"""训练 V341 长文本路线 A/B。

入口只读取显式传入的 Train、Dev-A2 和 Long-Dev。Test-B 路径在训练阶段会被
拒绝。两条路线共享数据、分块、损失、初始化与 checkpoint 选择协议。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from data.longdoc_chunker import ChunkConfig, HEAD_MODE, PLAIN_MODE
    from data.longdoc_collator import LongDocCollator, NUM_PARENT_CLASSES
    from models.longdoc_multitask_classifier import build_longdoc_model
    from training.longdoc_losses import longdoc_multitask_loss
    from training.longdoc_metrics import classification_metrics, stratified_metrics
    from training.longdoc_sampler import LongExposureSampler
except ImportError:
    from text_model.data.longdoc_chunker import ChunkConfig, HEAD_MODE, PLAIN_MODE
    from text_model.data.longdoc_collator import LongDocCollator, NUM_PARENT_CLASSES
    from text_model.models.longdoc_multitask_classifier import build_longdoc_model
    from text_model.training.longdoc_losses import longdoc_multitask_loss
    from text_model.training.longdoc_metrics import classification_metrics, stratified_metrics
    from text_model.training.longdoc_sampler import LongExposureSampler


BASELINE_DEV_ACCURACY = 0.9228206551915602
BASELINE_DEV_MACRO_F1 = 0.925528294873317


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    lowered = path.name.lower().replace("-", "_")
    if "test_b" in lowered or "testb" in lowered or "oof" in lowered:
        raise ValueError("longdoc training accepts Train/Dev only")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def child_mapping() -> dict[str, int]:
    try:
        from scripts.inference.official_subcategories import OFFICIAL_SUBCATEGORY_TO_INDEX
    except ImportError:
        from text_model.scripts.inference.official_subcategories import OFFICIAL_SUBCATEGORY_TO_INDEX
    return dict(OFFICIAL_SUBCATEGORY_TO_INDEX)


class RowsDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        asr_view_fraction: float = 0.0,
        seed: int = 202,
    ) -> None:
        if not 0.0 <= asr_view_fraction <= 1.0:
            raise ValueError("asr_view_fraction must be in [0, 1]")
        self.rows = rows
        self.asr_view_fraction = float(asr_view_fraction)
        self.seed = int(seed)
        self.paired_indices = [
            index
            for index, row in enumerate(rows)
            if str(row.get("clean_model_input_text") or "").strip()
            and str(row.get("asr_model_input_text") or "").strip()
        ]
        if self.asr_view_fraction and not self.paired_indices:
            raise ValueError("ASR view sampling requires paired clean/ASR rows")
        self.asr_indices: set[int] = set()
        self.set_epoch(0)

    @property
    def asr_rows_per_epoch(self) -> int:
        return round(len(self.paired_indices) * self.asr_view_fraction)

    def set_epoch(self, epoch: int) -> None:
        rng = random.Random(self.seed + 1009 * int(epoch))
        self.asr_indices = set(
            rng.sample(self.paired_indices, self.asr_rows_per_epoch)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if not self.paired_indices:
            return row
        selected = dict(row)
        use_asr = index in self.asr_indices
        selected["model_input_text"] = str(
            row["asr_model_input_text"]
            if use_asr
            else row["clean_model_input_text"]
        )
        selected["input_view"] = "asr" if use_asr else "clean"
        selected["disable_window_evidence"] = use_asr
        return selected


def _class_weights(rows: list[Mapping[str, Any]], cap: float = 1.25) -> torch.Tensor:
    counts = Counter(
        int(row.get("canonical_label_13", row.get("label_13", 0))) for row in rows
    )
    total = float(sum(counts.values()))
    values = [
        total / (NUM_PARENT_CLASSES * max(counts.get(index, 0), 1))
        for index in range(NUM_PARENT_CLASSES)
    ]
    mean = sum(values) / max(len(values), 1)
    return torch.tensor(
        [min(value / max(mean, 1e-8), cap) for value in values],
        dtype=torch.float32,
    )


def _child_weights(
    rows: list[Mapping[str, Any]], mapping: Mapping[str, int], cap: float = 3.0
) -> torch.Tensor:
    counts: Counter[int] = Counter()
    for row in rows:
        parent = int(row.get("canonical_label_13", row.get("label_13", 0)))
        code = str(
            row.get("canonical_subcategory_code") or row.get("subcategory_code") or ""
        )
        if parent != 0 and code in mapping:
            counts[mapping[code]] += 1
    total = float(sum(counts.values()))
    values = [
        (total / max(counts.get(index, 0), 1)) ** 0.5 for index in range(39)
    ]
    mean = sum(values) / max(len(values), 1)
    return torch.tensor(
        [min(value / max(mean, 1e-8), cap) for value in values],
        dtype=torch.float32,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    parent_weights: torch.Tensor,
    child_weights: torch.Tensor,
    amp_dtype: torch.dtype | None,
    *,
    chunk_length: int,
    loss_weights: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    subtype_labels: list[int] = []
    subtype_predictions: list[int] = []
    child_labels: list[int] = []
    child_predictions: list[int] = []
    records: list[dict[str, Any]] = []
    total_loss = 0.0
    batches = 0
    latencies_ms: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                chunk_mask=batch["chunk_mask"],
                chunk_positions=batch["chunk_positions"],
                global_length_features=batch["global_length_features"],
                hard_last=True,
            )
            loss, _ = longdoc_multitask_loss(
                outputs,
                batch,
                parent_class_weights=parent_weights,
                child_class_weights=child_weights,
                loss_weights=loss_weights,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - started) * 1000.0
        per_call = elapsed / max(len(raw_batch["sample_id"]), 1)
        latencies_ms.extend([per_call] * len(raw_batch["sample_id"]))

        parent_pred = outputs.logits.argmax(dim=-1)
        child_pred = outputs.official_subcategory_logits.argmax(dim=-1)
        subtype_pred = outputs.subtype_logits.argmax(dim=-1)
        total_loss += float(loss.detach().cpu())
        batches += 1
        labels.extend(batch["label_13"].detach().cpu().tolist())
        predictions.extend(parent_pred.detach().cpu().tolist())
        fraud = batch["subtype_label"] >= 0
        if bool(fraud.any()):
            subtype_labels.extend(batch["subtype_label"][fraud].detach().cpu().tolist())
            subtype_predictions.extend(subtype_pred[fraud].detach().cpu().tolist())
            child_labels.extend(batch["child_label"][fraud].detach().cpu().tolist())
            child_predictions.extend(child_pred[fraud].detach().cpu().tolist())

        parent_probs = outputs.logits.float().softmax(dim=-1).detach().cpu()
        binary_probs = outputs.binary_logits.float().softmax(dim=-1).detach().cpu()
        attention = outputs.evidence["attention_weights"].detach().cpu()
        risks = outputs.risk_scores.detach().cpu()
        top1 = outputs.evidence["top1_indices"].detach().cpu()
        top2 = outputs.evidence["top2_indices"].detach().cpu()
        last = outputs.evidence["last_indices"].detach().cpu()
        masks = batch["chunk_mask"].detach().cpu()
        for index, metadata in enumerate(raw_batch["metadata"]):
            valid_chunks = int(masks[index].sum())
            record = dict(metadata)
            record.update(
                {
                    "prediction_13": int(parent_pred[index].detach().cpu()),
                    "p_label_13": parent_probs[index].tolist(),
                    "p_binary_fraud": float(binary_probs[index, 1]),
                    "prediction_subcategory39_index": int(child_pred[index].detach().cpu()),
                    "chunk_attention_weights": attention[index, :valid_chunks].tolist(),
                    "chunk_risk_scores": risks[index, :valid_chunks].tolist(),
                    "top_evidence_chunk": int(top1[index]),
                    "top_evidence_chunks": [
                        int(value) for value in top2[index].tolist()[:valid_chunks]
                    ],
                    "last_high_risk_chunk": int(last[index]),
                    "full_text_coverage": float(metadata["coverage_ratio"]),
                }
            )
            records.append(record)

    metrics = classification_metrics(labels, predictions, NUM_PARENT_CLASSES)
    metrics.update(
        {
            "loss": total_loss / max(batches, 1),
            "fraud_subtype_accuracy": sum(
                a == b for a, b in zip(subtype_labels, subtype_predictions)
            )
            / max(len(subtype_labels), 1),
            "fraud_subtype_macro_f1": classification_metrics(
                subtype_labels, subtype_predictions, 12
            )["macro_f1"]
            if subtype_labels
            else 0.0,
            "subcategory39_accuracy": sum(
                a == b for a, b in zip(child_labels, child_predictions)
            )
            / max(len(child_labels), 1),
            "subcategory39_macro_f1": classification_metrics(
                child_labels, child_predictions, 39
            )["macro_f1"]
            if child_labels
            else 0.0,
            "coverage_min": min(
                (float(record["full_text_coverage"]) for record in records), default=0.0
            ),
            "average_chunks_per_call": statistics.fmean(
                [int(record["num_chunks"]) for record in records]
            )
            if records
            else 0.0,
            "latency_ms_p50": _percentile(latencies_ms, 0.50),
            "latency_ms_p95": _percentile(latencies_ms, 0.95),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
            "stratified": stratified_metrics(records, chunk_length=chunk_length),
        }
    )
    return metrics, records


def _save_checkpoint(
    path: Path, model: torch.nn.Module, epoch: int, metrics: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": int(epoch), "model_state_dict": model.state_dict(), "metrics": metrics},
        path,
    )


def _encoder_layers(backbone: torch.nn.Module) -> list[torch.nn.Module]:
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layer", None)
    return list(layers) if layers is not None else []


def set_backbone_stage(
    backbone: torch.nn.Module,
    epoch: int,
    *,
    frozen_epochs: int,
    top_layer_epochs: int,
    top_layers: int,
) -> str:
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    if epoch <= frozen_epochs:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        return "frozen"
    if epoch <= frozen_epochs + top_layer_epochs:
        layers = _encoder_layers(backbone)
        if not layers:
            return "full"
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for layer in layers[-min(top_layers, len(layers)) :]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        pooler = getattr(backbone, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad_(True)
        return f"top_{min(top_layers, len(layers))}"
    return "full"


def apply_backbone_train_mode(backbone: torch.nn.Module, stage: str) -> None:
    """让冻结层保持确定性，同时保留解冻层的训练态。"""

    if stage == "frozen":
        backbone.eval()
        return
    if stage.startswith("top_"):
        count = int(stage.split("_", 1)[1])
        backbone.eval()
        layers = _encoder_layers(backbone)
        for layer in layers[-min(count, len(layers)) :]:
            layer.train()
        pooler = getattr(backbone, "pooler", None)
        if pooler is not None:
            pooler.train()
        return
    backbone.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--long-dev-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--v341-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("A", "B"), required=True)
    parser.add_argument("--input-mode", choices=(PLAIN_MODE, HEAD_MODE), default=PLAIN_MODE)
    parser.add_argument("--chunk-length", type=int, choices=(256, 320, 384), default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--head-context-length", type=int, choices=(48, 64, 96), default=64)
    parser.add_argument("--diagnostic-max-chunks", type=int, default=12)
    parser.add_argument("--encoder-micro-batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--encoder-learning-rate", type=float, default=5e-6)
    parser.add_argument("--new-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--fusion-hidden", type=int, default=512)
    parser.add_argument("--document-layers", type=int, default=2)
    parser.add_argument("--document-heads", type=int, default=8)
    parser.add_argument("--document-intermediate", type=int, default=2048)
    parser.add_argument(
        "--evidence-components",
        default="attention,top1,top2,last,mean",
        help="逗号分隔的证据池化分支，用于 A/B 聚合消融",
    )
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--top-layer-epochs", type=int, default=2)
    parser.add_argument("--top-layers", type=int, default=6)
    parser.add_argument("--long-exposure", type=float, default=0.21)
    parser.add_argument("--mil-weight", type=float, default=0.2)
    parser.add_argument("--window-evidence-weight", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--baseline-dev-accuracy", type=float, default=BASELINE_DEV_ACCURACY)
    parser.add_argument("--baseline-dev-macro-f1", type=float, default=BASELINE_DEV_MACRO_F1)
    parser.add_argument("--dev-regression-tolerance", type=float, default=0.003)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--long-dev-limit", type=int)
    parser.add_argument("--max-train-rows", type=int, default=6500)
    parser.add_argument("--asr-view-fraction", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_components = tuple(
        part.strip() for part in args.evidence_components.split(",") if part.strip()
    )
    if not evidence_components:
        raise ValueError("--evidence-components must contain at least one component")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.mil_weight < 0 or args.window_evidence_weight < 0:
        raise ValueError("MIL and window evidence weights must be non-negative")
    loss_weights = {
        "mil": float(args.mil_weight),
        "window_evidence": float(args.window_evidence_weight),
    }
    train_rows = read_jsonl(args.train_jsonl, args.train_limit)
    dev_rows = read_jsonl(args.dev_jsonl, args.dev_limit)
    long_dev_rows = read_jsonl(args.long_dev_jsonl, args.long_dev_limit)
    if len(train_rows) > args.max_train_rows:
        raise ValueError(
            f"training rows exceed cap: {len(train_rows)} > {args.max_train_rows}"
        )
    mapping = child_mapping()

    from transformers import AutoTokenizer

    tokenizer_source = args.v341_checkpoint.parent if (args.v341_checkpoint.parent / "tokenizer.json").exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    chunk_config = ChunkConfig(
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        input_mode=args.input_mode,
        head_context_length=args.head_context_length,
        max_chunks_per_call=args.diagnostic_max_chunks,
    )
    collator = LongDocCollator(tokenizer, chunk_config, child_code_to_index=mapping)
    train_sampler = LongExposureSampler(
        train_rows, target_fraction=args.long_exposure, seed=args.seed
    )
    train_dataset = RowsDataset(
        train_rows,
        asr_view_fraction=args.asr_view_fraction,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        RowsDataset(dev_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    long_dev_loader = DataLoader(
        RowsDataset(long_dev_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    model = build_longdoc_model(
        args.model,
        route=args.route,
        projection_dim=args.projection_dim,
        fusion_hidden=args.fusion_hidden,
        document_layers=args.document_layers,
        document_heads=args.document_heads,
        document_intermediate=args.document_intermediate,
        encoder_micro_batch_size=args.encoder_micro_batch_size,
        evidence_components=evidence_components,
        v341_checkpoint=args.v341_checkpoint,
    )
    if args.gradient_checkpointing:
        backbone = model.chunk_encoder.backbone
        if not hasattr(backbone, "gradient_checkpointing_enable"):
            raise ValueError("backbone does not support gradient checkpointing")
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parent_weights = _class_weights(train_rows).to(device)
    child_weights = _child_weights(train_rows, mapping).to(device)

    inherited_modules = (
        model.chunk_encoder.backbone,
        model.binary_classifier,
        model.subtype_classifier,
        model.official_subcategory_classifier,
    )
    inherited_ids = {
        id(parameter)
        for module in inherited_modules
        for parameter in module.parameters()
    }
    inherited_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in inherited_ids
    ]
    new_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in inherited_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": inherited_parameters, "lr": args.encoder_learning_rate},
            {"params": new_parameters, "lr": args.new_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / max(args.gradient_accumulation_steps, 1)
    )
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=total_optimizer_steps,
    )
    amp_dtype = torch.float16 if args.amp and device.type == "cuda" else None
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype is not None)
    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        backbone_stage = set_backbone_stage(
            model.chunk_encoder.backbone,
            epoch,
            frozen_epochs=args.freeze_backbone_epochs,
            top_layer_epochs=args.top_layer_epochs,
            top_layers=args.top_layers,
        )
        model.train()
        apply_backbone_train_mode(model.chunk_encoder.backbone, backbone_stage)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = _move_batch(raw_batch, device)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    chunk_mask=batch["chunk_mask"],
                    chunk_positions=batch["chunk_positions"],
                    global_length_features=batch["global_length_features"],
                )
                loss, _ = longdoc_multitask_loss(
                    outputs,
                    batch,
                    parent_class_weights=parent_weights,
                    child_class_weights=child_weights,
                    loss_weights=loss_weights,
                )
                scaled_loss = loss / max(args.gradient_accumulation_steps, 1)
            scaler.scale(scaled_loss).backward()
            should_step = (
                step % max(args.gradient_accumulation_steps, 1) == 0
                or step == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())
            steps += 1

        dev_metrics, dev_predictions = evaluate(
            model,
            dev_loader,
            device,
            parent_weights,
            child_weights,
            amp_dtype,
            chunk_length=args.chunk_length,
            loss_weights=loss_weights,
        )
        long_metrics, long_predictions = evaluate(
            model,
            long_dev_loader,
            device,
            parent_weights,
            child_weights,
            amp_dtype,
            chunk_length=args.chunk_length,
            loss_weights=loss_weights,
        )
        guard_pass = (
            float(dev_metrics["accuracy"])
            >= args.baseline_dev_accuracy - args.dev_regression_tolerance
            and float(dev_metrics["macro_f1"])
            >= args.baseline_dev_macro_f1 - args.dev_regression_tolerance
            and float(dev_metrics["coverage_min"]) == 1.0
            and float(long_metrics["coverage_min"]) == 1.0
        )
        selection_score = float(long_metrics["macro_f1"]) if guard_pass else -1.0
        epoch_metrics = {
            "epoch": epoch,
            "backbone_stage": backbone_stage,
            "train_loss": running_loss / max(steps, 1),
            "selection_gate_pass": guard_pass,
            "selection_score": selection_score,
            "dev_a2": dev_metrics,
            "long_dev": long_metrics,
        }
        history.append(epoch_metrics)
        write_jsonl(
            args.output / "predictions" / f"epoch_{epoch}_dev_a2.jsonl",
            dev_predictions,
        )
        write_jsonl(
            args.output / "predictions" / f"epoch_{epoch}_long_dev.jsonl",
            long_predictions,
        )
        if args.save_every_epoch and not args.metrics_only:
            _save_checkpoint(
                args.output / f"epoch_{epoch}.pt", model, epoch, epoch_metrics
            )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            if not args.metrics_only:
                _save_checkpoint(
                    args.output / "best_model.pt", model, epoch, epoch_metrics
                )
            write_jsonl(args.output / "predictions" / "best_dev_a2.jsonl", dev_predictions)
            write_jsonl(args.output / "predictions" / "best_long_dev.jsonl", long_predictions)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    manifest = {
        "schema_version": "v361.longdoc_training_manifest.v1",
        "route": args.route,
        "seed": args.seed,
        "config": {**vars(args), "output": str(args.output)},
        "train_rows": len(train_rows),
        "paired_train_rows": len(train_dataset.paired_indices),
        "asr_view_fraction": train_dataset.asr_view_fraction,
        "asr_view_rows_per_epoch": train_dataset.asr_rows_per_epoch,
        "dev_rows": len(dev_rows),
        "long_dev_rows": len(long_dev_rows),
        "long_exposure": {
            "requested": args.long_exposure,
            "effective": train_sampler.effective_fraction,
            "epoch_draws": len(train_sampler),
            "long_draws": train_sampler.long_draws,
        },
        "loss_weights": loss_weights,
        "evidence_components": list(evidence_components),
        "optimizer_steps_planned": total_optimizer_steps,
        "checkpoint_policy": (
            "metrics_only"
            if args.metrics_only
            else "best_plus_every_epoch"
            if args.save_every_epoch
            else "best_only"
        ),
        "v341_load_report": model.v341_load_report,
        "best_epoch": best_epoch,
        "best_selection_metric": "long_dev_macro_f1_with_dev_a2_guard",
        "history": history,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output / "metrics.json").write_text(
        json.dumps(
            {"status": "done", "best_epoch": best_epoch, "history": history},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
