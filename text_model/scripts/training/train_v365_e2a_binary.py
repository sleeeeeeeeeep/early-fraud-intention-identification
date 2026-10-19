#!/usr/bin/env python3
"""Train E2-A for one frozen-backbone epoch from M0 using binary CE only.

This deliberately has no Dev input and no checkpoint-selection loop.  The sole
epoch is saved as ``best_model.pt`` for compatibility with the existing strict
inference loader; V365 Dev and V361 long-text Dev are evaluated only afterward.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from data.longdoc_chunker import ChunkConfig, HEAD_MODE
    from data.longdoc_collator import LongDocCollator
    from inference.train_longdoc import (
        apply_backbone_train_mode,
        child_mapping,
        set_backbone_stage,
    )
    from models.longdoc_multitask_classifier import build_longdoc_model
except ImportError:
    from text_model.data.longdoc_chunker import ChunkConfig, HEAD_MODE
    from text_model.data.longdoc_collator import LongDocCollator
    from text_model.inference.train_longdoc import (
        apply_backbone_train_mode,
        child_mapping,
        set_backbone_stage,
    )
    from text_model.models.longdoc_multitask_classifier import build_longdoc_model


EXPECTED_TRAIN_SUFFIX = (
    "data/processed/v365_e2a_te_prefix_60new_40replay_s202/"
    "train_e2a_binary_6500.jsonl"
)
EXPECTED_M0_SUFFIX = "runs/repro_v361b_b320_head96/best_model.pt"
EXPECTED_M0_SHA256 = "c6f7c97dc6dbcd1ab0a8fe148918b04b677d408eb133737acf831ec02ad2ea13"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_path(path: Path, suffix: str, description: str) -> Path:
    resolved = path.resolve()
    normalized = resolved.as_posix()
    if not normalized.endswith(suffix):
        raise SystemExit(f"refusing unexpected {description}: {normalized}")
    if "test" in normalized.lower() or "restricted" in normalized.lower():
        raise SystemExit(f"refusing Test/restricted path: {normalized}")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def weighted_binary_loss(
    logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    per_row = F.cross_entropy(logits.float(), targets.long(), reduction="none")
    weights = weights.float()
    return (per_row * weights).sum() / weights.sum().clamp_min(1e-8)


def load_full_checkpoint(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict M0 load failed: {incompatible}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
    }


def trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--v341-checkpoint", type=Path, required=True)
    parser.add_argument("--init-longdoc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", choices=("B",), default="B")
    parser.add_argument("--input-mode", choices=(HEAD_MODE,), default=HEAD_MODE)
    parser.add_argument("--chunk-length", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--head-context-length", type=int, default=96)
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--encoder-micro-batch-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=0.0)
    parser.add_argument("--task-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs != 1:
        raise SystemExit("E2-A contract fixes epochs=1")
    expected = {
        "chunk_length": 320,
        "overlap": 64,
        "head_context_length": 96,
        "max_chunks": 12,
        "encoder_micro_batch_size": 12,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "freeze_backbone_epochs": 1,
        "top_layers": 4,
        "encoder_learning_rate": 0.0,
        "task_learning_rate": 2e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "gradient_clip_norm": 1.0,
        "seed": 202,
    }
    mismatched = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if mismatched:
        raise SystemExit(f"E2-A parameters differ from E1-B: {mismatched}")

    train_path = require_path(args.train_jsonl, EXPECTED_TRAIN_SUFFIX, "train file")
    init_path = require_path(
        args.init_longdoc_checkpoint, EXPECTED_M0_SUFFIX, "M0 checkpoint"
    )
    if sha256_file(init_path) != EXPECTED_M0_SHA256:
        raise ValueError("M0 checkpoint SHA-256 mismatch")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"base model directory is invalid: {args.model}")
    if not args.v341_checkpoint.is_file():
        raise FileNotFoundError(args.v341_checkpoint)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output}")
    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes < 7 * 1024**3:
        raise RuntimeError(
            f"less than 7 GiB free for full E2-A checkpoint: {free_bytes / 1024**3:.2f} GiB"
        )

    rows = read_jsonl(train_path)
    if len(rows) != 6500:
        raise ValueError(f"expected 6500 training rows, got {len(rows)}")
    if any(float(row.get("binary_loss_mask", 0.0)) <= 0.0 for row in rows):
        raise ValueError("masked/ambiguous row exists in E2-A physical mixture")
    if any(int(row.get("binary_target", -1)) not in (0, 1) for row in rows):
        raise ValueError("invalid binary_target in E2-A physical mixture")
    source_counts = collections.Counter(str(row.get("e2a_source")) for row in rows)
    if source_counts != collections.Counter(
        v365_te_prefix=3900, v361_fulltext_replay=2600
    ):
        raise ValueError(f"unexpected source mixture: {dict(source_counts)}")
    new_rows = [row for row in rows if row["e2a_source"] == "v365_te_prefix"]
    if collections.Counter(int(row["binary_target"]) for row in new_rows) != collections.Counter(
        {0: 1950, 1: 1950}
    ):
        raise ValueError("new V365 prefix block is not exactly class-balanced")

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer_source = (
        args.v341_checkpoint.parent
        if (args.v341_checkpoint.parent / "tokenizer.json").exists()
        else args.model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    chunk_config = ChunkConfig(
        chunk_length=args.chunk_length,
        overlap=args.overlap,
        input_mode=args.input_mode,
        head_context_length=args.head_context_length,
        max_chunks_per_call=args.max_chunks,
    )
    collator = LongDocCollator(
        tokenizer, chunk_config, child_code_to_index=child_mapping()
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        RowsDataset(rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )

    model = build_longdoc_model(
        str(args.model),
        route=args.route,
        encoder_micro_batch_size=args.encoder_micro_batch_size,
        evidence_components=("attention", "top1", "top2", "last", "mean"),
        v341_checkpoint=args.v341_checkpoint,
    )
    init_report = load_full_checkpoint(model, init_path)
    for parameter in model.subtype_classifier.parameters():
        parameter.requires_grad_(False)
    for parameter in model.official_subcategory_classifier.parameters():
        parameter.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("E2-A training requires CUDA")
    model.to(device)
    amp_dtype = torch.float16 if args.amp else None
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype is not None)

    backbone_ids = {
        id(parameter) for parameter in model.chunk_encoder.backbone.parameters()
    }
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) in backbone_ids
    ]
    task_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.encoder_learning_rate},
            {"params": task_parameters, "lr": args.task_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    optimizer_steps = math.ceil(len(loader) / args.gradient_accumulation_steps)
    warmup_steps = int(optimizer_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )

    stage = set_backbone_stage(
        model.chunk_encoder.backbone,
        1,
        frozen_epochs=args.freeze_backbone_epochs,
        top_layer_epochs=args.epochs,
        top_layers=args.top_layers,
    )
    if stage != "frozen":
        raise RuntimeError(f"epoch 1 backbone must be frozen, got {stage}")
    summary = trainable_summary(model)
    if summary["trainable_parameters"] != 5_857_812:
        raise ValueError(f"unexpected trainable parameter count: {summary}")
    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    model.train()
    apply_backbone_train_mode(model.chunk_encoder.backbone, stage)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    started = time.perf_counter()
    optimizer_step_count = 0
    for step, raw_batch in enumerate(loader, 1):
        batch = move_batch(raw_batch, device)
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
            loss = weighted_binary_loss(
                outputs.binary_logits,
                batch["binary_target"],
                batch["binary_weight"],
            )
            scaled_loss = loss / args.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        if step % args.gradient_accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1
        running_loss += float(loss.detach().cpu())
        if step % args.progress_every == 0 or step == len(loader):
            elapsed = (time.perf_counter() - started) / 60.0
            print(
                f"[E2-A epoch 1/1] step {step}/{len(loader)} "
                f"loss={running_loss / step:.6f} elapsed={elapsed:.1f}m",
                flush=True,
            )

    if optimizer_step_count != optimizer_steps:
        raise RuntimeError(
            f"optimizer step mismatch: {optimizer_step_count} != {optimizer_steps}"
        )
    changed: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for name, parameter in model.named_parameters():
        if name not in trainable_before:
            continue
        after = parameter.detach().cpu()
        delta = float((after - trainable_before[name]).abs().max())
        if delta > 0.0:
            changed.append(
                {"name": name, "numel": parameter.numel(), "max_abs_delta": delta}
            )
        else:
            unchanged.append(name)
    del trainable_before
    if not changed:
        raise RuntimeError("no trainable E2-A parameter changed")

    elapsed_seconds = time.perf_counter() - started
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metrics = {
        "epoch": 1,
        "task": "binary fraud detection only",
        "backbone_stage": stage,
        "train_loss": running_loss / len(loader),
        "train_seconds": elapsed_seconds,
        "train_minutes": elapsed_seconds / 60.0,
        "optimizer_steps": optimizer_step_count,
        "warmup_steps": warmup_steps,
        "trainable": summary,
        "changed_trainable_tensors": len(changed),
        "unchanged_trainable_tensors": len(unchanged),
    }
    checkpoint_path = args.output / "best_model.pt"
    temp_checkpoint = args.output / "best_model.pt.tmp"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "e2a_config": config,
            "init_checkpoint": init_report,
        },
        temp_checkpoint,
    )
    temp_checkpoint.replace(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest = {
        "schema_version": "v365.e2a.binary_training.v1",
        "status": "complete",
        "task": "binary fraud detection only",
        "score": "p_binary[1]",
        "selection_policy": "none; fixed one epoch, saved final state",
        "dev_data_used_during_training": False,
        "test_data_used": False,
        "train_jsonl": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "init_checkpoint": init_report,
        "config": config,
        "metrics": metrics,
        "parameter_change_audit": {
            "changed": changed,
            "unchanged_trainable_names": unchanged,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha,
            "format": "full strict-compatible model_state_dict",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"E2-A training complete: loss={metrics['train_loss']:.6f} "
        f"minutes={metrics['train_minutes']:.2f} checkpoint={checkpoint_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
