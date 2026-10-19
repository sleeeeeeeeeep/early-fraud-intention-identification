#!/usr/bin/env python3
"""Train the pre-registered E2-B binary model for one frozen-backbone epoch.

The 6,500 physical examples receive the same binary cross-entropy objective as
E2-A.  V365 fraud tE examples additionally encode their same-call tE-1 view in
the same forward batch and receive fixed adjacent-boundary ranking and pre-tE
alarm-cap penalties.  No Dev data or checkpoint selection is used.
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
from typing import Any, Mapping, Sequence

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
    "data/processed/v365_e2b_boundary_balanced_60new_40replay_s202/"
    "train_e2b_binary_6500.jsonl"
)
EXPECTED_M0_SUFFIX = "runs/repro_v361b_b320_head96/best_model.pt"
EXPECTED_M0_SHA256 = "c6f7c97dc6dbcd1ab0a8fe148918b04b677d408eb133737acf831ec02ad2ea13"
V365_SOURCE = "v365_te_prefix"
REPLAY_SOURCE = "v361_fulltext_replay"


class RowsDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class BoundaryPairCollator:
    """Encode one physical row, plus its auxiliary tE-1 view when present."""

    def __init__(self, base: LongDocCollator) -> None:
        self.base = base

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if len(rows) != 1:
            raise ValueError("E2-B boundary collator requires physical batch_size=1")
        main = dict(rows[0])
        has_pair = bool(main.get("boundary_pair_available", False))
        encoded_rows: list[dict[str, Any]] = [main]
        if has_pair:
            if (
                str(main.get("e2b_source")) != V365_SOURCE
                or int(main.get("binary_target", -1)) != 1
            ):
                raise ValueError("only V365 fraud tE rows may carry a boundary pair")
            pre_text = str(main.get("boundary_pre_model_input_text") or "")
            if not pre_text:
                raise ValueError("boundary pair is missing tE-1 model input text")
            pre = dict(main)
            pre.update(
                {
                    "sample_id": f"{main['sample_id']}__TE_MINUS_1_AUX",
                    "model_input_text": pre_text,
                    "prefix_text": pre_text,
                    "text": pre_text,
                    "input_view": "te_minus_1_auxiliary_boundary_view",
                    # These labels are tensor placeholders only.  No CE is
                    # computed for the auxiliary row.
                    "decision": "insufficient_evidence_auxiliary",
                    "binary_target": -1,
                    "binary_loss_mask": 0.0,
                    "binary_loss_weight": 0.0,
                    "sample_weight": 0.0,
                    "parent_loss_weight": 0.0,
                    "child_loss_weight": 0.0,
                    "disable_window_evidence": True,
                }
            )
            encoded_rows.append(pre)
        batch = self.base(encoded_rows)
        batch["boundary_pair_present"] = has_pair
        return batch


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


def forbidden(path: Path) -> bool:
    normalized = path.resolve(strict=False).as_posix().lower()
    return "restricted" in normalized or any(
        "test" in part for part in normalized.split("/")
    )


def require_path(path: Path, suffix: str, description: str) -> Path:
    resolved = path.resolve()
    if forbidden(resolved):
        raise SystemExit(f"refusing Test/restricted path: {resolved}")
    if not resolved.as_posix().endswith(suffix):
        raise SystemExit(f"refusing unexpected {description}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
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
        "strict_missing_keys": list(incompatible.missing_keys),
        "strict_unexpected_keys": list(incompatible.unexpected_keys),
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
    parser.add_argument("--boundary-margin", type=float, default=0.5)
    parser.add_argument("--boundary-rank-weight", type=float, default=0.20)
    parser.add_argument("--pre-alarm-cap-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--progress-every", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixed = {
        "chunk_length": 320,
        "overlap": 64,
        "head_context_length": 96,
        "max_chunks": 12,
        "encoder_micro_batch_size": 12,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "epochs": 1,
        "freeze_backbone_epochs": 1,
        "top_layers": 4,
        "encoder_learning_rate": 0.0,
        "task_learning_rate": 2e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "gradient_clip_norm": 1.0,
        "boundary_margin": 0.5,
        "boundary_rank_weight": 0.20,
        "pre_alarm_cap_weight": 0.10,
        "seed": 202,
    }
    mismatched = {
        key: {"expected": value, "actual": getattr(args, key)}
        for key, value in fixed.items()
        if getattr(args, key) != value
    }
    if mismatched:
        raise SystemExit(f"E2-B parameters differ from preregistration: {mismatched}")

    train_path = require_path(args.train_jsonl, EXPECTED_TRAIN_SUFFIX, "train file")
    init_path = require_path(
        args.init_longdoc_checkpoint, EXPECTED_M0_SUFFIX, "M0 checkpoint"
    )
    if sha256_file(init_path) != EXPECTED_M0_SHA256:
        raise ValueError("M0 checkpoint SHA-256 mismatch")
    if forbidden(args.model) or forbidden(args.v341_checkpoint) or forbidden(args.output):
        raise SystemExit("refusing a Test/restricted model or output path")
    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        raise FileNotFoundError(f"base model directory is invalid: {args.model}")
    if not args.v341_checkpoint.is_file():
        raise FileNotFoundError(args.v341_checkpoint)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output}")
    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes < 7 * 1024**3:
        raise RuntimeError(
            f"less than 7 GiB free for full E2-B checkpoint: {free_bytes / 1024**3:.2f} GiB"
        )

    rows = read_jsonl(train_path)
    if len(rows) != 6500:
        raise ValueError(f"expected 6500 training rows, got {len(rows)}")
    if any(float(row.get("binary_loss_mask", 0.0)) <= 0.0 for row in rows):
        raise ValueError("masked/ambiguous row exists in E2-B physical mixture")
    if any(int(row.get("binary_target", -1)) not in (0, 1) for row in rows):
        raise ValueError("invalid binary_target in E2-B physical mixture")
    source_counts = collections.Counter(str(row.get("e2b_source")) for row in rows)
    if source_counts != collections.Counter({V365_SOURCE: 3900, REPLAY_SOURCE: 2600}):
        raise ValueError(f"unexpected source mixture: {dict(source_counts)}")
    binary_counts = collections.Counter(int(row["binary_target"]) for row in rows)
    if binary_counts != collections.Counter({0: 3250, 1: 3250}):
        raise ValueError(f"E2-B physical mixture is not balanced: {dict(binary_counts)}")
    v365_rows = [row for row in rows if row.get("e2b_source") == V365_SOURCE]
    v365_binary = collections.Counter(int(row["binary_target"]) for row in v365_rows)
    if v365_binary != collections.Counter({0: 2559, 1: 1341}):
        raise ValueError(f"unexpected V365 draw classes: {dict(v365_binary)}")
    normal_roles = collections.Counter(
        str(row.get("e2b_normal_role"))
        for row in v365_rows
        if int(row["binary_target"]) == 0
    )
    if normal_roles != collections.Counter(
        {"same_scenario_fraud_draw_control": 1341, "broad_normal_coverage": 1218}
    ):
        raise ValueError(f"unexpected E2-B normal control roles: {dict(normal_roles)}")
    if any(
        row.get("scenario_tag") != row.get("matched_fraud_scenario_tag")
        for row in v365_rows
        if row.get("e2b_normal_role") == "same_scenario_fraud_draw_control"
    ):
        raise ValueError("a same-scenario normal control has mismatched scenario metadata")
    new_fraud_rows = [
        row
        for row in rows
        if row.get("e2b_source") == V365_SOURCE and int(row["binary_target"]) == 1
    ]
    if len(new_fraud_rows) != 1341:
        raise ValueError(f"expected 1341 V365 fraud draws, got {len(new_fraud_rows)}")
    paired_rows = [row for row in new_fraud_rows if row.get("boundary_pair_available")]
    unpaired_rows = [
        row for row in new_fraud_rows if not row.get("boundary_pair_available")
    ]
    if any(
        int(row.get("tE_turn", -1)) != 1
        or row.get("boundary_pair_exclusion_reason") != "tE_is_first_turn"
        for row in unpaired_rows
    ):
        raise ValueError("only tE=1 V365 fraud rows may lack a boundary pair")
    if any(
        row.get("e2b_source") != V365_SOURCE
        or int(row["binary_target"]) != 1
        or int(row["boundary_pre_end_turn"]) + 1 != int(row["tE_turn"])
        for row in paired_rows
    ):
        raise ValueError("invalid tE-1 boundary pair metadata")

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
    base_collator = LongDocCollator(
        tokenizer, chunk_config, child_code_to_index=child_mapping()
    )
    collator = BoundaryPairCollator(base_collator)
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
        raise RuntimeError("E2-B training requires CUDA")
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
    aggregate = collections.defaultdict(float)
    started = time.perf_counter()
    optimizer_step_count = 0
    observed_pair_steps = 0
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
            main_logits = outputs.binary_logits[:1]
            ce_loss = weighted_binary_loss(
                main_logits,
                batch["binary_target"][:1],
                batch["binary_weight"][:1],
            )
            rank_loss = torch.zeros((), device=device, dtype=torch.float32)
            cap_loss = torch.zeros((), device=device, dtype=torch.float32)
            if bool(batch["boundary_pair_present"]):
                if outputs.binary_logits.shape[0] != 2:
                    raise RuntimeError("paired row did not produce exactly two documents")
                margins = (
                    outputs.binary_logits[:, 1].float()
                    - outputs.binary_logits[:, 0].float()
                )
                rank_loss = F.relu(args.boundary_margin - (margins[0] - margins[1]))
                pre_probability = torch.sigmoid(margins[1])
                cap_loss = F.relu(pre_probability - 0.5)
                observed_pair_steps += 1
            elif outputs.binary_logits.shape[0] != 1:
                raise RuntimeError("unpaired row produced an unexpected document count")
            loss = (
                ce_loss
                + args.boundary_rank_weight * rank_loss
                + args.pre_alarm_cap_weight * cap_loss
            )
            if not torch.isfinite(outputs.binary_logits.float()).all():
                raise FloatingPointError(f"non-finite binary logits at step {step}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite E2-B loss at step {step}")
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
        aggregate["total_loss"] += float(loss.detach().cpu())
        aggregate["ce_loss"] += float(ce_loss.detach().cpu())
        if bool(batch["boundary_pair_present"]):
            aggregate["rank_loss_on_pairs"] += float(rank_loss.detach().cpu())
            aggregate["cap_loss_on_pairs"] += float(cap_loss.detach().cpu())
        if step % args.progress_every == 0 or step == len(loader):
            elapsed = (time.perf_counter() - started) / 60.0
            pair_denominator = max(observed_pair_steps, 1)
            print(
                f"[E2-B epoch 1/1] step {step}/{len(loader)} "
                f"loss={aggregate['total_loss'] / step:.6f} "
                f"ce={aggregate['ce_loss'] / step:.6f} "
                f"rank_pair={aggregate['rank_loss_on_pairs'] / pair_denominator:.6f} "
                f"cap_pair={aggregate['cap_loss_on_pairs'] / pair_denominator:.6f} "
                f"pairs={observed_pair_steps} elapsed={elapsed:.1f}m",
                flush=True,
            )

    if optimizer_step_count != optimizer_steps:
        raise RuntimeError(
            f"optimizer step mismatch: {optimizer_step_count} != {optimizer_steps}"
        )
    if observed_pair_steps != len(paired_rows):
        raise RuntimeError(
            f"boundary pair step mismatch: {observed_pair_steps} != {len(paired_rows)}"
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
        raise RuntimeError("no trainable E2-B parameter changed")

    elapsed_seconds = time.perf_counter() - started
    pair_denominator = max(observed_pair_steps, 1)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metrics = {
        "epoch": 1,
        "task": "binary fraud detection with adjacent tE boundary regularisation",
        "backbone_stage": stage,
        "train_loss": aggregate["total_loss"] / len(loader),
        "ce_loss": aggregate["ce_loss"] / len(loader),
        "mean_rank_loss_on_pairs": aggregate["rank_loss_on_pairs"] / pair_denominator,
        "mean_cap_loss_on_pairs": aggregate["cap_loss_on_pairs"] / pair_denominator,
        "boundary_pair_steps": observed_pair_steps,
        "unpaired_te1_main_only_steps": len(unpaired_rows),
        "train_seconds": elapsed_seconds,
        "train_minutes": elapsed_seconds / 60.0,
        "optimizer_steps": optimizer_step_count,
        "warmup_steps": warmup_steps,
        "trainable": summary,
        "changed_trainable_tensors": len(changed),
        "unchanged_trainable_tensors": len(unchanged),
    }
    checkpoint_path = args.output / "best_model.pt"
    temporary_checkpoint = args.output / "best_model.pt.tmp"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "e2b_config": config,
            "init_checkpoint": init_report,
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest = {
        "schema_version": "v365.e2b.binary_training.v1",
        "status": "complete",
        "task": "binary fraud detection only",
        "score": "p_binary[1]",
        "loss": {
            "main": "binary cross entropy over the 6,500 physical rows",
            "rank": "relu(0.5 - (margin_tE - margin_tE_minus_1))",
            "pre_alarm_cap": "relu(sigmoid(margin_tE_minus_1) - 0.5)",
            "rank_weight": args.boundary_rank_weight,
            "pre_alarm_cap_weight": args.pre_alarm_cap_weight,
            "auxiliary_hard_normal_label": False,
            "unpaired_te1_policy": "main CE only; no empty auxiliary prefix",
        },
        "selection_policy": "none; fixed one epoch, saved final state",
        "dev_data_used_during_training": False,
        "test_data_used": False,
        "train_jsonl": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "binary_counts": {str(k): v for k, v in sorted(binary_counts.items())},
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
        f"E2-B training complete: loss={metrics['train_loss']:.6f} "
        f"minutes={metrics['train_minutes']:.2f} checkpoint={checkpoint_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
