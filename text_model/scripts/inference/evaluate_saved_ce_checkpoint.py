#!/usr/bin/env python3
"""Inference-only evaluation for a saved CE text checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from . import run_ce_supervised_13class_external_eval as core
except ImportError:
    import run_ce_supervised_13class_external_eval as core


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_file(value: str) -> Path:
    path = Path(value)
    return path / "pytorch_model.bin" if path.is_dir() else path


def load_evaluation_rows(
    eval_csv: str | None,
    eval_jsonl: str | None,
) -> tuple[Path, list[dict[str, Any]], str]:
    """Load exactly one evaluation input without silently changing its format."""
    if bool(eval_csv) == bool(eval_jsonl):
        raise ValueError("exactly one of eval_csv or eval_jsonl is required")
    if eval_csv:
        path = Path(eval_csv)
        return path, core.read_eval_csv(path), "csv"
    path = Path(str(eval_jsonl))
    return path, core.read_jsonl(path), "jsonl"


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        architecture=args.architecture,
        ce_neck_dropout=args.ce_neck_dropout,
        ce_neck_gate_bias=args.ce_neck_gate_bias,
        ce_mid_adapter_layer=args.ce_mid_adapter_layer,
        ce_mid_adapter_bottleneck=args.ce_mid_adapter_bottleneck,
        official_subcategory_loss_weight=args.official_subcategory_loss_weight,
        official_subcategory_loss_mode="flat",
        clear11_detach=False,
        residual12_detach=False,
        protected_subtype_weight=0.0,
        protected_gate_threshold=0.5,
        protected_official_weight=0.0,
        protected_official_rng_reference_head_classes=13,
        protected_fraud_top_layers=0,
        local_expert_residual_scale=1.0,
        local_expert_gate_bias=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--architecture", required=True)
    evaluation_input = parser.add_mutually_exclusive_group(required=True)
    evaluation_input.add_argument("--eval-csv")
    evaluation_input.add_argument("--eval-jsonl")
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--ce-neck-dropout", type=float, default=-1.0)
    parser.add_argument("--ce-neck-gate-bias", type=float, default=-1.0)
    parser.add_argument("--ce-mid-adapter-layer", type=int, default=-1)
    parser.add_argument("--ce-mid-adapter-bottleneck", type=int, default=256)
    parser.add_argument("--official-subcategory-loss-weight", type=float, default=0.0)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    checkpoint = checkpoint_file(args.checkpoint)
    eval_path, rows, eval_input_format = load_evaluation_rows(args.eval_csv, args.eval_jsonl)
    output = Path(args.output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    dataset = core.TextDataset(rows, tokenizer, args.max_length, args.eval_name, args.eval_name)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = core.build_model_from_args(model_args(args), [], device)
    state_dict = torch.load(checkpoint, map_location=device)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint did not load exactly: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    metrics, predictions = core.predict(model, dataset, args.eval_batch_size, device)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = core.write_prediction_artifacts(output, args.eval_name, predictions)
    core.write_json(output / "metrics.json", {
        "mode": "inference_only_saved_ce_checkpoint",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model": args.model,
        "architecture": args.architecture,
        "max_length": args.max_length,
        "eval_input": str(eval_path),
        "eval_input_format": eval_input_format,
        "eval_input_sha256": sha256_file(eval_path),
        "eval_name": args.eval_name,
        "eval_rows": len(rows),
        "metrics": metrics,
        "prediction_artifacts": {args.eval_name: artifacts},
    })


if __name__ == "__main__":
    main()
