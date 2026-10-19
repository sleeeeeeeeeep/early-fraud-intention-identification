from __future__ import annotations

from pathlib import Path

import pytest

from text_model.scripts.cascade.run_v361_fixed_controls_suite_entry import (
    BASELINE_KIND,
    FIXED_KIND,
    build_command,
    validate_spec,
)


def _spec() -> dict:
    return {
        "data": {
            "external_dev_a2_jsonl": "data/processed/v346/dev_a2.jsonl",
            "long_dev_jsonl": "data/processed/v361/long_dev.jsonl",
        },
        "evaluation": {
            "checkpoint_path": "runs/v341/best_model/pytorch_model.bin",
            "inference_script": "scripts/inference/evaluate_saved_ce_checkpoint.py",
        },
        "model": {
            "model": "models/megatronbert",
            "baseline_architecture": "ce_dual_head",
            "baseline_max_length": 256,
            "baseline_eval_batch_size": 16,
            "v361_inference_script": "inference/run_longdoc_inference.py",
            "chunk_length": 320,
            "overlap": 64,
            "fixed_eval_batch_size": 2,
        },
        "phase1": {
            "max_parallel": 2,
            "jobs": [
                {
                    "name": "baseline_dev",
                    "kind": BASELINE_KIND,
                    "data_key": "external_dev_a2_jsonl",
                    "eval_name": "dev_a2_1801",
                    "gpu": 0,
                },
                {
                    "name": "fixed_long_dev",
                    "kind": FIXED_KIND,
                    "data_key": "long_dev_jsonl",
                    "gpu": 1,
                },
            ]
        },
    }


def test_phase1_commands_keep_exact_baseline_separate_from_full_text_controls():
    spec = _spec()
    baseline, fixed = validate_spec(spec)
    baseline_command = build_command(spec, baseline, Path("runs/v361/baseline"))
    fixed_command = build_command(spec, fixed, Path("runs/v361/fixed"))
    assert baseline_command[baseline_command.index("--max-length") + 1] == "256"
    assert "--fixed-pooling" not in baseline_command
    assert fixed_command[fixed_command.index("--fixed-pooling") + 1] == "all"
    assert fixed_command[fixed_command.index("--chunk-length") + 1] == "320"
    assert fixed_command[fixed_command.index("--overlap") + 1] == "64"


def test_phase1_rejects_test_b_oof_and_more_than_two_gpus():
    spec = _spec()
    spec["data"]["long_dev_jsonl"] = "data/test_b_1580.jsonl"
    with pytest.raises(ValueError, match="rejects"):
        validate_spec(spec)

    spec = _spec()
    spec["data"]["external_dev_a2_jsonl"] = "data/oof_fold0.jsonl"
    with pytest.raises(ValueError, match="rejects"):
        validate_spec(spec)

    spec = _spec()
    spec["phase1"]["jobs"].append(
        {
            "name": "baseline_long_dev",
            "kind": BASELINE_KIND,
            "data_key": "long_dev_jsonl",
            "eval_name": "long_dev_240",
            "gpu": 2,
        }
    )
    with pytest.raises(ValueError, match="at most two GPUs"):
        validate_spec(spec)
