from __future__ import annotations

from pathlib import Path

import pytest

from text_model.scripts.cascade.run_v361_longdoc_suite_entry import (
    build_command,
    validate_spec,
)


def _spec() -> dict:
    return {
        "data": {
            "train_jsonl": "data/processed/v361/train.jsonl",
            "external_dev_a2_jsonl": "data/processed/v314/dev_a_complete_1801.jsonl",
            "long_dev_jsonl": "data/processed/v361/long_dev.jsonl",
        },
        "evaluation": {
            "checkpoint_path": "runs/v341/best_model/pytorch_model.bin"
        },
        "model": {
            "model": "models/megatronbert",
            "v361_training_script": "inference/train_longdoc.py",
            "route": "B",
            "input_mode": "plain_sliding_chunks",
            "chunk_length": 320,
            "overlap": 64,
            "head_context_length": 64,
            "evidence_components": "attention,top1,top2,last,mean",
            "epochs": 2,
            "gradient_checkpointing": True,
            "amp": True,
            "max_train_rows": 6500,
            "asr_view_fraction": 0.5,
        },
        "subruns": {"matrix": [{"name": "b320_s202", "gpu": 0, "seed": 202}]},
    }


def test_v361_suite_builds_guarded_longdoc_command():
    spec = _spec()
    matrix = validate_spec(spec)
    command = build_command(spec, matrix[0], Path("runs/v361/subruns/b320_s202"))
    assert "--train-jsonl" in command
    assert "--long-dev-jsonl" in command
    assert "--v341-checkpoint" in command
    assert command[command.index("--route") + 1] == "B"
    assert command[command.index("--chunk-length") + 1] == "320"
    assert "--gradient-checkpointing" in command
    assert "--amp" in command
    assert command[command.index("--max-train-rows") + 1] == "6500"
    assert command[command.index("--asr-view-fraction") + 1] == "0.5"


def test_v361_suite_rejects_test_b_and_oof_inputs():
    spec = _spec()
    spec["data"]["long_dev_jsonl"] = "data/test_b_1580.jsonl"
    with pytest.raises(ValueError, match="rejects"):
        validate_spec(spec)
    spec = _spec()
    spec["data"]["train_jsonl"] = "data/oof_fold0.jsonl"
    with pytest.raises(ValueError, match="rejects"):
        validate_spec(spec)


def test_v361_suite_allows_four_gpu_queues_and_rejects_a_fifth_gpu():
    spec = _spec()
    spec["subruns"]["matrix"].append(
        {"name": "a320_s42", "route": "A", "gpu": 0, "seed": 42}
    )
    spec["subruns"]["matrix"].append(
        {"name": "a320_s13", "route": "A", "gpu": 1, "seed": 13}
    )
    spec["subruns"]["matrix"].append(
        {"name": "a320_s7", "route": "A", "gpu": 2, "seed": 7}
    )
    spec["subruns"]["matrix"].append(
        {"name": "a320_s11", "route": "A", "gpu": 3, "seed": 11}
    )
    accepted = validate_spec(spec)
    assert [item["gpu"] for item in accepted] == [0, 0, 1, 2, 3]

    spec["subruns"]["matrix"].append(
        {"name": "a320_s19", "route": "A", "gpu": 4, "seed": 19}
    )
    with pytest.raises(ValueError, match="at most four GPUs"):
        validate_spec(spec)
