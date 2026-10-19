from pathlib import Path

from text_model.scripts.cascade.run_v361_final_eval_suite_entry import (
    BASELINE,
    ROUTE_A,
    ROUTE_B,
    build_commands,
    validate_spec,
)


def _spec() -> dict:
    return {
        "data": {"eval_csvs": {"fraud12k_eval": "data/test_b.csv"}, "long_dev_jsonl": "data/long_stress.jsonl"},
        "evaluation": {
            "checkpoint_path": "runs/v341/pytorch_model.bin",
            "longdoc_checkpoint_path": "runs/v361/best_model.pt",
            "inference_script": "scripts/inference/evaluate_saved_ce_checkpoint.py",
        },
        "model": {"model": "models/megatronbert", "v361_inference_script": "inference/run_longdoc_inference.py"},
        "subruns": {"matrix": [
            {"name": "route_b", "kind": ROUTE_B, "gpu": 2},
            {"name": "baseline", "kind": BASELINE, "gpu": 3},
        ]},
    }


def test_final_eval_builds_four_frozen_inference_commands():
    spec = _spec()
    route, baseline = validate_spec(spec)
    route_commands = build_commands(spec, route, Path("runs/final/route"))
    baseline_commands = build_commands(spec, baseline, Path("runs/final/baseline"))
    assert len(route_commands) == len(baseline_commands) == 2
    assert "--longdoc-checkpoint" in route_commands[0]
    assert "--allow-final-test-b" in route_commands[0]
    assert "--max-length" in baseline_commands[0]
    assert baseline_commands[0][baseline_commands[0].index("--max-length") + 1] == "256"


def test_route_a_uses_each_frozen_checkpoint_and_input_mode():
    spec = _spec()
    spec["evaluation"]["longdoc_checkpoint_paths"] = {
        "a320_plain": "runs/v361/a320_plain/best_model.pt",
        "a320_head64": "runs/v361/a320_head64/best_model.pt",
    }
    spec["subruns"]["matrix"] = [
        {"name": "a320_plain", "kind": ROUTE_A, "gpu": 2, "input_mode": "plain_sliding_chunks", "head_context_length": 64},
        {"name": "a320_head64", "kind": ROUTE_A, "gpu": 3, "input_mode": "head_context_enhanced_chunks", "head_context_length": 64},
    ]
    plain, head64 = validate_spec(spec)
    plain_command = build_commands(spec, plain, Path("runs/final/plain"))[0]
    head64_command = build_commands(spec, head64, Path("runs/final/head64"))[0]
    assert plain_command[plain_command.index("--route") + 1] == "A"
    assert plain_command[plain_command.index("--input-mode") + 1] == "plain_sliding_chunks"
    assert plain_command[plain_command.index("--longdoc-checkpoint") + 1].endswith("a320_plain/best_model.pt")
    assert head64_command[head64_command.index("--input-mode") + 1] == "head_context_enhanced_chunks"
    assert head64_command[head64_command.index("--longdoc-checkpoint") + 1].endswith("a320_head64/best_model.pt")
