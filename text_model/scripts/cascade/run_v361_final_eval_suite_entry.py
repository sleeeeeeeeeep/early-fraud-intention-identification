#!/usr/bin/env python3
"""V361 冻结模型在 Test-B 与 Long-Stress 上的最终对照评测。"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
BASELINE = "baseline_256"
ROUTE_B = "route_b_head96"
ROUTE_A = "route_a"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def validate_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    data, evaluation = spec.get("data"), spec.get("evaluation")
    matrix = spec.get("subruns", {}).get("matrix")
    if not isinstance(data, Mapping) or not data.get("eval_csvs", {}).get("fraud12k_eval") or not data.get("long_dev_jsonl"):
        raise ValueError("final evaluation requires Test-B CSV and Long-Stress JSONL")
    if not isinstance(evaluation, Mapping) or not evaluation.get("checkpoint_path"):
        raise ValueError("final evaluation requires the frozen V341 checkpoint")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("final evaluation requires at least one subrun")
    kinds = {str(row.get("kind")) for row in matrix}
    if kinds == {BASELINE, ROUTE_B}:
        if len(matrix) != 2 or not evaluation.get("longdoc_checkpoint_path"):
            raise ValueError("baseline/Route-B evaluation requires two frozen checkpoints")
    elif kinds == {ROUTE_A}:
        checkpoint_paths = evaluation.get("longdoc_checkpoint_paths")
        if not isinstance(checkpoint_paths, Mapping) or any(str(row.get("name")) not in checkpoint_paths for row in matrix):
            raise ValueError("Route-A evaluation requires one frozen checkpoint per subrun")
    else:
        raise ValueError("unsupported final-evaluation subrun combination")
    if len({str(row.get("gpu")) for row in matrix}) != len(matrix):
        raise ValueError("final-evaluation subruns must use separate GPUs")
    return [dict(row) for row in matrix]


def execution_path(path: str) -> str:
    return path if (ROOT / path).exists() else str(Path("text_model") / path)


def build_commands(spec: Mapping[str, Any], item: Mapping[str, Any], output: Path) -> list[list[str]]:
    data, model, evaluation = spec["data"], spec["model"], spec["evaluation"]
    evaluations = [
        ("test_b", str(data["eval_csvs"]["fraud12k_eval"]), "csv"),
        ("long_stress", str(data["long_dev_jsonl"]), "jsonl"),
    ]
    commands: list[list[str]] = []
    for name, path, input_format in evaluations:
        target = output / name
        if item["kind"] == BASELINE:
            command = [
                sys.executable, "-m", "scripts.inference.evaluate_saved_ce_checkpoint",
                "--checkpoint", str(evaluation["checkpoint_path"]),
                "--model", str(model["model"]),
                "--architecture", str(model.get("baseline_architecture", "ce_dual_head")),
                f"--eval-{input_format}", path,
                "--eval-name", name,
                "--output", str(target),
                "--max-length", str(model.get("baseline_max_length", 256)),
                "--eval-batch-size", str(model.get("baseline_eval_batch_size", 16)),
                "--official-subcategory-loss-weight", str(model.get("official_subcategory_loss_weight", 0.2)),
            ]
        else:
            checkpoint_paths = evaluation.get("longdoc_checkpoint_paths", {})
            longdoc_checkpoint = checkpoint_paths.get(str(item["name"]), evaluation.get("longdoc_checkpoint_path"))
            command = [
                sys.executable, execution_path(str(model["v361_inference_script"])),
                "--input-jsonl", path,
                "--model", str(model["model"]),
                "--v341-checkpoint", str(evaluation["checkpoint_path"]),
                "--longdoc-checkpoint", str(longdoc_checkpoint),
                "--route", "A" if item["kind"] == ROUTE_A else "B",
                "--output-dir", str(target),
                "--input-mode", str(item.get("input_mode", model.get("input_mode", "head_context_enhanced_chunks"))),
                "--chunk-length", str(model.get("chunk_length", 320)),
                "--overlap", str(model.get("overlap", 64)),
                "--head-context-length", str(item.get("head_context_length", model.get("head_context_length", 96))),
                "--encoder-micro-batch-size", str(model.get("encoder_micro_batch_size", 12)),
                "--evidence-components", str(model.get("evidence_components", "attention,top1,top2,last,mean")),
                "--batch-size", str(model.get("eval_batch_size", 2)),
            ]
            if name == "test_b":
                command.append("--allow-final-test-b")
        commands.append(command)
    return commands


def run_queue(spec: Mapping[str, Any], item: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    name, gpu = str(item["name"]), str(item["gpu"])
    output = run_dir / "subruns" / name
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "USE_TF": "0",
        "TRANSFORMERS_NO_TF": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src" / "v361"), str(ROOT), environment.get("PYTHONPATH", ""))),
    })
    started, evaluations = time.time(), {}
    for command in build_commands(spec, item, output):
        eval_name = Path(command[command.index("--output-dir") + 1] if "--output-dir" in command else command[command.index("--output") + 1]).name
        target = output / eval_name
        target.mkdir(parents=True, exist_ok=True)
        (target / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        with (target / "stdout.log").open("w", encoding="utf-8") as stdout, (target / "stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        metrics = read_json(target / "metrics.json") if (target / "metrics.json").exists() else {}
        evaluations[eval_name] = {"returncode": result.returncode, "metrics": metrics}
        if result.returncode:
            break
    return {
        "name": name,
        "kind": item["kind"],
        "gpu": gpu,
        "status": "done" if len(evaluations) == 2 and all(row["returncode"] == 0 for row in evaluations.values()) else "failed",
        "elapsed_sec": round(time.time() - started, 3),
        "evaluations": evaluations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    spec_path = args.experiment_spec if args.experiment_spec.is_absolute() else ROOT / args.experiment_spec
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    spec, run_dir = read_json(spec_path), run_dir
    matrix = validate_spec(spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "experiment_spec.json", spec)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: run_queue(spec, item, run_dir), matrix))
    returncode = 0 if all(row["status"] == "done" for row in results) else 1
    payload = {"status": "done" if returncode == 0 else "failed", "returncode": returncode, "runs": results}
    write_json(run_dir / "metrics.json", payload)
    write_json(run_dir / "remote_status.json", payload)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
