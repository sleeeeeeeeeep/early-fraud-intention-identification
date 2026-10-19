#!/usr/bin/env python3
"""V361 阶段一：V341 原始截断基线与固定滑窗聚合对照。"""

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


SCRIPT_PATH = Path(__file__).resolve()
ROOT_CANDIDATES = (SCRIPT_PATH.parents[3], SCRIPT_PATH.parents[2])
ROOT = next(
    (
        candidate
        for candidate in ROOT_CANDIDATES
        if (candidate / "configs").exists() and (candidate / "data").exists()
    ),
    SCRIPT_PATH.parents[2],
)
BASELINE_KIND = "v341_baseline_256"
FIXED_KIND = "fixed_320_all"
SUPPORTED_KINDS = {BASELINE_KIND, FIXED_KIND}


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def execution_path(relative_path: str) -> str:
    if (ROOT / relative_path).exists():
        return relative_path
    nested = Path("text_model") / relative_path
    if (ROOT / nested).exists():
        return str(nested)
    return relative_path


def _reject_selection_leak(field: str, value: Any) -> None:
    lowered = str(value).lower().replace("-", "_")
    if "test_b" in lowered or "testb" in lowered or "oof" in lowered:
        raise ValueError(f"V361 phase-1 rejects {field}: {value}")


def validate_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = spec.get("data")
    model = spec.get("model")
    evaluation = spec.get("evaluation")
    matrix = spec.get("phase1", {}).get("jobs")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("V361 phase-1 requires object-valued data and model")
    if not isinstance(evaluation, Mapping):
        raise ValueError("V361 phase-1 requires evaluation settings")
    for key in ("external_dev_a2_jsonl", "long_dev_jsonl"):
        if not data.get(key):
            raise ValueError(f"V361 phase-1 is missing data.{key}")
        _reject_selection_leak(f"data.{key}", data[key])
    checkpoint = evaluation.get("checkpoint_path")
    if not checkpoint:
        raise ValueError("V361 phase-1 is missing evaluation.checkpoint_path")
    _reject_selection_leak("evaluation.checkpoint_path", checkpoint)
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("V361 phase-1 requires a non-empty phase1.jobs list")

    names: set[str] = set()
    gpus: set[int] = set()
    seen_pairs: set[tuple[str, str]] = set()
    validated: list[dict[str, Any]] = []
    for raw in matrix:
        if not isinstance(raw, Mapping):
            raise ValueError("every V361 phase-1 subrun must be an object")
        item = dict(raw)
        name = str(item.get("name") or "").strip()
        kind = str(item.get("kind") or "").strip()
        data_key = str(item.get("data_key") or "").strip()
        gpu = int(item.get("gpu", -1))
        if not name or name in names:
            raise ValueError(f"invalid or duplicate subrun name: {name!r}")
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"{name}: unsupported kind {kind!r}")
        if data_key not in {"external_dev_a2_jsonl", "long_dev_jsonl"}:
            raise ValueError(f"{name}: unsupported data_key {data_key!r}")
        if gpu < 0:
            raise ValueError(f"{name}: gpu must be a non-negative integer")
        pair = (kind, data_key)
        if pair in seen_pairs:
            raise ValueError(f"duplicate phase-1 comparison: {pair}")
        names.add(name)
        gpus.add(gpu)
        seen_pairs.add(pair)
        validated.append(item)
    if len(validated) > 4:
        raise ValueError("V361 phase-1 supports at most four subruns")
    if len(gpus) > 2:
        raise ValueError("V361 phase-1 uses at most two GPUs")
    return validated


def _add_arg(command: list[str], flag: str, value: Any) -> None:
    if value not in (None, ""):
        command.extend((flag, str(value)))


def build_command(
    spec: Mapping[str, Any], item: Mapping[str, Any], output: Path
) -> list[str]:
    data = spec["data"]
    model = spec["model"]
    evaluation = spec["evaluation"]
    input_jsonl = str(data[str(item["data_key"])])
    checkpoint = str(evaluation["checkpoint_path"])
    if item["kind"] == BASELINE_KIND:
        script = str(
            evaluation.get(
                "inference_script",
                "scripts/inference/evaluate_saved_ce_checkpoint.py",
            )
        )
        command = [
            sys.executable,
            execution_path(script),
            "--checkpoint",
            checkpoint,
            "--model",
            str(model["model"]),
            "--architecture",
            str(model.get("baseline_architecture", "ce_dual_head")),
            "--eval-jsonl",
            input_jsonl,
            "--eval-name",
            str(item.get("eval_name") or item["data_key"]),
            "--output",
            str(output),
            "--max-length",
            str(model.get("baseline_max_length", 256)),
            "--eval-batch-size",
            str(model.get("baseline_eval_batch_size", 16)),
        ]
        for flag, key in (
            ("--ce-neck-dropout", "ce_neck_dropout"),
            ("--ce-neck-gate-bias", "ce_neck_gate_bias"),
            ("--ce-mid-adapter-layer", "ce_mid_adapter_layer"),
            ("--ce-mid-adapter-bottleneck", "ce_mid_adapter_bottleneck"),
            (
                "--official-subcategory-loss-weight",
                "official_subcategory_loss_weight",
            ),
        ):
            _add_arg(command, flag, model.get(key))
        return command

    script = str(
        model.get("v361_inference_script", "inference/run_longdoc_inference.py")
    )
    return [
        sys.executable,
        execution_path(script),
        "--input-jsonl",
        input_jsonl,
        "--model",
        str(model["model"]),
        "--v341-checkpoint",
        checkpoint,
        "--route",
        "B",
        "--fixed-pooling",
        "all",
        "--output-dir",
        str(output),
        "--input-mode",
        str(model.get("input_mode", "plain_sliding_chunks")),
        "--chunk-length",
        str(model.get("chunk_length", 320)),
        "--overlap",
        str(model.get("overlap", 64)),
        "--head-context-length",
        str(model.get("head_context_length", 64)),
        "--encoder-micro-batch-size",
        str(model.get("encoder_micro_batch_size", 12)),
        "--evidence-components",
        str(model.get("evidence_components", "attention,top1,top2,last,mean")),
        "--batch-size",
        str(model.get("fixed_eval_batch_size", 2)),
    ]


def run_one(
    spec: Mapping[str, Any], item: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    name = str(item["name"])
    output = run_dir / "subruns" / name
    output.mkdir(parents=True, exist_ok=True)
    command = build_command(spec, item, output)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(item["gpu"]),
            "USE_TF": "0",
            "TRANSFORMERS_NO_TF": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ROOT / "src" / "v361"),
            str(ROOT),
            str(ROOT / "scripts" / "cascade"),
            existing_python_path,
        )
        if part
    )
    (output / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = time.time()
    write_json(
        output / "status.json",
        {
            "status": "running",
            "name": name,
            "kind": item["kind"],
            "gpu": item["gpu"],
            "started_at": started,
        },
    )
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
        )
    result: dict[str, Any] = {
        "status": "done" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "name": name,
        "kind": item["kind"],
        "data_key": item["data_key"],
        "gpu": item["gpu"],
        "elapsed_sec": round(time.time() - started, 3),
        "output": str(output.relative_to(ROOT)),
    }
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        result["metrics"] = read_json(metrics_path)
    if process.returncode:
        result["stderr_tail"] = (output / "stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )[-6000:]
    write_json(output / "status.json", result)
    return result


def run_gpu_queue(
    spec: Mapping[str, Any], queue: list[dict[str, Any]], run_dir: Path
) -> list[dict[str, Any]]:
    """Run one GPU's subruns back to back so a single GPU never hosts two jobs."""
    return [run_one(spec, item, run_dir) for item in queue]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    spec_path = (
        args.experiment_spec
        if args.experiment_spec.is_absolute()
        else ROOT / args.experiment_spec
    )
    spec = read_json(spec_path)
    matrix = validate_spec(spec)
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "experiment_spec.json", spec)
    write_json(
        run_dir / "suite_status.json",
        {"status": "running", "phase": "v361_fixed_controls", "started_at": time.time()},
    )
    queues: dict[int, list[dict[str, Any]]] = {}
    for item in matrix:
        queues.setdefault(int(item["gpu"]), []).append(item)
    max_parallel = min(int(spec.get("phase1", {}).get("max_parallel", 2)), len(queues))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        pending = [
            executor.submit(run_gpu_queue, spec, queue, run_dir) for queue in queues.values()
        ]
        for future in concurrent.futures.as_completed(pending):
            results.extend(future.result())
            write_json(
                run_dir / "metrics.json",
                {"runs": sorted(results, key=lambda row: str(row["name"]))},
            )
    results.sort(key=lambda row: str(row["name"]))
    returncode = 0 if all(row["returncode"] == 0 for row in results) else 1
    write_json(
        run_dir / "suite_status.json",
        {
            "status": "done" if returncode == 0 else "failed",
            "phase": "done",
            "returncode": returncode,
            "runs": results,
        },
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
