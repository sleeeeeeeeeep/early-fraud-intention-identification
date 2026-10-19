#!/usr/bin/env python3
"""V361 文本长上下文路线 A/B 的受控四卡训练入口。"""

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


def resolved(item: Mapping[str, Any], model: Mapping[str, Any], key: str) -> Any:
    return item[key] if key in item else model.get(key)


def add_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, (list, tuple)):
        value = ",".join(map(str, value))
    command.extend((flag, str(value)))


def validate_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = spec.get("data")
    model = spec.get("model")
    matrix = spec.get("subruns", {}).get("matrix")
    if not isinstance(data, Mapping) or not isinstance(model, Mapping):
        raise ValueError("V361 requires object-valued data and model")
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("V361 requires a non-empty subrun matrix")
    required_data = ("train_jsonl", "long_dev_jsonl")
    for key in required_data:
        if not data.get(key):
            raise ValueError(f"V361 is missing data.{key}")
        lowered = str(data[key]).lower().replace("-", "_")
        if "test_b" in lowered or "testb" in lowered or "oof" in lowered:
            raise ValueError(f"V361 training rejects data.{key}: {data[key]}")
    dev_path = data.get("external_dev_a2_jsonl") or data.get("dev_a2_jsonl")
    if not dev_path:
        raise ValueError("V361 is missing data.external_dev_a2_jsonl")
    checkpoint = (
        spec.get("evaluation", {}).get("checkpoint_path")
        if isinstance(spec.get("evaluation"), Mapping)
        else None
    ) or data.get("v341_checkpoint")
    if not checkpoint:
        raise ValueError("V361 is missing evaluation.checkpoint_path")
    for field, value in (
        ("data.external_dev_a2_jsonl", dev_path),
        ("evaluation.checkpoint_path", checkpoint),
    ):
        lowered = str(value).lower().replace("-", "_")
        if "test_b" in lowered or "testb" in lowered or "oof" in lowered:
            raise ValueError(f"V361 training rejects {field}: {value}")
    seen_names: set[str] = set()
    for raw in matrix:
        if not isinstance(raw, Mapping):
            raise ValueError("every V361 subrun must be an object")
        name = str(raw.get("name") or "").strip()
        if not name or name in seen_names:
            raise ValueError(f"invalid or duplicate V361 subrun name: {name!r}")
        seen_names.add(name)
        if str(resolved(raw, model, "route")) not in {"A", "B"}:
            raise ValueError(f"{name}: route must be A or B")
        if int(raw.get("gpu", -1)) < 0:
            raise ValueError(f"{name}: gpu must be non-negative")
        if int(raw.get("seed", -1)) < 0:
            raise ValueError(f"{name}: seed must be non-negative")
    gpus = [int(item["gpu"]) for item in matrix]
    if len(set(gpus)) > 4:
        raise ValueError("one V361 suite uses at most four GPUs")
    return [dict(item) for item in matrix]


def build_command(
    spec: Mapping[str, Any], item: Mapping[str, Any], output: Path
) -> list[str]:
    data = spec["data"]
    model = spec["model"]
    dev_path = data.get("external_dev_a2_jsonl") or data.get("dev_a2_jsonl")
    evaluation = spec.get("evaluation", {})
    checkpoint = (
        evaluation.get("checkpoint_path")
        if isinstance(evaluation, Mapping)
        else None
    ) or data.get("v341_checkpoint")
    command = [
        sys.executable,
        execution_path(
            str(
                model.get(
                    "v361_training_script",
                    model.get("training_script", "inference/train_longdoc.py"),
                )
            )
        ),
        "--train-jsonl",
        str(data["train_jsonl"]),
        "--dev-jsonl",
        str(dev_path),
        "--long-dev-jsonl",
        str(data["long_dev_jsonl"]),
        "--model",
        str(model["model"]),
        "--v341-checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--route",
        str(resolved(item, model, "route")),
        "--seed",
        str(item["seed"]),
    ]
    for flag, key in (
        ("--input-mode", "input_mode"),
        ("--chunk-length", "chunk_length"),
        ("--overlap", "overlap"),
        ("--head-context-length", "head_context_length"),
        ("--diagnostic-max-chunks", "diagnostic_max_chunks"),
        ("--encoder-micro-batch-size", "encoder_micro_batch_size"),
        ("--epochs", "epochs"),
        ("--patience", "patience"),
        ("--batch-size", "batch_size"),
        ("--eval-batch-size", "eval_batch_size"),
        ("--gradient-accumulation-steps", "gradient_accumulation_steps"),
        ("--encoder-learning-rate", "encoder_learning_rate"),
        ("--new-learning-rate", "new_learning_rate"),
        ("--weight-decay", "weight_decay"),
        ("--warmup-ratio", "warmup_ratio"),
        ("--projection-dim", "projection_dim"),
        ("--fusion-hidden", "fusion_hidden"),
        ("--document-layers", "document_layers"),
        ("--document-heads", "document_heads"),
        ("--document-intermediate", "document_intermediate"),
        ("--evidence-components", "evidence_components"),
        ("--freeze-backbone-epochs", "freeze_backbone_epochs"),
        ("--top-layer-epochs", "top_layer_epochs"),
        ("--top-layers", "top_layers"),
        ("--long-exposure", "long_exposure"),
        ("--mil-weight", "mil_weight"),
        ("--window-evidence-weight", "window_evidence_weight"),
        ("--train-limit", "train_limit"),
        ("--dev-limit", "dev_limit"),
        ("--long-dev-limit", "long_dev_limit"),
        ("--max-train-rows", "max_train_rows"),
        ("--asr-view-fraction", "asr_view_fraction"),
    ):
        add_arg(command, flag, resolved(item, model, key))
    if bool(resolved(item, model, "amp")):
        command.append("--amp")
    if bool(resolved(item, model, "gradient_checkpointing")):
        command.append("--gradient-checkpointing")
    if bool(resolved(item, model, "metrics_only")):
        command.append("--metrics-only")
    if bool(resolved(item, model, "save_every_epoch")):
        command.append("--save-every-epoch")
    return command


def run_one(
    spec: Mapping[str, Any], item: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    name = str(item["name"])
    output = run_dir / "subruns" / name
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
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
    existing_python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ROOT / "src" / "v361"),
            str(ROOT),
            str(ROOT / "scripts" / "cascade"),
            existing_python_path,
        )
        if part
    )
    command = build_command(spec, item, output)
    (output / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    started = time.time()
    write_json(
        output / "status.json",
        {
            "status": "running",
            "name": name,
            "gpu": item["gpu"],
            "seed": item["seed"],
            "started_at": started,
        },
    )
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command, cwd=ROOT, env=env, text=True, stdout=stdout, stderr=stderr
        )
    result: dict[str, Any] = {
        "status": "done" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "name": name,
        "gpu": item["gpu"],
        "seed": item["seed"],
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
        {"status": "running", "phase": "v361_longdoc", "started_at": time.time()},
    )
    queues: dict[int, list[dict[str, Any]]] = {}
    for item in matrix:
        queues.setdefault(int(item["gpu"]), []).append(item)
    max_parallel = min(
        int(spec.get("execution", {}).get("max_parallel", 1)), len(queues)
    )
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
