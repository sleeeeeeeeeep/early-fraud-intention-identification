#!/usr/bin/env python3
"""Score and compare M0/M1 on V365 Dev without using any Test labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


EXPECTED_SOURCE_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_dev_final.jsonl"
)
THRESHOLD = 0.5
DIAGNOSTIC_MAX_CHUNKS = 12

# Fixed before looking at predictions.  Exact requested terms are retained as
# their own features; the three semantic groups are used only for stratification.
KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "exact_到期": ("到期",),
    "exact_扣费": ("扣费",),
    "exact_关闭": ("关闭",),
    "到期失效组": ("到期", "过期", "期满", "失效", "免费期结束", "试用期结束"),
    "扣费续费组": ("扣费", "扣款", "自动扣费", "自动续费", "续费", "划扣", "收费"),
    "关闭取消组": ("关闭", "取消", "注销", "退订", "解除", "终止"),
}
ALL_KEYWORDS = tuple(sorted({term for terms in KEYWORD_GROUPS.values() for term in terms}))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": (float(numerator / denominator) if denominator else None),
    }


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean(values: Sequence[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p for discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def validate_dev_source(path: Path) -> list[dict[str, Any]]:
    normalized = path.resolve().as_posix()
    if not normalized.endswith(EXPECTED_SOURCE_SUFFIX):
        raise SystemExit(f"refusing non-allowlisted source: {normalized}")
    if "test" in path.name.lower() or "restricted" in normalized.lower():
        raise SystemExit("refusing Test/restricted-label source")
    rows = read_jsonl(path)
    counts = Counter(str(row.get("decision")) for row in rows)
    expected = Counter(normal=80, early_fraud=15, explicit_fraud=65, ambiguous=40)
    if len(rows) != 200 or counts != expected:
        raise ValueError(f"unexpected Dev composition: n={len(rows)}, counts={dict(counts)}")
    if len({str(row["call_id"]) for row in rows}) != 200:
        raise ValueError("call_id is not unique")
    for row in rows:
        decision = str(row["decision"])
        expected_mask = 0 if decision == "ambiguous" else 1
        expected_target = -1 if decision == "ambiguous" else int(decision != "normal")
        if int(row.get("binary_loss_mask", -1)) != expected_mask:
            raise ValueError(f"{row['call_id']}: binary_loss_mask inconsistent with decision")
        if int(row.get("binary_target", -99)) != expected_target:
            raise ValueError(f"{row['call_id']}: binary_target inconsistent with decision")
        if int(row["n_full_turns"]) != len(row["turns"]):
            raise ValueError(f"{row['call_id']}: n_full_turns mismatch")
        n_turns = int(row["n_full_turns"])
        t_e = row.get("tE_turn")
        t_a = row.get("tA_turn")
        if decision in {"early_fraud", "explicit_fraud"}:
            if t_e is None or not 1 <= int(t_e) <= n_turns:
                raise ValueError(f"{row['call_id']}: fraud has invalid tE")
        elif t_e is not None:
            raise ValueError(f"{row['call_id']}: non-fraud has non-null tE")
        if decision == "explicit_fraud":
            if t_a is None or not 1 <= int(t_a) <= n_turns:
                raise ValueError(f"{row['call_id']}: explicit fraud has invalid tA")
        elif t_a is not None:
            raise ValueError(f"{row['call_id']}: non-explicit row has non-null tA")
    return rows


def canonical_text(turns: Iterable[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{int(turn['turn_id']):03d} "
        f"{str(turn['speaker']).strip()}：{str(turn['text']).strip()}"
        for turn in turns
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_timing(path: Path, expected_checkpoint: Path, expected_input: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    checkpoint = payload.get("checkpoint") or {}
    config = payload.get("config") or {}
    if not checkpoint or not metrics or not config:
        raise ValueError(f"missing inference diagnostics in {path}")
    missing = checkpoint.get("missing", [])
    unexpected = checkpoint.get("unexpected", [])
    if missing or unexpected:
        raise ValueError(f"non-strict checkpoint load in {path}: {checkpoint}")
    if Path(str(checkpoint.get("checkpoint", ""))).resolve() != expected_checkpoint.resolve():
        raise ValueError(f"checkpoint identity mismatch in {path}")
    if Path(str(config.get("input_jsonl", ""))).resolve() != expected_input.resolve():
        raise ValueError(f"input identity mismatch in {path}")
    expected_config = {
        "route": "B",
        "input_mode": "head_context_enhanced_chunks",
        "chunk_length": 320,
        "overlap": 64,
        "head_context_length": 96,
        "encoder_micro_batch_size": 12,
        "evidence_components": "attention,top1,top2,last,mean",
        "batch_size": 2,
    }
    mismatched = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatched:
        raise ValueError(f"inference config mismatch in {path}: {mismatched}")
    wall_time = metrics.get("wall_time_seconds")
    if wall_time is None or float(wall_time) <= 0:
        raise ValueError(f"missing/invalid wall time in {path}")
    if abs(float(metrics.get("coverage_min", -1.0)) - 1.0) > 1e-9:
        raise ValueError(f"incomplete raw coverage in {path}")
    return {
        "wall_time_seconds": float(wall_time),
        "latency_ms_mean_raw": metrics.get("latency_ms_mean"),
        "latency_ms_p50_raw": metrics.get("latency_ms_p50"),
        "latency_ms_p95_raw": metrics.get("latency_ms_p95"),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
        "coverage_min_raw": metrics.get("coverage_min"),
        "checkpoint_load_report": checkpoint,
        "note": "Only timing/coverage/load diagnostics are used; built-in labels are ignored.",
    }


def prediction_probability(row: Mapping[str, Any]) -> tuple[float, float]:
    values = row.get("p_binary")
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"missing two-class p_binary for {row.get('sample_id')}")
    p0, p1 = float(values[0]), float(values[1])
    if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0):
        raise ValueError(f"invalid probability for {row.get('sample_id')}: {values}")
    if abs(p0 + p1 - 1.0) > 1e-4:
        raise ValueError(f"probabilities do not sum to one: {values}")
    return p0, p1


def build_full_records(
    dev_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], model_id: str
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in raw_rows}
    expected_ids = {str(row["sample_id"]) for row in dev_rows}
    if len(by_id) != len(raw_rows) or set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))[:5]
        extra = sorted(set(by_id) - expected_ids)[:5]
        raise ValueError(f"{model_id} full ids mismatch; missing={missing}, extra={extra}")
    output: list[dict[str, Any]] = []
    for source in dev_rows:
        raw = by_id[str(source["sample_id"])]
        p_normal, p_fraud = prediction_probability(raw)
        decision = str(source["decision"])
        y_true = None if decision == "ambiguous" else int(decision != "normal")
        output.append(
            {
                "schema_version": "v365.dev.full_prediction.v1",
                "model_id": model_id,
                "sample_id": str(source["sample_id"]),
                "call_id": str(source["call_id"]),
                "decision": decision,
                "scenario_tag": source.get("scenario_tag"),
                "binary_loss_mask": int(source.get("binary_loss_mask", 0)),
                "y_true": y_true,
                "p_normal": p_normal,
                "p_fraud": p_fraud,
                "threshold": THRESHOLD,
                "prediction": int(p_fraud >= THRESHOLD),
                "n_full_turns": int(source["n_full_turns"]),
                "tE_turn": source.get("tE_turn"),
                "tA_turn": source.get("tA_turn"),
                "raw_wordpiece_length": int(raw.get("raw_wordpiece_length", 0)),
                "n_chunks": int(raw.get("num_chunks", 0)),
                "diagnostic_max_chunks": DIAGNOSTIC_MAX_CHUNKS,
                "max_chunks_exceeded": bool(raw.get("max_chunks_exceeded", False)),
                "was_chunk_capped": False,
                "full_text_coverage": float(raw.get("full_text_coverage", raw.get("coverage_ratio", -1.0))),
                "top_evidence_chunk": raw.get("top_evidence_chunk"),
                "top_evidence_chunks": raw.get("top_evidence_chunks", []),
            }
        )
        record = output[-1]
        if abs(float(record["full_text_coverage"]) - 1.0) > 1e-9:
            raise ValueError(f"{model_id}/{source['call_id']}: incomplete full-text coverage")
        if int(record["n_chunks"]) < 1:
            raise ValueError(f"{model_id}/{source['call_id']}: invalid chunk count")
        if bool(record["max_chunks_exceeded"]) != (int(record["n_chunks"]) > DIAGNOSTIC_MAX_CHUNKS):
            raise ValueError(f"{model_id}/{source['call_id']}: max_chunks_exceeded inconsistency")
    return output


def binary_metrics(records: list[dict[str, Any]], timing: dict[str, Any]) -> dict[str, Any]:
    valid = [
        row
        for row in records
        if row["y_true"] is not None and int(row["binary_loss_mask"]) > 0
    ]
    if len(valid) != 160:
        raise ValueError(f"expected 160 effective rows, got {len(valid)}")
    y = np.asarray([int(row["y_true"]) for row in valid], dtype=int)
    p = np.asarray([float(row["p_fraud"]) for row in valid], dtype=float)
    pred = (p >= THRESHOLD).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    accuracy = (tp + tn) / len(valid)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and precision + recall else None

    def hit_block(decision: str) -> dict[str, Any]:
        subset = [row for row in records if row["decision"] == decision]
        hits = sum(int(row["prediction"] == 1) for row in subset)
        block = rate(hits, len(subset))
        block["wilson_95"] = wilson_interval(hits, len(subset))
        return block

    fraud = [row for row in records if row["decision"] in {"early_fraud", "explicit_fraud"}]
    normal = [row for row in records if row["decision"] == "normal"]
    ambiguous = [row for row in records if row["decision"] == "ambiguous"]
    amb_probs = [float(row["p_fraud"]) for row in ambiguous]
    wall = float(timing["wall_time_seconds"])
    scenario_metrics: dict[str, Any] = {}
    for scenario in sorted({str(row.get("scenario_tag")) for row in records}):
        scenario_rows = [row for row in records if str(row.get("scenario_tag")) == scenario]
        scenario_fraud = [row for row in scenario_rows if row["decision"] in {"early_fraud", "explicit_fraud"}]
        scenario_early = [row for row in scenario_rows if row["decision"] == "early_fraud"]
        scenario_normal = [row for row in scenario_rows if row["decision"] == "normal"]
        scenario_metrics[scenario] = {
            "fraud_recall": rate(sum(int(row["prediction"] == 1) for row in scenario_fraud), len(scenario_fraud)),
            "early_fraud_recall": rate(sum(int(row["prediction"] == 1) for row in scenario_early), len(scenario_early)),
            "normal_fpr": rate(sum(int(row["prediction"] == 1) for row in scenario_normal), len(scenario_normal)),
        }
    fraud_scenario_rates = [block["fraud_recall"]["rate"] for block in scenario_metrics.values() if block["fraud_recall"]["denominator"]]
    normal_scenario_rates = [block["normal_fpr"]["rate"] for block in scenario_metrics.values() if block["normal_fpr"]["denominator"]]
    return {
        "schema_version": "v365.dev.binary_metrics.v1",
        "threshold": THRESHOLD,
        "probability_field": "p_binary[1]",
        "effective_samples": len(valid),
        "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
        "excluded_ambiguous": len(ambiguous),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": (float(precision) if precision is not None else None),
        "recall": float(recall),
        "f1": (float(f1) if f1 is not None else None),
        "normal_fpr": float(fp / (fp + tn)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_ap": float(average_precision_score(y, p)),
        "subgroups": {
            "early_fraud": hit_block("early_fraud"),
            "explicit_fraud": hit_block("explicit_fraud"),
            "all_fraud": {
                **rate(sum(int(row["prediction"] == 1) for row in fraud), len(fraud)),
                "wilson_95": wilson_interval(sum(int(row["prediction"] == 1) for row in fraud), len(fraud)),
            },
            "normal_false_positive": {
                **rate(sum(int(row["prediction"] == 1) for row in normal), len(normal)),
                "wilson_95": wilson_interval(sum(int(row["prediction"] == 1) for row in normal), len(normal)),
            },
        },
        "ambiguous": {
            "samples": len(ambiguous),
            "mean_p_fraud": mean(amb_probs),
            "median_p_fraud": median(amb_probs),
            "p90_p_fraud_linear": quantile(amb_probs, 0.90),
            "count_ge_0_5": sum(value >= THRESHOLD for value in amb_probs),
        },
        "scenario_metrics": scenario_metrics,
        "scenario_macro": {
            "fraud_recall": mean([float(value) for value in fraud_scenario_rates if value is not None]),
            "normal_fpr": mean([float(value) for value in normal_scenario_rates if value is not None]),
            "fraud_scenarios": len(fraud_scenario_rates),
            "normal_scenarios": len(normal_scenario_rates),
        },
        "diagnostics": {
            "coverage_min": min(float(row["full_text_coverage"]) for row in records),
            "max_chunks_observed": max(int(row["n_chunks"]) for row in records),
            "calls_exceeding_diagnostic_12_chunks": sum(bool(row["max_chunks_exceeded"]) for row in records),
            "calls_capped": 0,
            "max_chunks_semantics": "diagnostic only; all chunks retained for full coverage",
        },
        "timing": {
            **timing,
            "evaluated_calls": len(records),
            "seconds_per_call": wall / len(records),
            "milliseconds_per_call": wall * 1000 / len(records),
        },
    }


def build_prefix_records(
    dev_rows: list[dict[str, Any]],
    input_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    model_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_call = {str(row["call_id"]): row for row in dev_rows}
    inputs = {str(row["sample_id"]): row for row in input_rows}
    raw = {str(row["sample_id"]): row for row in raw_rows}
    expected_prefixes = sum(int(row["n_full_turns"]) for row in dev_rows)
    if len(inputs) != expected_prefixes or len(raw) != expected_prefixes or set(inputs) != set(raw):
        raise ValueError(
            f"{model_id} prefix mismatch: inputs={len(inputs)}, predictions={len(raw)}, expected={expected_prefixes}"
        )
    prefix_records: list[dict[str, Any]] = []
    by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id, input_row in inputs.items():
        call_id = str(input_row["call_id"])
        if call_id not in source_by_call:
            raise ValueError(f"unknown prefix call_id: {call_id}")
        source = source_by_call[call_id]
        pred_row = raw[sample_id]
        p_normal, p_fraud = prediction_probability(pred_row)
        turn = int(input_row["prefix_end_turn"])
        if not 1 <= turn <= int(source["n_full_turns"]):
            raise ValueError(f"{sample_id}: prefix turn out of range")
        expected_text = canonical_text(source["turns"][:turn])
        actual_text = str(input_row.get("model_input_text") or "")
        if actual_text != expected_text:
            raise ValueError(f"{sample_id}: prefix contains missing/extra/future text")
        expected_hash = sha256_text(expected_text)
        if str(input_row.get("input_text_sha256") or "") != expected_hash:
            raise ValueError(f"{sample_id}: prefix text hash mismatch")
        t_e = source.get("tE_turn")
        t_a = source.get("tA_turn")
        relation_e = "na" if t_e is None else ("pre" if turn < int(t_e) else "at" if turn == int(t_e) else "post")
        relation_a = "na" if t_a is None else ("pre" if turn < int(t_a) else "at" if turn == int(t_a) else "post")
        record = {
            "schema_version": "v365.dev.prefix_prediction.v1",
            "model_id": model_id,
            "sample_id": sample_id,
            "call_id": call_id,
            "decision": str(source["decision"]),
            "scenario_tag": source.get("scenario_tag"),
            "prefix_turn": turn,
            "n_full_turns": int(source["n_full_turns"]),
            "tE_turn": t_e,
            "tA_turn": t_a,
            "prefix_text_sha256": expected_hash,
            "p_normal": p_normal,
            "p_fraud": p_fraud,
            "raw_hit_ge_0_5": bool(p_fraud >= THRESHOLD),
            "alert_k1": False,
            "alert_k2": False,
            "relation_to_tE": relation_e,
            "relation_to_tA": relation_a,
            "is_future_free": bool(input_row.get("is_future_free", False)),
            "raw_wordpiece_length": int(pred_row.get("raw_wordpiece_length", 0)),
            "n_chunks": int(pred_row.get("num_chunks", 0)),
            "max_chunks_exceeded": bool(pred_row.get("max_chunks_exceeded", False)),
            "full_text_coverage": float(pred_row.get("full_text_coverage", -1.0)),
        }
        if abs(record["full_text_coverage"] - 1.0) > 1e-9:
            raise ValueError(f"{model_id}/{sample_id}: incomplete prefix coverage")
        if record["n_chunks"] < 1:
            raise ValueError(f"{model_id}/{sample_id}: invalid prefix chunk count")
        if record["max_chunks_exceeded"] != (record["n_chunks"] > DIAGNOSTIC_MAX_CHUNKS):
            raise ValueError(f"{model_id}/{sample_id}: max_chunks_exceeded inconsistency")
        by_call[call_id].append(record)
        prefix_records.append(record)

    call_summaries: list[dict[str, Any]] = []
    for source in dev_rows:
        call_id = str(source["call_id"])
        sequence = sorted(by_call[call_id], key=lambda row: int(row["prefix_turn"]))
        expected_turns = list(range(1, int(source["n_full_turns"]) + 1))
        actual_turns = [int(row["prefix_turn"]) for row in sequence]
        if actual_turns != expected_turns:
            raise ValueError(f"{call_id}: incomplete prefix sequence")
        if any(not row["is_future_free"] for row in sequence):
            raise ValueError(f"{call_id}: prefix not marked future-free")
        hits = [bool(row["raw_hit_ge_0_5"]) for row in sequence]
        alerts_by_k = {
            1: hits,
            2: [False] + [hits[index - 1] and hits[index] for index in range(1, len(hits))],
        }
        policies: dict[str, Any] = {}
        t_e_raw = source.get("tE_turn")
        t_a_raw = source.get("tA_turn")
        t_e = int(t_e_raw) if t_e_raw is not None else None
        t_a = int(t_a_raw) if t_a_raw is not None else None
        for k, alerts in alerts_by_k.items():
            for index, value in enumerate(alerts):
                sequence[index][f"alert_k{k}"] = bool(value)
            alarm_turns = [index + 1 for index, value in enumerate(alerts) if value]
            first_any = min(alarm_turns) if alarm_turns else None
            first_valid = (
                min((turn for turn in alarm_turns if t_e is not None and turn >= t_e), default=None)
                if t_e is not None
                else None
            )
            premature = bool(t_e is not None and any(turn < t_e for turn in alarm_turns))
            effective = first_valid is not None
            policies[f"k{k}"] = {
                "alarm_turns": alarm_turns,
                "first_any_turn": first_any,
                "first_any_offset_from_tE": (first_any - t_e if first_any is not None and t_e is not None else None),
                "premature": premature,
                "effective": effective,
                "clean_effective": bool(effective and not premature),
                "first_valid_turn": first_valid,
                "valid_delay": (first_valid - t_e if first_valid is not None and t_e is not None else None),
                "before_tA": bool(first_valid is not None and t_a is not None and first_valid < t_a),
                "advance_before_tA": (t_a - first_valid if first_valid is not None and t_a is not None and first_valid < t_a else None),
                "tA_minus_first_valid": (t_a - first_valid if first_valid is not None and t_a is not None else None),
                "normal_ever_false_alarm": bool(str(source["decision"]) == "normal" and alarm_turns),
            }
        probs = [float(row["p_fraud"]) for row in sequence]
        call_summaries.append(
            {
                "schema_version": "v365.dev.temporal_call.v1",
                "model_id": model_id,
                "call_id": call_id,
                "decision": str(source["decision"]),
                "scenario_tag": source.get("scenario_tag"),
                "n_full_turns": int(source["n_full_turns"]),
                "tE_turn": t_e,
                "tA_turn": t_a,
                "trajectory": {
                    "mean_p_fraud": mean(probs),
                    "max_p_fraud": max(probs),
                    "final_p_fraud": probs[-1],
                    "first_raw_hit_turn": next((i + 1 for i, hit in enumerate(hits) if hit), None),
                },
                "policies": policies,
            }
        )
    prefix_records.sort(key=lambda row: (str(row["call_id"]), int(row["prefix_turn"])))
    return prefix_records, call_summaries


def temporal_metrics(
    call_summaries: list[dict[str, Any]], timing: dict[str, Any]
) -> dict[str, Any]:
    by_decision = defaultdict(list)
    for row in call_summaries:
        by_decision[str(row["decision"])].append(row)
    subsets = {
        "early_fraud": by_decision["early_fraud"],
        "explicit_fraud": by_decision["explicit_fraud"],
        "all_fraud": by_decision["early_fraud"] + by_decision["explicit_fraud"],
    }
    policies: dict[str, Any] = {}
    for k in (1, 2):
        k_key = f"k{k}"
        subset_metrics: dict[str, Any] = {}
        for name, rows in subsets.items():
            eligible = [row for row in rows if row["tE_turn"] is not None]
            effective = [row for row in eligible if row["policies"][k_key]["effective"]]
            premature = [row for row in eligible if row["policies"][k_key]["premature"]]
            clean = [row for row in eligible if row["policies"][k_key]["clean_effective"]]
            opportunity = [
                row
                for row in eligible
                if int(row["tE_turn"]) > (1 if k == 1 else 2)
            ]
            premature_opportunity = [row for row in opportunity if row["policies"][k_key]["premature"]]
            delays = [int(row["policies"][k_key]["valid_delay"]) for row in effective]
            subset_metrics[name] = {
                "tE_after_effective_recall": rate(len(effective), len(eligible)),
                "tE_before_premature_alarm_rate": rate(len(premature), len(eligible)),
                "clean_effective_rate": rate(len(clean), len(eligible)),
                "premature_rate_among_calls_with_opportunity": rate(len(premature_opportunity), len(opportunity)),
                "delay": {
                    "detected_n": len(effective),
                    "missed_n": len(eligible) - len(effective),
                    "median_turns": median(delays),
                    "p25_turns": quantile(delays, 0.25),
                    "p75_turns": quantile(delays, 0.75),
                },
            }
        normal = by_decision["normal"]
        normal_fp = [row for row in normal if row["policies"][k_key]["normal_ever_false_alarm"]]
        explicit_ta = [
            row
            for row in by_decision["explicit_fraud"]
            if row["tA_turn"] is not None
            and 1 <= int(row["tA_turn"]) <= int(row["n_full_turns"])
        ]
        before_ta = [row for row in explicit_ta if row["policies"][k_key]["before_tA"]]
        t_a_opportunity = [
            row for row in explicit_ta
            if row["tE_turn"] is not None and int(row["tA_turn"]) > int(row["tE_turn"])
        ]
        before_ta_opportunity = [row for row in t_a_opportunity if row["policies"][k_key]["before_tA"]]
        advances = [int(row["policies"][k_key]["advance_before_tA"]) for row in before_ta]
        signed_ta_offsets = [
            int(row["policies"][k_key]["tA_minus_first_valid"])
            for row in explicit_ta
            if row["policies"][k_key]["tA_minus_first_valid"] is not None
        ]
        at_ta = [
            row for row in explicit_ta
            if row["policies"][k_key]["first_valid_turn"] == row["tA_turn"]
        ]
        after_ta = [
            row for row in explicit_ta
            if row["policies"][k_key]["first_valid_turn"] is not None
            and int(row["policies"][k_key]["first_valid_turn"]) > int(row["tA_turn"])
        ]
        ambiguous = by_decision["ambiguous"]
        amb_ever = [row for row in ambiguous if row["policies"][k_key]["alarm_turns"]]
        policies[k_key] = {
            "consecutive_hits_required": k,
            "alarm_time_semantics": "end turn of the consecutive-hit window",
            "subsets": subset_metrics,
            "normal_ever_false_alarm": rate(len(normal_fp), len(normal)),
            "explicit_before_tA": {
                **rate(len(before_ta), len(explicit_ta)),
                "strictly_before_tA": True,
                "among_calls_with_nonempty_tE_to_tA_window": rate(len(before_ta_opportunity), len(t_a_opportunity)),
                "no_strict_pre_tA_window_n": len(explicit_ta) - len(t_a_opportunity),
                "median_advance_turns": median(advances),
                "p25_advance_turns": quantile(advances, 0.25),
                "p75_advance_turns": quantile(advances, 0.75),
                "at_tA_n": len(at_ta),
                "after_tA_n": len(after_ta),
                "no_effective_detection_n": sum(
                    row["policies"][k_key]["first_valid_turn"] is None
                    for row in explicit_ta
                ),
                "signed_tA_minus_first_valid_detected_n": len(signed_ta_offsets),
                "signed_tA_minus_first_valid_median": median(signed_ta_offsets),
                "invalid_or_missing_tA": len(by_decision["explicit_fraud"]) - len(explicit_ta),
            },
            "ambiguous_trajectory_only": {
                "calls": len(ambiguous),
                "ever_alert_calls": len(amb_ever),
                "ever_alert": rate(len(amb_ever), len(ambiguous)),
                "mean_of_call_max_p_fraud": mean([float(row["trajectory"]["max_p_fraud"]) for row in ambiguous]),
                "median_of_call_max_p_fraud": median([float(row["trajectory"]["max_p_fraud"]) for row in ambiguous]),
            },
        }
    wall = float(timing["wall_time_seconds"])
    prefix_count = sum(int(row["n_full_turns"]) for row in call_summaries)
    return {
        "schema_version": "v365.dev.temporal_metrics.v1",
        "threshold": THRESHOLD,
        "prefixes": prefix_count,
        "calls": len(call_summaries),
        "policies": policies,
        "timing": {
            **timing,
            "evaluated_prefixes": prefix_count,
            "seconds_per_prefix": wall / prefix_count,
            "milliseconds_per_prefix": wall * 1000 / prefix_count,
        },
        "boundary_note": (
            "turn_stage_labels are not used as independent labels; tE/tA only locate "
            "evaluation boundaries from final adjudication."
        ),
    }


def source_evidence(source: Mapping[str, Any], temporal: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for field, kind in (
        ("core_evidence", "adjudicated_core"),
        ("supporting_evidence", "adjudicated_supporting"),
        ("operation_events", "adjudicated_operation"),
    ):
        items = source.get(field) or []
        if isinstance(items, list):
            for item in items[:3]:
                if isinstance(item, Mapping):
                    turn_value = item.get("turn_id")
                    turn_number = int(turn_value) if turn_value is not None else None
                    t_e = source.get("tE_turn")
                    t_a = source.get("tA_turn")
                    evidence.append(
                        {
                            "turn": turn_number,
                            "quote": item.get("quote"),
                            "kind": kind,
                            "subtype": item.get("evidence_type") or item.get("operation_type"),
                            "available_by_tE": bool(turn_number is not None and t_e is not None and turn_number <= int(t_e)),
                            "relation_to_tE": (
                                None if turn_number is None or t_e is None
                                else "pre" if turn_number < int(t_e)
                                else "at" if turn_number == int(t_e)
                                else "post"
                            ),
                            "relation_to_tA": (
                                None if turn_number is None or t_a is None
                                else "pre" if turn_number < int(t_a)
                                else "at" if turn_number == int(t_a)
                                else "post"
                            ),
                        }
                    )
    turns = {int(turn["turn_id"]): str(turn["text"]) for turn in source["turns"]}
    for k in (1, 2):
        first = temporal["policies"][f"k{k}"]["first_any_turn"]
        if first is not None:
            evidence.append(
                {
                    "turn": first,
                    "quote": turns.get(int(first)),
                    "kind": f"model_first_alarm_k{k}",
                    "subtype": None,
                }
            )
    seen: set[tuple[Any, Any, Any]] = set()
    unique: list[dict[str, Any]] = []
    for item in evidence:
        key = (item.get("turn"), item.get("quote"), item.get("kind"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:8]


def build_errors(
    dev_rows: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
    temporal_calls: list[dict[str, Any]],
    model_id: str,
) -> list[dict[str, Any]]:
    source = {str(row["call_id"]): row for row in dev_rows}
    full = {str(row["call_id"]): row for row in full_records}
    temporal = {str(row["call_id"]): row for row in temporal_calls}
    output: list[dict[str, Any]] = []
    for call_id in sorted(source):
        s = source[call_id]
        f = full[call_id]
        t = temporal[call_id]
        decision = str(s["decision"])
        errors: list[str] = []
        if decision == "early_fraud" and int(f["prediction"]) == 0:
            errors.append("fulltext_fn_early")
        elif decision == "explicit_fraud" and int(f["prediction"]) == 0:
            errors.append("fulltext_fn_explicit")
        elif decision == "normal" and int(f["prediction"]) == 1:
            errors.append("fulltext_fp_normal")
        if decision in {"early_fraud", "explicit_fraud"}:
            for k in (1, 2):
                policy = t["policies"][f"k{k}"]
                if policy["premature"]:
                    errors.append(f"premature_k{k}")
                if not policy["effective"]:
                    errors.append(f"miss_after_tE_k{k}")
                if decision == "explicit_fraud" and not policy["before_tA"]:
                    errors.append(f"not_before_tA_k{k}")
        elif decision == "normal":
            for k in (1, 2):
                if t["policies"][f"k{k}"]["normal_ever_false_alarm"]:
                    errors.append(f"normal_ever_fp_k{k}")
        if not errors:
            continue
        output.append(
            {
                "schema_version": "v365.dev.error.v1",
                "model_id": model_id,
                "call_id": call_id,
                "decision": decision,
                "scenario_tag": s.get("scenario_tag"),
                "n_full_turns": int(s["n_full_turns"]),
                "tE_turn": s.get("tE_turn"),
                "tA_turn": s.get("tA_turn"),
                "p_fraud": float(f["p_fraud"]),
                "prediction": int(f["prediction"]),
                "error_types": errors,
                "first_any_turn_k1": t["policies"]["k1"]["first_any_turn"],
                "first_valid_turn_k1": t["policies"]["k1"]["first_valid_turn"],
                "valid_delay_k1": t["policies"]["k1"]["valid_delay"],
                "premature_k1": t["policies"]["k1"]["premature"],
                "first_any_turn_k2": t["policies"]["k2"]["first_any_turn"],
                "first_valid_turn_k2": t["policies"]["k2"]["first_valid_turn"],
                "valid_delay_k2": t["policies"]["k2"]["valid_delay"],
                "premature_k2": t["policies"]["k2"]["premature"],
                "related_evidence": source_evidence(s, t),
            }
        )
    return output


def endpoint_consistency(
    full_records: list[dict[str, Any]], temporal_calls: list[dict[str, Any]], model_id: str
) -> dict[str, Any]:
    full = {str(row["call_id"]): row for row in full_records}
    temporal = {str(row["call_id"]): row for row in temporal_calls}
    if set(full) != set(temporal):
        raise ValueError(f"{model_id}: full/prefix endpoint call sets differ")
    deltas = {
        call_id: abs(float(full[call_id]["p_fraud"]) - float(temporal[call_id]["trajectory"]["final_p_fraud"]))
        for call_id in full
    }
    flips = [
        call_id
        for call_id in full
        if (float(full[call_id]["p_fraud"]) >= THRESHOLD)
        != (float(temporal[call_id]["trajectory"]["final_p_fraud"]) >= THRESHOLD)
    ]
    max_delta = max(deltas.values(), default=0.0)
    if flips:
        raise ValueError(f"{model_id}: full/final-prefix threshold flips: {flips}")
    if max_delta > 1e-4:
        worst = max(deltas, key=deltas.get)
        raise ValueError(f"{model_id}: full/final-prefix delta {max_delta} at {worst} exceeds 1e-4")
    return {
        "calls": len(full),
        "max_abs_p_fraud_delta": max_delta,
        "mean_abs_p_fraud_delta": mean(list(deltas.values())),
        "threshold_flip_count": len(flips),
        "tolerance": 1e-4,
        "passed": True,
    }


def compare_full(
    dev_rows: list[dict[str, Any]],
    m0: list[dict[str, Any]],
    m1: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = {str(row["call_id"]): row for row in dev_rows}
    a = {str(row["call_id"]): row for row in m0}
    b = {str(row["call_id"]): row for row in m1}
    output: list[dict[str, Any]] = []
    for call_id in sorted(source):
        s, r0, r1 = source[call_id], a[call_id], b[call_id]
        decision = str(s["decision"])
        p0, p1 = int(r0["prediction"]), int(r1["prediction"])
        transition = "unchanged"
        if decision in {"early_fraud", "explicit_fraud"}:
            if (p0, p1) == (0, 1):
                transition = "m1_new_hit"
            elif (p0, p1) == (1, 0):
                transition = "m1_lost_hit"
        elif decision == "normal":
            if (p0, p1) == (0, 1):
                transition = "m1_new_false_positive"
            elif (p0, p1) == (1, 0):
                transition = "m1_corrected_false_positive"
        else:
            transition = "ambiguous_score_only"
        output.append(
            {
                "call_id": call_id,
                "decision": decision,
                "scenario_tag": s.get("scenario_tag"),
                "n_full_turns": int(s["n_full_turns"]),
                "tE_turn": s.get("tE_turn"),
                "tA_turn": s.get("tA_turn"),
                "m0_p_fraud": float(r0["p_fraud"]),
                "m1_p_fraud": float(r1["p_fraud"]),
                "delta_p_m1_minus_m0": float(r1["p_fraud"] - r0["p_fraud"]),
                "m0_prediction": p0,
                "m1_prediction": p1,
                "transition": transition,
                "core_evidence": s.get("core_evidence", []),
            }
        )
    return output


def compare_temporal(
    dev_rows: list[dict[str, Any]],
    m0_calls: list[dict[str, Any]],
    m1_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = {str(row["call_id"]): row for row in dev_rows}
    m0 = {str(row["call_id"]): row for row in m0_calls}
    m1 = {str(row["call_id"]): row for row in m1_calls}
    if set(source) != set(m0) or set(source) != set(m1):
        raise ValueError("temporal model call sets do not match Dev")
    output: list[dict[str, Any]] = []
    for call_id in sorted(source):
        s = source[call_id]
        decision = str(s["decision"])
        policies: dict[str, Any] = {}
        for k in (1, 2):
            p0 = m0[call_id]["policies"][f"k{k}"]
            p1 = m1[call_id]["policies"][f"k{k}"]
            changes: list[str] = []
            if decision in {"early_fraud", "explicit_fraud"}:
                if not p0["effective"] and p1["effective"]:
                    changes.append("m1_new_effective_detection")
                elif p0["effective"] and not p1["effective"]:
                    changes.append("m1_lost_effective_detection")
                if not p0["premature"] and p1["premature"]:
                    changes.append("m1_new_premature_alarm")
                elif p0["premature"] and not p1["premature"]:
                    changes.append("m1_corrected_premature_alarm")
                if decision == "explicit_fraud":
                    if not p0["before_tA"] and p1["before_tA"]:
                        changes.append("m1_new_before_tA_detection")
                    elif p0["before_tA"] and not p1["before_tA"]:
                        changes.append("m1_lost_before_tA_detection")
            elif decision == "normal":
                if not p0["normal_ever_false_alarm"] and p1["normal_ever_false_alarm"]:
                    changes.append("m1_new_normal_ever_false_alarm")
                elif p0["normal_ever_false_alarm"] and not p1["normal_ever_false_alarm"]:
                    changes.append("m1_corrected_normal_ever_false_alarm")
            policies[f"k{k}"] = {
                "changes": changes,
                "m0": {
                    "first_any_turn": p0["first_any_turn"],
                    "first_valid_turn": p0["first_valid_turn"],
                    "effective": p0["effective"],
                    "premature": p0["premature"],
                    "normal_ever_false_alarm": p0["normal_ever_false_alarm"],
                    "before_tA": p0["before_tA"],
                },
                "m1": {
                    "first_any_turn": p1["first_any_turn"],
                    "first_valid_turn": p1["first_valid_turn"],
                    "effective": p1["effective"],
                    "premature": p1["premature"],
                    "normal_ever_false_alarm": p1["normal_ever_false_alarm"],
                    "before_tA": p1["before_tA"],
                },
            }
        output.append(
            {
                "schema_version": "v365.dev.temporal_comparison.v1",
                "call_id": call_id,
                "decision": decision,
                "scenario_tag": s.get("scenario_tag"),
                "n_full_turns": int(s["n_full_turns"]),
                "tE_turn": s.get("tE_turn"),
                "tA_turn": s.get("tA_turn"),
                "policies": policies,
            }
        )
    return output


def t_e_position(row: Mapping[str, Any]) -> str:
    t_e = row.get("tE_turn")
    if t_e is None:
        return "missing"
    ratio = int(t_e) / int(row["n_full_turns"])
    if ratio <= 1 / 3:
        return "early_le_1_3"
    if ratio <= 2 / 3:
        return "middle_1_3_to_2_3"
    return "late_gt_2_3"


def miss_analysis(
    dev_rows: list[dict[str, Any]],
    full_records: list[dict[str, Any]],
    temporal_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    source = {str(row["call_id"]): row for row in dev_rows}
    temporal = {str(row["call_id"]): row for row in temporal_calls}
    fraud = [row for row in full_records if row["decision"] in {"early_fraud", "explicit_fraud"}]
    q75 = quantile([float(row["raw_wordpiece_length"]) for row in fraud], 0.75)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "by_decision": dict(Counter(str(row["decision"]) for row in rows)),
            "by_scenario": dict(Counter(str(row.get("scenario_tag")) for row in rows)),
            "by_tE_relative_position": dict(Counter(t_e_position(source[str(row["call_id"])]) for row in rows)),
            "long_at_or_above_fraud_q75": sum(float(row["raw_wordpiece_length"]) >= float(q75) for row in rows) if q75 is not None else 0,
            "exceeded_diagnostic_12_chunks": sum(bool(row["max_chunks_exceeded"]) for row in rows),
            "call_ids": [str(row["call_id"]) for row in rows],
        }

    full_fn = [row for row in fraud if int(row["prediction"]) == 0]
    output: dict[str, Any] = {
        "fraud_wordpiece_q75_threshold": q75,
        "fulltext_false_negatives": aggregate(full_fn),
    }
    for k in (1, 2):
        missed_ids = {
            call_id
            for call_id, row in temporal.items()
            if row["decision"] in {"early_fraud", "explicit_fraud"}
            and not row["policies"][f"k{k}"]["effective"]
        }
        output[f"temporal_miss_k{k}"] = aggregate([row for row in fraud if row["call_id"] in missed_ids])
    return output


def first_keyword_turn(source: Mapping[str, Any], terms: Sequence[str]) -> int | None:
    for turn in source["turns"]:
        text = str(turn["text"])
        if any(term in text for term in terms):
            return int(turn["turn_id"])
    return None


def keyword_analysis(
    dev_rows: list[dict[str, Any]],
    full_by_model: Mapping[str, list[dict[str, Any]]],
    temporal_by_model: Mapping[str, list[dict[str, Any]]],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    source = {str(row["call_id"]): row for row in dev_rows}
    result: dict[str, Any] = {
        "fixed_before_prediction_review": True,
        "groups": {key: list(value) for key, value in KEYWORD_GROUPS.items()},
        "all_terms": list(ALL_KEYWORDS),
        "interpretation_limit": "Observational stratification only; it does not prove causal reliance.",
        "models": {},
    }
    groups_with_any = {**KEYWORD_GROUPS, "任一固定词": ALL_KEYWORDS}
    for model_id, records in full_by_model.items():
        model_result: dict[str, Any] = {"fulltext_strata": {}, "alarm_relative_to_first_keyword": {}}
        for group_name, terms in groups_with_any.items():
            strata: dict[str, Any] = {}
            for population_name, decisions in {
                "fraud": {"early_fraud", "explicit_fraud"},
                "normal": {"normal"},
            }.items():
                population = [row for row in records if row["decision"] in decisions]
                present = [row for row in population if any(term in str(source[row["call_id"]]["model_input_text"]) for term in terms)]
                absent = [row for row in population if row not in present]

                def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
                    return {
                        "n": len(rows),
                        "mean_p_fraud": mean([float(row["p_fraud"]) for row in rows]),
                        "predicted_fraud": rate(sum(int(row["prediction"] == 1) for row in rows), len(rows)),
                    }

                strata[population_name] = {"keyword_present": block(present), "keyword_absent": block(absent)}
            model_result["fulltext_strata"][group_name] = strata

        temporal = {str(row["call_id"]): row for row in temporal_by_model[model_id]}
        for k in (1, 2):
            relation_counts: Counter[str] = Counter()
            fraud_rows = [row for row in dev_rows if row["decision"] in {"early_fraud", "explicit_fraud"}]
            for row in fraud_rows:
                first_kw = first_keyword_turn(row, ALL_KEYWORDS)
                first_alarm = temporal[str(row["call_id"])]["policies"][f"k{k}"]["first_any_turn"]
                if first_alarm is None:
                    relation = "no_alarm"
                elif first_kw is None:
                    relation = "alarm_without_fixed_keyword"
                elif first_alarm < first_kw:
                    relation = "alarm_before_keyword"
                elif first_alarm == first_kw:
                    relation = "alarm_at_keyword_turn"
                else:
                    relation = "alarm_after_keyword"
                relation_counts[relation] += 1
            model_result["alarm_relative_to_first_keyword"][f"k{k}"] = {
                "denominator_fraud_calls": len(fraud_rows),
                "counts": dict(relation_counts),
            }
        result["models"][model_id] = model_result

    result["m1_transitions_by_any_keyword"] = {}
    for transition_name in (
        "m1_new_hit",
        "m1_lost_hit",
        "m1_new_false_positive",
        "m1_corrected_false_positive",
    ):
        rows = [row for row in transitions if row["transition"] == transition_name]
        present = [
            row
            for row in rows
            if any(term in str(source[row["call_id"]]["model_input_text"]) for term in ALL_KEYWORDS)
        ]
        result["m1_transitions_by_any_keyword"][transition_name] = {
            "total": len(rows),
            "keyword_present": len(present),
            "keyword_absent": len(rows) - len(present),
            "call_ids": [str(row["call_id"]) for row in rows],
        }
    return result


def fnum(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def fpct(value: Any) -> str:
    return "NA" if value is None else f"{float(value) * 100:.2f}%"


def frac(block: Mapping[str, Any]) -> str:
    return f"{block['numerator']}/{block['denominator']} ({fpct(block['rate'])})"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    metrics: Mapping[str, Mapping[str, Any]],
    temporal: Mapping[str, Mapping[str, Any]],
    transitions: list[dict[str, Any]],
    temporal_transitions: list[dict[str, Any]],
    misses: Mapping[str, Mapping[str, Any]],
    keywords: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    m0, m1 = metrics["m0_v361"], metrics["m1_e1b"]
    early0 = m0["subgroups"]["early_fraud"]
    early1 = m1["subgroups"]["early_fraud"]
    early_delta = int(early1["numerator"]) - int(early0["numerator"])
    early_word = "提高" if early_delta > 0 else "下降" if early_delta < 0 else "未改变"
    fp0 = m0["subgroups"]["normal_false_positive"]
    fp1 = m1["subgroups"]["normal_false_positive"]
    fp_delta = int(fp1["numerator"]) - int(fp0["numerator"])
    fp_word = "升高" if fp_delta > 0 else "降低" if fp_delta < 0 else "未改变"

    early_pairs = [row for row in transitions if row["decision"] == "early_fraud"]
    b = sum(row["m0_prediction"] == 0 and row["m1_prediction"] == 1 for row in early_pairs)
    c = sum(row["m0_prediction"] == 1 and row["m1_prediction"] == 0 for row in early_pairs)
    p_value = exact_mcnemar_p(b, c)
    m0_k1_early = temporal["m0_v361"]["policies"]["k1"]["subsets"]["early_fraud"]
    m1_k1_early = temporal["m1_e1b"]["policies"]["k1"]["subsets"]["early_fraud"]
    m0_k2_early = temporal["m0_v361"]["policies"]["k2"]["subsets"]["early_fraud"]
    m1_k2_early = temporal["m1_e1b"]["policies"]["k2"]["subsets"]["early_fraud"]
    m0_k1_normal = temporal["m0_v361"]["policies"]["k1"]["normal_ever_false_alarm"]
    m1_k1_normal = temporal["m1_e1b"]["policies"]["k1"]["normal_ever_false_alarm"]
    m0_k2_normal = temporal["m0_v361"]["policies"]["k2"]["normal_ever_false_alarm"]
    m1_k2_normal = temporal["m1_e1b"]["policies"]["k2"]["normal_ever_false_alarm"]

    lines = [
        "# V365 Dev 现有模型零训练基线评测",
        "",
        "## 结论摘要",
        "",
        f"- 完整文本上，E1-B 对 early_fraud 的命中数{early_word}：M0 为 {frac(early0)}，M1 为 {frac(early1)}，净变化 {early_delta:+d}/15。",
        f"- 正常通话完整文本误报{fp_word}：M0 为 {frac(fp0)}，M1 为 {frac(fp1)}，净变化 {fp_delta:+d}/80。",
        f"- 15 条 early_fraud 的配对变化为 M0漏/M1中={b}、M0中/M1漏={c}；精确 McNemar p={p_value:.4f}。样本仅15条，不能只凭百分点宣称稳定显著改善。",
        f"- 时序上，early 的tE后有效检测：k1由{frac(m0_k1_early['tE_after_effective_recall'])}升至{frac(m1_k1_early['tE_after_effective_recall'])}，k2同样由{frac(m0_k2_early['tE_after_effective_recall'])}升至{frac(m1_k2_early['tE_after_effective_recall'])}；两种策略都只净增1条。",
        f"- early抢报没有改善：k1维持{frac(m1_k1_early['tE_before_premature_alarm_rate'])}，k2由{frac(m0_k2_early['tE_before_premature_alarm_rate'])}升至{frac(m1_k2_early['tE_before_premature_alarm_rate'])}。normal曾误报方面，k1由{frac(m0_k1_normal)}升至{frac(m1_k1_normal)}，k2维持{frac(m1_k2_normal)}。因此E1-B改善了召回，但没有解决抢报，连续2轮策略更适合作为当前报警基线。",
        "",
        "## 评测约束与数据",
        "",
        "- 只读取 V365 Dev 200 条：normal 80、early_fraud 15、explicit_fraud 65、ambiguous 40；主二分类有效样本160条。",
        "- ambiguous 不计入混淆矩阵、Accuracy、Precision、Recall、F1、ROC-AUC 或 PR-AUC，仅报告分数分布与轨迹。",
        "- 两模型均使用 Route B、head_context_enhanced_chunks、chunk=320、overlap=64、head=96、阈值0.5，只采用 p_binary[1]。",
        "- `max_chunks=12` 在现有实现中是诊断阈值，不是截断上限；所有块均保留，全文覆盖率必须为1。",
        "- 未训练、未反向传播、未写出新模型权重；推理输入是从 Dev 派生的标签盲文件。",
        "",
        "## 完整文本结果",
        "",
    ]
    rows = []
    for model_id, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B")):
        m = metrics[model_id]
        cm = m["confusion"]
        rows.append(
            [
                label,
                f"{cm['tp']}/{cm['fp']}/{cm['tn']}/{cm['fn']}",
                fpct(m["accuracy"]),
                fpct(m["balanced_accuracy"]),
                fpct(m["precision"]),
                fpct(m["recall"]),
                fpct(m["f1"]),
                fpct(m["normal_fpr"]),
                fnum(m["roc_auc"]),
                fnum(m["pr_auc_ap"]),
            ]
        )
    lines += [
        markdown_table(
            ["模型", "TP/FP/TN/FN", "Accuracy", "Bal.Acc", "Precision", "Recall", "F1", "Normal FPR", "ROC-AUC", "PR-AUC(AP)"],
            rows,
        ),
        "",
    ]
    subgroup_rows = []
    for model_id, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B")):
        sub = metrics[model_id]["subgroups"]
        subgroup_rows.append([label, frac(sub["early_fraud"]), frac(sub["explicit_fraud"]), frac(sub["all_fraud"]), frac(sub["normal_false_positive"])])
    lines += [
        markdown_table(["模型", "early命中", "explicit命中", "全部fraud命中", "normal误报"], subgroup_rows),
        "",
        "### 按场景分层",
        "",
        markdown_table(
            ["场景", "M0 fraud", "M1 fraud", "M0 early", "M1 early", "M0 normal FP", "M1 normal FP"],
            [
                [
                    scenario,
                    frac(m0["scenario_metrics"][scenario]["fraud_recall"]),
                    frac(m1["scenario_metrics"][scenario]["fraud_recall"]),
                    frac(m0["scenario_metrics"][scenario]["early_fraud_recall"]),
                    frac(m1["scenario_metrics"][scenario]["early_fraud_recall"]),
                    frac(m0["scenario_metrics"][scenario]["normal_fpr"]),
                    frac(m1["scenario_metrics"][scenario]["normal_fpr"]),
                ]
                for scenario in sorted(m0["scenario_metrics"])
            ],
        ),
        "",
        f"场景宏平均：M0 fraud recall={fpct(m0['scenario_macro']['fraud_recall'])}、normal FPR={fpct(m0['scenario_macro']['normal_fpr'])}；M1 fraud recall={fpct(m1['scenario_macro']['fraud_recall'])}、normal FPR={fpct(m1['scenario_macro']['normal_fpr'])}。early_fraud 只有15条且仅覆盖部分场景，整体结论不得外推为所有诈骗场景均已改善。",
        "",
        "### ambiguous 分数分布",
        "",
        markdown_table(
            ["模型", "N", "均值", "中位数", "P90", ">=0.5"],
            [
                [label, m["ambiguous"]["samples"], fnum(m["ambiguous"]["mean_p_fraud"]), fnum(m["ambiguous"]["median_p_fraud"]), fnum(m["ambiguous"]["p90_p_fraud_linear"]), m["ambiguous"]["count_ge_0_5"]]
                for label, m in (("M0 V361", m0), ("M1 E1-B", m1))
            ],
        ),
        "",
        "### 推理时间",
        "",
        markdown_table(
            ["模型", "完整文本总时长(s)", "每通平均(ms)", "前缀数", "前缀总时长(s)", "每前缀(ms)"],
            [
                [
                    label,
                    fnum(metrics[mid]["timing"]["wall_time_seconds"], 3),
                    fnum(metrics[mid]["timing"]["milliseconds_per_call"], 2),
                    temporal[mid]["prefixes"],
                    fnum(temporal[mid]["timing"]["wall_time_seconds"], 3),
                    fnum(temporal[mid]["timing"]["milliseconds_per_prefix"], 2),
                ]
                for mid, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B"))
            ],
        ),
        "",
        "## 累计前缀时间序列",
        "",
        "报警时刻按连续命中窗口的末端轮次计算。premature 与有效检测可同时成立；延迟只在 tE 及以后首次有效报警上计算。",
        "",
    ]
    temporal_rows = []
    for k in (1, 2):
        for mid, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B")):
            p = temporal[mid]["policies"][f"k{k}"]
            for subset, subset_label in (("early_fraud", "early"), ("explicit_fraud", "explicit"), ("all_fraud", "all fraud")):
                s = p["subsets"][subset]
                temporal_rows.append(
                    [
                        f"k{k}", label, subset_label,
                        frac(s["tE_after_effective_recall"]),
                        frac(s["tE_before_premature_alarm_rate"]),
                        frac(s["clean_effective_rate"]),
                        f"{fnum(s['delay']['median_turns'], 1)} (N={s['delay']['detected_n']}, miss={s['delay']['missed_n']})",
                        frac(p["normal_ever_false_alarm"]),
                        frac(p["explicit_before_tA"]),
                        fnum(p["explicit_before_tA"]["signed_tA_minus_first_valid_median"], 1),
                    ]
                )
    lines += [
        markdown_table(
            ["策略", "模型", "子集", "tE后有效", "tE前抢报", "无抢报有效", "延迟中位数", "normal曾误报", "explicit在tA前", "tA-first-valid中位"],
            temporal_rows,
        ),
        "",
        "tA前成功的主分母是全部具有合法tA的65条explicit通话；其中1条tE=tA，没有严格的tE≤alarm<tA窗口。`temporal_metrics.json`另报有非空窗口的64条机会分母及有符号tA-first-valid分布。",
        "",
        "### ambiguous 时序轨迹（不判对错）",
        "",
        markdown_table(
            ["策略", "模型", "曾报警通话", "每通最大p均值", "每通最大p中位数"],
            [
                [
                    f"k{k}", label,
                    frac(temporal[mid]["policies"][f"k{k}"]["ambiguous_trajectory_only"]["ever_alert"]),
                    fnum(temporal[mid]["policies"][f"k{k}"]["ambiguous_trajectory_only"]["mean_of_call_max_p_fraud"]),
                    fnum(temporal[mid]["policies"][f"k{k}"]["ambiguous_trajectory_only"]["median_of_call_max_p_fraud"]),
                ]
                for k in (1, 2)
                for mid, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B"))
            ],
        ),
        "",
        "## 漏报、文本位置与长度",
        "",
    ]
    miss_rows = []
    for mid, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B")):
        for key, view in (("fulltext_false_negatives", "完整文本FN"), ("temporal_miss_k1", "k1 tE后漏"), ("temporal_miss_k2", "k2 tE后漏")):
            block = misses[mid][key]
            miss_rows.append([label, view, block["n"], json.dumps(block["by_tE_relative_position"], ensure_ascii=False), block["long_at_or_above_fraud_q75"], block["exceeded_diagnostic_12_chunks"], json.dumps(block["by_scenario"], ensure_ascii=False)])
    lines += [
        markdown_table(["模型", "口径", "漏报N", "tE位置", ">=fraud token Q75", ">12块", "场景"], miss_rows),
        "",
        f"长文本阈值按各模型相同输入上的 fraud Dev token 数 Q75 固定；M0阈值={fnum(misses['m0_v361']['fraud_wordpiece_q75_threshold'], 1)}，M1阈值={fnum(misses['m1_e1b']['fraud_wordpiece_q75_threshold'], 1)}。tE位置分箱固定为 <=1/3、(1/3,2/3]、>2/3。",
        "",
        "## 高频词依赖检查",
        "",
        "固定词表包括到期/失效、扣费/续费、关闭/取消三组，并单列“到期、扣费、关闭”。以下是分层相关性，不是因果证明。",
        "",
    ]
    kw_rows = []
    for mid, label in (("m0_v361", "M0 V361"), ("m1_e1b", "M1 E1-B")):
        block = keywords["models"][mid]["fulltext_strata"]["任一固定词"]
        kw_rows.append(
            [
                label,
                frac(block["fraud"]["keyword_present"]["predicted_fraud"]),
                frac(block["fraud"]["keyword_absent"]["predicted_fraud"]),
                frac(block["normal"]["keyword_present"]["predicted_fraud"]),
                frac(block["normal"]["keyword_absent"]["predicted_fraud"]),
                json.dumps(keywords["models"][mid]["alarm_relative_to_first_keyword"]["k1"]["counts"], ensure_ascii=False),
            ]
        )
    lines += [
        markdown_table(["模型", "fraud有词命中", "fraud无词命中", "normal有词误报", "normal无词误报", "k1报警相对首次关键词(80 fraud)"], kw_rows),
        "",
        "本次Dev没有显示模型明显依赖这些词：两模型对无固定词的16条fraud均命中16/16；normal含词与不含词的误报率没有一致升高；多数fraud的首次报警发生在关键词之前或文本根本不含固定词。这个结果只能排除“仅靠这些词”的简单解释，不能证明模型不存在其他词面捷径；若要下因果结论，还需预注册的遮蔽/改写对照实验。",
        "",
        "## M1 相对 M0 的逐条变化",
        "",
    ]
    transition_counts = Counter(str(row["transition"]) for row in transitions)
    lines += [
        markdown_table(["变化", "条数"], [[key, value] for key, value in sorted(transition_counts.items())]),
        "",
        "### M1 新增完整文本命中",
        "",
        markdown_table(
            ["call_id", "decision", "场景", "M0 p", "M1 p", "tE", "tA"],
            [
                [row["call_id"], row["decision"], row["scenario_tag"], fnum(row["m0_p_fraud"]), fnum(row["m1_p_fraud"]), row["tE_turn"], row["tA_turn"]]
                for row in transitions if row["transition"] == "m1_new_hit"
            ],
        ),
        "",
        "### M1 仍然完整文本漏报",
        "",
        markdown_table(
            ["call_id", "decision", "场景", "M1 p", "轮数", "tE", "核心证据"],
            [
                [
                    row["call_id"], row["decision"], row["scenario_tag"], fnum(row["m1_p_fraud"]), row["n_full_turns"], row["tE_turn"],
                    ((row.get("core_evidence") or [{}])[0].get("quote") or "")[:80],
                ]
                for row in transitions
                if row["decision"] in {"early_fraud", "explicit_fraud"} and int(row["m1_prediction"]) == 0
            ],
        ),
        "",
        "### M1 完整文本正常误报",
        "",
        markdown_table(
            ["call_id", "场景", "M0 p", "M1 p", "轮数", "相对M0"],
            [
                [row["call_id"], row["scenario_tag"], fnum(row["m0_p_fraud"]), fnum(row["m1_p_fraud"]), row["n_full_turns"], row["transition"]]
                for row in transitions if row["decision"] == "normal" and int(row["m1_prediction"]) == 1
            ],
        ),
        "",
        "### M1 相对 M0 的时序变化",
        "",
        markdown_table(
            ["策略", "变化", "条数", "call_id"],
            [
                [f"k{k}", change, len(rows), "、".join(row["call_id"] for row in rows)]
                for k in (1, 2)
                for change in (
                    "m1_new_effective_detection",
                    "m1_lost_effective_detection",
                    "m1_new_premature_alarm",
                    "m1_corrected_premature_alarm",
                    "m1_new_normal_ever_false_alarm",
                    "m1_corrected_normal_ever_false_alarm",
                )
                for rows in [[row for row in temporal_transitions if change in row["policies"][f"k{k}"]["changes"]]]
            ],
        ),
        "",
        "逐条完整变化、时序变化和错误证据均已写入 JSONL；错误清单包含模型报警轮次及裁定证据引文。",
        "",
        "## 合规与可复现性",
        "",
        f"- Dev SHA256：`{manifest['data']['sha256']}`。",
        f"- M0 checkpoint SHA256：`{manifest['models']['m0_v361']['checkpoint_sha256']}`。",
        f"- M1 checkpoint SHA256：`{manifest['models']['m1_e1b']['checkpoint_sha256']}`。",
        "- Test/restricted Test 文件访问数：0（脚本使用精确 Dev allowlist，不扫描数据目录）。",
        "- 原始推理程序自带 metrics 把缺少 label_13 的V365行视为正常，因此其分类指标被明确弃用；只取时间、覆盖率和严格加载诊断，全部分类指标由本报告按 Dev decision/mask 重算。",
        "- `turn_stage_labels` 未作为逐前缀独立标签或训练监督；仅使用最终裁定的 tE/tA 作为评测边界。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", type=Path, default=Path(EXPECTED_SOURCE_SUFFIX))
    parser.add_argument("--prefix-input", type=Path, required=True)
    parser.add_argument("--m0-full-pred", type=Path, required=True)
    parser.add_argument("--m0-full-metrics", type=Path, required=True)
    parser.add_argument("--m0-prefix-pred", type=Path, required=True)
    parser.add_argument("--m0-prefix-metrics", type=Path, required=True)
    parser.add_argument("--m1-full-pred", type=Path, required=True)
    parser.add_argument("--m1-full-metrics", type=Path, required=True)
    parser.add_argument("--m1-prefix-pred", type=Path, required=True)
    parser.add_argument("--m1-prefix-metrics", type=Path, required=True)
    parser.add_argument("--m0-checkpoint", type=Path, required=True)
    parser.add_argument("--m1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dev_rows = validate_dev_source(args.dev_source)
    prefix_inputs = read_jsonl(args.prefix_input)
    models = {
        "m0_v361": {
            "full_pred": args.m0_full_pred,
            "full_metrics": args.m0_full_metrics,
            "prefix_pred": args.m0_prefix_pred,
            "prefix_metrics": args.m0_prefix_metrics,
            "checkpoint": args.m0_checkpoint,
        },
        "m1_e1b": {
            "full_pred": args.m1_full_pred,
            "full_metrics": args.m1_full_metrics,
            "prefix_pred": args.m1_prefix_pred,
            "prefix_metrics": args.m1_prefix_metrics,
            "checkpoint": args.m1_checkpoint,
        },
    }
    full_by_model: dict[str, list[dict[str, Any]]] = {}
    prefix_by_model: dict[str, list[dict[str, Any]]] = {}
    temporal_calls_by_model: dict[str, list[dict[str, Any]]] = {}
    metrics_by_model: dict[str, dict[str, Any]] = {}
    temporal_metrics_by_model: dict[str, dict[str, Any]] = {}
    errors_by_model: dict[str, list[dict[str, Any]]] = {}
    miss_by_model: dict[str, dict[str, Any]] = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_input = args.prefix_input.parent / "v365_dev_fulltext_label_blind.jsonl"
    for model_id, files in models.items():
        raw_full = read_jsonl(files["full_pred"])
        raw_prefix = read_jsonl(files["prefix_pred"])
        full_timing = raw_timing(files["full_metrics"], files["checkpoint"], full_input)
        prefix_timing = raw_timing(files["prefix_metrics"], files["checkpoint"], args.prefix_input)
        full = build_full_records(dev_rows, raw_full, model_id)
        metric = binary_metrics(full, full_timing)
        prefix, temporal_calls = build_prefix_records(dev_rows, prefix_inputs, raw_prefix, model_id)
        temporal_metric = temporal_metrics(temporal_calls, prefix_timing)
        consistency = endpoint_consistency(full, temporal_calls, model_id)
        metric["diagnostics"]["full_vs_final_prefix"] = consistency
        temporal_metric["endpoint_consistency"] = consistency
        errors = build_errors(dev_rows, full, temporal_calls, model_id)
        misses = miss_analysis(dev_rows, full, temporal_calls)
        full_by_model[model_id] = full
        prefix_by_model[model_id] = prefix
        temporal_calls_by_model[model_id] = temporal_calls
        metrics_by_model[model_id] = metric
        temporal_metrics_by_model[model_id] = temporal_metric
        errors_by_model[model_id] = errors
        miss_by_model[model_id] = misses

        model_dir = args.output_dir / model_id
        write_jsonl(model_dir / "predictions.jsonl", full)
        write_json(model_dir / "metrics.json", metric)
        write_jsonl(model_dir / "prefix_predictions.jsonl", prefix)
        write_jsonl(model_dir / "temporal_call_details.jsonl", temporal_calls)
        write_json(model_dir / "temporal_metrics.json", temporal_metric)
        write_jsonl(model_dir / "errors.jsonl", errors)

    transitions = compare_full(dev_rows, full_by_model["m0_v361"], full_by_model["m1_e1b"])
    temporal_transitions = compare_temporal(
        dev_rows,
        temporal_calls_by_model["m0_v361"],
        temporal_calls_by_model["m1_e1b"],
    )
    keywords = keyword_analysis(dev_rows, full_by_model, temporal_calls_by_model, transitions)
    reports = args.output_dir / "reports"
    write_jsonl(reports / "comparison_transitions.jsonl", transitions)
    write_jsonl(reports / "temporal_comparison_transitions.jsonl", temporal_transitions)
    write_jsonl(reports / "m1_new_hits.jsonl", [row for row in transitions if row["transition"] == "m1_new_hit"])
    write_jsonl(reports / "m1_new_false_positives.jsonl", [row for row in transitions if row["transition"] == "m1_new_false_positive"])
    write_jsonl(reports / "m1_lost_hits.jsonl", [row for row in transitions if row["transition"] == "m1_lost_hit"])
    write_jsonl(reports / "all_model_errors.jsonl", errors_by_model["m0_v361"] + errors_by_model["m1_e1b"])
    write_json(reports / "keyword_analysis.json", keywords)
    write_json(reports / "miss_analysis.json", miss_by_model)

    comparison_rows: list[dict[str, Any]] = []
    for model_id in ("m0_v361", "m1_e1b"):
        m = metrics_by_model[model_id]
        comparison_rows.append(
            {
                "model_id": model_id,
                **m["confusion"],
                "accuracy": m["accuracy"],
                "balanced_accuracy": m["balanced_accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "normal_fpr": m["normal_fpr"],
                "roc_auc": m["roc_auc"],
                "pr_auc_ap": m["pr_auc_ap"],
                "early_hits": m["subgroups"]["early_fraud"]["numerator"],
                "explicit_hits": m["subgroups"]["explicit_fraud"]["numerator"],
                "normal_false_positives": m["subgroups"]["normal_false_positive"]["numerator"],
            }
        )
    csv_path = reports / "fulltext_model_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    manifest = {
        "schema_version": "v365.dev.zero_train.baseline.v1",
        "created_for": "V365 Dev existing-model zero-training baseline",
        "data": {
            "path": str(args.dev_source),
            "sha256": sha256_file(args.dev_source),
            "counts": {"total": 200, "normal": 80, "early_fraud": 15, "explicit_fraud": 65, "ambiguous": 40, "effective_binary": 160},
        },
        "models": {
            model_id: {
                "checkpoint_path": str(files["checkpoint"]),
                "checkpoint_sha256": sha256_file(files["checkpoint"]),
                "strict_load": True,
            }
            for model_id, files in models.items()
        },
        "config": {
            "binary_head_only": True,
            "fraud_score": "p_binary[1]",
            "route": "B",
            "input_mode": "head_context_enhanced_chunks",
            "chunk_length": 320,
            "overlap": 64,
            "head_context_length": 96,
            "diagnostic_max_chunks": 12,
            "hard_chunk_cap": None,
            "all_chunks_retained": True,
            "threshold": THRESHOLD,
            "batch_size": 2,
            "encoder_micro_batch_size": 12,
            "evidence_components": ["attention", "top1", "top2", "last", "mean"],
        },
        "safety": {
            "training_performed": False,
            "backward_or_optimizer_used": False,
            "new_model_weights_written": False,
            "test_or_restricted_test_files_opened": 0,
            "source_allowlist_enforced": True,
            "turn_stage_labels_used_for_training": False,
        },
        "outputs": {
            "models": ["m0_v361", "m1_e1b"],
            "reports_dir": str(reports),
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    comparison = {
        "fulltext": metrics_by_model,
        "temporal": temporal_metrics_by_model,
        "miss_analysis": miss_by_model,
        "transition_counts": dict(Counter(str(row["transition"]) for row in transitions)),
        "temporal_transition_counts": {
            f"k{k}": dict(
                Counter(
                    change
                    for row in temporal_transitions
                    for change in row["policies"][f"k{k}"]["changes"]
                )
            )
            for k in (1, 2)
        },
    }
    write_json(reports / "model_comparison.json", comparison)
    report = build_report(metrics_by_model, temporal_metrics_by_model, transitions, temporal_transitions, miss_by_model, keywords, manifest)
    (reports / "V365_DEV_BASELINE_REPORT.md").write_text(report + "\n", encoding="utf-8")

    forbidden_weights = [
        str(path)
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".bin", ".safetensors"}
    ]
    if forbidden_weights:
        raise AssertionError(f"unexpected model weights in output: {forbidden_weights}")
    print(json.dumps({"status": "done", "output": str(args.output_dir), "transition_counts": comparison["transition_counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
