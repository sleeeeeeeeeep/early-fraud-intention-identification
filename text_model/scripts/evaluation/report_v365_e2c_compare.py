#!/usr/bin/env python3
"""Dev-only five-model report and frozen E2-C promotion gate.

M0/M1/E2-A are reconstructed from the audited E2-A scored outputs, E2-B is
reconstructed from the canonical audited E2-B scored outputs, and only E2-C is
reconstructed from fresh raw inference.  Every read path is rejected before
opening when any path component contains ``test`` or ``restricted``.  This
program never trains, selects a threshold, changes an alarm policy, or opens a
V365 Test artefact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


THRESHOLD = 0.5
KEYWORD_TERMS = ("到期", "扣费", "关闭")
MODEL_ORDER = ("m0_v361", "m1_e1b", "e2a", "e2b", "e2c")
MODEL_LABELS = {
    "m0_v361": "M0 V361",
    "m1_e1b": "M1 E1-B",
    "e2a": "E2-A",
    "e2b": "E2-B",
    "e2c": "E2-C",
}
EXPECTED_DEV_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl"
)
EXPECTED_LONG_SUFFIX = (
    "data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl"
)
EXPECTED_E2A_ROOT_SUFFIX = "runs/v365_e2a_dev_compare_20261011"
EXPECTED_E2B_ROOT_SUFFIX = "runs/v365_e2b_dev_compare_20261012"
EXPECTED_E2B_MANIFEST_SCHEMA = "v365.e2b.dev_compare.v1"
EXPECTED_PREREG_SCHEMA = "v365.e2c.preregistration.v1"


def forbidden_path(path: Path) -> bool:
    """Reject a path if *any* component contains test/restricted."""
    for candidate in (path.absolute(), path.resolve(strict=False)):
        parts = [part.lower() for part in candidate.parts]
        if any("test" in part or "restricted" in part for part in parts):
            return True
    return False


def safe_file(path: Path, suffix: str | None = None) -> Path:
    if forbidden_path(path):
        raise SystemExit(f"refusing Test/restricted path: {path}")
    resolved = path.resolve()
    if suffix is not None and not resolved.as_posix().endswith(suffix):
        raise SystemExit(f"refusing non-allowlisted path: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def safe_dir(path: Path, suffix: str | None = None) -> Path:
    if forbidden_path(path):
        raise SystemExit(f"refusing Test/restricted directory: {path}")
    resolved = path.resolve()
    if suffix is not None and not resolved.as_posix().endswith(suffix):
        raise SystemExit(f"refusing non-allowlisted directory: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def safe_output(path: Path) -> Path:
    if forbidden_path(path):
        raise SystemExit(f"refusing Test/restricted output: {path}")
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    safe_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    safe_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_baseline_module() -> tuple[ModuleType, Path]:
    candidates = [
        Path(__file__).with_name("report_v365_dev_baseline.py"),
        Path.cwd() / "text_model/scripts/evaluation/report_v365_dev_baseline.py",
        Path.cwd() / "scripts/evaluation/report_v365_dev_baseline.py",
    ]
    module_path = next((p.resolve() for p in candidates if p.is_file()), None)
    if module_path is None:
        raise FileNotFoundError("report_v365_dev_baseline.py is required")
    safe_file(module_path)
    spec = importlib.util.spec_from_file_location("v365_baseline_report", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "validate_dev_source", "read_jsonl", "prediction_probability",
        "build_full_records", "binary_metrics", "build_prefix_records",
        "temporal_metrics", "raw_timing", "wilson_interval",
        "exact_mcnemar_p",
    }
    missing = sorted(required - set(dir(module)))
    if missing:
        raise ImportError(f"baseline report missing functions: {missing}")
    return module, module_path


def require_number(block: Mapping[str, Any], key: str) -> float:
    if key not in block or isinstance(block[key], bool) or not isinstance(block[key], (int, float)):
        raise ValueError(f"preregistration requires numeric field {key}")
    return float(block[key])


def validate_prereg(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("preregistration must be a JSON object")
    if payload.get("schema_version") != EXPECTED_PREREG_SCHEMA:
        raise ValueError("unexpected preregistration schema")
    if payload.get("status") != "frozen_before_data_build_and_training":
        raise ValueError("preregistration is not frozen-before-training")
    if payload.get("score") != "p_binary[1]" or float(payload.get("threshold", -1)) != THRESHOLD:
        raise ValueError("preregistered score/threshold differs from fixed evaluator")
    alarm = payload.get("primary_alarm_policy") or {}
    if alarm.get("consecutive_positive_turns") != 2 or alarm.get("k1_role") != "diagnostic_only":
        raise ValueError("preregistered primary alarm policy must be k2; k1 diagnostic only")
    validity = payload.get("validity") or {}
    noninf = ((payload.get("gates") or {}).get("noninferiority") or {})
    improve = ((payload.get("gates") or {}).get("improvement") or {})
    for key in (
        "expected_v365_dev_calls", "expected_v365_prefix_rows",
        "expected_v361_longdev_rows", "full_final_prefix_threshold_flips_max",
        "full_final_prefix_max_abs_probability_delta_max", "coverage_required",
        "strict_missing_keys_max", "strict_unexpected_keys_max",
        "test_or_restricted_files_read_max",
    ):
        require_number(validity, key)
    required_gate_fields = {
        "fulltext": ("early_hits_min", "explicit_hits_min", "fraud_hits_min", "normal_fp_max"),
        "temporal_k2": (
            "early_te_after_min", "all_te_after_min", "normal_ever_fp_max",
            "all_premature_max", "clean_effective_min", "explicit_before_ta_min",
            "all_delay_median_max", "all_delay_p75_max",
        ),
        "longdev": ("fraud_hits_min", "normal_fp_max", "new_false_negative_max"),
    }
    for section, keys in required_gate_fields.items():
        current = noninf.get(section) or {}
        for key in keys:
            require_number(current, key)
    current = improve.get("temporal_k2") or {}
    for key in (
        "normal_ever_fp_max", "all_premature_max",
        "normal_fp_plus_premature_max", "clean_effective_min",
    ):
        require_number(current, key)
    rules = payload.get("status_rules") or {}
    if set(rules) != {"invalid", "fail", "hold", "pass_for_freeze"}:
        raise ValueError("preregistration status_rules are incomplete")
    if "never read" not in str(payload.get("test_policy", "")).lower():
        raise ValueError("preregistration does not keep Test locked")
    return payload


def timing_from_scored(path: Path, expected_checkpoint: Path) -> dict[str, Any]:
    payload = read_json(path)
    timing = dict(payload.get("timing") or {})
    if float(timing.get("wall_time_seconds", 0)) <= 0:
        raise ValueError(f"missing audited timing: {path}")
    report = timing.get("checkpoint_load_report") or {}
    if report.get("missing") or report.get("unexpected"):
        raise ValueError(f"audited output used non-strict checkpoint load: {path}")
    if Path(str(report.get("checkpoint", ""))).resolve() != expected_checkpoint.resolve():
        raise ValueError(f"audited checkpoint identity mismatch: {path}")
    if abs(float(timing.get("coverage_min_raw", -1.0)) - 1.0) > 1e-9:
        raise ValueError(f"audited output has incomplete coverage: {path}")
    return timing


def raw_from_scored(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        p0, p1 = float(row["p_normal"]), float(row["p_fraud"])
        output.append(
            {
                "sample_id": row["sample_id"],
                "p_binary": [p0, p1],
                "raw_wordpiece_length": int(row.get("raw_wordpiece_length", 0)),
                "num_chunks": int(row.get("n_chunks", 0)),
                "max_chunks_exceeded": bool(row.get("max_chunks_exceeded", False)),
                "full_text_coverage": float(row.get("full_text_coverage", -1.0)),
                "top_evidence_chunk": row.get("top_evidence_chunk"),
                "top_evidence_chunks": row.get("top_evidence_chunks", []),
            }
        )
    return output


def long_label(row: Mapping[str, Any]) -> int:
    value = row.get("canonical_label_13", row.get("label_13"))
    if value is None:
        raise ValueError(f"{row.get('sample_id')}: missing long label")
    return int(value)


def wordpiece_bin(length: int) -> str:
    if length <= 512:
        return "wp_0000_0512"
    if length <= 1024:
        return "wp_0513_1024"
    if length <= 1600:
        return "wp_1025_1600"
    if length <= 2600:
        return "wp_1601_2600"
    return "wp_2601_plus"


def build_long_records(
    source_rows: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, Any]],
    model_id: str, baseline: ModuleType,
) -> list[dict[str, Any]]:
    raw = {str(row.get("sample_id")): row for row in raw_rows}
    source = {str(row.get("sample_id")): row for row in source_rows}
    if len(raw) != len(raw_rows) or set(raw) != set(source):
        raise ValueError(f"{model_id}: long Dev ids differ from source")
    output: list[dict[str, Any]] = []
    for sample_id, src in source.items():
        row = raw[sample_id]
        p0, p1 = baseline.prediction_probability(row)
        label = long_label(src)
        wp = int(row.get("raw_wordpiece_length", src.get("raw_wordpiece_length", 0)))
        record = {
            "model_id": model_id, "sample_id": sample_id, "label_13": label,
            "y_true": int(label != 0), "p_normal": p0, "p_fraud": p1,
            "prediction": int(p1 >= THRESHOLD), "scenario_tag": src.get("scenario_tag"),
            "length_bucket": str(src.get("length_bucket") or wordpiece_bin(wp)),
            "raw_wordpiece_length": wp, "n_chunks": int(row.get("num_chunks", 0)),
            "full_text_coverage": float(row.get("full_text_coverage", row.get("coverage_ratio", -1.0))),
        }
        if record["raw_wordpiece_length"] <= 0 or record["n_chunks"] <= 0:
            raise ValueError(f"{model_id}/{sample_id}: missing long diagnostics")
        output.append(record)
    return output


def generic_binary_metrics(records: Sequence[Mapping[str, Any]], timing: Mapping[str, Any]) -> dict[str, Any]:
    y = np.asarray([int(row["y_true"]) for row in records], dtype=int)
    p = np.asarray([float(row["p_fraud"]) for row in records], dtype=float)
    pred = (p >= THRESHOLD).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    normalized_timing = dict(timing)
    wall_time = float(normalized_timing.get("wall_time_seconds", 0.0))
    if wall_time <= 0.0 or not records:
        raise ValueError("long-Dev timing requires positive wall time and records")
    normalized_timing.update({
        "evaluated_calls": len(records),
        "seconds_per_call": wall_time / len(records),
        "milliseconds_per_call": 1000.0 * wall_time / len(records),
    })
    return {
        "samples": len(records), "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": (tp + tn) / len(records),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "precision": precision, "recall": recall, "f1": f1,
        "normal_fpr": fp / (fp + tn) if fp + tn else None,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_ap": float(average_precision_score(y, p)),
        "timing": normalized_timing,
        "diagnostics": {
            "coverage_min": min(float(row["full_text_coverage"]) for row in records),
            "max_chunks_observed": max(int(row["n_chunks"]) for row in records),
            "calls_exceeding_diagnostic_12_chunks": sum(int(row["n_chunks"]) > 12 for row in records),
            "diagnostic_max_chunks": 12,
            "hard_chunk_cap": None,
            "max_chunks_semantics": "diagnostic only; all chunks retained for full coverage",
        },
    }


def endpoint_stats(
    full_records: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    full = {str(row["call_id"]): row for row in full_records}
    temporal = {str(row["call_id"]): row for row in calls}
    if set(full) != set(temporal):
        raise ValueError("full/final-prefix call sets differ")
    deltas = {
        call_id: abs(float(full[call_id]["p_fraud"]) - float(temporal[call_id]["trajectory"]["final_p_fraud"]))
        for call_id in full
    }
    flips = [
        call_id for call_id in full
        if (float(full[call_id]["p_fraud"]) >= THRESHOLD)
        != (float(temporal[call_id]["trajectory"]["final_p_fraud"]) >= THRESHOLD)
    ]
    return {
        "calls": len(full), "threshold_flip_count": len(flips),
        "threshold_flip_call_ids": flips,
        "max_abs_p_fraud_delta": max(deltas.values(), default=0.0),
        "mean_abs_p_fraud_delta": float(np.mean(list(deltas.values()))) if deltas else 0.0,
    }


def validate_recomputed_against_audited(
    model_id: str, audited: Mapping[str, Any], full: Mapping[str, Any],
    temporal: Mapping[str, Any], long_metric: Mapping[str, Any],
) -> None:
    """Ensure a reconstructed canonical model retains its audited integers."""
    old_full = audited["v365_fulltext"][model_id]
    old_temporal = audited["v365_temporal"][model_id]
    old_long = audited["v361_longdev"][model_id]
    if dict(old_full["confusion"]) != dict(full["confusion"]):
        raise ValueError(f"{model_id}: recomputed full confusion differs from its audited comparison")
    for subset in ("early_fraud", "explicit_fraud", "all_fraud", "normal_false_positive"):
        if int(old_full["subgroups"][subset]["numerator"]) != int(full["subgroups"][subset]["numerator"]):
            raise ValueError(f"{model_id}: recomputed full subgroup differs: {subset}")
    for k in ("k1", "k2"):
        for subset in ("early_fraud", "explicit_fraud", "all_fraud"):
            for endpoint_name in (
                "tE_after_effective_recall", "tE_before_premature_alarm_rate", "clean_effective_rate",
            ):
                old_n = old_temporal["policies"][k]["subsets"][subset][endpoint_name]["numerator"]
                new_n = temporal["policies"][k]["subsets"][subset][endpoint_name]["numerator"]
                if int(old_n) != int(new_n):
                    raise ValueError(f"{model_id}: recomputed temporal endpoint differs: {k}/{subset}/{endpoint_name}")
        old_fp = old_temporal["policies"][k]["normal_ever_false_alarm"]["numerator"]
        new_fp = temporal["policies"][k]["normal_ever_false_alarm"]["numerator"]
        if int(old_fp) != int(new_fp):
            raise ValueError(f"{model_id}: recomputed normal trajectory endpoint differs: {k}")
    if dict(old_long["confusion"]) != dict(long_metric["confusion"]):
        raise ValueError(f"{model_id}: recomputed long confusion differs from its audited comparison")


def rate_observation(n: int, d: int, baseline: ModuleType) -> dict[str, Any]:
    return {
        "numerator": int(n), "denominator": int(d),
        "rate": n / d if d else None,
        "wilson_95": baseline.wilson_interval(n, d),
    }


def paired_result(
    endpoint: str, baseline_id: str, candidate: Mapping[str, bool],
    reference: Mapping[str, bool], baseline: ModuleType,
) -> dict[str, Any]:
    if set(candidate) != set(reference):
        raise ValueError(f"paired endpoint ids differ: {endpoint}/{baseline_id}")
    ids = sorted(candidate)
    gain = [key for key in ids if candidate[key] and not reference[key]]
    loss = [key for key in ids if reference[key] and not candidate[key]]
    c_success = sum(candidate.values())
    r_success = sum(reference.values())
    return {
        "endpoint": endpoint, "baseline_model": baseline_id,
        "paired_n": len(ids),
        "candidate_success": rate_observation(c_success, len(ids), baseline),
        "baseline_success": rate_observation(r_success, len(ids), baseline),
        "risk_difference_percentage_points": 100.0 * (c_success - r_success) / len(ids) if ids else None,
        "candidate_only_success_n": len(gain),
        "baseline_only_success_n": len(loss),
        "candidate_only_call_ids": gain, "baseline_only_call_ids": loss,
        "exact_binomial_mcnemar_two_sided_p": baseline.exact_mcnemar_p(len(loss), len(gain)),
    }


def endpoint_maps(
    full: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]],
    long_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, bool]]:
    result: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in full:
        call_id, decision = str(row["call_id"]), str(row["decision"])
        pred = int(row["prediction"])
        if decision == "early_fraud":
            result["full_early_hit"][call_id] = pred == 1
        if decision in {"early_fraud", "explicit_fraud"}:
            result["full_all_fraud_hit"][call_id] = pred == 1
        if decision == "normal":
            result["full_normal_no_fp"][call_id] = pred == 0
    for row in calls:
        call_id, decision = str(row["call_id"]), str(row["decision"])
        policy = row["policies"]["k2"]
        if decision == "early_fraud":
            result["k2_early_effective"][call_id] = bool(policy["effective"])
        if decision in {"early_fraud", "explicit_fraud"}:
            result["k2_all_effective"][call_id] = bool(policy["effective"])
            result["k2_all_no_premature"][call_id] = not bool(policy["premature"])
            result["k2_all_clean_effective"][call_id] = bool(policy["clean_effective"])
        if decision == "normal":
            result["k2_normal_no_ever_fp"][call_id] = not bool(policy["normal_ever_false_alarm"])
    for row in long_rows:
        sample_id = str(row["sample_id"])
        if int(row["y_true"]) == 1:
            result["long_fraud_hit"][sample_id] = int(row["prediction"]) == 1
        else:
            result["long_normal_no_fp"][sample_id] = int(row["prediction"]) == 0
    return dict(result)


def full_transition(decision: str, old_pred: int, new_pred: int) -> str:
    if decision in {"early_fraud", "explicit_fraud"}:
        if not old_pred and new_pred:
            return "e2c_new_hit"
        if old_pred and not new_pred:
            return "e2c_lost_hit"
        return "unchanged_hit" if new_pred else "unchanged_miss"
    if decision == "normal":
        if not old_pred and new_pred:
            return "e2c_new_false_positive"
        if old_pred and not new_pred:
            return "e2c_corrected_false_positive"
        return "unchanged_false_positive" if new_pred else "unchanged_true_negative"
    if not old_pred and new_pred:
        return "ambiguous_crossed_up"
    if old_pred and not new_pred:
        return "ambiguous_crossed_down"
    return "ambiguous_unchanged_side"


def binary_change(old: bool, new: bool, positive: str, negative: str) -> str:
    if not old and new:
        return f"e2c_new_{positive}"
    if old and not new:
        return f"e2c_lost_{positive}"
    return f"unchanged_{positive}" if new else f"unchanged_{negative}"


def build_transitions(
    dev_rows: Sequence[Mapping[str, Any]], old_full: Sequence[Mapping[str, Any]],
    new_full: Sequence[Mapping[str, Any]], old_calls: Sequence[Mapping[str, Any]],
    new_calls: Sequence[Mapping[str, Any]], baseline_id: str,
) -> list[dict[str, Any]]:
    source = {str(row["call_id"]): row for row in dev_rows}
    old_f = {str(row["call_id"]): row for row in old_full}
    new_f = {str(row["call_id"]): row for row in new_full}
    old_t = {str(row["call_id"]): row for row in old_calls}
    new_t = {str(row["call_id"]): row for row in new_calls}
    if not (set(source) == set(old_f) == set(new_f) == set(old_t) == set(new_t)):
        raise ValueError(f"{baseline_id}/E2-C transition ids differ")
    output: list[dict[str, Any]] = []
    for call_id in sorted(source):
        src = source[call_id]
        decision = str(src["decision"])
        row: dict[str, Any] = {
            "baseline_model": baseline_id, "call_id": call_id,
            "decision": decision, "scenario_tag": src.get("scenario_tag"),
            "n_full_turns": int(src["n_full_turns"]),
            "tE_turn": src.get("tE_turn"), "tA_turn": src.get("tA_turn"),
            "baseline_p_fraud": float(old_f[call_id]["p_fraud"]),
            "e2c_p_fraud": float(new_f[call_id]["p_fraud"]),
            "delta_p_fraud": float(new_f[call_id]["p_fraud"] - old_f[call_id]["p_fraud"]),
            "full_transition": full_transition(
                decision, int(old_f[call_id]["prediction"]), int(new_f[call_id]["prediction"])
            ),
        }
        for k in (1, 2):
            old_p, new_p = old_t[call_id]["policies"][f"k{k}"], new_t[call_id]["policies"][f"k{k}"]
            if decision in {"early_fraud", "explicit_fraud"}:
                row[f"k{k}_effective_transition"] = binary_change(
                    bool(old_p["effective"]), bool(new_p["effective"]), "effective_detection", "miss"
                )
                row[f"k{k}_premature_transition"] = binary_change(
                    bool(old_p["premature"]), bool(new_p["premature"]), "premature_alarm", "no_premature_alarm"
                )
                row[f"k{k}_clean_transition"] = binary_change(
                    bool(old_p["clean_effective"]), bool(new_p["clean_effective"]), "clean_effective", "not_clean_effective"
                )
            elif decision == "normal":
                row[f"k{k}_normal_fp_transition"] = binary_change(
                    bool(old_p["normal_ever_false_alarm"]), bool(new_p["normal_ever_false_alarm"]),
                    "normal_ever_false_alarm", "normal_never_false_alarm",
                )
        output.append(row)
    return output


def build_e2c_errors(
    dev_rows: Sequence[Mapping[str, Any]], full_rows: Sequence[Mapping[str, Any]],
    temporal_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build an error audit whose quotations occur only at real alarm turns."""
    source = {str(row["call_id"]): row for row in dev_rows}
    full = {str(row["call_id"]): row for row in full_rows}
    temporal = {str(row["call_id"]): row for row in temporal_rows}
    if not (set(source) == set(full) == set(temporal)):
        raise ValueError("E2-C error-audit call sets differ")
    output: list[dict[str, Any]] = []
    for call_id in sorted(source):
        src, frow, trow = source[call_id], full[call_id], temporal[call_id]
        decision = str(src["decision"])
        if decision == "ambiguous":
            continue
        errors: list[str] = []
        if decision in {"early_fraud", "explicit_fraud"}:
            if int(frow["prediction"]) == 0:
                errors.append("fulltext_false_negative")
            if not bool(trow["policies"]["k2"]["effective"]):
                errors.append("k2_miss_after_tE")
            if bool(trow["policies"]["k2"]["premature"]):
                errors.append("k2_premature_alarm")
        elif decision == "normal":
            if int(frow["prediction"]) == 1:
                errors.append("fulltext_false_positive")
            if bool(trow["policies"]["k2"]["normal_ever_false_alarm"]):
                errors.append("k2_normal_ever_false_alarm")
        if not errors:
            continue
        turn_text = {int(turn["turn_id"]): str(turn["text"]) for turn in src["turns"]}
        alarm_evidence: dict[str, Any] = {}
        for k in (1, 2):
            policy = trow["policies"][f"k{k}"]
            alarms = [int(value) for value in policy["alarm_turns"]]
            first_alarm = policy.get("first_any_turn")
            trigger_turns: list[int] = []
            if first_alarm is not None:
                first_alarm = int(first_alarm)
                start = first_alarm if k == 1 else max(1, first_alarm - 1)
                trigger_turns = list(range(start, first_alarm + 1))
            alarm_evidence[f"k{k}"] = {
                "first_alarm_turn": first_alarm,
                "all_alarm_turns": alarms,
                "trigger_window_turns": trigger_turns,
                "quotes_at_trigger_window": [
                    {"turn": turn, "quote": turn_text[turn]}
                    for turn in trigger_turns if turn in turn_text
                ],
                "quotes_after_first_alarm_included": False,
            }
        miss_evidence: list[dict[str, Any]] = []
        if "k2_miss_after_tE" in errors:
            t_e = int(src["tE_turn"])
            miss_evidence = [
                {"turn": int(item["turn_id"]), "quote": str(item["quote"]),
                 "evidence_type": item.get("evidence_type")}
                for item in src.get("core_evidence", [])
                if int(item.get("turn_id", t_e + 1)) <= t_e
            ]
            if not miss_evidence and t_e in turn_text:
                miss_evidence = [{"turn": t_e, "quote": turn_text[t_e],
                                  "evidence_type": "tE_anchor"}]
        output.append({
            "call_id": call_id, "decision": decision,
            "scenario_tag": src.get("scenario_tag"),
            "n_full_turns": int(src["n_full_turns"]),
            "tE_turn": src.get("tE_turn"), "tA_turn": src.get("tA_turn"),
            "p_fraud": float(frow["p_fraud"]),
            "prediction": int(frow["prediction"]), "error_types": errors,
            "alarm_evidence": alarm_evidence,
            "miss_evidence_at_or_before_tE": miss_evidence,
            "evidence_policy": "alarm quotations are restricted to the first trigger window; miss anchors are restricted to turn <= tE",
        })
    return output


def _observed_rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": numerator / denominator if denominator else None,
    }


def build_keyword_diagnostics(
    dev_rows: Sequence[Mapping[str, Any]],
    full_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Observational keyword strata; this is not a counterfactual causal test."""
    source = {str(row["call_id"]): row for row in dev_rows}
    result: dict[str, Any] = {
        "terms": list(KEYWORD_TERMS),
        "scope": "V365 Dev full text; ambiguous excluded from hit/false-positive rates",
        "causal_interpretation": "descriptive only; keyword presence is confounded with scenario and decision",
        "models": {},
    }
    for model_id in MODEL_ORDER:
        rows = list(full_records[model_id])
        by_id = {str(row["call_id"]): row for row in rows}
        if set(by_id) != set(source):
            raise ValueError(f"{model_id}: keyword diagnostic call ids differ")
        model_result: dict[str, Any] = {}
        for bucket_name, predicate in (
            ("any_keyword_present", lambda text: any(term in text for term in KEYWORD_TERMS)),
            ("all_keywords_absent", lambda text: not any(term in text for term in KEYWORD_TERMS)),
        ):
            selected = [
                by_id[call_id] for call_id, src in source.items()
                if predicate(str(src.get("model_input_text", "")))
            ]
            fraud = [row for row in selected if row["decision"] in {"early_fraud", "explicit_fraud"}]
            normal = [row for row in selected if row["decision"] == "normal"]
            model_result[bucket_name] = {
                "calls_including_ambiguous": len(selected),
                "fraud_hit": _observed_rate(sum(int(row["prediction"]) == 1 for row in fraud), len(fraud)),
                "normal_false_positive": _observed_rate(sum(int(row["prediction"]) == 1 for row in normal), len(normal)),
                "mean_p_fraud_all_decisions": float(np.mean([float(row["p_fraud"]) for row in selected])) if selected else None,
            }
        result["models"][model_id] = model_result
    return result


def check(
    check_id: str, group: str, observed: float, operator: str, limit: float,
    evidence: str,
) -> dict[str, Any]:
    passed = observed <= limit if operator == "<=" else observed >= limit if operator == ">=" else observed == limit
    return {
        "id": check_id, "group": group, "observed": observed,
        "operator": operator, "limit": limit, "passed": bool(passed),
        "evidence": evidence,
    }


def evaluate_gate(
    prereg: Mapping[str, Any], dev_rows: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, Any]], long_rows: Sequence[Mapping[str, Any]],
    full_metrics: Mapping[str, Mapping[str, Any]],
    temporal_metrics: Mapping[str, Mapping[str, Any]],
    long_metrics: Mapping[str, Mapping[str, Any]],
    full_records: Mapping[str, Sequence[Mapping[str, Any]]],
    endpoint: Mapping[str, Mapping[str, Any]], timing_e2c: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validity = prereg["validity"]
    noninf = prereg["gates"]["noninferiority"]
    improve = prereg["gates"]["improvement"]["temporal_k2"]
    e2_full = full_metrics["e2c"]
    e2_temp = temporal_metrics["e2c"]["policies"]["k2"]
    e2_long = long_metrics["e2c"]
    early = e2_full["subgroups"]["early_fraud"]
    explicit = e2_full["subgroups"]["explicit_fraud"]
    fraud = e2_full["subgroups"]["all_fraud"]
    normal = e2_full["subgroups"]["normal_false_positive"]
    te_early = e2_temp["subsets"]["early_fraud"]["tE_after_effective_recall"]
    te_all = e2_temp["subsets"]["all_fraud"]["tE_after_effective_recall"]
    premature = e2_temp["subsets"]["all_fraud"]["tE_before_premature_alarm_rate"]
    clean = e2_temp["subsets"]["all_fraud"]["clean_effective_rate"]
    normal_ever = e2_temp["normal_ever_false_alarm"]
    delay = e2_temp["subsets"]["all_fraud"]["delay"]
    before_ta = e2_temp["explicit_before_tA"]
    e2c_long_by_id = {str(row["sample_id"]): row for row in full_records["e2c_long"]}
    m1_long_by_id = {str(row["sample_id"]): row for row in full_records["m1_long"]}
    new_fn = sum(
        int(row["y_true"]) == 1 and int(row["prediction"]) == 0
        and int(m1_long_by_id[sample_id]["prediction"]) == 1
        for sample_id, row in e2c_long_by_id.items()
    )
    missing_max = max(len((item.get("checkpoint_load_report") or {}).get("missing", [])) for item in timing_e2c)
    unexpected_max = max(len((item.get("checkpoint_load_report") or {}).get("unexpected", [])) for item in timing_e2c)
    coverage_min = min(
        float(e2_full["diagnostics"]["coverage_min"]),
        float(e2_long["diagnostics"]["coverage_min"]),
        *(float(item.get("coverage_min_raw", -1.0)) for item in timing_e2c),
    )
    checks: list[dict[str, Any]] = []
    add = checks.append
    add(check("valid.dev_calls", "validity", len(dev_rows), "==", validity["expected_v365_dev_calls"], "V365 Dev source"))
    add(check("valid.prefix_rows", "validity", len(prefix_rows), "==", validity["expected_v365_prefix_rows"], "future-free prefix input"))
    add(check("valid.long_rows", "validity", len(long_rows), "==", validity["expected_v361_longdev_rows"], "V361 long Dev source"))
    add(check("valid.endpoint_flips", "validity", endpoint["e2c"]["threshold_flip_count"], "<=", validity["full_final_prefix_threshold_flips_max"], "E2-C full vs final prefix"))
    add(check("valid.endpoint_delta", "validity", endpoint["e2c"]["max_abs_p_fraud_delta"], "<=", validity["full_final_prefix_max_abs_probability_delta_max"], "E2-C full vs final prefix"))
    add(check("valid.coverage", "validity", coverage_min, ">=", validity["coverage_required"], "minimum E2-C raw/scored coverage"))
    add(check("valid.missing_keys", "validity", missing_max, "<=", validity["strict_missing_keys_max"], "three E2-C inference loads"))
    add(check("valid.unexpected_keys", "validity", unexpected_max, "<=", validity["strict_unexpected_keys_max"], "three E2-C inference loads"))
    add(check("valid.test_reads", "validity", 0, "<=", validity["test_or_restricted_files_read_max"], "path guard audit counter"))
    nf = noninf["fulltext"]
    nk2 = noninf["temporal_k2"]
    nlong = noninf["longdev"]
    add(check("noninf.full.early_hits", "noninferiority", early["numerator"], ">=", nf["early_hits_min"], "E2-C full early_fraud"))
    add(check("noninf.full.explicit_hits", "noninferiority", explicit["numerator"], ">=", nf["explicit_hits_min"], "E2-C full explicit_fraud"))
    add(check("noninf.full.fraud_hits", "noninferiority", fraud["numerator"], ">=", nf["fraud_hits_min"], "E2-C full all fraud"))
    add(check("noninf.full.normal_fp", "noninferiority", normal["numerator"], "<=", nf["normal_fp_max"], "E2-C full normal"))
    add(check("noninf.k2.early_effective", "noninferiority", te_early["numerator"], ">=", nk2["early_te_after_min"], "E2-C k2 early tE-after"))
    add(check("noninf.k2.all_effective", "noninferiority", te_all["numerator"], ">=", nk2["all_te_after_min"], "E2-C k2 all fraud tE-after"))
    add(check("noninf.k2.normal_ever_fp", "noninferiority", normal_ever["numerator"], "<=", nk2["normal_ever_fp_max"], "E2-C k2 normal"))
    add(check("noninf.k2.premature", "noninferiority", premature["numerator"], "<=", nk2["all_premature_max"], "E2-C k2 all fraud"))
    add(check("noninf.k2.clean_effective", "noninferiority", clean["numerator"], ">=", nk2["clean_effective_min"], "E2-C k2 all fraud"))
    add(check("noninf.k2.explicit_before_ta", "noninferiority", before_ta["numerator"], ">=", nk2["explicit_before_ta_min"], "E2-C k2 explicit; fixed legacy denominator includes no-window call"))
    add(check("noninf.k2.delay_median", "noninferiority", delay["median_turns"], "<=", nk2["all_delay_median_max"], "E2-C k2 detected fraud"))
    add(check("noninf.k2.delay_p75", "noninferiority", delay["p75_turns"], "<=", nk2["all_delay_p75_max"], "E2-C k2 detected fraud"))
    add(check("noninf.long.fraud_hits", "noninferiority", e2_long["confusion"]["tp"], ">=", nlong["fraud_hits_min"], "E2-C V361 long Dev"))
    add(check("noninf.long.normal_fp", "noninferiority", e2_long["confusion"]["fp"], "<=", nlong["normal_fp_max"], "E2-C V361 long Dev"))
    add(check("noninf.long.new_fn", "noninferiority", new_fn, "<=", nlong["new_false_negative_max"], "E2-C vs M1 V361 long Dev"))
    add(check("improve.k2.normal_ever_fp", "improvement", normal_ever["numerator"], "<=", improve["normal_ever_fp_max"], "E2-C k2 normal"))
    add(check("improve.k2.premature", "improvement", premature["numerator"], "<=", improve["all_premature_max"], "E2-C k2 all fraud"))
    add(check("improve.k2.fp_plus_premature", "improvement", normal_ever["numerator"] + premature["numerator"], "<=", improve["normal_fp_plus_premature_max"], "sum of two preregistered k2 harm counts"))
    add(check("improve.k2.clean_effective", "improvement", clean["numerator"], ">=", improve["clean_effective_min"], "E2-C k2 all fraud"))
    valid_pass = all(item["passed"] for item in checks if item["group"] == "validity")
    noninf_pass = all(item["passed"] for item in checks if item["group"] == "noninferiority")
    improve_pass = all(item["passed"] for item in checks if item["group"] == "improvement")
    if not valid_pass:
        status = "INVALID"
    elif not noninf_pass:
        status = "FAIL"
    elif not improve_pass:
        status = "HOLD"
    else:
        status = "PASS_FOR_FREEZE"
    return {
        "schema_version": "v365.e2c.gate.v1", "status": status,
        "status_rule_applied": prereg["status_rules"][status.lower()],
        "validity_passed": valid_pass, "noninferiority_passed": noninf_pass,
        "improvement_passed": improve_pass, "checks": checks,
        "test_policy": "LOCKED: PASS_FOR_FREEZE does not authorize or trigger Test access",
        "test_or_restricted_files_opened": 0,
    }


def number(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def ratio(block: Mapping[str, Any]) -> str:
    return f"{block['numerator']}/{block['denominator']} ({100 * block['rate']:.2f}%)" if block.get("rate") is not None else f"{block['numerator']}/{block['denominator']} (NA)"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(
        ["| " + " | ".join(map(clean, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
    )


def build_report(
    full: Mapping[str, Mapping[str, Any]], temporal: Mapping[str, Mapping[str, Any]],
    long_metrics: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]], transitions: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Mapping[str, Any], e2c_errors: Sequence[Mapping[str, Any]],
    keyword_diagnostics: Mapping[str, Any],
) -> str:
    full_rows, long_rows = [], []
    for model_id in MODEL_ORDER:
        m = full[model_id]
        full_rows.append([
            MODEL_LABELS[model_id], m["effective_samples"],
            m["confusion"]["tp"], m["confusion"]["fp"],
            m["confusion"]["tn"], m["confusion"]["fn"], number(m["accuracy"]),
            number(m["balanced_accuracy"]), number(m["precision"]), number(m["recall"]),
            number(m["f1"]), number(m["normal_fpr"]), number(m["roc_auc"]),
            number(m["pr_auc_ap"]), ratio(m["subgroups"]["early_fraud"]),
            ratio(m["subgroups"]["explicit_fraud"]), ratio(m["subgroups"]["normal_false_positive"]),
        ])
        lm = long_metrics[model_id]
        long_rows.append([
            MODEL_LABELS[model_id], lm["confusion"]["tp"], lm["confusion"]["fp"],
            lm["confusion"]["tn"], lm["confusion"]["fn"], number(lm["accuracy"]),
            number(lm["recall"]), number(lm["normal_fpr"]), number(lm["roc_auc"]),
        ])
    timing_rows = []
    for model_id in MODEL_ORDER:
        full_t = full[model_id]["timing"]
        prefix_t = temporal[model_id]["timing"]
        long_t = long_metrics[model_id]["timing"]
        timing_rows.append([
            MODEL_LABELS[model_id],
            number(full_t.get("wall_time_seconds"), 2),
            number(1000.0 * float(full_t.get("seconds_per_call", 0.0)), 2),
            number(prefix_t.get("wall_time_seconds"), 2),
            number(1000.0 * float(prefix_t.get("seconds_per_prefix", 0.0)), 2),
            number(long_t.get("wall_time_seconds"), 2),
            number(long_t.get("milliseconds_per_call"), 2),
            number(max(
                float(full_t.get("peak_gpu_memory_mb", 0.0)),
                float(prefix_t.get("peak_gpu_memory_mb", 0.0)),
                float(long_t.get("peak_gpu_memory_mb", 0.0)),
            ), 1),
        ])
    def temporal_rows(k: int) -> list[list[Any]]:
        rows = []
        for model_id in MODEL_ORDER:
            policy = temporal[model_id]["policies"][f"k{k}"]
            all_fraud = policy["subsets"]["all_fraud"]
            rows.append([
                MODEL_LABELS[model_id],
                ratio(policy["subsets"]["early_fraud"]["tE_after_effective_recall"]),
                ratio(all_fraud["tE_after_effective_recall"]),
                ratio(policy["normal_ever_false_alarm"]),
                ratio(all_fraud["tE_before_premature_alarm_rate"]),
                ratio(all_fraud["clean_effective_rate"]),
                number(all_fraud["delay"]["median_turns"], 1),
                number(all_fraud["delay"]["p75_turns"], 1),
                ratio(policy["explicit_before_tA"]),
            ])
        return rows
    k2_subgroup_rows: list[list[Any]] = []
    for model_id in MODEL_ORDER:
        for subset_id, subset_label in (
            ("early_fraud", "early_fraud"),
            ("explicit_fraud", "explicit_fraud"),
            ("all_fraud", "all fraud"),
        ):
            block = temporal[model_id]["policies"]["k2"]["subsets"][subset_id]
            k2_subgroup_rows.append([
                MODEL_LABELS[model_id], subset_label,
                ratio(block["tE_after_effective_recall"]),
                ratio(block["tE_before_premature_alarm_rate"]),
                ratio(block["clean_effective_rate"]),
                number(block["delay"]["median_turns"], 1),
                number(block["delay"]["p75_turns"], 1),
            ])
    ambiguous_full_rows = []
    ambiguous_temporal_rows = []
    for model_id in MODEL_ORDER:
        ambiguous = full[model_id]["ambiguous"]
        ambiguous_full_rows.append([
            MODEL_LABELS[model_id], ambiguous["samples"],
            number(ambiguous["mean_p_fraud"]), number(ambiguous["median_p_fraud"]),
            number(ambiguous["p90_p_fraud_linear"]), ambiguous["count_ge_0_5"],
        ])
        for k in (1, 2):
            trajectory = temporal[model_id]["policies"][f"k{k}"]["ambiguous_trajectory_only"]
            ambiguous_temporal_rows.append([
                MODEL_LABELS[model_id], f"k{k}", trajectory["calls"],
                ratio(trajectory["ever_alert"]),
                number(trajectory["mean_of_call_max_p_fraud"]),
                number(trajectory["median_of_call_max_p_fraud"]),
            ])
    keyword_rows: list[list[Any]] = []
    for model_id in MODEL_ORDER:
        for bucket_id, bucket_label in (
            ("any_keyword_present", "含任一关键词"),
            ("all_keywords_absent", "三个关键词均无"),
        ):
            block = keyword_diagnostics["models"][model_id][bucket_id]
            keyword_rows.append([
                MODEL_LABELS[model_id], bucket_label,
                block["calls_including_ambiguous"], ratio(block["fraud_hit"]),
                ratio(block["normal_false_positive"]),
                number(block["mean_p_fraud_all_decisions"]),
            ])
    error_counts = Counter(
        error_type for row in e2c_errors for error_type in row["error_types"]
    )
    error_summary_rows = [[name, count] for name, count in sorted(error_counts.items())]
    critical_error_rows = []
    critical_types = {"fulltext_false_negative", "fulltext_false_positive", "k2_miss_after_tE"}
    for row in e2c_errors:
        selected_types = sorted(set(row["error_types"]) & critical_types)
        if selected_types:
            critical_error_rows.append([
                row["call_id"], row["decision"], row["scenario_tag"],
                row["n_full_turns"], row["tE_turn"], row["tA_turn"],
                number(row["p_fraud"]), ", ".join(selected_types),
            ])
    m1_change_rows = []
    for row in transitions["m1_e1b"]:
        for field in (
            "full_transition", "k2_effective_transition",
            "k2_premature_transition", "k2_normal_fp_transition",
            "k2_clean_transition",
        ):
            value = row.get(field)
            if isinstance(value, str) and value.startswith("e2c_"):
                m1_change_rows.append([
                    row["call_id"], row["decision"], row["scenario_tag"],
                    field, value, number(row["baseline_p_fraud"]),
                    number(row["e2c_p_fraud"]),
                ])
    gate_rows = [[item["group"], item["id"], item["observed"], item["operator"], item["limit"], "PASS" if item["passed"] else "FAIL"] for item in gate["checks"]]
    paired_rows = [[
        MODEL_LABELS[item["baseline_model"]], item["endpoint"], item["paired_n"],
        item["candidate_success"]["numerator"], item["baseline_success"]["numerator"],
        number(item["risk_difference_percentage_points"], 2),
        item["candidate_only_success_n"], item["baseline_only_success_n"],
        number(item["exact_binomial_mcnemar_two_sided_p"]),
    ] for item in paired]
    transition_rows = []
    for baseline_id, rows in transitions.items():
        for field in (
            "full_transition", "k2_effective_transition", "k2_premature_transition",
            "k2_normal_fp_transition", "k2_clean_transition",
        ):
            counts = Counter(str(row[field]) for row in rows if field in row)
            for name, count in sorted(counts.items()):
                transition_rows.append([MODEL_LABELS[baseline_id], field, name, count])
    opportunity_rows = []
    for model_id in MODEL_ORDER:
        block = temporal[model_id]["policies"]["k2"]["explicit_before_tA"]
        opportunity_rows.append([
            MODEL_LABELS[model_id], ratio(block),
            ratio(block["among_calls_with_nonempty_tE_to_tA_window"]),
            block["no_strict_pre_tA_window_n"],
        ])
    scenario_names = sorted({
        scenario for model_id in MODEL_ORDER
        for scenario in full[model_id]["scenario_metrics"]
    })
    scenario_rows = []
    macro_rows = []
    for model_id in MODEL_ORDER:
        macro = full[model_id]["scenario_macro"]
        macro_rows.append([
            MODEL_LABELS[model_id], number(macro["fraud_recall"]),
            number(macro["normal_fpr"]), macro["fraud_scenarios"], macro["normal_scenarios"],
        ])
        for scenario in scenario_names:
            block = full[model_id]["scenario_metrics"].get(scenario)
            if block is None:
                continue
            scenario_rows.append([
                MODEL_LABELS[model_id], scenario,
                ratio(block["fraud_recall"]), ratio(block["early_fraud_recall"]),
                ratio(block["normal_fpr"]),
            ])
    e2c_keyword = keyword_diagnostics["models"]["e2c"]
    keyword_present = e2c_keyword["any_keyword_present"]
    keyword_absent = e2c_keyword["all_keywords_absent"]
    keyword_conclusion = (
        "E2-C在含任一关键词的样本中诈骗命中"
        f"{ratio(keyword_present['fraud_hit'])}、正常误报{ratio(keyword_present['normal_false_positive'])}；"
        "三个关键词均不存在时分别为"
        f"{ratio(keyword_absent['fraud_hit'])}和{ratio(keyword_absent['normal_false_positive'])}。"
        "本Dev结果不支持把误报简单归因于这三个词，但该分层受场景与标签混杂影响，"
        "不能代替关键词遮蔽或反事实重写实验。"
    )
    return "\n".join([
        "# V365 Dev：M0 / M1 / E2-A / E2-B / E2-C 二分类对比", "",
        f"**预登记门禁结论：{gate['status']}。** 固定分数 `p_binary[1]`、阈值0.5；k2为主策略，k1只作诊断。无论结论为何，V365 Test仍封存。", "",
        "## 完整文本", "",
        table(["模型", "有效N", "TP", "FP", "TN", "FN", "Acc", "BAcc", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "AP", "early", "explicit", "normal FP"], full_rows), "",
        "### ambiguous完整文本分数（不计对错）", "",
        table(["模型", "N", "平均p_fraud", "中位p_fraud", "P90 p_fraud", ">=0.5条数"], ambiguous_full_rows), "",
        "## 累计前缀：k2主端点", "",
        table(["模型", "early tE后", "全部tE后", "normal曾误报", "tE前抢报", "clean effective", "延迟中位", "延迟P75", "explicit tA前"], temporal_rows(2)), "",
        "### k2按诈骗阶段分别统计", "",
        table(["模型", "子集", "tE后有效检测", "tE前抢报", "clean effective", "延迟中位", "延迟P75"], k2_subgroup_rows), "",
        "## 累计前缀：k1诊断", "",
        table(["模型", "early tE后", "全部tE后", "normal曾误报", "tE前抢报", "clean effective", "延迟中位", "延迟P75", "explicit tA前"], temporal_rows(1)), "",
        "### ambiguous累计轨迹（不计对错）", "",
        table(["模型", "策略", "通话数", "曾报警", "通话内最大分数均值", "通话内最大分数中位"], ambiguous_temporal_rows), "",
        "### explicit_fraud 的tA机会窗口诊断", "",
        "预登记门槛继续使用固定旧口径（全部65条）。另列仅 `tA>tE` 的机会调整口径；无窗口通话不从门槛分母中悄然删除。", "",
        table(["模型", "固定旧口径", "仅有严格窗口", "无严格窗口"], opportunity_rows), "",
        "## 完整文本场景诊断（不参与promotion gate）", "",
        "场景宏平均与逐场景原始分子/分母用于识别由高频订阅扣费场景主导的表面提升。", "",
        table(["模型", "诈骗Recall宏平均", "Normal FPR宏平均", "诈骗场景数", "正常场景数"], macro_rows), "",
        table(["模型", "场景", "全部诈骗命中", "early命中", "normal误报"], scenario_rows), "",
        "## 错误与转移审计", "",
        "完整逐轮证据保存在 `reports/e2c_error_audit_future_filtered.jsonl`；报警引文只保留首次触发窗口，漏报证据只保留 `tE` 及以前。", "",
        table(["E2-C错误类型", "通话数"], error_summary_rows), "",
        table(["call_id", "decision", "scenario", "轮数", "tE", "tA", "完整文本p_fraud", "关键错误"], critical_error_rows), "",
        "### E2-C相对M1发生的离散变化", "",
        table(["call_id", "decision", "scenario", "端点", "变化", "M1完整文本p", "E2-C完整文本p"], m1_change_rows), "",
        "## 高频词依赖的观察性诊断", "",
        "关键词固定为“到期、扣费、关闭”。ambiguous计入总通话数和均值，但不计诈骗命中或正常误报。", "",
        table(["模型", "文本分层", "通话数（含ambiguous）", "诈骗命中", "正常误报", "全决策平均p_fraud"], keyword_rows), "",
        keyword_conclusion, "",
        "## 原V361长文本Dev", "",
        table(["模型", "TP", "FP", "TN", "FN", "Acc", "Recall", "FPR", "ROC-AUC"], long_rows), "",
        "## 推理时间与资源", "",
        table(["模型", "full总秒", "full毫秒/通", "prefix总秒", "毫秒/前缀", "long总秒", "long毫秒/通", "峰值显存MB"], timing_rows), "",
        "## 冻结门禁逐项结果", "",
        table(["组", "检查", "观测", "关系", "界值", "结果"], gate_rows), "",
        "## 主要配对端点（E2-C相对M0/M1/E2-A/E2-B）", "",
        "成功率均附Wilson 95%区间于JSON；p值为discordant pairs的双侧exact-binomial McNemar检验，未作多重比较校正。", "",
        table(["基线", "端点", "N", "E2-C成功", "基线成功", "风险差pp", "E2-C独有", "基线独有", "exact p"], paired_rows), "",
        "## 离散转移计数", "",
        table(["基线", "字段", "转移", "条数"], transition_rows), "",
        "## 合规与复现", "",
        f"- 预登记SHA256：`{manifest['preregistration']['sha256']}`。",
        f"- 本报告代码SHA256：`{manifest['code']['report']['sha256']}`。",
        f"- 推理代码SHA256：`{manifest['code']['inference']['sha256']}`。",
        "- M0/M1/E2-A从已审计E2-A scored输出重建并重算；E2-B从canonical E2-B scored输出重建并重算；仅E2-C来自本次raw inference。",
        "- 旧模型耗时是历史审计记录，E2-C耗时是本次运行记录；模型结构相同，但跨运行时间不作promotion gate。",
        "- `max_chunks=12`沿用项目既有语义：它是诊断阈值而非硬截断；`hard_chunk_cap=None`，所有块均保留且覆盖率必须为1。",
        "- Test/restricted读取数为0；路径的任一组成部分含test/restricted都会在打开前被拒绝。",
        "- 本脚本不训练、不调阈值、不调报警策略、不保存模型权重；PASS_FOR_FREEZE也不自动触发Test。", "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--prefix-input", type=Path, required=True)
    parser.add_argument("--full-input", type=Path)
    parser.add_argument("--e2a-root", type=Path, required=True)
    parser.add_argument("--e2b-root", type=Path, required=True)
    parser.add_argument("--e2c-full-pred", type=Path, required=True)
    parser.add_argument("--e2c-full-metrics", type=Path, required=True)
    parser.add_argument("--e2c-prefix-pred", type=Path, required=True)
    parser.add_argument("--e2c-prefix-metrics", type=Path, required=True)
    parser.add_argument("--long-dev-source", type=Path, required=True)
    parser.add_argument("--e2c-long-pred", type=Path, required=True)
    parser.add_argument("--e2c-long-metrics", type=Path, required=True)
    for name in ("m0", "m1", "e2a", "e2b", "e2c"):
        parser.add_argument(f"--{name}-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--inference-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    baseline, baseline_path = load_baseline_module()
    dev_source = safe_file(args.dev_source, EXPECTED_DEV_SUFFIX)
    long_source = safe_file(args.long_dev_source, EXPECTED_LONG_SUFFIX)
    e2a_root = safe_dir(args.e2a_root, EXPECTED_E2A_ROOT_SUFFIX)
    e2b_root = safe_dir(args.e2b_root, EXPECTED_E2B_ROOT_SUFFIX)
    prefix_input = safe_file(args.prefix_input)
    full_input = safe_file(args.full_input if args.full_input else prefix_input.parent / "v365_dev_fulltext_label_blind.jsonl")
    prereg_path = safe_file(args.preregistration)
    inference_code = safe_file(args.inference_code)
    output_dir = safe_output(args.output_dir)
    prereg = validate_prereg(read_json(prereg_path))
    checkpoints = {
        "m0_v361": safe_file(args.m0_checkpoint), "m1_e1b": safe_file(args.m1_checkpoint),
        "e2a": safe_file(args.e2a_checkpoint), "e2b": safe_file(args.e2b_checkpoint),
        "e2c": safe_file(args.e2c_checkpoint),
    }
    e2c_files = {
        "full_pred": safe_file(args.e2c_full_pred), "full_metrics": safe_file(args.e2c_full_metrics),
        "prefix_pred": safe_file(args.e2c_prefix_pred), "prefix_metrics": safe_file(args.e2c_prefix_metrics),
        "long_pred": safe_file(args.e2c_long_pred), "long_metrics": safe_file(args.e2c_long_metrics),
    }
    if output_dir.exists():
        unexpected = [p for p in output_dir.iterdir() if p.name not in {"logs", "raw_inference"}]
        if unexpected:
            raise SystemExit(f"refusing to overwrite report outputs: {unexpected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dev_rows = baseline.validate_dev_source(dev_source)
    prefix_rows = read_jsonl(prefix_input)
    full_input_rows = read_jsonl(full_input)
    long_source_rows = read_jsonl(long_source)
    if len(full_input_rows) != len(dev_rows):
        raise ValueError("unexpected V365 full-input row count")
    if len(long_source_rows) != int(prereg["validity"]["expected_v361_longdev_rows"]):
        raise ValueError("unexpected V361 long Dev row count")
    old_manifest_path = safe_file(e2a_root / "run_manifest.json")
    old_manifest = read_json(old_manifest_path)
    old_comparison_path = safe_file(e2a_root / "reports" / "model_comparison.json")
    old_comparison = read_json(old_comparison_path)
    e2b_manifest_path = safe_file(e2b_root / "run_manifest.json")
    e2b_manifest = read_json(e2b_manifest_path)
    e2b_comparison_path = safe_file(e2b_root / "reports" / "model_comparison.json")
    e2b_comparison = read_json(e2b_comparison_path)
    if e2b_manifest.get("schema_version") != EXPECTED_E2B_MANIFEST_SCHEMA:
        raise ValueError("unexpected canonical E2-B manifest schema")
    if e2b_comparison.get("manifest") != e2b_manifest:
        raise ValueError("canonical E2-B run manifest and comparison manifest differ")
    if int(((e2b_manifest.get("safety") or {}).get("test_or_restricted_files_opened", -1))) != 0:
        raise ValueError("canonical E2-B manifest does not attest zero Test/restricted access")
    for key, actual in (
        ("v365_dev", dev_source), ("v365_full_input", full_input),
        ("v365_prefix_input", prefix_input), ("v361_long_dev", long_source),
    ):
        expected = ((old_manifest.get("inputs") or {}).get(key) or {}).get("sha256")
        if expected != sha256_file(actual):
            raise ValueError(f"audited E2-A input hash differs for {key}")
    for model_id in ("m0_v361", "m1_e1b", "e2a"):
        prior = (old_manifest.get("models") or {}).get(model_id) or {}
        if Path(str(prior.get("checkpoint", ""))).resolve() != checkpoints[model_id].resolve():
            raise ValueError(f"audited checkpoint path differs for {model_id}")
        expected_checkpoint_sha = str(prior.get("checkpoint_sha256") or "")
        if not expected_checkpoint_sha:
            raise ValueError(f"audited checkpoint SHA missing for {model_id}")
        if sha256_file(checkpoints[model_id]) != expected_checkpoint_sha:
            raise ValueError(f"checkpoint bytes differ from audited SHA for {model_id}")
    e2b_prior = (e2b_manifest.get("models") or {}).get("e2b") or {}
    if Path(str(e2b_prior.get("checkpoint", ""))).resolve() != checkpoints["e2b"].resolve():
        raise ValueError("canonical E2-B checkpoint path differs")
    e2b_checkpoint_sha = str(e2b_prior.get("sha256") or e2b_prior.get("checkpoint_sha256") or "")
    if not e2b_checkpoint_sha or sha256_file(checkpoints["e2b"]) != e2b_checkpoint_sha:
        raise ValueError("canonical E2-B checkpoint bytes differ from audited SHA")

    def canonical_e2b_input_hash(suffix: str) -> str:
        matches = [
            str(value) for path, value in (e2b_manifest.get("inputs") or {}).items()
            if Path(str(path)).as_posix().endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"canonical E2-B manifest input is missing or ambiguous: {suffix}")
        return matches[0]

    for suffix, actual in (
        (EXPECTED_DEV_SUFFIX, dev_source),
        ("runs/v365_dev_baseline_compare_20261010/inputs/v365_dev_fulltext_label_blind.jsonl", full_input),
        ("runs/v365_dev_baseline_compare_20261010/inputs/v365_dev_cumulative_prefix_label_blind.jsonl", prefix_input),
        (EXPECTED_LONG_SUFFIX, long_source),
    ):
        if canonical_e2b_input_hash(suffix) != sha256_file(actual):
            raise ValueError(f"canonical E2-B input hash differs: {suffix}")

    full_records: dict[str, list[dict[str, Any]]] = {}
    prefix_records: dict[str, list[dict[str, Any]]] = {}
    calls_by_model: dict[str, list[dict[str, Any]]] = {}
    long_by_model: dict[str, list[dict[str, Any]]] = {}
    full_metrics: dict[str, dict[str, Any]] = {}
    temporal_metrics: dict[str, dict[str, Any]] = {}
    long_metrics: dict[str, dict[str, Any]] = {}
    endpoint: dict[str, dict[str, Any]] = {}
    consumed_paths: list[Path] = [
        dev_source, prefix_input, full_input, long_source, prereg_path,
        inference_code, baseline_path, old_manifest_path, old_comparison_path,
        e2b_manifest_path, e2b_comparison_path,
    ]

    def load_audited_scored(
        model_id: str, root: Path, audited_comparison: Mapping[str, Any],
    ) -> None:
        root = safe_dir(root)
        paths = {
            "full": safe_file(root / "v365_fulltext_predictions.jsonl"),
            "full_metric": safe_file(root / "v365_fulltext_metrics.json"),
            "prefix": safe_file(root / "v365_prefix_predictions.jsonl"),
            "prefix_metric": safe_file(root / "v365_temporal_metrics.json"),
            "long": safe_file(root / "v361_longdev_predictions.jsonl"),
            "long_metric": safe_file(root / "v361_longdev_metrics.json"),
        }
        consumed_paths.extend(paths.values())
        old_full = read_jsonl(paths["full"])
        old_prefix = read_jsonl(paths["prefix"])
        timing_full = timing_from_scored(paths["full_metric"], checkpoints[model_id])
        timing_prefix = timing_from_scored(paths["prefix_metric"], checkpoints[model_id])
        normalized_full = baseline.build_full_records(dev_rows, raw_from_scored(old_full), model_id)
        normalized_prefix, calls = baseline.build_prefix_records(
            dev_rows, prefix_rows, raw_from_scored(old_prefix), model_id
        )
        old_long = read_jsonl(paths["long"])
        timing_long = timing_from_scored(paths["long_metric"], checkpoints[model_id])
        normalized_long = build_long_records(long_source_rows, raw_from_scored(old_long), model_id, baseline)
        full_records[model_id], prefix_records[model_id] = normalized_full, normalized_prefix
        calls_by_model[model_id], long_by_model[model_id] = calls, normalized_long
        full_metrics[model_id] = baseline.binary_metrics(normalized_full, timing_full)
        temporal_metrics[model_id] = baseline.temporal_metrics(calls, timing_prefix)
        long_metrics[model_id] = generic_binary_metrics(normalized_long, timing_long)
        endpoint[model_id] = endpoint_stats(normalized_full, calls)
        validate_recomputed_against_audited(
            model_id, audited_comparison, full_metrics[model_id],
            temporal_metrics[model_id], long_metrics[model_id],
        )

    for model_id in ("m0_v361", "m1_e1b", "e2a"):
        load_audited_scored(model_id, e2a_root / "scored" / model_id, old_comparison)
    load_audited_scored("e2b", e2b_root / "scored" / "e2b", e2b_comparison)

    timing_e2c_full = baseline.raw_timing(e2c_files["full_metrics"], checkpoints["e2c"], full_input)
    timing_e2c_prefix = baseline.raw_timing(e2c_files["prefix_metrics"], checkpoints["e2c"], prefix_input)
    timing_e2c_long = baseline.raw_timing(e2c_files["long_metrics"], checkpoints["e2c"], long_source)
    raw_full = read_jsonl(e2c_files["full_pred"])
    raw_prefix = read_jsonl(e2c_files["prefix_pred"])
    raw_long = read_jsonl(e2c_files["long_pred"])
    consumed_paths.extend(e2c_files.values())
    e2c_full = baseline.build_full_records(dev_rows, raw_full, "e2c")
    e2c_prefix, e2c_calls = baseline.build_prefix_records(dev_rows, prefix_rows, raw_prefix, "e2c")
    e2c_long = build_long_records(long_source_rows, raw_long, "e2c", baseline)
    full_records["e2c"], prefix_records["e2c"] = e2c_full, e2c_prefix
    calls_by_model["e2c"], long_by_model["e2c"] = e2c_calls, e2c_long
    full_metrics["e2c"] = baseline.binary_metrics(e2c_full, timing_e2c_full)
    temporal_metrics["e2c"] = baseline.temporal_metrics(e2c_calls, timing_e2c_prefix)
    long_metrics["e2c"] = generic_binary_metrics(e2c_long, timing_e2c_long)
    endpoint["e2c"] = endpoint_stats(e2c_full, e2c_calls)

    reference_chunks = {row["sample_id"]: (row["raw_wordpiece_length"], row["n_chunks"]) for row in long_by_model["m0_v361"]}
    for model_id in MODEL_ORDER[1:]:
        observed = {row["sample_id"]: (row["raw_wordpiece_length"], row["n_chunks"]) for row in long_by_model[model_id]}
        if observed != reference_chunks:
            raise ValueError(f"{model_id}: long Dev chunking differs from M0")

    transitions = {
        baseline_id: build_transitions(
            dev_rows, full_records[baseline_id], e2c_full,
            calls_by_model[baseline_id], e2c_calls, baseline_id,
        )
        for baseline_id in ("m0_v361", "m1_e1b", "e2a", "e2b")
    }
    e2c_errors = build_e2c_errors(dev_rows, e2c_full, e2c_calls)
    keyword_diagnostics = build_keyword_diagnostics(dev_rows, full_records)
    maps = {
        model_id: endpoint_maps(full_records[model_id], calls_by_model[model_id], long_by_model[model_id])
        for model_id in MODEL_ORDER
    }
    paired: list[dict[str, Any]] = []
    for baseline_id in ("m0_v361", "m1_e1b", "e2a", "e2b"):
        for endpoint_name in sorted(maps["e2c"]):
            paired.append(paired_result(
                endpoint_name, baseline_id, maps["e2c"][endpoint_name],
                maps[baseline_id][endpoint_name], baseline,
            ))

    gate = evaluate_gate(
        prereg, dev_rows, prefix_rows, long_source_rows, full_metrics,
        temporal_metrics, long_metrics,
        {**full_records, "e2c_long": e2c_long, "m1_long": long_by_model["m1_e1b"]},
        endpoint, [timing_e2c_full, timing_e2c_prefix, timing_e2c_long],
    )
    reports = output_dir / "reports"
    scored = output_dir / "scored" / "e2c"
    write_jsonl(scored / "v365_fulltext_predictions.jsonl", e2c_full)
    write_jsonl(scored / "v365_prefix_predictions.jsonl", e2c_prefix)
    write_jsonl(scored / "v365_temporal_calls.jsonl", e2c_calls)
    write_json(scored / "v365_fulltext_metrics.json", full_metrics["e2c"])
    write_json(scored / "v365_temporal_metrics.json", temporal_metrics["e2c"])
    write_jsonl(scored / "v361_longdev_predictions.jsonl", e2c_long)
    write_json(scored / "v361_longdev_metrics.json", long_metrics["e2c"])
    for baseline_id, rows in transitions.items():
        write_jsonl(reports / f"e2c_vs_{baseline_id}_transitions.jsonl", rows)
        write_csv(reports / f"e2c_vs_{baseline_id}_transitions.csv", list(rows))
    write_jsonl(reports / "e2c_error_audit_future_filtered.jsonl", e2c_errors)
    write_json(reports / "paired_endpoint_inference.json", paired)
    write_csv(reports / "paired_endpoint_inference.csv", [
        {
            "baseline_model": row["baseline_model"], "endpoint": row["endpoint"],
            "paired_n": row["paired_n"],
            "e2c_success": row["candidate_success"]["numerator"],
            "e2c_wilson_low": row["candidate_success"]["wilson_95"][0],
            "e2c_wilson_high": row["candidate_success"]["wilson_95"][1],
            "baseline_success": row["baseline_success"]["numerator"],
            "risk_difference_percentage_points": row["risk_difference_percentage_points"],
            "baseline_wilson_low": row["baseline_success"]["wilson_95"][0],
            "baseline_wilson_high": row["baseline_success"]["wilson_95"][1],
            "e2c_only": row["candidate_only_success_n"],
            "baseline_only": row["baseline_only_success_n"],
            "exact_p": row["exact_binomial_mcnemar_two_sided_p"],
        }
        for row in paired
    ])

    old_checkpoint_sha = {
        model_id: old_manifest["models"][model_id]["checkpoint_sha256"]
        for model_id in ("m0_v361", "m1_e1b", "e2a")
    }
    manifest = {
        "schema_version": "v365.e2c.dev_compare.v1",
        "threshold": THRESHOLD, "primary_policy": "k2", "k1_role": "diagnostic_only",
        "preregistration": {"path": str(prereg_path), "sha256": sha256_file(prereg_path)},
        "models": {
            **{
                model_id: {
                    "checkpoint": str(checkpoints[model_id]),
                    "sha256": old_checkpoint_sha[model_id],
                    "sha_source": "audited E2-A run_manifest.json",
                }
                for model_id in ("m0_v361", "m1_e1b", "e2a")
            },
            "e2b": {
                "checkpoint": str(checkpoints["e2b"]),
                "sha256": e2b_checkpoint_sha,
                "sha_source": "canonical E2-B run_manifest.json",
            },
            "e2c": {
                "checkpoint": str(checkpoints["e2c"]),
                "sha256": sha256_file(checkpoints["e2c"]),
                "sha_source": "computed by this report",
            },
        },
        "inputs": {
            str(path): sha256_file(path) for path in dict.fromkeys(consumed_paths)
        },
        "code": {
            "report": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "inference": {"path": str(inference_code), "sha256": sha256_file(inference_code)},
        },
        "safety": {
            "training_performed": False, "threshold_or_policy_tuned": False,
            "test_or_restricted_files_opened": 0, "path_guard_before_every_open": True,
            "test_remains_locked_after_pass": True,
        },
    }
    comparison = {
        "manifest": manifest, "gate": gate, "v365_fulltext": full_metrics,
        "v365_temporal": temporal_metrics, "v361_longdev": long_metrics,
        "endpoint_consistency": endpoint, "paired_endpoint_inference": paired,
        "e2c_error_audit_count": len(e2c_errors),
        "keyword_diagnostics": keyword_diagnostics,
        "transition_counts": {
            baseline_id: {
                field: dict(Counter(str(row[field]) for row in rows if field in row))
                for field in (
                    "full_transition", "k1_effective_transition", "k1_premature_transition",
                    "k1_normal_fp_transition", "k1_clean_transition",
                    "k2_effective_transition", "k2_premature_transition",
                    "k2_normal_fp_transition", "k2_clean_transition",
                )
            }
            for baseline_id, rows in transitions.items()
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    write_json(reports / "gate_decision.json", gate)
    write_json(reports / "model_comparison.json", comparison)
    report = build_report(
        full_metrics, temporal_metrics, long_metrics, gate, paired, transitions,
        manifest, e2c_errors, keyword_diagnostics,
    )
    (reports / "V365_E2C_DEV_COMPARISON_REPORT.md").write_text(report + "\n", encoding="utf-8")
    weights = [p for p in output_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".bin", ".safetensors"}]
    if weights:
        raise AssertionError(f"report output unexpectedly contains weights: {weights}")
    print(json.dumps({
        "status": gate["status"],
        "gate": str(reports / "gate_decision.json"),
        "report": str(reports / "V365_E2C_DEV_COMPARISON_REPORT.md"),
        "test_accessed": False,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    output_dir = safe_output(args.output_dir)
    try:
        return run(args)
    except (ValueError, KeyError, TypeError, AssertionError) as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        invalid = {
            "schema_version": "v365.e2c.gate.v1", "status": "INVALID",
            "reason": f"{type(exc).__name__}: {exc}",
            "test_policy": "LOCKED; invalid evaluation never authorizes Test",
            "test_or_restricted_files_opened": 0,
        }
        write_json(output_dir / "reports" / "gate_decision.json", invalid)
        print(json.dumps(invalid, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
