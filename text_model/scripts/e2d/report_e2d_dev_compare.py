#!/usr/bin/env python3
"""Dev-only report and frozen promotion gate for E2-D.

E2-D is a causal veto layered on the already frozen M1 ``k2`` policy.  It may
remove an M1 k2 alert, but it can never create one.  M1 probabilities, the k1
diagnostic policy, full-text predictions, and V361 long-Dev predictions are
immutable.  Every input path is checked before opening and any path component
containing ``test`` or ``restricted`` is rejected.

This program performs no training, threshold selection, or model selection.
Even ``PASS_FOR_FREEZE`` leaves V365 Test locked.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


THRESHOLD = 0.5
MODEL_ORDER = ("m0_v361", "m1_e1b", "e2a", "e2b", "e2c", "e2d")
MODEL_LABELS = {
    "m0_v361": "M0 V361",
    "m1_e1b": "M1 E1-B",
    "e2a": "E2-A",
    "e2b": "E2-B",
    "e2c": "E2-C",
    "e2d": "E2-D gate",
}
EXPECTED_DEV_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/v365_calls_dev_final.jsonl"
)
EXPECTED_PREFIX_SUFFIX = (
    "runs/v365_dev_baseline_compare_20261010/inputs/"
    "v365_dev_cumulative_prefix_label_blind.jsonl"
)
EXPECTED_LONG_SUFFIX = (
    "data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl"
)
EXPECTED_E2A_SUFFIX = "runs/v365_e2a_dev_compare_20261011"
EXPECTED_E2B_SUFFIX = "runs/v365_e2b_dev_compare_20261012"
EXPECTED_E2C_SUFFIX = "runs/v365_e2c_dev_compare_20261013"
EXPECTED_GATE_SCHEMA = "v365.e2d.independent_gate.v1"
EXPECTED_PREREG_SCHEMA = "v365.e2d.preregistration.v1"

EXPECTED_NI_K2 = {
    "early_te_after_min": 14,
    "all_te_after_min": 79,
    "normal_ever_fp_max": 11,
    "all_premature_max": 57,
    "clean_effective_min": 22,
    "explicit_before_ta_min": 63,
    "all_delay_median_max": 0,
    "all_delay_p75_max": 1,
}
EXPECTED_IMPROVE_K2 = {
    "normal_ever_fp_max": 10,
    "all_premature_max": 56,
    "normal_fp_plus_premature_max": 65,
    "clean_effective_min": 23,
}
EXPECTED_IMMUTABLE_FULL = {
    "early_hits": 13,
    "explicit_hits": 65,
    "fraud_hits": 78,
    "normal_fp": 6,
}
EXPECTED_IMMUTABLE_LONG = {"fraud_hits": 120, "normal_fp": 2}
EXPECTED_PREREG_FULL_NI = {
    "early_hits_min": 13,
    "explicit_hits_min": 65,
    "fraud_hits_min": 78,
    "normal_fp_max": 6,
}
EXPECTED_PREREG_LONG_NI = {
    "fraud_hits_min": 120,
    "normal_fp_max": 2,
    "new_false_negative_max": 0,
}


def forbidden_path(path: Path) -> bool:
    for candidate in (path.absolute(), path.resolve(strict=False)):
        if any(
            "test" in part.lower() or "restricted" in part.lower()
            for part in candidate.parts
        ):
            return True
    return False


def safe_file(path: Path, suffix: str | None = None) -> Path:
    if forbidden_path(path):
        raise ValueError(f"refusing Test/restricted path: {path}")
    resolved = path.resolve()
    if suffix is not None and not resolved.as_posix().endswith(suffix):
        raise ValueError(f"refusing non-allowlisted path: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def safe_dir(path: Path, suffix: str | None = None) -> Path:
    if forbidden_path(path):
        raise ValueError(f"refusing Test/restricted directory: {path}")
    resolved = path.resolve()
    if suffix is not None and not resolved.as_posix().endswith(suffix):
        raise ValueError(f"refusing non-allowlisted directory: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def safe_output(path: Path) -> Path:
    if forbidden_path(path):
        raise ValueError(f"refusing Test/restricted output: {path}")
    return path.resolve()


def sha256_file(path: Path) -> str:
    path = safe_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    path = safe_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = safe_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    if forbidden_path(path):
        raise ValueError(f"refusing Test/restricted output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if forbidden_path(path):
        raise ValueError(f"refusing Test/restricted output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if forbidden_path(path):
        raise ValueError(f"refusing Test/restricted output: {path}")
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
        for row in rows:
            writer.writerow(dict(row))


def copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    source = safe_file(source)
    if forbidden_path(destination):
        raise ValueError(f"refusing Test/restricted output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    source_hash = sha256_file(source)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise ValueError(f"byte-copy hash mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "copy": str(destination.resolve()),
        "source_sha256": source_hash,
        "copy_sha256": destination_hash,
        "byte_identical": True,
    }


def load_module(path: Path, name: str) -> ModuleType:
    path = safe_file(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": numerator / denominator if denominator else None,
    }


def quantile(values: Sequence[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def raw_from_scored(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        converted.append(
            {
                "sample_id": row["sample_id"],
                "p_binary": [float(row["p_normal"]), float(row["p_fraud"])],
                "raw_wordpiece_length": int(row["raw_wordpiece_length"]),
                "num_chunks": int(row["n_chunks"]),
                "max_chunks_exceeded": bool(row.get("max_chunks_exceeded", False)),
                "full_text_coverage": float(row["full_text_coverage"]),
            }
        )
    return converted


def validate_probability_rows(
    rows: Sequence[Mapping[str, Any]], expected_count: int, id_field: str
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(rows)}")
    identifiers = [str(row[id_field]) for row in rows]
    if len(set(identifiers)) != expected_count:
        raise ValueError(f"duplicate {id_field} values")
    for row in rows:
        p0, p1 = float(row["p_normal"]), float(row["p_fraud"])
        if not (math.isfinite(p0) and math.isfinite(p1)):
            raise ValueError(f"non-finite probability: {row[id_field]}")
        if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0):
            raise ValueError(f"out-of-range probability: {row[id_field]}")
        if abs(p0 + p1 - 1.0) > 1e-4:
            raise ValueError(f"probabilities do not sum to one: {row[id_field]}")
        threshold_prediction = int(p1 >= THRESHOLD)
        if "prediction" in row and int(row["prediction"]) != threshold_prediction:
            raise ValueError(f"prediction/probability mismatch: {row[id_field]}")
        if "raw_hit_ge_0_5" in row and bool(row["raw_hit_ge_0_5"]) != bool(threshold_prediction):
            raise ValueError(f"raw-hit/probability mismatch: {row[id_field]}")


def identical_prediction_values(
    reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]], id_field: str
) -> bool:
    ref = {str(row[id_field]): row for row in reference}
    new = {str(row[id_field]): row for row in candidate}
    if set(ref) != set(new):
        return False
    fields = ("p_normal", "p_fraud", "prediction", "y_true")
    return all(
        all(ref[key].get(field) == new[key].get(field) for field in fields)
        for key in ref
    )


def veto_alerts(
    m1_alerts: Sequence[bool], gate_probabilities: Sequence[float | None],
    threshold: float = THRESHOLD,
) -> list[bool]:
    """Apply a veto without ever creating an alert (pure, unit-testable core)."""
    if len(m1_alerts) != len(gate_probabilities):
        raise ValueError("alert/gate lengths differ")
    output: list[bool] = []
    for original, probability in zip(m1_alerts, gate_probabilities):
        if probability is not None and not 0.0 <= float(probability) <= 1.0:
            raise ValueError("gate probability is outside [0, 1]")
        output.append(bool(original and probability is not None and probability >= threshold))
    if any(new and not old for old, new in zip(m1_alerts, output)):
        raise AssertionError("veto created an alert")
    return output


def policy_from_alerts(source: Mapping[str, Any], alerts: Sequence[bool]) -> dict[str, Any]:
    alarm_turns = [index + 1 for index, value in enumerate(alerts) if value]
    t_e_raw, t_a_raw = source.get("tE_turn"), source.get("tA_turn")
    t_e = int(t_e_raw) if t_e_raw is not None else None
    t_a = int(t_a_raw) if t_a_raw is not None else None
    first_any = min(alarm_turns) if alarm_turns else None
    first_valid = (
        min((turn for turn in alarm_turns if turn >= t_e), default=None)
        if t_e is not None
        else None
    )
    premature = bool(t_e is not None and any(turn < t_e for turn in alarm_turns))
    effective = first_valid is not None
    return {
        "alarm_turns": alarm_turns,
        "first_any_turn": first_any,
        "first_any_offset_from_tE": (
            first_any - t_e if first_any is not None and t_e is not None else None
        ),
        "premature": premature,
        "effective": effective,
        "clean_effective": bool(effective and not premature),
        "first_valid_turn": first_valid,
        "valid_delay": (
            first_valid - t_e if first_valid is not None and t_e is not None else None
        ),
        "before_tA": bool(first_valid is not None and t_a is not None and first_valid < t_a),
        "advance_before_tA": (
            t_a - first_valid
            if first_valid is not None and t_a is not None and first_valid < t_a
            else None
        ),
        "tA_minus_first_valid": (
            t_a - first_valid
            if first_valid is not None and t_a is not None
            else None
        ),
        "normal_ever_false_alarm": bool(
            str(source["decision"]) == "normal" and alarm_turns
        ),
    }


def build_gated_temporal(
    dev_rows: Sequence[Mapping[str, Any]],
    m1_prefix: Sequence[Mapping[str, Any]],
    m1_calls: Sequence[Mapping[str, Any]],
    common: ModuleType,
    model: Any,
    mean: Any,
    scale: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_by_call = {str(row["call_id"]): row for row in dev_rows}
    call_by_id = {str(row["call_id"]): row for row in m1_calls}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for original in m1_prefix:
        row = copy.deepcopy(dict(original))
        row["model_id"] = "e2d"
        row["m1_alert_k2"] = bool(original["alert_k2"])
        row["gate_probability"] = None
        row["gate_pass_ge_0_5"] = False
        row["gate_feature_values"] = None
        grouped[str(row["call_id"])].append(row)

    feature_rows: list[list[float]] = []
    feature_targets: list[dict[str, Any]] = []
    for call_id, sequence in grouped.items():
        sequence.sort(key=lambda row: int(row["prefix_turn"]))
        expected = list(range(1, int(source_by_call[call_id]["n_full_turns"]) + 1))
        if [int(row["prefix_turn"]) for row in sequence] != expected:
            raise ValueError(f"{call_id}: incomplete prefix sequence")
        for previous, current in zip(sequence, sequence[1:]):
            values = [float(value) for value in common.pair_features(previous, current)]
            if len(values) != len(common.FEATURE_NAMES) or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError(f"{call_id}: invalid gate feature vector")
            feature_rows.append(values)
            feature_targets.append(current)

    started = time.perf_counter()
    probabilities = [float(value) for value in common.predict_gate(
        model, mean, scale, feature_rows
    )]
    gate_seconds = time.perf_counter() - started
    if len(probabilities) != len(feature_targets):
        raise ValueError("gate prediction count mismatch")
    for target, probability, features in zip(feature_targets, probabilities, feature_rows):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid gate probability")
        target["gate_probability"] = probability
        target["gate_pass_ge_0_5"] = bool(probability >= THRESHOLD)
        target["gate_feature_values"] = {
            name: value for name, value in zip(common.FEATURE_NAMES, features)
        }

    output_prefix: list[dict[str, Any]] = []
    output_calls: list[dict[str, Any]] = []
    subset_violations: list[dict[str, Any]] = []
    suppressed = 0
    for source in dev_rows:
        call_id = str(source["call_id"])
        sequence = grouped[call_id]
        original_alerts = [bool(row["m1_alert_k2"]) for row in sequence]
        gate_probs = [row["gate_probability"] for row in sequence]
        gated_alerts = veto_alerts(original_alerts, gate_probs)
        for row, gated in zip(sequence, gated_alerts):
            row["alert_k2"] = bool(gated)
            row["e2d_alert_k2"] = bool(gated)
            row["alert_k1"] = bool(row["alert_k1"])
            suppressed += int(row["m1_alert_k2"] and not gated)
            if gated and not row["m1_alert_k2"]:
                subset_violations.append(
                    {"call_id": call_id, "prefix_turn": int(row["prefix_turn"])}
                )
        base_call = copy.deepcopy(dict(call_by_id[call_id]))
        base_call["model_id"] = "e2d"
        base_call["policies"]["k2"] = policy_from_alerts(source, gated_alerts)
        output_calls.append(base_call)
        output_prefix.extend(sequence)

    output_prefix.sort(key=lambda row: (str(row["call_id"]), int(row["prefix_turn"])))
    diagnostics = {
        "gate_threshold": THRESHOLD,
        "evaluated_adjacent_pairs": len(feature_rows),
        "expected_adjacent_pairs": len(m1_prefix) - len(dev_rows),
        "m1_k2_alert_windows": sum(bool(row["m1_alert_k2"]) for row in output_prefix),
        "e2d_k2_alert_windows": sum(bool(row["alert_k2"]) for row in output_prefix),
        "suppressed_m1_k2_windows": suppressed,
        "alert_subset_violation_count": len(subset_violations),
        "alert_subset_violations": subset_violations,
        "k1_mismatch_count": sum(
            bool(new["alert_k1"]) != bool(old["alert_k1"])
            for new, old in zip(output_prefix, sorted(
                m1_prefix,
                key=lambda row: (str(row["call_id"]), int(row["prefix_turn"])),
            ))
        ),
        "probability_mismatch_count": sum(
            float(new["p_fraud"]) != float(old["p_fraud"])
            or float(new["p_normal"]) != float(old["p_normal"])
            for new, old in zip(output_prefix, sorted(
                m1_prefix,
                key=lambda row: (str(row["call_id"]), int(row["prefix_turn"])),
            ))
        ),
        "gate_cpu_seconds": gate_seconds,
    }
    return output_prefix, output_calls, diagnostics


def evidence_turn(source: Mapping[str, Any], turn: int, role: str) -> dict[str, Any]:
    turns = source["turns"]
    if not 1 <= turn <= len(turns):
        raise ValueError(f"{source['call_id']}: evidence turn out of range")
    item = turns[turn - 1]
    return {
        "turn": turn,
        "role": role,
        "speaker": item.get("speaker"),
        "quote": item.get("text"),
    }


def build_errors(
    dev_rows: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    call_by_id = {str(row["call_id"]): row for row in calls}
    errors: list[dict[str, Any]] = []
    for source in dev_rows:
        call_id = str(source["call_id"])
        decision = str(source["decision"])
        policy = call_by_id[call_id]["policies"]["k2"]
        error_types: list[str] = []
        if decision in {"early_fraud", "explicit_fraud"}:
            if not policy["effective"]:
                error_types.append("missed_after_tE")
            if policy["premature"]:
                error_types.append("premature_alarm")
        elif decision == "normal" and policy["normal_ever_false_alarm"]:
            error_types.append("normal_ever_false_alarm")
        if not error_types:
            continue
        first_alarm = policy["first_any_turn"]
        trigger_window: list[dict[str, Any]] = []
        if first_alarm is not None:
            for turn in range(max(1, int(first_alarm) - 1), int(first_alarm) + 1):
                trigger_window.append(evidence_turn(source, turn, "current_trigger_window"))
        t_e = source.get("tE_turn")
        anchor = (
            evidence_turn(source, int(t_e), "tE_anchor") if t_e is not None else None
        )
        evidence_turns = [item["turn"] for item in trigger_window]
        if anchor is not None:
            evidence_turns.append(int(anchor["turn"]))
        if anchor is not None and int(anchor["turn"]) > int(t_e):
            raise AssertionError("future evidence leaked past tE anchor")
        errors.append(
            {
                "call_id": call_id,
                "decision": decision,
                "scenario_tag": source.get("scenario_tag"),
                "n_full_turns": int(source["n_full_turns"]),
                "tE_turn": t_e,
                "tA_turn": source.get("tA_turn"),
                "error_types": error_types,
                "first_alarm_turn": first_alarm,
                "first_valid_turn": policy["first_valid_turn"],
                "trigger_window": trigger_window,
                "tE_anchor": anchor,
                "max_evidence_turn": max(evidence_turns) if evidence_turns else None,
                "evidence_policy": (
                    "Only the current two-prefix trigger window and the adjudicated "
                    "tE anchor are quoted; no later adjudicated evidence is copied."
                ),
            }
        )
    return errors


def build_transitions(
    m1_calls: Sequence[Mapping[str, Any]], e2d_calls: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    old = {str(row["call_id"]): row for row in m1_calls}
    new = {str(row["call_id"]): row for row in e2d_calls}
    if set(old) != set(new):
        raise ValueError("M1/E2-D call identities differ")
    rows: list[dict[str, Any]] = []
    for call_id in sorted(old):
        old_policy, new_policy = old[call_id]["policies"]["k2"], new[call_id]["policies"]["k2"]
        row = {
            "call_id": call_id,
            "decision": old[call_id]["decision"],
            "scenario_tag": old[call_id].get("scenario_tag"),
        }
        for field in ("effective", "premature", "clean_effective", "normal_ever_false_alarm"):
            before, after = bool(old_policy[field]), bool(new_policy[field])
            row[f"m1_{field}"] = before
            row[f"e2d_{field}"] = after
            row[f"{field}_transition"] = f"{int(before)}->{int(after)}"
        rows.append(row)
    return rows


def manifest_hash_for_suffix(manifest: Mapping[str, Any], suffix: str) -> str | None:
    matches: list[str] = []
    for key, value in (manifest.get("inputs") or {}).items():
        if str(key).replace("\\", "/").endswith(suffix):
            if isinstance(value, Mapping):
                value = value.get("sha256")
            if value:
                matches.append(str(value))
    if not matches:
        return None
    if len(set(matches)) != 1:
        raise ValueError(f"manifest has ambiguous hashes for suffix: {suffix}")
    return matches[0]


def require_manifest_input_hash(
    manifest: Mapping[str, Any], logical_name: str, actual: Path
) -> None:
    block = (manifest.get("inputs") or {}).get(logical_name)
    if not isinstance(block, Mapping) or not block.get("sha256"):
        raise ValueError(f"canonical manifest lacks input hash: {logical_name}")
    if str(block["sha256"]) != sha256_file(actual):
        raise ValueError(f"canonical input bytes differ: {logical_name}")


def normalize_historical_m1_metrics(
    comparison: Mapping[str, Any], endpoint: Mapping[str, Any],
    long_rows: Sequence[Mapping[str, Any]],
    canonical_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconcile only independently verified E2-A/E2-C reporting-schema deltas.

    This does not relax metric equality.  Auxiliary diagnostics must reproduce
    the frozen predictions, and all remaining keys/values must match exactly.
    Neither historical on-disk report is modified.
    """
    normalized = copy.deepcopy(comparison)
    full = normalized["v365_fulltext"]["m1_e1b"]
    temporal = normalized["v365_temporal"]["m1_e1b"]
    long_metric = normalized["v361_longdev"]["m1_e1b"]
    full["diagnostics"].setdefault("full_vs_final_prefix", copy.deepcopy(endpoint))
    temporal.setdefault("endpoint_consistency", copy.deepcopy(endpoint))

    expected_long_diagnostics = {
        "coverage_min": min(float(row["full_text_coverage"]) for row in long_rows),
        "max_chunks_observed": max(int(row["n_chunks"]) for row in long_rows),
        "calls_exceeding_diagnostic_12_chunks": sum(
            int(row["n_chunks"]) > 12 for row in long_rows
        ),
        "diagnostic_max_chunks": 12,
        "hard_chunk_cap": None,
        "max_chunks_semantics": "diagnostic only; all chunks retained for full coverage",
    }
    if "diagnostics" in long_metric:
        if long_metric["diagnostics"] != expected_long_diagnostics:
            raise ValueError("historical M1 long diagnostics differ from frozen predictions")
        del long_metric["diagnostics"]
    count = len(long_rows)
    wall = float(long_metric["timing"]["wall_time_seconds"])
    expected_timing_extras = {
        "evaluated_calls": count,
        "seconds_per_call": wall / count,
        "milliseconds_per_call": 1000.0 * wall / count,
    }
    for key, expected in expected_timing_extras.items():
        if key in long_metric["timing"]:
            if long_metric["timing"][key] != expected:
                raise ValueError(f"historical M1 long timing derivation differs: {key}")
            del long_metric["timing"][key]
    long_metric.setdefault("threshold", THRESHOLD)
    long_metric.setdefault("probability_field", "p_binary[1]")
    for section, canonical in canonical_metrics.items():
        if normalized[section]["m1_e1b"] != canonical:
            raise ValueError(f"historical comparison changed canonical M1 metrics: {section}")
    return normalized, {
        "policy": "known auxiliary schema deltas only; exact metric equality retained",
        "endpoint_recomputed_from_frozen_predictions": dict(endpoint),
        "long_diagnostics_recomputed_from_frozen_predictions": expected_long_diagnostics,
        "long_timing_derivations_verified": expected_timing_extras,
        "canonical_m1_exact_after_normalization": True,
        "historical_files_modified": False,
    }


def validate_gate_contract(
    payload: Mapping[str, Any], training: Mapping[str, Any], preregistration: Path,
    prereg: Mapping[str, Any], gate_checkpoint: Path, common: ModuleType,
    mean: Any, scale: Any,
) -> dict[str, Any]:
    prereg_hash = sha256_file(preregistration)
    gate_hash = sha256_file(gate_checkpoint)
    if payload.get("schema_version") != EXPECTED_GATE_SCHEMA:
        raise ValueError("unexpected gate checkpoint schema")
    required = {
        "training_data_scope": "v365_train_only",
        "scaler_fit_scope": "train_only",
        "dev_rows_used_for_training": 0,
        "test_or_restricted_files_opened": 0,
        "training_used_dev": False,
        "training_used_test": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected or training.get(key) != expected:
            raise ValueError(f"gate isolation contract mismatch: {key}")
    if training.get("status") != "complete":
        raise ValueError("gate training manifest is not complete")
    if training.get("selection_policy") != "fixed final epoch; no Dev selection":
        raise ValueError("gate selection policy was not frozen")
    if training.get("m1_parameters_loaded_in_trainer") != 0:
        raise ValueError("gate trainer unexpectedly loaded M1 parameters")
    if training.get("fulltext_model_updated") is not False:
        raise ValueError("gate trainer reports full-text model modification")
    if payload.get("preregistration_sha256") != prereg_hash:
        raise ValueError("checkpoint/preregistration hash mismatch")
    if training.get("preregistration_sha256") != prereg_hash:
        raise ValueError("training-manifest/preregistration hash mismatch")
    if training.get("checkpoint_sha256") != gate_hash:
        raise ValueError("training-manifest/checkpoint hash mismatch")
    for key in (
        "feature_names", "m1_sha256", "config", "train_pair_sha256",
        "train_prediction_sha256", "train_pair_rows", "train_call_ids_sha256",
    ):
        if training.get(key) != payload.get(key):
            raise ValueError(f"checkpoint/training-manifest mismatch: {key}")
    if payload.get("feature_names") != list(common.FEATURE_NAMES):
        raise ValueError("feature contract differs from common module")
    if payload.get("m1_sha256") != common.M1_SHA256:
        raise ValueError("M1 contract differs from common module")
    if payload.get("m1_sha256") != prereg.get("m1_sha256"):
        raise ValueError("M1 contract differs from preregistration")
    if payload.get("config") != prereg.get("gate_training"):
        raise ValueError("gate training config differs from preregistration")
    if payload.get("feature_names") != prereg.get("feature_names"):
        raise ValueError("feature order differs from preregistration")
    pair_data = prereg.get("pair_data") or {}
    if int(payload.get("train_pair_rows", -1)) != int(pair_data.get("pairs", -2)):
        raise ValueError("gate pair count differs from preregistration")
    if int(training.get("unique_calls", -1)) != int(pair_data.get("unique_train_calls", -2)):
        raise ValueError("gate Train-call count differs from preregistration")
    if training.get("group_counts") != pair_data.get("pair_counts"):
        raise ValueError("gate group counts differ from preregistration")
    expected_parameters = 10 * 16 + 16 + 16 + 1
    if int(training.get("parameters", -1)) != expected_parameters:
        raise ValueError("unexpected gate parameter count")
    mean_values = list(mean.tolist() if hasattr(mean, "tolist") else mean)
    scale_values = list(scale.tolist() if hasattr(scale, "tolist") else scale)
    if len(mean_values) != len(common.FEATURE_NAMES) or len(scale_values) != len(common.FEATURE_NAMES):
        raise ValueError("scaler shape differs from feature contract")
    if not all(math.isfinite(float(value)) for value in mean_values):
        raise ValueError("non-finite scaler mean")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in scale_values):
        raise ValueError("invalid scaler scale")
    return {
        "gate_checkpoint_sha256": gate_hash,
        "preregistration_sha256": prereg_hash,
        "training_manifest_isolation_verified": True,
        "scaler_fit_scope": "train_only",
        "dev_rows_used_for_training": 0,
        "test_or_restricted_files_opened": 0,
    }


def validate_preregistration(prereg: Mapping[str, Any]) -> None:
    if prereg.get("schema_version") != EXPECTED_PREREG_SCHEMA:
        raise ValueError("unexpected E2-D preregistration schema")
    if float(prereg.get("gate_threshold", THRESHOLD)) != THRESHOLD:
        raise ValueError("preregistered gate threshold is not 0.5")
    gates = prereg.get("gates") or {}
    noninferiority = gates.get("noninferiority") or {}
    noninf = noninferiority.get("temporal_k2") or {}
    improve = ((gates.get("improvement") or {}).get("temporal_k2") or {})
    if (noninferiority.get("fulltext") or {}) != EXPECTED_PREREG_FULL_NI:
        raise ValueError("full-text noninferiority gates differ from frozen contract")
    if noninf != EXPECTED_NI_K2:
        raise ValueError(f"noninferiority gates differ from frozen contract: {noninf}")
    if (noninferiority.get("longdev") or {}) != EXPECTED_PREREG_LONG_NI:
        raise ValueError("long-Dev noninferiority gates differ from frozen contract")
    if improve != EXPECTED_IMPROVE_K2:
        raise ValueError(f"improvement gates differ from frozen contract: {improve}")


def check(
    check_id: str, group: str, observed: Any, operator: str, limit: Any, evidence: str
) -> dict[str, Any]:
    if operator == "==":
        passed = observed == limit
    elif operator == "<=":
        passed = observed is not None and observed <= limit
    elif operator == ">=":
        passed = observed is not None and observed >= limit
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "id": check_id,
        "group": group,
        "observed": observed,
        "operator": operator,
        "limit": limit,
        "passed": bool(passed),
        "evidence": evidence,
    }


def evaluate_gate(
    dev_rows: Sequence[Mapping[str, Any]], prefix_rows: Sequence[Mapping[str, Any]],
    long_rows: Sequence[Mapping[str, Any]], full_metric: Mapping[str, Any],
    temporal_metric: Mapping[str, Any], long_metric: Mapping[str, Any],
    diagnostics: Mapping[str, Any], identity: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    full = full_metric["subgroups"]
    k2 = temporal_metric["policies"]["k2"]
    early = k2["subsets"]["early_fraud"]["tE_after_effective_recall"]
    all_effective = k2["subsets"]["all_fraud"]["tE_after_effective_recall"]
    premature = k2["subsets"]["all_fraud"]["tE_before_premature_alarm_rate"]
    clean = k2["subsets"]["all_fraud"]["clean_effective_rate"]
    normal_fp = k2["normal_ever_false_alarm"]
    before_ta = k2["explicit_before_tA"]
    delay = k2["subsets"]["all_fraud"]["delay"]
    checks: list[dict[str, Any]] = []
    add = checks.append
    add(check("valid.dev_calls", "validity", len(dev_rows), "==", 200, "V365 Dev source"))
    add(check("valid.prefix_rows", "validity", len(prefix_rows), "==", 4492, "causal Dev prefixes"))
    add(check("valid.long_rows", "validity", len(long_rows), "==", 240, "V361 long Dev source"))
    add(check("valid.full_probabilities_immutable", "validity", identity["full_values_identical"], "==", True, "E2-D aliases frozen M1 full probabilities"))
    add(check("valid.long_probabilities_immutable", "validity", identity["long_values_identical"], "==", True, "E2-D aliases frozen M1 long probabilities"))
    add(check("valid.full_byte_copy", "validity", identity["full_byte_copy"]["byte_identical"], "==", True, "unchanged M1 full artifact proof"))
    add(check("valid.long_byte_copy", "validity", identity["long_byte_copy"]["byte_identical"], "==", True, "unchanged M1 long artifact proof"))
    add(check("valid.alert_subset", "validity", diagnostics["alert_subset_violation_count"], "==", 0, "E2-D alerts must be a subset of M1 k2"))
    add(check("valid.k1_unchanged", "validity", diagnostics["k1_mismatch_count"], "==", 0, "M1 k1 remains diagnostic and unchanged"))
    add(check("valid.prefix_probabilities_unchanged", "validity", diagnostics["probability_mismatch_count"], "==", 0, "gate cannot alter M1 probabilities"))
    add(check("valid.gate_pair_coverage", "validity", diagnostics["evaluated_adjacent_pairs"], "==", diagnostics["expected_adjacent_pairs"], "every adjacent Dev prefix pair"))
    add(check("valid.train_isolation", "validity", contract["training_manifest_isolation_verified"], "==", True, "checkpoint and scaler fitted on Train only"))
    add(check("valid.test_reads", "validity", contract["test_or_restricted_files_opened"], "==", 0, "path guard and training manifest"))

    add(check("noninf.full.early_exact", "noninferiority", full["early_fraud"]["numerator"], "==", EXPECTED_IMMUTABLE_FULL["early_hits"], "frozen M1 full text"))
    add(check("noninf.full.explicit_exact", "noninferiority", full["explicit_fraud"]["numerator"], "==", EXPECTED_IMMUTABLE_FULL["explicit_hits"], "frozen M1 full text"))
    add(check("noninf.full.fraud_exact", "noninferiority", full["all_fraud"]["numerator"], "==", EXPECTED_IMMUTABLE_FULL["fraud_hits"], "frozen M1 full text"))
    add(check("noninf.full.normal_fp_exact", "noninferiority", full["normal_false_positive"]["numerator"], "==", EXPECTED_IMMUTABLE_FULL["normal_fp"], "frozen M1 full text"))
    add(check("noninf.k2.early_effective", "noninferiority", early["numerator"], ">=", EXPECTED_NI_K2["early_te_after_min"], "E2-D k2 early"))
    add(check("noninf.k2.all_effective", "noninferiority", all_effective["numerator"], ">=", EXPECTED_NI_K2["all_te_after_min"], "E2-D k2 all fraud"))
    add(check("noninf.k2.normal_fp", "noninferiority", normal_fp["numerator"], "<=", EXPECTED_NI_K2["normal_ever_fp_max"], "E2-D k2 normal"))
    add(check("noninf.k2.premature", "noninferiority", premature["numerator"], "<=", EXPECTED_NI_K2["all_premature_max"], "E2-D k2 all fraud"))
    add(check("noninf.k2.clean", "noninferiority", clean["numerator"], ">=", EXPECTED_NI_K2["clean_effective_min"], "E2-D k2 all fraud"))
    add(check("noninf.k2.before_ta", "noninferiority", before_ta["numerator"], ">=", EXPECTED_NI_K2["explicit_before_ta_min"], "E2-D k2 explicit"))
    add(check("noninf.k2.delay_median", "noninferiority", delay["median_turns"], "<=", EXPECTED_NI_K2["all_delay_median_max"], "detected fraud"))
    add(check("noninf.k2.delay_p75", "noninferiority", delay["p75_turns"], "<=", EXPECTED_NI_K2["all_delay_p75_max"], "detected fraud"))
    add(check("noninf.long.tp_exact", "noninferiority", long_metric["confusion"]["tp"], "==", EXPECTED_IMMUTABLE_LONG["fraud_hits"], "frozen M1 V361 long Dev"))
    add(check("noninf.long.fp_exact", "noninferiority", long_metric["confusion"]["fp"], "==", EXPECTED_IMMUTABLE_LONG["normal_fp"], "frozen M1 V361 long Dev"))

    add(check("improve.k2.normal_fp", "improvement", normal_fp["numerator"], "<=", EXPECTED_IMPROVE_K2["normal_ever_fp_max"], "E2-D k2 normal"))
    add(check("improve.k2.premature", "improvement", premature["numerator"], "<=", EXPECTED_IMPROVE_K2["all_premature_max"], "E2-D k2 all fraud"))
    add(check("improve.k2.fp_plus_premature", "improvement", normal_fp["numerator"] + premature["numerator"], "<=", EXPECTED_IMPROVE_K2["normal_fp_plus_premature_max"], "fixed harm-count sum"))
    add(check("improve.k2.clean", "improvement", clean["numerator"], ">=", EXPECTED_IMPROVE_K2["clean_effective_min"], "E2-D k2 all fraud"))

    validity = all(item["passed"] for item in checks if item["group"] == "validity")
    noninferiority = all(item["passed"] for item in checks if item["group"] == "noninferiority")
    improvement = all(item["passed"] for item in checks if item["group"] == "improvement")
    if not validity:
        status = "INVALID"
    elif not noninferiority:
        status = "FAIL"
    elif not improvement:
        status = "HOLD"
    else:
        status = "PASS_FOR_FREEZE"
    return {
        "schema_version": "v365.e2d.gate.v1",
        "status": status,
        "validity_passed": validity,
        "noninferiority_passed": noninferiority,
        "improvement_passed": improvement,
        "checks": checks,
        "test_policy": "LOCKED: no status authorizes or triggers V365 Test access",
        "test_or_restricted_files_opened": 0,
    }


def ratio(block: Mapping[str, Any]) -> str:
    if block.get("rate") is None:
        return f"{block['numerator']}/{block['denominator']}"
    return f"{block['numerator']}/{block['denominator']} ({100.0 * block['rate']:.2f}%)"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(
        [
            "| " + " | ".join(clean(value) for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
    )


def build_report(
    full: Mapping[str, Mapping[str, Any]], temporal: Mapping[str, Mapping[str, Any]],
    long_metrics: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any],
    diagnostics: Mapping[str, Any], transitions: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]], identity: Mapping[str, Any],
) -> str:
    available = [model_id for model_id in MODEL_ORDER if model_id in full and model_id in temporal and model_id in long_metrics]
    full_rows: list[list[Any]] = []
    temporal_rows: list[list[Any]] = []
    long_rows: list[list[Any]] = []
    for model_id in available:
        f = full[model_id]["subgroups"]
        t = temporal[model_id]["policies"]["k2"]
        full_rows.append([
            MODEL_LABELS[model_id], ratio(f["early_fraud"]), ratio(f["explicit_fraud"]),
            ratio(f["all_fraud"]), ratio(f["normal_false_positive"]),
        ])
        temporal_rows.append([
            MODEL_LABELS[model_id],
            ratio(t["subsets"]["early_fraud"]["tE_after_effective_recall"]),
            ratio(t["subsets"]["all_fraud"]["tE_after_effective_recall"]),
            ratio(t["normal_ever_false_alarm"]),
            ratio(t["subsets"]["all_fraud"]["tE_before_premature_alarm_rate"]),
            ratio(t["subsets"]["all_fraud"]["clean_effective_rate"]),
            ratio(t["explicit_before_tA"]),
            t["subsets"]["all_fraud"]["delay"]["median_turns"],
            t["subsets"]["all_fraud"]["delay"]["p75_turns"],
        ])
        confusion = long_metrics[model_id]["confusion"]
        long_rows.append([MODEL_LABELS[model_id], confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"]])
    failed = [item for item in gate["checks"] if not item["passed"]]
    transition_counts = {
        field: dict(Counter(str(row[field]) for row in transitions))
        for field in (
            "effective_transition", "premature_transition", "clean_effective_transition",
            "normal_ever_false_alarm_transition",
        )
    }
    return "\n".join([
        "# V365 E2-D Dev比较与冻结门禁", "",
        f"- 最终状态：**{gate['status']}**。",
        "- E2-D仅在冻结M1的k2报警上执行因果门控；不能新增报警。",
        "- k1、M1诈骗概率、完整文本输出和V361长文本输出均保持不变。",
        "- 本报告只使用Train派生gate和Dev评测；V365 Test/restricted读取数为0，继续锁定。", "",
        "## 完整文本（二分类，精确复用M1）", "",
        markdown_table(["模型", "early", "explicit", "全部诈骗", "normal误报"], full_rows), "",
        "## 累计前缀主策略k2", "",
        markdown_table([
            "模型", "early有效", "全部诈骗有效", "normal曾误报", "过早报警",
            "clean effective", "strict tA前", "延迟中位数", "延迟P75",
        ], temporal_rows), "",
        "## V361长文本（二分类，精确复用M1）", "",
        markdown_table(["模型", "TP", "FP", "TN", "FN"], long_rows), "",
        "## 门控审计", "",
        f"- 相邻前缀pair：{diagnostics['evaluated_adjacent_pairs']}/{diagnostics['expected_adjacent_pairs']}。",
        f"- M1 k2报警窗口：{diagnostics['m1_k2_alert_windows']}；E2-D保留：{diagnostics['e2d_k2_alert_windows']}；抑制：{diagnostics['suppressed_m1_k2_windows']}。",
        f"- 新增报警违规：{diagnostics['alert_subset_violation_count']}。",
        f"- 冻结M1全文字节副本SHA相同：{identity['full_byte_copy']['byte_identical']}。",
        f"- 冻结M1长文本字节副本SHA相同：{identity['long_byte_copy']['byte_identical']}。", "",
        "## 门禁", "",
        markdown_table(
            ["组", "检查", "观测", "关系", "限值", "通过"],
            [[item["group"], item["id"], item["observed"], item["operator"], item["limit"], item["passed"]] for item in gate["checks"]],
        ), "",
        f"未通过项：{json.dumps([item['id'] for item in failed], ensure_ascii=False)}。", "",
        "## M1→E2-D离散转移", "",
        "```json", json.dumps(transition_counts, ensure_ascii=False, indent=2), "```", "",
        f"错误审计记录数：{len(errors)}。证据仅保留当前触发窗口和tE锚点，不复制更晚轮次。", "",
        "## 解释边界", "",
        "- gate训练边界由Train的tE机械派生，属于弱监督，不宣称逐前缀独立盲标。",
        "- Dev只用于一次冻结评测；checkpoint、scaler、阈值和门禁均不得用Dev拟合或选择。",
        "- PASS_FOR_FREEZE仅表示可冻结候选，不能自动读取或运行V365 Test。",
    ])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--prefix-input", type=Path, required=True)
    parser.add_argument("--long-dev-source", type=Path, required=True)
    parser.add_argument("--e2a-root", type=Path, required=True)
    parser.add_argument("--e2b-root", type=Path, required=True)
    parser.add_argument("--e2c-root", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    dev_source = safe_file(args.dev_source, EXPECTED_DEV_SUFFIX)
    prefix_input = safe_file(args.prefix_input, EXPECTED_PREFIX_SUFFIX)
    long_source = safe_file(args.long_dev_source, EXPECTED_LONG_SUFFIX)
    e2a_root = safe_dir(args.e2a_root, EXPECTED_E2A_SUFFIX)
    e2b_root = safe_dir(args.e2b_root, EXPECTED_E2B_SUFFIX)
    e2c_root = safe_dir(args.e2c_root, EXPECTED_E2C_SUFFIX)
    gate_checkpoint = safe_file(args.gate_checkpoint)
    preregistration = safe_file(args.preregistration)
    training_manifest_path = safe_file(gate_checkpoint.parent / "training_manifest.json")
    output = safe_output(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E2-D evaluation output: {output}")

    script_dir = Path(__file__).resolve().parent
    baseline_path = safe_file(
        script_dir.parent / "evaluation" / "report_v365_dev_baseline.py"
    )
    common_path = safe_file(script_dir / "e2d_gate_common.py")
    baseline = load_module(baseline_path, "v365_dev_baseline_for_e2d")
    common = load_module(common_path, "e2d_gate_common_for_report")

    prereg = read_json(preregistration)
    validate_preregistration(prereg)
    training_manifest = read_json(training_manifest_path)
    model, mean, scale, gate_payload = common.load_gate(gate_checkpoint)
    contract = validate_gate_contract(
        gate_payload, training_manifest, preregistration, prereg,
        gate_checkpoint, common, mean, scale,
    )

    e2a_manifest_path = safe_file(e2a_root / "run_manifest.json")
    e2a_comparison_path = safe_file(e2a_root / "reports" / "model_comparison.json")
    e2a_manifest = read_json(e2a_manifest_path)
    e2a_comparison = read_json(e2a_comparison_path)
    if e2a_manifest.get("schema_version") != "v365.e2a.dev_compare.v1":
        raise ValueError("unexpected canonical E2-A manifest")
    if e2a_comparison.get("manifest") != e2a_manifest:
        raise ValueError("E2-A manifest and comparison disagree")
    if int((e2a_manifest.get("safety") or {}).get("test_or_restricted_files_opened", -1)) != 0:
        raise ValueError("canonical E2-A does not attest zero Test/restricted reads")
    require_manifest_input_hash(e2a_manifest, "v365_dev", dev_source)
    require_manifest_input_hash(e2a_manifest, "v365_prefix_input", prefix_input)
    require_manifest_input_hash(e2a_manifest, "v361_long_dev", long_source)
    m1_model = (e2a_manifest.get("models") or {}).get("m1_e1b") or {}
    if m1_model.get("checkpoint_sha256") != common.M1_SHA256:
        raise ValueError("canonical M1 SHA differs from gate contract")

    e2b_manifest_path = safe_file(e2b_root / "run_manifest.json")
    e2b_comparison_path = safe_file(e2b_root / "reports" / "model_comparison.json")
    e2c_manifest_path = safe_file(e2c_root / "run_manifest.json")
    e2c_comparison_path = safe_file(e2c_root / "reports" / "model_comparison.json")
    e2b_manifest, e2b_comparison = read_json(e2b_manifest_path), read_json(e2b_comparison_path)
    e2c_manifest, e2c_comparison = read_json(e2c_manifest_path), read_json(e2c_comparison_path)
    for name, manifest, comparison, schema in (
        ("E2-B", e2b_manifest, e2b_comparison, "v365.e2b.dev_compare.v1"),
        ("E2-C", e2c_manifest, e2c_comparison, "v365.e2c.dev_compare.v1"),
    ):
        if manifest.get("schema_version") != schema or comparison.get("manifest") != manifest:
            raise ValueError(f"unexpected or inconsistent {name} canonical report")
        if int((manifest.get("safety") or {}).get("test_or_restricted_files_opened", -1)) != 0:
            raise ValueError(f"canonical {name} does not attest zero Test/restricted reads")

    dev_rows = baseline.validate_dev_source(dev_source)
    prefix_inputs = read_jsonl(prefix_input)
    long_source_rows = read_jsonl(long_source)
    if len(prefix_inputs) != 4492 or len(long_source_rows) != 240:
        raise ValueError("unexpected fixed evaluation row count")

    m1_dir = safe_dir(e2a_root / "scored" / "m1_e1b")
    m1_full_path = safe_file(m1_dir / "v365_fulltext_predictions.jsonl")
    m1_prefix_path = safe_file(m1_dir / "v365_prefix_predictions.jsonl")
    m1_long_path = safe_file(m1_dir / "v361_longdev_predictions.jsonl")
    m1_full_metric_path = safe_file(m1_dir / "v365_fulltext_metrics.json")
    m1_temporal_metric_path = safe_file(m1_dir / "v365_temporal_metrics.json")
    m1_long_metric_path = safe_file(m1_dir / "v361_longdev_metrics.json")
    for path in (
        m1_full_path, m1_prefix_path, m1_long_path,
        m1_full_metric_path, m1_temporal_metric_path, m1_long_metric_path,
    ):
        suffix = path.resolve().as_posix().split("runs/", 1)[-1]
        expected_hash = manifest_hash_for_suffix(e2c_manifest, "runs/" + suffix)
        if expected_hash is None or expected_hash != sha256_file(path):
            raise ValueError(f"canonical E2-C audit does not bind M1 scored bytes: {path.name}")

    m1_full_rows = read_jsonl(m1_full_path)
    m1_scored_prefix = read_jsonl(m1_prefix_path)
    m1_long_rows = read_jsonl(m1_long_path)
    validate_probability_rows(m1_full_rows, 200, "call_id")
    validate_probability_rows(m1_scored_prefix, 4492, "sample_id")
    validate_probability_rows(m1_long_rows, 240, "sample_id")
    normalized_prefix, m1_calls = baseline.build_prefix_records(
        dev_rows, prefix_inputs, raw_from_scored(m1_scored_prefix), "m1_e1b"
    )
    if [row["sample_id"] for row in normalized_prefix] != [row["sample_id"] for row in m1_scored_prefix]:
        by_id = {str(row["sample_id"]): row for row in normalized_prefix}
        if set(by_id) != {str(row["sample_id"]) for row in m1_scored_prefix}:
            raise ValueError("normalized/canonical M1 prefix identities differ")
        normalized_prefix = [by_id[str(row["sample_id"])] for row in m1_scored_prefix]
    for canonical, normalized in zip(m1_scored_prefix, normalized_prefix):
        if (
            float(canonical["p_fraud"]) != float(normalized["p_fraud"])
            or bool(canonical["alert_k1"]) != bool(normalized["alert_k1"])
            or bool(canonical["alert_k2"]) != bool(normalized["alert_k2"])
        ):
            raise ValueError("canonical M1 prefix differs after causal reconstruction")

    e2d_prefix, e2d_calls, diagnostics = build_gated_temporal(
        dev_rows, normalized_prefix, m1_calls, common, model, mean, scale
    )
    m1_full_metric = read_json(m1_full_metric_path)
    m1_temporal_metric = read_json(m1_temporal_metric_path)
    m1_long_metric = read_json(m1_long_metric_path)
    if m1_full_metric != e2a_comparison["v365_fulltext"]["m1_e1b"]:
        raise ValueError("canonical M1 full metric file differs from audited E2-A comparison")
    if m1_temporal_metric != e2a_comparison["v365_temporal"]["m1_e1b"]:
        raise ValueError("canonical M1 temporal metric file differs from audited E2-A comparison")
    if m1_long_metric != e2a_comparison["v361_longdev"]["m1_e1b"]:
        raise ValueError("canonical M1 long metric file differs from audited E2-A comparison")
    recomputed_full_metric = baseline.binary_metrics(
        m1_full_rows, copy.deepcopy(m1_full_metric["timing"])
    )
    endpoint = baseline.endpoint_consistency(m1_full_rows, m1_calls, "m1_e1b")
    recomputed_full_metric["diagnostics"]["full_vs_final_prefix"] = endpoint
    if recomputed_full_metric != m1_full_metric:
        raise ValueError("M1 full predictions do not reproduce the canonical metric file")
    recomputed_temporal_metric = baseline.temporal_metrics(
        m1_calls, copy.deepcopy(m1_temporal_metric["timing"])
    )
    recomputed_temporal_metric["endpoint_consistency"] = endpoint
    if recomputed_temporal_metric != m1_temporal_metric:
        raise ValueError("M1 prefix predictions do not reproduce the canonical metric file")
    long_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in m1_long_rows:
        truth, prediction = int(row["y_true"]), int(row["prediction"])
        long_confusion[
            "tp" if truth == 1 and prediction == 1
            else "fn" if truth == 1
            else "fp" if prediction == 1
            else "tn"
        ] += 1
    if long_confusion != m1_long_metric["confusion"]:
        raise ValueError("M1 long predictions do not reproduce the canonical confusion matrix")
    timing = copy.deepcopy(m1_temporal_metric["timing"])
    timing.update({
        "frozen_m1_inference_reused": True,
        "gate_cpu_seconds": diagnostics["gate_cpu_seconds"],
        "gate_parameters": int(training_manifest["parameters"]),
    })
    e2d_temporal_metric = baseline.temporal_metrics(e2d_calls, timing)
    e2d_temporal_metric["endpoint_consistency"] = endpoint

    normalized_history, schema_compatibility = normalize_historical_m1_metrics(
        e2c_comparison, endpoint, m1_long_rows,
        {
            "v365_fulltext": m1_full_metric,
            "v365_temporal": m1_temporal_metric,
            "v361_longdev": m1_long_metric,
        },
    )

    e2d_full_rows = [dict(row, model_id="e2d") for row in m1_full_rows]
    e2d_long_rows = [dict(row, model_id="e2d") for row in m1_long_rows]
    identity = {
        "full_values_identical": identical_prediction_values(m1_full_rows, e2d_full_rows, "call_id"),
        "long_values_identical": identical_prediction_values(m1_long_rows, e2d_long_rows, "sample_id"),
    }

    output.mkdir(parents=True)
    scored = output / "scored" / "e2d"
    proof = output / "proof" / "m1_unchanged"
    reports = output / "reports"
    write_jsonl(scored / "v365_fulltext_predictions.jsonl", e2d_full_rows)
    write_jsonl(scored / "v365_prefix_predictions.jsonl", e2d_prefix)
    write_jsonl(scored / "v365_temporal_calls.jsonl", e2d_calls)
    e2d_full_metric = copy.deepcopy(m1_full_metric)
    e2d_long_metric = copy.deepcopy(m1_long_metric)
    write_json(scored / "v365_fulltext_metrics.json", e2d_full_metric)
    write_json(scored / "v365_temporal_metrics.json", e2d_temporal_metric)
    write_jsonl(scored / "v361_longdev_predictions.jsonl", e2d_long_rows)
    write_json(scored / "v361_longdev_metrics.json", e2d_long_metric)
    identity["full_byte_copy"] = copy_exact(
        m1_full_path, proof / "v365_fulltext_predictions.jsonl"
    )
    identity["long_byte_copy"] = copy_exact(
        m1_long_path, proof / "v361_longdev_predictions.jsonl"
    )
    identity["full_metric_byte_copy"] = copy_exact(
        m1_full_metric_path, proof / "v365_fulltext_metrics.json"
    )
    identity["long_metric_byte_copy"] = copy_exact(
        m1_long_metric_path, proof / "v361_longdev_metrics.json"
    )

    full_metrics = copy.deepcopy(normalized_history["v365_fulltext"])
    temporal_metrics = copy.deepcopy(normalized_history["v365_temporal"])
    long_metrics = copy.deepcopy(normalized_history["v361_longdev"])
    full_metrics["e2d"] = e2d_full_metric
    temporal_metrics["e2d"] = e2d_temporal_metric
    long_metrics["e2d"] = e2d_long_metric
    for container, canonical in (
        (full_metrics, e2a_comparison["v365_fulltext"]["m1_e1b"]),
        (temporal_metrics, e2a_comparison["v365_temporal"]["m1_e1b"]),
        (long_metrics, e2a_comparison["v361_longdev"]["m1_e1b"]),
    ):
        if container["m1_e1b"] != canonical:
            raise ValueError("historical comparison changed canonical M1 metrics")

    gate = evaluate_gate(
        dev_rows, prefix_inputs, long_source_rows, e2d_full_metric,
        e2d_temporal_metric, e2d_long_metric, diagnostics, identity, contract,
    )
    errors = build_errors(dev_rows, e2d_calls)
    transitions = build_transitions(m1_calls, e2d_calls)
    write_jsonl(reports / "e2d_error_audit_future_filtered.jsonl", errors)
    write_jsonl(reports / "e2d_vs_m1_transitions.jsonl", transitions)
    write_csv(reports / "e2d_vs_m1_transitions.csv", transitions)
    write_json(reports / "gate_decision.json", gate)

    consumed = [
        dev_source, prefix_input, long_source, gate_checkpoint, preregistration,
        training_manifest_path, e2a_manifest_path, e2a_comparison_path,
        e2b_manifest_path, e2b_comparison_path, e2c_manifest_path, e2c_comparison_path,
        m1_full_path, m1_prefix_path, m1_long_path, m1_full_metric_path,
        m1_temporal_metric_path, m1_long_metric_path, baseline_path, common_path,
    ]
    manifest = {
        "schema_version": "v365.e2d.dev_compare.v1",
        "threshold": THRESHOLD,
        "primary_policy": "frozen_m1_k2_and_gate_ge_0_5",
        "k1_role": "unchanged_m1_diagnostic_only",
        "models": {
            "m1_e1b": {
                "checkpoint": m1_model.get("checkpoint"),
                "sha256": common.M1_SHA256,
                "fulltext_and_longdoc_frozen": True,
            },
            "e2d_gate": {
                "checkpoint": str(gate_checkpoint),
                "sha256": contract["gate_checkpoint_sha256"],
                "parameters": int(training_manifest["parameters"]),
                "feature_names": list(common.FEATURE_NAMES),
            },
        },
        "preregistration": {
            "path": str(preregistration),
            "sha256": contract["preregistration_sha256"],
        },
        "training_contract": contract,
        "identity_proof": identity,
        "diagnostics": diagnostics,
        "historical_schema_compatibility": schema_compatibility,
        "inputs": {str(path): sha256_file(path) for path in consumed},
        "code": {
            "report": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "gate_common": {"path": str(common_path), "sha256": sha256_file(common_path)},
        },
        "safety": {
            "training_performed_by_report": False,
            "dev_used_for_gate_or_scaler_fit": False,
            "threshold_or_policy_tuned": False,
            "test_or_restricted_files_opened": 0,
            "path_guard_before_every_open": True,
            "test_remains_locked_after_pass": True,
            "turn_stage_labels_used_as_independent_blind_labels": False,
        },
    }
    comparison = {
        "manifest": manifest,
        "gate": gate,
        "v365_fulltext": full_metrics,
        "v365_temporal": temporal_metrics,
        "v361_longdev": long_metrics,
        "m1_to_e2d_transition_counts": {
            field: dict(Counter(str(row[field]) for row in transitions))
            for field in (
                "effective_transition", "premature_transition", "clean_effective_transition",
                "normal_ever_false_alarm_transition",
            )
        },
        "e2d_error_audit_count": len(errors),
    }
    write_json(output / "run_manifest.json", manifest)
    write_json(reports / "model_comparison.json", comparison)
    report = build_report(
        full_metrics, temporal_metrics, long_metrics, gate, diagnostics,
        transitions, errors, identity,
    )
    (reports / "V365_E2D_DEV_COMPARISON_REPORT.md").write_text(
        report + "\n", encoding="utf-8"
    )
    weights = [
        path for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".bin", ".safetensors"}
    ]
    if weights:
        raise AssertionError(f"report output unexpectedly contains weights: {weights}")
    print(json.dumps({
        "status": gate["status"],
        "gate": str(reports / "gate_decision.json"),
        "report": str(reports / "V365_E2D_DEV_COMPARISON_REPORT.md"),
        "test_accessed": False,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = safe_output(args.output_dir)
    if output.exists():
        print(f"refusing to overwrite E2-D evaluation output: {output}")
        return 2
    try:
        return run(args)
    except (ValueError, KeyError, TypeError, AssertionError, FileNotFoundError) as exc:
        if not output.exists():
            output.mkdir(parents=True, exist_ok=True)
        invalid = {
            "schema_version": "v365.e2d.gate.v1",
            "status": "INVALID",
            "reason": f"{type(exc).__name__}: {exc}",
            "test_policy": "LOCKED; invalid evaluation never authorizes Test",
            "test_or_restricted_files_opened": 0,
        }
        write_json(output / "reports" / "gate_decision.json", invalid)
        print(json.dumps(invalid, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
