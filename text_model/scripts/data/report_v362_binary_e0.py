#!/usr/bin/env python3
"""Generate the final V362 E0 report using only the dedicated binary head.

The 13-class joint head is deliberately ignored.  The fraud probability used
everywhere in this report is ``softmax(binary_logits)[fraud]``, serialized by
the inference entry point as ``p_binary[1]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FRAUD_DECISIONS = {"early_fraud", "explicit_fraud"}
NORMAL_DECISION = "normal"
AMBIGUOUS_DECISION = "ambiguous"
TURN_SEPARATOR = "@@t"
FIXED_THRESHOLD = 0.5
TEMPORAL_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
TEMPORAL_K_GRID = (1, 2, 3, 4)


class ReportError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReportError(f"missing file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReportError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ReportError(f"empty file: {path}")
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def describe(values: Sequence[float | int]) -> dict[str, Any]:
    cast = [float(value) for value in values]
    if not cast:
        return {"count": 0}
    return {
        "count": len(cast),
        "min": min(cast),
        "p25": quantile(cast, 0.25),
        "median": quantile(cast, 0.50),
        "p75": quantile(cast, 0.75),
        "max": max(cast),
        "mean": statistics.fmean(cast),
    }


def binary_score(record: Mapping[str, Any]) -> float:
    probabilities = record.get("p_binary")
    if not isinstance(probabilities, list) or len(probabilities) != 2:
        raise ReportError(f"{record.get('sample_id')}: missing two-value p_binary")
    normal, fraud = (float(probabilities[0]), float(probabilities[1]))
    if not (math.isfinite(normal) and math.isfinite(fraud)):
        raise ReportError(f"{record.get('sample_id')}: p_binary is not finite")
    if abs(normal + fraud - 1.0) > 1e-4:
        raise ReportError(f"{record.get('sample_id')}: p_binary does not sum to one")
    predicted = int(record.get("prediction_binary", -1))
    expected = 1 if fraud >= normal else 0
    if predicted != expected:
        raise ReportError(f"{record.get('sample_id')}: prediction_binary disagrees with p_binary")
    return fraud


def binary_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    if len(labels) != len(scores) or not labels:
        raise ReportError("binary metrics received empty or mismatched inputs")
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = int(float(score) >= threshold)
        if label == 1 and predicted == 1:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted == 1:
            fp += 1
        else:
            tn += 1
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = (
        0.0
        if precision is None or recall is None or precision + recall == 0.0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "threshold": threshold,
        "samples": len(labels),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": safe_div(tp + tn, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_fpr": safe_div(fp, fp + tn),
        "specificity": specificity,
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in ordered:
        tp += int(label == 1)
        fp += int(label == 0)
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def with_ranking(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    result = binary_metrics(labels, scores, threshold)
    result["roc_auc"] = roc_auc(labels, scores)
    result["pr_auc"] = pr_auc(labels, scores)
    return result


def prediction_path(run_dir: Path, view: str) -> Path:
    return run_dir / "predictions" / view / "predictions.jsonl"


def load_predictions(run_dir: Path, view: str) -> list[dict[str, Any]]:
    rows = read_jsonl(prediction_path(run_dir, view))
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in seen:
            raise ReportError(f"{view}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
        binary_score(row)
    return rows


def endpoint_rows(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Mapping[str, Any]] | None,
    *,
    split: str,
    view: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in predictions:
        sample_id = str(record["sample_id"])
        score = binary_score(record)
        if annotations is None:
            truth = int(int(record["label_13"]) != 0)
            decision = "eventual_fraud" if truth else NORMAL_DECISION
        else:
            if sample_id not in annotations:
                raise ReportError(f"{view}: missing annotation for {sample_id}")
            decision = str(annotations[sample_id]["decision"])
            if decision == AMBIGUOUS_DECISION:
                excluded.append({"sample_id": sample_id, "score": score, "decision": decision})
                continue
            if decision not in FRAUD_DECISIONS | {NORMAL_DECISION}:
                raise ReportError(f"{sample_id}: unknown decision {decision}")
            truth = int(decision in FRAUD_DECISIONS)
        counted.append({
            "sample_id": sample_id,
            "split": split,
            "view": view,
            "decision": decision,
            "truth_binary": truth,
            "p_fraud_binary": score,
            "prediction_binary_0_5": int(score >= FIXED_THRESHOLD),
        })
    return counted, excluded


def endpoint_thresholds(dev_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["truth_binary"]) for row in dev_rows]
    scores = [float(row["p_fraud_binary"]) for row in dev_rows]
    candidates = sorted({0.0, 0.5, 1.0, *scores})
    sweep = [binary_metrics(labels, scores, threshold) for threshold in candidates]
    best_f1 = max(sweep, key=lambda row: (row["f1"] or 0.0, row["recall"] or 0.0, row["threshold"]))
    feasible = [row for row in sweep if (row["normal_fpr"] or 0.0) <= 0.05]
    best_fpr5 = max(feasible, key=lambda row: (row["recall"] or 0.0, row["precision"] or 0.0, row["threshold"]))
    return {"fixed_0_5": binary_metrics(labels, scores, 0.5), "dev_best_f1": best_f1, "dev_best_recall_at_fpr_le_5pct": best_fpr5}


def parse_temporal_id(value: str) -> tuple[str, int]:
    source, separator, turn = value.rpartition(TURN_SEPARATOR)
    if not separator or not source or not turn.isdigit() or int(turn) < 1:
        raise ReportError(f"invalid temporal sample_id: {value}")
    return source, int(turn)


def load_curves(run_dir: Path, split: str) -> dict[str, list[tuple[int, float]]]:
    rows = load_predictions(run_dir, f"cumulative_prefix_{split}")
    curves: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        source_id, turn = parse_temporal_id(str(row["sample_id"]))
        curves.setdefault(source_id, []).append((turn, binary_score(row)))
    for sample_id, points in curves.items():
        points.sort()
        turns = [turn for turn, _ in points]
        if turns != list(range(1, len(turns) + 1)):
            raise ReportError(f"{sample_id}: temporal turns are not contiguous from one")
    return curves


def alarm_turns(points: Sequence[tuple[int, float]], threshold: float, k: int) -> list[int]:
    consecutive = 0
    turns: list[int] = []
    for turn, score in points:
        consecutive = consecutive + 1 if score >= threshold else 0
        if consecutive >= k:
            turns.append(turn)
    return turns


def temporal_policy_metrics(
    curves: Mapping[str, Sequence[tuple[int, float]]],
    annotations: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for sample_id, points in curves.items():
        annotation = annotations[sample_id]
        decision = str(annotation["decision"])
        events = alarm_turns(points, threshold, k)
        t_e_raw = annotation.get("tE_turn")
        t_e = int(t_e_raw) if t_e_raw is not None else None
        pre_te = [turn for turn in events if t_e is not None and turn < t_e]
        post_te = [turn for turn in events if t_e is not None and turn >= t_e]
        point_by_turn = dict(points)
        details.append({
            "sample_id": sample_id,
            "decision": decision,
            "threshold": threshold,
            "consecutive_k": k,
            "prefix_end_turn": int(annotation["prefix_end_turn"]),
            "tE_turn": t_e,
            "ever_alarm_turn": events[0] if events else None,
            "premature_alarm_before_tE": bool(pre_te),
            "valid_detection_turn_at_or_after_tE": post_te[0] if post_te else None,
            "valid_detection_delay": (post_te[0] - t_e) if post_te else None,
            "probability_at_tE": point_by_turn.get(t_e) if t_e is not None else None,
            "positive_at_tE": (point_by_turn.get(t_e, -1.0) >= threshold) if t_e is not None else None,
            "endpoint_probability": float(points[-1][1]),
            "endpoint_positive": float(points[-1][1]) >= threshold,
        })

    fraud = [row for row in details if row["decision"] in FRAUD_DECISIONS]
    normal = [row for row in details if row["decision"] == NORMAL_DECISION]
    ambiguous = [row for row in details if row["decision"] == AMBIGUOUS_DECISION]
    valid = [row for row in fraud if row["valid_detection_turn_at_or_after_tE"] is not None]
    premature = [row for row in fraud if row["premature_alarm_before_tE"]]
    normal_alarm = [row for row in normal if row["ever_alarm_turn"] is not None]
    at_te = [row for row in fraud if row["positive_at_tE"]]
    endpoint = [row for row in fraud if row["endpoint_positive"]]
    return {
        "threshold": threshold,
        "consecutive_k": k,
        "definition": "alert is emitted on the kth consecutive cut with p_binary[1] >= threshold",
        "fraud": {
            "samples": len(fraud),
            "valid_detection_at_or_after_tE": len(valid),
            "valid_detection_recall": safe_div(len(valid), len(fraud)),
            "valid_detection_delay": describe([row["valid_detection_delay"] for row in valid]),
            "detected_within_tE_plus_2": safe_div(sum(row["valid_detection_delay"] <= 2 for row in valid), len(fraud)),
            "detected_within_tE_plus_4": safe_div(sum(row["valid_detection_delay"] <= 4 for row in valid), len(fraud)),
            "premature_alarm_before_tE": len(premature),
            "premature_alarm_rate": safe_div(len(premature), len(fraud)),
            "positive_exactly_at_tE": len(at_te),
            "positive_exactly_at_tE_rate": safe_div(len(at_te), len(fraud)),
            "endpoint_positive": len(endpoint),
            "endpoint_positive_rate": safe_div(len(endpoint), len(fraud)),
        },
        "normal": {
            "samples": len(normal),
            "calls_with_any_false_alarm": len(normal_alarm),
            "per_call_ever_false_alarm_rate": safe_div(len(normal_alarm), len(normal)),
            "first_false_alarm_turn": describe([row["ever_alarm_turn"] for row in normal_alarm]),
            "endpoint_false_positives": sum(row["endpoint_positive"] for row in normal),
            "endpoint_false_positive_rate": safe_div(sum(row["endpoint_positive"] for row in normal), len(normal)),
        },
        "ambiguous": {
            "samples": len(ambiguous),
            "policy": "reported descriptively and excluded from binary correctness",
            "endpoint_probability": describe([row["endpoint_probability"] for row in ambiguous]),
        },
    }, details


def select_temporal_policy(
    curves: Mapping[str, Sequence[tuple[int, float]]],
    annotations: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    for threshold in TEMPORAL_THRESHOLD_GRID:
        for k in TEMPORAL_K_GRID:
            metrics, _ = temporal_policy_metrics(curves, annotations, threshold=threshold, k=k)
            row = {
                "threshold": threshold,
                "consecutive_k": k,
                "valid_detection_recall": metrics["fraud"]["valid_detection_recall"],
                "median_valid_delay": metrics["fraud"]["valid_detection_delay"].get("median"),
                "premature_alarm_rate": metrics["fraud"]["premature_alarm_rate"],
                "normal_ever_false_alarm_rate": metrics["normal"]["per_call_ever_false_alarm_rate"],
            }
            grid.append(row)
    feasible = [row for row in grid if (row["normal_ever_false_alarm_rate"] or 0.0) <= 0.05]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["valid_detection_recall"] or 0.0,
                -(row["median_valid_delay"] or 0.0),
                -(row["premature_alarm_rate"] or 0.0),
                -row["consecutive_k"],
            ),
        )
        rule = "maximum valid post-tE recall subject to dev normal per-call ever-FPR <= 5%"
    else:
        selected = min(
            grid,
            key=lambda row: (
                row["normal_ever_false_alarm_rate"] if row["normal_ever_false_alarm_rate"] is not None else 1.0,
                -(row["valid_detection_recall"] or 0.0),
                row["median_valid_delay"] or 0.0,
            ),
        )
        rule = "no grid point met dev FPR <= 5%; chose minimum dev normal per-call ever-FPR, then maximum recall"
    return {"selection_rule": rule, **selected}, grid


def normalize_text(value: Any) -> str:
    return "".join(unicodedata.normalize("NFKC", str(value or "")).split())


def derive_operation_turns(
    annotations: Mapping[str, Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sample_id, annotation in sorted(annotations.items()):
        if annotation["decision"] not in FRAUD_DECISIONS:
            continue
        source = source_rows[sample_id]
        evidence = source.get("risk_evidence") or []
        if isinstance(evidence, str):
            evidence = [evidence]
        quote_hits: list[dict[str, Any]] = []
        all_hits: list[int] = []
        for quote in evidence:
            normalized_quote = normalize_text(quote)
            hits = [
                index + 1
                for index, turn in enumerate(source.get("turns") or [])
                if normalized_quote in normalize_text(turn.get("text"))
                or normalize_text(turn.get("text")) in normalized_quote
            ]
            quote_hits.append({"quote": quote, "turn_hits": hits})
            all_hits.extend(hits)
        unique_hits = sorted(set(all_hits))
        operation_turn = min(unique_hits) if unique_hits else None
        t_e = int(annotation["tE_turn"])
        output.append({
            "sample_id": sample_id,
            "split": str(annotation["source_split"]),
            "decision": str(annotation["decision"]),
            "tE_turn": t_e,
            "explicit_operation_turn": operation_turn,
            "lead_turns_from_tE_to_operation": operation_turn - t_e if operation_turn is not None else None,
            "derivation_source": "v361.risk_evidence exact normalized text match to source turns",
            "match_status": "matched" if operation_turn is not None else "unmatched",
            "requires_manual_spot_check": True,
            "quote_matches": quote_hits,
        })
    return output


def attach_operation_lead(
    details: list[dict[str, Any]], operation_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    model_leads: list[int] = []
    annotation_leads: list[int] = []
    unmatched: list[str] = []
    for row in details:
        if row["decision"] not in FRAUD_DECISIONS:
            continue
        operation = operation_by_id.get(str(row["sample_id"]))
        operation_turn = operation.get("explicit_operation_turn") if operation else None
        row["explicit_operation_turn"] = operation_turn
        row["model_lead_turns_before_operation"] = None
        if operation_turn is None:
            unmatched.append(str(row["sample_id"]))
            continue
        annotation_leads.append(int(operation_turn) - int(row["tE_turn"]))
        detection = row["valid_detection_turn_at_or_after_tE"]
        if detection is not None:
            lead = int(operation_turn) - int(detection)
            row["model_lead_turns_before_operation"] = lead
            model_leads.append(lead)
    return {
        "operation_turn_source": "derived from V361 risk_evidence; not used as model input",
        "manual_review_status": "all mappings are deterministic; manual spot-check is still recommended",
        "unmatched_sample_ids": unmatched,
        "tE_to_operation_lead_turns": describe(annotation_leads),
        "model_detection_to_operation_lead_turns": describe(model_leads),
        "detections_before_or_at_operation": safe_div(sum(value >= 0 for value in model_leads), len(model_leads)),
    }


def load_runtime(run_dir: Path, view: str) -> dict[str, Any]:
    path = run_dir / "predictions" / view / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    return {
        "wall_time_seconds": metrics.get("wall_time_seconds"),
        "throughput_samples_per_second": metrics.get("throughput_samples_per_second"),
        "latency_ms_mean": metrics.get("latency_ms_mean"),
        "latency_ms_p50": metrics.get("latency_ms_p50"),
        "latency_ms_p95": metrics.get("latency_ms_p95"),
        "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v362_e0_binary"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/processed/v362_early_intent"))
    parser.add_argument("--source", type=Path, default=Path("data/processed/v361_longdoc_text_route_20260824/generated_long_all_v361_980.jsonl"))
    parser.add_argument("--report-name", default="E0_BINARY")
    parser.add_argument("--schema-stage", default="e0")
    parser.add_argument("--checkpoint", default="runs/repro_v361b_b320_head96/best_model.pt")
    args = parser.parse_args()
    report_dir = args.run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    annotations_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    all_annotations: dict[str, dict[str, Any]] = {}
    for split in ("dev", "stress"):
        rows = read_jsonl(args.annotation_dir / f"prefix_{split}.jsonl")
        annotations_by_split[split] = {str(row["sample_id"]): row for row in rows}
        all_annotations.update(annotations_by_split[split])
    source_rows = {str(row["sample_id"]): row for row in read_jsonl(args.source)}

    views: dict[str, list[dict[str, Any]]] = {}
    endpoint_details: dict[str, list[dict[str, Any]]] = {}
    endpoint_excluded: dict[str, list[dict[str, Any]]] = {}
    endpoint_report: dict[str, Any] = {}
    for split in ("dev", "stress"):
        for view in ("fulltext", "strict_prefix"):
            name = f"{view}_{split}"
            views[name] = load_predictions(args.run_dir, name)
            rows, excluded = endpoint_rows(
                views[name], None if view == "fulltext" else annotations_by_split[split], split=split, view=view
            )
            endpoint_details[name] = rows
            endpoint_excluded[name] = excluded
            labels = [int(row["truth_binary"]) for row in rows]
            scores = [float(row["p_fraud_binary"]) for row in rows]
            endpoint_report[name] = {
                "truth_definition": "label_13 != 0" if view == "fulltext" else "early_fraud|explicit_fraud vs normal; ambiguous excluded",
                "fixed_0_5": with_ranking(labels, scores, FIXED_THRESHOLD),
                "excluded_ambiguous": {"samples": len(excluded), "p_fraud_binary": describe([row["score"] for row in excluded])},
                "runtime": load_runtime(args.run_dir, name),
            }

    endpoint_selection = endpoint_thresholds(endpoint_details["strict_prefix_dev"])
    endpoint_application: dict[str, Any] = {}
    for split in ("dev", "stress"):
        rows = endpoint_details[f"strict_prefix_{split}"]
        labels = [int(row["truth_binary"]) for row in rows]
        scores = [float(row["p_fraud_binary"]) for row in rows]
        endpoint_application[split] = {
            name: binary_metrics(labels, scores, float(point["threshold"]))
            for name, point in endpoint_selection.items()
        }

    threshold_rows: list[dict[str, Any]] = []
    for name, point in endpoint_selection.items():
        threshold_rows.append({"selected_on": "strict_prefix_dev", "policy": name, **point})
    write_csv(report_dir / "endpoint_thresholds_selected_on_dev.csv", threshold_rows)

    operation_rows = derive_operation_turns(all_annotations, source_rows)
    operation_by_id = {str(row["sample_id"]): row for row in operation_rows}
    write_jsonl(report_dir / "explicit_operation_turns_from_risk_evidence.jsonl", operation_rows)

    curves = {split: load_curves(args.run_dir, split) for split in ("dev", "stress")}
    selected_temporal, temporal_grid = select_temporal_policy(curves["dev"], annotations_by_split["dev"])
    write_csv(report_dir / "temporal_policy_grid_selected_on_dev.csv", temporal_grid)
    temporal_report: dict[str, Any] = {
        "selection": {"selected_on": "dev only", **selected_temporal},
        "stress_role": "fixed diagnostic stress set; never used for selecting threshold or k",
        "policies": {},
    }
    policy_specs = {
        "fixed_0_5_k1": (0.5, 1),
        "fixed_0_5_k2": (0.5, 2),
        "dev_selected": (float(selected_temporal["threshold"]), int(selected_temporal["consecutive_k"])),
    }
    for policy_name, (threshold, k) in policy_specs.items():
        temporal_report["policies"][policy_name] = {}
        for split in ("dev", "stress"):
            metrics, details = temporal_policy_metrics(
                curves[split], annotations_by_split[split], threshold=threshold, k=k
            )
            metrics["operation_lead_time"] = attach_operation_lead(details, operation_by_id)
            metrics["runtime"] = load_runtime(args.run_dir, f"cumulative_prefix_{split}")
            temporal_report["policies"][policy_name][split] = metrics
            write_jsonl(report_dir / f"temporal_details_{policy_name}_{split}.jsonl", details)

    regression: dict[str, Any] = {}
    for split in ("dev", "stress"):
        endpoint_by_id = {
            str(row["sample_id"]): binary_score(row)
            for row in views[f"strict_prefix_{split}"]
        }
        differences = []
        for sample_id, points in curves[split].items():
            differences.append(abs(float(points[-1][1]) - endpoint_by_id[sample_id]))
        regression[split] = {
            "samples": len(differences),
            "max_abs_probability_difference": max(differences),
            "median_abs_probability_difference": quantile(differences, 0.5),
            "decision_flips_at_0_5": sum(
                (float(points[-1][1]) >= 0.5) != (endpoint_by_id[sample_id] >= 0.5)
                for sample_id, points in curves[split].items()
            ),
            "status": "pass" if max(differences) <= 1e-4 else "fail",
        }

    report = {
        "schema_version": f"v362.{args.schema_stage}.binary_head.v1",
        "task": "binary fraud detection only",
        "score_definition": "p_fraud = p_binary[1] = softmax(binary_logits)[1]",
        "forbidden_primary_score": "1 - p_label_13[0] is joint-13-derived and is auxiliary only",
        "checkpoint": args.checkpoint,
        "data": {
            "annotation_dir": str(args.annotation_dir),
            "source": str(args.source),
            "quarantine": str(args.annotation_dir / "quarantine_samples.jsonl"),
        },
        "endpoint": endpoint_report,
        "endpoint_threshold_selection": {
            "selected_on": "strict_prefix_dev only",
            "policies": endpoint_selection,
            "application": endpoint_application,
        },
        "temporal": temporal_report,
        "temporal_endpoint_regression": regression,
        "explicit_operation_turns": {
            "rows": len(operation_rows),
            "matched": sum(row["match_status"] == "matched" for row in operation_rows),
            "unmatched": [row["sample_id"] for row in operation_rows if row["match_status"] != "matched"],
            "file": str(report_dir / "explicit_operation_turns_from_risk_evidence.jsonl"),
        },
        "evaluation_discipline": {
            "dev": "threshold and k may be selected here",
            "stress": "diagnostic regression only; settings are frozen from dev",
            "final_generalization": "company hidden test or a new untouched holdout",
        },
    }
    report_slug = args.report_name.lower()
    metrics_path = report_dir / f"{report_slug}_metrics.json"
    write_json(metrics_path, report)

    for name, rows in endpoint_details.items():
        write_jsonl(report_dir / f"endpoint_details_{name}.jsonl", rows)

    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{100.0 * float(value):.2f}%"

    md: list[str] = [
        f"# V362 {args.report_name} 二分类评测\n\n",
        "本报告只使用专用二分类头：`p_fraud = p_binary[1]`。13 分类联合概率仅保留作调试，不参与结论。\n\n",
        "## 固定阈值 0.5 的终点结果\n\n",
        "| 数据视图 | 样本 | Accuracy | Precision | Recall | F1 | 正常 FPR | PR-AUC |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for name in ("fulltext_dev", "fulltext_stress", "strict_prefix_dev", "strict_prefix_stress"):
        metric = endpoint_report[name]["fixed_0_5"]
        md.append(
            f"| {name} | {metric['samples']} | {pct(metric['accuracy'])} | {pct(metric['precision'])} | "
            f"{pct(metric['recall'])} | {pct(metric['f1'])} | {pct(metric['normal_fpr'])} | {metric['pr_auc']:.4f} |\n"
        )
    md.extend([
        "\n## 时间序列主口径\n\n",
        "`premature_alarm` 表示 tE 以前报警；有效检测延迟只从 tE 开始计算。正常样本的误报率按通话计，只要任一累计前缀报警即记为一次误报。\n\n",
        "| 策略 | Split | 有效检测召回 | 过早报警率 | 正常通话曾误报率 | 延迟中位数(轮) |\n",
        "|---|---|---:|---:|---:|---:|\n",
    ])
    for policy in ("fixed_0_5_k1", "fixed_0_5_k2", "dev_selected"):
        for split in ("dev", "stress"):
            metric = temporal_report["policies"][policy][split]
            md.append(
                f"| {policy} | {split} | {pct(metric['fraud']['valid_detection_recall'])} | "
                f"{pct(metric['fraud']['premature_alarm_rate'])} | "
                f"{pct(metric['normal']['per_call_ever_false_alarm_rate'])} | "
                f"{metric['fraud']['valid_detection_delay'].get('median', 'n/a')} |\n"
            )
    md.extend([
        "\n## 口径与使用限制\n\n",
        "- strict-prefix 主二分类：`early_fraud + explicit_fraud` 为正类，`normal` 为负类，`ambiguous` 不计正确/错误。\n",
        "- 阈值和连续轮数只允许在 dev 上选择；stress 只套用冻结设置。\n",
        "- stress 已在前期反复查看，不能再称为完全未触碰的最终测试集。最终泛化结论以公司隐私测试或新留出的测试集为准。\n",
        "- 显式操作轮次由 V361 `risk_evidence` 与原始轮次做规范化文本匹配得到，未作为模型输入；建议人工抽查映射。\n",
    ])
    (report_dir / f"{args.report_name}_REPORT.md").write_text("".join(md), encoding="utf-8")

    manifest_rows = []
    for path in sorted(args.run_dir.rglob("*")):
        if path.is_file() and path.name != "file_manifest_sha256.json":
            manifest_rows.append({"path": str(path.relative_to(args.run_dir)), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(report_dir / "file_manifest_sha256.json", {"files": manifest_rows})
    print(f"binary E0 report -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
