"""长文本分类与分层评测指标。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def confusion_matrix(
    labels: Iterable[int], predictions: Iterable[int], num_classes: int
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels, predictions):
        if 0 <= int(label) < num_classes and 0 <= int(prediction) < num_classes:
            matrix[int(label)][int(prediction)] += 1
    return matrix


def macro_f1(
    labels: Iterable[int],
    predictions: Iterable[int],
    num_classes: int,
    *,
    observed_only: bool = False,
) -> float:
    labels = [int(value) for value in labels]
    predictions = [int(value) for value in predictions]
    matrix = confusion_matrix(labels, predictions, num_classes)
    observed = set(labels) | set(predictions)
    indices = sorted(observed) if observed_only else list(range(num_classes))
    scores: list[float] = []
    for index in indices:
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(num_classes)) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2.0 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / max(len(scores), 1)


def classification_metrics(
    labels: Iterable[int], predictions: Iterable[int], num_classes: int = 13
) -> dict[str, float | int]:
    labels = [int(value) for value in labels]
    predictions = [int(value) for value in predictions]
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions have different lengths")
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    fraud_labels = [label != 0 for label in labels]
    fraud_predictions = [prediction != 0 for prediction in predictions]
    fraud_tp = sum(a and b for a, b in zip(fraud_labels, fraud_predictions))
    fraud_fn = sum(a and not b for a, b in zip(fraud_labels, fraud_predictions))
    normal_fp = sum((not a) and b for a, b in zip(fraud_labels, fraud_predictions))
    normal_tn = sum((not a) and (not b) for a, b in zip(fraud_labels, fraud_predictions))
    normal_count = sum(not value for value in fraud_labels)
    fraud_count = sum(fraud_labels)
    return {
        "count": len(labels),
        "accuracy": correct / max(len(labels), 1),
        "macro_f1": macro_f1(labels, predictions, num_classes),
        "observed_macro_f1": macro_f1(
            labels, predictions, num_classes, observed_only=True
        ),
        "binary_accuracy": (fraud_tp + normal_tn) / max(len(labels), 1),
        "fraud_recall": fraud_tp / max(fraud_tp + fraud_fn, 1),
        "normal_fpr": normal_fp / max(normal_count, 1),
        "fraud_count": fraud_count,
        "normal_count": normal_count,
    }


def evidence_position_bucket(start: int, end: int, total: int) -> str:
    if total <= 0:
        return "unknown"
    midpoint = (float(start) + float(end)) / 2.0 / float(total)
    if midpoint < 0.25:
        return "Q1"
    if midpoint < 0.50:
        return "Q2"
    if midpoint < 0.75:
        return "Q3"
    return "Q4"


def length_band(raw_wordpiece_length: int, chunk_length: int) -> str:
    length = int(raw_wordpiece_length)
    if length <= chunk_length:
        return "short"
    if length <= 512:
        return "medium"
    if length <= 1024:
        return "long"
    return "very_long"


def _group_records(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(key) or "unknown")].append(record)
    return dict(grouped)


def stratified_metrics(
    records: Sequence[Mapping[str, Any]], *, chunk_length: int
) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["length_band"] = length_band(
            int(item.get("raw_wordpiece_length", 0)), chunk_length
        )
        enriched.append(item)

    result: dict[str, Any] = {}
    for axis in (
        "length_band",
        "evidence_position_bin",
        "scenario_tag",
        "generation_kind",
        "content_group",
        "cross_window_evidence",
        "subcategory_code",
    ):
        result[axis] = {}
        for value, group in sorted(_group_records(enriched, axis).items()):
            result[axis][value] = classification_metrics(
                [int(row["label_13"]) for row in group],
                [int(row["prediction_13"]) for row in group],
                13,
            )
    return result
