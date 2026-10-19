#!/usr/bin/env python3
"""Strict Dev-only comparison for M0, M1 (E1-B), and E2-A.

The scorer consumes the already frozen, label-blind V365 Dev full/prefix inputs,
the existing M0/M1 raw inference under the V365 baseline directory, new E2-A
raw inference, and fresh three-model inference on the original V361 long-text
Dev.  Every user-supplied read path is checked before it is opened; paths that
contain ``test`` or ``restricted`` are rejected deliberately.
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
EXPECTED_DEV_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_dev_final.jsonl"
)
EXPECTED_LONG_DEV_SUFFIX = (
    "data/processed/v361_longdoc_text_route_20260824/long_dev_v361_240.jsonl"
)
EXPECTED_BASELINE_SUFFIX = "runs/v365_dev_baseline_compare_20261010"
MODEL_ORDER = ("m0_v361", "m1_e1b", "e2a")
MODEL_LABELS = {
    "m0_v361": "M0 V361",
    "m1_e1b": "M1 E1-B",
    "e2a": "E2-A",
}


def forbidden_path(path: Path) -> bool:
    """Return True for Test/restricted paths, including resolved symlinks."""
    for candidate in (path.absolute(), path.resolve(strict=False)):
        normalized = candidate.as_posix().lower()
        if "restricted" in normalized:
            return True
        if any("test" in part for part in normalized.split("/")):
            return True
    return False


def require_safe_file(path: Path, *, suffix: str | None = None) -> Path:
    if forbidden_path(path):
        raise SystemExit(f"refusing Test/restricted path: {path}")
    resolved = path.resolve()
    if suffix is not None and not resolved.as_posix().endswith(suffix):
        raise SystemExit(f"refusing non-allowlisted path: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_safe_dir(path: Path, *, suffix: str | None = None) -> Path:
    if forbidden_path(path):
        raise SystemExit(f"refusing Test/restricted directory: {path}")
    resolved = path.resolve()
    if suffix is not None and not resolved.as_posix().endswith(suffix):
        raise SystemExit(f"refusing non-allowlisted directory: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def require_safe_output(path: Path) -> Path:
    if forbidden_path(path):
        raise SystemExit(f"refusing Test/restricted output: {path}")
    return path.resolve()


def load_baseline_module() -> ModuleType:
    candidates = [
        Path(__file__).with_name("report_v365_dev_baseline.py"),
        Path.cwd() / "scripts/evaluation/report_v365_dev_baseline.py",
        Path.cwd() / "text_model/scripts/evaluation/report_v365_dev_baseline.py",
    ]
    module_path = next((item.resolve() for item in candidates if item.is_file()), None)
    if module_path is None:
        raise FileNotFoundError(
            "report_v365_dev_baseline.py is required to preserve the audited "
            "V365 boundary and temporal definitions"
        )
    if forbidden_path(module_path):
        raise SystemExit(f"refusing unsafe baseline module path: {module_path}")
    spec = importlib.util.spec_from_file_location("v365_baseline_report", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = {
        "validate_dev_source",
        "read_jsonl",
        "write_jsonl",
        "write_json",
        "sha256_file",
        "canonical_text",
        "sha256_text",
        "raw_timing",
        "prediction_probability",
        "build_full_records",
        "binary_metrics",
        "build_prefix_records",
        "temporal_metrics",
        "endpoint_consistency",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise ImportError(f"baseline report is missing functions: {missing}")
    return module


def read_jsonl_safe(path: Path, baseline: ModuleType) -> list[dict[str, Any]]:
    require_safe_file(path)
    return baseline.read_jsonl(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fixed_full_input(
    dev_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]], baseline: ModuleType
) -> None:
    by_id = {str(row.get("sample_id")): row for row in full_rows}
    expected = {str(row["sample_id"]): row for row in dev_rows}
    if len(by_id) != len(full_rows) or set(by_id) != set(expected):
        raise ValueError("fixed V365 full input ids do not match the 200 Dev calls")
    for sample_id, source in expected.items():
        row = by_id[sample_id]
        text = str(source["model_input_text"])
        if str(row.get("model_input_text") or "") != text:
            raise ValueError(f"{sample_id}: fixed full input text mismatch")
        if str(row.get("input_text_sha256") or "") != baseline.sha256_text(text):
            raise ValueError(f"{sample_id}: fixed full input hash mismatch")
        if float(row.get("binary_loss_mask", -1.0)) != 0.0:
            raise ValueError(f"{sample_id}: full inference input is not label-blind")


def safe_div(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def generic_binary_metrics(
    records: Sequence[Mapping[str, Any]], timing: Mapping[str, Any]
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot score an empty record set")
    y = np.asarray([int(row["y_true"]) for row in records], dtype=int)
    p = np.asarray([float(row["p_fraud"]) for row in records], dtype=float)
    pred = (p >= THRESHOLD).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    both_classes = len(set(y.tolist())) == 2
    return {
        "threshold": THRESHOLD,
        "probability_field": "p_binary[1]",
        "samples": len(records),
        "positive_samples": int(np.sum(y == 1)),
        "negative_samples": int(np.sum(y == 0)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": safe_div(tp + tn, len(records)),
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_fpr": safe_div(fp, fp + tn),
        "roc_auc": (
            finite_or_none(roc_auc_score(y, p)) if both_classes else None
        ),
        "pr_auc_ap": (
            finite_or_none(average_precision_score(y, p)) if both_classes else None
        ),
        "timing": dict(timing),
    }


def validate_long_dev(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 240:
        raise ValueError(f"expected 240 original long Dev rows, got {len(rows)}")
    ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("original long Dev has empty/duplicate sample_id")
    for row in rows:
        primary = long_label_13(row)
        if primary is None or not 0 <= int(primary) <= 12:
            raise ValueError(f"{row.get('sample_id')}: invalid long Dev label")
        if row.get("canonical_label_13") is not None and row.get("label_13") is not None:
            if int(row["canonical_label_13"]) != int(row["label_13"]):
                raise ValueError(f"{row['sample_id']}: conflicting long Dev labels")


def long_label_13(row: Mapping[str, Any]) -> int | None:
    value = row.get("canonical_label_13")
    if value is None:
        value = row.get("label_13")
    return None if value is None else int(value)


def long_records(
    source_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    model_id: str,
    baseline: ModuleType,
) -> list[dict[str, Any]]:
    raw_by_id = {str(row.get("sample_id")): row for row in raw_rows}
    expected_ids = {str(row["sample_id"]) for row in source_rows}
    if len(raw_by_id) != len(raw_rows) or set(raw_by_id) != expected_ids:
        raise ValueError(f"{model_id}: long Dev prediction ids do not match source")
    output: list[dict[str, Any]] = []
    for source in source_rows:
        sample_id = str(source["sample_id"])
        raw = raw_by_id[sample_id]
        p_normal, p_fraud = baseline.prediction_probability(raw)
        label_13_value = long_label_13(source)
        if label_13_value is None:
            raise ValueError(f"{sample_id}: missing long Dev label")
        label_13 = int(label_13_value)
        wordpieces = int(raw.get("raw_wordpiece_length", source.get("raw_wordpiece_length", 0)))
        record = {
            "model_id": model_id,
            "sample_id": sample_id,
            "label_13": label_13,
            "y_true": int(label_13 != 0),
            "p_normal": p_normal,
            "p_fraud": p_fraud,
            "prediction": int(p_fraud >= THRESHOLD),
            "scenario_tag": source.get("scenario_tag"),
            "length_bucket": str(source.get("length_bucket") or wordpiece_bin(wordpieces)),
            "raw_wordpiece_length": wordpieces,
            "n_chunks": int(raw.get("num_chunks", 0)),
            "full_text_coverage": float(raw.get("full_text_coverage", raw.get("coverage_ratio", -1.0))),
        }
        if record["raw_wordpiece_length"] <= 0 or record["n_chunks"] <= 0:
            raise ValueError(f"{model_id}/{sample_id}: missing length/chunk diagnostics")
        if abs(record["full_text_coverage"] - 1.0) > 1e-9:
            raise ValueError(f"{model_id}/{sample_id}: incomplete long Dev coverage")
        output.append(record)
    return output


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


def length_strata(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["length_bucket"])].append(row)
    return {
        bucket: generic_binary_metrics(rows, timing={})
        for bucket, rows in sorted(groups.items())
    }


def compare_e2(
    dev_rows: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    e2_records: list[dict[str, Any]],
    baseline_id: str,
) -> list[dict[str, Any]]:
    source = {str(row["call_id"]): row for row in dev_rows}
    old = {str(row["call_id"]): row for row in baseline_records}
    new = {str(row["call_id"]): row for row in e2_records}
    if set(old) != set(new) or set(old) != set(source):
        raise ValueError(f"{baseline_id}/E2 call ids differ")
    output: list[dict[str, Any]] = []
    for call_id in sorted(source):
        src, lhs, rhs = source[call_id], old[call_id], new[call_id]
        decision = str(src["decision"])
        old_pred, e2_pred = int(lhs["prediction"]), int(rhs["prediction"])
        if decision in {"early_fraud", "explicit_fraud"}:
            if old_pred == 0 and e2_pred == 1:
                transition = "e2_new_hit"
            elif old_pred == 1 and e2_pred == 0:
                transition = "e2_lost_hit"
            else:
                transition = "unchanged_hit" if e2_pred else "unchanged_miss"
        elif decision == "normal":
            if old_pred == 0 and e2_pred == 1:
                transition = "e2_new_false_positive"
            elif old_pred == 1 and e2_pred == 0:
                transition = "e2_corrected_false_positive"
            else:
                transition = "unchanged_false_positive" if e2_pred else "unchanged_true_negative"
        else:
            if old_pred == 0 and e2_pred == 1:
                transition = "ambiguous_crossed_up"
            elif old_pred == 1 and e2_pred == 0:
                transition = "ambiguous_crossed_down"
            else:
                transition = "ambiguous_unchanged_side"
        output.append(
            {
                "baseline_model": baseline_id,
                "call_id": call_id,
                "decision": decision,
                "scenario_tag": src.get("scenario_tag"),
                "n_full_turns": int(src["n_full_turns"]),
                "tE_turn": src.get("tE_turn"),
                "tA_turn": src.get("tA_turn"),
                "baseline_p_fraud": float(lhs["p_fraud"]),
                "baseline_prediction": old_pred,
                "e2_p_fraud": float(rhs["p_fraud"]),
                "e2_prediction": e2_pred,
                "delta_p_fraud": float(rhs["p_fraud"] - lhs["p_fraud"]),
                "transition": transition,
            }
        )
    return output


def ratio_text(block: Mapping[str, Any]) -> str:
    numerator = int(block["numerator"])
    denominator = int(block["denominator"])
    value = block.get("rate")
    suffix = "NA" if value is None else f"{100 * float(value):.2f}%"
    return f"{numerator}/{denominator} ({suffix})"


def number(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(clean(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


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


def flatten_full_metrics(model_id: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": model_id,
        **dict(metrics["confusion"]),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "normal_fpr": metrics["normal_fpr"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc_ap": metrics["pr_auc_ap"],
        "early_hits": metrics["subgroups"]["early_fraud"]["numerator"],
        "early_total": metrics["subgroups"]["early_fraud"]["denominator"],
        "explicit_hits": metrics["subgroups"]["explicit_fraud"]["numerator"],
        "explicit_total": metrics["subgroups"]["explicit_fraud"]["denominator"],
        "normal_false_positives": metrics["subgroups"]["normal_false_positive"]["numerator"],
        "normal_total": metrics["subgroups"]["normal_false_positive"]["denominator"],
        "ambiguous_mean_p_fraud": metrics["ambiguous"]["mean_p_fraud"],
        "ambiguous_median_p_fraud": metrics["ambiguous"]["median_p_fraud"],
        "ambiguous_p90_p_fraud": metrics["ambiguous"]["p90_p_fraud_linear"],
        "ambiguous_count_ge_0_5": metrics["ambiguous"]["count_ge_0_5"],
    }


def flatten_temporal(model_id: str, temporal: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in (1, 2):
        policy = temporal["policies"][f"k{k}"]
        for subset in ("early_fraud", "explicit_fraud", "all_fraud"):
            block = policy["subsets"][subset]
            rows.append(
                {
                    "model_id": model_id,
                    "policy": f"k{k}",
                    "subset": subset,
                    "te_after_n": block["tE_after_effective_recall"]["numerator"],
                    "te_after_d": block["tE_after_effective_recall"]["denominator"],
                    "te_after_rate": block["tE_after_effective_recall"]["rate"],
                    "premature_n": block["tE_before_premature_alarm_rate"]["numerator"],
                    "premature_d": block["tE_before_premature_alarm_rate"]["denominator"],
                    "premature_rate": block["tE_before_premature_alarm_rate"]["rate"],
                    "delay_median_turns": block["delay"]["median_turns"],
                    "delay_detected_n": block["delay"]["detected_n"],
                    "delay_missed_n": block["delay"]["missed_n"],
                    "normal_ever_fp_n": policy["normal_ever_false_alarm"]["numerator"],
                    "normal_ever_fp_d": policy["normal_ever_false_alarm"]["denominator"],
                    "normal_ever_fp_rate": policy["normal_ever_false_alarm"]["rate"],
                    "explicit_before_ta_n": policy["explicit_before_tA"]["numerator"],
                    "explicit_before_ta_d": policy["explicit_before_tA"]["denominator"],
                    "explicit_before_ta_rate": policy["explicit_before_tA"]["rate"],
                }
            )
    return rows


def build_report(
    full: Mapping[str, Mapping[str, Any]],
    temporal: Mapping[str, Mapping[str, Any]],
    long_metrics: Mapping[str, Mapping[str, Any]],
    long_strata: Mapping[str, Mapping[str, Mapping[str, Any]]],
    transitions: Mapping[str, list[dict[str, Any]]],
    manifest: Mapping[str, Any],
) -> str:
    full_rows = []
    ambiguous_rows = []
    for model_id in MODEL_ORDER:
        metric = full[model_id]
        full_rows.append(
            [
                MODEL_LABELS[model_id], metric["confusion"]["tp"], metric["confusion"]["fp"],
                metric["confusion"]["tn"], metric["confusion"]["fn"],
                number(metric["accuracy"]), number(metric["balanced_accuracy"]),
                number(metric["precision"]), number(metric["recall"]), number(metric["f1"]),
                number(metric["normal_fpr"]), number(metric["roc_auc"]), number(metric["pr_auc_ap"]),
                ratio_text(metric["subgroups"]["early_fraud"]),
                ratio_text(metric["subgroups"]["explicit_fraud"]),
                ratio_text(metric["subgroups"]["normal_false_positive"]),
            ]
        )
        ambiguous_rows.append(
            [
                MODEL_LABELS[model_id], metric["ambiguous"]["samples"],
                number(metric["ambiguous"]["mean_p_fraud"]),
                number(metric["ambiguous"]["median_p_fraud"]),
                number(metric["ambiguous"]["p90_p_fraud_linear"]),
                metric["ambiguous"]["count_ge_0_5"],
            ]
        )

    temporal_rows = []
    for k in (1, 2):
        for model_id in MODEL_ORDER:
            policy = temporal[model_id]["policies"][f"k{k}"]
            for subset in ("early_fraud", "explicit_fraud", "all_fraud"):
                block = policy["subsets"][subset]
                temporal_rows.append(
                    [
                        f"k{k}", MODEL_LABELS[model_id], subset,
                        ratio_text(block["tE_after_effective_recall"]),
                        ratio_text(block["tE_before_premature_alarm_rate"]),
                        ratio_text(policy["normal_ever_false_alarm"]),
                        number(block["delay"]["median_turns"], 1),
                        ratio_text(policy["explicit_before_tA"]),
                    ]
                )

    long_rows = []
    for model_id in MODEL_ORDER:
        metric = long_metrics[model_id]
        long_rows.append(
            [
                MODEL_LABELS[model_id], metric["samples"], metric["confusion"]["tp"],
                metric["confusion"]["fp"], metric["confusion"]["tn"], metric["confusion"]["fn"],
                number(metric["accuracy"]), number(metric["balanced_accuracy"]),
                number(metric["precision"]), number(metric["recall"]), number(metric["f1"]),
                number(metric["normal_fpr"]), number(metric["roc_auc"]), number(metric["pr_auc_ap"]),
            ]
        )
    strata_rows = []
    for model_id in MODEL_ORDER:
        for bucket, metric in long_strata[model_id].items():
            strata_rows.append(
                [
                    MODEL_LABELS[model_id], bucket, metric["samples"], metric["positive_samples"],
                    metric["negative_samples"], number(metric["recall"]),
                    number(metric["normal_fpr"]), number(metric["accuracy"]),
                    number(metric["balanced_accuracy"]),
                ]
            )

    transition_rows = []
    detail_sections: list[str] = []
    for baseline_id in ("m0_v361", "m1_e1b"):
        counter = Counter(row["transition"] for row in transitions[baseline_id])
        for transition, count in sorted(counter.items()):
            transition_rows.append([f"E2 vs {MODEL_LABELS[baseline_id]}", transition, count])
        material = [
            row for row in transitions[baseline_id]
            if row["transition"] in {"e2_new_hit", "e2_new_false_positive"}
        ]
        detail_sections.extend(
            [
                f"### E2 相对 {MODEL_LABELS[baseline_id]} 的新增命中/新增误报",
                "",
                markdown_table(
                    ["call_id", "decision", "场景", "变化", "baseline p", "E2 p", "tE", "tA"],
                    [
                        [
                            row["call_id"], row["decision"], row["scenario_tag"], row["transition"],
                            number(row["baseline_p_fraud"]), number(row["e2_p_fraud"]),
                            row["tE_turn"], row["tA_turn"],
                        ]
                        for row in material
                    ],
                ),
                "",
            ]
        )

    lines = [
        "# V365 Dev：M0 / M1 / E2-A 二分类对比",
        "",
        "固定阈值为0.5，诈骗分数只使用 `p_binary[1]`。ambiguous 不计入对错；Test 与 restricted 文件被读取前硬拒绝。",
        "",
        "## V365 Dev 完整文本",
        "",
        markdown_table(
            ["模型", "TP", "FP", "TN", "FN", "Acc", "BAcc", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "PR-AUC", "early", "explicit", "normal FP"],
            full_rows,
        ),
        "",
        "### ambiguous 分数（不判对错）",
        "",
        markdown_table(["模型", "N", "均值", "中位数", "P90", ">=0.5"], ambiguous_rows),
        "",
        "## V365 Dev 累计前缀时序",
        "",
        "报警轮次采用连续命中窗口的末轮；tE/tA 仅作评测边界。premature 与之后的有效检测可同时发生。",
        "",
        markdown_table(
            ["策略", "模型", "子集", "tE后检测", "tE前抢报", "normal曾误报", "有效检测延迟中位", "explicit在tA前"],
            temporal_rows,
        ),
        "",
        "## 原 V361 长文本 Dev",
        "",
        markdown_table(
            ["模型", "N", "TP", "FP", "TN", "FN", "Acc", "BAcc", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "PR-AUC"],
            long_rows,
        ),
        "",
        "### 长度分层",
        "",
        markdown_table(["模型", "length_bucket", "N", "fraud", "normal", "Recall", "FPR", "Acc", "BAcc"], strata_rows),
        "",
        "## E2-A 相对变化",
        "",
        markdown_table(["对比", "变化", "条数"], transition_rows),
        "",
        *detail_sections,
        "## 合规与复现",
        "",
        f"- V365 Dev SHA256：`{manifest['inputs']['v365_dev']['sha256']}`。",
        f"- 固定完整文本输入 SHA256：`{manifest['inputs']['v365_full_input']['sha256']}`。",
        f"- 固定累计前缀输入 SHA256：`{manifest['inputs']['v365_prefix_input']['sha256']}`。",
        f"- 原长文本 Dev SHA256：`{manifest['inputs']['v361_long_dev']['sha256']}`。",
        "- Test/restricted 文件读取数：0；全部输入先经过路径拒绝检查。",
        "- 没有训练、阈值选择或报警策略调参；本脚本只重算 Dev 指标。",
        "- `turn_stage_labels` 未用作逐前缀监督；只使用最终裁定的 tE/tA 定位评测边界。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-source", type=Path, required=True)
    parser.add_argument("--prefix-input", type=Path, required=True)
    parser.add_argument("--full-input", type=Path)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--e2-full-pred", type=Path, required=True)
    parser.add_argument("--e2-full-metrics", type=Path, required=True)
    parser.add_argument("--e2-prefix-pred", type=Path, required=True)
    parser.add_argument("--e2-prefix-metrics", type=Path, required=True)
    parser.add_argument("--long-dev-source", type=Path, required=True)
    for prefix in ("m0", "m1", "e2"):
        parser.add_argument(f"--{prefix}-long-pred", type=Path, required=True)
        parser.add_argument(f"--{prefix}-long-metrics", type=Path, required=True)
        parser.add_argument(f"--{prefix}-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = load_baseline_module()

    dev_source = require_safe_file(args.dev_source, suffix=EXPECTED_DEV_SUFFIX)
    long_source = require_safe_file(args.long_dev_source, suffix=EXPECTED_LONG_DEV_SUFFIX)
    baseline_root = require_safe_dir(args.baseline_root, suffix=EXPECTED_BASELINE_SUFFIX)
    prefix_input = require_safe_file(args.prefix_input)
    full_input = require_safe_file(
        args.full_input
        if args.full_input is not None
        else prefix_input.parent / "v365_dev_fulltext_label_blind.jsonl"
    )
    output_dir = require_safe_output(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        # Raw inference already lives below this output in the standard runner.
        allowed_top = {"logs", "raw_inference"}
        unexpected = [path for path in output_dir.iterdir() if path.name not in allowed_top]
        if unexpected:
            raise SystemExit(f"refusing to overwrite existing report outputs: {unexpected}")

    checkpoint_paths = {
        "m0_v361": require_safe_file(args.m0_checkpoint),
        "m1_e1b": require_safe_file(args.m1_checkpoint),
        "e2a": require_safe_file(args.e2_checkpoint),
    }
    for path in (
        args.e2_full_pred, args.e2_full_metrics, args.e2_prefix_pred,
        args.e2_prefix_metrics, args.m0_long_pred, args.m0_long_metrics,
        args.m1_long_pred, args.m1_long_metrics, args.e2_long_pred,
        args.e2_long_metrics,
    ):
        require_safe_file(path)

    dev_rows = baseline.validate_dev_source(dev_source)
    fixed_full_rows = read_jsonl_safe(full_input, baseline)
    prefix_rows = read_jsonl_safe(prefix_input, baseline)
    validate_fixed_full_input(dev_rows, fixed_full_rows, baseline)

    model_files: dict[str, dict[str, Path]] = {
        "m0_v361": {
            "full_pred": baseline_root / "raw_inference/m0_v361/fulltext/predictions.jsonl",
            "full_metrics": baseline_root / "raw_inference/m0_v361/fulltext/metrics.json",
            "prefix_pred": baseline_root / "raw_inference/m0_v361/cumulative_prefix/predictions.jsonl",
            "prefix_metrics": baseline_root / "raw_inference/m0_v361/cumulative_prefix/metrics.json",
            "long_pred": args.m0_long_pred,
            "long_metrics": args.m0_long_metrics,
        },
        "m1_e1b": {
            "full_pred": baseline_root / "raw_inference/m1_e1b/fulltext/predictions.jsonl",
            "full_metrics": baseline_root / "raw_inference/m1_e1b/fulltext/metrics.json",
            "prefix_pred": baseline_root / "raw_inference/m1_e1b/cumulative_prefix/predictions.jsonl",
            "prefix_metrics": baseline_root / "raw_inference/m1_e1b/cumulative_prefix/metrics.json",
            "long_pred": args.m1_long_pred,
            "long_metrics": args.m1_long_metrics,
        },
        "e2a": {
            "full_pred": args.e2_full_pred,
            "full_metrics": args.e2_full_metrics,
            "prefix_pred": args.e2_prefix_pred,
            "prefix_metrics": args.e2_prefix_metrics,
            "long_pred": args.e2_long_pred,
            "long_metrics": args.e2_long_metrics,
        },
    }
    for files in model_files.values():
        for path in files.values():
            require_safe_file(path)

    long_rows = read_jsonl_safe(long_source, baseline)
    validate_long_dev(long_rows)
    full_records: dict[str, list[dict[str, Any]]] = {}
    prefix_records: dict[str, list[dict[str, Any]]] = {}
    temporal_calls: dict[str, list[dict[str, Any]]] = {}
    full_metrics: dict[str, dict[str, Any]] = {}
    temporal_metrics: dict[str, dict[str, Any]] = {}
    long_by_model: dict[str, list[dict[str, Any]]] = {}
    long_metrics: dict[str, dict[str, Any]] = {}
    long_strata: dict[str, dict[str, dict[str, Any]]] = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_ORDER:
        files = model_files[model_id]
        checkpoint = checkpoint_paths[model_id]
        raw_full = read_jsonl_safe(files["full_pred"], baseline)
        raw_prefix = read_jsonl_safe(files["prefix_pred"], baseline)
        timing_full = baseline.raw_timing(files["full_metrics"], checkpoint, full_input)
        timing_prefix = baseline.raw_timing(files["prefix_metrics"], checkpoint, prefix_input)
        normalized_full = baseline.build_full_records(dev_rows, raw_full, model_id)
        metric_full = baseline.binary_metrics(normalized_full, timing_full)
        normalized_prefix, calls = baseline.build_prefix_records(
            dev_rows, prefix_rows, raw_prefix, model_id
        )
        metric_temporal = baseline.temporal_metrics(calls, timing_prefix)
        endpoint = baseline.endpoint_consistency(normalized_full, calls, model_id)
        metric_full["diagnostics"]["full_vs_final_prefix"] = endpoint
        metric_temporal["endpoint_consistency"] = endpoint

        raw_long = read_jsonl_safe(files["long_pred"], baseline)
        timing_long = baseline.raw_timing(files["long_metrics"], checkpoint, long_source)
        normalized_long = long_records(long_rows, raw_long, model_id, baseline)
        metric_long = generic_binary_metrics(normalized_long, timing_long)
        strata = length_strata(normalized_long)

        full_records[model_id] = normalized_full
        prefix_records[model_id] = normalized_prefix
        temporal_calls[model_id] = calls
        full_metrics[model_id] = metric_full
        temporal_metrics[model_id] = metric_temporal
        long_by_model[model_id] = normalized_long
        long_metrics[model_id] = metric_long
        long_strata[model_id] = strata

        model_out = output_dir / "scored" / model_id
        baseline.write_jsonl(model_out / "v365_fulltext_predictions.jsonl", normalized_full)
        baseline.write_jsonl(model_out / "v365_prefix_predictions.jsonl", normalized_prefix)
        baseline.write_jsonl(model_out / "v365_temporal_calls.jsonl", calls)
        baseline.write_json(model_out / "v365_fulltext_metrics.json", metric_full)
        baseline.write_json(model_out / "v365_temporal_metrics.json", metric_temporal)
        baseline.write_jsonl(model_out / "v361_longdev_predictions.jsonl", normalized_long)
        baseline.write_json(model_out / "v361_longdev_metrics.json", metric_long)
        baseline.write_json(model_out / "v361_longdev_length_strata.json", strata)

    # Length/chunk diagnostics must be input-invariant across checkpoints.
    reference = {
        row["sample_id"]: (row["raw_wordpiece_length"], row["n_chunks"])
        for row in long_by_model["m0_v361"]
    }
    for model_id in ("m1_e1b", "e2a"):
        observed = {
            row["sample_id"]: (row["raw_wordpiece_length"], row["n_chunks"])
            for row in long_by_model[model_id]
        }
        if observed != reference:
            raise ValueError(f"{model_id}: long Dev chunking differs from M0")

    transitions = {
        baseline_id: compare_e2(
            dev_rows, full_records[baseline_id], full_records["e2a"], baseline_id
        )
        for baseline_id in ("m0_v361", "m1_e1b")
    }
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for baseline_id, rows in transitions.items():
        baseline.write_jsonl(reports / f"e2_vs_{baseline_id}_transitions.jsonl", rows)
        write_csv(reports / f"e2_vs_{baseline_id}_transitions.csv", rows)
        baseline.write_jsonl(
            reports / f"e2_vs_{baseline_id}_new_hits.jsonl",
            [row for row in rows if row["transition"] == "e2_new_hit"],
        )
        baseline.write_jsonl(
            reports / f"e2_vs_{baseline_id}_new_false_positives.jsonl",
            [row for row in rows if row["transition"] == "e2_new_false_positive"],
        )

    full_csv = [flatten_full_metrics(model_id, full_metrics[model_id]) for model_id in MODEL_ORDER]
    temporal_csv = [
        row for model_id in MODEL_ORDER for row in flatten_temporal(model_id, temporal_metrics[model_id])
    ]
    long_csv: list[dict[str, Any]] = []
    length_csv: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        metric = long_metrics[model_id]
        long_csv.append(
            {
                "model_id": model_id,
                **metric["confusion"],
                "samples": metric["samples"],
                "accuracy": metric["accuracy"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "normal_fpr": metric["normal_fpr"],
                "roc_auc": metric["roc_auc"],
                "pr_auc_ap": metric["pr_auc_ap"],
            }
        )
        for bucket, block in long_strata[model_id].items():
            length_csv.append(
                {
                    "model_id": model_id,
                    "length_bucket": bucket,
                    "samples": block["samples"],
                    "positive_samples": block["positive_samples"],
                    "negative_samples": block["negative_samples"],
                    **block["confusion"],
                    "accuracy": block["accuracy"],
                    "balanced_accuracy": block["balanced_accuracy"],
                    "precision": block["precision"],
                    "recall": block["recall"],
                    "f1": block["f1"],
                    "normal_fpr": block["normal_fpr"],
                }
            )
    write_csv(reports / "v365_fulltext_model_comparison.csv", full_csv)
    write_csv(reports / "v365_temporal_model_comparison.csv", temporal_csv)
    write_csv(reports / "v361_longdev_model_comparison.csv", long_csv)
    write_csv(reports / "v361_longdev_length_strata.csv", length_csv)

    manifest = {
        "schema_version": "v365.e2a.dev_compare.v1",
        "threshold": THRESHOLD,
        "score": "p_binary[1]",
        "models": {
            model_id: {
                "checkpoint": str(checkpoint_paths[model_id]),
                "checkpoint_sha256": sha256_file(checkpoint_paths[model_id]),
            }
            for model_id in MODEL_ORDER
        },
        "inputs": {
            "v365_dev": {"path": str(dev_source), "sha256": sha256_file(dev_source)},
            "v365_full_input": {"path": str(full_input), "sha256": sha256_file(full_input)},
            "v365_prefix_input": {"path": str(prefix_input), "sha256": sha256_file(prefix_input)},
            "v361_long_dev": {"path": str(long_source), "sha256": sha256_file(long_source)},
        },
        "safety": {
            "training_performed_by_report": False,
            "threshold_or_policy_tuned": False,
            "test_or_restricted_files_opened": 0,
            "all_read_paths_checked_before_open": True,
            "turn_stage_labels_used_as_supervision": False,
        },
    }
    comparison = {
        "manifest": manifest,
        "v365_fulltext": full_metrics,
        "v365_temporal": temporal_metrics,
        "v361_longdev": long_metrics,
        "v361_longdev_length_strata": long_strata,
        "e2_transition_counts": {
            baseline_id: dict(Counter(row["transition"] for row in rows))
            for baseline_id, rows in transitions.items()
        },
    }
    baseline.write_json(output_dir / "run_manifest.json", manifest)
    baseline.write_json(reports / "model_comparison.json", comparison)
    report = build_report(
        full_metrics, temporal_metrics, long_metrics, long_strata, transitions, manifest
    )
    (reports / "V365_E2A_DEV_COMPARISON_REPORT.md").write_text(
        report + "\n", encoding="utf-8"
    )

    forbidden_weights = [
        str(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".bin", ".safetensors"}
    ]
    if forbidden_weights:
        raise AssertionError(f"report output unexpectedly contains model weights: {forbidden_weights}")
    print(
        json.dumps(
            {
                "status": "done",
                "report": str(reports / "V365_E2A_DEV_COMPARISON_REPORT.md"),
                "e2_transition_counts": comparison["e2_transition_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
