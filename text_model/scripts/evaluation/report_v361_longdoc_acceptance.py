#!/usr/bin/env python3
"""根据逐条预测生成 V361 文本长上下文验收与路线选择报告。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from text_model.training.longdoc_metrics import classification_metrics
except ModuleNotFoundError:
    from training.longdoc_metrics import classification_metrics


BASELINE_DEV_ACCURACY = 0.9228
BASELINE_DEV_MACRO_F1 = 0.9255
TARGET_SCENARIOS = (
    "douyin_live_membership",
    "million_protection",
    "business_license_annual_review",
    "private_number_douyin_official",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    values = list(rows)
    return classification_metrics(
        [int(row["label_13"]) for row in values],
        [int(row["prediction_13"]) for row in values],
        13,
    )


def group(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    output: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row.get(key) or "unknown")].append(row)
    return dict(output)


def route_summary(
    dev_rows: list[dict[str, Any]], long_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    scenario_groups = group(
        [row for row in long_rows if str(row.get("scenario_tag")) in TARGET_SCENARIOS],
        "scenario_tag",
    )
    target_rows = [
        row for row in long_rows if str(row.get("scenario_tag")) in TARGET_SCENARIOS
    ]
    q4_rows = [
        row for row in long_rows if str(row.get("evidence_position_bin")) == "Q4"
    ]
    cross_rows = [row for row in long_rows if bool(row.get("cross_window_evidence"))]
    return {
        "dev_a2": metrics(dev_rows),
        "long_dev": metrics(long_rows),
        "q4": metrics(q4_rows),
        "cross_window": metrics(cross_rows),
        "target_scenarios_combined": metrics(target_rows),
        "target_scenarios": {
            scenario: metrics(scenario_groups.get(scenario, []))
            for scenario in TARGET_SCENARIOS
        },
        "coverage_min": min(
            (float(row.get("full_text_coverage", 0.0)) for row in dev_rows + long_rows),
            default=0.0,
        ),
    }


def acceptance(
    candidate: Mapping[str, Any], baseline_long: Mapping[str, Any]
) -> dict[str, Any]:
    dev = candidate["dev_a2"]
    long_dev = candidate["long_dev"]
    q4 = candidate["q4"]
    baseline_q4 = baseline_long["q4"]
    scenario = candidate["target_scenarios"]
    checks = {
        "dev_accuracy_regression_le_0_3pp": float(dev["accuracy"])
        >= BASELINE_DEV_ACCURACY - 0.003,
        "dev_macro_f1_regression_le_0_3pp": float(dev["macro_f1"])
        >= BASELINE_DEV_MACRO_F1 - 0.003,
        "long_fraud_recall_gain_ge_5pp": float(long_dev["fraud_recall"])
        >= float(baseline_long["long_dev"]["fraud_recall"]) + 0.05,
        "q4_fraud_recall_gain_ge_10pp": float(q4["fraud_recall"])
        >= float(baseline_q4["fraud_recall"]) + 0.10,
        "four_scenario_combined_recall_ge_95pct": float(
            candidate["target_scenarios_combined"]["fraud_recall"]
        )
        >= 0.95,
        "each_scenario_recall_ge_90pct": all(
            float(scenario[name]["fraud_recall"]) >= 0.90
            for name in TARGET_SCENARIOS
        ),
        "each_scenario_label_accuracy_ge_90pct": all(
            float(scenario[name]["accuracy"]) >= 0.90 for name in TARGET_SCENARIOS
        ),
        "long_normal_fpr_increase_le_1pp": float(long_dev["normal_fpr"])
        <= float(baseline_long["long_dev"]["normal_fpr"]) + 0.01,
        "full_text_coverage_100pct": float(candidate["coverage_min"]) == 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks}


def choose_route(route_a: Mapping[str, Any], route_b: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "long_dev_macro_f1_gain_ge_0_7pp": float(route_a["long_dev"]["macro_f1"])
        >= float(route_b["long_dev"]["macro_f1"]) + 0.007,
        "q4_recall_gain_ge_1_5pp": float(route_a["q4"]["fraud_recall"])
        >= float(route_b["q4"]["fraud_recall"]) + 0.015,
        "cross_window_accuracy_gain_ge_2pp": float(
            route_a["cross_window"]["accuracy"]
        )
        >= float(route_b["cross_window"]["accuracy"]) + 0.02,
        "normal_fpr_gap_le_0_3pp": abs(
            float(route_a["long_dev"]["normal_fpr"])
            - float(route_b["long_dev"]["normal_fpr"])
        )
        <= 0.003,
    }
    passed = sum(checks.values())
    return {
        "selected_route": "A" if passed >= 2 else "B",
        "criteria_passed": passed,
        "checks": checks,
    }


def pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["# V361 文本长上下文验收报告", ""]
    for name in ("route_a", "route_b"):
        if name not in report:
            continue
        summary = report[name]["summary"]
        gate = report[name]["acceptance"]
        lines.extend(
            [
                f"## {name.replace('_', ' ').upper()}",
                "",
                "| 指标 | Dev-A2 | Long-Dev | Q4 | 跨窗口 |",
                "|---|---:|---:|---:|---:|",
                "| Accuracy | "
                + " | ".join(
                    pct(summary[key]["accuracy"])
                    for key in ("dev_a2", "long_dev", "q4", "cross_window")
                )
                + " |",
                "| Macro-F1 | "
                + " | ".join(
                    pct(summary[key]["macro_f1"])
                    for key in ("dev_a2", "long_dev", "q4", "cross_window")
                )
                + " |",
                "| Fraud recall | "
                + " | ".join(
                    pct(summary[key]["fraud_recall"])
                    for key in ("dev_a2", "long_dev", "q4", "cross_window")
                )
                + " |",
                "",
                f"验收结论：{'通过' if gate['passed'] else '未通过'}。",
                "",
            ]
        )
        for key, value in gate["checks"].items():
            lines.append(f"- {'通过' if value else '未通过'}：`{key}`")
        lines.append("")
    if "route_choice" in report:
        choice = report["route_choice"]
        lines.extend(
            [
                "## 路线选择",
                "",
                f"选择路线 {choice['selected_route']}，4 项规则通过 {choice['criteria_passed']} 项。",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-long", type=Path, required=True)
    parser.add_argument("--route-a-dev", type=Path)
    parser.add_argument("--route-a-long", type=Path)
    parser.add_argument("--route-b-dev", type=Path)
    parser.add_argument("--route-b-long", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    baseline_rows = read_jsonl(args.baseline_long)
    baseline_summary = route_summary([], baseline_rows)
    report: dict[str, Any] = {"baseline_long": baseline_summary}
    for key, dev_path, long_path in (
        ("route_a", args.route_a_dev, args.route_a_long),
        ("route_b", args.route_b_dev, args.route_b_long),
    ):
        if (dev_path is None) != (long_path is None):
            raise ValueError(f"{key} requires both Dev-A2 and Long-Dev predictions")
        if dev_path is None:
            continue
        summary = route_summary(read_jsonl(dev_path), read_jsonl(long_path))
        report[key] = {
            "summary": summary,
            "acceptance": acceptance(summary, baseline_summary),
        }
    if "route_a" in report and "route_b" in report:
        report["route_choice"] = choose_route(
            report["route_a"]["summary"], report["route_b"]["summary"]
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
