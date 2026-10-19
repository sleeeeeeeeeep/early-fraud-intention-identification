#!/usr/bin/env python3
"""汇总 V361 的 40 条 Luna 冒烟生成与独立审核质量。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    rows: list[dict[str, Any]] = []
    for item in paths:
        rows.extend(read_jsonl(item))
    return rows


def grouped_ready(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for value, group in sorted(groups.items()):
        ready = sum(row.get("automatic_validation") == "READY" for row in group)
        result[value] = {
            "rows": len(group),
            "ready": ready,
            "ready_rate": ready / max(len(group), 1),
        }
    return result


def build_report(
    candidates: list[dict[str, Any]], reviews: list[dict[str, Any]]
) -> dict[str, Any]:
    review_by_id = {
        str(row.get("task_id")): row for row in reviews if row.get("task_id")
    }
    ready_rows = [
        row for row in candidates if row.get("automatic_validation") == "READY"
    ]
    reviewed_ready = [
        (row, review_by_id.get(str(row["task_id"]))) for row in ready_rows
    ]
    accepted = sum(
        review is not None
        and str(review.get("decision") or "").lower().startswith("accept")
        for _, review in reviewed_ready
    )
    error_counts = Counter(
        error
        for row in candidates
        for error in row.get("automatic_validation_errors", [])
    )
    length = grouped_ready(candidates, "length_bucket")
    scenario = grouped_ready(candidates, "scenario_tag")
    automatic_gate = {
        "all_40_results_present": len(candidates) == 40,
        "automatic_ready_at_least_32": len(ready_rows) >= 32,
        "each_length_bucket_ready_rate_at_least_70pct": bool(length)
        and all(float(item["ready_rate"]) >= 0.70 for item in length.values()),
        "role_marker_policy_100pct": len(candidates) == 40
        and all(
            row.get("role_marker_token_policy")
            == "excluded_from_model_input_and_wordpiece_counts"
            for row in candidates
        ),
        "full_text_coverage_100pct": len(candidates) == 40
        and all(float(row.get("full_text_coverage", 0.0)) == 1.0 for row in candidates),
    }
    review_gate = {
        "every_ready_candidate_reviewed": len(ready_rows) > 0
        and all(review is not None for _, review in reviewed_ready),
        "accepted_ready_rate_at_least_90pct": accepted
        / max(len(ready_rows), 1)
        >= 0.90,
        "review_label_pass_rate_at_least_90pct": sum(
            bool(review and review.get("label_pass")) for _, review in reviewed_ready
        )
        / max(len(ready_rows), 1)
        >= 0.90,
        "review_role_logic_pass_rate_at_least_90pct": sum(
            bool(review and review.get("role_logic_pass"))
            for _, review in reviewed_ready
        )
        / max(len(ready_rows), 1)
        >= 0.90,
    }
    return {
        "schema_version": "v361.longdoc_smoke_quality.v1",
        "generated_rows": len(candidates),
        "automatic_ready": len(ready_rows),
        "automatic_ready_rate": len(ready_rows) / max(len(candidates), 1),
        "review_rows": len(reviews),
        "review_accepted": accepted,
        "review_accept_rate_over_ready": accepted / max(len(ready_rows), 1),
        "automatic_errors": dict(error_counts.most_common()),
        "by_generation_kind": grouped_ready(candidates, "generation_kind"),
        "by_length_bucket": length,
        "by_scenario": scenario,
        "automatic_gate": automatic_gate,
        "review_gate": review_gate,
        "automatic_gate_passed": all(automatic_gate.values()),
        "review_gate_passed": all(review_gate.values()),
        "full_smoke_gate_passed": all(automatic_gate.values())
        and all(review_gate.values()),
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V361 Luna 长文本 40 条冒烟质量报告",
        "",
        "| 项目 | 结果 |",
        "|---|---:|",
        f"| 生成结果 | {report['generated_rows']} |",
        f"| 自动 READY | {report['automatic_ready']} ({100 * report['automatic_ready_rate']:.2f}%) |",
        f"| 独立审核 | {report['review_rows']} |",
        f"| 审核接受 | {report['review_accepted']} ({100 * report['review_accept_rate_over_ready']:.2f}%) |",
        "",
        f"最终冒烟门：{'通过' if report['full_smoke_gate_passed'] else '未通过'}。",
        "",
        "## 自动失败原因",
        "",
    ]
    if report["automatic_errors"]:
        lines.extend(
            f"- `{name}`：{count} 条"
            for name, count in report["automatic_errors"].items()
        )
    else:
        lines.append("- 无")
    lines.extend(["", "## 分长度结果", "", "| 长度档 | 数量 | READY | READY率 |", "|---|---:|---:|---:|"])
    for name, item in report["by_length_bucket"].items():
        lines.append(
            f"| {name} | {item['rows']} | {item['ready']} | {100 * item['ready_rate']:.2f}% |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(read_jsonl(args.candidates), collect_jsonl(args.reviews))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
