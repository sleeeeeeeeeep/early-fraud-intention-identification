#!/usr/bin/env python3
"""为自动通过的长文本候选构建独立 Luna 子代理审核任务。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATES = ROOT / "data/processed/v361_longdoc_text_route_20260824/automatic_ready.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/v361_longdoc_subagent_review_tasks_20260824"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def prompt(row: Mapping[str, Any]) -> str:
    return f"""你是独立数据审核员。审核下面的中文长电话，只输出一个JSON对象。

task_id：{row['task_id']}
目标父类：{row['label_13']}
目标子类：{row['subcategory_code']}
目标场景：{row['scenario_tag']}
样本类型：{row['generation_kind']}
证据位置：{row['evidence_position_bin']}

对话：
{row['text']}

审核标准：
1. 对话自然连续，身份、称呼、时间和业务事实一致，没有机械复读。
2. 诈骗样本由来电人推进明确可执行风险动作，标签与最终完整机制一致。
3. 正常样本形成合法闭环，引用、否定和反诈提醒不会被当成来电人风险要求。
4. 风险动作角色、否定、引用和反问关系清楚。
5. 私人号码只作为元数据，不应在正文中直接念出。

输出：
{{"task_id":"{row['task_id']}","decision":"accept或reject","naturalness_pass":true,"label_pass":true,"role_logic_pass":true,"reason":"不超过60字"}}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-size", type=int, default=10)
    args = parser.parse_args()
    rows = read_jsonl(args.candidates)
    tasks = [
        {
            "schema_version": "v361.longdoc_review_task.v1",
            "task_id": row["task_id"],
            "slot_id": row["slot_id"],
            "review_model": "gpt-5.6-luna",
            "reasoning_effort": "medium",
            "prompt": prompt(row),
        }
        for row in rows
    ]
    write_jsonl(args.output_dir / "all_review_tasks.jsonl", tasks)
    shard_paths: list[str] = []
    for index, start in enumerate(range(0, len(tasks), args.shard_size)):
        path = args.output_dir / "shards" / f"review_{index:03d}.jsonl"
        write_jsonl(path, tasks[start : start + args.shard_size])
        shard_paths.append(str(path))
    manifest = {
        "schema_version": "v361.longdoc_review_manifest.v1",
        "rows": len(tasks),
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "http_api_used": False,
        "shards": shard_paths,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
