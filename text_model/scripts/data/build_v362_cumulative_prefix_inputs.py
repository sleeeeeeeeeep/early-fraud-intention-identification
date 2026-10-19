#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V362 第三阶段（时间检测评测）的逐轮前缀输入构建。

对 ``prefix_dev`` / ``prefix_stress`` 的每条样本，从第 1 轮切到它自己的
``prefix_end_turn``，每一刀产出一行推理输入。跑完推理后，聚合脚本按
``sample_id`` 把同一通电话的几十个点拼回一条概率时间曲线，用于扫描
"模型第一次报警发生在第几轮"。

与 ``build_v362_prefix_eval_inputs.py`` 的关系
----------------------------------------------
复用同一套严格前缀校验，只是把"切一刀"换成"切 N 刀"：

* 第 k 刀写出的文本必须是原始 ``model_input_text`` 的**逐字节前缀**；
* 第 k 刀必须满足 ``1 <= k <= prefix_end_turn``，**绝不越过人工给定的切点**；
* 真值不变：``label_13`` / ``subcategory_code`` 沿用该通话的完整标签，
  评测的"可见信息量"是唯一变量。

两个必须注意的实现细节
----------------------
1. **轮次只能编码进 sample_id**。推理入口
   (``src/v361/inference/run_longdoc_inference.py``) 写出的 metadata 是
   白名单字段（见其 ``record = dict(metadata)`` 上游的 collator），
   自定义字段 ``cut_turn`` 会被丢弃。因此本脚本把轮次写进
   ``sample_id``，格式 ``<原始 sample_id>@@t<三位零填充轮次>``，
   由聚合脚本解析回来。``cut_turn`` 字段仍然写出，仅作人工核对用。
2. **输出目录独立**。绝不覆盖 ``prefix_eval_{dev,stress}.jsonl``，
   也不覆盖任何已有 runs。

用法::

    python build_v362_temporal_eval_inputs.py \
        --prefix-dir  data/processed/v362_early_intent \
        --source      data/processed/v361_longdoc_text_route_20260824/generated_long_all_v361_980.jsonl \
        --output-dir  runs/e0_v362_temporal/inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PREFIX_DIR = Path("data/processed/v362_early_intent")
DEFAULT_SOURCE = Path(
    "data/processed/v361_longdoc_text_route_20260824/generated_long_all_v361_980.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("runs/v362_e0_binary/inputs")

#: sample_id 中编码轮次的分隔符。原始 sample_id 形如 ``V361_LONG_dev_0000``，
#: 不含 ``@@``，因此可以无歧义地 rsplit。
TURN_SEPARATOR = "@@t"
TURN_WIDTH = 3

#: 从原始 980 行透传的静态样本属性，保证与全文/前缀基线的分层统计口径一致。
PASSTHROUGH_FIELDS = (
    "split",
    "generation_kind",
    "scenario_tag",
    "claimed_identity_type",
    "caller_number_type",
    "length_bucket",
    "evidence_position_bin",
    "cross_window_evidence",
    "content_group",
    "review_status",
    "longdoc_generated",
    "input_view",
    "canonical_label_13",
    "canonical_subcategory_code",
)

#: 前缀数据集自身字段，供聚合脚本做分析（推理输出会丢弃，仅在本文件中保留）。
PREFIX_FIELDS = (
    "decision",
    "stage_target",
    "binary_target",
    "binary_loss_mask",
    "stage_loss_mask",
    "sample_weight",
    "prefix_relation",
    "tE_turn",
    "early_detectable",
    "original_full_binary_label",
    "prefix_end_turn",
    "n_prefix_turns",
    "n_full_turns",
    "prefix_char_length",
    "prefix_wordpiece_length",
    "confidence",
    "review_required",
    "annotation_file",
)

#: 描述完整文本、对前缀行无效的字段，禁止透传。
FORBIDDEN_FIELDS = (
    "turns",
    "text",
    "audio_text",
    "model_input_text",
    "chunk_token_ranges",
    "evidence_token_ranges",
    "context_evidence_token_ranges",
    "raw_wordpiece_length",
    "full_text_coverage",
    "chunking_profile",
    "prefix_text",
    "prefix_turns",
)


class TemporalInputError(RuntimeError):
    """构建逐轮前缀评测输入时的校验失败。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TemporalInputError(f"{path}:{line_number}: 期望 JSON 对象")
            rows.append(row)
    if not rows:
        raise TemporalInputError(f"空数据集: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_text_from_turns(prefix_turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(turn["text"]) for turn in prefix_turns)


def temporal_sample_id(sample_id: str, cut_turn: int) -> str:
    """把轮次编码进 sample_id —— 推理入口的 metadata 白名单会丢弃自定义字段。"""
    return f"{sample_id}{TURN_SEPARATOR}{cut_turn:0{TURN_WIDTH}d}"


def build_rows(
    prefix_rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    *,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set()
    built: list[dict[str, Any]] = []
    per_sample_cuts: dict[str, int] = {}
    decision_counter: Counter[str] = Counter()

    for row in prefix_rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise TemporalInputError(f"{split_name}: sample_id 重复: {sample_id}")
        seen.add(sample_id)
        if sample_id not in source_rows:
            raise TemporalInputError(f"{split_name}: 原始数据中不存在 {sample_id}")
        source = source_rows[sample_id]

        prefix_turns = row["prefix_turns"]
        declared = int(row["prefix_end_turn"])
        if declared != len(prefix_turns):
            raise TemporalInputError(
                f"{sample_id}: prefix_end_turn={declared} 与 prefix_turns 长度 "
                f"{len(prefix_turns)} 不一致"
            )
        if declared < 1:
            raise TemporalInputError(f"{sample_id}: prefix_end_turn={declared} < 1")
        full_turns = source.get("turns") or []
        if declared > len(full_turns):
            raise TemporalInputError(
                f"{sample_id}: prefix_end_turn={declared} 超出完整轮次数 {len(full_turns)}"
            )

        # 前缀轮次必须与原始完整轮次逐字一致（严格盲前缀）。
        for index, turn in enumerate(prefix_turns):
            original = full_turns[index]
            if str(turn["text"]) != str(original["text"]):
                raise TemporalInputError(
                    f"{sample_id}: 第 {index + 1} 轮文本与原文不一致"
                )
            if int(turn.get("turn_id", index + 1)) != index + 1:
                raise TemporalInputError(f"{sample_id}: turn_id 非连续")

        text_from_turns = prefix_text_from_turns(prefix_turns)
        if str(row.get("prefix_text") or "") != text_from_turns:
            raise TemporalInputError(f"{sample_id}: prefix_text 与 prefix_turns 拼接不一致")

        # 核心断言：整条前缀必须是原始 model_input_text 的逐字节前缀。
        full_model_input = str(source.get("model_input_text") or "")
        if not full_model_input.startswith(text_from_turns):
            raise TemporalInputError(
                f"{sample_id}: 前缀不是原始 model_input_text 的逐字节前缀"
            )

        decision = str(row.get("decision") or "")
        decision_counter[decision] += 1

        base: dict[str, Any] = {}
        for field in PASSTHROUGH_FIELDS:
            if field in source and source[field] is not None:
                base[field] = source[field]
            elif field in row and row[field] is not None:
                base[field] = row[field]
        for field in PREFIX_FIELDS:
            if field in row:
                base[field] = row[field]
        for field in FORBIDDEN_FIELDS:
            base.pop(field, None)
        leaked = sorted(set(base) & set(FORBIDDEN_FIELDS))
        if leaked:
            raise TemporalInputError(f"{sample_id}: 透传了完整文本派生字段 {leaked}")

        if int(row["label_13"]) != 0 and not str(row["subcategory_code"]).strip():
            raise TemporalInputError(f"{sample_id}: 欺诈行缺少 subcategory_code")

        # ---- 逐轮切片：k = 1 .. declared，绝不越过 declared ----
        for k in range(1, declared + 1):
            cut_text = prefix_text_from_turns(prefix_turns[:k])
            # 累加前缀的逐字节断言（传递性上成立，仍显式校验一次）。
            if not full_model_input.startswith(cut_text):
                raise TemporalInputError(
                    f"{sample_id} 切到第 {k} 轮: 不是原始 model_input_text 的前缀"
                )
            if not text_from_turns.startswith(cut_text):
                raise TemporalInputError(
                    f"{sample_id} 切到第 {k} 轮: 不是人工前缀的前缀"
                )
            # 绝不越过人工切点：第 k 刀的文本必须短于或等于整条人工前缀。
            if len(cut_text) > len(text_from_turns):
                raise TemporalInputError(f"{sample_id}: 第 {k} 刀越过了 prefix_end_turn")

            payload = dict(base)
            payload.update(
                {
                    "sample_id": temporal_sample_id(sample_id, k),
                    "source_sample_id": sample_id,
                    "cut_turn": k,
                    "label_13": int(row["label_13"]),
                    "subcategory_code": str(row["subcategory_code"]),
                    "model_input_text": cut_text,
                    "eval_view": "temporal_prefix",
                    "eval_split": split_name,
                }
            )
            built.append(payload)

        per_sample_cuts[sample_id] = declared

    if not built:
        raise TemporalInputError(f"{split_name}: 未生成任何行")

    cuts = list(per_sample_cuts.values())
    stats = {
        "samples": len(per_sample_cuts),
        "rows": len(built),
        "cut_turns_min": min(cuts),
        "cut_turns_max": max(cuts),
        "cut_turns_mean": round(sum(cuts) / len(cuts), 4),
        "cut_turns_total": sum(cuts),
        "distinct_cut_turns": len(set(cuts)),
        "decision_counts": dict(sorted(decision_counter.items())),
    }
    return built, stats


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-dir", type=Path, default=DEFAULT_PREFIX_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", default="dev,stress", help="要构建的 split，逗号分隔")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = {str(row["sample_id"]): row for row in read_jsonl(args.source)}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "v362.temporal_eval_input.v1",
        "purpose": "时间检测评测：逐轮累计前缀，用于扫描首次报警轮次",
        "turn_encoding": f"sample_id = <原 sample_id>{TURN_SEPARATOR}<轮次>",
        "threshold_policy": "评测阈值固定 0.5，判定用概率阈值而非 argmax",
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "splits": {},
    }
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        prefix_path = args.prefix_dir / f"prefix_{split}.jsonl"
        prefix_rows = read_jsonl(prefix_path)
        built, stats = build_rows(prefix_rows, source_rows, split_name=split)
        out_path = args.output_dir / f"cumulative_prefix_{split}.jsonl"
        write_jsonl(out_path, built)

        manifest["splits"][split] = {
            "prefix_file": str(prefix_path),
            "prefix_sha256": sha256_file(prefix_path),
            "output_file": str(out_path),
            "output_sha256": sha256_file(out_path),
            "strict_prefix_verified": True,
            "never_exceeds_prefix_end_turn": True,
            "duplicate_sample_ids": 0,
            **stats,
        }
        print(
            f"[{split}] {stats['samples']} 条样本 -> {stats['rows']} 行逐轮输入 "
            f"(切点 {stats['cut_turns_min']}~{stats['cut_turns_max']}, "
            f"均值 {stats['cut_turns_mean']}) -> {out_path}"
        )
        print(f"        decision 分布: {stats['decision_counts']}")

    manifest_path = args.output_dir / "cumulative_prefix_inputs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(s["rows"] for s in manifest["splits"].values())
    print(f"合计 {total} 行 -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
