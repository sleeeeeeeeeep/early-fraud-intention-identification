#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0 前缀基线的评测输入构建。

把 V362 前缀数据集转换为长文本模型评测入口
（``src/v361/inference/run_longdoc_inference.py``）可消费的 JSONL 行。

设计要点
--------
1. **严格前缀**：写入 ``model_input_text`` 的文本必须逐字节等于原始
   980 数据 ``model_input_text`` 的前缀。脚本对每条样本做该断言，
   不满足即失败，避免基线与全文基线之间混入格式偏差。
2. **真值不变**：``label_13`` / ``subcategory_code`` 沿用该通话的原始
   完整标签，使前缀基线与全文基线只有"可见信息量"一个变量，
   从而可以量化截断带来的信息损失，作为 E1/E2 必须超越的下界。
3. **不携带全文派生字段**：``turns`` / ``chunk_token_ranges`` /
   ``evidence_token_ranges`` / ``raw_wordpiece_length`` /
   ``full_text_coverage`` / ``chunking_profile`` 描述的是完整文本，
   对前缀行无效，一律不写出，由 chunker 按前缀文本重新计算。

用法::

    python build_v362_prefix_eval_inputs.py \
        --prefix-dir  data/processed/v362_early_intent \
        --source      data/processed/v361_longdoc_text_route_20260824/generated_long_all_v361_980.jsonl \
        --output-dir  runs/e0_v362_prefix_baseline/inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PREFIX_DIR = Path("data/processed/v362_early_intent")
DEFAULT_SOURCE = Path(
    "data/processed/v361_longdoc_text_route_20260824/generated_long_all_v361_980.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("runs/v362_e0_binary/inputs")

#: 从原始 980 行透传的静态样本属性，保证分层统计口径与全文基线一致。
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

#: 前缀数据集自身的字段，供后续按 decision / 盲标真值做二次分析。
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
    "prefix_text_hash",
    "normalized_text_hash",
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


class EvalInputError(RuntimeError):
    """构建前缀评测输入时的校验失败。"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise EvalInputError(f"{path}:{line_number}: 期望 JSON 对象")
            rows.append(row)
    if not rows:
        raise EvalInputError(f"空数据集: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prefix_text_from_turns(prefix_turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(turn["text"]) for turn in prefix_turns)


def build_rows(
    prefix_rows: Sequence[Mapping[str, Any]],
    source_rows: Mapping[str, Mapping[str, Any]],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    built: list[dict[str, Any]] = []
    for row in prefix_rows:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise EvalInputError(f"{split_name}: sample_id 重复: {sample_id}")
        seen.add(sample_id)
        if sample_id not in source_rows:
            raise EvalInputError(f"{split_name}: 原始数据中不存在 {sample_id}")
        source = source_rows[sample_id]

        prefix_turns = row["prefix_turns"]
        declared = int(row["prefix_end_turn"])
        if declared != len(prefix_turns):
            raise EvalInputError(
                f"{sample_id}: prefix_end_turn={declared} 与 prefix_turns 长度 "
                f"{len(prefix_turns)} 不一致"
            )
        full_turns = source.get("turns") or []
        if declared > len(full_turns):
            raise EvalInputError(
                f"{sample_id}: prefix_end_turn={declared} 超出完整轮次数 {len(full_turns)}"
            )

        # 前缀轮次必须与原始完整轮次逐字一致（严格盲前缀，不多切也不少切）。
        for index, turn in enumerate(prefix_turns):
            original = full_turns[index]
            if str(turn["text"]) != str(original["text"]):
                raise EvalInputError(
                    f"{sample_id}: 第 {index + 1} 轮文本与原文不一致"
                )
            if int(turn.get("turn_id", index + 1)) != index + 1:
                raise EvalInputError(f"{sample_id}: turn_id 非连续")

        text_from_turns = prefix_text_from_turns(prefix_turns)
        if str(row.get("prefix_text") or "") != text_from_turns:
            raise EvalInputError(f"{sample_id}: prefix_text 与 prefix_turns 拼接不一致")
        if int(row.get("prefix_char_length", -1)) != len(text_from_turns):
            raise EvalInputError(f"{sample_id}: prefix_char_length 与实际长度不一致")

        # 核心断言：前缀必须是原始 model_input_text 的逐字节前缀。
        full_model_input = str(source.get("model_input_text") or "")
        if not full_model_input.startswith(text_from_turns):
            raise EvalInputError(
                f"{sample_id}: 前缀不是原始 model_input_text 的逐字节前缀"
            )

        payload: dict[str, Any] = {}
        for field in PASSTHROUGH_FIELDS:
            if field in source and source[field] is not None:
                payload[field] = source[field]
            elif field in row and row[field] is not None:
                payload[field] = row[field]
        for field in PREFIX_FIELDS:
            if field in row:
                payload[field] = row[field]
        # 先清掉全文派生的字段，再写入前缀文本，避免被误删。
        for field in FORBIDDEN_FIELDS:
            payload.pop(field, None)
        leaked = sorted(set(payload) & set(FORBIDDEN_FIELDS))
        if leaked:
            raise EvalInputError(f"{sample_id}: 透传了完整文本派生字段 {leaked}")

        payload.update(
            {
                "sample_id": sample_id,
                "label_13": int(row["label_13"]),
                "subcategory_code": str(row["subcategory_code"]),
                "model_input_text": text_from_turns,
                "eval_view": "prefix",
                "eval_split": split_name,
            }
        )

        # 欺诈行必须带有可解析的官方子类，否则 collator 会直接报错。
        if int(row["label_13"]) != 0 and not str(row["subcategory_code"]).strip():
            raise EvalInputError(f"{sample_id}: 欺诈行缺少 subcategory_code")
        built.append(payload)
    return built


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
    parser.add_argument(
        "--splits", default="dev,stress", help="要构建的 split，逗号分隔"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = {
        str(row["sample_id"]): row for row in read_jsonl(args.source)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "v362.e0_prefix_eval_input.v1",
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "splits": {},
    }
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        prefix_path = args.prefix_dir / f"prefix_{split}.jsonl"
        prefix_rows = read_jsonl(prefix_path)
        built = build_rows(prefix_rows, source_rows, split_name=split)
        out_path = args.output_dir / f"strict_prefix_{split}.jsonl"
        write_jsonl(out_path, built)

        labels = [int(row["label_13"]) for row in built]
        manifest["splits"][split] = {
            "prefix_file": str(prefix_path),
            "prefix_sha256": sha256_file(prefix_path),
            "output_file": str(out_path),
            "output_sha256": sha256_file(out_path),
            "rows": len(built),
            "fraud_rows": sum(1 for value in labels if value != 0),
            "normal_rows": sum(1 for value in labels if value == 0),
            "prefix_chars_total": sum(len(row["model_input_text"]) for row in built),
            "strict_prefix_verified": True,
            "duplicate_sample_ids": 0,
        }
        print(
            f"[{split}] {len(built)} 行 -> {out_path} "
            f"(fraud={manifest['splits'][split]['fraud_rows']}, "
            f"normal={manifest['splits'][split]['normal_rows']})"
        )

    manifest_path = args.output_dir / "strict_prefix_inputs_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
