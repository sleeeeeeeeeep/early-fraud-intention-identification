#!/usr/bin/env python3
"""Attach one deterministic adjacent-prefix auxiliary pair to E2-B rows.

E2-C keeps the 6,500 E2-B physical examples, their order, labels, weights, and
main model text unchanged.  It only adds an auxiliary pair to eligible V365
Train draws.  Fraud pairs end strictly before tE; normal pairs come from the
same Train call.  Replay rows do not receive an auxiliary pair.

No Dev, Test, restricted-label, or ambiguous row is read or emitted.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_E2B_MIXTURE_SUFFIX = (
    "data/processed/v365_e2b_boundary_balanced_60new_40replay_s202/"
    "train_e2b_binary_6500.jsonl"
)
EXPECTED_E2B_MANIFEST_SUFFIX = (
    "data/processed/v365_e2b_boundary_balanced_60new_40replay_s202/manifest.json"
)
EXPECTED_TRAIN_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_train_final.jsonl"
)
EXPECTED_E2B_MIXTURE_SHA256 = (
    "ec27494a74924a8e9e41cdf7e859bdc19cf9a6c02e23cb0e41472d3026d348c4"
)
EXPECTED_E2B_MANIFEST_SHA256 = (
    "127f6f3a21eda6009a8acaf527a3ffce4f68d6aa66c5622ba7951b9fb026a091"
)
EXPECTED_TRAIN_SHA256 = (
    "189cd9c7e38a2312eebf30bb6abac21ecd70644db7b3ae3e5eb2809b9da0957f"
)
EXPECTED_TOKENIZER_SUFFIX = "models/IDEA-CCNL-Erlangshen-MegatronBert-1.3B"
EXPECTED_VOCAB_SHA256 = (
    "366fe2b1318c774fd28bdea58dbcfbd6d6aeae313fdcd9063d170cdc602fd51a"
)
EXPECTED_DECISIONS = collections.Counter(
    normal=600, early_fraud=151, explicit_fraud=449, ambiguous=300
)
V365_SOURCE = "v365_te_prefix"
REPLAY_SOURCE = "v361_fulltext_replay"


def forbidden(path: Path) -> bool:
    normalized = path.resolve(strict=False).as_posix().lower()
    return "restricted" in normalized or any(
        "test" in part for part in normalized.split("/")
    )


def require_file(path: Path, suffix: str, description: str) -> Path:
    resolved = path.resolve()
    if forbidden(resolved):
        raise SystemExit(f"refusing Test/restricted {description}: {resolved}")
    if not resolved.as_posix().endswith(suffix):
        raise SystemExit(f"refusing non-allowlisted {description}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{int(turn['turn_id']):03d} "
        f"{str(turn['speaker']).strip()}：{str(turn['text']).strip()}"
        for turn in turns
    )


def token_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def contiguous_turns(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    turns = source.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{source.get('call_id')}: missing turns")
    copied = [dict(turn) for turn in turns]
    observed = [int(turn["turn_id"]) for turn in copied]
    if observed != list(range(1, len(copied) + 1)):
        raise ValueError(f"{source.get('call_id')}: non-contiguous turns")
    return copied


def ordered_pair_currents(
    *, decision: str, t_e: int | None, n_turns: int
) -> list[int]:
    if decision in {"early_fraud", "explicit_fraud"}:
        if t_e is None:
            raise ValueError("fraud source is missing tE")
        legal = list(range(2, t_e))
        preferred: list[int] = []
        low, high = 2, t_e - 1
        while low <= high:
            preferred.append(low)
            if high != low:
                preferred.append(high)
            low += 1
            high -= 1
    elif decision == "normal":
        legal = list(range(2, n_turns + 1))
        preferred = [2, 3]
        low, high = 4, n_turns
        while low <= high:
            preferred.append(high)
            if high != low:
                preferred.append(low)
            low += 1
            high -= 1
    else:
        raise ValueError(f"unsupported E2-C decision: {decision}")
    result: list[int] = []
    for current in preferred + legal:
        if current in legal and current not in result:
            result.append(current)
    return result


def strip_e2c(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("e2c_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2b-mixture", type=Path, required=True)
    parser.add_argument("--e2b-manifest", type=Path, required=True)
    parser.add_argument("--v365-train", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()

    if args.seed != 202:
        raise SystemExit("E2-C contract fixes seed=202")
    mixture_path = require_file(
        args.e2b_mixture, EXPECTED_E2B_MIXTURE_SUFFIX, "E2-B mixture"
    )
    manifest_path = require_file(
        args.e2b_manifest, EXPECTED_E2B_MANIFEST_SUFFIX, "E2-B manifest"
    )
    train_path = require_file(args.v365_train, EXPECTED_TRAIN_SUFFIX, "V365 Train")
    tokenizer_path = args.tokenizer.resolve()
    if (
        forbidden(tokenizer_path)
        or not tokenizer_path.is_dir()
        or not tokenizer_path.as_posix().endswith(EXPECTED_TOKENIZER_SUFFIX)
    ):
        raise SystemExit(f"invalid/refused tokenizer directory: {args.tokenizer}")
    vocab_path = tokenizer_path / "vocab.txt"
    if not vocab_path.is_file() or sha256_file(vocab_path) != EXPECTED_VOCAB_SHA256:
        raise ValueError("frozen tokenizer vocab hash mismatch")
    if forbidden(args.output_dir):
        raise SystemExit(f"refusing Test/restricted output: {args.output_dir}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")

    observed_hashes = {
        "e2b_mixture": sha256_file(mixture_path),
        "e2b_manifest": sha256_file(manifest_path),
        "v365_train_before": sha256_file(train_path),
    }
    expected_hashes = {
        "e2b_mixture": EXPECTED_E2B_MIXTURE_SHA256,
        "e2b_manifest": EXPECTED_E2B_MANIFEST_SHA256,
        "v365_train_before": EXPECTED_TRAIN_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            f"source hash mismatch: observed={observed_hashes} expected={expected_hashes}"
        )
    e2b_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if e2b_manifest.get("safety", {}).get("test_or_restricted_files_read") != 0:
        raise ValueError("E2-B manifest does not attest zero Test reads")
    if e2b_manifest.get("mixture", {}).get("sha256") != EXPECTED_E2B_MIXTURE_SHA256:
        raise ValueError("E2-B manifest mixture hash differs from frozen contract")

    train_rows = read_jsonl(train_path)
    train_counts = collections.Counter(str(row.get("decision")) for row in train_rows)
    if len(train_rows) != 1500 or train_counts != EXPECTED_DECISIONS:
        raise ValueError(
            f"unexpected V365 Train composition: n={len(train_rows)} {dict(train_counts)}"
        )
    if any(str(row.get("split")) != "train" for row in train_rows):
        raise ValueError("V365 source contains a non-Train row")
    source_by_call = {str(row["call_id"]): row for row in train_rows}
    if len(source_by_call) != 1500:
        raise ValueError("V365 Train call_id is not unique")

    rows = read_jsonl(mixture_path)
    if len(rows) != 6500 or len({str(row["sample_id"]) for row in rows}) != 6500:
        raise ValueError("E2-B physical mixture is not exactly 6,500 unique rows")
    source_counts = collections.Counter(str(row.get("e2b_source")) for row in rows)
    binary_counts = collections.Counter(int(row.get("binary_target", -1)) for row in rows)
    if source_counts != collections.Counter({V365_SOURCE: 3900, REPLAY_SOURCE: 2600}):
        raise ValueError(f"unexpected source counts: {dict(source_counts)}")
    if binary_counts != collections.Counter({0: 3250, 1: 3250}):
        raise ValueError(f"unexpected binary counts: {dict(binary_counts)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    indices_by_call: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if str(row.get("e2b_source")) != V365_SOURCE:
            continue
        call_id = str(row.get("call_id"))
        if call_id not in source_by_call:
            raise ValueError(f"mixture call absent from V365 Train: {call_id}")
        indices_by_call[call_id].append(index)
    if len(indices_by_call) != 1200:
        raise ValueError(f"expected 1,200 unique V365 calls, got {len(indices_by_call)}")

    output_rows = [dict(row) for row in rows]
    audit_rows: list[dict[str, Any]] = []
    current_histogram: collections.Counter[int] = collections.Counter()
    kind_counts: collections.Counter[str] = collections.Counter()
    unavailable_counts: collections.Counter[str] = collections.Counter()
    extended_beyond_main = 0

    for call_id in sorted(indices_by_call):
        source = source_by_call[call_id]
        decision = str(source.get("decision"))
        if decision == "ambiguous":
            raise ValueError(f"ambiguous call leaked into E2-C mixture: {call_id}")
        turns = contiguous_turns(source)
        t_e_raw = source.get("tE_turn")
        t_e = int(t_e_raw) if t_e_raw is not None else None
        if decision in {"early_fraud", "explicit_fraud"}:
            if t_e is None or not 1 <= t_e <= len(turns):
                raise ValueError(f"{call_id}: invalid fraud tE={t_e_raw}")
        elif decision != "normal":
            raise ValueError(f"{call_id}: unsupported source decision={decision}")

        priority = ordered_pair_currents(
            decision=decision, t_e=t_e, n_turns=len(turns)
        )
        row_indices = sorted(indices_by_call[call_id], key=lambda i: str(rows[i]["sample_id"]))
        for occurrence, row_index in enumerate(row_indices):
            original = rows[row_index]
            row = output_rows[row_index]
            row["e2c_schema_version"] = "v365.e2c.k2_adjacent_pair.v1"
            row["e2c_dataset_version"] = "v365_e2c_k2_pair_20261012"
            row["e2c_source"] = str(original["e2b_source"])
            row["e2c_disable_legacy_boundary_aux"] = True
            row["e2c_pair_schedule_occurrence"] = occurrence
            row["e2c_base_row_sha256"] = sha256_text(
                json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            if not priority:
                reason = "fraud_te_le_2_no_fully_pre_te_adjacent_pair"
                row.update(
                    {
                        "e2c_pair_available": False,
                        "e2c_aux_mask": 0,
                        "e2c_pair_aux_mask": 0,
                        "e2c_pair_exclusion_reason": reason,
                        "e2c_pair_uses_post_pair_endpoint_content": False,
                    }
                )
                unavailable_counts[reason] += 1
                audit_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "call_id": call_id,
                        "decision": decision,
                        "scenario_tag": source.get("scenario_tag"),
                        "pair_available": False,
                        "exclusion_reason": reason,
                        "tE_turn": t_e,
                    }
                )
                continue

            current_turn = priority[occurrence % len(priority)]
            previous_turn = current_turn - 1
            previous_text = canonical_text(turns[:previous_turn])
            current_text = canonical_text(turns[:current_turn])
            if not previous_text or not current_text.startswith(previous_text):
                raise ValueError(f"{call_id}: malformed cumulative pair")
            if decision in {"early_fraud", "explicit_fraud"} and current_turn >= int(t_e):
                raise ValueError(f"{call_id}: fraud auxiliary pair is not fully pre-tE")
            pair_kind = (
                "fraud_fully_pre_te_adjacent"
                if decision in {"early_fraud", "explicit_fraud"}
                else "normal_same_call_adjacent"
            )
            main_end = original.get("prefix_end_turn")
            beyond_main = main_end is not None and current_turn > int(main_end)
            extended_beyond_main += int(beyond_main)
            pair_id = (
                f"E2C_K2__{call_id}__{previous_turn:03d}_{current_turn:03d}__"
                f"OCC{occurrence:02d}"
            )
            content_id = f"E2C_K2_CONTENT__{call_id}__{previous_turn:03d}_{current_turn:03d}"
            row.update(
                {
                    "e2c_pair_available": True,
                    "e2c_aux_mask": 1,
                    "e2c_pair_aux_mask": 1,
                    "e2c_pair_id": pair_id,
                    "e2c_pair_assignment_id": pair_id,
                    "e2c_pair_content_id": content_id,
                    "e2c_pair_kind": pair_kind,
                    "e2c_pair_prev_end_turn": previous_turn,
                    "e2c_pair_curr_end_turn": current_turn,
                    "e2c_pair_schedule_rank": occurrence % len(priority),
                    "e2c_pair_prev_model_input_text": previous_text,
                    "e2c_pair_curr_model_input_text": current_text,
                    "e2c_pair_prev_text_sha256": sha256_text(previous_text),
                    "e2c_pair_curr_text_sha256": sha256_text(current_text),
                    "e2c_pair_prev_raw_wordpiece_length": token_length(
                        tokenizer, previous_text
                    ),
                    "e2c_pair_curr_raw_wordpiece_length": token_length(
                        tokenizer, current_text
                    ),
                    "e2c_pair_strictly_before_te": (
                        current_turn < int(t_e)
                        if decision in {"early_fraud", "explicit_fraud"}
                        else None
                    ),
                    "e2c_pair_extends_beyond_main_prefix": beyond_main,
                    "e2c_pair_within_main_view": not beyond_main,
                    "e2c_pair_uses_post_pair_endpoint_content": False,
                }
            )
            current_histogram[current_turn] += 1
            kind_counts[pair_kind] += 1
            audit_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "call_id": call_id,
                    "decision": decision,
                    "scenario_tag": source.get("scenario_tag"),
                    "pair_available": True,
                    "pair_id": pair_id,
                    "pair_content_id": content_id,
                    "pair_kind": pair_kind,
                    "previous_end_turn": previous_turn,
                    "current_end_turn": current_turn,
                    "tE_turn": t_e,
                    "main_prefix_end_turn": main_end,
                    "extends_beyond_main_prefix": beyond_main,
                    "previous_text_sha256": row["e2c_pair_prev_text_sha256"],
                    "current_text_sha256": row["e2c_pair_curr_text_sha256"],
                }
            )

    for index, row in enumerate(output_rows):
        if str(row.get("e2b_source")) == REPLAY_SOURCE:
            row.update(
                {
                    "e2c_schema_version": "v365.e2c.k2_adjacent_pair.v1",
                    "e2c_dataset_version": "v365_e2c_k2_pair_20261012",
                    "e2c_source": REPLAY_SOURCE,
                    "e2c_disable_legacy_boundary_aux": True,
                    "e2c_base_row_sha256": sha256_text(
                        json.dumps(
                            rows[index],
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                    "e2c_pair_available": False,
                    "e2c_aux_mask": 0,
                    "e2c_pair_aux_mask": 0,
                    "e2c_pair_exclusion_reason": "replay_main_ce_only",
                    "e2c_pair_uses_post_pair_endpoint_content": False,
                }
            )
        if strip_e2c(row) != rows[index]:
            raise RuntimeError(f"physical row changed beyond e2c metadata: index={index}")

    paired = [row for row in output_rows if bool(row.get("e2c_pair_available"))]
    paired_fraud = [row for row in paired if int(row["binary_target"]) == 1]
    paired_normal = [row for row in paired if int(row["binary_target"]) == 0]
    if len(audit_rows) != 3900 or len(paired) != len(paired_fraud) + len(paired_normal):
        raise RuntimeError("E2-C pair accounting mismatch")
    if (len(paired), len(paired_fraud), len(paired_normal)) != (3838, 1279, 2559):
        raise ValueError(
            "frozen E2-C pair counts changed: "
            f"{len(paired)}/{len(paired_fraud)}/{len(paired_normal)}"
        )
    if any(int(row["e2c_pair_curr_end_turn"]) >= int(row["tE_turn"]) for row in paired_fraud):
        raise ValueError("a fraud auxiliary pair reaches or crosses tE")
    if any(row.get("e2c_source") == REPLAY_SOURCE and row.get("e2c_pair_available") for row in output_rows):
        raise ValueError("a replay row received an auxiliary pair")
    pair_content_keys = {
        (
            str(row["call_id"]),
            int(row["e2c_pair_prev_end_turn"]),
            int(row["e2c_pair_curr_end_turn"]),
        )
        for row in paired
    }
    fraud_content_keys = {
        (
            str(row["call_id"]),
            int(row["e2c_pair_prev_end_turn"]),
            int(row["e2c_pair_curr_end_turn"]),
        )
        for row in paired_fraud
    }
    normal_content_keys = pair_content_keys - fraud_content_keys
    if (len(pair_content_keys), len(fraud_content_keys), len(normal_content_keys)) != (
        3710,
        1156,
        2554,
    ):
        raise ValueError("frozen E2-C unique pair-content coverage changed")
    if extended_beyond_main != 969:
        raise ValueError(
            f"frozen normal beyond-main count changed: {extended_beyond_main} != 969"
        )
    paired_endpoints_by_call: dict[str, set[int]] = collections.defaultdict(set)
    for row in paired:
        paired_endpoints_by_call[str(row["call_id"])].add(
            int(row["e2c_pair_curr_end_turn"])
        )
    eligible_fraud_calls = 0
    for call_id, source in source_by_call.items():
        decision = str(source.get("decision"))
        endpoints = paired_endpoints_by_call.get(call_id, set())
        if decision in {"early_fraud", "explicit_fraud"}:
            t_e = int(source["tE_turn"])
            if t_e <= 2:
                if endpoints:
                    raise ValueError(f"{call_id}: ineligible fraud call has a pair")
                continue
            eligible_fraud_calls += 1
            if not {2, t_e - 1}.issubset(endpoints):
                raise ValueError(f"{call_id}: fraud early/boundary pair coverage changed")
        elif decision == "normal":
            n_turns = len(contiguous_turns(source))
            if not {2, 3, n_turns}.issubset(endpoints):
                raise ValueError(f"{call_id}: normal early/final pair coverage changed")
    if eligible_fraud_calls != 573 or len(paired_endpoints_by_call) != 1173:
        raise ValueError("frozen unique-call pair coverage changed")
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("V365 Train changed while constructing E2-C data")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mixture_out = args.output_dir / "train_e2c_binary_6500.jsonl"
    audit_out = args.output_dir / "v365_train_k2_pair_assignments_3900.jsonl"
    write_jsonl(mixture_out, output_rows)
    write_jsonl(audit_out, audit_rows)

    manifest = {
        "schema_version": "v365.e2c.k2_pair_mixture.v1",
        "created_for": "E2-C binary-only k2-aligned adjacent-prefix regularisation",
        "seed": 202,
        "safety": {
            "source_provenance": "V365 Train and frozen E2-B physical mixture only",
            "dev_files_read": 0,
            "test_or_restricted_files_read": 0,
            "ambiguous_rows_used": 0,
            "original_source_files_modified": False,
        },
        "source_integrity": {
            "e2b_mixture_path": str(mixture_path),
            "e2b_mixture_sha256": observed_hashes["e2b_mixture"],
            "e2b_manifest_path": str(manifest_path),
            "e2b_manifest_sha256": observed_hashes["e2b_manifest"],
            "v365_train_path": str(train_path),
            "v365_train_sha256_before": observed_hashes["v365_train_before"],
            "v365_train_sha256_after": sha256_file(train_path),
            "tokenizer_directory": str(tokenizer_path),
            "vocab_sha256": sha256_file(vocab_path),
        },
        "physical_mixture": {
            "rows": 6500,
            "row_order_unchanged": True,
            "sample_ids_unchanged": True,
            "main_text_labels_weights_unchanged": True,
            "deep_equal_after_removing_e2c_fields": True,
            "source_counts": dict(sorted(source_counts.items())),
            "binary_counts": {str(k): v for k, v in sorted(binary_counts.items())},
        },
        "pair_policy": {
            "one_pair_max_per_physical_v365_draw": True,
            "fraud_legal_current_turns": "2..tE-1; both prefixes strictly before tE",
            "fraud_priority": [2, "tE-1", 3, "tE-2", "alternate inward"],
            "normal_legal_current_turns": "2..n_full_turns from the same Train call",
            "normal_priority": [2, 3, "n_full_turns", 4, "n_full_turns-1", "alternate inward"],
            "assignment": "sample_id-sorted occurrences cycle through deterministic priority",
            "auxiliary_hard_label_assigned": False,
            "included_in_physical_row_count": False,
            "causality": "each view contains only turns 1..its endpoint; normal pairs may extend beyond the separate main-row endpoint",
        },
        "pair_assignments": {
            "v365_draws": len(audit_rows),
            "paired_draws": len(paired),
            "paired_fraud_draws": len(paired_fraud),
            "paired_normal_draws": len(paired_normal),
            "unavailable_draws": len(audit_rows) - len(paired),
            "unavailable_reasons": dict(sorted(unavailable_counts.items())),
            "pair_kind_counts": dict(sorted(kind_counts.items())),
            "current_turn_histogram": {
                str(k): v for k, v in sorted(current_histogram.items())
            },
            "extends_beyond_main_prefix_draws": extended_beyond_main,
            "unique_pair_contents": len(pair_content_keys),
            "unique_fraud_pair_contents": len(fraud_content_keys),
            "unique_normal_pair_contents": len(normal_content_keys),
            "duplicate_fraud_assignments": len(paired_fraud) - len(fraud_content_keys),
            "duplicate_normal_assignments": len(paired_normal) - len(normal_content_keys),
            "unique_paired_calls": len(
                {str(row["call_id"]) for row in audit_rows if row["pair_available"]}
            ),
        },
        "files": {
            mixture_out.name: {
                "rows": len(output_rows),
                "sha256": sha256_file(mixture_out),
            },
            audit_out.name: {
                "rows": len(audit_rows),
                "sha256": sha256_file(audit_out),
            },
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    audit_md = f"""# E2-C 数据审计

- 物理训练行：6500；去除 `e2c_*` 字段后与冻结 E2-B 混合文件逐行完全一致。
- 来源：V365 Train 派生行3900；同一批 V361 replay 2600。
- 类别：fraud 3250、normal 3250；ambiguous 0。
- 辅助相邻前缀对：{len(paired)} 个物理draw带pair，其中fraud {len(paired_fraud)}、normal {len(paired_normal)}；其余 {len(audit_rows) - len(paired)} 个V365 draw无合法pair并已记录。
- fraud pair 的当前端严格小于 tE；normal pair 来自同一Train通话。辅助pair不计入6500条，不赋硬标签，只参与连续两轮报警约束。
- Dev读取0；Test/restricted读取0；原始数据未修改。
"""
    (args.output_dir / "DATASET_AUDIT.md").write_text(audit_md, encoding="utf-8")
    print(
        f"E2-C data complete: rows={len(output_rows)} pairs={len(paired)} "
        f"fraud_pairs={len(paired_fraud)} normal_pairs={len(paired_normal)} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
