#!/usr/bin/env python3
"""Build the V365 E2-A tE-inclusive binary training mixture.

Only the allow-listed V365 Train file is read.  Fraud calls contribute exactly
one cumulative prefix ending at tE (inclusive).  Each normal call contributes
one cumulative prefix selected by a global one-to-one matching procedure over
turn count and exact model-tokenizer length.  Ambiguous calls are never emitted.

The final physical 6,500-row mixture contains 3,900 deterministic draws from
the new balanced V365 prefix pool and the exact same 2,600 V361 replay examples
that E1-B used.  Reusing the exact replay subset isolates the changed prefix
supervision in the M1-versus-E2-A comparison.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp


EXPECTED_TRAIN_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_train_final.jsonl"
)
EXPECTED_E1B_SUFFIX = (
    "data/processed/v362_e1b_binary_60prefix_40replay/"
    "train_e1b_binary_6500.jsonl"
)
EXPECTED_REPLAY_POOL_SUFFIX = (
    "data/processed/v361_longdoc_text_route_20260824/"
    "train_v361_longdoc_6500.jsonl"
)
EXPECTED_DECISIONS = collections.Counter(
    normal=600, early_fraud=151, explicit_fraud=449, ambiguous=300
)
FRAUD_DECISIONS = {"early_fraud", "explicit_fraud"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_allowlisted(path: Path, suffix: str, description: str) -> Path:
    resolved = path.resolve()
    normalized = resolved.as_posix()
    if not normalized.endswith(suffix):
        raise SystemExit(f"refusing non-allowlisted {description}: {normalized}")
    lowered = normalized.lower()
    if "test" in lowered or "restricted" in lowered:
        raise SystemExit(f"refusing Test/restricted path for {description}: {normalized}")
    return resolved


def canonical_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{int(turn['turn_id']):03d} "
        f"{str(turn['speaker']).strip()}：{str(turn['text']).strip()}"
        for turn in turns
    )


def validate_turns(row: Mapping[str, Any]) -> tuple[bool, str]:
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        return False, "turns_missing_or_empty"
    expected = list(range(1, len(turns) + 1))
    try:
        observed = [int(turn["turn_id"]) for turn in turns]
    except (KeyError, TypeError, ValueError):
        return False, "turn_id_not_integer"
    if observed != expected:
        return False, "turn_id_sequence_not_contiguous_from_1"
    if int(row.get("n_full_turns", -1)) != len(turns):
        return False, "n_full_turns_mismatch"
    return True, "ok"


def token_length(tokenizer: Any, text: str) -> int:
    result = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    ids = result["input_ids"] if isinstance(result, Mapping) else result.input_ids
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return len(ids)


def percentile(values: Sequence[float | int], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q, method="linear"))


def distribution(values: Sequence[int]) -> dict[str, Any]:
    return {
        "n": len(values),
        "min": int(min(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p90": percentile(values, 0.90),
        "max": int(max(values)),
    }


def base_prefix_row(
    source: Mapping[str, Any],
    prefix_turns: list[dict[str, Any]],
    *,
    sample_id: str,
    decision: str,
    target: int,
    prefix_kind: str,
    raw_wordpiece_length: int,
) -> dict[str, Any]:
    text = canonical_text(prefix_turns)
    source_decision = str(source["decision"])
    row = {
        "schema_version": "v365.e2a.te_prefix.v1",
        "dataset_version": "v365_e2a_te_prefix_20261011",
        "sample_id": sample_id,
        "call_id": str(source["call_id"]),
        "source_sample_id": str(source["sample_id"]),
        "source_call_id": str(source["call_id"]),
        "source_record_sha256": source.get("source_record_sha256"),
        "source_text_sha256": source.get("text_sha256"),
        "source_content_group_id": source.get("content_group_id"),
        "split": "train",
        "source_split": str(source.get("split")),
        "source_fulltext_decision": source_decision,
        "decision": decision,
        "binary_target": int(target),
        "binary_loss_mask": 1,
        "stage_target": 2 if target else 0,
        "stage_loss_mask": 0,
        # No 13-class label is synthesized.  The collator uses binary_target
        # independently, while parent/child losses are disabled below.
        "parent_loss_weight": 0.0,
        "child_loss_weight": 0.0,
        "sample_weight": float(source.get("sample_weight", 1.0)),
        "turns": prefix_turns,
        "n_full_turns": len(prefix_turns),
        "source_n_full_turns": int(source["n_full_turns"]),
        "n_prefix_turns": len(prefix_turns),
        "prefix_end_turn": len(prefix_turns),
        "prefix_text": text,
        "model_input_text": text,
        "text": text,
        "prefix_text_hash": sha256_text(text),
        "text_sha256": sha256_text(text),
        "raw_wordpiece_length": int(raw_wordpiece_length),
        "prefix_kind": prefix_kind,
        "prefix_relation": "at_te" if target else "matched_normal",
        "input_view": "te_inclusive_prefix" if target else "matched_normal_prefix",
        "e2a_source": "v365_te_prefix",
        "content_group": "v365_e2a_te_prefix",
        "scenario_tag": source.get("scenario_tag"),
        "claimed_identity_type": source.get("claimed_identity_type"),
        "difficulty_tag": source.get("difficulty_tag"),
        "length_bucket": source.get("length_bucket"),
        "evidence_position_bin": source.get("evidence_position_bin"),
        "disable_window_evidence": True,
        "future_content_used": False,
        "label_leakage_flag": False,
    }
    return row


def repeated_draws(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot sample an empty source")
    output: list[dict[str, Any]] = []
    while len(output) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        output.extend(cycle[: count - len(output)])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v365-train", type=Path, required=True)
    parser.add_argument("--e1b-mixture", type=Path, required=True)
    parser.add_argument("--replay-pool", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=6500)
    parser.add_argument("--new-prefix-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()

    train_path = require_allowlisted(
        args.v365_train, EXPECTED_TRAIN_SUFFIX, "V365 Train"
    )
    e1b_path = require_allowlisted(
        args.e1b_mixture, EXPECTED_E1B_SUFFIX, "E1-B mixture"
    )
    replay_pool_path = require_allowlisted(
        args.replay_pool, EXPECTED_REPLAY_POOL_SUFFIX, "V361 replay pool"
    )
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    if args.total_rows != 6500 or abs(args.new_prefix_fraction - 0.60) > 1e-12:
        raise SystemExit("E2-A contract requires exactly 6500 rows and 60% new prefixes")

    source_sha_before = sha256_file(train_path)
    rows = read_jsonl(train_path)
    counts = collections.Counter(str(row.get("decision")) for row in rows)
    if len(rows) != 1500 or counts != EXPECTED_DECISIONS:
        raise ValueError(f"unexpected V365 Train composition: n={len(rows)} {dict(counts)}")
    if any(str(row.get("split")) != "train" for row in rows):
        raise ValueError("V365 input contains a non-Train row")
    if len({str(row.get("call_id")) for row in rows}) != len(rows):
        raise ValueError("V365 Train call_id is not unique")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    excluded_invalid: list[dict[str, Any]] = []
    fraud_records: list[dict[str, Any]] = []
    for source in rows:
        if str(source["decision"]) not in FRAUD_DECISIONS:
            continue
        valid, reason = validate_turns(source)
        t_e_raw = source.get("tE_turn")
        try:
            t_e = int(t_e_raw)
        except (TypeError, ValueError):
            valid, reason, t_e = False, "tE_missing_or_not_integer", -1
        turns = source.get("turns") if isinstance(source.get("turns"), list) else []
        if valid and not 1 <= t_e <= len(turns):
            valid, reason = False, "tE_out_of_range"
        if valid and int(turns[t_e - 1].get("turn_id", -1)) != t_e:
            valid, reason = False, "tE_does_not_map_to_turn"
        if not valid:
            excluded_invalid.append(
                {
                    "call_id": source.get("call_id"),
                    "sample_id": source.get("sample_id"),
                    "decision": source.get("decision"),
                    "n_full_turns": source.get("n_full_turns"),
                    "tE_turn": t_e_raw,
                    "reason": reason,
                }
            )
            continue
        prefix_turns = [dict(turn) for turn in turns[:t_e]]
        text = canonical_text(prefix_turns)
        n_tokens = token_length(tokenizer, text)
        prefix = base_prefix_row(
            source,
            prefix_turns,
            sample_id=f"V365_E2A_TE__{source['call_id']}",
            decision="early_fraud",
            target=1,
            prefix_kind="fraud_tE_inclusive",
            raw_wordpiece_length=n_tokens,
        )
        prefix["tE_turn"] = t_e
        prefix["source_tE_turn"] = t_e
        prefix["tE_end_speaker"] = str(prefix_turns[-1]["speaker"]).strip()
        fraud_records.append(prefix)

    normal_sources = [row for row in rows if str(row["decision"]) == "normal"]
    if len(fraud_records) != len(normal_sources):
        raise ValueError(
            "one-to-one normal matching requires equal valid fraud/normal counts; "
            f"fraud={len(fraud_records)} normal={len(normal_sources)}"
        )

    # Precompute every role-compatible cumulative normal prefix.  Role matching
    # controls for whether the endpoint is a caller or receiver turn.
    candidate_sets: list[list[dict[str, Any]]] = []
    target_speakers = {str(row["tE_end_speaker"]) for row in fraud_records}
    for source in normal_sources:
        valid, reason = validate_turns(source)
        if not valid:
            raise ValueError(f"normal {source.get('call_id')} invalid: {reason}")
        turns = source["turns"]
        candidates: list[dict[str, Any]] = []
        for end_turn in range(1, len(turns) + 1):
            speaker = str(turns[end_turn - 1]["speaker"]).strip()
            if speaker not in target_speakers:
                continue
            prefix_turns = [dict(turn) for turn in turns[:end_turn]]
            text = canonical_text(prefix_turns)
            candidates.append(
                {
                    "end_turn": end_turn,
                    "speaker": speaker,
                    "token_length": token_length(tokenizer, text),
                    "turns": prefix_turns,
                }
            )
        if not candidates:
            raise ValueError(f"normal {source['call_id']} has no role-compatible prefix")
        candidate_sets.append(candidates)

    fraud_turns = np.asarray([int(row["prefix_end_turn"]) for row in fraud_records])
    fraud_tokens = np.asarray([int(row["raw_wordpiece_length"]) for row in fraud_records])
    n = len(fraud_records)
    maximum_token_length = max(
        int(fraud_tokens.max()),
        max(
            int(candidate["token_length"])
            for candidates in candidate_sets
            for candidate in candidates
        ),
    )
    # Lexicographic objective: minimize total turn-count difference first, then
    # total token-length difference.  The multiplier exceeds the maximum
    # possible token-difference sum, so token gains can never buy a worse turn
    # match.  This dataset admits an exact-turn one-to-one solution for all 600.
    turn_priority = (maximum_token_length + 1) * n + 1
    cost_matrix = np.empty((n, n), dtype=np.float64)
    best_candidate = np.empty((n, n), dtype=np.int16)
    for fraud_index, fraud in enumerate(fraud_records):
        target_turn = int(fraud["prefix_end_turn"])
        target_token = int(fraud["raw_wordpiece_length"])
        target_speaker = str(fraud["tE_end_speaker"])
        for normal_index, candidates in enumerate(candidate_sets):
            compatible = [
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if candidate["speaker"] == target_speaker
            ]
            if not compatible:
                compatible = list(enumerate(candidates))
            costs = [
                abs(int(candidate["end_turn"]) - target_turn) * turn_priority
                + abs(int(candidate["token_length"]) - target_token)
                for _, candidate in compatible
            ]
            local_best = int(np.argmin(costs))
            candidate_index = compatible[local_best][0]
            # Tiny stable tie breaker makes the assignment reproducible across
            # SciPy implementations without changing any material cost.
            tie_breaker = (normal_index + candidate_index / 1000.0) * 1e-12
            cost_matrix[fraud_index, normal_index] = costs[local_best] + tie_breaker
            best_candidate[fraud_index, normal_index] = candidate_index

    fraud_indices, normal_indices = linear_sum_assignment(cost_matrix)
    if list(fraud_indices) != list(range(n)) or len(set(normal_indices)) != n:
        raise ValueError("global matching did not produce a one-to-one assignment")

    normal_records: list[dict[str, Any]] = []
    matching_pairs: list[dict[str, Any]] = []
    for fraud_index, normal_index in zip(fraud_indices, normal_indices):
        fraud = fraud_records[int(fraud_index)]
        source = normal_sources[int(normal_index)]
        candidate_index = int(best_candidate[int(fraud_index), int(normal_index)])
        candidate = candidate_sets[int(normal_index)][candidate_index]
        normal = base_prefix_row(
            source,
            candidate["turns"],
            sample_id=f"V365_E2A_NORMAL__{source['call_id']}",
            decision="normal",
            target=0,
            prefix_kind="normal_matched_to_fraud_tE",
            raw_wordpiece_length=int(candidate["token_length"]),
        )
        normal["tE_turn"] = None
        normal["matched_fraud_call_id"] = fraud["call_id"]
        normal["matched_target_turns"] = int(fraud["prefix_end_turn"])
        normal["matched_target_wordpiece_length"] = int(fraud["raw_wordpiece_length"])
        normal_records.append(normal)
        matching_pairs.append(
            {
                "fraud_call_id": fraud["call_id"],
                "fraud_source_decision": fraud["source_fulltext_decision"],
                "fraud_tE_turn": int(fraud["prefix_end_turn"]),
                "fraud_wordpiece_length": int(fraud["raw_wordpiece_length"]),
                "fraud_tE_end_speaker": fraud["tE_end_speaker"],
                "normal_call_id": normal["call_id"],
                "normal_prefix_end_turn": int(normal["prefix_end_turn"]),
                "normal_wordpiece_length": int(normal["raw_wordpiece_length"]),
                "normal_end_speaker": candidate["speaker"],
                "absolute_turn_delta": abs(
                    int(normal["prefix_end_turn"]) - int(fraud["prefix_end_turn"])
                ),
                "absolute_token_delta": abs(
                    int(normal["raw_wordpiece_length"])
                    - int(fraud["raw_wordpiece_length"])
                ),
                "lexicographic_matching_cost": float(
                    cost_matrix[int(fraud_index), int(normal_index)]
                ),
            }
        )

    unique_prefix_rows = fraud_records + normal_records
    if len(unique_prefix_rows) != 1200:
        raise ValueError(f"expected 1200 unique prefixes, got {len(unique_prefix_rows)}")
    if any(str(row.get("source_split")) != "train" for row in unique_prefix_rows):
        raise ValueError("derived prefix contains a non-Train source")
    if any(str(row.get("source_fulltext_decision")) == "ambiguous" for row in unique_prefix_rows):
        raise ValueError("ambiguous row leaked into derived prefixes")
    for row in fraud_records:
        if len(row["turns"]) != int(row["tE_turn"]):
            raise ValueError(f"{row['call_id']}: fraud prefix does not end exactly at tE")
        if max(int(turn["turn_id"]) for turn in row["turns"]) != int(row["tE_turn"]):
            raise ValueError(f"{row['call_id']}: post-tE turn leaked into prefix")

    # Extract exactly the E1-B replay subset, and verify that every row traces
    # back to the allow-listed original 6,500-row pool.
    e1b_rows = read_jsonl(e1b_path)
    replay_rows = [row for row in e1b_rows if row.get("e1b_source") == "fulltext_replay"]
    if len(e1b_rows) != 6500 or len(replay_rows) != 2600:
        raise ValueError(
            f"unexpected E1-B mixture/replay counts: {len(e1b_rows)}/{len(replay_rows)}"
        )
    replay_pool = read_jsonl(replay_pool_path)
    if len(replay_pool) != 6500:
        raise ValueError(f"unexpected replay pool size: {len(replay_pool)}")
    pool_by_id = {str(row["sample_id"]): row for row in replay_pool}
    if len(pool_by_id) != 6500:
        raise ValueError("original replay pool sample_id is not unique")
    replay_source_ids: list[str] = []
    for row in replay_rows:
        source_id = str(row.get("source_sample_id") or "")
        if source_id not in pool_by_id:
            raise ValueError(f"E1-B replay source is absent from original pool: {source_id}")
        source = pool_by_id[source_id]
        if str(row.get("model_input_text")) != str(source.get("model_input_text")):
            raise ValueError(f"E1-B replay text differs from original pool: {source_id}")
        replay_source_ids.append(source_id)
    if len(set(replay_source_ids)) != 2600:
        raise ValueError("E1-B replay subset is not 2,600 unique source rows")

    new_count = round(args.total_rows * args.new_prefix_fraction)
    replay_count = args.total_rows - new_count
    if new_count != 3900 or replay_count != 2600:
        raise ValueError("mixture contract arithmetic mismatch")
    rng = random.Random(args.seed)
    fraud_draws = repeated_draws(fraud_records, new_count // 2, rng)
    normal_draws = repeated_draws(normal_records, new_count // 2, rng)
    mixed: list[dict[str, Any]] = []
    for draw_index, source in enumerate(fraud_draws + normal_draws):
        row = dict(source)
        source_prefix_id = str(source["sample_id"])
        row.update(
            {
                "sample_id": f"E2A_PREFIX_{draw_index:04d}__{source_prefix_id}",
                "source_prefix_sample_id": source_prefix_id,
                "e2a_source": "v365_te_prefix",
            }
        )
        mixed.append(row)
    for draw_index, source in enumerate(replay_rows):
        row = dict(source)
        original_source_id = str(source["source_sample_id"])
        row.update(
            {
                "sample_id": f"E2A_REPLAY_{draw_index:04d}__{original_source_id}",
                "source_sample_id": original_source_id,
                "e2a_source": "v361_fulltext_replay",
                "input_view": "fulltext_replay",
                "content_group": "v361_fulltext_replay",
                "binary_target": int(row.get("canonical_label_13", row.get("label_13", 0)) != 0),
                "binary_loss_mask": 1,
            }
        )
        mixed.append(row)
    rng.shuffle(mixed)
    if len(mixed) != 6500 or len({str(row["sample_id"]) for row in mixed}) != 6500:
        raise ValueError("final mixture size or instance-ID uniqueness mismatch")
    if any(float(row.get("binary_loss_mask", 0)) <= 0 for row in mixed):
        raise ValueError("masked/ambiguous row leaked into final mixture")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    unique_path = args.output_dir / "v365_train_te_prefix_unique_1200.jsonl"
    fraud_path = args.output_dir / "v365_train_fraud_te_prefix_600.jsonl"
    normal_path = args.output_dir / "v365_train_normal_matched_prefix_600.jsonl"
    pair_path = args.output_dir / "normal_fraud_matching_pairs.jsonl"
    excluded_path = args.output_dir / "excluded_invalid_te.jsonl"
    mixture_path = args.output_dir / "train_e2a_binary_6500.jsonl"
    write_jsonl(unique_path, unique_prefix_rows)
    write_jsonl(fraud_path, fraud_records)
    write_jsonl(normal_path, normal_records)
    write_jsonl(pair_path, matching_pairs)
    write_jsonl(excluded_path, excluded_invalid)
    write_jsonl(mixture_path, mixed)

    fraud_turn_values = [int(row["prefix_end_turn"]) for row in fraud_records]
    normal_turn_values = [int(row["prefix_end_turn"]) for row in normal_records]
    fraud_token_values = [int(row["raw_wordpiece_length"]) for row in fraud_records]
    normal_token_values = [int(row["raw_wordpiece_length"]) for row in normal_records]
    turn_deltas = [int(row["absolute_turn_delta"]) for row in matching_pairs]
    token_deltas = [int(row["absolute_token_delta"]) for row in matching_pairs]
    if any(turn_deltas):
        raise ValueError(
            "expected exact turn-count matching for all 600 pairs, got "
            f"{sum(delta == 0 for delta in turn_deltas)}/600 exact"
        )
    source_sha_after = sha256_file(train_path)
    if source_sha_after != source_sha_before:
        raise RuntimeError("V365 Train source changed while building derived data")

    source_counts = collections.Counter(str(row["e2a_source"]) for row in mixed)
    binary_counts = collections.Counter(int(row["binary_target"]) for row in mixed)
    by_source_binary: dict[str, dict[str, int]] = {}
    for source_name in sorted(source_counts):
        members = [row for row in mixed if row["e2a_source"] == source_name]
        by_source_binary[source_name] = {
            str(key): value
            for key, value in sorted(
                collections.Counter(int(row["binary_target"]) for row in members).items()
            )
        }
    manifest = {
        "schema_version": "v365.e2a.te_prefix_mixture.v1",
        "created_for": "E2-A binary-only training",
        "seed": args.seed,
        "safety": {
            "v365_source_allowlist": str(train_path),
            "source_split": "train only",
            "test_or_restricted_paths_read": False,
            "ambiguous_rows_used": 0,
            "original_v365_file_modified": False,
            "source_sha256_before": source_sha_before,
            "source_sha256_after": source_sha_after,
        },
        "v365_train_composition": dict(sorted(counts.items())),
        "fraud_prefix": {
            "policy": "exactly turns 1..tE inclusive; no later turn",
            "eligible": 600,
            "emitted": len(fraud_records),
            "early_fraud_source_rows": sum(
                row["source_fulltext_decision"] == "early_fraud" for row in fraud_records
            ),
            "explicit_fraud_source_rows": sum(
                row["source_fulltext_decision"] == "explicit_fraud" for row in fraud_records
            ),
            "invalid_excluded": len(excluded_invalid),
            "other_pre_tE_prefixes_generated": 0,
        },
        "normal_prefix": {
            "policy": "one prefix per normal call; global one-to-one minimum-cost match to fraud tE prefixes by endpoint role, turn count and tokenizer length",
            "eligible": len(normal_sources),
            "emitted": len(normal_records),
            "unique_normal_calls": len({row["call_id"] for row in normal_records}),
            "matching_objective": "lexicographic: exact/minimum turn difference first, then minimum absolute WordPiece-token difference",
            "turn_priority_multiplier": turn_priority,
        },
        "matching_quality": {
            "fraud_turns": distribution(fraud_turn_values),
            "normal_turns": distribution(normal_turn_values),
            "fraud_wordpiece_length": distribution(fraud_token_values),
            "normal_wordpiece_length": distribution(normal_token_values),
            "absolute_turn_delta": distribution(turn_deltas),
            "absolute_token_delta": distribution(token_deltas),
            "turn_count_ks_statistic": float(
                ks_2samp(fraud_turn_values, normal_turn_values).statistic
            ),
            "turn_count_ks_pvalue": float(
                ks_2samp(fraud_turn_values, normal_turn_values).pvalue
            ),
            "token_length_ks_statistic": float(
                ks_2samp(fraud_token_values, normal_token_values).statistic
            ),
            "token_length_ks_pvalue": float(
                ks_2samp(fraud_token_values, normal_token_values).pvalue
            ),
        },
        "mixture": {
            "output": str(mixture_path.resolve()),
            "output_sha256": sha256_file(mixture_path),
            "rows": len(mixed),
            "new_prefix_fraction": args.new_prefix_fraction,
            "source_counts": dict(sorted(source_counts.items())),
            "binary_counts": {str(k): v for k, v in sorted(binary_counts.items())},
            "by_source_binary_counts": by_source_binary,
            "new_prefix_unique_rows": len(unique_prefix_rows),
            "new_prefix_draws": new_count,
            "new_prefix_fraud_draws": len(fraud_draws),
            "new_prefix_normal_draws": len(normal_draws),
            "replay_draws": replay_count,
        },
        "replay_control": {
            "policy": "reuse the exact 2,600 full-text replay examples used by E1-B",
            "e1b_mixture": str(e1b_path),
            "e1b_mixture_sha256": sha256_file(e1b_path),
            "original_pool": str(replay_pool_path),
            "original_pool_sha256": sha256_file(replay_pool_path),
            "original_pool_rows": len(replay_pool),
            "reused_rows": len(replay_rows),
            "unique_source_ids": len(set(replay_source_ids)),
            "source_id_set_sha256": sha256_text("\n".join(sorted(replay_source_ids))),
        },
        "tokenizer_and_solver": {
            "tokenizer_directory": str(args.tokenizer.resolve()),
            "tokenizer_json_sha256": (
                sha256_file(args.tokenizer / "tokenizer.json")
                if (args.tokenizer / "tokenizer.json").is_file()
                else None
            ),
            "vocab_sha256": (
                sha256_file(args.tokenizer / "vocab.txt")
                if (args.tokenizer / "vocab.txt").is_file()
                else None
            ),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "solver": "scipy.optimize.linear_sum_assignment",
        },
        "files": {
            path.name: {"rows": count, "sha256": sha256_file(path)}
            for path, count in [
                (unique_path, len(unique_prefix_rows)),
                (fraud_path, len(fraud_records)),
                (normal_path, len(normal_records)),
                (pair_path, len(matching_pairs)),
                (excluded_path, len(excluded_invalid)),
                (mixture_path, len(mixed)),
            ]
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    report = f"""# V365 E2-A tE前缀训练数据审计报告

- V365来源：仅 `{train_path}`（Train 1500条）；未读取Dev/Test/受限Test标签。
- 诈骗前缀：{len(fraud_records)}条，每通电话恰好保留第1轮至tE轮，包含tE，未生成其他tE前逐轮前缀。
- 无效tE排除：{len(excluded_invalid)}条，明细见 `excluded_invalid_te.jsonl`。
- 正常前缀：{len(normal_records)}条，每个正常call恰好1条；与诈骗前缀做全局一对一轮数+token长度匹配。
- 唯一前缀池：{len(unique_prefix_rows)}条（诈骗600、正常600）；ambiguous使用0条。
- 最终训练集：6500条，其中V365新前缀3900条（诈骗1950、正常1950），E1-B原样回放2600条。
- 回放控制：复用E1-B实际使用的同一批2600条，均可追溯至原V361 6500行池。

匹配质量：轮数平均绝对差 {statistics.fmean(turn_deltas):.3f}，P90 {percentile(turn_deltas, 0.9):.3f}；token平均绝对差 {statistics.fmean(token_deltas):.3f}，P90 {percentile(token_deltas, 0.9):.3f}。完整统计与哈希见 `manifest.json`。
"""
    (args.output_dir / "DATASET_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
