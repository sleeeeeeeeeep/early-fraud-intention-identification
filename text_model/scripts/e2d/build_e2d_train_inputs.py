#!/usr/bin/env python3
"""Build causal, Train-only adjacent-prefix inputs for the E2-D gate.

This builder opens exactly one allow-listed V365 Train file.  Pair labels and
label-blind model inputs are deliberately written to separate files.  It never
scans the dataset directory, never reads Dev/Test/restricted files, and refuses
to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEED = 202
EXPECTED_TRAIN_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_train_final.jsonl"
)
EXPECTED_OUTPUT_SUFFIX = "data/processed/v365_e2d_gate_train_s202"
EXPECTED_TRAIN_SHA256 = (
    "189cd9c7e38a2312eebf30bb6abac21ecd70644db7b3ae3e5eb2809b9da0957f"
)
EXPECTED_DECISIONS = collections.Counter(
    normal=600, early_fraud=151, explicit_fraud=449, ambiguous=300
)
FRAUD_DECISIONS = {"early_fraud", "explicit_fraud"}
EXPECTED_PAIR_COUNTS = collections.Counter(
    normal_negative=2388,
    fraud_pre_negative=1045,
    fraud_post_positive=1200,
)
EXPECTED_PREFIX_ROWS = 8258


def is_forbidden(path: Path) -> bool:
    parts = path.resolve(strict=False).parts
    return any(
        "restricted" in part.lower() or "test" in part.lower()
        for part in parts
    )


def require_allowlisted_train(path: Path) -> Path:
    resolved = path.resolve()
    if is_forbidden(resolved):
        raise SystemExit(f"refusing Test/restricted input: {resolved}")
    if not resolved.as_posix().endswith(EXPECTED_TRAIN_SUFFIX):
        raise SystemExit(f"refusing non-allowlisted Train input: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_new_output(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if is_forbidden(resolved):
        raise SystemExit(f"refusing Test/restricted output path: {resolved}")
    if not resolved.as_posix().endswith(EXPECTED_OUTPUT_SUFFIX):
        raise SystemExit(f"refusing non-contract output directory: {resolved}")
    if resolved.exists():
        raise SystemExit(f"refusing to overwrite output directory: {resolved}")
    staging = resolved.with_name(resolved.name + ".building")
    if staging.exists():
        raise SystemExit(f"refusing to overwrite staging directory: {staging}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            count += 1
    temporary.replace(path)
    return count


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"{int(turn['turn_id']):03d} "
        f"{str(turn['speaker']).strip()}：{str(turn['text']).strip()}"
        for turn in turns
    )


def validate_turns(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    call_id = str(row.get("call_id") or "")
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{call_id}: missing turns")
    copied = [dict(turn) for turn in turns]
    observed = [int(turn.get("turn_id", -1)) for turn in copied]
    if observed != list(range(1, len(copied) + 1)):
        raise ValueError(f"{call_id}: non-contiguous turn ids")
    if int(row.get("n_full_turns", -1)) != len(copied):
        raise ValueError(f"{call_id}: n_full_turns mismatch")
    if canonical_text(copied) != str(row.get("model_input_text") or ""):
        raise ValueError(f"{call_id}: canonical full text mismatch")
    return copied


def choose_normal_currents(n_turns: int, rng: random.Random) -> list[int]:
    """At most four: first legal pair, full call, then two seeded draws."""
    selected: list[int] = []
    for current in (2, n_turns):
        if 2 <= current <= n_turns and current not in selected:
            selected.append(current)
    remaining = [
        current
        for current in range(2, n_turns + 1)
        if current not in selected
    ]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, 4 - len(selected))])
    return selected


def choose_fraud_currents(
    t_e: int, n_turns: int, rng: random.Random
) -> tuple[list[int], list[int]]:
    """Choose up to two pre-tE negatives and two tE-or-later positives."""
    legal_pre = list(range(2, t_e))
    pre: list[int] = []
    if legal_pre:
        # The boundary-adjacent negative is always retained.
        pre.append(t_e - 1)
        remaining = [current for current in legal_pre if current not in pre]
        rng.shuffle(remaining)
        pre.extend(remaining[:1])

    post: list[int] = []
    for current in (max(2, t_e), n_turns):
        if 2 <= current <= n_turns and current not in post:
            post.append(current)
    return pre, post


def make_pair(
    *,
    row: Mapping[str, Any],
    current: int,
    group: str,
    target: int,
    t_e: int | None,
) -> dict[str, Any]:
    call_id = str(row["call_id"])
    previous = current - 1
    previous_sample_id = f"{call_id}__P{previous:03d}"
    current_sample_id = f"{call_id}__P{current:03d}"
    return {
        "pair_id": f"E2D_GATE__{call_id}__P{previous:03d}_P{current:03d}",
        "split": "train",
        "decision": str(row["decision"]),
        "group": group,
        "target": target,
        "scenario_tag": str(row["scenario_tag"]),
        "call_id": call_id,
        "current_turn": current,
        "previous_turn": previous,
        "tE_turn": t_e,
        "previous_sample_id": previous_sample_id,
        "current_sample_id": current_sample_id,
    }


def blind_prefix_row(
    call_id: str, endpoint: int, turns: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    text = canonical_text(turns[:endpoint])
    return {
        "sample_id": f"{call_id}__P{endpoint:03d}",
        "call_id": call_id,
        "prefix_end_turn": endpoint,
        "n_prefix_turns": endpoint,
        "model_input_text": text,
        "binary_target": 0,
        "binary_loss_mask": 0.0,
        "sample_weight": 0.0,
        "input_view": "v365_train_e2d_prefix_label_blind",
        "input_text_sha256": sha256_text(text),
        "is_future_free": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v365-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.seed != SEED:
        raise SystemExit("E2-D contract fixes seed=202")
    source_path = require_allowlisted_train(args.v365_train)
    output_dir = require_new_output(args.output_dir)
    source_hash_before = sha256_file(source_path)
    if source_hash_before != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            f"V365 Train hash mismatch: {source_hash_before} != "
            f"{EXPECTED_TRAIN_SHA256}"
        )

    rows = read_jsonl(source_path)
    decision_counts = collections.Counter(str(row.get("decision")) for row in rows)
    if len(rows) != 1500 or decision_counts != EXPECTED_DECISIONS:
        raise ValueError(
            f"unexpected Train composition: n={len(rows)} {dict(decision_counts)}"
        )
    if any(str(row.get("split")) != "train" for row in rows):
        raise ValueError("V365 source contains a non-Train row")
    call_ids = [str(row.get("call_id") or "") for row in rows]
    if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(rows):
        raise ValueError("V365 Train call_id is empty or non-unique")

    rng = random.Random(SEED)
    pairs: list[dict[str, Any]] = []
    turns_by_call: dict[str, list[dict[str, Any]]] = {}
    boundary_exclusions: list[dict[str, str]] = []
    ambiguous_excluded = 0

    for row in sorted(rows, key=lambda item: str(item["call_id"])):
        call_id = str(row["call_id"])
        decision = str(row["decision"])
        turns = validate_turns(row)
        turns_by_call[call_id] = turns
        n_turns = len(turns)
        if n_turns < 2:
            boundary_exclusions.append(
                {"call_id": call_id, "reason": "fewer_than_two_turns"}
            )
            continue
        if decision == "ambiguous":
            ambiguous_excluded += 1
            continue
        if decision == "normal":
            if int(row.get("binary_target", -1)) != 0:
                raise ValueError(f"{call_id}: normal binary target mismatch")
            for current in choose_normal_currents(n_turns, rng):
                pairs.append(
                    make_pair(
                        row=row,
                        current=current,
                        group="normal_negative",
                        target=0,
                        t_e=None,
                    )
                )
            continue
        if decision not in FRAUD_DECISIONS:
            raise ValueError(f"{call_id}: unsupported decision={decision}")
        if int(row.get("binary_target", -1)) != 1:
            raise ValueError(f"{call_id}: fraud binary target mismatch")
        try:
            t_e = int(row["tE_turn"])
        except (KeyError, TypeError, ValueError):
            boundary_exclusions.append(
                {"call_id": call_id, "reason": "missing_or_non_integer_tE"}
            )
            continue
        if not 1 <= t_e <= n_turns:
            boundary_exclusions.append(
                {"call_id": call_id, "reason": "tE_out_of_range"}
            )
            continue
        pre, post = choose_fraud_currents(t_e, n_turns, rng)
        for current in pre:
            pairs.append(
                make_pair(
                    row=row,
                    current=current,
                    group="fraud_pre_negative",
                    target=0,
                    t_e=t_e,
                )
            )
        for current in post:
            pairs.append(
                make_pair(
                    row=row,
                    current=current,
                    group="fraud_post_positive",
                    target=1,
                    t_e=t_e,
                )
            )

    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("duplicate pair_id")
    currents_by_call: dict[str, set[int]] = collections.defaultdict(set)
    for pair in pairs:
        call_id = str(pair["call_id"])
        current = int(pair["current_turn"])
        previous = int(pair["previous_turn"])
        if current != previous + 1 or current < 2:
            raise RuntimeError(f"{pair['pair_id']}: non-adjacent pair")
        if current in currents_by_call[call_id]:
            raise RuntimeError(f"{pair['pair_id']}: duplicate current turn in call")
        currents_by_call[call_id].add(current)
        if pair["group"] == "normal_negative":
            if pair["decision"] != "normal" or int(pair["target"]) != 0:
                raise RuntimeError(f"{pair['pair_id']}: invalid normal pair")
        elif pair["group"] == "fraud_pre_negative":
            if int(pair["target"]) != 0 or current >= int(pair["tE_turn"]):
                raise RuntimeError(f"{pair['pair_id']}: invalid pre-tE pair")
        elif pair["group"] == "fraud_post_positive":
            if int(pair["target"]) != 1 or current < int(pair["tE_turn"]):
                raise RuntimeError(f"{pair['pair_id']}: invalid post-tE pair")
        else:
            raise RuntimeError(f"{pair['pair_id']}: unexpected group")

    group_counts = collections.Counter(str(pair["group"]) for pair in pairs)
    if group_counts != EXPECTED_PAIR_COUNTS or len(pairs) != 4633:
        raise ValueError(
            f"frozen pair coverage changed: n={len(pairs)} {dict(group_counts)}"
        )
    if boundary_exclusions:
        raise ValueError(
            "frozen Train unexpectedly contains invalid calls: "
            + json.dumps(boundary_exclusions, ensure_ascii=False)
        )

    endpoint_keys = {
        (str(pair["call_id"]), int(pair[key]))
        for pair in pairs
        for key in ("previous_turn", "current_turn")
    }
    prefix_rows = [
        blind_prefix_row(call_id, endpoint, turns_by_call[call_id])
        for call_id, endpoint in sorted(endpoint_keys)
    ]
    if len(prefix_rows) != EXPECTED_PREFIX_ROWS:
        raise ValueError(
            f"frozen prefix coverage changed: {len(prefix_rows)} != "
            f"{EXPECTED_PREFIX_ROWS}"
        )
    if len({str(row["sample_id"]) for row in prefix_rows}) != len(prefix_rows):
        raise RuntimeError("duplicate label-blind prefix sample_id")
    for row in prefix_rows:
        allowed = {
            "sample_id",
            "call_id",
            "prefix_end_turn",
            "n_prefix_turns",
            "model_input_text",
            "binary_target",
            "binary_loss_mask",
            "sample_weight",
            "input_view",
            "input_text_sha256",
            "is_future_free",
        }
        if set(row) != allowed:
            raise RuntimeError("label-blind input field contract changed")
        if row["binary_loss_mask"] != 0.0 or row["sample_weight"] != 0.0:
            raise RuntimeError("label-blind placeholder is active")
        if sha256_text(str(row["model_input_text"])) != row["input_text_sha256"]:
            raise RuntimeError("prefix text hash mismatch")

    source_hash_after = sha256_file(source_path)
    if source_hash_after != source_hash_before:
        raise RuntimeError("V365 Train changed during E2-D input construction")

    staging = output_dir.with_name(output_dir.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    pair_path = staging / "train_pair_labels.jsonl"
    prefix_path = staging / "train_prefix_inputs_label_blind.jsonl"
    pair_count = write_jsonl(pair_path, pairs)
    prefix_count = write_jsonl(prefix_path, prefix_rows)

    per_call_counts = collections.Counter(
        len(currents) for currents in currents_by_call.values()
    )
    decision_group_counts = collections.Counter(
        (str(pair["decision"]), str(pair["group"])) for pair in pairs
    )
    scenario_group_counts = collections.Counter(
        (str(pair["scenario_tag"]), str(pair["group"])) for pair in pairs
    )
    manifest = {
        "schema_version": "v365.e2d.gate_train_inputs.v1",
        "created_for": "E2-D independent temporal gate; frozen M1 feature extraction",
        "seed": SEED,
        "safety": {
            "source_scope": "V365 Train only",
            "dev_files_read": 0,
            "test_or_restricted_files_read": 0,
            "ambiguous_rows_used": 0,
            "ambiguous_rows_excluded": ambiguous_excluded,
            "original_source_files_modified": False,
            "labels_present_in_prefix_inference_input": False,
        },
        "source": {
            "path": str(source_path),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "rows": len(rows),
            "decision_counts": dict(sorted(decision_counts.items())),
        },
        "selection_policy": {
            "adjacent_pairs_only": True,
            "previous_turn": "current_turn - 1",
            "normal_negative": "current=2, full-call, then up to two seed-202 random unique currents",
            "fraud_pre_negative": "current=tE-1 first, then one seed-202 random unique current; all current<tE",
            "fraud_post_positive": "current=max(2,tE) first, then full-call; all current>=tE",
            "maximum_pairs_per_normal_call": 4,
            "maximum_pre_negative_pairs_per_fraud_call": 2,
            "maximum_post_positive_pairs_per_fraud_call": 2,
            "random_iteration_order": "call_id lexical order with one Python random.Random(202) stream",
            "same_current_turn_not_reused_within_call": True,
            "prefix_causality": "each input contains exactly turns 1..prefix_end_turn",
            "endpoint_inference_rows_deduplicated_by": ["call_id", "prefix_end_turn"],
        },
        "validation": {
            "boundary_exclusions": boundary_exclusions,
            "boundary_exclusion_count": len(boundary_exclusions),
            "usable_calls": len(currents_by_call),
            "pairs": pair_count,
            "unique_prefix_endpoints": prefix_count,
            "group_counts": dict(sorted(group_counts.items())),
            "pair_count_per_call_histogram": {
                str(key): value for key, value in sorted(per_call_counts.items())
            },
            "decision_group_counts": {
                f"{decision}|{group}": count
                for (decision, group), count in sorted(decision_group_counts.items())
            },
            "scenario_group_counts": {
                f"{scenario}|{group}": count
                for (scenario, group), count in sorted(scenario_group_counts.items())
            },
            "pair_ids_unique": True,
            "prefix_sample_ids_unique": True,
            "turn_ids_contiguous": True,
            "full_text_rebuild_exact": True,
        },
        "outputs": {
            "pairs": {
                "path": str(output_dir / pair_path.name),
                "rows": pair_count,
                "sha256": sha256_file(pair_path),
            },
            "prefixes": {
                "path": str(output_dir / prefix_path.name),
                "rows": prefix_count,
                "sha256": sha256_file(prefix_path),
            },
        },
    }
    write_json(staging / "input_manifest.json", manifest)
    # Re-check immediately before publication.  A concurrent creator must win;
    # this process must never replace even an empty pre-existing directory.
    if output_dir.exists():
        raise RuntimeError(f"output directory appeared during build: {output_dir}")
    staging.replace(output_dir)

    print(
        json.dumps(
            {
                "status": "complete",
                "pairs": pair_count,
                "prefixes": prefix_count,
                "group_counts": dict(sorted(group_counts.items())),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
