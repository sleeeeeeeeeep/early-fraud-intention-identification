#!/usr/bin/env python3
"""Build the pre-registered E2-B binary training mixture.

E2-B reuses the already audited E2-A unique V365 Train prefix pool and the
exact E1-B replay subset embedded in the E2-A physical mixture.  It changes
only two training inputs: (1) draw counts make the full 6,500-row physical
mixture exactly class-balanced and (2) each V365 fraud tE row carries a
same-call tE-1 auxiliary view for boundary regularisation.

No Dev, Test, restricted-label, or ambiguous row is read or emitted.
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
from scipy.optimize import linear_sum_assignment


EXPECTED_E2A_DIR_SUFFIX = (
    "data/processed/v365_e2a_te_prefix_60new_40replay_s202"
)
EXPECTED_UNIQUE_NAME = "v365_train_te_prefix_unique_1200.jsonl"
EXPECTED_E2A_MIXTURE_NAME = "train_e2a_binary_6500.jsonl"
EXPECTED_MANIFEST_NAME = "manifest.json"
EXPECTED_TRAIN_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_train_final.jsonl"
)
EXPECTED_DECISIONS = collections.Counter(
    normal=600, early_fraud=151, explicit_fraud=449, ambiguous=300
)
FRAUD_SOURCE = "v365_te_prefix"
REPLAY_SOURCE = "v361_fulltext_replay"


def forbidden(path: Path) -> bool:
    normalized = path.resolve(strict=False).as_posix().lower()
    return "restricted" in normalized or any(
        "test" in part for part in normalized.split("/")
    )


def require_e2a_file(path: Path, filename: str) -> Path:
    resolved = path.resolve()
    if forbidden(resolved):
        raise SystemExit(f"refusing Test/restricted path: {resolved}")
    expected_suffix = f"{EXPECTED_E2A_DIR_SUFFIX}/{filename}"
    if not resolved.as_posix().endswith(expected_suffix):
        raise SystemExit(f"refusing non-allowlisted E2-A input: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_train_file(path: Path) -> Path:
    resolved = path.resolve()
    if forbidden(resolved):
        raise SystemExit(f"refusing Test/restricted path: {resolved}")
    if not resolved.as_posix().endswith(EXPECTED_TRAIN_SUFFIX):
        raise SystemExit(f"refusing non-allowlisted V365 Train path: {resolved}")
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


def proportional_without_replacement(
    rows: list[dict[str, Any]],
    extra_count: int,
    key_fields: tuple[str, ...],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Hamilton-allocate extras across strata, then sample within each stratum."""

    if not 0 <= extra_count <= len(rows):
        raise ValueError("extra_count must permit sampling without replacement")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(field)) for field in key_fields)].append(row)
    quotas = {
        key: extra_count * len(members) / len(rows)
        for key, members in groups.items()
    }
    allocations = {key: int(np.floor(value)) for key, value in quotas.items()}
    remaining = extra_count - sum(allocations.values())
    ranked = sorted(
        groups,
        key=lambda key: (-(quotas[key] - allocations[key]), key),
    )
    for key in ranked[:remaining]:
        allocations[key] += 1
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = list(groups[key])
        rng.shuffle(members)
        take = allocations[key]
        if take > len(members):
            raise ValueError(f"stratum {key} cannot supply {take} unique extras")
        selected.extend(members[:take])
    if len(selected) != extra_count:
        raise RuntimeError("Hamilton allocation did not produce the requested count")
    return selected


def repeated_draws(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count < len(rows):
        raise ValueError(
            f"draw count {count} would omit unique calls from a {len(rows)}-row pool"
        )
    output: list[dict[str, Any]] = []
    while len(output) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        output.extend(cycle[: count - len(output)])
    return output


def source_count(rows: list[dict[str, Any]]) -> collections.Counter[str]:
    return collections.Counter(str(row.get("e2b_source")) for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2a-unique", type=Path, required=True)
    parser.add_argument("--e2a-mixture", type=Path, required=True)
    parser.add_argument("--e2a-manifest", type=Path, required=True)
    parser.add_argument("--v365-train", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, default=6500)
    parser.add_argument("--new-prefix-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()

    unique_path = require_e2a_file(args.e2a_unique, EXPECTED_UNIQUE_NAME)
    e2a_mixture_path = require_e2a_file(
        args.e2a_mixture, EXPECTED_E2A_MIXTURE_NAME
    )
    e2a_manifest_path = require_e2a_file(
        args.e2a_manifest, EXPECTED_MANIFEST_NAME
    )
    train_path = require_train_file(args.v365_train)
    if forbidden(args.tokenizer):
        raise SystemExit(f"refusing Test/restricted tokenizer path: {args.tokenizer}")
    if not args.tokenizer.is_dir():
        raise NotADirectoryError(args.tokenizer)
    if forbidden(args.output_dir):
        raise SystemExit(f"refusing Test/restricted output: {args.output_dir}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    if (
        args.total_rows != 6500
        or abs(args.new_prefix_fraction - 0.60) > 1e-12
        or args.seed != 202
    ):
        raise SystemExit("E2-B contract fixes rows=6500, new fraction=0.60, seed=202")

    e2a_manifest = json.loads(e2a_manifest_path.read_text(encoding="utf-8"))
    expected_unique_sha = e2a_manifest["files"][EXPECTED_UNIQUE_NAME]["sha256"]
    expected_mixture_sha = e2a_manifest["mixture"]["output_sha256"]
    observed_unique_sha = sha256_file(unique_path)
    observed_mixture_sha = sha256_file(e2a_mixture_path)
    if observed_unique_sha != expected_unique_sha:
        raise ValueError("E2-A unique-prefix hash differs from its manifest")
    if observed_mixture_sha != expected_mixture_sha:
        raise ValueError("E2-A mixture hash differs from its manifest")
    if e2a_manifest.get("safety", {}).get("test_or_restricted_paths_read") is not False:
        raise ValueError("E2-A source manifest does not attest Test isolation")

    train_sha_before = sha256_file(train_path)
    train_rows = read_jsonl(train_path)
    train_counts = collections.Counter(str(row.get("decision")) for row in train_rows)
    if len(train_rows) != 1500 or train_counts != EXPECTED_DECISIONS:
        raise ValueError(
            f"unexpected V365 Train composition: n={len(train_rows)} {dict(train_counts)}"
        )
    if any(str(row.get("split")) != "train" for row in train_rows):
        raise ValueError("V365 source contains a non-Train row")
    if len({str(row.get("call_id")) for row in train_rows}) != len(train_rows):
        raise ValueError("V365 Train call_id is not unique")
    normal_sources = [row for row in train_rows if str(row["decision"]) == "normal"]
    normal_source_by_call = {str(row["call_id"]): row for row in normal_sources}
    if len(normal_source_by_call) != 600:
        raise ValueError("expected 600 unique normal V365 Train calls")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    unique_rows = read_jsonl(unique_path)
    if len(unique_rows) != 1200:
        raise ValueError(f"expected 1200 unique E2-A prefixes, got {len(unique_rows)}")
    unique_binary = collections.Counter(int(row["binary_target"]) for row in unique_rows)
    if unique_binary != collections.Counter({0: 600, 1: 600}):
        raise ValueError(f"unexpected unique-prefix classes: {dict(unique_binary)}")
    if any(str(row.get("source_split")) != "train" for row in unique_rows):
        raise ValueError("non-Train source in E2-A unique prefix pool")
    if any(str(row.get("source_fulltext_decision")) == "ambiguous" for row in unique_rows):
        raise ValueError("ambiguous row in E2-A unique prefix pool")

    fraud_unique: list[dict[str, Any]] = []
    normal_unique: list[dict[str, Any]] = []
    boundary_audit: list[dict[str, Any]] = []
    for source in unique_rows:
        row = dict(source)
        row["schema_version"] = "v365.e2b.boundary_prefix.v1"
        row["dataset_version"] = "v365_e2b_boundary_balanced_20261012"
        if int(row["binary_target"]) == 0:
            row["boundary_pair_available"] = False
            normal_unique.append(row)
            continue

        turns = row.get("turns")
        t_e = int(row.get("tE_turn", -1))
        if not isinstance(turns, list) or len(turns) != t_e or t_e < 1:
            raise ValueError(f"{row.get('call_id')}: invalid tE prefix")
        if [int(turn["turn_id"]) for turn in turns] != list(range(1, t_e + 1)):
            raise ValueError(f"{row.get('call_id')}: non-contiguous tE prefix")
        if t_e == 1:
            row.update(
                {
                    "boundary_pair_available": False,
                    "boundary_aux_mask": 0,
                    "boundary_pair_id": None,
                    "boundary_pair_kind": None,
                    "boundary_pair_exclusion_reason": "tE_is_first_turn",
                    "boundary_te_end_turn": 1,
                    "boundary_future_content_used": False,
                }
            )
            fraud_unique.append(row)
            boundary_audit.append(
                {
                    "call_id": row["call_id"],
                    "source_sample_id": row["source_sample_id"],
                    "source_fulltext_decision": row["source_fulltext_decision"],
                    "scenario_tag": row.get("scenario_tag"),
                    "tE_turn": 1,
                    "pair_available": False,
                    "exclusion_reason": "tE_is_first_turn",
                }
            )
            continue
        pre_turns = [dict(turn) for turn in turns[:-1]]
        pre_text = canonical_text(pre_turns)
        if (
            not pre_text
            or max(int(turn["turn_id"]) for turn in pre_turns) != t_e - 1
            or any(line.startswith(f"{t_e:03d} ") for line in pre_text.splitlines())
        ):
            raise ValueError(f"{row.get('call_id')}: malformed tE-1 text")
        row.update(
            {
                "boundary_pair_available": True,
                "boundary_aux_mask": 1,
                "boundary_pair_id": f"E2B_BOUNDARY__{row['call_id']}",
                "boundary_pair_kind": "same_call_te_minus_1_to_te",
                "boundary_gap_turns": 1,
                "boundary_pre_end_turn": t_e - 1,
                "boundary_te_end_turn": t_e,
                "boundary_pre_model_input_text": pre_text,
                "boundary_pre_text_sha256": sha256_text(pre_text),
                "boundary_pre_raw_wordpiece_length": token_length(tokenizer, pre_text),
                "boundary_pre_n_turns": len(pre_turns),
                "boundary_future_content_used": False,
            }
        )
        fraud_unique.append(row)
        boundary_audit.append(
            {
                "call_id": row["call_id"],
                "source_sample_id": row["source_sample_id"],
                "source_fulltext_decision": row["source_fulltext_decision"],
                "scenario_tag": row.get("scenario_tag"),
                "tE_turn": t_e,
                "pair_available": True,
                "pre_end_turn": t_e - 1,
                "pre_text_sha256": row["boundary_pre_text_sha256"],
                "te_text_sha256": row["text_sha256"],
            }
        )

    if len(fraud_unique) != 600 or len(normal_unique) != 600:
        raise ValueError("E2-B unique fraud/normal pools must both contain 600 calls")
    if len({str(row["call_id"]) for row in fraud_unique}) != 600:
        raise ValueError("fraud call_id is not unique")
    if len({str(row["call_id"]) for row in normal_unique}) != 600:
        raise ValueError("normal call_id is not unique")

    e2a_rows = read_jsonl(e2a_mixture_path)
    replay_rows = [
        row for row in e2a_rows if str(row.get("e2a_source")) == REPLAY_SOURCE
    ]
    if len(e2a_rows) != 6500 or len(replay_rows) != 2600:
        raise ValueError(
            f"unexpected E2-A mixture/replay counts: {len(e2a_rows)}/{len(replay_rows)}"
        )
    if len({str(row["source_sample_id"]) for row in replay_rows}) != 2600:
        raise ValueError("E2-A replay source IDs are not 2600 unique rows")
    replay_binary = collections.Counter(int(row["binary_target"]) for row in replay_rows)
    if replay_binary != collections.Counter({0: 691, 1: 1909}):
        raise ValueError(f"unexpected replay class counts: {dict(replay_binary)}")

    new_count = round(args.total_rows * args.new_prefix_fraction)
    replay_count = args.total_rows - new_count
    target_per_class = args.total_rows // 2
    fraud_draw_count = target_per_class - replay_binary[1]
    normal_draw_count = target_per_class - replay_binary[0]
    if (new_count, replay_count, fraud_draw_count, normal_draw_count) != (
        3900,
        2600,
        1341,
        2559,
    ):
        raise ValueError("E2-B class-balance arithmetic changed unexpectedly")

    rng = random.Random(args.seed)
    # Cover every fraud call twice, then Hamilton-allocate the 141 remaining
    # draws over (decision, scenario) strata.  This avoids allowing the order
    # of the JSONL file to distort the small early_fraud subset.
    fraud_draws = [row for row in fraud_unique for _ in range(2)]
    fraud_draws.extend(
        proportional_without_replacement(
            fraud_unique,
            fraud_draw_count - len(fraud_draws),
            ("source_fulltext_decision", "scenario_tag"),
            rng,
        )
    )
    if len(fraud_draws) != fraud_draw_count:
        raise RuntimeError("stratified fraud draw count mismatch")
    fraud_call_draw_counts = collections.Counter(
        str(row["call_id"]) for row in fraud_draws
    )
    if set(fraud_call_draw_counts.values()) - {2, 3}:
        raise ValueError("every fraud call must be drawn exactly two or three times")

    normal_template_by_call = {str(row["call_id"]): row for row in normal_unique}
    if set(normal_template_by_call) != set(normal_source_by_call):
        raise ValueError("E2-A normal template calls differ from V365 Train normals")
    fraud_by_scenario: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    normal_by_scenario: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in fraud_draws:
        fraud_by_scenario[str(row.get("scenario_tag"))].append(row)
    for row in normal_sources:
        normal_by_scenario[str(row.get("scenario_tag"))].append(row)
    if set(fraud_by_scenario) != set(normal_by_scenario):
        raise ValueError("fraud and normal scenario sets differ")

    candidate_cache: dict[tuple[str, int], dict[str, Any] | None] = {}

    def normal_candidate(source: Mapping[str, Any], end_turn: int) -> dict[str, Any] | None:
        key = (str(source["call_id"]), int(end_turn))
        if key in candidate_cache:
            return candidate_cache[key]
        turns = source.get("turns")
        if not isinstance(turns, list) or not 1 <= end_turn <= len(turns):
            candidate_cache[key] = None
            return None
        prefix_turns = [dict(turn) for turn in turns[:end_turn]]
        if [int(turn["turn_id"]) for turn in prefix_turns] != list(
            range(1, end_turn + 1)
        ):
            raise ValueError(f"normal {source['call_id']} has invalid turn sequence")
        text = canonical_text(prefix_turns)
        result = {
            "turns": prefix_turns,
            "text": text,
            "token_length": token_length(tokenizer, text),
            "speaker": str(prefix_turns[-1]["speaker"]).strip(),
        }
        candidate_cache[key] = result
        return result

    scenario_normal_draws: list[dict[str, Any]] = []
    scenario_matching_pairs: list[dict[str, Any]] = []
    normal_pair_use_counts: collections.Counter[str] = collections.Counter()
    impossible_cost = 1e12
    for scenario in sorted(fraud_by_scenario):
        fraud_members = list(fraud_by_scenario[scenario])
        normal_members = sorted(
            normal_by_scenario[scenario], key=lambda row: str(row["call_id"])
        )
        demand = len(fraud_members)
        if not normal_members:
            raise ValueError(f"scenario {scenario} has no normal controls")
        base_capacity, remainder = divmod(demand, len(normal_members))
        if base_capacity < 1:
            raise ValueError(f"scenario {scenario} demand cannot cover every normal call")
        # Enforce floor(demand / normals) uses for every normal call and let
        # the assignment solver choose which ``remainder`` calls receive one
        # extra use.  A dominant reward on mandatory slots implements the
        # lower bound while token distance decides the optional capacity.
        slots: list[tuple[dict[str, Any], bool]] = []
        for source in normal_members:
            slots.extend((source, True) for _ in range(base_capacity))
            if remainder:
                slots.append((source, False))
        mandatory_slot_count = base_capacity * len(normal_members)
        expected_slots = mandatory_slot_count + (len(normal_members) if remainder else 0)
        if len(slots) != expected_slots or mandatory_slot_count > demand:
            raise RuntimeError(f"scenario {scenario} capacity arithmetic mismatch")

        mandatory_reward = 1e9
        cost = np.full((demand, len(slots)), impossible_cost, dtype=np.float64)
        for fraud_index, fraud in enumerate(fraud_members):
            target_turn = int(fraud["tE_turn"])
            target_tokens = int(fraud["raw_wordpiece_length"])
            target_speaker = str(fraud["tE_end_speaker"])
            for slot_index, (source, mandatory) in enumerate(slots):
                candidate = normal_candidate(source, target_turn)
                if candidate is None or candidate["speaker"] != target_speaker:
                    continue
                tie_breaker = slot_index * 1e-9
                cost[fraud_index, slot_index] = (
                    abs(int(candidate["token_length"]) - target_tokens)
                    - (mandatory_reward if mandatory else 0.0)
                    + tie_breaker
                )
        fraud_indices, slot_indices = linear_sum_assignment(cost)
        if len(fraud_indices) != demand or any(
            cost[int(i), int(j)] >= impossible_cost for i, j in zip(fraud_indices, slot_indices)
        ):
            raise ValueError(f"scenario {scenario} has no exact-turn/role assignment")
        selected_slot_indices = {int(index) for index in slot_indices}
        mandatory_indices = {
            index for index, (_, mandatory) in enumerate(slots) if mandatory
        }
        if not mandatory_indices.issubset(selected_slot_indices):
            raise ValueError(f"scenario {scenario} failed normal-call lower capacity")
        if len(selected_slot_indices - mandatory_indices) != remainder:
            raise ValueError(f"scenario {scenario} optional capacity mismatch")

        for fraud_index, slot_index in zip(fraud_indices, slot_indices):
            fraud = fraud_members[int(fraud_index)]
            source, _mandatory = slots[int(slot_index)]
            call_id = str(source["call_id"])
            target_turn = int(fraud["tE_turn"])
            candidate = normal_candidate(source, target_turn)
            if candidate is None:
                raise RuntimeError("selected normal candidate unexpectedly missing")
            template = dict(normal_template_by_call[call_id])
            template.update(
                {
                    "schema_version": "v365.e2b.scenario_normal_prefix.v1",
                    "dataset_version": "v365_e2b_boundary_balanced_20261012",
                    "sample_id": (
                        f"V365_E2B_SCENARIO_NORMAL__{scenario}__"
                        f"{len(scenario_normal_draws):04d}__{call_id}"
                    ),
                    "turns": candidate["turns"],
                    "n_full_turns": target_turn,
                    "n_prefix_turns": target_turn,
                    "prefix_end_turn": target_turn,
                    "prefix_text": candidate["text"],
                    "model_input_text": candidate["text"],
                    "text": candidate["text"],
                    "prefix_text_hash": sha256_text(candidate["text"]),
                    "text_sha256": sha256_text(candidate["text"]),
                    "raw_wordpiece_length": int(candidate["token_length"]),
                    "prefix_kind": "normal_same_scenario_turn_matched_to_fraud_tE",
                    "prefix_relation": "same_scenario_matched_normal",
                    "input_view": "same_scenario_turn_matched_normal_prefix",
                    "matched_fraud_call_id": fraud["call_id"],
                    "matched_fraud_scenario_tag": fraud["scenario_tag"],
                    "matched_target_turns": target_turn,
                    "matched_target_wordpiece_length": int(
                        fraud["raw_wordpiece_length"]
                    ),
                    "e2b_normal_role": "same_scenario_fraud_draw_control",
                    "boundary_pair_available": False,
                    "future_content_used": False,
                }
            )
            scenario_normal_draws.append(template)
            normal_pair_use_counts[call_id] += 1
            scenario_matching_pairs.append(
                {
                    "scenario_tag": scenario,
                    "fraud_call_id": fraud["call_id"],
                    "fraud_source_decision": fraud["source_fulltext_decision"],
                    "fraud_tE_turn": target_turn,
                    "fraud_wordpiece_length": int(fraud["raw_wordpiece_length"]),
                    "normal_call_id": call_id,
                    "normal_prefix_end_turn": target_turn,
                    "normal_wordpiece_length": int(candidate["token_length"]),
                    "absolute_token_delta": abs(
                        int(candidate["token_length"])
                        - int(fraud["raw_wordpiece_length"])
                    ),
                    "same_endpoint_speaker": True,
                }
            )

    if len(scenario_normal_draws) != fraud_draw_count:
        raise ValueError("same-scenario normal controls do not equal fraud draws")
    if set(normal_pair_use_counts) != set(normal_source_by_call):
        raise ValueError("same-scenario matching did not cover every normal call")
    if min(normal_pair_use_counts.values()) < 1 or max(normal_pair_use_counts.values()) > 5:
        raise ValueError("normal-control capacity is outside the pre-registered 1..5 range")

    # The remaining 1,218 normal draws preserve broad normal-call coverage and
    # the E2-A length/turn controls: every unique normal prefix appears twice,
    # plus 18 scenario-proportional extras.
    normal_coverage_draws = [row for row in normal_unique for _ in range(2)]
    normal_coverage_draws.extend(
        proportional_without_replacement(
            normal_unique,
            normal_draw_count - len(scenario_normal_draws) - len(normal_coverage_draws),
            ("scenario_tag",),
            rng,
        )
    )
    for row in normal_coverage_draws:
        row["e2b_normal_role"] = "broad_normal_coverage"
        row["boundary_pair_available"] = False
    normal_draws = scenario_normal_draws + normal_coverage_draws
    if len(normal_draws) != normal_draw_count:
        raise RuntimeError("E2-B normal draw count mismatch")

    mixed: list[dict[str, Any]] = []
    for draw_index, source in enumerate(fraud_draws + normal_draws):
        row = dict(source)
        source_prefix_id = str(source["sample_id"])
        row.update(
            {
                "sample_id": f"E2B_PREFIX_{draw_index:04d}__{source_prefix_id}",
                "source_prefix_sample_id": source_prefix_id,
                "e2a_origin_source": row.pop("e2a_source", None),
                "e2b_source": FRAUD_SOURCE,
            }
        )
        mixed.append(row)
    for draw_index, source in enumerate(replay_rows):
        row = dict(source)
        original_source_id = str(source["source_sample_id"])
        row.update(
            {
                "sample_id": f"E2B_REPLAY_{draw_index:04d}__{original_source_id}",
                "source_sample_id": original_source_id,
                "e2a_origin_source": row.pop("e2a_source", None),
                "e2b_source": REPLAY_SOURCE,
                "input_view": "fulltext_replay",
                "content_group": "v361_fulltext_replay",
                "binary_loss_mask": 1,
            }
        )
        mixed.append(row)
    rng.shuffle(mixed)

    if len(mixed) != 6500 or len({str(row["sample_id"]) for row in mixed}) != 6500:
        raise ValueError("final E2-B mixture size or sample IDs are invalid")
    final_binary = collections.Counter(int(row["binary_target"]) for row in mixed)
    if final_binary != collections.Counter({0: 3250, 1: 3250}):
        raise ValueError(f"E2-B mixture is not exactly balanced: {dict(final_binary)}")
    expected_sources = collections.Counter(
        {FRAUD_SOURCE: 3900, REPLAY_SOURCE: 2600}
    )
    if source_count(mixed) != expected_sources:
        raise ValueError(f"unexpected E2-B sources: {dict(source_count(mixed))}")
    paired_rows = [row for row in mixed if row.get("boundary_pair_available")]
    unpaired_v365_fraud = [
        row
        for row in mixed
        if row.get("e2b_source") == FRAUD_SOURCE
        and int(row["binary_target"]) == 1
        and not row.get("boundary_pair_available")
    ]
    if len(paired_rows) + len(unpaired_v365_fraud) != fraud_draw_count:
        raise ValueError("paired/unpaired V365 fraud draw accounting mismatch")
    if any(
        int(row.get("tE_turn", -1)) != 1
        or row.get("boundary_pair_exclusion_reason") != "tE_is_first_turn"
        for row in unpaired_v365_fraud
    ):
        raise ValueError("only tE=1 fraud rows may lack an auxiliary boundary view")
    if any(int(row["boundary_pre_end_turn"]) + 1 != int(row["tE_turn"]) for row in paired_rows):
        raise ValueError("a boundary auxiliary view is not exactly tE-1")
    if any(
        float(row.get("binary_loss_mask", 0.0)) <= 0.0
        or int(row.get("binary_target", -1)) not in (0, 1)
        for row in mixed
    ):
        raise ValueError("masked or invalid binary row in E2-B mixture")

    train_sha_after = sha256_file(train_path)
    if train_sha_after != train_sha_before:
        raise RuntimeError("V365 Train changed while constructing E2-B data")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    boundary_path = args.output_dir / "v365_train_fraud_boundary_audit.jsonl"
    matching_path = args.output_dir / "v365_train_same_scenario_normal_matching_1341.jsonl"
    mixture_path = args.output_dir / "train_e2b_binary_6500.jsonl"
    write_jsonl(boundary_path, boundary_audit)
    write_jsonl(matching_path, scenario_matching_pairs)
    write_jsonl(mixture_path, mixed)

    matching_token_deltas = [
        int(row["absolute_token_delta"]) for row in scenario_matching_pairs
    ]
    pair_use_histogram = collections.Counter(normal_pair_use_counts.values())
    weighted_binary_counts = {
        str(label): float(
            sum(
                float(row.get("sample_weight", 1.0))
                for row in mixed
                if int(row["binary_target"]) == label
            )
        )
        for label in (0, 1)
    }

    final_by_source: dict[str, dict[str, int]] = {}
    for name in sorted(expected_sources):
        members = [row for row in mixed if row["e2b_source"] == name]
        final_by_source[name] = {
            str(key): value
            for key, value in sorted(
                collections.Counter(int(row["binary_target"]) for row in members).items()
            )
        }
    manifest = {
        "schema_version": "v365.e2b.boundary_balanced_mixture.v1",
        "created_for": "E2-B binary-only boundary-calibrated training",
        "seed": args.seed,
        "safety": {
            "sources": [
                str(train_path),
                str(unique_path),
                str(e2a_mixture_path),
                str(e2a_manifest_path),
            ],
            "source_provenance": "V365 Train plus its audited E2-A derivatives only",
            "dev_files_read": 0,
            "test_or_restricted_files_read": 0,
            "ambiguous_rows_used": 0,
            "original_source_files_modified": False,
        },
        "source_integrity": {
            "e2a_unique_sha256": observed_unique_sha,
            "e2a_mixture_sha256": observed_mixture_sha,
            "e2a_manifest_sha256": sha256_file(e2a_manifest_path),
            "v365_train_sha256_before": train_sha_before,
            "v365_train_sha256_after": train_sha_after,
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
        },
        "unique_pool": {
            "rows": len(unique_rows),
            "fraud_calls": len(fraud_unique),
            "normal_calls": len(normal_unique),
            "all_unique_calls_covered_by_draws": True,
        },
        "boundary_auxiliary": {
            "policy": "same-call cumulative prefix ending exactly at tE-1",
            "eligible_unique_fraud_calls": len(boundary_audit),
            "unique_pairs": sum(bool(row["pair_available"]) for row in boundary_audit),
            "unique_unpaired_te1": sum(not bool(row["pair_available"]) for row in boundary_audit),
            "physical_paired_draws": len(paired_rows),
            "physical_unpaired_te1_draws": len(unpaired_v365_fraud),
            "hard_normal_label_assigned": False,
            "included_in_physical_row_count": False,
            "future_content_used": False,
            "limitation": "local tE-1 to tE boundary only; does not constrain every earlier prefix",
        },
        "same_scenario_normal_control": {
            "policy": "one capacity-balanced same-scenario, exact-turn and endpoint-role normal control per fraud draw",
            "matched_draws": len(scenario_normal_draws),
            "unique_normal_calls_used": len(normal_pair_use_counts),
            "normal_use_count_histogram": {
                str(k): v for k, v in sorted(pair_use_histogram.items())
            },
            "minimum_use_count": min(normal_pair_use_counts.values()),
            "maximum_use_count": max(normal_pair_use_counts.values()),
            "token_delta": {
                "mean": float(statistics.fmean(matching_token_deltas)),
                "median": float(statistics.median(matching_token_deltas)),
                "p90": float(np.quantile(matching_token_deltas, 0.9)),
                "max": max(matching_token_deltas),
            },
            "broad_normal_coverage_draws": len(normal_coverage_draws),
        },
        "replay_control": {
            "policy": "exact same 2,600 replay rows embedded in E2-A/E1-B",
            "rows": len(replay_rows),
            "unique_source_ids": len({str(row["source_sample_id"]) for row in replay_rows}),
            "binary_counts": {str(k): v for k, v in sorted(replay_binary.items())},
        },
        "mixture": {
            "path": str(mixture_path.resolve()),
            "sha256": sha256_file(mixture_path),
            "rows": len(mixed),
            "new_prefix_fraction": args.new_prefix_fraction,
            "source_counts": dict(sorted(source_count(mixed).items())),
            "binary_counts": {str(k): v for k, v in sorted(final_binary.items())},
            "weighted_binary_counts": weighted_binary_counts,
            "by_source_binary_counts": final_by_source,
            "new_fraud_draws": fraud_draw_count,
            "new_normal_draws": normal_draw_count,
            "replay_draws": replay_count,
        },
        "files": {
            boundary_path.name: {
                "rows": len(boundary_audit),
                "sha256": sha256_file(boundary_path),
            },
            matching_path.name: {
                "rows": len(scenario_matching_pairs),
                "sha256": sha256_file(matching_path),
            },
            mixture_path.name: {
                "rows": len(mixed),
                "sha256": sha256_file(mixture_path),
            },
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    report = f"""# V365 E2-B 训练数据审计

- 输入仅为已审计的 V365 Train 派生 E2-A 前缀与同一批 E1-B replay；Dev/Test/restricted 读取数均为0。
- 物理主训练集：6500条；V365前缀3900条、V361回放2600条。
- 全集类别严格平衡：fraud 3250、normal 3250。
- V365抽样：fraud tE前缀{fraud_draw_count}次、normal前缀{normal_draw_count}次；其中{len(scenario_normal_draws)}条为同场景、同轮数、同结束说话人控制，另{len(normal_coverage_draws)}条保持全部normal call广覆盖；600+600个唯一call全部覆盖。
- 同场景normal控制的token绝对差：mean={statistics.fmean(matching_token_deltas):.2f}、median={statistics.median(matching_token_deltas):.1f}、P90={np.quantile(matching_token_deltas, 0.9):.1f}、max={max(matching_token_deltas)}；单个normal call配对使用1至{max(normal_pair_use_counts.values())}次。
- 边界辅助：{sum(bool(row['pair_available']) for row in boundary_audit)}个 tE>1 的唯一fraud call由同通话第1..tE-1轮构造；{sum(not bool(row['pair_available']) for row in boundary_audit)}个 tE=1 call无合法前缀并记录排除。物理训练中{len(paired_rows)}个draw带辅助视图、{len(unpaired_v365_fraud)}个tE=1 draw只使用主CE。辅助视图不计入6500条，也不硬标为normal。
- replay：复用E2-A中嵌入的同一批2600条（fraud {replay_binary[1]}、normal {replay_binary[0]}）。
- 完整哈希、抽样和安全断言见 `manifest.json`。
"""
    (args.output_dir / "DATASET_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
