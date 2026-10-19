#!/usr/bin/env python3
"""Build a deterministic V362 E1-B binary training mixture.

The physical 6,500-row file contains 60% supervised strict-prefix draws and
40% V361 full-text replay draws.  V362 ambiguous rows are excluded because
their binary_loss_mask is zero.  Repeated prefix draws receive unique training
instance IDs while retaining source_sample_id for auditability.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty JSONL: {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repeated_draws(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    while len(output) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        output.extend(cycle[: count - len(output)])
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix-jsonl",
        type=Path,
        default=Path("runs/v362_e0_binary/inputs/strict_prefix_train.jsonl"),
    )
    parser.add_argument(
        "--replay-jsonl",
        type=Path,
        default=Path("data/processed/v361_longdoc_text_route_20260824/train_v361_longdoc_6500.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/processed/v362_e1b_binary_60prefix_40replay/train_e1b_binary_6500.jsonl"),
    )
    parser.add_argument("--total-rows", type=int, default=6500)
    parser.add_argument("--prefix-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=202)
    args = parser.parse_args()
    if args.total_rows <= 0 or not 0.0 < args.prefix_fraction < 1.0:
        raise ValueError("invalid mixture size/fraction")

    prefix_all = read_jsonl(args.prefix_jsonl)
    replay_all = read_jsonl(args.replay_jsonl)
    prefix_supervised = [
        row
        for row in prefix_all
        if float(row.get("binary_loss_mask", 0.0)) > 0.0
        and int(row.get("binary_target", -1)) in (0, 1)
    ]
    excluded_prefix = [row for row in prefix_all if row not in prefix_supervised]
    prefix_count = round(args.total_rows * args.prefix_fraction)
    replay_count = args.total_rows - prefix_count
    if replay_count > len(replay_all):
        raise ValueError("requested replay rows exceed available rows")

    rng = random.Random(args.seed)
    prefix_draws = repeated_draws(prefix_supervised, prefix_count, rng)
    replay_draws = rng.sample(replay_all, replay_count)
    mixed: list[dict[str, Any]] = []

    for draw_index, source in enumerate(prefix_draws):
        row = dict(source)
        source_id = str(source["sample_id"])
        row.update(
            {
                "sample_id": f"E1B_PREFIX_{draw_index:04d}__{source_id}",
                "source_sample_id": source_id,
                "e1b_source": "strict_prefix",
                "input_view": "strict_prefix",
                "content_group": "v362_strict_prefix",
                "binary_target": int(source["binary_target"]),
                "binary_loss_mask": 1,
                "disable_window_evidence": True,
            }
        )
        mixed.append(row)

    for draw_index, source in enumerate(replay_draws):
        row = dict(source)
        source_id = str(source["sample_id"])
        parent = int(source.get("canonical_label_13", source.get("label_13", 0)))
        row.update(
            {
                "sample_id": f"E1B_REPLAY_{draw_index:04d}__{source_id}",
                "source_sample_id": source_id,
                "e1b_source": "fulltext_replay",
                "input_view": "fulltext_replay",
                "content_group": "v361_fulltext_replay",
                "binary_target": int(parent != 0),
                "binary_loss_mask": 1,
            }
        )
        mixed.append(row)

    rng.shuffle(mixed)
    identifiers = [str(row["sample_id"]) for row in mixed]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("training instance IDs are not unique")
    if len(mixed) != args.total_rows:
        raise ValueError("mixture row count mismatch")
    write_jsonl(args.output_jsonl, mixed)

    source_counter = collections.Counter(str(row["e1b_source"]) for row in mixed)
    binary_counter = collections.Counter(int(row["binary_target"]) for row in mixed)
    by_source_binary: dict[str, dict[str, int]] = {}
    for source_name in sorted(source_counter):
        members = [row for row in mixed if row["e1b_source"] == source_name]
        by_source_binary[source_name] = dict(
            sorted(collections.Counter(str(row["binary_target"]) for row in members).items())
        )
    manifest = {
        "schema_version": "v362.e1b.binary_mixture.v1",
        "seed": args.seed,
        "policy": "physical deterministic mixture; 60% supervised strict prefix, 40% V361 full-text replay",
        "output": str(args.output_jsonl),
        "output_sha256": sha256(args.output_jsonl),
        "rows": len(mixed),
        "source_counts": dict(sorted(source_counter.items())),
        "binary_counts": {str(key): value for key, value in sorted(binary_counter.items())},
        "by_source_binary_counts": by_source_binary,
        "prefix": {
            "input": str(args.prefix_jsonl),
            "input_sha256": sha256(args.prefix_jsonl),
            "all_rows": len(prefix_all),
            "supervised_unique_rows": len(prefix_supervised),
            "ambiguous_or_masked_rows_excluded": len(excluded_prefix),
            "draws": prefix_count,
            "minimum_repetitions_per_source": prefix_count // len(prefix_supervised),
            "maximum_repetitions_per_source": (prefix_count + len(prefix_supervised) - 1) // len(prefix_supervised),
        },
        "replay": {
            "input": str(args.replay_jsonl),
            "input_sha256": sha256(args.replay_jsonl),
            "available_rows": len(replay_all),
            "sampled_without_replacement": replay_count,
        },
    }
    manifest_path = args.output_jsonl.parent / "mixture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
