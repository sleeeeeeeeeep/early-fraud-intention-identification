#!/usr/bin/env python3
"""Build label-blind full-call and cumulative-prefix inputs for V365 Dev only.

This program deliberately accepts exactly one source file.  It does not scan the
dataset directory and it never opens Test or restricted-label files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SOURCE_SUFFIX = (
    "data/processed/v365_2000_final_20260923/data/"
    "v365_calls_dev_final.jsonl"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dev_only(path: Path) -> list[dict[str, Any]]:
    normalized = path.resolve().as_posix()
    if not normalized.endswith(EXPECTED_SOURCE_SUFFIX):
        raise SystemExit(
            "Refusing non-allowlisted input; expected the V365 Dev final JSONL, "
            f"got: {normalized}"
        )
    lowered = normalized.lower()
    if "test" in Path(normalized).name.lower() or "restricted" in lowered:
        raise SystemExit("Refusing Test/restricted-label input")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def canonical_text(turns: Iterable[dict[str, Any]]) -> str:
    return "\n".join(
        f"{int(turn['turn_id']):03d} "
        f"{str(turn['speaker']).strip()}：{str(turn['text']).strip()}"
        for turn in turns
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    digest = hashlib.sha256()
    with temp.open("wb") as handle:
        for row in rows:
            raw = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            handle.write(raw)
            digest.update(raw)
            count += 1
    temp.replace(path)
    return count, digest.hexdigest()


def validate_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 200:
        raise ValueError(f"expected 200 Dev calls, got {len(rows)}")
    counts = Counter(str(row.get("decision")) for row in rows)
    expected = {
        "normal": 80,
        "early_fraud": 15,
        "explicit_fraud": 65,
        "ambiguous": 40,
    }
    if dict(counts) != expected:
        raise ValueError(f"unexpected decision counts: {dict(counts)}")
    call_ids = [str(row.get("call_id") or "") for row in rows]
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if any(not value for value in call_ids + sample_ids):
        raise ValueError("empty call_id/sample_id")
    if len(set(call_ids)) != len(call_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate call_id/sample_id in Dev")

    total_prefixes = 0
    for row in rows:
        decision = str(row["decision"])
        expected_mask = 0 if decision == "ambiguous" else 1
        expected_target = -1 if decision == "ambiguous" else int(decision != "normal")
        if int(row.get("binary_loss_mask", -1)) != expected_mask:
            raise ValueError(f"{row['call_id']}: binary_loss_mask inconsistent with decision")
        if int(row.get("binary_target", -99)) != expected_target:
            raise ValueError(f"{row['call_id']}: binary_target inconsistent with decision")
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{row['call_id']}: missing turns")
        expected_ids = list(range(1, len(turns) + 1))
        actual_ids = [int(turn.get("turn_id", -1)) for turn in turns]
        if actual_ids != expected_ids:
            raise ValueError(
                f"{row['call_id']}: non-contiguous turn ids: {actual_ids[:8]}"
            )
        if int(row.get("n_full_turns", -1)) != len(turns):
            raise ValueError(f"{row['call_id']}: n_full_turns mismatch")
        rebuilt = canonical_text(turns)
        if rebuilt != str(row.get("model_input_text") or ""):
            raise ValueError(f"{row['call_id']}: canonical full text mismatch")
        total_prefixes += len(turns)
    return {
        "calls": len(rows),
        "decision_counts": expected,
        "effective_binary_calls": 160,
        "cumulative_prefixes": total_prefixes,
        "turn_ids_contiguous": True,
        "n_full_turns_consistent": True,
        "final_prefix_equals_fulltext": True,
        "binary_target_and_mask_consistent": True,
    }


def blind_full_rows(rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        text = str(row["model_input_text"])
        yield {
            "sample_id": str(row["sample_id"]),
            "call_id": str(row["call_id"]),
            "model_input_text": text,
            # Placeholders are ignored by model.forward and keep the inference file
            # label-blind.  All scoring joins predictions back to the Dev source.
            "binary_target": 0,
            "binary_loss_mask": 0.0,
            "sample_weight": 0.0,
            "input_view": "v365_dev_fulltext_label_blind",
            "input_text_sha256": sha256_bytes(text.encode("utf-8")),
        }


def blind_prefix_rows(rows: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        call_id = str(row["call_id"])
        turns = row["turns"]
        n_full = len(turns)
        for prefix_end in range(1, n_full + 1):
            text = canonical_text(turns[:prefix_end])
            # The source sample id is not reused because inference output is keyed
            # by sample_id.  The call id plus 1-based turn is deterministic.
            sample_id = f"{call_id}__P{prefix_end:03d}"
            yield {
                "sample_id": sample_id,
                "call_id": call_id,
                "prefix_end_turn": prefix_end,
                "n_prefix_turns": prefix_end,
                "n_full_turns": n_full,
                "model_input_text": text,
                "binary_target": 0,
                "binary_loss_mask": 0.0,
                "sample_weight": 0.0,
                "input_view": "v365_dev_cumulative_prefix_label_blind",
                "input_text_sha256": sha256_bytes(text.encode("utf-8")),
                "is_future_free": True,
            }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(EXPECTED_SOURCE_SUFFIX),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_dev_only(args.source)
    validation = validate_source(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.output_dir / "v365_dev_fulltext_label_blind.jsonl"
    prefix_path = args.output_dir / "v365_dev_cumulative_prefix_label_blind.jsonl"
    full_count, full_hash = write_jsonl(full_path, blind_full_rows(rows))
    prefix_count, prefix_hash = write_jsonl(prefix_path, blind_prefix_rows(rows))
    if full_count != 200 or prefix_count != validation["cumulative_prefixes"]:
        raise AssertionError("derived row count mismatch")

    # Re-read only the last derived prefix of each call and verify text identity.
    final_by_call: dict[str, str] = {}
    with prefix_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if int(item["prefix_end_turn"]) == int(item["n_full_turns"]):
                final_by_call[str(item["call_id"])] = str(item["model_input_text"])
    for row in rows:
        if final_by_call.get(str(row["call_id"])) != str(row["model_input_text"]):
            raise AssertionError(f"{row['call_id']}: final prefix mismatch after write")

    manifest = {
        "schema_version": "v365.dev.zero_train.inputs.v1",
        "scope": "V365 Dev only",
        "source_path": str(args.source),
        "source_sha256": sha256_file(args.source),
        "prohibited_inputs_opened": [],
        "labels_in_inference_inputs": False,
        "validation": validation,
        "outputs": {
            "fulltext": {
                "path": str(full_path),
                "rows": full_count,
                "sha256": full_hash,
            },
            "cumulative_prefix": {
                "path": str(prefix_path),
                "rows": prefix_count,
                "sha256": prefix_hash,
            },
        },
    }
    manifest_path = args.output_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
