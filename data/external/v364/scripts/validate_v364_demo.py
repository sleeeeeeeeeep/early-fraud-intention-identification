from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CALLS_PATH = DATA / "v364_calls_demo.jsonl"
REPORT_PATH = DATA / "v364_validation_report.json"
MANIFEST_PATH = ROOT / "v364_manifest.json"

DECISION_MAP = {
    "normal": (0, 0, 1),
    "ambiguous": (1, -1, 0),
    "early_fraud": (2, 1, 1),
    "explicit_fraud": (3, 1, 1),
}

OPERATION_TYPES = {
    "transfer",
    "otp",
    "download",
    "screen_share",
    "remote_control",
    "bank_operation",
    "unofficial_link",
    "account_or_password",
    "identity_document",
    "cash_gold_giftcard_or_courier",
    "crypto_asset",
    "sim_or_esim_operation",
}

REQUIRED_FIELDS = {
    "schema_version",
    "dataset_version",
    "policy_version",
    "call_id",
    "sample_id",
    "source_sample_id",
    "content_group_id",
    "duplicate_cluster_id",
    "split",
    "source_type",
    "source_name",
    "source_record_sha256",
    "text_sha256",
    "turns",
    "n_full_turns",
    "prefix_end_turn",
    "n_prefix_turns",
    "prefix_text",
    "model_input_text",
    "prefix_text_hash",
    "normalized_text_hash",
    "language",
    "scenario_tag",
    "scenario_pair_id",
    "claimed_identity_type",
    "length_bucket",
    "evidence_position_bin",
    "final_label",
    "final_binary_label",
    "decision",
    "stage_target",
    "binary_target",
    "binary_loss_mask",
    "stage_loss_mask",
    "sample_weight",
    "prefix_relation",
    "tS_turn",
    "tE_turn",
    "tA_turn",
    "tFinal_turn",
    "early_detectable",
    "explicit_operation_present",
    "operation_events",
    "core_evidence",
    "supporting_evidence",
    "insufficient_reason",
    "boundary_rationale",
    "turn_stage_labels",
    "annotation_method",
    "annotator_id",
    "annotation_model",
    "annotation_prompt_version",
    "reviewer_id",
    "adjudicator_id",
    "confidence",
    "review_required",
    "review_status",
    "disagreement_type",
    "future_content_used",
    "label_leakage_flag",
    "quality_flags",
    "annotation_timestamp",
}


def digest_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def render_text(turns: list[dict]) -> str:
    return "\n".join(f"{turn['turn_id']:03d} {turn['speaker']}：{turn['text']}" for turn in turns)


def bigram_set(text: str) -> set[str]:
    value = normalize(text)
    return {value[index : index + 2] for index in range(max(0, len(value) - 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    records = [json.loads(line) for line in CALLS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(len(records) == 30, f"expected 30 calls, got {len(records)}")
    call_ids: set[str] = set()
    sample_ids: set[str] = set()
    text_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    pairs: dict[str, set[str]] = defaultdict(set)
    decision_counts = Counter()
    full_texts: list[tuple[str, str]] = []

    for record in records:
        call_id = record.get("call_id", "<missing>")
        check(REQUIRED_FIELDS <= set(record), f"{call_id}: missing required fields {sorted(REQUIRED_FIELDS - set(record))}")
        check(call_id not in call_ids, f"duplicate call_id: {call_id}")
        call_ids.add(call_id)
        check(record["sample_id"] not in sample_ids, f"duplicate sample_id: {record['sample_id']}")
        sample_ids.add(record["sample_id"])
        check(record["text_sha256"] not in text_hashes, f"duplicate text hash: {call_id}")
        text_hashes.add(record["text_sha256"])
        check(record["normalized_text_hash"] not in normalized_hashes, f"duplicate normalized hash: {call_id}")
        normalized_hashes.add(record["normalized_text_hash"])
        check(record["split"] == "demo", f"{call_id}: split must be demo")
        check(record["source_type"] == "synthetic_authored", f"{call_id}: unexpected source type")
        check(record["future_content_used"] is False, f"{call_id}: future content used")
        check(record["label_leakage_flag"] is False, f"{call_id}: label leakage flag")
        check(record["review_required"] is False, f"{call_id}: unresolved review requirement")

        turns = record["turns"]
        turn_ids = [turn["turn_id"] for turn in turns]
        check(turn_ids == list(range(1, len(turns) + 1)), f"{call_id}: non-sequential turn ids")
        check(all(turn["speaker"] in {"来电人", "接电人"} for turn in turns), f"{call_id}: invalid speaker")
        check(
            all(turns[index]["speaker"] != turns[index - 1]["speaker"] for index in range(1, len(turns))),
            f"{call_id}: adjacent turns have the same speaker",
        )
        check(all(turn["text"].strip() == turn["text"] and turn["text"] for turn in turns), f"{call_id}: blank/dirty turn")
        check(record["n_full_turns"] == len(turns), f"{call_id}: n_full_turns mismatch")
        check(record["prefix_end_turn"] == len(turns), f"{call_id}: prefix_end_turn mismatch")
        check(record["n_prefix_turns"] == len(turns), f"{call_id}: n_prefix_turns mismatch")
        check(record["tFinal_turn"] == len(turns), f"{call_id}: tFinal mismatch")
        full_text = render_text(turns)
        check(record["prefix_text"] == full_text, f"{call_id}: prefix_text mismatch")
        check(record["model_input_text"] == full_text, f"{call_id}: model_input_text mismatch")
        check(record["text_sha256"] == digest_text(full_text), f"{call_id}: text hash mismatch")
        check(record["prefix_text_hash"] == digest_text(full_text), f"{call_id}: prefix hash mismatch")
        check(record["normalized_text_hash"] == digest_text(normalize(full_text)), f"{call_id}: normalized hash mismatch")
        check(
            not re.search(r"这是诈骗|虚构工单|诈骗话术|诈骗样本|冒充某", full_text),
            f"{call_id}: label-leaking wording in dialogue",
        )
        full_texts.append((call_id, full_text))

        decision = record["decision"]
        check(decision in DECISION_MAP, f"{call_id}: invalid decision")
        expected_stage, expected_binary, expected_mask = DECISION_MAP[decision]
        check(record["stage_target"] == expected_stage, f"{call_id}: stage target mismatch")
        check(record["binary_target"] == expected_binary, f"{call_id}: binary target mismatch")
        check(record["binary_loss_mask"] == expected_mask, f"{call_id}: binary mask mismatch")
        expected_final = 1 if decision in {"early_fraud", "explicit_fraud"} else 0 if decision == "normal" else -1
        check(record["final_binary_label"] == expected_final, f"{call_id}: final binary mismatch")
        check(record["early_detectable"] == (decision in {"early_fraud", "explicit_fraud"}), f"{call_id}: early_detectable mismatch")
        if decision == "normal":
            check(record["prefix_relation"] == "clean_normal", f"{call_id}: normal prefix relation")
            check(record["tS_turn"] is None and record["tE_turn"] is None and record["tA_turn"] is None, f"{call_id}: normal time marker")
            check(not record["core_evidence"], f"{call_id}: normal has core evidence")
            check(not record["operation_events"], f"{call_id}: normal has operation events")
        elif decision == "ambiguous":
            check(record["tS_turn"] is not None, f"{call_id}: ambiguous missing tS")
            check(record["tE_turn"] is None and record["tA_turn"] is None, f"{call_id}: ambiguous has fraud time")
            check(bool(record["insufficient_reason"]), f"{call_id}: ambiguous lacks reason")
            check(not record["core_evidence"], f"{call_id}: ambiguous has core evidence")
        else:
            check(isinstance(record["tS_turn"], int) and isinstance(record["tE_turn"], int), f"{call_id}: fraud time missing")
            check(1 <= record["tS_turn"] <= record["tE_turn"] <= len(turns), f"{call_id}: fraud time order")
            check(bool(record["core_evidence"]), f"{call_id}: fraud lacks core evidence")
            if decision == "early_fraud":
                check(record["tA_turn"] is None, f"{call_id}: early fraud has tA")
                check(not record["operation_events"], f"{call_id}: early fraud has explicit operation")
            else:
                check(isinstance(record["tA_turn"], int), f"{call_id}: explicit fraud missing tA")
                check(bool(record["operation_events"]), f"{call_id}: explicit fraud lacks operation event")
                check(record["tA_turn"] == min(event["turn_id"] for event in record["operation_events"]), f"{call_id}: tA not earliest operation")

        by_id = {turn["turn_id"]: turn for turn in turns}
        for evidence in record["core_evidence"] + record["supporting_evidence"]:
            turn_id = evidence["turn_id"]
            check(turn_id in by_id, f"{call_id}: evidence turn missing")
            if turn_id in by_id:
                check(by_id[turn_id]["speaker"] == "来电人", f"{call_id}: evidence not caller speech")
                check(evidence["quote"] in by_id[turn_id]["text"], f"{call_id}: evidence quote mismatch")
            check(turn_id <= record["prefix_end_turn"], f"{call_id}: evidence beyond prefix")
        if decision in {"early_fraud", "explicit_fraud"}:
            check(any(item["turn_id"] == record["tE_turn"] for item in record["core_evidence"]), f"{call_id}: no core evidence at tE")

        present = record["explicit_operation_present"]
        check(set(present) == OPERATION_TYPES, f"{call_id}: operation flag keys mismatch")
        event_types = {event["operation_type"] for event in record["operation_events"]}
        check({name for name, value in present.items() if value} == event_types, f"{call_id}: operation flags/events mismatch")
        for event in record["operation_events"]:
            check(event["operation_type"] in OPERATION_TYPES, f"{call_id}: unknown operation type")
            check(event["turn_id"] in by_id, f"{call_id}: operation turn missing")
            if event["turn_id"] in by_id:
                check(by_id[event["turn_id"]]["speaker"] == "来电人", f"{call_id}: operation not caller speech")
                check(event["quote"] in by_id[event["turn_id"]]["text"], f"{call_id}: operation quote mismatch")

        trajectory = record["turn_stage_labels"]
        check(len(trajectory) == len(turns), f"{call_id}: trajectory length mismatch")
        check([item["prefix_end_turn"] for item in trajectory] == turn_ids, f"{call_id}: trajectory turn mismatch")
        check(trajectory[-1]["decision"] == decision, f"{call_id}: trajectory final decision mismatch")
        check(all(trajectory[index]["stage_target"] <= trajectory[index + 1]["stage_target"] for index in range(len(trajectory) - 1)), f"{call_id}: stage trajectory regresses")
        if record["tE_turn"] is not None:
            check(trajectory[record["tE_turn"] - 1]["decision"] in {"early_fraud", "explicit_fraud"}, f"{call_id}: tE stage mismatch")
            if record["tE_turn"] > 1:
                check(trajectory[record["tE_turn"] - 2]["decision"] in {"normal", "ambiguous"}, f"{call_id}: pre-tE already fraud")
        if record["tA_turn"] is not None:
            check(trajectory[record["tA_turn"] - 1]["decision"] == "explicit_fraud", f"{call_id}: tA stage mismatch")

        pairs[record["scenario_pair_id"]].add(record["final_label"])
        decision_counts[decision] += 1

    check(decision_counts == Counter({"normal": 12, "ambiguous": 6, "early_fraud": 5, "explicit_fraud": 7}), f"decision counts mismatch: {dict(decision_counts)}")
    check(len(pairs) == 12, f"expected 12 scenario pairs, got {len(pairs)}")
    for pair, labels in sorted(pairs.items()):
        check({"fraud", "normal"}.issubset(labels), f"{pair}: missing matched fraud/normal pair")

    near_duplicates = []
    grams = [(call_id, bigram_set(text)) for call_id, text in full_texts]
    for index, (left_id, left_grams) in enumerate(grams):
        for right_id, right_grams in grams[index + 1 :]:
            similarity = jaccard(left_grams, right_grams)
            if similarity >= 0.80:
                near_duplicates.append({"left": left_id, "right": right_id, "bigram_jaccard": round(similarity, 4)})
    check(not near_duplicates, f"near-duplicate calls found: {near_duplicates}")

    statistics = json.loads((DATA / "v364_statistics.json").read_text(encoding="utf-8"))
    check(statistics["calls"] == len(records), "statistics call count mismatch")
    check(statistics["matched_fraud_normal_scenario_pairs"] == 12, "statistics paired coverage mismatch")
    check(bool(statistics["demo_limitations"]), "demo limitations missing")
    warnings.extend(statistics["demo_limitations"])

    report = {
        "schema_version": "v364.section8_demo.validation.v1",
        "dataset_version": records[0]["dataset_version"] if records else None,
        "status": "pass" if not errors else "fail",
        "checks_executed": checks,
        "errors": errors,
        "warnings": warnings,
        "calls": len(records),
        "unique_call_ids": len(call_ids),
        "unique_text_hashes": len(text_hashes),
        "decision_counts": dict(sorted(decision_counts.items())),
        "scenario_pairs": len(pairs),
        "near_duplicate_pairs": near_duplicates,
        "strict_client_boundary_applied": True,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_files = [
        ROOT / "README.md",
        ROOT / "V364_SECTION8_BOUNDARY_POLICY.md",
        ROOT / "V364_REVIEW_REPORT.md",
        ROOT / "scripts" / "build_v364_demo.py",
        ROOT / "scripts" / "validate_v364_demo.py",
        CALLS_PATH,
        DATA / "v364_statistics.json",
        REPORT_PATH,
    ]
    manifest = {
        "schema_version": "v364.section8_demo.manifest.v1",
        "dataset_version": records[0]["dataset_version"] if records else None,
        "created_at": records[0]["annotation_timestamp"] if records else None,
        "files": [
            {
                "file": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest_bytes(path),
            }
            for path in manifest_files
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
