from __future__ import annotations

import collections
import hashlib
import json
import re
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORT = DATA / "v365_validation_report.json"
MANIFEST = ROOT / "v365_manifest.json"
FILES = {
    "train": DATA / "v365_calls_train.jsonl",
    "dev": DATA / "v365_calls_dev.jsonl",
    "stress": DATA / "v365_calls_stress.jsonl",
    "blind": DATA / "v365_calls_blind_input.jsonl",
    "blind_labels": DATA / "restricted_v365_calls_blind_labels.jsonl",
}
EXPECTED = {"train": 1500, "dev": 200, "stress": 150, "blind": 150}
EXPECTED_LABELS = {"fraud": 800, "normal": 800, "ambiguous": 400}
BLIND_ALLOWED = {
    "schema_version", "dataset_version", "policy_version", "call_id", "sample_id",
    "split", "turns", "n_full_turns", "prefix_end_turn", "n_prefix_turns",
    "prefix_text", "model_input_text", "text_sha256", "prefix_text_hash",
    "normalized_text_hash", "language", "scenario_tag", "claimed_identity_type",
    "length_bucket", "tFinal_turn",
}
META_LEAK_FIELDS = {
    "evidence_position_bin", "final_label", "final_binary_label", "decision", "stage_target",
    "binary_target", "binary_loss_mask", "stage_loss_mask", "sample_weight", "prefix_relation",
    "tS_turn", "tE_turn", "tA_turn", "early_detectable", "explicit_operation_present",
    "operation_events", "core_evidence", "supporting_evidence", "insufficient_reason",
    "boundary_rationale", "turn_stage_labels", "confidence", "review_required", "review_status",
    "quality_flags", "future_content_used", "label_leakage_flag", "annotation_method",
    "annotator_id", "annotation_model", "annotation_prompt_version", "reviewer_id", "adjudicator_id",
    "content_group_id", "dialogue_blueprint_id", "duplicate_cluster_id", "source_type", "source_name",
}
META_PHRASES = (
    "个人业务结论", "尚未断言", "当前信息不足", "不需要强行下结论", "重新判断风险",
    "当前只算", "只停留在事实说明", "只作为待核信息", "系统把任务归到",
)


def load(path):
    rows, errors = [], []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                errors.append(f"{path.name}:{n}: {exc}")
    return rows, errors


def render(turns):
    return "\n".join(f"{t['turn_id']:03d} {t['speaker']}：{t['text']}" for t in turns)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def grams(text, n=4):
    value = normalize(text)
    return {value[i:i+n] for i in range(max(0, len(value)-n+1))}


def signature(gs, size=24):
    prime = 4294967311
    base = [zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF for x in gs]
    if not base:
        return tuple([0] * size)
    return tuple(min(((2654435761+2*i)*x + 2246822519*(i+1)) % prime for x in base) for i in range(size))


def near_duplicate_pairs(records):
    texts = {x["call_id"]: render(x["turns"]) for x in records}
    splits = {x["call_id"]: x["split"] for x in records}
    g = {k: grams(v) for k, v in texts.items()}
    buckets = collections.defaultdict(list)
    for cid, value in g.items():
        sig = signature(value)
        for band in range(6):
            buckets[(band, sig[band*4:(band+1)*4])].append(cid)
    candidates = set()
    for ids in buckets.values():
        if 1 < len(ids) <= 250:
            for i, left in enumerate(ids):
                for right in ids[i+1:]:
                    candidates.add(tuple(sorted((left, right))))
    pairs = []
    for left, right in sorted(candidates):
        score = len(g[left] & g[right]) / len(g[left] | g[right])
        if score >= 0.85:
            pairs.append({"left": left, "left_split": splits[left], "right": right, "right_split": splits[right], "char4_jaccard": round(score, 6), "cross_split": splits[left] != splits[right]})
    return pairs, len(candidates)


def main():
    errors, warnings = [], []
    data, parse_errors = {}, []
    for split in ("train", "dev", "stress", "blind"):
        data[split], errs = load(FILES[split]); parse_errors.extend(errs)
    labels, errs = load(FILES["blind_labels"]); parse_errors.extend(errs)
    errors.extend(parse_errors)
    label_map = {x.get("call_id"): x for x in labels}
    if len(labels) != 150 or len(label_map) != 150:
        errors.append("restricted blind labels must contain 150 unique call_ids")

    blind_field_union = set().union(*(set(x) for x in data["blind"])) if data["blind"] else set()
    blind_forbidden = sorted(blind_field_union - BLIND_ALLOWED)
    if blind_forbidden:
        errors.append(f"blind input contains non-allowlisted fields: {blind_forbidden}")
    leaked = sorted(blind_field_union & META_LEAK_FIELDS)
    if leaked:
        errors.append(f"blind input contains answer-inferable metadata: {leaked}")

    merged = []
    for split in ("train", "dev", "stress"):
        merged.extend(data[split])
    for x in data["blind"]:
        truth = label_map.get(x.get("call_id"))
        if truth is None:
            errors.append(f"missing restricted label for {x.get('call_id')}")
        else:
            merged.append({**x, **truth})
    if {x.get("call_id") for x in data["blind"]} != set(label_map):
        errors.append("blind input/label call_id sets differ")
    for split, expected in EXPECTED.items():
        if len(data[split]) != expected:
            errors.append(f"{split} count={len(data[split])}, expected={expected}")
    if len(merged) != 2000:
        errors.append(f"merged call count={len(merged)}")

    call_ids, sample_ids, hashes, nhashes = set(), set(), set(), set()
    turn_counts = collections.Counter()
    label_counts = collections.Counter()
    decision_counts = collections.Counter()
    evidence_errors = 0
    operation_errors = 0
    meta_phrase_calls = []
    full_records = []
    group_splits = collections.defaultdict(set)

    for row in merged:
        cid = row.get("call_id")
        if cid in call_ids: errors.append(f"duplicate call_id: {cid}")
        call_ids.add(cid)
        sid = row.get("sample_id")
        if sid in sample_ids: errors.append(f"duplicate sample_id: {sid}")
        sample_ids.add(sid)
        turns = row.get("turns") or []
        ids = [t.get("turn_id") for t in turns]
        speakers = [t.get("speaker") for t in turns]
        if ids != list(range(1, len(turns)+1)): errors.append(f"{cid}: discontinuous turn ids")
        if any(speakers[i] == speakers[i-1] for i in range(1, len(speakers))): errors.append(f"{cid}: speakers do not alternate")
        if row.get("n_full_turns") != len(turns) or row.get("tFinal_turn") != len(turns): errors.append(f"{cid}: turn count mismatch")
        full = render(turns)
        full_records.append(row)
        if row.get("prefix_text") != full or row.get("model_input_text") != full: errors.append(f"{cid}: rendered text mismatch")
        h, nh = digest(full), digest(normalize(full))
        if row.get("text_sha256") != h or row.get("prefix_text_hash") != h or row.get("normalized_text_hash") != nh: errors.append(f"{cid}: text hash mismatch")
        if h in hashes or nh in nhashes: errors.append(f"{cid}: duplicate full text hash")
        hashes.add(h); nhashes.add(nh)
        by_id = {t["turn_id"]: t for t in turns}
        for ev in (row.get("core_evidence") or []) + (row.get("supporting_evidence") or []):
            tid, quote = ev.get("turn_id"), ev.get("quote")
            if tid not in by_id or quote not in by_id[tid].get("text", "") or by_id[tid].get("speaker") != "来电人":
                evidence_errors += 1
        for ev in row.get("operation_events") or []:
            tid, quote = ev.get("turn_id"), ev.get("quote")
            if tid not in by_id or quote not in by_id[tid].get("text", "") or by_id[tid].get("speaker") != "来电人":
                operation_errors += 1
        if len(row.get("turn_stage_labels") or []) != len(turns): errors.append(f"{cid}: stage trajectory length mismatch")
        if any(p in full for p in META_PHRASES): meta_phrase_calls.append(cid)
        turn_counts.update(t["text"] for t in turns)
        label_counts[row.get("final_label")] += 1
        decision_counts[row.get("decision")] += 1
        if row.get("content_group_id") is not None:
            group_splits[row["content_group_id"]].add(row.get("split"))

    if evidence_errors: errors.append(f"invalid evidence quotes={evidence_errors}")
    if operation_errors: errors.append(f"invalid operation events={operation_errors}")
    if dict(label_counts) != EXPECTED_LABELS: errors.append(f"label counts={dict(label_counts)}")
    if meta_phrase_calls: errors.append(f"annotation-like language remains in {len(meta_phrase_calls)} calls")
    reused = [(text, n) for text, n in turn_counts.items() if n > 20]
    if reused: errors.append(f"exact turn reuse >20 remains: {len(reused)} phrases")
    max_reuse = max(turn_counts.values()) if turn_counts else 0
    pairs, candidate_count = near_duplicate_pairs(full_records)
    if pairs: errors.append(f"char4 Jaccard >=0.85 near duplicates remain: {len(pairs)}")
    if any(len(v) > 1 for v in group_splits.values()): errors.append("content_group_id crosses split")

    report = {
        "schema_version": "v365.validation.clean1",
        "dataset_version": "v365_2000.clean1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": [
            "All samples remain synthetic_compositional; SOURCE_COMPOSITION_PASS=false.",
            "Independent review disagreements still require adjudication; RELEASE_READY=false.",
        ],
        "calls": len(merged),
        "unique_call_ids": len(call_ids),
        "unique_sample_ids": len(sample_ids),
        "unique_text_sha256": len(hashes),
        "unique_normalized_text_sha256": len(nhashes),
        "final_label_counts": dict(sorted(label_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "blind_input_fields": sorted(blind_field_union),
        "blind_input_non_allowlisted_fields": blind_forbidden,
        "blind_input_answer_inferable_metadata_absent": not leaked,
        "blind_alignment_ok": {x.get("call_id") for x in data["blind"]} == set(label_map),
        "invalid_evidence_quote_count": evidence_errors,
        "invalid_operation_event_count": operation_errors,
        "annotation_like_language_call_count": len(meta_phrase_calls),
        "exact_turn_phrases_reused_over_20": len(reused),
        "max_exact_turn_reuse": max_reuse,
        "near_duplicate_pairs_jaccard_ge_085": pairs,
        "near_duplicate_candidate_pairs_checked": candidate_count,
        "cross_split_near_duplicate_count": sum(x["cross_split"] for x in pairs),
        "structure_pass": not parse_errors and evidence_errors == 0 and operation_errors == 0,
        "blind_isolation_pass": not leaked and not blind_forbidden,
        "duplication_pass": not pairs and not reused,
        "semantic_quality_pass": not meta_phrase_calls and max_reuse <= 20,
        "source_composition_pass": False,
        "release_ready": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_paths = [
        ROOT / "README.md", ROOT / "V365_BOUNDARY_POLICY.md", ROOT / "V365_BUILD_REPORT.md",
        ROOT / "scripts/build_v365_2000.py", ROOT / "scripts/validate_v365_2000.py",
        ROOT / "scripts/rebuild_v365_clean1.py", *FILES.values(), DATA / "v365_statistics.json", REPORT,
    ]
    manifest = {
        "schema_version": "v365.manifest.clean1",
        "dataset_version": "v365_2000.clean1",
        "files": [{"file": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in manifest_paths if p.exists()],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
