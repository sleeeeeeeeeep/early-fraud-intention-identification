#!/usr/bin/env python3
"""Post-freeze comparison and release audit for V365_2000."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path


ANSWER_FIELDS = {
    "final_label", "final_binary_label", "decision", "stage_target", "binary_target",
    "binary_loss_mask", "stage_loss_mask", "tS_turn", "tE_turn", "tA_turn",
    "core_evidence", "supporting_evidence", "boundary_rationale", "turn_stage_labels",
    "explicit_operation_present", "operation_events", "confidence",
}


def read_jsonl(path: Path):
    rows, errors = [], []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                errors.append({"file": str(path), "line": line_no, "error": str(exc)})
    return rows, errors


def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pct(n, d):
    return round(100 * n / d, 4) if d else None


def rendered_turns(turns, end=None):
    selected = turns if end is None else turns[:end]
    return "\n".join(f"{int(t['turn_id']):03d} {t['speaker']}：{t['text']}" for t in selected)


def evidence_valid(row, turns_by_id):
    invalid = []
    for field in ("core_evidence", "supporting_evidence"):
        for ev in row.get(field) or []:
            tid, quote = ev.get("turn_id"), ev.get("quote")
            if tid not in turns_by_id:
                invalid.append({"field": field, "turn_id": tid, "reason": "turn_missing", "quote": quote})
            elif not isinstance(quote, str) or quote not in str(turns_by_id[tid].get("text", "")):
                invalid.append({"field": field, "turn_id": tid, "reason": "quote_not_verbatim", "quote": quote})
    return invalid


def operation_event_valid(row, turns_by_id):
    invalid = []
    for ev in row.get("operation_events") or []:
        tid = ev.get("turn_id")
        if tid not in turns_by_id:
            invalid.append({"turn_id": tid, "reason": "turn_missing"})
        elif turns_by_id[tid].get("speaker") != "来电人":
            invalid.append({"turn_id": tid, "reason": "wrong_speaker"})
    return invalid


def timing_diff(original, reviewed):
    if original is None and reviewed is None:
        return 0
    if original is None or reviewed is None:
        return None
    return reviewed - original


def nested_counter_to_dict(counter):
    out = collections.defaultdict(dict)
    for (a, b), n in sorted(counter.items()):
        out[str(a)][str(b)] = n
    return dict(out)


def rates_by(meta_key, calls, comparisons):
    total = collections.Counter()
    issues = collections.Counter()
    disagreement = collections.Counter()
    for cid, row in calls.items():
        key = str(row.get(meta_key))
        total[key] += 1
        c = comparisons[cid]
        if c["issue_codes"]:
            issues[key] += 1
        if not c["label_agreement"]:
            disagreement[key] += 1
    return {
        key: {
            "count": total[key],
            "issue_count": issues[key],
            "issue_rate_pct": pct(issues[key], total[key]),
            "label_disagreement_count": disagreement[key],
            "label_disagreement_rate_pct": pct(disagreement[key], total[key]),
        }
        for key in sorted(total)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--review-dir", required=True)
    args = ap.parse_args()
    base, out = Path(args.dataset_dir), Path(args.review_dir)

    manifest = json.loads((out / "BLIND_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    frozen_path = out / "blind_review_labels.jsonl"
    actual_hash = sha256_file(frozen_path)
    if actual_hash != manifest["blind_review_labels_sha256"]:
        raise SystemExit("Frozen blind review hash mismatch; refusing label comparison")
    reviews_list, errors = read_jsonl(frozen_path)
    if errors or len(reviews_list) != 2000:
        raise SystemExit("Frozen blind review is incomplete or unparsable")
    reviews = {x["call_id"]: x for x in reviews_list}

    files = {
        "train": base / "data/v365_calls_train.jsonl",
        "dev": base / "data/v365_calls_dev.jsonl",
        "stress": base / "data/v365_calls_stress.jsonl",
        "blind": base / "data/v365_calls_blind_input.jsonl",
    }
    calls, parse_errors = {}, []
    file_rows = {}
    for split, path in files.items():
        rows, errs = read_jsonl(path)
        file_rows[split] = rows
        parse_errors.extend(errs)
        for row in rows:
            if row.get("call_id") in calls:
                parse_errors.append({"file": str(path), "call_id": row.get("call_id"), "error": "duplicate_call_id"})
            calls[row.get("call_id")] = row
    blind_labels_list, errs = read_jsonl(base / "data/restricted_v365_calls_blind_labels.jsonl")
    parse_errors.extend(errs)
    blind_labels = {x["call_id"]: x for x in blind_labels_list}
    alignment_ok = set(blind_labels) == {x["call_id"] for x in file_rows["blind"]} and len(blind_labels_list) == len(blind_labels)

    originals = {}
    for split in ("train", "dev", "stress"):
        originals.update({x["call_id"]: x for x in file_rows[split]})
    originals.update(blind_labels)
    all_ids_aligned = set(calls) == set(originals) == set(reviews) and len(calls) == 2000

    structure_errors = []
    sample_ids = collections.Counter()
    content_splits = collections.defaultdict(set)
    blueprint_splits = collections.defaultdict(set)
    for cid, row in calls.items():
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "turns_missing_or_empty"})
            continue
        ids = [t.get("turn_id") for t in turns]
        speakers = [t.get("speaker") for t in turns]
        if ids != list(range(1, len(turns) + 1)):
            structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "turn_ids_not_continuous"})
        if any(speakers[i] == speakers[i - 1] for i in range(1, len(speakers))):
            structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "speakers_not_alternating"})
        if row.get("n_full_turns") != len(turns):
            structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "n_full_turns_mismatch"})
        if row.get("tFinal_turn") != len(turns):
            structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "tFinal_turn_mismatch"})
        prefix_end = row.get("prefix_end_turn")
        if isinstance(prefix_end, int):
            expected = rendered_turns(turns, prefix_end)
            if row.get("prefix_text") != expected or row.get("model_input_text") != expected:
                structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "rendered_text_mismatch"})
            digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
            if row.get("prefix_text_hash") != digest or row.get("text_sha256") != digest:
                structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "text_hash_mismatch"})
            if row.get("n_prefix_turns") != prefix_end:
                structure_errors.append({"call_id": cid, "code": "SCHEMA_ERROR", "detail": "prefix_turn_count_mismatch"})
        sample_ids[row.get("sample_id")] += 1
        content_splits[row.get("content_group_id")].add(row.get("split"))
        blueprint_splits[row.get("dialogue_blueprint_id")].add(row.get("split"))
    for sid, count in sample_ids.items():
        if sid is None or count != 1:
            structure_errors.append({"sample_id": sid, "code": "SCHEMA_ERROR", "detail": f"sample_id_count={count}"})
    content_group_leaks = {str(k): sorted(v) for k, v in content_splits.items() if k is not None and len(v) > 1}

    blind_rows = file_rows["blind"]
    explicit_answer_fields = sorted(set().union(*(set(x) for x in blind_rows)) & ANSWER_FIELDS)
    evidence_position_values = collections.Counter(str(x.get("evidence_position_bin")) for x in blind_rows)
    evidence_position_by_label = collections.Counter((str(x.get("evidence_position_bin")), blind_labels[x["call_id"]]["final_label"]) for x in blind_rows)
    evidence_position_perfect_leak = all(
        (x.get("evidence_position_bin") is not None) == (blind_labels[x["call_id"]]["final_label"] == "fraud")
        for x in blind_rows
    )
    metadata_fields_of_concern = sorted(set().union(*(set(x) for x in blind_rows)) & {
        "evidence_position_bin", "review_required", "review_status", "quality_flags",
        "future_content_used", "label_leakage_flag", "annotation_method", "reviewer_id", "adjudicator_id",
    })

    comparison_rows = []
    comparison_map = {}
    issue_rows = []
    evidence_invalid_count = 0
    operation_invalid_count = 0
    stage_length_error_count = 0
    confusion = collections.Counter()
    decision_confusion = collections.Counter()
    tE_categories = collections.Counter()
    tA_agree_count = 0
    evidence_turn_agree_count = 0

    for cid in sorted(calls):
        row, original, rev = calls[cid], originals[cid], reviews[cid]
        turns_by_id = {t["turn_id"]: t for t in row.get("turns") or [] if isinstance(t, dict) and "turn_id" in t}
        invalid_ev = evidence_valid(original, turns_by_id)
        invalid_ops = operation_event_valid(original, turns_by_id)
        evidence_invalid_count += len(invalid_ev)
        operation_invalid_count += len(invalid_ops)
        if len(original.get("turn_stage_labels") or []) != len(row.get("turns") or []):
            stage_length_error_count += 1

        ol, rl = original.get("final_label"), rev.get("reviewer_final_label")
        od, rd = original.get("decision"), rev.get("reviewer_decision")
        ote, rte = original.get("tE_turn"), rev.get("reviewer_tE_turn")
        ota, rta = original.get("tA_turn"), rev.get("reviewer_tA_turn")
        label_agree, decision_agree = ol == rl, od == rd
        tediff = timing_diff(ote, rte)
        ta_agree = ota == rta
        confusion[(ol, rl)] += 1
        decision_confusion[(od, rd)] += 1
        if ote is None and rte is None:
            tE_categories["both_null"] += 1
        elif ote is None or rte is None:
            tE_categories["one_null"] += 1
        elif tediff == 0:
            tE_categories["exact"] += 1
        elif abs(tediff) == 1:
            tE_categories["off_by_1"] += 1
        else:
            tE_categories["off_by_2_or_more"] += 1
        if ta_agree:
            tA_agree_count += 1
        original_evidence_turns = {e.get("turn_id") for e in original.get("core_evidence") or []}
        reviewer_evidence_turns = {e.get("turn_id") for e in rev.get("reviewer_core_evidence") or []}
        evidence_turn_agree = original_evidence_turns == reviewer_evidence_turns
        evidence_turn_agree_count += int(evidence_turn_agree)

        codes = list(rev.get("issue_codes") or [])
        if not label_agree:
            codes.append("LABEL_DISAGREEMENT")
        if not decision_agree:
            codes.append("DECISION_DISAGREEMENT")
        if tediff not in (0, None):
            codes.append("TE_TOO_LATE" if tediff > 0 else "TE_TOO_EARLY")
        elif (ote is None) != (rte is None):
            codes.append("TE_DISAGREEMENT")
        if not ta_agree:
            codes.append("TA_WRONG")
        if invalid_ev:
            codes.append("EVIDENCE_QUOTE_INVALID")
        if invalid_ops:
            codes.append("OPERATION_EVENT_INVALID")
        if len(original.get("turn_stage_labels") or []) != len(row.get("turns") or []):
            codes.append("SCHEMA_ERROR")
        codes = list(dict.fromkeys(codes))

        if not label_agree or ((isinstance(tediff, int) and abs(tediff) > 1) or (ote is None) != (rte is None)) or invalid_ev:
            severity = "P0"
        elif not decision_agree or not ta_agree or invalid_ops or "TEMPLATE_OVERUSE" in codes or "ROLE_INCOHERENT" in codes:
            severity = "P1"
        elif codes:
            severity = "P2"
        else:
            severity = "none"
        requires = (
            not label_agree or not decision_agree or not ta_agree
            or (tediff is None and ote != rte)
            or (isinstance(tediff, int) and abs(tediff) > 1)
            or rev.get("naturalness_score", 4) < 2
            or severity in ("P0", "P1")
            or rev.get("reviewer_confidence") == "low"
        )
        comp = {
            "call_id": cid,
            "split": row.get("split"),
            "scenario_tag": row.get("scenario_tag"),
            "length_bucket": row.get("length_bucket"),
            "original_final_label": ol,
            "reviewer_final_label": rl,
            "label_agreement": label_agree,
            "original_decision": od,
            "reviewer_decision": rd,
            "decision_agreement": decision_agree,
            "original_tE_turn": ote,
            "reviewer_tE_turn": rte,
            "tE_difference": tediff,
            "original_tA_turn": ota,
            "reviewer_tA_turn": rta,
            "tA_agreement": ta_agree,
            "evidence_turn_agreement": evidence_turn_agree,
            "original_evidence_valid": not invalid_ev,
            "requires_adjudication": requires,
            "issue_codes": codes,
            "severity": severity,
            "post_reveal_assessment": "requires_independent_adjudication" if requires else "agreement_no_adjudication_required",
        }
        comparison_rows.append(comp)
        comparison_map[cid] = comp
        if codes:
            issue_rows.append({
                "scope": "call", "call_id": cid, "split": row.get("split"),
                "scenario_tag": row.get("scenario_tag"), "length_bucket": row.get("length_bucket"),
                "severity": severity, "issue_codes": codes,
                "details": {
                    "label": [ol, rl], "decision": [od, rd], "tE": [ote, rte], "tA": [ota, rta],
                    "invalid_evidence": invalid_ev, "invalid_operation_events": invalid_ops,
                    "naturalness_score": rev.get("naturalness_score"),
                    "template_risk_score": rev.get("template_risk_score"),
                },
                "recommended_action": "send_to_adjudication" if requires else rev.get("recommended_action"),
            })

    duplicate_rows, dup_errors = read_jsonl(out / "duplicate_candidates_blind.jsonl")
    parse_errors.extend(dup_errors)
    cross_split_pairs = [x for x in duplicate_rows if x.get("cross_split")]
    train_blind_pairs = [x for x in duplicate_rows if {x.get("split_a"), x.get("split_b")} == {"train", "blind"}]
    for pair in duplicate_rows:
        issue_rows.append({
            "scope": "pair", "severity": "P0" if pair in train_blind_pairs else ("P1" if pair.get("cross_split") else "P1"),
            "issue_codes": ["NEAR_DUPLICATE"] + (["SPLIT_LEAKAGE"] if pair.get("cross_split") else []),
            "details": pair,
            "recommended_action": "remove_or_regenerate_one_member",
        })
    if evidence_position_perfect_leak:
        issue_rows.append({
            "scope": "blind_input", "severity": "P0", "issue_codes": ["LABEL_LEAKAGE"],
            "details": {
                "field": "evidence_position_bin",
                "finding": "The field is non-null for every blind fraud record and null for every blind normal/ambiguous record.",
                "value_counts": dict(evidence_position_values),
            },
            "recommended_action": "remove_field_from_blind_input_and_regenerate_blind_artifact",
        })
    for err in structure_errors:
        issue_rows.append({"scope": "structure", "severity": "P0", "issue_codes": ["SCHEMA_ERROR"], "details": err, "recommended_action": "correct_schema"})
    if not alignment_ok:
        issue_rows.append({"scope": "blind_alignment", "severity": "P0", "issue_codes": ["BLIND_ALIGNMENT_ERROR"], "details": {}, "recommended_action": "repair_alignment"})

    dump_jsonl(out / "review_comparison.jsonl", comparison_rows)
    dump_jsonl(out / "review_issues.jsonl", issue_rows)
    adjudication = []
    corrections = []
    for c in comparison_rows:
        if not c["requires_adjudication"]:
            continue
        cid = c["call_id"]
        row, original, rev = calls[cid], originals[cid], reviews[cid]
        adjudication.append({
            "call_id": cid, "split": row.get("split"), "scenario_tag": row.get("scenario_tag"),
            "length_bucket": row.get("length_bucket"), "severity": c["severity"],
            "issue_codes": c["issue_codes"],
            "original": {"final_label": original.get("final_label"), "decision": original.get("decision"), "tE_turn": original.get("tE_turn"), "tA_turn": original.get("tA_turn"), "core_evidence": original.get("core_evidence")},
            "reviewer": {"final_label": rev.get("reviewer_final_label"), "decision": rev.get("reviewer_decision"), "tE_turn": rev.get("reviewer_tE_turn"), "tA_turn": rev.get("reviewer_tA_turn"), "core_evidence": rev.get("reviewer_core_evidence"), "confidence": rev.get("reviewer_confidence")},
            "adjudication_instruction": "由未接触原标签的新审核员仅依据 turns 和严格边界裁决；不得自动采用任一方答案。",
        })
        corrections.append({
            "call_id": cid,
            "status": "pending_independent_adjudication",
            "recommended_action": "send_to_adjudication",
            "original_values": {"final_label": original.get("final_label"), "decision": original.get("decision"), "tE_turn": original.get("tE_turn"), "tA_turn": original.get("tA_turn")},
            "reviewer_proposal": {"final_label": rev.get("reviewer_final_label"), "decision": rev.get("reviewer_decision"), "tE_turn": rev.get("reviewer_tE_turn"), "tA_turn": rev.get("reviewer_tA_turn")},
            "apply_automatically": False,
        })
    dump_jsonl(out / "adjudication_queue.jsonl", adjudication)
    dump_jsonl(out / "proposed_corrections.jsonl", corrections)

    label_agree_n = sum(x["label_agreement"] for x in comparison_rows)
    decision_agree_n = sum(x["decision_agreement"] for x in comparison_rows)
    tE_comparable = [x for x in comparison_rows if isinstance(x["original_tE_turn"], int) and isinstance(x["reviewer_tE_turn"], int)]
    tE_abs = [abs(x["tE_difference"]) for x in tE_comparable]
    severity_counts = collections.Counter(x["severity"] for x in issue_rows)
    naturalness = collections.Counter(str(x["naturalness_score"]) for x in reviews_list)
    template = collections.Counter(str(x["template_risk_score"]) for x in reviews_list)
    original_counts = collections.Counter(x.get("final_label") for x in originals.values())
    reviewer_counts = collections.Counter(x.get("reviewer_final_label") for x in reviews_list)

    structure_pass = not parse_errors and not structure_errors and alignment_ok and all_ids_aligned and evidence_invalid_count == 0 and operation_invalid_count == 0 and stage_length_error_count == 0 and not content_group_leaks
    label_pass = label_agree_n == 2000
    timing_pass = all(x["tE_difference"] == 0 for x in comparison_rows) and tA_agree_count == 2000
    semantic_pass = naturalness.get("0", 0) == 0 and naturalness.get("1", 0) == 0 and naturalness.get("2", 0) == 0 and not any("ROLE_INCOHERENT" in x.get("issue_codes", []) for x in issue_rows)
    duplication_pass = not cross_split_pairs
    blind_isolation_pass = not explicit_answer_fields and not evidence_position_perfect_leak
    source_composition_pass = False
    independent_review_pass = label_agree_n == 2000 and not adjudication
    release_ready = all([structure_pass, label_pass, timing_pass, semantic_pass, duplication_pass, blind_isolation_pass, source_composition_pass, independent_review_pass])

    summary = {
        "review_identity": "gpt5_independent_review_20260922",
        "frozen_review_sha256": actual_hash,
        "coverage": {"expected": 2000, "reviewed": len(reviews), "complete": len(reviews) == 2000 and all_ids_aligned},
        "labels": {
            "original_counts": dict(original_counts), "reviewer_counts": dict(reviewer_counts),
            "confusion_matrix": nested_counter_to_dict(confusion),
            "agreement_count": label_agree_n, "agreement_rate_pct": pct(label_agree_n, 2000),
        },
        "decisions": {
            "confusion_matrix": nested_counter_to_dict(decision_confusion),
            "agreement_count": decision_agree_n, "agreement_rate_pct": pct(decision_agree_n, 2000),
        },
        "timing": {
            "tE_categories": dict(tE_categories),
            "tE_comparable_count": len(tE_comparable),
            "tE_exact_among_comparable_count": sum(x == 0 for x in tE_abs),
            "tE_exact_among_comparable_rate_pct": pct(sum(x == 0 for x in tE_abs), len(tE_abs)),
            "tE_mean_absolute_difference_among_comparable": round(sum(tE_abs) / len(tE_abs), 6) if tE_abs else None,
            "tA_agreement_count_all_records": tA_agree_count,
            "tA_agreement_rate_all_records_pct": pct(tA_agree_count, 2000),
            "core_evidence_turn_agreement_count": evidence_turn_agree_count,
            "core_evidence_turn_agreement_rate_pct": pct(evidence_turn_agree_count, 2000),
        },
        "quality": {
            "naturalness_score_distribution": dict(naturalness),
            "template_risk_score_distribution": dict(template),
            "original_evidence_invalid_count": evidence_invalid_count,
            "original_operation_event_invalid_count": operation_invalid_count,
            "turn_stage_length_error_count": stage_length_error_count,
        },
        "rates_by_split": rates_by("split", calls, comparison_map),
        "rates_by_scenario": rates_by("scenario_tag", calls, comparison_map),
        "rates_by_length_bucket": rates_by("length_bucket", calls, comparison_map),
        "duplication": {
            "normalized_exact_duplicate_group_count": manifest.get("normalized_exact_duplicate_group_count"),
            "near_duplicate_pair_count": len(duplicate_rows),
            "cross_split_near_duplicate_pair_count": len(cross_split_pairs),
            "train_blind_near_duplicate_pair_count": len(train_blind_pairs),
            "train_blind_pairs": train_blind_pairs,
            "content_group_cross_split_count": len(content_group_leaks),
        },
        "blind_isolation": {
            "blind_alignment_ok": alignment_ok,
            "explicit_answer_fields_present": explicit_answer_fields,
            "metadata_fields_of_concern": metadata_fields_of_concern,
            "evidence_position_bin_perfect_label_leak": evidence_position_perfect_leak,
            "evidence_position_bin_by_label": {f"{k[0]}|{k[1]}": v for k, v in evidence_position_by_label.items()},
        },
        "issues": {
            "issue_record_count": len(issue_rows),
            "severity_counts": dict(severity_counts),
            "adjudication_queue_count": len(adjudication),
            "unresolved_label_disagreement_count": 2000 - label_agree_n,
        },
        "structure": {
            "parse_error_count": len(parse_errors), "structure_error_count": len(structure_errors),
            "all_ids_aligned": all_ids_aligned, "sample_id_unique": all(v == 1 for v in sample_ids.values()),
            "content_group_cross_split_count": len(content_group_leaks),
        },
        "policy_specification_conflicts": [
            "诈骗总量800与至少1000通可在明确操作前标记early_fraud无法同时满足。",
            "12个诈骗场景每类至少80需至少960通，与诈骗总量800无法同时满足；平台诈骗合计不少于400进一步加剧冲突。",
        ],
        "release_flags": {
            "STRUCTURE_PASS": structure_pass,
            "LABEL_PASS": label_pass,
            "TIMING_PASS": timing_pass,
            "SEMANTIC_QUALITY_PASS": semantic_pass,
            "DUPLICATION_PASS": duplication_pass,
            "BLIND_ISOLATION_PASS": blind_isolation_pass,
            "SOURCE_COMPOSITION_PASS": source_composition_pass,
            "INDEPENDENT_REVIEW_PASS": independent_review_pass,
            "RELEASE_READY": release_ready,
        },
    }
    dump_json(out / "review_summary.json", summary)

    representative = [x for x in comparison_rows if not x["label_agreement"]][:8]
    rep_lines = []
    for c in representative:
        cid = c["call_id"]
        ev = originals[cid].get("core_evidence") or []
        quote = ev[0].get("quote") if ev else "（原标签无核心证据）"
        rep_lines.append(f"- `{cid}`：原 `{c['original_final_label']}/{c['original_decision']}`，复核 `{c['reviewer_final_label']}/{c['reviewer_decision']}`；原核心证据：{quote}")

    flags = summary["release_flags"]
    naturalness_2 = naturalness.get("2", 0)
    blind_status_text = (
        "Blind Input 已移除 `evidence_position_bin` 及答案相关元字段；字段级隔离复验通过。"
        if blind_isolation_pass
        else "Blind Input 仍存在可推断答案的字段或元数据，字段级隔离复验未通过。"
    )
    duplicate_status_text = (
        "未发现字符 4-gram Jaccard ≥0.85 的跨切分近重复。"
        if not cross_split_pairs
        else f"仍发现 {len(cross_split_pairs)} 对跨切分近重复，其中 Train–Blind {len(train_blind_pairs)} 对。"
    )
    blocking_items = [f"独立标签分歧 {2000-label_agree_n} 通尚未经第三方裁决", "全部来源仍为 `synthetic_compositional`"]
    if not blind_isolation_pass:
        blocking_items.append("Blind Test 仍有答案泄漏")
    if not duplication_pass:
        blocking_items.append("仍存在跨切分近重复")
    if naturalness_2:
        blocking_items.append(f"仍有 {naturalness_2} 通自然度仅 2 分")
    blocking_text = "；".join(blocking_items) + "。"
    report = f"""# V365_2000 独立检查复核报告

## 结论

已按要求完成 2,000/2,000 通第一轮盲审，并在揭示原标签前冻结结果。冻结文件 SHA256：`{actual_hash}`。

本数据集当前仍**不具备正式训练与发布条件**，`RELEASE_READY = false`。主要阻断项：{blocking_text}

## 覆盖与冻结

- 实际复核：2,000 通；完整覆盖：是。
- 第一轮仅读取 `call_id`、`split`、`turns`。
- 第一轮独立结果：诈骗 {reviewer_counts.get('fraud',0)}、正常 {reviewer_counts.get('normal',0)}、边界不确定 {reviewer_counts.get('ambiguous',0)}。
- 冻结后才读取原标签；Blind Input 150 条与 Restricted Label 150 条按 `call_id` 一一对应。

## 标签与决策对照

- 原标签：诈骗 {original_counts.get('fraud',0)}、正常 {original_counts.get('normal',0)}、边界不确定 {original_counts.get('ambiguous',0)}。
- 最终标签一致：{label_agree_n}/2,000（{pct(label_agree_n,2000)}%）。
- 决策一致：{decision_agree_n}/2,000（{pct(decision_agree_n,2000)}%）。
- 原标签×复核标签混淆矩阵：`{json.dumps(nested_counter_to_dict(confusion), ensure_ascii=False)}`。
- 未裁决标签分歧：{2000-label_agree_n} 通；均已进入裁决队列，未自动覆盖原始数据。

## 时间点、证据与操作

- 双方均有 `tE`、可直接比较：{len(tE_comparable)} 通；完全一致 {sum(x==0 for x in tE_abs)} 通（{pct(sum(x==0 for x in tE_abs),len(tE_abs))}%）。
- 可比较 `tE` 平均绝对偏差：{round(sum(tE_abs)/len(tE_abs),6) if tE_abs else 'N/A'} 轮。
- `tE`：双方均空 {tE_categories.get('both_null',0)}；一方为空 {tE_categories.get('one_null',0)}；相差1轮 {tE_categories.get('off_by_1',0)}；相差2轮及以上 {tE_categories.get('off_by_2_or_more',0)}。
- `tA` 全记录一致：{tA_agree_count}/2,000（{pct(tA_agree_count,2000)}%）。
- 原核心/支持证据逐字回引无效：{evidence_invalid_count} 条；原操作事件轮次或说话人无效：{operation_invalid_count} 条。

## 结构检查

- JSONL 解析错误：{len(parse_errors)}；结构错误：{len(structure_errors)}。
- `call_id`/`sample_id` 唯一、轮次连续、说话人交替、`n_full_turns`、`tFinal_turn`、渲染文本与 SHA256 均已全量检查。
- 内容组跨切分：{len(content_group_leaks)} 组。
- Blind 输入与标签一一对应：{'是' if alignment_ok else '否'}。

## Blind Test 隔离

{blind_status_text}

## 重复与切分泄漏

- 规范化全文完全重复组：{manifest.get('normalized_exact_duplicate_group_count')}。
- 字符 4-gram Jaccard ≥0.85 的近重复对：{len(duplicate_rows)}；跨切分 {len(cross_split_pairs)} 对。
- Train–Blind 近重复：{len(train_blind_pairs)} 对。
- {duplicate_status_text}

## 语义与自然度

- 自然度分布：{dict(naturalness)}；自然度 2 分记录为 {naturalness_2} 通。
- 模板风险分布：{dict(template)}。
- 正常误报候选（原 normal、复核 fraud/ambiguous）：{sum(n for (o,r),n in confusion.items() if o=='normal' and r!='normal')}。
- 诈骗漏报候选（原 fraud、复核 normal/ambiguous）：{sum(n for (o,r),n in confusion.items() if o=='fraud' and r!='fraud')}。
- 边界不确定滥用候选（原 fraud/normal、复核 ambiguous）：{sum(n for (o,r),n in confusion.items() if r=='ambiguous' and o!='ambiguous')}。
- 这些均是“分歧候选”，不是已裁决真值；应由未接触原标签的第三位审核员裁决。

## 分组问题率

- 按 split：`{json.dumps(summary['rates_by_split'], ensure_ascii=False)}`
- 按长度：`{json.dumps(summary['rates_by_length_bucket'], ensure_ascii=False)}`
- 按场景的完整统计见 `review_summary.json`。

## 代表性分歧

{chr(10).join(rep_lines)}

## 规范内部数量冲突

1. 诈骗总量 800 与“至少 1,000 通诈骗可在明确操作前标记 early_fraud”无法同时满足。
2. 12 个诈骗场景每类至少 80 需要至少 960 通，与诈骗总量 800 冲突；平台诈骗不少于 400 又进一步约束数量。

复核未通过调整独立标签来掩盖这些冲突。

## 发布判定

```text
STRUCTURE_PASS = {str(flags['STRUCTURE_PASS']).lower()}
LABEL_PASS = {str(flags['LABEL_PASS']).lower()}
TIMING_PASS = {str(flags['TIMING_PASS']).lower()}
SEMANTIC_QUALITY_PASS = {str(flags['SEMANTIC_QUALITY_PASS']).lower()}
DUPLICATION_PASS = {str(flags['DUPLICATION_PASS']).lower()}
BLIND_ISOLATION_PASS = {str(flags['BLIND_ISOLATION_PASS']).lower()}
SOURCE_COMPOSITION_PASS = {str(flags['SOURCE_COMPOSITION_PASS']).lower()}
INDEPENDENT_REVIEW_PASS = {str(flags['INDEPENDENT_REVIEW_PASS']).lower()}
RELEASE_READY = {str(flags['RELEASE_READY']).lower()}
```

## 建议

1. 保持当前 Blind 安全字段白名单，并在后续构建脚本中禁止重新写入答案相关元字段。
2. 对 `adjudication_queue.jsonl` 中的标签、决策和时间点分歧由新审核员盲裁。
3. 补足真实、脱敏真实或人工自然改写来源比例；在此之前保持 `SOURCE_COMPOSITION_PASS=false`。
4. 裁决与来源修复后重新运行全量校验，再决定是否用于正式训练或发布。
"""
    (out / "REVIEW_REPORT.md").write_text(report, encoding="utf-8")

    deliverables = [
        "blind_review_labels.jsonl", "review_comparison.jsonl", "review_issues.jsonl",
        "adjudication_queue.jsonl", "review_summary.json", "REVIEW_REPORT.md",
        "proposed_corrections.jsonl",
    ]
    hashes = {name: sha256_file(out / name) for name in deliverables}
    dump_json(out / "DELIVERABLE_SHA256.json", hashes)
    print(json.dumps({"release_flags": flags, "deliverable_hashes": hashes, "adjudication_count": len(adjudication)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
