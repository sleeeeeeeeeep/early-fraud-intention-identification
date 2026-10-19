"""Validate audited source annotation corrections without changing original files.

This module is intentionally standard-library-only and compatible with Python 3.8.
Copy it together with its audit ledger for independent offline verification. The
ledger embeds the original review evidence; applying it does not approve a rewrite.
"""
import copy
import hashlib
import json
import re
from pathlib import Path


AUDIT_FILENAME = "source_adjudication_override_audit_20261009.json"
AUDIT_LEDGER_SHA256 = "a1f939ce9771721c0d24503e63772b5de1fac8c4f3a08a43898e06e2123c4723"
EARLY_CONFLICT_SOURCE_SHA256 = {
    "V365_TRAIN_0457": "68f8c60a4c2775afba6a0f77b53e1eb37ef6fc4c753c8d0897f526dc422595fd",
    "V365_TRAIN_0458": "8f573517ea4d58cc21ce10574108682fc1a8b382c1a1f04a4afda69deea6f733",
    "V365_TRAIN_0460": "020908859812459459a8bf4c40af5e210f906c032835dd4d52d87a3b03635f15",
    "V365_TRAIN_0461": "98dee772fe271bf5c98609a60f0bcd50633257f401d3c68fc2c430494003dc15",
    "V365_TRAIN_0465": "6e4228b2b7dfc259130cf6246ec21f62a617217aa10bf4c45dd5f40ed48cda3f",
    "V365_TRAIN_0469": "e9c50d107b2699f115d0e44f90a3ed400acb495877114ef67f713c86201d649b",
    "V365_TRAIN_0475": "b658a25114985763150c147b2de3229cb30b3f3962c8479010457930e1c91f84",
    "V365_STRESS_1747": "4ba432433ce4cb67fc39b365e36bba28271c99691d59110d44ff494b112f3231",
    "V365_BLIND_1897": "5b172759198dc2c1ebfd372880412609f3199b29b07c3191443c91a9e506e08d",
    "V365_BLIND_1898": "08ffef201d6fc487e5ab9f301409c883b758c79b98c238859ba20dca7d0aed8a",
}
CHANGEABLE_FIELDS = {
    "decision", "tS_turn", "tE_turn", "tA_turn", "core_evidence",
    "supporting_evidence", "operation_events", "adjudication_rationale",
}
MODEL = "gpt-6-luna"
OPERATION_TYPES = {
    "account_or_password", "bank_operation", "cash_gold_giftcard_or_courier",
    "crypto_asset", "download", "identity_document", "otp", "remote_control",
    "screen_share", "sim_or_esim_operation", "transfer", "unofficial_link",
}


class OverrideValidationError(ValueError):
    """An override must not be applied when its provenance is incomplete."""


def _require(condition, reason):
    if not condition:
        raise OverrideValidationError(reason)


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return _sha_bytes(encoded)


def _audit_path(audit_root=None):
    root = Path(audit_root) if audit_root is not None else Path(__file__).resolve().parent
    return root if root.is_file() else root / AUDIT_FILENAME


def audit_ledger_sha256(audit_root=None):
    path = _audit_path(audit_root)
    _require(path.is_file(), "audited source override ledger is missing: " + str(path))
    digest = _sha_bytes(path.read_bytes())
    _require(digest == AUDIT_LEDGER_SHA256, "override ledger byte hash differs from its module binding")
    return digest


def _quote_supported(evidence, source, context):
    _require(isinstance(evidence, dict), context + ": evidence object is missing")
    turn = evidence.get("turn_id")
    quote = evidence.get("quote")
    turns = source.get("turns", [])
    _require(type(turn) is int and 1 <= turn <= len(turns), context + ": invalid turn")
    _require(isinstance(quote, str) and bool(quote.strip()) and
             quote in turns[turn - 1].get("text", ""), context + ": quote is not verbatim")


def _operations_without_explanation(events):
    return [{key: item.get(key) for key in ("turn_id", "quote", "operation_type")}
            for item in events]


def _validate_review(review, source, context):
    _require(review.get("observed_decision") == "explicit_fraud", context + ": source decision is not explicit fraud")
    bounds = [review.get("earliest_" + key + "_turn") for key in ("tS", "tE", "tA")]
    _require(all(type(value) is int and 1 <= value <= len(source["turns"]) for value in bounds),
             context + ": source boundary is outside its transcript")
    ts, te, ta = bounds
    _require(ts <= te <= ta, context + ": boundary order is inconsistent")
    for key, boundary in (("tS_evidence", ts), ("tE_evidence", te), ("first_operation", ta)):
        _quote_supported(review.get(key), source, context + ":" + key)
        _require(review[key]["turn_id"] == boundary, context + ": boundary evidence turn differs")
    chain = review.get("evidence_chain_scan", {})
    component_turns = []
    for key in ("claimed_identity", "individualized_state", "connected_consequence"):
        _quote_supported(chain.get(key), source, context + ":chain:" + key)
        component_turns.append(chain[key]["turn_id"])
    _require(chain.get("first_complete_turn") == te and max(component_turns) == te,
             context + ": first complete chain does not match source E")
    prefix = review.get("pre_E_prefix", {})
    _require(prefix.get("end_turn") == te - 1 and bool(prefix.get("missing_element")) and
             bool(prefix.get("reason")), context + ": earlier prefix explanation is incomplete")
    events = review.get("all_operation_events")
    _require(isinstance(events, list) and bool(events), context + ": complete operation scan is missing")
    previous = 0
    for event in events:
        _quote_supported(event, source, context + ":operation")
        _require(event.get("operation_type") in OPERATION_TYPES, context + ": unsupported operation category")
        _require(event["turn_id"] >= previous, context + ": operation scan is unordered")
        previous = event["turn_id"]
    _require(min(item["turn_id"] for item in events) == ta, context + ": A is not the earliest observed operation")
    first = review["first_operation"]
    _require(any(item["turn_id"] == ta and item["operation_type"] == first.get("operation_type")
                 for item in events), context + ": first operation is absent from full scan")
    _require(bool(review.get("boundary_rationale")) and isinstance(review.get("reasons"), list) and
             len(review["reasons"]) >= 3, context + ": independent source reasoning is missing")


def _embedded_files(files, required, optional, context):
    _require(required.issubset(files) and set(files).issubset(required.union(optional)),
             context + ": embedded evidence file coverage differs")
    payloads = {}
    for filename, item in files.items():
        _require(isinstance(item.get("text"), str), context + ": embedded evidence is missing")
        raw = item["text"].encode("utf-8")
        _require(len(raw) == item.get("size_bytes") and _sha_bytes(raw) == item.get("sha256"),
                 context + ": embedded file byte hash differs: " + filename)
        payloads[filename] = item["text"]
    return payloads


def _reassessment_proofs(ledger):
    bundle = ledger.get("independent_validation_reassessment")
    if not bundle:
        return {}
    files = bundle.get("files", {})
    required = {"revalidated_proposals.jsonl", "validation_reassessment_records.jsonl",
                "validation_reassessment_summary.json", "validation_reassessment.py",
                "boundary_batch_verifier.py", "original_proposals_aggregate.jsonl"}
    texts = _embedded_files(files, required, set(), "mechanical validation reassessment")
    summary = json.loads(texts["validation_reassessment_summary.json"])
    original = {row["source_call_id"]: row for row in
                [json.loads(line) for line in texts["original_proposals_aggregate.jsonl"].splitlines() if line.strip()]}
    revalidated = {row["source_call_id"]: row for row in
                   [json.loads(line) for line in texts["revalidated_proposals.jsonl"].splitlines() if line.strip()]}
    records = {row["source_call_id"]: row for row in
               [json.loads(line) for line in texts["validation_reassessment_records.jsonl"].splitlines() if line.strip()]}
    _require(set(original) == set(revalidated) == set(records) and len(records) == summary.get("records"),
             "mechanical validation reassessment ID coverage differs")
    _require(summary.get("model_called") is False and summary.get("applied") is False and
             summary.get("original_aggregate_sha256_unchanged") == files["original_proposals_aggregate.jsonl"]["sha256"] and
             summary.get("validation_reassessment_script_sha256") == files["validation_reassessment.py"]["sha256"],
             "mechanical reassessment execution or original-byte binding differs")
    for filename in ("revalidated_proposals.jsonl", "validation_reassessment_records.jsonl"):
        _require(summary.get("files", {}).get(filename) == files[filename]["sha256"],
                 "mechanical reassessment output hash differs")
    pattern = "操作|请求|要求|付款|operation|request|payment"
    _require(pattern in texts["boundary_batch_verifier.py"], "formal reassessment rule evidence is absent")
    verified = {}
    for cid, new in revalidated.items():
        old, record = original[cid], records[cid]
        issues = old.get("validation_issues")
        _require(issues in ([], ["operation_first_sufficiency_not_explained"]),
                 cid + ": reassessment attempts to waive a substantive source review issue")
        _require(all(new.get(key) == value for key, value in old.items() if key != "validation_issues") and
                 new.get("validation_issues") == [] and new.get("original_validation_issues") == issues,
                 cid + ": reassessment changed a model observation or concealed original issues")
        _require(record.get("original_validation_issues") == issues and record.get("revalidated_validation_issues") == [] and
                 new.get("original_proposal_row_sha256") == record.get("original_proposal_row_sha256") == canonical_sha(old) and
                 new.get("validation_reassessment_record_sha256") == canonical_sha(record),
                 cid + ": reassessment original/record binding differs")
        _require(new.get("formal_validator_sha256") == record.get("formal_validator_file_sha256") ==
                 files["boundary_batch_verifier.py"]["sha256"] and
                 new.get("validation_reassessment_script_sha256") == files["validation_reassessment.py"]["sha256"],
                 cid + ": reassessment validator provenance differs")
        _require(record.get("model_called") is False and record.get("applied") is False and
                 all(record.get(key) is True for key in ("source_sha_binding_verified",
                     "prompt_input_schema_binding_verified", "response_session_turn_completed_verified",
                     "literal_quote_stage_validation_verified")), cid + ": reassessment checks are incomplete")
        basis = old.get("review", {}).get("evidence_chain_scan", {}).get("sufficiency_basis", "")
        _require(record.get("formal_basis_pattern") == pattern and record.get("actual_sufficiency_basis") == basis and
                 bool(record.get("reason")), cid + ": reassessment basis differs from real source reviewer")
        if issues:
            _require(re.search(pattern, basis, re.I) is not None,
                     cid + ": saved explanation does not satisfy the formal operation-completes-chain rule")
        verified[cid] = {"original": old, "revalidated": new, "record": record}
    return verified


def _validate_batch(name, batch, sources, reassessments=None, rejected_versions=None):
    files = batch.get("files", {})
    required = {"prompt.txt", "schema.json", "binding.json", "review.json",
                "review.json.cli.json", "proposals.jsonl"}
    payloads = _embedded_files(files, required, {"input.json", "batch_status.json"}, name)
    try:
        binding = json.loads(payloads["binding.json"])
        response = json.loads(payloads["review.json"])
        shape = json.loads(payloads["schema.json"])
        diagnostic = json.loads(payloads["review.json.cli.json"])
        proposals = [json.loads(line) for line in payloads["proposals.jsonl"].splitlines() if line.strip()]
        inputs = json.loads(payloads["prompt.txt"].split("\n材料：\n", 1)[1])
    except (ValueError, IndexError) as exc:
        raise OverrideValidationError(name + ": embedded review evidence is unparsable") from exc
    input_by_id = {item.get("audit_id"): item for item in inputs}
    _require(1 <= len(binding) <= 5 and len(input_by_id) == len(inputs) and
             set(binding) == set(response) == set(input_by_id) == set(shape.get("properties", {})) and
             set(shape.get("required", [])) == set(binding), name + ": opaque audit ID binding differs")
    command = diagnostic.get("command", [])
    _require(diagnostic.get("returncode") == 0 and diagnostic.get("model_requested") == MODEL and
             diagnostic.get("reasoning_effort") == "high" and "-m" in command and
             command[command.index("-m") + 1] == MODEL and "--ephemeral" in command and
             "--output-schema" in command, name + ": source reviewer invocation proof is invalid")
    session = diagnostic.get("cli_session_id")
    _require(isinstance(session, str) and re.fullmatch(r"[0-9a-fA-F-]{36}", session) is not None,
             name + ": actual CLI session is missing")
    event_types = diagnostic.get("event_types", [])
    _require("thread.started" in event_types and "turn.completed" in event_types and
             "turn.failed" not in event_types, name + ": reviewer did not complete")
    tail_events = []
    for line in diagnostic.get("stdout", "").splitlines():
        try:
            tail_events.append(json.loads(line))
        except ValueError:
            pass
    _require(any(item.get("type") == "turn.completed" for item in tail_events),
             name + ": actual stdout tail does not confirm completion")
    proposal_by_source = {item.get("source_call_id"): item for item in proposals}
    _require(len(proposal_by_source) == len(proposals) == len(binding), name + ": source proposal coverage differs")
    verified = {}
    for audit_id, bound in binding.items():
        cid = bound.get("source_call_id")
        _require(cid in sources and cid in proposal_by_source, name + ": source record is missing")
        source, proposal, review, input_item = sources[cid], proposal_by_source[cid], response[audit_id], input_by_id[audit_id]
        source_sha = canonical_sha(source)
        _require(bound.get("source_record_sha256") == proposal.get("source_record_sha256") == source_sha,
                 name + ": original source record hash differs for " + cid)
        _require(set(input_item) == {"audit_id", "language", "turns"} and input_item["turns"] == source["turns"],
                 name + ": source-review prompt provenance is exposed or transcript differs")
        _require(shape["properties"][audit_id]["properties"]["audit_id"].get("enum") == [audit_id],
                 name + ": source-review schema audit ID enum differs")
        _require(review.get("audit_id") == audit_id and proposal.get("review") == review and
                 proposal.get("applied") is False,
                 name + ": reviewed proposal is unbound or already applied")
        _require(proposal.get("actual_cli_session_id") == session and proposal.get("model_requested") == MODEL and
                 proposal.get("returncode") == 0 and proposal.get("prompt_sha256") == files["prompt.txt"]["sha256"] and
                 proposal.get("response_sha256") == canonical_sha(response), name + ": proposal execution hashes differ")
        rejected = (rejected_versions or {}).get((name, cid, canonical_sha(proposal)))
        if rejected:
            # Retain a genuine failed model row alongside valid rows from the
            # same raw batch. It must never become a selectable override proof.
            _require(proposal.get("validation_issues") == ["first_operation_not_verbatim"] and
                     rejected.get("original_validation_issues") == proposal["validation_issues"] and
                     rejected.get("applied") is False and bool(rejected.get("reason")),
                     cid + ": rejected source-review version is not bound to its real failure")
            failed = False
            try:
                _validate_review(review, source, cid)
            except OverrideValidationError as error:
                failed = "first_operation: quote is not verbatim" in str(error)
            _require(failed, cid + ": declared failed first-operation quote did not actually fail validation")
            rejected_versions.pop((name, cid, canonical_sha(proposal)))
            continue
        _validate_review(review, source, cid)
        reassessed = (reassessments or {}).get(cid)
        if reassessed and proposal != reassessed["original"]:
            # A later independent model session is a new source-review version;
            # the mechanical reassessment applies only to its original raw row.
            reassessed = None
        if reassessed:
            record = reassessed["record"]
            _require(proposal == reassessed["original"] and
                     record.get("original_proposal_file_sha256") == files["proposals.jsonl"]["sha256"] and
                     record.get("raw_response_file_sha256") == files["review.json"]["sha256"] and
                     record.get("raw_response_record_sha256") == canonical_sha(review) and
                     record.get("actual_cli_session_id") == session,
                     cid + ": reassessment does not bind the original CLI batch evidence")
            if "input.json" in files:
                _require(json.loads(payloads["input.json"]) == inputs and
                         proposal.get("input_file_sha256") == files["input.json"]["sha256"],
                         cid + ": source input-file binding differs")
            verified[cid] = reassessed["revalidated"]
        else:
            _require(proposal.get("validation_issues") == [], name + ": source review has unresolved validation issues")
            verified[cid] = proposal
    return verified


def validated_audit(labels, sources, audit_root=None):
    """Return a verified independent audit, or raise without modifying inputs."""
    audit_ledger_sha256(audit_root)
    ledger = json.loads(_audit_path(audit_root).read_text(encoding="utf-8"))
    _require(ledger.get("schema_version") == "v365.source_adjudication_override.audit.v1" and
             ledger.get("source_files_modified") is False and ledger.get("main_label_counts_changed") is False,
             "override ledger scope is invalid")
    _require(len(labels) == ledger.get("original_labels_record_count") == 2000,
             "complete adjudication mapping record count differs")
    _require(set(ledger.get("fixed_conflict_ids", [])) == set(EARLY_CONFLICT_SOURCE_SHA256),
             "original early-fraud conflict group is not completely represented")
    basis = ledger.get("authorization_and_policy_basis", {})
    _require(basis.get("current_user_request_verbatim") ==
             "把剩余未做的数据集修改全部完成，之后把无关的中间文件删除。最终留下清晰明了的结果与报告" and
             basis.get("earlier_question_is_explicit_approval") is False and bool(basis.get("authorization_interpretation")),
             "override authorization account is incomplete or misstates the earlier question")
    proof_versions = {}
    reassessments = _reassessment_proofs(ledger)
    rejected_versions = {}
    for rejected in ledger.get("rejected_source_review_versions", []):
        key = (rejected.get("batch"), rejected.get("source_call_id"), rejected.get("proposal_sha256"))
        _require(key not in rejected_versions and key[0] in ledger.get("independent_review_batches", {}) and
                 key[1] in sources, "rejected source-review version binding is invalid")
        rejected_versions[key] = rejected
    for name, batch in ledger.get("independent_review_batches", {}).items():
        reviewed = _validate_batch(name, batch, sources, reassessments, rejected_versions)
        for cid, proposal in reviewed.items():
            proof_versions.setdefault(cid, {})[name] = proposal
    _require(not rejected_versions, "declared rejected source-review version is absent from the bound raw evidence")
    proofs = {}
    selection = ledger.get("source_review_selection", {})
    if selection:
        _require(set(selection) == set(proof_versions), "source-review version selection coverage differs")
    for cid, versions in proof_versions.items():
        if selection:
            choice = selection[cid]
            _require(isinstance(choice, dict) and choice.get("batch") in versions,
                     cid + ": selected source-review batch is missing")
            selected = versions[choice["batch"]]
            _require(choice.get("proposal_sha256") == canonical_sha(selected),
                     cid + ": selected source-review proposal hash differs")
        else:
            _require(len(versions) == 1, cid + ": duplicate source reviews require explicit version selection")
            selected = next(iter(versions.values()))
        proofs[cid] = selected
    entries = ledger.get("records", [])
    ids = [entry.get("source_call_id") for entry in entries]
    unapplied = ledger.get("unapplied_review_proposal_ids", [])
    _require(len(unapplied) == len(set(unapplied)) and not set(unapplied).intersection(ids) and
             len(ids) == len(set(ids)) and set(ids).union(unapplied) == set(proofs) and
             set(EARLY_CONFLICT_SOURCE_SHA256).issubset(ids), "override entry/proof coverage differs")
    original_labels, effective_labels = copy.deepcopy(labels), copy.deepcopy(labels)
    for entry in entries:
        cid = entry["source_call_id"]
        _require(cid in sources and cid in labels, "override entry is missing a source/adjudication: " + cid)
        old, new = entry.get("original_adjudication", {}), entry.get("effective_adjudication", {})
        _require(canonical_sha(old) == entry.get("original_adjudication_sha256") and
                 canonical_sha(new) == entry.get("effective_adjudication_sha256") and labels[cid] in (old, new),
                 cid + ": original/effective adjudication record hash differs")
        _require(canonical_sha(sources[cid]) == entry.get("source_record_sha256") and
                 entry.get("independent_source_proposal") == proofs[cid], cid + ": independent source proof differs")
        changed = {key for key in set(old).union(new) if old.get(key) != new.get(key)}
        _require(changed == set(entry.get("changed_fields", [])) and changed.issubset(CHANGEABLE_FIELDS),
                 cid + ": unaudited immutable adjudication field changed")
        _require(old.get("final_label") == new.get("final_label") == "fraud" and
                 old.get("final_binary_label") == new.get("final_binary_label") == 1 and
                 old.get("final_action") == new.get("final_action") == "rewrite_dialogue" and
                 new.get("decision") == "explicit_fraud" and bool(entry.get("change_reason")),
                 cid + ": primary label/action/mechanism correction scope differs")
        review = proofs[cid]["review"]
        for key in ("tS", "tE", "tA"):
            _require(new.get(key + "_turn") == review.get("earliest_" + key + "_turn"),
                     cid + ": corrected source boundary differs from independent review")
        for field in ("core_evidence", "supporting_evidence"):
            _require({item.get("evidence_type") for item in old.get(field, [])} ==
                     {item.get("evidence_type") for item in new.get(field, [])}, cid + ": baseline evidence types changed")
            for evidence in new.get(field, []):
                _quote_supported(evidence, sources[cid], cid + ":" + field)
        _require(any(item.get("turn_id") == new["tE_turn"] for item in new.get("core_evidence", [])),
                 cid + ": corrected E lacks a supported core evidence entry")
        _require(_operations_without_explanation(new.get("operation_events", [])) ==
                 _operations_without_explanation(review["all_operation_events"]),
                 cid + ": corrected operation scan differs from independent source review")
        if cid in EARLY_CONFLICT_SOURCE_SHA256:
            _require(entry["source_record_sha256"] == EARLY_CONFLICT_SOURCE_SHA256[cid] and
                     old.get("decision") == "early_fraud" and old.get("tA_turn") is None and
                     old.get("operation_events") == [] and
                     {event["operation_type"] for event in new["operation_events"]} == {"transfer"},
                     cid + ": original fixed early-fraud conflict correction differs")
            payment_turns = [turn["turn_id"] for turn in sources[cid]["turns"]
                             if re.search(r"需绕过财务审批.*由个人先行垫付", turn.get("text", ""))]
            _require(payment_turns and min(payment_turns) == new["tA_turn"],
                     cid + ": original positive personal-payment request is not at the reviewed A")
        else:
            _require(old.get("decision") == "explicit_fraud", cid + ": additional correction must preserve explicit decision")
        original_labels[cid], effective_labels[cid] = copy.deepcopy(old), copy.deepcopy(new)
    _require(canonical_sha(original_labels) == ledger.get("original_labels_mapping_sha256") and
             canonical_sha(effective_labels) == ledger.get("effective_labels_mapping_sha256"),
             "effective/original complete adjudication mapping is not reproducible")
    return copy.deepcopy(ledger)


def apply_overrides(labels, sources, audit_root=None):
    """Deep-copy adjudications and apply only fully verified source corrections."""
    ledger = validated_audit(labels, sources, audit_root)
    effective = copy.deepcopy(labels)
    for entry in ledger["records"]:
        effective[entry["source_call_id"]] = copy.deepcopy(entry["effective_adjudication"])
    return effective


def resolved_conflict_ids(labels, sources, audit_root=None):
    """Return the resolved original early-fraud IDs only after full proof checks."""
    ledger = validated_audit(labels, sources, audit_root)
    return set(ledger["fixed_conflict_ids"])


def overridden_ids(labels, sources, audit_root=None):
    """All override IDs, including independently audited operation omissions."""
    ledger = validated_audit(labels, sources, audit_root)
    return {entry["source_call_id"] for entry in ledger["records"]}


def override_entry_hashes(labels, sources, audit_root=None):
    """Stable per-entry bindings; adding unrelated entries does not change these."""
    ledger = validated_audit(labels, sources, audit_root)
    return {entry["source_call_id"]: canonical_sha(entry) for entry in ledger["records"]}
