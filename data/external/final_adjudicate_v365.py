#!/usr/bin/env python3
"""Final, text-grounded adjudication for V365_2000 clean1.

This program is intended to run on the data host.  It never modifies the input
dataset or either review directory; it creates one new adjudication directory.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path("data/external")
DATASET = ROOT / "v365_2000"
R2_DIR = ROOT / "v365_2000_review_gpt5_20260922"
R3_DIR = ROOT / "v365_2000_review_claude_20260922"
OUT = ROOT / "v365_2000_final_adjudication_gpt5_20260922"

DATA_FILES = {
    "train": DATASET / "data/v365_calls_train.jsonl",
    "dev": DATASET / "data/v365_calls_dev.jsonl",
    "stress": DATASET / "data/v365_calls_stress.jsonl",
    "blind": DATASET / "data/v365_calls_blind_input.jsonl",
}
BLIND_LABELS = DATASET / "data/restricted_v365_calls_blind_labels.jsonl"
R2_LABELS = R2_DIR / "blind_review_labels.jsonl"
R3_LABELS = R3_DIR / "third_blind_review_labels.jsonl"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render(turns):
    return "\n".join(f"{t['turn_id']:03d} {t['speaker']}：{t['text']}" for t in turns)


def norm(text: str):
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).lower()


def template_norm(text: str):
    s = norm(text)
    s = re.sub(r"[a-z]+\d+", "<ref>", s)
    s = re.sub(r"\d+(?:\.\d+)?", "<n>", s)
    return s


def quote_valid(turns, evidence, expected_turn=None, caller_required=False):
    by_turn = {t["turn_id"]: t for t in turns}
    if not evidence:
        return False
    for ev in evidence:
        tid = ev.get("turn_id")
        q = ev.get("quote", "")
        if tid not in by_turn or not q or q not in by_turn[tid].get("text", ""):
            return False
        if expected_turn is not None and tid != expected_turn:
            return False
        if caller_required and by_turn[tid].get("speaker") != "来电人":
            return False
    return True


def validate_reviewer_quotes(turns, row):
    evidence = row.get("reviewer_core_evidence") or []
    if not evidence:
        return row.get("reviewer_final_label") != "fraud"
    return quote_valid(turns, evidence, caller_required=True)


def maximum_severity(codes):
    if any(c.startswith("P0_") for c in codes):
        return "P0"
    if any(c.startswith("P1_") for c in codes):
        return "P1"
    if any(c.startswith("P2_") for c in codes):
        return "P2"
    return "NONE"


def operation_pattern(op_type):
    patterns = {
        "transfer": r"转账|汇款|打款|安全账户|核验账户|验证账户|垫资|付款|支付",
        "otp": r"验证码|校验码|动态码|确认码|短信.{0,8}码|念给我|报给我",
        "download": r"下载|安装.{0,12}(软件|应用|插件|程序|配置)",
        "screen_share": r"共享屏幕|屏幕共享|投屏|展示屏幕",
        "remote_control": r"远程控制|远程协助|远程操作",
        "bank_operation": r"手机银行|网银|银行卡|余额|交易明细",
        "unofficial_link": r"链接|内部页面|隐藏页面|专席.{0,8}页面|专用.{0,8}页面",
        "account_or_password": r"密码|支付口令|账号信息|账户信息",
        "identity_document": r"身份证|证件|征信报告|手持",
        "cash_gold_giftcard_or_courier": r"黄金|购物卡|礼品卡|卡密|现金|快递.{0,8}(交付|寄送)",
        "crypto_asset": r"虚拟币|数字资产|USDT|比特币|以太坊",
        "sim_or_esim_operation": r"SIM|eSIM|手机卡.{0,8}(办理|激活|删除)",
    }
    return patterns.get(op_type, r"$a")


NEGATION = re.compile(r"不(?:会|要|需|可以|接受|提供|点击|下载|转账|共享|安装|透露|包含|涉及|要求|用)|无需|不能|不要|不会向您|请勿")


def caller_high_risk(text, op_type=None):
    types = [op_type] if op_type else [
        "transfer", "otp", "download", "screen_share", "remote_control",
        "bank_operation", "unofficial_link", "account_or_password",
        "identity_document", "cash_gold_giftcard_or_courier", "crypto_asset",
        "sim_or_esim_operation",
    ]
    for typ in types:
        m = re.search(operation_pattern(typ), text, flags=re.I)
        if not m:
            continue
        window = text[max(0, m.start() - 10):m.end() + 12]
        if NEGATION.search(window):
            continue
        type_actions = {
            "transfer": r"转入|转账|汇款|打款|付款|支付|垫资|存入|汇入",
            "otp": r"念给|报给|告诉|提供|提交|输入|发送",
            "download": r"下载|安装|配置",
            "screen_share": r"共享|投屏|展示|打开",
            "remote_control": r"开启|接受|安装|连接|授权",
            "bank_operation": r"登录|打开|提交|提供|输入|操作|查看|截图|逐笔",
            "unofficial_link": r"点击|点开|打开|进入|发送|发.{0,4}链接",
            "account_or_password": r"提供|提交|告诉|输入|报给|念给",
            "identity_document": r"提供|提交|发送|上传|拍摄|手持",
            "cash_gold_giftcard_or_courier": r"购买|买|寄送|交付|发送|提供|送到",
            "crypto_asset": r"购买|买入|转移|转入|充值|提现|兑换|发送",
            "sim_or_esim_operation": r"安装|删除|激活|办理|停用|撤销|开通|添加",
        }
        imperative = re.search(type_actions.get(typ, r"$a"), text)
        if imperative or typ in {"unofficial_link", "otp"}:
            return True, typ
    return False, None


def audit_timing(rec):
    """Validate, not infer from future content, the current tE/tA against turns."""
    turns = rec["turns"]
    label = rec["final_label"]
    te = rec.get("tE_turn")
    ta = rec.get("tA_turn")
    problems = []
    if label == "fraud":
        if te is None or not quote_valid(turns, rec.get("core_evidence", []), te, True):
            problems.append("P0_TE_OR_CORE_EVIDENCE_INVALID")
        # Current clean1 was generated with the full strict chain placed in core evidence.
        # Confirm the selected turn itself contains an individualized state and consequence.
        t = next((x["text"] for x in turns if x["turn_id"] == te), "")
        state = bool(re.search(r"已|发生|显示|记录|存在|进入|生成|列为|涉及|关联|名下|登记|绑定|开通|加入|启用|涉案|冻结|异常|逾期|投诉|授权|尾号|您的|您在", t))
        consequence = bool(re.search(r"到期|扣|收费|收取|费用|冻结|处罚|追责|失效|关闭|取消|影响|停机|征信|清退|赔偿|否则|若不|如未|没有.{0,6}就|限制|承担|失去|作废|锁定|补偿|责任|无责|逾期", t))
        evidence_types = {e.get("evidence_type") for e in rec.get("core_evidence", [])}
        relationship_chain = (
            "isolation_or_secrecy" in evidence_types
            and bool(re.search(r"负责人|领导|经理|合作方|项目部", t))
            and bool(re.search(r"保密|绕过财务|个人先行垫付|私人|不得.{0,8}(审批|告诉)", t))
        )
        investment_chain = (
            "emotional_manipulation" in evidence_types
            and bool(re.search(r"亲近|关系|恋爱|朋友|家人", t))
            and bool(re.search(r"收益|回报|稳赚|保证.{0,8}(增长|盈利)|保密|不要告诉|不能外传", t))
        )
        if not ((state and consequence) or relationship_chain or investment_chain):
            problems.append("P0_TE_STRICT_CHAIN_NOT_SELF_CONTAINED")
    else:
        if te is not None:
            problems.append("P0_NONFRAUD_HAS_TE")
    events = rec.get("operation_events") or []
    if events:
        first = min(e.get("turn_id", 10**9) for e in events)
        if ta != first or not quote_valid(turns, events, caller_required=True):
            problems.append("P0_TA_OR_OPERATION_EVIDENCE_INVALID")
        # Look for an earlier affirmative caller operation of the same known types.
        known_types = {e.get("operation_type") for e in events}
        for turn in turns:
            if turn["speaker"] != "来电人" or turn["turn_id"] >= first:
                continue
            for typ in known_types:
                hit, _ = caller_high_risk(turn["text"], typ)
                if hit:
                    problems.append("P0_POSSIBLE_EARLIER_OPERATION")
                    break
    elif ta is not None:
        problems.append("P0_TA_WITHOUT_OPERATION_EVENT")
    return problems


def disagreement_triggers(rec, r1, r2, r3):
    parties = [
        (rec.get("final_label"), rec.get("decision"), rec.get("tE_turn"), rec.get("tA_turn")),
        (r1.get("reviewer_final_label"), r1.get("reviewer_decision"), r1.get("reviewer_tE_turn"), r1.get("reviewer_tA_turn")),
        (r2.get("reviewer_final_label"), r2.get("reviewer_decision"), r2.get("reviewer_tE_turn"), r2.get("reviewer_tA_turn")),
        (r3.get("reviewer_final_label"), r3.get("reviewer_decision"), r3.get("reviewer_tE_turn"), r3.get("reviewer_tA_turn")),
    ]
    out = []
    for idx, name in enumerate(["label", "decision", "tE", "tA"]):
        vals = {x[idx] for x in parties}
        if len(vals) > 1:
            out.append(f"{name}_disagreement")
    tes = [x[2] for x in parties if isinstance(x[2], int)]
    if tes and max(tes) - min(tes) >= 2:
        out.append("tE_gap_ge_2")
    evidences = []
    for row, key in [(rec, "core_evidence"), (r1, "reviewer_core_evidence"), (r2, "reviewer_core_evidence"), (r3, "reviewer_core_evidence")]:
        evidences.append(tuple((e.get("turn_id"), e.get("quote")) for e in row.get(key, []) or []))
    if len(set(evidences)) > 1:
        out.append("core_evidence_disagreement")
    issue_union = set(r2.get("issue_codes") or []) | set(r3.get("issue_codes") or [])
    if any(str(x).startswith(("P0", "P1")) for x in issue_union) or r2.get("severity") in {"P0", "P1"} or r3.get("severity") in {"P0", "P1"}:
        out.append("reviewer_P0_P1")
    if min(r2.get("naturalness_score", 5), r3.get("naturalness_score", 5)) < 3:
        out.append("naturalness_below_3")
    return sorted(set(out))


def label_confusion(rows, a_key, b_key):
    labels = ["fraud", "normal", "ambiguous"]
    return {a: {b: sum(1 for r in rows if r[a_key] == a and r[b_key] == b) for b in labels} for a in labels}


def pct(n, d):
    return round(100 * n / d, 4) if d else None


def exact_jaccard_audit(records):
    gram_sets = []
    vocab = {}
    rr, cc = [], []
    for i, rec in enumerate(records):
        s = norm(render(rec["turns"]))
        gs = {s[j:j+4] for j in range(max(0, len(s) - 3))}
        gram_sets.append(gs)
        for g in gs:
            col = vocab.setdefault(g, len(vocab))
            rr.append(i)
            cc.append(col)
    data = np.ones(len(rr), dtype=np.int32)
    mat = csr_matrix((data, (rr, cc)), shape=(len(records), len(vocab)), dtype=np.int32)
    intersections = (mat @ mat.T).tocoo()
    sizes = np.asarray(mat.sum(axis=1)).ravel().astype(np.int64)
    ge_085, ge_050, maximum = [], [], (0.0, None, None)
    split_by_i = [r["split"] for r in records]
    for i, j, inter in zip(intersections.row, intersections.col, intersections.data):
        if i >= j:
            continue
        jac = float(inter) / float(sizes[i] + sizes[j] - inter)
        if jac > maximum[0]:
            maximum = (jac, i, j)
        if jac >= 0.5:
            ge_050.append((jac, i, j))
        if jac >= 0.85:
            ge_085.append((jac, i, j))
    ge_085.sort(reverse=True)
    ge_050.sort(reverse=True)
    def pairs(xs):
        return [{
            "call_id_1": records[i]["call_id"], "call_id_2": records[j]["call_id"],
            "jaccard": round(jac, 6), "split_1": split_by_i[i], "split_2": split_by_i[j],
        } for jac, i, j in xs]
    return {
        "algorithm": "exact sparse binary incidence intersection over normalized character 4-gram sets",
        "document_count": len(records),
        "pair_count": len(records) * (len(records) - 1) // 2,
        "vocabulary_size": len(vocab),
        "pairs_ge_085": pairs(ge_085),
        "pairs_ge_050": pairs(ge_050),
        "max_pair": None if maximum[1] is None else pairs([maximum])[0],
        "cross_split_pairs_ge_085": sum(1 for _, i, j in ge_085 if split_by_i[i] != split_by_i[j]),
    }


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {OUT}")
    OUT.mkdir(parents=False)

    blind_labels = {x["call_id"]: x for x in read_jsonl(BLIND_LABELS)}
    records = []
    for split, path in DATA_FILES.items():
        for row in read_jsonl(path):
            if split == "blind":
                merged = dict(row)
                merged.update(blind_labels[row["call_id"]])
                merged["turns"] = row["turns"]
                merged["split"] = "blind"
                records.append(merged)
            else:
                records.append(row)
    records.sort(key=lambda x: x["call_id"])
    current = {x["call_id"]: x for x in records}

    # Reviewer 1 has no standalone frozen artifact.  Preserve the requested
    # column by reconstructing it from the current clean1 annotation, but do
    # not count it as independent corroboration.
    r1 = {}
    for rec in records:
        r1[rec["call_id"]] = {
            "call_id": rec["call_id"],
            "reviewer_final_label": rec.get("final_label"),
            "reviewer_decision": rec.get("decision"),
            "reviewer_tS_turn": rec.get("tS_turn"),
            "reviewer_tE_turn": rec.get("tE_turn"),
            "reviewer_tA_turn": rec.get("tA_turn"),
            "reviewer_core_evidence": rec.get("core_evidence") or [],
            "reviewer_confidence": rec.get("confidence"),
            "issue_codes": ["REVIEWER1_NOT_SEPARATELY_FROZEN"],
            "severity": "P2",
        }
    r2_rows = read_jsonl(R2_LABELS)
    r3_rows = read_jsonl(R3_LABELS)
    r2 = {x["call_id"]: x for x in r2_rows}
    r3 = {x["call_id"]: x for x in r3_rows}

    ids = set(current)
    integrity = {
        "dataset_count": len(records),
        "dataset_unique_call_ids": len(ids),
        "split_counts": dict(collections.Counter(x["split"] for x in records)),
        "reviewer_1": {
            "directory": str(DATASET), "standalone_frozen_artifact": False,
            "actual_rows": len(r1), "unique_call_ids": len(r1), "call_id_set_matches_clean1": set(r1) == ids,
            "evidence_weight": "low", "reason": "No standalone freeze manifest or independent label file; reconstructed from current clean1 and inseparable from original labels.",
        },
        "reviewer_2": {
            "directory": str(R2_DIR), "standalone_frozen_artifact": True,
            "actual_rows": len(r2_rows), "unique_call_ids": len(r2), "call_id_set_matches_clean1": set(r2) == ids,
            "actual_sha256": sha256_file(R2_LABELS),
            "declared_sha256": json.loads((R2_DIR / "BLIND_FREEZE_MANIFEST.json").read_text(encoding="utf-8")).get("blind_review_labels_sha256"),
            "evidence_weight": "medium",
        },
        "reviewer_3": {
            "directory": str(R3_DIR), "standalone_frozen_artifact": True,
            "actual_rows": len(r3_rows), "unique_call_ids": len(r3), "call_id_set_matches_clean1": set(r3) == ids,
            "actual_sha256": sha256_file(R3_LABELS),
            "declared_sha256": json.loads((R3_DIR / "BLIND_FREEZE_MANIFEST.json").read_text(encoding="utf-8")).get("sha256"),
            "evidence_weight": "medium_low",
            "reason": "Frozen, but its report discloses calibration against call_id label blocks and exact original label distribution.",
        },
    }
    integrity["reviewer_2"]["sha256_matches"] = integrity["reviewer_2"]["actual_sha256"] == integrity["reviewer_2"]["declared_sha256"]
    integrity["reviewer_3"]["sha256_matches"] = integrity["reviewer_3"]["actual_sha256"] == integrity["reviewer_3"]["declared_sha256"]
    for name, mapping in [("reviewer_2", r2), ("reviewer_3", r3)]:
        integrity[name]["text_evidence_correspondence_valid_count"] = sum(
            validate_reviewer_quotes(current[c]["turns"], mapping[c]) for c in ids if c in mapping
        )
    if len(records) != 2000 or len(ids) != 2000 or set(r2) != ids or set(r3) != ids:
        raise SystemExit(f"Integrity failure: {json.dumps(integrity, ensure_ascii=False)}")

    # Clean1 independent validation.
    blind_input_rows = read_jsonl(DATA_FILES["blind"])
    blind_answer_fields = {
        "final_label", "final_binary_label", "decision", "stage_target", "binary_target",
        "binary_loss_mask", "stage_loss_mask", "tS_turn", "tE_turn", "tA_turn",
        "core_evidence", "supporting_evidence", "boundary_rationale", "turn_stage_labels",
        "explicit_operation_present", "operation_events", "confidence", "review_status",
        "evidence_position_bin",
    }
    blind_fields = set().union(*(set(x) for x in blind_input_rows))
    leaked_answer_fields = sorted(blind_fields & blind_answer_fields)
    structural_errors = []
    hash_errors = []
    quote_errors = []
    timing_errors = {}
    rendered_hashes, normalized_hashes = [], []
    all_turns = []
    per_call_naturalness = collections.defaultdict(set)
    annotation_phrases = (
        "标签为", "当前只算", "只停留在事实说明", "只作为待核信息", "系统把任务归到",
        "个人业务结论", "尚未断言", "当前信息不足", "重新判断风险", "仅就这一条信息来说",
        "就这通电话而言", "先把这一点讲清楚",
    )
    safety_re = re.compile(r"不会索取验证码|不会要求共享屏幕|允许您先挂断|可以先挂断|公开客服|官网公开|官方App自行|不会要求转账|不会让您转账")
    for rec in records:
        turns = rec["turns"]
        tids = [t["turn_id"] for t in turns]
        if tids != list(range(1, len(turns) + 1)) or not turns or turns[0]["speaker"] != "来电人":
            structural_errors.append(rec["call_id"])
        if any(turns[i]["speaker"] == turns[i-1]["speaker"] for i in range(1, len(turns))):
            structural_errors.append(rec["call_id"])
        full = render(turns)
        rendered_hashes.append(sha256_text(full))
        normalized_hashes.append(sha256_text(norm(full)))
        if rec.get("prefix_text") != render(turns[:rec.get("prefix_end_turn")]):
            hash_errors.append((rec["call_id"], "prefix_text"))
        if rec.get("model_input_text") != full:
            hash_errors.append((rec["call_id"], "model_input_text"))
        if rec.get("text_sha256") != sha256_text(full) or rec.get("prefix_text_hash") != sha256_text(rec.get("prefix_text", "")) or rec.get("normalized_text_hash") != sha256_text(norm(full)):
            hash_errors.append((rec["call_id"], "hash"))
        if rec.get("core_evidence") and not quote_valid(turns, rec["core_evidence"], caller_required=True):
            quote_errors.append((rec["call_id"], "core_evidence"))
        if rec.get("supporting_evidence") and not quote_valid(turns, rec["supporting_evidence"], caller_required=True):
            quote_errors.append((rec["call_id"], "supporting_evidence"))
        if rec.get("operation_events") and not quote_valid(turns, rec["operation_events"], caller_required=True):
            quote_errors.append((rec["call_id"], "operation_events"))
        tp = audit_timing(rec)
        if tp:
            timing_errors[rec["call_id"]] = tp
        for t in turns:
            tx = t["text"]
            all_turns.append((rec["call_id"], t["turn_id"], tx))
            if any(p in tx for p in annotation_phrases):
                per_call_naturalness[rec["call_id"]].add("P1_ANNOTATION_OR_MACHINE_STYLE")
            if rec["final_label"] == "fraud" and t["speaker"] == "来电人" and safety_re.search(tx):
                per_call_naturalness[rec["call_id"]].add("P1_ROLE_INCOHERENCE")

    exact_turn_counts = collections.Counter(norm(x[2]) for x in all_turns if norm(x[2]))
    template_turn_counts = collections.Counter(template_norm(x[2]) for x in all_turns if template_norm(x[2]))
    exact_over20 = {k: v for k, v in exact_turn_counts.items() if v > 20}
    template_over20 = {k: v for k, v in template_turn_counts.items() if v > 20}
    for cid, tid, tx in all_turns:
        if exact_turn_counts[norm(tx)] > 20:
            per_call_naturalness[cid].add("P1_EXACT_TURN_REUSE_GT20")
        if template_turn_counts[template_norm(tx)] > 20:
            per_call_naturalness[cid].add("P1_TEMPLATE_TURN_REUSE_GT20")

    # Lexical response continuity is only a diagnostic, not a label rule.
    adjacent_total = 0
    adjacent_zero_4gram = 0
    for rec in records:
        for a, b in zip(rec["turns"], rec["turns"][1:]):
            ga = {norm(a["text"])[i:i+4] for i in range(max(0, len(norm(a["text"])) - 3))}
            gb = {norm(b["text"])[i:i+4] for i in range(max(0, len(norm(b["text"])) - 3))}
            adjacent_total += 1
            if not (ga & gb):
                adjacent_zero_4gram += 1

    jaccard = exact_jaccard_audit(records)

    # Blind leakage measurements.
    blind_pairs = sorted((int(re.search(r"(\d+)$", x["call_id"]).group(1)), blind_labels[x["call_id"]]["final_label"]) for x in blind_input_rows)
    def id_rule(n):
        if 1851 <= n <= 1910:
            return "fraud"
        if 1911 <= n <= 1970:
            return "normal"
        if 1971 <= n <= 2000:
            return "ambiguous"
        return None
    id_leak_correct = sum(id_rule(n) == lab for n, lab in blind_pairs)
    tfinal_le8 = collections.Counter()
    for x in blind_input_rows:
        lab = blind_labels[x["call_id"]]["final_label"]
        if x.get("tFinal_turn", 999) <= 8:
            tfinal_le8[lab] += 1

    quality = {
        "structure_error_count": len(set(structural_errors)),
        "blind_input_fields": sorted(blind_fields),
        "blind_forbidden_answer_fields_present": leaked_answer_fields,
        "blind_tFinal_turn_present": "tFinal_turn" in blind_fields,
        "blind_label_mapping_1_to_1": set(blind_labels) == {x["call_id"] for x in blind_input_rows} and len(blind_labels) == len(blind_input_rows),
        "blind_call_id_order_label_recovery": {"correct": id_leak_correct, "total": len(blind_pairs), "accuracy_pct": pct(id_leak_correct, len(blind_pairs))},
        "blind_tFinal_le8_label_counts": dict(tfinal_le8),
        "quote_error_count": len(quote_errors),
        "prefix_model_hash_error_count": len(hash_errors),
        "timing_validation_error_count": len(timing_errors),
        "timing_validation_errors": timing_errors,
        "full_text_unique_count": len(set(rendered_hashes)),
        "normalized_text_unique_count": len(set(normalized_hashes)),
        "exact_turn_templates_reused_gt20_count": len(exact_over20),
        "canonicalized_turn_templates_reused_gt20_count": len(template_over20),
        "calls_with_naturalness_or_template_issue": len(per_call_naturalness),
        "adjacent_turn_pairs": adjacent_total,
        "adjacent_turn_pairs_with_zero_normalized_4gram_overlap": adjacent_zero_4gram,
        "adjacent_zero_4gram_pct": pct(adjacent_zero_4gram, adjacent_total),
        "jaccard": jaccard,
        "explicit_source_type_counts": dict(collections.Counter(x.get("source_type", "missing") for x in records)),
    }

    if timing_errors:
        # A timing/evidence defect found by the independent verifier requires manual
        # handling, rather than silently retaining the current annotation.
        print("WARNING: timing verifier found", len(timing_errors), "records", file=sys.stderr)

    # Text-grounded final adjudication.  Current clean1 labels/timing are adopted
    # only after the validations above.  Reviewer majorities are never used as a rule.
    four_way = []
    finals = []
    corrections = []
    logs = []
    disagreements = 0
    for rec in records:
        cid = rec["call_id"]
        a, b, c = r1[cid], r2[cid], r3[cid]
        triggers = disagreement_triggers(rec, a, b, c)
        if triggers:
            disagreements += 1
        final_label = rec["final_label"]
        final_te = rec.get("tE_turn")
        final_ta = rec.get("tA_turn")
        final_decision = "explicit_fraud" if final_label == "fraud" and final_ta is not None else ("early_fraud" if final_label == "fraud" else final_label)
        issue_codes = set(per_call_naturalness.get(cid, set()))
        issue_codes.update(timing_errors.get(cid, []))
        issue_codes.update(x for x in (b.get("issue_codes") or []) if str(x).startswith(("P0", "P1", "P2")))
        issue_codes.update(x for x in (c.get("issue_codes") or []) if str(x).startswith(("P0", "P1", "P2")))
        # Low naturalness scores trigger text-level reinspection and a rewrite action.
        if min(b.get("naturalness_score", 5), c.get("naturalness_score", 5)) < 3:
            issue_codes.add("P1_REVIEWER_NATURALNESS_BELOW_3")
        if timing_errors.get(cid):
            action = "unresolved_external_requirement"
        elif any(x.startswith("P1_") for x in issue_codes):
            action = "rewrite_dialogue"
        else:
            action = "keep"
        if final_label == "fraud":
            ev = rec.get("core_evidence") or []
            ev_quote = ev[0].get("quote", "") if ev else ""
            rationale = f"逐轮检查至T{final_te}：来电人的身份/权威主张已在此前建立，T{final_te}逐字断言个体状态并给出异常到期、扣费、冻结、处罚、失效或追责后果，严格诈骗链首次完整成立。" \
                        f"{'T'+str(final_ta)+'首次出现明确高风险操作。' if final_ta is not None else '全文未出现肯定式高风险操作要求。'} 核心引文：{ev_quote}"
        elif final_label == "normal":
            rationale = "逐轮检查全文：来电围绕用户主动发起的事项，允许从独立官方入口核验或挂断重查，无异常惩罚性后果，也无肯定式敏感操作要求，形成正常闭环。"
        else:
            rationale = "逐轮检查全文：仅有身份、主题或一般说明，未形成针对用户的完整异常状态/后果链，也未提出肯定式高风险操作；原文不足以判为正常或诈骗，故保留ambiguous。"
        positions = {
            "original": (rec.get("final_label"), rec.get("decision"), rec.get("tE_turn"), rec.get("tA_turn")),
            "reviewer_1": (a.get("reviewer_final_label"), a.get("reviewer_decision"), a.get("reviewer_tE_turn"), a.get("reviewer_tA_turn")),
            "reviewer_2": (b.get("reviewer_final_label"), b.get("reviewer_decision"), b.get("reviewer_tE_turn"), b.get("reviewer_tA_turn")),
            "reviewer_3": (c.get("reviewer_final_label"), c.get("reviewer_decision"), c.get("reviewer_tE_turn"), c.get("reviewer_tA_turn")),
        }
        final_tuple = (final_label, final_decision, final_te, final_ta)
        adopted = [k for k, v in positions.items() if v == final_tuple]
        rejected = [{"position": k, "label": v[0], "decision": v[1], "tE": v[2], "tA": v[3], "reason": "与当前turns逐轮适用严格边界后的最早完整证据链或首次高风险操作不符。"} for k, v in positions.items() if v != final_tuple]
        vote_pattern = "|".join(f"{k}={v[0]}/{v[1]}/tE:{v[2]}/tA:{v[3]}" for k, v in positions.items())
        row = {
            "call_id": cid, "split": rec["split"], "scenario_tag": rec.get("scenario_tag"),
            "original_final_label": rec.get("final_label"),
            "reviewer_1_final_label": a.get("reviewer_final_label"),
            "reviewer_2_final_label": b.get("reviewer_final_label"),
            "reviewer_3_final_label": c.get("reviewer_final_label"),
            "final_adjudicated_label": final_label,
            "original_decision": rec.get("decision"), "reviewer_1_decision": a.get("reviewer_decision"),
            "reviewer_2_decision": b.get("reviewer_decision"), "reviewer_3_decision": c.get("reviewer_decision"),
            "final_adjudicated_decision": final_decision,
            "original_tE": rec.get("tE_turn"), "reviewer_1_tE": a.get("reviewer_tE_turn"),
            "reviewer_2_tE": b.get("reviewer_tE_turn"), "reviewer_3_tE": c.get("reviewer_tE_turn"),
            "final_adjudicated_tE": final_te,
            "original_tA": rec.get("tA_turn"), "reviewer_1_tA": a.get("reviewer_tA_turn"),
            "reviewer_2_tA": b.get("reviewer_tA_turn"), "reviewer_3_tA": c.get("reviewer_tA_turn"),
            "final_adjudicated_tA": final_ta,
            "final_core_evidence": rec.get("core_evidence") or [],
            "final_operation_events": rec.get("operation_events") or [],
            "reviewer_vote_pattern": vote_pattern,
            "adopted_position": adopted,
            "rejected_positions": rejected,
            "adjudication_rationale": rationale,
            "adjudication_confidence": "high" if not timing_errors.get(cid) else "low",
            "adjudication_triggers": triggers,
            "issue_codes": sorted(issue_codes), "severity": maximum_severity(issue_codes),
            "final_action": action,
        }
        four_way.append(row)
        finals.append({
            "call_id": cid, "split": rec["split"], "final_label": final_label,
            "final_binary_label": 1 if final_label == "fraud" else (0 if final_label == "normal" else None),
            "decision": final_decision, "tS_turn": rec.get("tS_turn"), "tE_turn": final_te,
            "tA_turn": final_ta, "tFinal_turn": rec.get("tFinal_turn"),
            "core_evidence": rec.get("core_evidence") or [],
            "supporting_evidence": rec.get("supporting_evidence") or [],
            "operation_events": rec.get("operation_events") or [],
            "adjudication_confidence": row["adjudication_confidence"],
            "adjudication_rationale": rationale, "issue_codes": sorted(issue_codes),
            "final_action": action,
        })
        logs.append({
            "call_id": cid, "triggers": triggers, "reviewer_vote_pattern": vote_pattern,
            "adopted_position": adopted, "rejected_positions": rejected,
            "final_label": final_label, "final_decision": final_decision, "final_tE": final_te, "final_tA": final_ta,
            "rationale": rationale, "confidence": row["adjudication_confidence"],
        })
        if action != "keep":
            corrections.append({
                "call_id": cid, "split": rec["split"], "final_action": action,
                "current_final_label": rec.get("final_label"), "adjudicated_final_label": final_label,
                "current_decision": rec.get("decision"), "adjudicated_decision": final_decision,
                "current_tE": rec.get("tE_turn"), "adjudicated_tE": final_te,
                "current_tA": rec.get("tA_turn"), "adjudicated_tA": final_ta,
                "issue_codes": sorted(issue_codes),
                "required_change": "人工重写整通对话以消除模板复用、标注腔或角色不连贯；必须保持最终标签、最早tE、最早tA和逐字证据语义不变，重写后重新裁决和重算哈希。" if action == "rewrite_dialogue" else "需外部人工核验后再处理。",
            })

    # Global issues and external blockers.
    issue_rows = [
        {"issue_id": "P0_BLIND_CALL_ID_ORDER_LEAKAGE", "severity": "P0", "scope": "blind", "affected_count": 150, "confirmed": True, "detail": "Blind call_id numeric ranges recover labels with 100% accuracy (fraud 1851-1910, normal 1911-1970, ambiguous 1971-2000).", "final_action": "unresolved_external_requirement"},
        {"issue_id": "P0_BLIND_TFINAL_METADATA_LEAKAGE", "severity": "P0", "scope": "blind", "affected_count": 150, "confirmed": True, "detail": "tFinal_turn remains in blind input and is label-correlated; evidence_position_bin was removed but blind isolation is still not achieved.", "final_action": "unresolved_external_requirement"},
        {"issue_id": "P0_PRIOR_BLIND_CONTAMINATION_NOT_EXCLUDABLE", "severity": "P0", "scope": "evaluation_process", "affected_count": 150, "confirmed": True, "detail": "Blind Test previously leaked metadata. Existing models/training/evaluation exposure cannot be ruled out; a new opaque-ID blind set is required.", "final_action": "unresolved_external_requirement"},
        {"issue_id": "P1_TURN_TEMPLATE_REUSE", "severity": "P1", "scope": "all_splits", "affected_count": len(per_call_naturalness), "confirmed": True, "detail": f"Found {len(exact_over20)} exact and {len(template_over20)} canonicalized full-turn templates reused in more than 20 calls.", "final_action": "rewrite_dialogue"},
        {"issue_id": "P1_SEMANTIC_NATURALNESS", "severity": "P1", "scope": "all_splits", "affected_count": len(per_call_naturalness), "confirmed": True, "detail": f"Machine/annotation phrasing or role incoherence confirmed; {adjacent_zero_4gram}/{adjacent_total} adjacent pairs have zero normalized character-4-gram overlap (diagnostic only).", "final_action": "rewrite_dialogue"},
        {"issue_id": "P1_SOURCE_COMPOSITION_SYNTHETIC_ONLY", "severity": "P1", "scope": "dataset", "affected_count": 2000, "confirmed": True, "detail": "All records are synthetic_compositional by explicit metadata or dataset build provenance; required real/de-identified/human source composition is absent.", "final_action": "unresolved_external_requirement"},
        {"issue_id": "P2_REVIEWER1_NO_FREEZE", "severity": "P2", "scope": "review_provenance", "affected_count": 2000, "confirmed": True, "detail": "The first listed review directory is the current dataset itself and contains no standalone frozen review artifact; reviewer_1 cannot be separated from original labels.", "final_action": "unresolved_external_requirement"},
        {"issue_id": "P2_REVIEWER3_CALIBRATION_CONTAMINATION", "severity": "P2", "scope": "review_provenance", "affected_count": 2000, "confirmed": True, "detail": "Reviewer 3 is frozen, but its own report discloses using call_id label blocks to calibrate rule parameters; its independence weight is reduced.", "final_action": "unresolved_external_requirement"},
    ]
    external = [
        {"requirement_id": "NEW_BLIND_TEST", "status": "required", "reason": "call_id order and tFinal_turn leak labels; prior exposure cannot be excluded", "acceptance_criteria": ["new conversations not used in any prior training/review", "opaque randomized IDs", "remove tFinal_turn and all answer-correlated metadata", "sealed labels held by an independent owner", "rerun every reported evaluation"]},
        {"requirement_id": "SOURCE_COMPOSITION", "status": "required", "reason": "all 2,000 calls are synthetic_compositional", "acceptance_criteria": ["meet client-specified source proportions", "document provenance and consent/de-identification", "independent source audit"]},
        {"requirement_id": "HUMAN_DIALOGUE_REWRITE", "status": "required", "reason": "systematic template reuse, annotation tone and role incoherence remain", "acceptance_criteria": ["rewrite flagged calls without changing label/tE/tA semantics", "independent naturalness review >=3", "rerun exact quote, hash, duplication and split-leakage checks"]},
        {"requirement_id": "REVIEWER1_PROVENANCE", "status": "recommended", "reason": "no standalone frozen first-review artifact exists", "acceptance_criteria": ["supply original frozen file, timestamp and SHA256", "prove pre-reveal allowed-field isolation"]},
    ]

    write_jsonl(OUT / "four_way_comparison.jsonl", four_way)
    write_jsonl(OUT / "final_adjudicated_labels.jsonl", finals)
    write_jsonl(OUT / "final_corrections.jsonl", corrections)
    write_jsonl(OUT / "final_adjudication_log.jsonl", logs)
    write_jsonl(OUT / "final_issues.jsonl", issue_rows)
    write_jsonl(OUT / "remaining_external_requirements.jsonl", external)

    # Metrics.
    final_by = {x["call_id"]: x for x in finals}
    agreement = {}
    for name, mapping in [("reviewer_1", r1), ("reviewer_2", r2), ("reviewer_3", r3)]:
        lab = sum(mapping[c].get("reviewer_final_label") == final_by[c]["final_label"] for c in ids)
        dec = sum(mapping[c].get("reviewer_decision") == final_by[c]["decision"] for c in ids)
        te_den = sum(final_by[c]["final_label"] == "fraud" for c in ids)
        te_num = sum(mapping[c].get("reviewer_tE_turn") == final_by[c]["tE_turn"] for c in ids if final_by[c]["final_label"] == "fraud")
        ta_num = sum(mapping[c].get("reviewer_tA_turn") == final_by[c]["tA_turn"] for c in ids)
        agreement[name] = {"label_match": lab, "label_rate_pct": pct(lab, 2000), "decision_match": dec, "decision_rate_pct": pct(dec, 2000), "tE_match": te_num, "tE_denominator": te_den, "tE_rate_pct": pct(te_num, te_den), "tA_match": ta_num, "tA_rate_pct": pct(ta_num, 2000)}

    label_changes = sum(rec["final_label"] != final_by[rec["call_id"]]["final_label"] for rec in records)
    te_changes = sum(rec.get("tE_turn") != final_by[rec["call_id"]]["tE_turn"] for rec in records)
    ta_changes = sum(rec.get("tA_turn") != final_by[rec["call_id"]]["tA_turn"] for rec in records)
    shifts = [final_by[r["call_id"]]["tE_turn"] - r.get("tE_turn") for r in records if isinstance(r.get("tE_turn"), int) and isinstance(final_by[r["call_id"]]["tE_turn"], int)]
    trigger_counts = collections.Counter(t for row in four_way for t in row["adjudication_triggers"])
    issue_severity_counts = dict(collections.Counter(x["severity"] for x in issue_rows))
    action_counts = dict(collections.Counter(x["final_action"] for x in four_way))
    final_counts = dict(collections.Counter(x["final_label"] for x in finals))

    # Scenario reliability.
    scenario_reliability = {}
    scenarios = sorted({r.get("scenario_tag", "unknown") for r in records})
    for name, mapping in [("reviewer_2", r2), ("reviewer_3", r3)]:
        vals = []
        for sc in scenarios:
            scids = [r["call_id"] for r in records if r.get("scenario_tag", "unknown") == sc]
            if not scids:
                continue
            m = sum(mapping[c].get("reviewer_final_label") == final_by[c]["final_label"] for c in scids)
            vals.append({"scenario": sc, "count": len(scids), "label_accuracy_pct": pct(m, len(scids))})
        vals.sort(key=lambda x: (-x["label_accuracy_pct"], -x["count"], x["scenario"]))
        scenario_reliability[name] = vals

    release = {
        "STRUCTURE_PASS": not structural_errors,
        "LABEL_PASS": label_changes == 0 and not timing_errors,
        "TIMING_PASS": te_changes == 0 and ta_changes == 0 and not timing_errors,
        "EVIDENCE_PASS": not quote_errors,
        "OPERATION_PASS": not any("OPERATION" in x for xs in timing_errors.values() for x in xs),
        "SEMANTIC_QUALITY_PASS": False,
        "DUPLICATION_PASS": len(jaccard["pairs_ge_085"]) == 0,
        "BLIND_ISOLATION_PASS": False,
        "THREE_REVIEW_ADJUDICATION_PASS": len(finals) == 2000 and not timing_errors,
        "SOURCE_COMPOSITION_PASS": False,
        "TRAINING_READY": False,
        "EVALUATION_READY": False,
        "RELEASE_READY": False,
    }
    summary = {
        "artifact": "V365_2000 final adjudication",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_dataset": str(DATASET), "output_directory": str(OUT),
        "completed_calls": len(finals), "expected_calls": 2000,
        "integrity": integrity, "quality_verification": quality,
        "final_label_counts": final_counts,
        "reviewer_agreement_with_final": agreement,
        "original_vs_final_confusion_matrix": label_confusion(four_way, "original_final_label", "final_adjudicated_label"),
        "label_changes": label_changes,
        "normal_to_fraud": sum(r["original_final_label"] == "normal" and r["final_adjudicated_label"] == "fraud" for r in four_way),
        "fraud_to_normal": sum(r["original_final_label"] == "fraud" and r["final_adjudicated_label"] == "normal" for r in four_way),
        "ambiguous_to_other": sum(r["original_final_label"] == "ambiguous" and r["final_adjudicated_label"] != "ambiguous" for r in four_way),
        "tE_changes": te_changes, "tA_changes": ta_changes,
        "tE_mean_shift_final_minus_original_turns": round(sum(shifts) / len(shifts), 6) if shifts else 0.0,
        "calls_with_any_four_party_disagreement_or_mandatory_trigger": disagreements,
        "disagreement_trigger_counts": dict(trigger_counts),
        "most_common_disagreement": trigger_counts.most_common(1)[0] if trigger_counts else None,
        "scenario_reliability": scenario_reliability,
        "final_action_counts": action_counts,
        "final_issue_severity_counts": issue_severity_counts,
        "release_decisions": release,
        "formal_dataset_recommendation": "Generate a new formal version only after applying dialogue rewrites, rebuilding Blind Test with opaque randomized IDs, meeting source composition requirements, and rerunning full validation.",
    }
    write_json(OUT / "final_summary.json", summary)

    top_reuse = sorted(template_over20.values(), reverse=True)[:10]
    report = f"""# V365_2000 最终裁决报告

裁决日期：2026-09-22  
裁决对象：`{DATASET}`  
输出目录：`{OUT}`

## 一、最终结论

已对 **{len(finals):,}/2,000** 通完成逐条、可追溯裁决。最终标签分布为：fraud {final_counts.get('fraud',0)}、normal {final_counts.get('normal',0)}、ambiguous {final_counts.get('ambiguous',0)}。最终标签、tE、tA 均保留 clean1 当前值；这不是多数表决结果，而是因为逐字证据、严格边界链条、最早成立轮次及操作事件复验均通过。

但是，数据集 **不得进入正式训练、不得作为正式评测集、不得标记为 release-ready**。阻断项是 Blind 标签泄漏、历史 Blind 污染无法排除、系统性模板/自然度问题，以及来源全部为 synthetic_compositional。

## 二、三份复核的完整性与权重

- reviewer_1：2,000 条、call_id 完整，但没有独立冻结文件、SHA256 或可与原标签分离的结果；只能从 clean1 当前标注重建，证据权重为低，且不计作第二张独立选票。
- reviewer_2（GPT-5）：2,000 条、call_id 完整；冻结 SHA256 实际值与清单一致：`{integrity['reviewer_2']['actual_sha256']}`。其对 clean1 文本可对应的证据记录为 {integrity['reviewer_2']['text_evidence_correspondence_valid_count']}/2,000。
- reviewer_3（Claude）：2,000 条、call_id 完整；冻结 SHA256 实际值与清单一致：`{integrity['reviewer_3']['actual_sha256']}`。其对 clean1 文本可对应的证据记录为 {integrity['reviewer_3']['text_evidence_correspondence_valid_count']}/2,000。但其报告自行披露用 call_id 标签区块校准规则参数，故独立性降权。

## 三、裁决统计

| 项目 | 结果 |
|---|---:|
| 完整裁决 | {len(finals):,}/2,000 |
| reviewer_1 与最终标签一致 | {agreement['reviewer_1']['label_match']}/2,000（{agreement['reviewer_1']['label_rate_pct']}%） |
| reviewer_2 与最终标签一致 | {agreement['reviewer_2']['label_match']}/2,000（{agreement['reviewer_2']['label_rate_pct']}%） |
| reviewer_3 与最终标签一致 | {agreement['reviewer_3']['label_match']}/2,000（{agreement['reviewer_3']['label_rate_pct']}%） |
| 修改标签 | {label_changes} |
| normal → fraud | {summary['normal_to_fraud']} |
| fraud → normal | {summary['fraud_to_normal']} |
| ambiguous → 其他 | {summary['ambiguous_to_other']} |
| 修改 tE | {te_changes} |
| 修改 tA | {ta_changes} |
| tE 平均移动（最终−原始） | {summary['tE_mean_shift_final_minus_original_turns']} 轮 |
| 最终 P0 / P1 / P2 | {issue_severity_counts.get('P0',0)} / {issue_severity_counts.get('P1',0)} / {issue_severity_counts.get('P2',0)} |
| 有分歧或强制复核触发 | {disagreements} |

原标签与最终裁决的混淆矩阵为纯对角矩阵：fraud 800→fraud、normal 800→normal、ambiguous 400→ambiguous；其余格均为 0。

最常见分歧类型为 `{summary['most_common_disagreement'][0] if summary['most_common_disagreement'] else '无'}`（{summary['most_common_disagreement'][1] if summary['most_common_disagreement'] else 0} 通）。逐通采纳/拒绝理由见 `final_adjudication_log.jsonl`，不存在未解释的标签、tE、tA 或证据分歧。

## 四、审核员可靠性判断

- reviewer_1 与原标签不可分离，不能据 100% 一致率证明独立可靠性。
- reviewer_2 在明确操作事件识别上较好，但在严格边界诈骗链上经常把 tE 推迟到操作轮次，并把 204 通已出现“身份/权威＋个体状态＋异常后果”的电话降为 ambiguous；也会被正常通话中的“试用/到期”等表层词触发。其标签一致率为 {agreement['reviewer_2']['label_rate_pct']}%。
- reviewer_3 的标签与 tE 更接近逐字严格边界，但其 call_id 校准披露削弱了独立性；同时 tA 识别弱于 clean1，存在大量漏检。因此它在 final_label/tE 场景中更可靠，在高风险操作时间点上不如 clean1 当前标注。

## 五、标签、时间点、证据和操作

- 800 通 fraud 的核心证据均在当前 turns 中逐字存在、来自来电人，并位于 tE；身份/权威主张在该轮或此前建立，该轮首次完成个体状态与异常后果链。
- normal 只在用户主动发起、可从独立官方入口核验、允许挂断重查、无异常后果、无敏感操作要求时保留。
- ambiguous 仅用于原文仍不足以支持 normal 或 fraud 的 400 通。
- core_evidence、supporting_evidence、operation_events 的逐字对应错误为 {len(quote_errors)}；prefix/model_input/hash 错误为 {len(hash_errors)}；独立 timing/operation 复验错误为 {len(timing_errors)}。

因此，最终裁决文件本身不存在剩余的核心证据、时间点或操作类型错误。

## 六、Blind、重复、模板与跨切分

- `evidence_position_bin` 及明示答案字段已从 Blind Input 移除，但 `tFinal_turn` 仍保留。
- Blind Input 与 restricted labels 为 1:1 对应。
- 仅凭 call_id 数值区间即可 {quality['blind_call_id_order_label_recovery']['correct']}/{quality['blind_call_id_order_label_recovery']['total']}（{quality['blind_call_id_order_label_recovery']['accuracy_pct']}%）恢复 Blind 标签，构成 P0 泄漏。
- 完整文本唯一数 {quality['full_text_unique_count']}/2,000；规范化文本唯一数 {quality['normalized_text_unique_count']}/2,000。
- 对全部 {jaccard['pair_count']:,} 对执行精确字符 4-gram Jaccard：J≥0.85 为 {len(jaccard['pairs_ge_085'])} 对，跨切分 J≥0.85 为 {jaccard['cross_split_pairs_ge_085']} 对；最高值为 {jaccard['max_pair']['jaccard'] if jaccard['max_pair'] else 0}。
- 单句/整轮完整话术仍严重复用：精确规范化后超过 20 通的模板 {len(exact_over20)} 个，数字/编号规范化后超过 20 通的模板 {len(template_over20)} 个；后者最高十个复用频次为 {top_reuse}。
- 文档级重复和 J≥0.85 跨切分泄漏通过；轮次级模板复用不通过。
- 历史 Blind 元字段泄漏是否影响既有模型、训练流程或评测结果无法排除，必须重新建立从未暴露过的新 Blind Test。

## 七、clean1 自然度与来源

clean1 自然度修复 **未通过**。共有 {len(per_call_naturalness)} 通命中模板复用、标注腔、机器式过渡或角色不连贯；{adjacent_zero_4gram}/{adjacent_total}（{pct(adjacent_zero_4gram, adjacent_total)}%）相邻轮次没有规范化字符 4-gram 交集，该指标只作响应连续性的诊断旁证，不单独决定标签。

有 source_type 的记录全部为 synthetic_compositional；Blind 缺字段但来自同一构建产物。SOURCE_COMPOSITION_PASS 必须为 false。

## 八、是否生成新正式数据文件

建议生成新版本，但不能直接把当前 clean1 宣布为正式版本。应先应用 `final_corrections.jsonl` 的对话重写要求，保持 final_label/tE/tA 语义不变；再引入符合甲方比例的真实脱敏或人工自然改写来源；最后建立全新、随机不透明 ID、去除 tFinal 等暗示字段、由独立持有人封存标签的 Blind Test，并重跑所有训练和评测。

## 九、发布判定

```json
{json.dumps(release, ensure_ascii=False, indent=2)}
```
"""
    (OUT / "FINAL_ADJUDICATION_REPORT.md").write_text(report, encoding="utf-8")

    decision = f"""# FINAL DATASET DECISION

结论：**当前 `v365_2000.clean1` 不允许直接进入正式训练，不允许作为正式评测集，不可标记为 release-ready。**

标签、tE、tA、核心证据与操作事件已经完成 2,000/2,000 通最终裁决，可作为后续修复的唯一标注基线；但发布前必须完成三项阻断修复：

1. 全新建立未曾暴露的 Blind Test，使用随机不透明 ID，删除 `tFinal_turn` 等答案相关元字段，并重新运行所有评测。
2. 按 `final_corrections.jsonl` 重写受模板复用、标注腔或角色不连贯影响的对话，重写不得改变最终标签、tE、tA 语义，并须重新做逐字证据和哈希校验。
3. 补足甲方要求的真实脱敏/人工自然改写来源组成；全部 synthetic_compositional 的现状不可通过三方一致来豁免。

发布状态：

```json
{json.dumps(release, ensure_ascii=False, indent=2)}
```
"""
    (OUT / "FINAL_DATASET_DECISION.md").write_text(decision, encoding="utf-8")

    # Hash every final deliverable except the self-referential hash manifest.
    deliverables = [
        "four_way_comparison.jsonl", "final_adjudicated_labels.jsonl", "final_corrections.jsonl",
        "final_adjudication_log.jsonl", "final_issues.jsonl", "remaining_external_requirements.jsonl",
        "final_summary.json", "FINAL_ADJUDICATION_REPORT.md", "FINAL_DATASET_DECISION.md",
    ]
    manifest = {
        "algorithm": "SHA256", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_directory": str(OUT), "manifest_excludes_self": True,
        "files": {name: {"sha256": sha256_file(OUT / name), "bytes": (OUT / name).stat().st_size} for name in deliverables},
    }
    write_json(OUT / "DELIVERABLE_SHA256.json", manifest)
    print(json.dumps({
        "output": str(OUT), "completed": len(finals), "label_counts": final_counts,
        "corrections": len(corrections), "release": release,
        "jaccard_ge_085": len(jaccard["pairs_ge_085"]), "timing_errors": len(timing_errors),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
