#!/usr/bin/env python3
"""Blind semantic review for V365_2000.

This program deliberately projects every input row to call_id/split/turns and
never reads answer fields.  It produces and freezes the first-pass review
before any comparison program is allowed to inspect the original labels.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import unicodedata
import zlib
from pathlib import Path


SAFE_KEYS = ("call_id", "split", "turns")
NEGATIONS = (
    "不需要", "无需", "不会", "不得", "不要", "不能", "不要求", "未要求",
    "没有要求", "不提供", "不发送", "不接受", "不操作", "不转账", "不付款",
    "不涉及", "不索取", "不安装", "不下载", "不共享", "不远程", "不提交",
    "不告诉", "不交出", "自行", "自己从", "自己打开", "自己操作", "以正式",
    "以官方", "以您主动", "仅供您独立核对", "只做说明", "只解释", "不能替您",
)

OP_PATTERNS = collections.OrderedDict([
    ("transfer", [
        r"(?:向|给|往|转入|汇入).{0,16}(?:账户|卡|二维码).{0,8}(?:转账|汇款|打款|付款)",
        r"(?:转账|汇款|打款|付款).{0,18}(?:安全账户|核验账户|私人账户|个人账户|二维码)",
        r"(?:先|立即|现在).{0,8}(?:转账|汇款|打款|支付|付款)",
        r"个人垫付", r"扫码(?:支付|付款)",
    ]),
    ("otp", [
        r"(?:把|将|提供|报出|念出|告诉我).{0,12}(?:验证码|动态码|短信码)",
        r"(?:验证码|动态码|短信码).{0,12}(?:告诉|提供|发给|报给|念给)",
    ]),
    ("download", [
        r"(?:下载|安装).{0,18}(?:应用|软件|APP|App|app|插件|安装包|配置文件)",
        r"(?:应用|软件|插件|安装包|配置文件).{0,10}(?:下载|安装)",
    ]),
    ("screen_share", [r"(?:开启|打开|进行|接受|发起|共享).{0,8}屏幕", r"屏幕共享"]),
    ("remote_control", [r"(?:开启|打开|接受|允许|进行).{0,8}远程(?:控制|协助)", r"远程控制"]),
    ("bank_operation", [
        r"(?:登录|打开|进入).{0,10}(?:手机银行|网银|银行App|银行APP)",
        r"(?:手机银行|网银|银行App|银行APP).{0,12}(?:按照|按我|跟着|操作)",
    ]),
    ("unofficial_link", [
        r"(?:点击|打开|进入|访问).{0,12}(?:链接|网址|短链|页面)",
        r"(?:链接|网址|短链).{0,12}(?:发给|发送|点开|打开)",
    ]),
    ("account_or_password", [
        r"(?:提供|提交|告诉|发来|报出|念出|填写).{0,12}(?:银行卡|卡号|账户|密码|支付口令)",
        r"(?:银行卡|卡号|账户|密码|支付口令).{0,12}(?:提供|提交|告诉|发来|报出|念出|填写)",
    ]),
    ("identity_document", [
        r"(?:提供|提交|上传|发送|拍摄).{0,16}(?:身份证|手持身份证|征信报告|证件照)",
        r"(?:身份证|手持身份证|征信报告|证件照).{0,12}(?:提供|提交|上传|发送|拍摄)",
    ]),
    ("cash_gold_giftcard_or_courier", [
        r"(?:购买|买|准备|邮寄|交付).{0,12}(?:黄金|金条|购物卡|礼品卡|卡密|现金)",
        r"(?:黄金|金条|购物卡|礼品卡|卡密|现金).{0,12}(?:购买|买|邮寄|交付)",
    ]),
    ("crypto_asset", [
        r"(?:购买|买入|兑换|转移|提到|转到).{0,16}(?:虚拟币|数字货币|稳定币|USDT|钱包地址|数字资产)",
        r"(?:虚拟币|数字货币|稳定币|USDT|数字资产).{0,12}(?:购买|买入|兑换|转移|提到|转到)",
        r"钱包地址",
    ]),
    ("sim_or_esim_operation", [
        r"(?:办理|开通|激活|删除|注销|转移|补办).{0,10}(?:SIM|sim|eSIM|esim|手机卡)",
        r"(?:SIM|sim|eSIM|esim|手机卡).{0,10}(?:办理|开通|激活|删除|注销|转移|补办)",
    ]),
])

PERSONAL_STATE = re.compile(
    r"(?:系统|后台|内部记录|记录|平台|页面|名单).{0,14}(?:显示|认定|标记|登记|显示您|认为)|"
    r"您(?:已|已经|曾|在|名下|当前|此前).{0,24}(?:开通|办理|领取|申请|进入|存在|关联|占用|涉案|逾期|签署|注册)|"
    r"(?:自动|已经).{0,12}(?:登记|开通|续期|绑定).{0,12}(?:您|名下)|"
    r"(?:尾号|账号|账户|保单|订单|贷款|企业|号码).{0,14}(?:异常|冻结|涉案|逾期|到期|失效|风控)"
)

ABNORMAL = re.compile(
    r"(?:免费期|试用|会员|保障|服务).{0,16}(?:到期|结束|自动续|自动扣)|"
    r"(?:即将|将|会|开始|每月|随后).{0,10}(?:自动扣|扣费|收费|冻结|停机|失效|追责|处罚|影响征信)|"
    r"(?:不|未|逾期).{0,10}(?:处理|完成|关闭|激活|配合).{0,14}(?:扣费|收费|冻结|停机|失效|追责|处罚|影响|违约|承担)|"
    r"(?:冻结|涉案|风控|黑名单|征信|违约费用|处罚|追责|失去资格|占用名额)|"
    r"(?:只能|必须).{0,16}(?:电话专席|当前专席|内部通道|隐藏页面|特殊通道)|"
    r"(?:官方|普通|公开).{0,12}(?:无法|不能|查不到|处理不了)"
)

RELATIONSHIP_SCAM = re.compile(
    r"(?:收益保证|保证每天|稳赚|内幕消息|内部消息|高收益).{0,30}(?:投资|账户|增长|回报|名额)?|"
    r"(?:不要|不能).{0,10}(?:告诉|联系).{0,10}(?:家人|亲友|同事|警方)|"
    r"(?:关系|名额).{0,16}(?:保密|不能外传|不要外传)|"
    r"(?:绕过|跳过).{0,10}(?:财务|法务|审批)|"
    r"(?:负责人|领导|老板).{0,20}(?:个人垫付|私人账户|保密)"
)

NORMAL_STRONG = re.compile(
    r"(?:您|用户).{0,8}(?:此前|之前|主动).{0,8}(?:发起|提交|申请|下单|咨询)|"
    r"(?:可以|允许|建议).{0,12}(?:挂断|结束通话).{0,16}(?:官网|官方|公开|原页面|App|APP|内网|网点|重拨|核验)|"
    r"(?:官网|官方 ?App|官方 ?APP|公开电话|原页面|站内消息|公司内网|正式页面).{0,20}(?:核对|核验|查询|查到|看到|重新呼入)|"
    r"(?:身份和事项|工单|订单|编号).{0,18}(?:独立核对|独立验证|可查|查询)|"
    r"不会因核验而.{0,12}(?:提高费用|失去资格)|"
    r"(?:不得|不需要|不会|不索取).{0,16}(?:个人垫付|私人账户|验证码|密码|共享屏幕|远程控制)"
)

AMBIGUOUS_STRONG = re.compile(
    r"(?:尚未|还未|没有|未).{0,12}(?:断言|确认|给出|涉及).{0,12}(?:申请|业务|账户|交易|信用|状态|额度|个人业务)|"
    r"当前只有.{0,30}没有个人业务结论|"
    r"(?:只|仅).{0,10}(?:介绍|说明|询问).{0,16}(?:一般|规则|主题|是否愿意)|"
    r"(?:无法|不能).{0,14}(?:确认|区分).{0,20}(?:对应|业务|保单|申请|旧咨询|误留号码)|"
    r"没有具体(?:订单|协议|状态|业务结论)"
)

META_PHRASES = (
    "个人业务结论", "尚未断言", "当前信息", "证据不足", "不能判断", "不提前判断",
    "不需要强行下结论", "待核", "独立核验", "重新判断风险", "当前只算", "当前没有足够字段",
    "只停留在事实说明", "当前解释已经偏离", "后果需要", "不代表我", "只作为待核信息",
)
LEAK_TERMS = ("这是诈骗", "诈骗样本", "应标为诈骗", "标签为诈骗", "标成诈骗", "early_fraud", "explicit_fraud")


def dump_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clauses(text: str):
    return [x.strip() for x in re.split(r"[。！？!?；;\n]", text) if x.strip()]


def negated(clause: str, match_start: int) -> bool:
    left = clause[max(0, match_start - 18): match_start + 4]
    return any(n in left or n in clause for n in NEGATIONS)


def operations_in_text(text: str):
    found = []
    for clause in clauses(text):
        for op, patterns in OP_PATTERNS.items():
            for pattern in patterns:
                m = re.search(pattern, clause, flags=re.I)
                if m and not negated(clause, m.start()):
                    found.append(op)
                    break
    return list(dict.fromkeys(found))


def suspicious_early(text: str) -> bool:
    for clause in clauses(text):
        if any(n in clause for n in NEGATIONS) and not RELATIONSHIP_SCAM.search(clause):
            continue
        if RELATIONSHIP_SCAM.search(clause):
            return True
        if ABNORMAL.search(clause) and (PERSONAL_STATE.search(clause) or re.search(r"您|你的|您名下|账户|订单|保单|贷款", clause)):
            return True
    return False


def normalize_dialogue(turns):
    text = "\n".join(str(t.get("text", "")) for t in turns)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[a-z]\d{4,}", "<编号>", text, flags=re.I)
    text = re.sub(r"\d+(?:\.\d+)?(?:元|万元|块|%|％)?", "<数值>", text)
    text = re.sub(r"(?:北京|上海|广州|深圳|重庆|天津|南京|苏州|杭州|武汉|长沙|成都|西安|郑州|青岛|厦门|合肥|福州|济南|昆明|南宁|沈阳|大连|哈尔滨|长春|石家庄|太原|兰州|贵阳|海口|乌鲁木齐)", "<城市>", text)
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def ngrams(text: str, n=4):
    return {text[i:i+n] for i in range(max(0, len(text) - n + 1))}


def minhash_signature(gs, permutations=24):
    prime = 4294967311
    base = [zlib.crc32(g.encode("utf-8")) & 0xFFFFFFFF for g in gs]
    if not base:
        return tuple([0] * permutations)
    sig = []
    for i in range(permutations):
        a = 2654435761 + 2 * i
        b = 2246822519 * (i + 1)
        sig.append(min(((a * x + b) % prime) for x in base))
    return tuple(sig)


def duplicate_analysis(records):
    norm = {r["call_id"]: normalize_dialogue(r["turns"]) for r in records}
    split = {r["call_id"]: r["split"] for r in records}
    ids = list(norm)
    exact_groups = collections.defaultdict(list)
    for cid in ids:
        exact_groups[hashlib.sha256(norm[cid].encode("utf-8")).hexdigest()].append(cid)
    exact_groups = [v for v in exact_groups.values() if len(v) > 1]

    gram_sets = {cid: ngrams(norm[cid]) for cid in ids}
    buckets = collections.defaultdict(list)
    for cid in ids:
        sig = minhash_signature(gram_sets[cid])
        for band in range(6):
            buckets[(band, sig[band*4:(band+1)*4])].append(cid)
    candidates = set()
    for members in buckets.values():
        if 1 < len(members) <= 250:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if abs(len(norm[a]) - len(norm[b])) <= max(30, int(max(len(norm[a]), len(norm[b])) * 0.2)):
                        candidates.add((a, b) if a < b else (b, a))
    pairs = []
    for a, b in sorted(candidates):
        ga, gb = gram_sets[a], gram_sets[b]
        union = len(ga | gb)
        jac = len(ga & gb) / union if union else 1.0
        if jac >= 0.85:
            pairs.append({
                "call_id_a": a, "split_a": split[a], "call_id_b": b, "split_b": split[b],
                "jaccard_4gram": round(jac, 6), "cross_split": split[a] != split[b],
                "normalized_exact": norm[a] == norm[b],
            })
    adjacency = collections.Counter()
    for p in pairs:
        adjacency[p["call_id_a"]] += 1
        adjacency[p["call_id_b"]] += 1
    return norm, exact_groups, pairs, adjacency


def safe_turn_texts(record):
    return [(int(t.get("turn_id", i + 1)), str(t.get("speaker", "")), str(t.get("text", "")))
            for i, t in enumerate(record["turns"])]


def select_normal_basis(caller_turns):
    basis = []
    for tid, text in caller_turns:
        if NORMAL_STRONG.search(text):
            basis.append({"turn_id": tid, "quote": text, "explanation": "存在可独立核验、可挂断重查或安全办理闭环"})
        if len(basis) >= 3:
            break
    return basis or None


def review_one(record, turn_freq, duplicate_degree):
    cid, split, turns = record["call_id"], record["split"], record["turns"]
    issue_codes = []
    severity = "none"
    valid_turns = isinstance(turns, list) and len(turns) > 0
    parsed = safe_turn_texts(record) if valid_turns else []
    ids = [x[0] for x in parsed]
    speakers = [x[1] for x in parsed]
    expected_ids = list(range(1, len(parsed) + 1))
    alternates = all(speakers[i] != speakers[i-1] for i in range(1, len(speakers)))
    if not valid_turns or ids != expected_ids or not alternates:
        issue_codes.append("SCHEMA_ERROR")
        severity = "P0"

    caller = [(tid, text) for tid, spk, text in parsed if spk == "来电人"]
    all_text = "\n".join(text for _, _, text in parsed)
    early_turn = next((tid for tid, text in caller if suspicious_early(text)), None)
    op_events = []
    for tid, text in caller:
        ops = operations_in_text(text)
        if ops:
            op_events.append((tid, ops, text))
    tA = op_events[0][0] if op_events else None
    if early_turn is None and tA is not None:
        early_turn = tA

    normal_hits = sum(1 for _, text in caller if NORMAL_STRONG.search(text))
    ambiguous_hits = sum(1 for _, text in caller if AMBIGUOUS_STRONG.search(text))
    if early_turn is not None:
        label = "fraud"
        decision = "explicit_fraud" if tA is not None else "early_fraud"
        tE = early_turn
        evidence_text = next(text for tid, text in caller if tid == tE)
        evidence = [{
            "turn_id": tE,
            "quote": evidence_text,
            "explanation": "该轮首次形成个体化异常状态/后果链，或首次提出明确高风险操作",
        }]
        normal_basis = None
        ambiguous_missing = None
        confidence = "high" if (suspicious_early(evidence_text) or tA == tE) else "medium"
        rationale = f"第{tE}轮首次形成严格诈骗边界"
        if tA is not None:
            rationale += f"；第{tA}轮首次出现明确高风险操作。"
        else:
            rationale += "，尚无明确高风险操作。"
        tS = caller[0][0] if caller else tE
    elif normal_hits >= 2 or (normal_hits >= 1 and ambiguous_hits == 0):
        label, decision, tE, tA = "normal", "normal", None, None
        evidence = []
        normal_basis = select_normal_basis(caller)
        ambiguous_missing = None
        confidence = "high" if normal_hits >= 2 else "medium"
        rationale = "对话允许从独立官方入口核验或挂断重查，未出现异常后果及敏感操作要求。"
        tS = None
    else:
        label, decision, tE, tA = "ambiguous", "ambiguous", None, None
        evidence = []
        normal_basis = None
        ambiguous_missing = "缺少可确认的个体化业务/账户状态、异常后果及明确高风险操作。"
        confidence = "high" if ambiguous_hits >= 1 else "medium"
        rationale = "当前只有身份或一般业务主题，尚不足以确认正常业务闭环或诈骗链条。"
        tS = caller[0][0] if caller else None

    repeated_word = bool(re.search(r"([\u4e00-\u9fff]{2,4})\1", all_text))
    leak = any(x in all_text for x in LEAK_TERMS)
    meta_count = sum(all_text.count(x) for x in META_PHRASES)
    long_filler = len(parsed) >= 30 and meta_count >= 4
    caller_role_weird = "系统把任务归到网络朋友名下" in all_text or "系统把任务归到单位负责人名下" in all_text
    if repeated_word or long_filler or caller_role_weird:
        issue_codes.append("UNNATURAL_LANGUAGE")
        severity = "P1" if long_filler or caller_role_weird else ("P2" if severity == "none" else severity)
    if caller_role_weird:
        issue_codes.append("ROLE_INCOHERENT")
        severity = "P1"
    if leak:
        issue_codes.append("LABEL_LEAKAGE")
        severity = "P0"

    common_turns = sum(1 for _, _, text in parsed if turn_freq[text.strip()] > 20)
    common_ratio = common_turns / max(1, len(parsed))
    dup_degree = duplicate_degree.get(cid, 0)
    if dup_degree > 0 or common_ratio >= 0.30:
        issue_codes.append("TEMPLATE_OVERUSE")
        if severity == "none":
            severity = "P1"

    if not valid_turns:
        naturalness, logic, template = 0, 0, 4
    else:
        naturalness = 4
        if meta_count >= 2 or common_ratio >= 0.2:
            naturalness = 3
        if long_filler or repeated_word or caller_role_weird or meta_count >= 6:
            naturalness = 2
        logic = 4 if alternates else 1
        if caller_role_weird:
            logic = min(logic, 2)
        template = 4 if dup_degree >= 2 or common_ratio >= 0.6 else 3 if dup_degree or common_ratio >= 0.3 else 2 if common_ratio >= 0.15 else 1

    if naturalness <= 1:
        action = "remove"
    elif naturalness == 2 or template >= 3 or "ROLE_INCOHERENT" in issue_codes:
        action = "rewrite_dialogue"
    elif confidence == "low":
        action = "send_to_adjudication"
    else:
        action = "keep"

    return {
        "call_id": cid,
        "split": split,
        "reviewer_final_label": label,
        "reviewer_decision": decision,
        "reviewer_tS_turn": tS,
        "reviewer_tE_turn": tE,
        "reviewer_tA_turn": tA,
        "reviewer_core_evidence": evidence,
        "normal_verification_basis": normal_basis,
        "ambiguous_missing_information": ambiguous_missing,
        "reviewer_confidence": confidence,
        "naturalness_score": naturalness,
        "logic_consistency_score": logic,
        "template_risk_score": template,
        "detected_operation_events": [{"turn_id": tid, "operation_types": ops} for tid, ops, _ in op_events],
        "issue_codes": list(dict.fromkeys(issue_codes)),
        "severity": severity,
        "recommended_action": action,
        "review_rationale": rationale[:150],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    base, out = Path(args.dataset_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    inputs = [
        base / "data/v365_calls_train.jsonl",
        base / "data/v365_calls_dev.jsonl",
        base / "data/v365_calls_stress.jsonl",
        base / "data/v365_calls_blind_input.jsonl",
    ]
    records = []
    parse_errors = []
    for path in inputs:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    raw = json.loads(line)
                    # Strict blind projection. No other raw key is ever accessed.
                    records.append({key: raw.get(key) for key in SAFE_KEYS})
                except Exception as exc:
                    parse_errors.append({"file": str(path), "line": line_no, "error": str(exc)})
    if parse_errors:
        dump_jsonl(out / "blind_parse_errors.jsonl", parse_errors)
        raise SystemExit(f"Input parse errors: {len(parse_errors)}")
    if len(records) != 2000:
        raise SystemExit(f"Expected 2000 records, got {len(records)}")
    ids = [r["call_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("call_id is not globally unique")

    turn_freq = collections.Counter(
        str(t.get("text", "")).strip()
        for r in records for t in (r["turns"] or []) if isinstance(t, dict)
    )
    _, exact_groups, pairs, duplicate_degree = duplicate_analysis(records)
    dump_jsonl(out / "duplicate_candidates_blind.jsonl", pairs)
    reviews = [review_one(r, turn_freq, duplicate_degree) for r in records]
    result_path = out / "blind_review_labels.jsonl"
    dump_jsonl(result_path, reviews)

    manifest = {
        "status": "FROZEN_BEFORE_LABEL_REVEAL",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "record_count": len(reviews),
        "allowed_source_fields": list(SAFE_KEYS),
        "source_files": [str(p) for p in inputs],
        "blind_review_labels_sha256": sha256_file(result_path),
        "duplicate_candidates_sha256": sha256_file(out / "duplicate_candidates_blind.jsonl"),
        "normalized_exact_duplicate_group_count": len(exact_groups),
        "near_duplicate_pair_count": len(pairs),
        "cross_split_near_duplicate_pair_count": sum(1 for p in pairs if p["cross_split"]),
        "reviewer_label_counts": dict(collections.Counter(r["reviewer_final_label"] for r in reviews)),
        "reviewer_decision_counts": dict(collections.Counter(r["reviewer_decision"] for r in reviews)),
        "reviewer_naturalness_counts": dict(collections.Counter(str(r["naturalness_score"]) for r in reviews)),
    }
    manifest_path = out / "BLIND_FREEZE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
