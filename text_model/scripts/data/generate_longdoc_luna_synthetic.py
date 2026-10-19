#!/usr/bin/env python3
"""使用本地 Luna 从零生成长文本电话对话候选。

与 generate_longdoc_luna.py 的区别：

* 不读取、拼接或改写任何已有通话正文；
* 只读取官方 39 子类定义，用于分配标签和约束生成主题；
* 每条样本由 Luna 一次生成完整的、角色交替的电话对话；
* ``text``/``audio_text`` 保留“来电人/接电人”，WordPiece 统计使用无标识文本；
* 所有结果仍是 ``pending`` 候选，必须经过人工审核后才能进入训练集。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Iterable, Mapping, Sequence

try:
    from text_model.data.role_markers import role_contract, role_turns, strip_role_markers
    from text_model.scripts.data.generate_longdoc_luna import (
        BertWordPieceCounter,
        RateLimiter,
        call_luna,
        sha256_text,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:  # 允许直接以脚本路径运行
    ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
    from text_model.data.role_markers import role_contract, role_turns, strip_role_markers
    from text_model.scripts.data.generate_longdoc_luna import (
        BertWordPieceCounter,
        RateLimiter,
        call_luna,
        sha256_text,
        write_json,
        write_jsonl,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY = ROOT / "data/processed/v329_qm_audited_backfill6500_20260725/label_subcategory_rubrics_official_v1.json"
DEFAULT_VOCAB = ROOT / (
    "runs/r_260727_042528_V341D1_Policy6500_MegatronBERT_DataCausal_DevOnly_s202_"
    "auto_a1/subruns/seed-202/best_model/vocab.txt"
)
DEFAULT_OUTPUT = ROOT / "data/processed/longdoc_v1_20260806_luna_synthetic"
DEFAULT_ENDPOINT = "http://127.0.0.1:4000/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-luna"
DATASET_VERSION = "longdoc_v1_20260806_luna_synthetic"
PROMPT_VERSION = "synthetic_full_dialogue_v1"

BUCKETS: dict[str, tuple[int, int, int]] = {
    "q1_320_480": (320, 480, 400),
    "q2_481_720": (481, 720, 600),
    "q3_721_960": (721, 960, 840),
    "q4_961_1280": (961, 1280, 1080),
}
BUCKET_FRACTIONS = {
    "q1_320_480": 0.35,
    "q2_481_720": 0.35,
    "q3_721_960": 0.20,
    "q4_961_1280": 0.10,
}
EVIDENCE_BINS = ("0_25", "25_50", "50_75", "75_100")
NORMAL_TYPES = (
    "ordinary_customer_service",
    "bank_or_payment_official_process",
    "order_refund_or_service_resolution",
    "anti_fraud_education_or_quote",
    "user_initiated_security_check",
    "long_idle_and_business_confirmation",
)

# 这些词只用于最低限度的结构校验，不作为模型输入之外的标签来源。
RISK_TERM_RE = re.compile(
    r"转账|汇款|付款|缴费|垫付|保证金|手续费|验资|充值|入金|验证码|银行卡|"
    r"身份证|链接|下载|扫码|二维码|添加微信|私信|远程|屏幕共享|收款码|安全账户|"
    r"提现|账户|密码|授权码"
)
ASSERTED_ACTION_RE = re.compile(
    r"(?:请|需要|必须|先|马上|立即|赶紧|现在|点击|提供|提交|转到|发到|缴纳|支付|"
    r"告诉|输入|下载|打开).{0,20}(?:转账|汇款|付款|缴费|垫付|保证金|手续费|验资|"
    r"充值|入金|验证码|银行卡|身份证|链接|下载|扫码|二维码|添加微信|私信|远程|"
    r"屏幕共享|收款码|安全账户|提现|密码|授权码)"
)
NEGATED_RISK_RE = re.compile(
    r"(?:不要|无需|不需要|不会|不能|切勿|请勿|拒绝|不接受|不向|没有|未曾|"
    r"从不).{0,24}(?:转账|汇款|付款|缴费|垫付|保证金|手续费|验资|充值|入金|"
    r"验证码|银行卡|身份证|链接|下载|扫码|二维码|添加微信|私信|远程|屏幕共享|"
    r"收款码|安全账户|提现|密码|授权码)"
)
NORMAL_UNSAFE_CALLER_RE = re.compile(
    r"(?:向(?:个人|指定|陌生|安全)账户转账|转到(?:个人|指定|陌生|安全)账户|"
    r"扫(?:码|描)收款码|告诉(?:我|对方)验证码|提供完整银行卡|"
    r"下载(?:陌生|不明|非官方)(?:软件|应用|App)|点击(?:陌生|不明|外部|私聊|聊天)链接|"
    r"添加微信|转到私下账户|屏幕共享|远程控制|开启远程|提交银行卡密码)"
)
ROLE_PREFIX_RE = re.compile(r"(?:来电人|接电人|来电方|接听方|caller|callee)\s*[:：]", re.I)
PHONE_OR_URL_RE = re.compile(r"(?:https?://|www\.|\b1[3-9]\d{9}\b|\b\d{3,4}-\d{7,8}\b)")


def stable_int(value: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:12], 16)


def load_taxonomy(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parents = payload.get("parents") or {}
    parent_map: dict[str, dict[str, Any]] = {}
    child_map: dict[str, dict[str, Any]] = {}
    for parent_code, parent in sorted(parents.items(), key=lambda item: int(item[0])):
        parent_map[str(parent_code)] = {
            "label_13": int(parent_code),
            "name": str(parent.get("name") or ""),
        }
        for child in parent.get("subcategories") or []:
            code = str(child["code"])
            child_map[code] = {
                "label_13": int(parent_code),
                "parent_name": str(parent.get("name") or ""),
                "code": code,
                "name": str(child.get("name") or ""),
                "rubric": str(child.get("rubric") or ""),
            }
    if len(parent_map) != 13 or len(child_map) != 39:
        raise ValueError(f"官方标签定义数量异常：parents={len(parent_map)}, children={len(child_map)}")
    return parent_map, child_map


def allocate_integer(total: int, keys: Sequence[str], *, salt: str) -> dict[str, int]:
    """将总量尽量平均分到 keys，并用稳定哈希决定余数归属。"""

    if not keys:
        return {}
    base, remainder = divmod(total, len(keys))
    ordered = sorted(keys, key=lambda key: stable_int(key, salt))
    result = {key: base for key in keys}
    for key in ordered[:remainder]:
        result[key] += 1
    return result


def make_fraud_quota(total: int, child_map: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    fraud_children = [code for code, item in child_map.items() if int(item["label_13"]) != 0]
    by_parent: dict[int, list[str]] = defaultdict(list)
    for code in fraud_children:
        by_parent[int(child_map[code]["label_13"])].append(code)
    parent_quota = allocate_integer(total, [str(parent) for parent in sorted(by_parent)], salt="parent")
    result: dict[str, int] = {}
    for parent, codes in sorted(by_parent.items()):
        result.update(allocate_integer(parent_quota[str(parent)], sorted(codes), salt=f"child-{parent}"))
    return result


def make_normal_quota(total: int, child_map: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    normal = [code for code, item in child_map.items() if int(item["label_13"]) == 0]
    return allocate_integer(total, sorted(normal), salt="normal-child")


def choose_bucket(key: str) -> str:
    value = stable_int(key, "length") % 10000 / 10000.0
    cursor = 0.0
    for bucket, fraction in BUCKET_FRACTIONS.items():
        cursor += fraction
        if value < cursor:
            return bucket
    return next(reversed(BUCKET_FRACTIONS))


def choose_position(key: str) -> str:
    return EVIDENCE_BINS[stable_int(key, "evidence") % len(EVIDENCE_BINS)]


def choose_normal_type(key: str) -> str:
    return NORMAL_TYPES[stable_int(key, "normal-type") % len(NORMAL_TYPES)]


def target_info(bucket: str) -> tuple[int, int, int]:
    return BUCKETS[bucket]


def length_guidance(bucket: str) -> str:
    """把长度目标转换为模型更容易遵守的轮次数和单轮长度约束。"""

    guidance = {
        "q1_320_480": "严格使用 12–14 轮；每轮只说一个要点，通常 18–28 个汉字，不要写成长段解释。",
        "q2_481_720": "使用 16–18 轮；每轮通常 24–34 个汉字，保留必要追问，不要重复同一提醒。",
        "q3_721_960": "使用 20–24 轮；每轮通常 30–40 个汉字，让业务过程有连续推进。",
        "q4_961_1280": "使用 24–28 轮；每轮通常 36–46 个汉字，增加具体核验、追问和收束，但不重复口号。",
    }
    return guidance[bucket]


def make_tasks(
    *,
    output_dir: Path,
    taxonomy_path: Path,
    train_fraud: int,
    train_normal: int,
    dev_fraud: int,
    dev_normal: int,
    stress_fraud: int,
    stress_normal: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_map, child_map = load_taxonomy(taxonomy_path)
    split_counts = {
        "train": (train_fraud, train_normal),
        "dev": (dev_fraud, dev_normal),
        "stress": (stress_fraud, stress_normal),
    }
    tasks: list[dict[str, Any]] = []
    quota_manifest: dict[str, Any] = {}
    for split, (fraud_total, normal_total) in split_counts.items():
        fraud_quota = make_fraud_quota(fraud_total, child_map)
        normal_quota = make_normal_quota(normal_total, child_map)
        quota_manifest[split] = {
            "fraud_by_subcategory": fraud_quota,
            "normal_by_subcategory": normal_quota,
        }
        serial = 0
        for code, count in sorted({**fraud_quota, **normal_quota}.items()):
            item = child_map[code]
            label_13 = int(item["label_13"])
            for variant_index in range(count):
                task_key = f"{split}|{code}|{variant_index}"
                bucket = choose_bucket(task_key)
                low, high, mid = target_info(bucket)
                position = choose_position(task_key)
                normal_type = choose_normal_type(task_key) if label_13 == 0 else ""
                task = {
                    "task_id": f"SYN_LONGDOC_{split}_{serial:05d}",
                    "sample_id": f"SYN_LONGDOC_V1_{split}_{serial:05d}",
                    "split": split,
                    "serial": serial,
                    "variant_index": variant_index,
                    "synthetic_from_scratch": True,
                    "source_lineage": {
                        "source_type": "synthetic_from_scratch",
                        "source_sample_id": None,
                        "source_text_used": False,
                        "reference_distribution_only": True,
                    },
                    "label_13": label_13,
                    "canonical_label_13": label_13,
                    "subcategory_code": code,
                    "canonical_subcategory_code": code,
                    "parent_name": item["parent_name"],
                    "subcategory_name": item["name"],
                    "subcategory_rubric": item["rubric"],
                    "generation_kind": "normal" if label_13 == 0 else "fraud",
                    "normal_case_type": normal_type,
                    "evidence_position_bin": position,
                    "length_bucket": bucket,
                    "target_wordpiece_min": low,
                    "target_wordpiece_max": high,
                    "target_wordpiece_mid": mid,
                    "review_status": "pending",
                    "generation_provider": "local_luna_4000",
                    "generation_model": DEFAULT_MODEL,
                    "thinking_effort": "medium",
                    "prompt_version": PROMPT_VERSION,
                }
                tasks.append(task)
                serial += 1
    manifest = {
        "schema_version": "longdoc_luna_synthetic_generation_manifest.v1",
        "dataset_version": DATASET_VERSION,
        "generation_mode": "from_scratch_full_dialogue",
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "source_texts_read": False,
        "test_b_read": False,
        "oof_read": False,
        "client_example_read": False,
        "requested_counts": {
            "train_fraud": train_fraud,
            "train_normal": train_normal,
            "dev_fraud": dev_fraud,
            "dev_normal": dev_normal,
            "stress_fraud": stress_fraud,
            "stress_normal": stress_normal,
        },
        "task_counts": dict(Counter(task["split"] for task in tasks)),
        "task_label_counts": {
            split: dict(Counter("normal" if task["label_13"] == 0 else "fraud" for task in tasks if task["split"] == split))
            for split in split_counts
        },
        "quota_manifest": quota_manifest,
        "length_policy": {
            "wordpiece_buckets": BUCKETS,
            "bucket_proportions": BUCKET_FRACTIONS,
            "count_on": "model_input_text_after_role_marker_removal",
            "global_range": [320, 1280],
        },
        "role_contract": {
            "marked_fields": ["text", "audio_text", "text_with_role_markers"],
            "model_field": "model_input_text",
            "token_policy": "role_markers_excluded_from_model_input_and_wordpiece_counts",
            "markers": ["来电人", "接电人"],
        },
        "generation": {
            "endpoint": DEFAULT_ENDPOINT,
            "model": DEFAULT_MODEL,
            "thinking_effort": "medium",
            "rate_limit_rpm": 1500,
            "worker_cap": 96,
            "max_retries": 3,
            "prompt_version": PROMPT_VERSION,
        },
    }
    write_jsonl(output_dir / "generation_tasks.jsonl", tasks)
    write_json(output_dir / "generation_manifest.json", manifest)
    return tasks, manifest


def prompt_for(task: Mapping[str, Any], feedback: str = "") -> str:
    low = int(task["target_wordpiece_min"])
    high = int(task["target_wordpiece_max"])
    position = task["evidence_position_bin"]
    turn_guidance = length_guidance(str(task["length_bucket"]))
    if int(task["label_13"]) != 0:
        position_rule = {
            "0_25": "在前四分之一内出现第一个明确风险动作，后面继续自然展开。",
            "25_50": "前段先进行较充分的身份/业务铺垫，在四分之一至一半位置出现第一个明确风险动作。",
            "50_75": "前半段保持真实业务铺垫和追问，在中后段才出现第一个明确风险动作。",
            "75_100": "前面大部分内容都应是自然铺垫、核验和犹豫，直到最后四分之一才出现第一个明确风险动作。",
        }[position]
        return f"""你是中文电话反诈分类数据生成器。现在要从零创造一通全新的电话对话，不能引用、改写、拼接任何已有样本。只输出一个 JSON 对象，不要输出解释、Markdown 或代码围栏。

目标标签：13类父类 {task['label_13']}「{task['parent_name']}」；39子类 {task['subcategory_code']}「{task['subcategory_name']}」。
子类判定要求：{task['subcategory_rubric']}
场景覆盖：{position_rule}
完整对话去掉角色标识后的 WordPiece 长度必须落在 {low}–{high}，目标约 {task['target_wordpiece_mid']}。

请生成一通完整、连续、自然的来电对话。长度控制：{turn_guidance} 对话要有：
1. 来电人自然开场和身份/事项铺垫；
2. 接电人有真实的疑问、核对、犹豫或反应；
3. 逐步发展到该子类唯一主导的诈骗机制；
4. 至少一个明确的风险动作或资金/敏感信息推进，使标签证据完整；
5. 合理收束，可以是受害人准备操作、提出怀疑、拒绝或挂断。

硬性要求：
- turns 中 speaker 只能是 caller 或 callee，必须从 caller 开始，严格交替，不能连续同一角色；
- 每个 text 只写口语内容，不要写角色名、冒号、旁白、括号、舞台说明或审核结论；
- 不能加入真实姓名、真实手机号、网址、银行卡号、验证码数字、二维码内容或可识别个人信息；
- 不要把其他诈骗父类写成主线；子类「{task['subcategory_name']}」必须是唯一主要机制；
- 不要用重复的“提高警惕”宣传口号填充长度；每一轮都应推动通话；
- 不要把两通电话拼在一起，不要重新开始第二次问候；
- 可以使用泛化的“指定账户、官方页面、验证码”等词，但不写真实可执行信息。

输出格式：
{{"turns":[{{"speaker":"caller","text":"..."}},{{"speaker":"callee","text":"..."}}],"risk_evidence":["对话中已经出现的关键风险动作短语"],"scenario_summary":"不超过40字","generation_notes":"不超过50字"}}

上次校验反馈：{feedback or '无'}"""
    return f"""你是中文电话对话分类数据生成器。现在要从零创造一通全新的、标签为正常的长电话对话，不能引用、改写、拼接任何已有样本。只输出一个 JSON 对象，不要输出解释、Markdown 或代码围栏。

目标标签：13类父类 0「正常通话或非诈骗」；39子类 00_0000「正常通话或非诈骗」。
本条正常场景：{task['normal_case_type']}。
完整对话去掉角色标识后的 WordPiece 长度必须落在 {low}–{high}，目标约 {task['target_wordpiece_mid']}。

请生成一通完整、连续、自然的真实业务电话。长度控制：{turn_guidance} 对话应围绕一个具体事项展开，有来回确认、用户问题、处理结果和自然收尾。可以出现付款、退款、银行卡、验证码、链接、下载等词，但必须处于合法官方流程、用户主动核验、过去被骗经历的引用或反诈提醒语境；来电人不能推进陌生账户转账、索取验证码、远程控制、私下加联系方式或非官方链接操作。

硬性要求：
- turns 中 speaker 只能是 caller 或 callee，必须从 caller 开始，严格交替，不能连续同一角色；
- 每个 text 只写口语内容，不要写角色名、冒号、旁白、括号、舞台说明或审核结论；
- 不能加入真实姓名、真实手机号、网址、银行卡号、验证码数字、二维码内容或可识别个人信息；
- 不要把两个不同电话拼在一起，不要在中途重新开始问候；
- 不要机械堆叠“提高警惕”等口号，要有具体业务进展和自然收尾；
- 如果涉及高风险词，必须明确表现为官方渠道、自主核验、拒绝陌生要求或历史引用，不能形成诈骗闭环。

输出格式：
{{"turns":[{{"speaker":"caller","text":"..."}},{{"speaker":"callee","text":"..."}}],"risk_evidence":[],"scenario_summary":"不超过40字","generation_notes":"不超过50字"}}

上次校验反馈：{feedback or '无'}"""


def normalize_speaker(value: Any) -> str:
    value = str(value or "").strip().lower()
    return {
        "caller": "caller",
        "来电人": "caller",
        "来电方": "caller",
        "甲": "caller",
        "callee": "callee",
        "接电人": "callee",
        "接听方": "callee",
        "接听人": "callee",
        "乙": "callee",
    }.get(value, "")


def build_text(parsed: Mapping[str, Any]) -> tuple[str, list[str]]:
    raw_turns = parsed.get("turns")
    if not isinstance(raw_turns, list):
        raise ValueError("turns_missing_or_not_list")
    turns: list[tuple[str, str]] = []
    for raw in raw_turns:
        if not isinstance(raw, Mapping):
            raise ValueError("turn_not_object")
        speaker = normalize_speaker(raw.get("speaker"))
        content = str(raw.get("text") or "").strip()
        content = re.sub(r"\s+", " ", content)
        if not speaker or not content:
            raise ValueError("turn_speaker_or_text_invalid")
        if ROLE_PREFIX_RE.search(content):
            raise ValueError("role_marker_inside_turn_text")
        if "```" in content or re.search(
            r"(?:模型自述|审核结论|作为(?:语言)?模型|作为(?:人工智能|AI)|生成说明|以上要求)",
            content,
            re.I,
        ):
            raise ValueError("meta_text_in_turn")
        turns.append((speaker, content))
    if len(turns) < 8 or len(turns) > 30:
        raise ValueError(f"turn_count_invalid:{len(turns)}")
    if turns[0][0] != "caller":
        raise ValueError("must_start_with_caller")
    if any(left[0] == right[0] for left, right in zip(turns, turns[1:])):
        raise ValueError("speaker_not_alternating")
    text = "\n".join(
        f"{'来电人' if speaker == 'caller' else '接电人'}：{content}"
        for speaker, content in turns
    )
    evidence = parsed.get("risk_evidence")
    if not isinstance(evidence, list):
        evidence = []
    return text, [str(item).strip() for item in evidence if str(item).strip()]


def target_terms(code: str) -> tuple[str, ...]:
    terms: dict[str, tuple[str, ...]] = {
        "01_0101": ("电商", "订单", "退款", "会员", "平台", "售后"),
        "01_0102": ("快递", "物流", "包裹", "运单", "派送", "理赔"),
        "01_0103": ("平台", "商户", "结算", "账户", "订单"),
        "02_0201": ("公安", "警察", "法院", "检察院", "涉案", "冻结"),
        "02_0202": ("社保", "医保", "税务", "海关", "监管", "单位"),
        "03_0301": ("刷单", "任务", "点赞", "返利", "佣金", "垫付", "补单"),
        "04_0401": ("信用卡", "办卡", "下卡", "征信", "开卡"),
        "04_0402": ("提额", "额度", "套现", "解冻"),
        "04_0403": ("贷款", "放款", "审批", "利息", "保证金"),
        "04_0404": ("贷款", "信用卡", "授信", "借款", "费用"),
        "05_0501": ("部队", "军警", "消防", "采购", "订货", "履约"),
        "06_0601": ("投资", "理财", "股票", "收益", "带单", "入金"),
        "06_0602": ("博彩", "投注", "下注", "盘口", "竞猜", "提现"),
        "06_0603": ("恋爱", "感情", "信任", "见面", "借钱", "周转"),
        "07_0701": ("购物", "商品", "低价", "订金", "尾款", "发货"),
        "07_0702": ("维修", "家政", "租赁", "服务", "上门", "费用"),
        "07_0703": ("商品", "服务", "订单", "交易", "付款"),
        "08_0801": ("外国军人", "境外", "包裹", "清关", "保险", "津贴"),
        "08_0802": ("恋爱", "结婚", "对象", "感情", "见面", "借款"),
        "08_0803": ("网友", "交友", "朋友", "借钱", "周转", "费用"),
        "08_0804": ("网友", "认识", "关系", "借款", "费用", "转账"),
        "09_0901": ("校园贷", "学生", "校园", "贷款记录", "注销"),
        "09_0902": ("征信", "逾期", "不良记录", "信用报告", "修复"),
        "09_0903": ("征信", "信用", "记录", "修复", "注销"),
        "10_1001": ("领导", "老板", "上级", "单位", "紧急", "转账"),
        "10_1002": ("亲友", "同事", "同学", "熟人", "借款", "周转"),
        "10_1003": ("明星", "主播", "公众人物", "粉丝", "活动", "费用"),
        "10_1004": ("身份", "熟人", "冒充", "借款", "周转", "转账"),
        "11_1101": ("游戏币", "点卡", "充值", "代充", "虚拟道具"),
        "11_1102": ("游戏账号", "装备", "皮肤", "交易", "过户"),
        "11_1103": ("游戏", "代练", "道具", "账号", "交易"),
        "12_1201": ("中奖", "领奖", "奖品", "税费", "手续费"),
        "12_1202": ("招聘", "求职", "兼职", "入职", "培训", "押金"),
        "12_1203": ("充值", "红包", "返利", "福利", "活动"),
        "12_1204": ("机票", "航班", "退票", "改签", "航空"),
        "12_1205": ("截图", "凭证", "图片", "回执", "PS"),
        "12_1206": ("慈善", "捐款", "求子", "资助", "善款"),
        "12_1207": ("活动", "补贴", "资格", "费用", "账户"),
    }
    return terms.get(code, ())


def validate_candidate(
    task: Mapping[str, Any],
    parsed: Mapping[str, Any],
    counter: BertWordPieceCounter,
    reserved_hashes: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        text, risk_evidence = build_text(parsed)
    except ValueError as exc:
        return None, [str(exc)]
    model_input = strip_role_markers(text)
    actual_wp = counter.count(text)
    if not 320 <= actual_wp <= 1280:
        errors.append(f"wordpiece_out_of_global_range:{actual_wp} not in 320-1280")
    if PHONE_OR_URL_RE.search(model_input):
        errors.append("real_phone_or_url_like_text")
    if len(role_turns(text)) != len(parsed.get("turns") or []):
        errors.append("role_turn_count_mismatch")
    if int(task["label_13"]) != 0:
        caller_text = "\n".join(
            turn["text"]
            for turn in role_turns(text)
            if turn.get("canonical_role") == "caller"
        )
        safe_context = NEGATED_RISK_RE.sub("", caller_text)
        terms = target_terms(str(task["subcategory_code"]))
        if terms and sum(term in model_input for term in terms) < 2:
            errors.append("target_subcategory_evidence_too_weak")
        if not RISK_TERM_RE.search(safe_context):
            errors.append("fraud_caller_risk_term_missing")
        if not risk_evidence:
            errors.append("risk_evidence_missing")
    else:
        caller_text = "\n".join(
            turn["text"]
            for turn in role_turns(text)
            if turn.get("canonical_role") == "caller"
        )
        safe_caller = NEGATED_RISK_RE.sub("", caller_text)
        # 只拦截来电方直接推动的高危闭环；官方 App 内的自主操作不应被误杀。
        if NORMAL_UNSAFE_CALLER_RE.search(safe_caller):
            errors.append("normal_caller_pushes_unsafe_action")
        if parsed.get("risk_evidence") not in (None, [], ""):
            # 正常样本允许出现风险词，但不应把风险动作标成自身证据。
            evidence = parsed.get("risk_evidence")
            if isinstance(evidence, list) and any(str(item).strip() for item in evidence):
                errors.append("normal_risk_evidence_not_empty")
    normalized_hash = sha256_text(re.sub(r"\s+", "", model_input).lower())
    if normalized_hash in reserved_hashes:
        errors.append("duplicate_or_near_duplicate")
    if errors:
        return None, errors
    contract = role_contract(text)
    row = {
        **dict(task),
        **contract,
        "text": text,
        "text_hash": sha256_text(text),
        "text_sha256": sha256_text(text),
        "model_input_wordpiece_length": actual_wp,
        "actual_length_bucket": next(
            bucket for bucket, (bucket_low, bucket_high, _) in BUCKETS.items()
            if bucket_low <= actual_wp <= bucket_high
        ),
        "model_input_char_length": len(model_input),
        "raw_char_length_with_role_markers": len(text),
        "label_preserved": True,
        "role_consistency": True,
        "human_review_status": "pending",
        "generation_result": "READY",
        "normalized_text_hash": normalized_hash,
        "risk_evidence": risk_evidence,
        "scenario_summary": str(parsed.get("scenario_summary") or "").strip()[:100],
        "generation_notes": str(parsed.get("generation_notes") or "").strip()[:120],
    }
    return row, []


def generate_one(
    task: Mapping[str, Any],
    *,
    endpoint: str,
    model: str,
    counter: BertWordPieceCounter,
    limiter: RateLimiter,
    reserved_hashes: set[str],
    retries: int,
    thinking_effort: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    feedback = ""
    for attempt in range(1, retries + 2):
        try:
            parsed = call_luna(
                endpoint,
                model,
                prompt_for(task, feedback),
                limiter=limiter,
                thinking_effort=thinking_effort,
            )
            row, errors = validate_candidate(task, parsed, counter, reserved_hashes)
            attempts.append({"attempt": attempt, "errors": errors})
            if row is not None:
                row["luna_attempts"] = attempts
                return row
            feedback = "；".join(errors)
        except Exception as exc:  # noqa: BLE001 - 单条失败不影响其他任务
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "error": error})
            feedback = error
    return {
        **dict(task),
        "generation_result": "FAILED",
        "human_review_status": "not_generated",
        "luna_attempts": attempts,
        "failure_reason": feedback,
    }


def read_existing_hashes(output_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for path in (output_dir / "accepted_candidates.jsonl", output_dir / "pilot_candidates.jsonl"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if value.get("normalized_text_hash"):
                        hashes.add(str(value["normalized_text_hash"]))
    return hashes


def choose_tasks(tasks: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(tasks):
        return list(tasks)
    # pilot 同时覆盖正常/诈骗、父类、长度桶和证据位置，并保持两类数量接近。
    def order(pool: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            pool,
            key=lambda task: stable_int(
                f"{task['split']}|{task['canonical_subcategory_code']}|{task['length_bucket']}|{task['evidence_position_bin']}|{task['serial']}",
                "pilot",
            ),
        )

    def select(pool: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        ordered = order(pool)
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for task in ordered:
            group = f"{task['canonical_subcategory_code']}|{task['length_bucket']}|{task['evidence_position_bin']}"
            if group not in seen:
                selected.append(task)
                seen.add(group)
            if len(selected) >= count:
                return selected
        selected_ids = {task["task_id"] for task in selected}
        selected.extend(task for task in ordered if task["task_id"] not in selected_ids)
        return selected[:count]

    fraud = [task for task in tasks if int(task["label_13"]) != 0]
    normal = [task for task in tasks if int(task["label_13"]) == 0]
    fraud_count = min(len(fraud), (limit + 1) // 2)
    normal_count = min(len(normal), limit - fraud_count)
    if fraud_count + normal_count < limit:
        extra_pool = [task for task in tasks if task not in select(fraud, fraud_count) and task not in select(normal, normal_count)]
        selected = select(fraud, fraud_count) + select(normal, normal_count) + order(extra_pool)[: limit - fraud_count - normal_count]
    else:
        selected = select(fraud, fraud_count) + select(normal, normal_count)
    return sorted(selected[:limit], key=lambda task: task["task_id"])


def run_generation(
    tasks: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    endpoint: str,
    model: str,
    vocab: Path,
    workers: int,
    rpm: int,
    retries: int,
    limit: int,
    thinking_effort: str,
) -> dict[str, Any]:
    selected = choose_tasks(tasks, limit)
    counter = BertWordPieceCounter(vocab)
    limiter = RateLimiter(rpm / 60.0)
    reserved_hashes = read_existing_hashes(output_dir)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max(1, workers), 96)) as executor:
        futures = {
            executor.submit(
                generate_one,
                task,
                endpoint=endpoint,
                model=model,
                counter=counter,
                limiter=limiter,
                reserved_hashes=reserved_hashes,
                retries=retries,
                thinking_effort=thinking_effort,
            ): task
            for task in selected
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.get("generation_result") == "READY":
                reserved_hashes.add(str(result["normalized_text_hash"]))
            if index % 10 == 0 or index == len(selected):
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "total": len(selected),
                            "ready": sum(item.get("generation_result") == "READY" for item in results),
                            "failed": sum(item.get("generation_result") == "FAILED" for item in results),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    results.sort(key=lambda row: row["task_id"])
    is_pilot = limit > 0 and limit < len(tasks)
    result_path = output_dir / ("pilot_results.jsonl" if is_pilot else "all_results.jsonl")
    ready_path = output_dir / ("pilot_candidates.jsonl" if is_pilot else "accepted_candidates.jsonl")
    failed_path = output_dir / ("pilot_failures.jsonl" if is_pilot else "failed_tasks.jsonl")
    write_jsonl(result_path, results)
    write_jsonl(ready_path, [row for row in results if row.get("generation_result") == "READY"])
    write_jsonl(failed_path, [row for row in results if row.get("generation_result") == "FAILED"])
    summary = {
        "dataset_version": DATASET_VERSION,
        "model": model,
        "thinking_effort": thinking_effort,
        "endpoint": endpoint,
        "generation_mode": "from_scratch_full_dialogue",
        "requested": len(selected),
        "ready": sum(row.get("generation_result") == "READY" for row in results),
        "failed": sum(row.get("generation_result") == "FAILED" for row in results),
        "label_counts": dict(Counter("normal" if row.get("label_13") == 0 else "fraud" for row in results)),
        "split_counts": dict(Counter(str(row.get("split")) for row in results)),
        "ready_length_buckets": dict(Counter(str(row.get("actual_length_bucket")) for row in results if row.get("generation_result") == "READY")),
        "ready_evidence_positions": dict(Counter(str(row.get("evidence_position_bin")) for row in results if row.get("generation_result") == "READY")),
        "ready_subcategories": dict(Counter(str(row.get("canonical_subcategory_code")) for row in results if row.get("generation_result") == "READY")),
        "role_marker_policy": "marked text retained; model_input_text and WordPiece count are marker-free",
        "files": {
            "results": str(result_path),
            "ready": str(ready_path),
            "failed": str(failed_path),
        },
    }
    write_json(output_dir / ("pilot_summary.json" if is_pilot else "generation_summary.json"), summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--thinking-effort", choices=["off", "medium", "high", "max"], default="medium")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="pilot条数；0表示生成全部任务")
    parser.add_argument("--only-length-bucket", choices=sorted(BUCKETS), help="只生成指定长度桶，用于定向冒烟")
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--rpm", type=int, default=1500)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--train-fraud", type=int, default=1500)
    parser.add_argument("--train-normal", type=int, default=1500)
    parser.add_argument("--dev-fraud", type=int, default=500)
    parser.add_argument("--dev-normal", type=int, default=500)
    parser.add_argument("--stress-fraud", type=int, default=250)
    parser.add_argument("--stress-normal", type=int, default=250)
    args = parser.parse_args()

    tasks, manifest = make_tasks(
        output_dir=args.output_dir,
        taxonomy_path=args.taxonomy,
        train_fraud=args.train_fraud,
        train_normal=args.train_normal,
        dev_fraud=args.dev_fraud,
        dev_normal=args.dev_normal,
        stress_fraud=args.stress_fraud,
        stress_normal=args.stress_normal,
    )
    print(json.dumps(manifest["task_counts"], ensure_ascii=False, sort_keys=True))
    if args.prepare_only:
        return 0
    if args.only_length_bucket:
        tasks = [task for task in tasks if task["length_bucket"] == args.only_length_bucket]
    summary = run_generation(
        tasks,
        output_dir=args.output_dir,
        endpoint=args.endpoint,
        model=args.model,
        vocab=args.vocab,
        workers=args.workers,
        rpm=args.rpm,
        retries=args.retries,
        limit=args.limit,
        thinking_effort=args.thinking_effort,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
