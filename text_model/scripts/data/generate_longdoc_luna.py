#!/usr/bin/env python3
"""使用本地 Luna 生成第一阶段长文本增强候选。

这份脚本严格区分两个字段：

* ``text``/``audio_text``：保留来电人、接电人等角色标识，供后续音频合成；
* ``model_input_text``：去掉角色标识，供 tokenizer、WordPiece 长度和模型输入。

脚本默认只生成候选并标记 ``review_status=pending``，不会把未经审核的候选
伪装成最终训练数据。默认来源不读取 Test-B、OOF 或甲方示例。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.request

try:
    from text_model.data.role_markers import (
        role_contract,
        role_turns,
        strip_role_markers,
    )
except ModuleNotFoundError:  # 允许直接以脚本路径启动
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from text_model.data.role_markers import (  # type: ignore[no-redef]
        role_contract,
        role_turns,
        strip_role_markers,
    )


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_SOURCE = ROOT / (
    "data/processed/v340_policy_aligned_coretrain6500/"
    "train_v340_policy_aligned_coretrain_6500.jsonl"
)
DEFAULT_DEV_SOURCE = ROOT / (
    "data/processed/v314_auditclosed6500_deva1801_testb1580_20260721/"
    "dev_a_complete_1801.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/processed/longdoc_v1_20260805_luna"
DEFAULT_VOCAB = ROOT / (
    "runs/r_260727_042528_V341D1_Policy6500_MegatronBERT_DataCausal_DevOnly_s202_"
    "auto_a1/subruns/seed-202/best_model/vocab.txt"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:4000/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-luna"
DATASET_VERSION = "longdoc_v1_20260805_luna"

ROLE_LINE_RE = re.compile(
    r"^\s*(?:来电方|来电人|接听方|接听人|接电人|接收方|客服|客户|用户|对方|"
    r"诈骗分子|受害人|说话人[一二三四五六七八九十]?|caller|callee|"
    r"speaker[_ ]?[0-9]+|甲|乙|A|B)\s*[:：]",
    re.IGNORECASE,
)
RISK_TERM_RE = re.compile(
    r"转账|汇款|付款|缴费|垫付|保证金|手续费|验资|充值|入金|验证码|银行卡|"
    r"身份证|链接|下载|扫码|二维码|添加微信|私信|远程|屏幕共享|收款码|安全账户"
)
ASSERTED_RISK_RE = re.compile(
    r"(?:请|需要|必须|先|马上|立即|赶紧|现在|点击|提供|提交|转到|发到).{0,16}"
    r"(?:转账|汇款|付款|缴费|垫付|保证金|手续费|验资|充值|入金|验证码|银行卡|"
    r"身份证|链接|下载|扫码|二维码|添加微信|私信|远程|屏幕共享|收款码|安全账户)"
)
NEGATED_RISK_RE = re.compile(
    r"(?:不要|无需|不需要|不会|不能|切勿|请勿|拒绝|不接受|不向).{0,18}"
    r"(?:转账|汇款|付款|缴费|垫付|保证金|手续费|验资|充值|入金|验证码|银行卡|"
    r"身份证|链接|下载|扫码|二维码|添加微信|私信|远程|屏幕共享|收款码|安全账户)"
)
NORMAL_UNSAFE_ACTION_RE = re.compile(
    r"(?:请|需要|必须|先|马上|立即|赶紧|现在|要求|让您|让你).{0,20}"
    r"(?:向陌生账户转账|转到(?:个人|指定|陌生|安全)账户|扫(?:码|描)收款码|"
    r"提供(?:短信|动态)?验证码|告诉(?:我|对方)验证码|提供完整银行卡|"
    r"下载(?:陌生|不明|非官方)|点击(?:陌生|不明|外部|私聊|聊天)链接|"
    r"添加微信|转到私下账户|屏幕共享|远程控制|开启远程|提交银行卡密码)"
)

BUCKETS: dict[str, tuple[int, int, int]] = {
    "q1_320_480": (320, 480, 410),
    "q2_481_720": (481, 720, 600),
    "q3_721_960": (721, 960, 840),
    "q4_961_1280": (961, 1280, 1080),
}
BUCKET_ORDER = tuple(BUCKETS)
METHOD_FRACTIONS = {
    "benign_prefix_extension": 0.45,
    "middle_context_extension": 0.25,
    "evidence_position_control": 0.20,
    "multi_stage_fraud": 0.10,
}
NORMAL_FRACTIONS = {
    "ordinary_legitimate_dialogue": 0.50,
    "high_risk_hard_negative": 0.30,
    "anti_fraud_quote": 0.20,
}
HARD_NEGATIVE_TYPES = (
    "official_payment_or_refund",
    "otp_and_bankcard_safety_reminder",
    "download_app_official_channel",
    "past_scam_description",
    "anti_fraud_quote",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    name = path.name.lower().replace("-", "_")
    if "test_b" in name or "testb" in name or "oof" in str(path).lower():
        raise ValueError(f"拒绝读取受保护来源：{path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON 行不是对象")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def row_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("sample_id") or row.get("base_sample_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("来源行缺少 sample_id")
    return value


def label(row: Mapping[str, Any]) -> int:
    value = row.get("canonical_label_13", row.get("label_13", row.get("label", 0)))
    return int(value)


def child_code(row: Mapping[str, Any]) -> str:
    return str(
        row.get("canonical_subcategory_code")
        or row.get("subcategory_code")
        or row.get("child_code")
        or ("00_0000" if label(row) == 0 else "")
    ).strip()


def marked_text(row: Mapping[str, Any]) -> str:
    value = row.get("text_with_role_markers") or row.get("audio_text") or row.get("text")
    if value in (None, ""):
        value = row.get("model_input_text") or row.get("transcript") or ""
    return str(value).strip()


def normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", strip_role_markers(text)).lower()


def source_quality_ok(row: Mapping[str, Any], *, fraud: bool) -> tuple[bool, str]:
    text = marked_text(row)
    if len(text) < 80:
        return False, "text_too_short"
    if bool(row.get("ambiguity_flag")):
        return False, "ambiguity_flag"
    if str(row.get("policy_action") or "").lower() in {"exclude", "drop", "reject"}:
        return False, "policy_excluded"
    if fraud and not RISK_TERM_RE.search(text):
        return False, "no_risk_evidence_candidate"
    if "甲方" in text or "client_long_example" in row_id(row).lower():
        return False, "client_example"
    return True, ""


def allocate_quota(groups: Mapping[Any, Sequence[Any]], desired: int, *, minimum: int = 0) -> dict[Any, int]:
    if desired < 0 or desired > sum(len(values) for values in groups.values()):
        raise ValueError(f"配额 {desired} 超过来源容量 {sum(len(v) for v in groups.values())}")
    quotas = {key: min(len(values), minimum) for key, values in groups.items()}
    if sum(quotas.values()) > desired:
        quotas = {key: 0 for key in groups}
    remaining = desired - sum(quotas.values())
    if remaining <= 0:
        return quotas
    weights = {key: max(1.0, len(values) ** 0.5) for key, values in groups.items()}
    total_weight = sum(weights.values())
    fractions: list[tuple[float, Any]] = []
    for key, values in groups.items():
        raw = remaining * weights[key] / total_weight
        add = min(len(values) - quotas[key], int(raw))
        quotas[key] += add
        fractions.append((raw - add, key))
    left = desired - sum(quotas.values())
    for _, key in sorted(fractions, key=lambda pair: (-pair[0], str(pair[1]))):
        if left <= 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            left -= 1
    if sum(quotas.values()) != desired:
        raise AssertionError("配额分配未闭合")
    return quotas


def stratified_select(
    rows: Sequence[dict[str, Any]],
    desired: int,
    *,
    salt: str,
    key_fn: Any,
    minimum: int = 0,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    quotas = allocate_quota(groups, desired, minimum=minimum)
    selected: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(values, key=lambda row: stable_key(row_id(row), salt))
        selected.extend(ordered[: quotas[key]])
    if len(selected) != desired:
        raise AssertionError(f"实际选择 {len(selected)}，期望 {desired}")
    return selected


def select_fraud_sources(rows: Sequence[dict[str, Any]], desired: int, *, salt: str) -> tuple[list[dict[str, Any]], list[str]]:
    by_child = Counter(child_code(row) for row in rows)
    rare = sorted(code for code, count in by_child.items() if count < 10)
    eligible = [row for row in rows if by_child[child_code(row)] >= 10]
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_parent[label(row)].append(row)
    selected = stratified_select(
        eligible,
        desired,
        salt=salt,
        key_fn=lambda row: label(row),
        minimum=60 if desired >= 720 and len(by_parent) >= 12 else 1,
    )
    return selected, rare


def source_role_style(text: str) -> tuple[str, str]:
    caller = "来电人"
    callee = "接电人"
    for line in str(text).splitlines():
        if re.match(r"\s*(来电方|来电人|caller|甲|A)\s*[:：]", line, re.I):
            caller = re.match(r"\s*([^:：]+)\s*[:：]", line).group(1).strip()  # type: ignore[union-attr]
        if re.match(r"\s*(接听方|接听人|接电人|接收方|callee|乙|B)\s*[:：]", line, re.I):
            callee = re.match(r"\s*([^:：]+)\s*[:：]", line).group(1).strip()  # type: ignore[union-attr]
    return caller, callee


def choose_bucket(key: str) -> str:
    value = int(stable_key(key, "length-bucket")[:8], 16) % 100
    if value < 35:
        return BUCKET_ORDER[0]
    if value < 70:
        return BUCKET_ORDER[1]
    if value < 90:
        return BUCKET_ORDER[2]
    return BUCKET_ORDER[3]


def choose_fractional(key: str, fractions: Mapping[str, float]) -> str:
    value = int(stable_key(key, "method")[:8], 16) % 10000 / 10000.0
    cursor = 0.0
    for name, fraction in fractions.items():
        cursor += fraction
        if value < cursor:
            return name
    return next(reversed(fractions))


def evidence_position_bin(key: str) -> str:
    value = int(stable_key(key, "evidence-position")[:8], 16) % 100
    if value < 20:
        return "0_25"
    if value < 45:
        return "25_50"
    if value < 70:
        return "50_75"
    return "75_100"


def target_info(bucket: str) -> tuple[int, int, int]:
    return BUCKETS[bucket]


def first_risk_line(text: str) -> int:
    lines = str(text).splitlines()
    for index, line in enumerate(lines):
        body = strip_role_markers(line)
        if ASSERTED_RISK_RE.search(body) or RISK_TERM_RE.search(body):
            return index
    return max(0, len(lines) // 2)


def make_tasks(
    *,
    train_source: Path,
    dev_source: Path,
    output_dir: Path,
    include_splits: set[str],
    train_fraud: int,
    train_normal: int,
    dev_fraud: int,
    dev_normal: int,
    stress_fraud: int,
    stress_normal: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_rows = read_jsonl(train_source)
    if len(train_rows) != 6500:
        raise ValueError(f"Core-Train 应为6500条，实际{len(train_rows)}")
    if len({row_id(row) for row in train_rows}) != len(train_rows):
        raise ValueError("Core-Train sample_id 不唯一")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in train_rows:
        ok, reason = source_quality_ok(row, fraud=label(row) != 0)
        (accepted if ok else rejected).append({**row, "rejection_reason": reason} if not ok else row)
    accepted_train = [row for row in accepted if isinstance(row, dict)]
    fraud_train_pool = [row for row in accepted_train if label(row) != 0]
    normal_train_pool = [row for row in accepted_train if label(row) == 0]
    fraud_sources, rare_subclasses = select_fraud_sources(
        fraud_train_pool, train_fraud, salt="longdoc-v1-train-fraud"
    ) if "train" in include_splits else ([], [])
    # 训练正常样本遵守“多数来源1个、部分高质量来源2个”：
    # 第一阶段1500条对应1000个来源，其中500个来源生成第2个版本。
    normal_source_count = max(1, train_normal - train_normal // 3)
    normal_sources = stratified_select(
        normal_train_pool,
        normal_source_count,
        salt="longdoc-v1-train-normal",
        key_fn=lambda row: "normal",
    ) if "train" in include_splits else []

    tasks: list[dict[str, Any]] = []

    def add_task(row: Mapping[str, Any], split: str, variant_index: int, kind: str) -> None:
        source_id = row_id(row)
        task_key = f"{split}|{source_id}|{variant_index}"
        bucket = choose_bucket(task_key)
        _, _, target_wp = target_info(bucket)
        method = (
            choose_fractional(task_key, METHOD_FRACTIONS)
            if label(row) != 0
            else choose_fractional(task_key, NORMAL_FRACTIONS)
        )
        caller, callee = source_role_style(marked_text(row))
        tasks.append(
            {
                "task_id": f"LONGDOC_LUNA_{split}_{source_id}_{variant_index}",
                "sample_id": f"LONGDOC_V1_{split}_{source_id}_{variant_index}",
                "base_sample_id": source_id,
                "source_sample_id": source_id,
                "source_split": split,
                "split": split,
                "variant_index": variant_index,
                "variant_kind": kind,
                "label_13": label(row),
                "canonical_label_13": label(row),
                "subcategory_code": child_code(row),
                "canonical_subcategory_code": child_code(row),
                "source_text": marked_text(row),
                "source_text_sha256": sha256_text(marked_text(row)),
                "source_model_input_text": strip_role_markers(marked_text(row)),
                "caller_marker": caller,
                "callee_marker": callee,
                "augmentation_method": method,
                "evidence_position_bin": evidence_position_bin(task_key),
                "length_bucket": bucket,
                "target_wordpiece_min": target_info(bucket)[0],
                "target_wordpiece_max": target_info(bucket)[1],
                "target_wordpiece_mid": target_wp,
                "target_body_chars": max(360, int(target_wp * 1.15)),
                "normal_hard_negative_type": (
                    HARD_NEGATIVE_TYPES[int(stable_key(task_key, "normal-type")[:8], 16) % len(HARD_NEGATIVE_TYPES)]
                    if label(row) == 0 else ""
                ),
                "review_status": "pending",
                "generation_provider": "local_luna_4000",
                "generation_model": DEFAULT_MODEL,
            }
        )

    if "train" in include_splits:
        for row in fraud_sources:
            add_task(row, "train", 0, "long_train")
        duplicate_ids = set(
            row_id(row)
            for row in sorted(normal_sources, key=lambda item: stable_key(row_id(item), "normal-duplicate"))
        )
        # 训练正常样本：大多数来源1个版本，前半部分来源再生成第2个版本。
        duplicate_ids = set(list(duplicate_ids)[: max(0, train_normal - normal_source_count)])
        for row in normal_sources:
            add_task(row, "train", 0, "long_train")
            if row_id(row) in duplicate_ids:
                add_task(row, "train", 1, "long_train")
        if len([task for task in tasks if task["split"] == "train" and task["label_13"] == 0]) != train_normal:
            raise AssertionError("Long-Train normal任务数量未闭合")
        if len([task for task in tasks if task["split"] == "train" and task["label_13"] != 0]) != train_fraud:
            raise AssertionError("Long-Train fraud任务数量未闭合")

    if include_splits & {"dev", "stress"}:
        dev_rows = read_jsonl(dev_source)
        if len({row_id(row) for row in dev_rows}) != len(dev_rows):
            raise ValueError("Dev-A sample_id 不唯一")
        eligible_dev = [row for row in dev_rows if source_quality_ok(row, fraud=label(row) != 0)[0]]
        used_dev_ids: set[str] = set()
        for split, fraud_count, normal_count in (
            ("dev", dev_fraud, dev_normal),
            ("stress", stress_fraud, stress_normal),
        ):
            if split not in include_splits:
                continue
            fraud_pool = [row for row in eligible_dev if label(row) != 0 and row_id(row) not in used_dev_ids]
            normal_pool = [row for row in eligible_dev if label(row) == 0 and row_id(row) not in used_dev_ids]
            if len(fraud_pool) < fraud_count or len(normal_pool) < normal_count:
                raise ValueError(
                    f"{split} 来源不足：需要 fraud={fraud_count}, normal={normal_count}；"
                    f"可用 fraud={len(fraud_pool)}, normal={len(normal_pool)}。"
                    "不自动复用base_sample_id，避免泄漏。"
                )
            selected_fraud = stratified_select(
                fraud_pool, fraud_count, salt=f"longdoc-v1-{split}-fraud", key_fn=lambda row: label(row)
            )
            selected_normal = stratified_select(
                normal_pool, normal_count, salt=f"longdoc-v1-{split}-normal", key_fn=lambda row: "normal"
            )
            for row in selected_fraud + selected_normal:
                used_dev_ids.add(row_id(row))
                add_task(row, split, 0, "long_dev" if split == "dev" else "long_stress")

    lineage = [
        {
            "task_id": task["task_id"],
            "split": task["split"],
            "base_sample_id": task["base_sample_id"],
            "label_13": task["label_13"],
            "subcategory_code": task["subcategory_code"],
            "source_text_sha256": task["source_text_sha256"],
        }
        for task in tasks
    ]
    by_split = {split: {task["base_sample_id"] for task in tasks if task["split"] == split} for split in {"train", "dev", "stress"}}
    for left, right in (("train", "dev"), ("train", "stress"), ("dev", "stress")):
        overlap = by_split.get(left, set()) & by_split.get(right, set())
        if overlap:
            raise AssertionError(f"{left}/{right} base_sample_id overlap: {len(overlap)}")
    write_jsonl(output_dir / "generation_tasks.jsonl", tasks)
    write_jsonl(output_dir / "source_lineage.jsonl", sorted(lineage, key=lambda row: row["task_id"]))
    write_jsonl(output_dir / "rejected_bases.jsonl", rejected)
    manifest = {
        "schema_version": "longdoc_luna_generation_manifest.v1",
        "dataset_version": DATASET_VERSION,
        "source": {
            "train_path": str(train_source),
            "train_rows": len(train_rows),
            "train_sha256": sha256_file(train_source),
            "dev_path": str(dev_source) if include_splits & {"dev", "stress"} else None,
            "test_b_read": False,
            "oof_read": False,
            "client_example_read": False,
        },
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
            for split in sorted({task["split"] for task in tasks})
        },
        "rare_subclasses_excluded_from_auto_batch": rare_subclasses,
        "role_contract": {
            "marked_fields": ["text", "audio_text", "text_with_role_markers"],
            "model_field": "model_input_text",
            "token_policy": "role_markers_excluded_from_model_input_and_wordpiece_counts",
            "accepted_markers": ["来电人", "接电人", "来电方", "接听方", "接听人", "接收方"],
        },
        "length_policy": {
            "wordpiece_buckets": BUCKETS,
            "bucket_proportions": {"q1_320_480": 0.35, "q2_481_720": 0.35, "q3_721_960": 0.20, "q4_961_1280": 0.10},
            "count_on": "model_input_text_after_role_marker_removal",
        },
        "generation": {
            "endpoint": DEFAULT_ENDPOINT,
            "model": DEFAULT_MODEL,
            "max_retries": 2,
            "rate_limit_rpm": 1500,
            "worker_cap": 96,
            "fraud_methods": METHOD_FRACTIONS,
            "normal_methods": NORMAL_FRACTIONS,
        },
    }
    write_json(output_dir / "generation_manifest.json", manifest)
    return tasks, manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BertWordPieceCounter:
    """只用于长度审计的最小 BERT WordPiece 计数器。"""

    def __init__(self, vocab_path: Path):
        self.vocab = {line.rstrip("\n") for line in vocab_path.open(encoding="utf-8")}

    @staticmethod
    def is_chinese(char: str) -> bool:
        cp = ord(char)
        return (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F
            or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF
        )

    def tokens(self, text: str) -> list[str]:
        basic: list[str] = []
        for char in str(text):
            if char.isspace():
                basic.append(" ")
            elif self.is_chinese(char) or unicodedata.category(char).startswith("P"):
                basic.extend((" ", char, " "))
            else:
                basic.append(char.lower())
        pieces: list[str] = []
        for token in "".join(basic).split():
            if token in self.vocab:
                pieces.append(token)
                continue
            start = 0
            local: list[str] = []
            while start < len(token):
                end = len(token)
                found = None
                while start < end:
                    part = token[start:end]
                    candidate = part if start == 0 else "##" + part
                    if candidate in self.vocab:
                        found = candidate
                        break
                    end -= 1
                if found is None:
                    local = ["[UNK]"]
                    break
                local.append(found)
                start = end
            pieces.extend(local)
        return pieces

    def count(self, text: str) -> int:
        return len(self.tokens(strip_role_markers(text)))


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / max(requests_per_second, 0.01)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.next_time = max(now, self.next_time) + self.interval
            delay = self.next_time - now
        if delay > 0:
            time.sleep(delay)


def extract_json(content: str) -> dict[str, Any]:
    value = str(content or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # 个别兼容服务会在 JSON 后追加一句说明；只接受首个完整对象。
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Luna 返回不是 JSON 对象")
    return parsed


def prompt_for(task: Mapping[str, Any], feedback: str = "") -> str:
    source = str(task["source_text"])
    caller = task["caller_marker"]
    callee = task["callee_marker"]
    low = task["target_wordpiece_min"]
    high = task["target_wordpiece_max"]
    method = task["augmentation_method"]
    pos = task["evidence_position_bin"]
    if int(task["label_13"]) != 0:
        return f"""你是中文电话通话长文本数据生成器。只输出一个 JSON 对象，不要解释。

目标：在不改变原始标签和最终诈骗机制的情况下，为下面的原始通话增加自然的前置或中间上下文。
父类标签：{task['label_13']}；官方子类：{task['subcategory_code']}。
增强方式：{method}；首次风险动作目标位置：{pos}；完整文本 WordPiece 目标：{low}–{high}。
角色标识必须保留，使用“{caller}：”和“{callee}：”，每轮单独一行。

原始通话（风险核心必须按原文保留，不能改写）：
{source}

输出字段：
{{"prefix_text":"新增前置通话，若不需要则为空","middle_text":"新增中间铺垫，若不需要则为空","risk_actions":["原文已经存在的风险动作短语"],"generation_notes":"不超过50字"}}

硬性要求：
1. 只能生成与原场景一致的背景、寒暄、身份核对、业务解释、用户追问和流程过渡；不要凭空添加另一种完整诈骗机制。
2. “{caller}”和“{callee}”角色必须交替合理，身份、称谓和时间线一致。
3. 诈骗核心证据来自原文，必须保持原文轮次的文字和顺序；不得删除、改写或弱化最终风险动作。
4. 新增内容不能包含旁白、括号、网址、电话号码、模型自述或审核结论。
5. 目标是完整文本去掉角色标识后的 WordPiece 长度落入 {low}–{high}；不要输出长度说明。
6. 方法 multi_stage_fraud 只能增加入口和信任铺垫，最终机制仍以原文为准，返回后必须人工深审。
7. 上次校验反馈：{feedback or '无'}
"""
    return f"""你是中文电话通话长文本数据生成器。只输出一个 JSON 对象，不要解释。

目标：把下面的真实正常通话扩展成长正常或硬负样本，标签必须保持正常（label_13=0）。
增强方式：{method}；硬负类型：{task['normal_hard_negative_type']}；完整文本 WordPiece 目标：{low}–{high}。
新增内容使用“{caller}：”和“{callee}：”，每轮单独一行；如果原文没有角色标识，仍要给新增对话使用这两个标识。

原始正常通话：
{source}

输出字段：
{{"extension_text":"新增的正常对话或业务上下文","quoted_risk_language":true,"caller_requests_action":false,"generation_notes":"不超过50字"}}

硬性要求：
1. 只写合法业务、正常客服、用户主动操作、过去被骗经历或反诈提醒。
2. 可以出现付款、退款、验证码、银行卡、下载 App、链接、转账等词，但不能出现来电方推进的可执行诈骗闭环。
3. 官方渠道、用户自主核验、拒绝陌生账户和“不提供验证码”等安全语境要自然，不要机械堆口号。
4. 不新增虚假身份、私域引流、远程控制、强制付款、索取验证码或陌生链接点击。
5. 不得出现旁白、网址、电话号码、括号、模型自述或审核结论。
6. 完整文本去掉角色标识后的 WordPiece 长度落入 {low}–{high}。
7. 上次校验反馈：{feedback or '无'}
"""


def call_luna(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    limiter: RateLimiter,
    thinking_effort: str = "off",
    timeout: int = 180,
) -> dict[str, Any]:
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("MIMO_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 LITELLM_API_KEY/MIMO_API_KEY")
    limiter.wait()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.55,
        "top_p": 0.90,
        "max_tokens": 1800,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if thinking_effort == "off":
        payload["extra_body"] = {"enable_thinking": False}
    else:
        # 本地 LiteLLM 代理使用 extra_body 传 provider 参数；同时保留
        # OpenAI-compatible 根字段，便于代理把 reasoning_effort 透传给 Luna。
        payload["reasoning_effort"] = thinking_effort
        payload["allowed_openai_params"] = ["reasoning_effort"]
        payload["extra_body"] = {
            "enable_thinking": True,
        }
        payload["max_tokens"] = 2400
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Luna HTTP {exc.code}: {detail}") from exc
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("Luna 返回缺少 choices")
    message = choices[0].get("message") or {}
    return extract_json(str(message.get("content") or ""))


def ordered_source_preserved(source: str, generated: str) -> bool:
    source_lines = [normalize_for_compare(line) for line in str(source).splitlines() if normalize_for_compare(line)]
    haystack = normalize_for_compare(generated)
    cursor = 0
    for line in source_lines:
        position = haystack.find(line, cursor)
        if position < 0:
            return False
        cursor = position + len(line)
    return bool(source_lines)


def compose_fraud(task: Mapping[str, Any], output: Mapping[str, Any]) -> str:
    source = str(task["source_text"]).strip()
    prefix = str(output.get("prefix_text") or "").strip()
    middle = str(output.get("middle_text") or "").strip()
    method = str(task["augmentation_method"])
    if method == "middle_context_extension" and middle:
        lines = source.splitlines()
        index = first_risk_line(source)
        body = lines[:index] + [middle] + lines[index:]
        return "\n".join(part.strip() for part in body if part.strip())
    if prefix:
        return f"{prefix}\n{source}".strip()
    return source


def compose_normal(task: Mapping[str, Any], output: Mapping[str, Any]) -> str:
    source = str(task["source_text"]).strip()
    extension = str(output.get("extension_text") or "").strip()
    return f"{source}\n{extension}".strip() if extension else source


def validate_candidate(
    task: Mapping[str, Any],
    output: Mapping[str, Any],
    counter: BertWordPieceCounter,
    reserved_hashes: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if int(task["label_13"]) != 0:
        text = compose_fraud(task, output)
        if not ordered_source_preserved(str(task["source_text"]), text):
            errors.append("fraud_source_turns_not_preserved")
        context = "\n".join(
            part for key in ("prefix_text", "middle_text") if (part := str(output.get(key) or "").strip())
        )
        safe_context = NEGATED_RISK_RE.sub("", strip_role_markers(context))
        if task["augmentation_method"] != "multi_stage_fraud" and ASSERTED_RISK_RE.search(safe_context):
            errors.append("new_asserted_risk_in_generated_context")
        if not output.get("risk_actions"):
            errors.append("missing_risk_actions")
    else:
        text = compose_normal(task, output)
        if output.get("caller_requests_action") not in {False, 0, "false", "False", None}:
            errors.append("normal_caller_requests_action_true")
        safe_text = NEGATED_RISK_RE.sub("", strip_role_markers(text))
        if NORMAL_UNSAFE_ACTION_RE.search(safe_text):
            errors.append("normal_asserted_risk")
    if not text.strip():
        errors.append("empty_text")
    if len(role_turns(text)) < 2:
        errors.append("role_turns_lt_2")
    actual_wp = counter.count(text)
    # Luna 很难按单条任务精确命中四个子桶；生成阶段只拒绝超出总边界的样本，
    # 最终按实际 WordPiece 长度重新分桶，避免把长度控制误变成大量无效重试。
    if not 320 <= actual_wp <= 1280:
        errors.append(
            f"wordpiece_out_of_global_range:{actual_wp} not in 320-1280"
        )
    normalized_hash = sha256_text(normalize_for_compare(text))
    if normalized_hash in reserved_hashes:
        errors.append("near_or_exact_reserved_duplicate")
    if errors:
        return None, errors
    contract = role_contract(text)
    row = {
        **{key: value for key, value in task.items() if key not in {"source_text", "source_model_input_text"}},
        **contract,
        "text": text,
        "text_hash": sha256_text(text),
        "text_sha256": sha256_text(text),
        "model_input_wordpiece_length": actual_wp,
        "actual_length_bucket": next(
            bucket
            for bucket, (low, high, _) in BUCKETS.items()
            if low <= actual_wp <= high
        ),
        "model_input_char_length": len(strip_role_markers(text)),
        "raw_char_length_with_role_markers": len(text),
        "label_preserved": True,
        "role_consistency": True,
        "human_review_status": "pending",
        "generation_result": "READY",
        "normalized_text_hash": normalized_hash,
        "risk_actions": output.get("risk_actions", []),
        "quoted_risk_language": bool(output.get("quoted_risk_language", False)),
        "caller_requests_action": output.get("caller_requests_action", int(task["label_13"]) != 0),
        "generation_notes": str(output.get("generation_notes") or "").strip(),
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
            raw = call_luna(
                endpoint,
                model,
                prompt_for(task, feedback),
                limiter=limiter,
                thinking_effort=thinking_effort,
            )
            row, errors = validate_candidate(task, raw, counter, reserved_hashes)
            attempts.append({"attempt": attempt, "errors": errors})
            if row is not None:
                row["luna_attempts"] = attempts
                return row
            feedback = "；".join(errors)
        except Exception as exc:  # noqa: BLE001 - 每条任务需落失败原因并继续
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


def choose_pilot(tasks: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(tasks):
        return list(tasks)
    fraud = [task for task in tasks if task["label_13"] != 0]
    normal = [task for task in tasks if task["label_13"] == 0]
    half = limit // 2
    selected = fraud[:half] + normal[: limit - half]
    return selected if len(selected) == limit else list(tasks[:limit])


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
    selected = choose_pilot(tasks, limit)
    counter = BertWordPieceCounter(vocab)
    limiter = RateLimiter(rpm / 60.0)
    reserved_hashes = {
        str(row.get("normalized_text_hash"))
        for path in (output_dir / "accepted_candidates.jsonl", output_dir / "pilot_candidates.jsonl")
        if path.exists()
        for row in read_jsonl(path)
        if row.get("normalized_text_hash")
    }
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
        "requested": len(selected),
        "ready": sum(row.get("generation_result") == "READY" for row in results),
        "failed": sum(row.get("generation_result") == "FAILED" for row in results),
        "label_counts": dict(Counter("normal" if row.get("label_13") == 0 else "fraud" for row in results)),
        "ready_length_buckets": dict(Counter(str(row.get("length_bucket")) for row in results if row.get("generation_result") == "READY")),
        "ready_method_counts": dict(Counter(str(row.get("augmentation_method")) for row in results if row.get("generation_result") == "READY")),
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
    parser.add_argument("--train-source", type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument("--dev-source", type=Path, default=DEFAULT_DEV_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--thinking-effort",
        choices=["off", "medium", "high", "max"],
        default="medium",
        help="Luna思考强度；off关闭，当前默认medium",
    )
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--split", choices=["train", "dev", "stress", "all"], default="train")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="pilot条数；0表示当前选择的全部任务")
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--rpm", type=int, default=1500)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--train-fraud", type=int, default=1500)
    parser.add_argument("--train-normal", type=int, default=1500)
    parser.add_argument("--dev-fraud", type=int, default=500)
    parser.add_argument("--dev-normal", type=int, default=500)
    parser.add_argument("--stress-fraud", type=int, default=250)
    parser.add_argument("--stress-normal", type=int, default=250)
    args = parser.parse_args()

    include_splits = {"train", "dev", "stress"} if args.split == "all" else {args.split}
    tasks, manifest = make_tasks(
        train_source=args.train_source,
        dev_source=args.dev_source,
        output_dir=args.output_dir,
        include_splits=include_splits,
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
