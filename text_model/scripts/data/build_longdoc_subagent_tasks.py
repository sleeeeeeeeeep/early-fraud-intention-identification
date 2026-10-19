#!/usr/bin/env python3
"""构建 V361 Luna 子代理长文本生成任务。

脚本只建立任务包和分片，不调用 HTTP API，也不生成对话正文。每个最终数据槽位
至少有一个候选任务，25%槽位附加一个同约束备选候选，共1225个任务。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY = ROOT / "data/processed/v329_qm_audited_backfill6500_20260725/label_subcategory_rubrics_official_v1.json"
DEFAULT_TRAIN = ROOT / "data/processed/v340_policy_aligned_coretrain6500/train_v340_policy_aligned_coretrain_6500.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/v361_longdoc_subagent_tasks_20260824"

LENGTH_BUCKETS = {
    "wp_385_512": (385, 512),
    "wp_513_1024": (513, 1024),
    "wp_1025_1600": (1025, 1600),
    "wp_1601_2600": (1601, 2600),
}
LENGTH_SLOT_COUNTS = {
    "train": {"wp_385_512": 155, "wp_513_1024": 248, "wp_1025_1600": 155, "wp_1601_2600": 62},
    "dev": {"wp_385_512": 60, "wp_513_1024": 96, "wp_1025_1600": 60, "wp_1601_2600": 24},
    "stress": {"wp_385_512": 30, "wp_513_1024": 48, "wp_1025_1600": 30, "wp_1601_2600": 12},
}
CONTENT_SLOT_COUNTS = {
    "train": {"target_fraud": 240, "general_fraud": 70, "target_normal": 120, "general_normal": 190},
    "dev": {"target_fraud": 80, "general_fraud": 40, "target_normal": 80, "general_normal": 40},
    "stress": {"target_fraud": 40, "general_fraud": 20, "target_normal": 40, "general_normal": 20},
}
SCENARIOS = (
    "douyin_live_membership",
    "million_protection",
    "business_license_annual_review",
    "private_number_douyin_official",
)
EVIDENCE_BINS = ("Q1", "Q2", "Q3", "Q4")
GENERAL_NORMAL_TYPES = (
    "official_customer_service",
    "official_bank_notice",
    "official_tax_guidance",
    "anti_fraud_education",
    "customer_reports_past_scam",
    "order_refund_official_path",
    "business_consultation",
    "long_service_complaint",
)


def stable_int(value: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:16], 16)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_taxonomy(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    children: dict[str, dict[str, Any]] = {}
    for parent_code, parent in sorted(payload["parents"].items(), key=lambda item: int(item[0])):
        for child in parent["subcategories"]:
            code = str(child["code"])
            children[code] = {
                "label_13": int(parent_code),
                "parent_name": str(parent["name"]),
                "subcategory_code": code,
                "subcategory_name": str(child["name"]),
                "rubric": str(child["rubric"]),
            }
    if len(children) != 39:
        raise ValueError(f"taxonomy must contain 39 subcategories, got {len(children)}")
    fraud_codes = [code for code, item in children.items() if item["label_13"] != 0]
    return children, fraud_codes


def train_child_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            code = str(row.get("canonical_subcategory_code") or row.get("subcategory_code") or "")
            counts[code] += 1
    return counts


def balanced_values(values: list[str], count: int, salt: str) -> list[str]:
    if not values:
        raise ValueError("cannot allocate from an empty value list")
    output = [values[index % len(values)] for index in range(count)]
    random.Random(stable_int(salt, "balanced-values")).shuffle(output)
    return output


def target_label(scenario: str, occurrence: int, split: str) -> tuple[int, str, str]:
    if scenario in {"douyin_live_membership", "private_number_douyin_official"}:
        return 1, "01_0101", "douyin_platform_customer_service"
    if scenario == "business_license_annual_review":
        return 2, "02_0202", "tax_or_market_regulator"
    identities = (
        (1, "01_0101", "platform_customer_service"),
        (2, "02_0202", "financial_regulator"),
        (12, "12_1207", "bank_or_insurance_staff"),
    )
    return identities[stable_int(f"{split}|{occurrence}", "million-identity") % len(identities)]


def general_fraud_codes(
    count: int,
    fraud_codes: list[str],
    source_counts: Counter[str],
    split: str,
) -> list[str]:
    # Train槽位先覆盖全部38个诈骗子类；剩余额度优先补训练集中稀缺子类。
    ordered = sorted(fraud_codes, key=lambda code: (source_counts.get(code, 0), code))
    output: list[str] = []
    while len(output) < count:
        output.extend(ordered)
    output = output[:count]
    random.Random(stable_int(split, "general-fraud")).shuffle(output)
    return output


def make_content_slots(
    split: str,
    children: Mapping[str, Mapping[str, Any]],
    fraud_codes: list[str],
    source_counts: Counter[str],
) -> list[dict[str, Any]]:
    quotas = CONTENT_SLOT_COUNTS[split]
    slots: list[dict[str, Any]] = []
    per_scenario_fraud = quotas["target_fraud"] // len(SCENARIOS)
    per_scenario_normal = quotas["target_normal"] // len(SCENARIOS)
    for scenario in SCENARIOS:
        for occurrence in range(per_scenario_fraud):
            label, code, identity = target_label(scenario, occurrence, split)
            slots.append(
                {
                    "content_group": "target_fraud",
                    "generation_kind": "fraud",
                    "scenario_tag": scenario,
                    "label_13": label,
                    "subcategory_code": code,
                    "claimed_identity_type": identity,
                    "caller_number_type": "private" if scenario == "private_number_douyin_official" else "unknown",
                }
            )
        for occurrence in range(per_scenario_normal):
            slots.append(
                {
                    "content_group": "target_normal",
                    "generation_kind": "normal",
                    "scenario_tag": scenario,
                    "label_13": 0,
                    "subcategory_code": "00_0000",
                    "claimed_identity_type": "legitimate_or_verified_channel",
                    "caller_number_type": "private" if scenario == "private_number_douyin_official" else "official_or_unknown",
                }
            )
    target_fraud_created = per_scenario_fraud * len(SCENARIOS)
    target_normal_created = per_scenario_normal * len(SCENARIOS)
    if target_fraud_created != quotas["target_fraud"] or target_normal_created != quotas["target_normal"]:
        raise AssertionError(f"{split}: target scenario quotas must divide by four")

    for code in general_fraud_codes(
        quotas["general_fraud"], fraud_codes, source_counts, split
    ):
        item = children[code]
        slots.append(
            {
                "content_group": "general_fraud",
                "generation_kind": "fraud",
                "scenario_tag": "general_taxonomy_longcall",
                "label_13": int(item["label_13"]),
                "subcategory_code": code,
                "claimed_identity_type": "taxonomy_constrained",
                "caller_number_type": "unknown",
            }
        )
    normal_types = balanced_values(
        list(GENERAL_NORMAL_TYPES), quotas["general_normal"], f"{split}-normal"
    )
    for normal_type in normal_types:
        slots.append(
            {
                "content_group": "general_normal",
                "generation_kind": "normal",
                "scenario_tag": normal_type,
                "label_13": 0,
                "subcategory_code": "00_0000",
                "claimed_identity_type": "legitimate_or_verified_channel",
                "caller_number_type": "official_or_unknown",
            }
        )
    if len(slots) != sum(quotas.values()):
        raise AssertionError(f"{split}: built {len(slots)} slots")
    random.Random(stable_int(split, "content-shuffle")).shuffle(slots)
    return slots


def assign_slot_axes(slots: list[dict[str, Any]], split: str) -> None:
    lengths: list[str] = []
    for bucket, count in LENGTH_SLOT_COUNTS[split].items():
        lengths.extend([bucket] * count)
    random.Random(stable_int(split, "length-shuffle")).shuffle(lengths)
    fraud_indices = [index for index, slot in enumerate(slots) if slot["generation_kind"] == "fraud"]
    evidence_bins = balanced_values(list(EVIDENCE_BINS), len(fraud_indices), f"{split}-evidence")
    evidence_cursor = 0
    for index, (slot, bucket) in enumerate(zip(slots, lengths)):
        slot["length_bucket"] = bucket
        low, high = LENGTH_BUCKETS[bucket]
        slot["target_wordpiece_min"] = low
        slot["target_wordpiece_max"] = high
        if slot["generation_kind"] == "fraud":
            slot["evidence_position_bin"] = evidence_bins[evidence_cursor]
            evidence_cursor += 1
        else:
            slot["evidence_position_bin"] = "normal"
            slot["cross_window_evidence"] = False
    cross_count = round(len(fraud_indices) * 0.20)
    cross_candidates = [
        index
        for index in fraud_indices
        if slots[index]["length_bucket"] != "wp_385_512"
        and slots[index]["evidence_position_bin"] in {"Q3", "Q4"}
    ]
    if len(cross_candidates) < cross_count:
        raise AssertionError(
            f"{split}: only {len(cross_candidates)} feasible cross-window slots"
        )
    cross_indices = set(
        sorted(
            cross_candidates,
            key=lambda index: stable_int(f"{split}|{index}", "cross-window"),
        )[:cross_count]
    )
    for index in fraud_indices:
        slots[index]["cross_window_evidence"] = index in cross_indices


def scenario_instruction(slot: Mapping[str, Any], children: Mapping[str, Mapping[str, Any]]) -> str:
    scenario = str(slot["scenario_tag"])
    if slot["content_group"] == "general_fraud":
        item = children[str(slot["subcategory_code"])]
        return f"围绕官方子类“{item['subcategory_name']}”生成，严格遵守：{item['rubric']}"
    if slot["content_group"] == "general_normal":
        return "生成合法、自然、可核验的长通话；可以出现付款、验证码、下载等词，但来电方不得推进高风险动作。"
    fraud = slot["generation_kind"] == "fraud"
    instructions = {
        "douyin_live_membership": "围绕抖音直播会员、平台会员权益或会员到期展开。",
        "million_protection": "围绕‘百万保障’到期、续费、关闭或账户整改展开。",
        "business_license_annual_review": "围绕营业执照逾期年审、电子税务、零申报或罚款威胁展开。",
        "private_number_douyin_official": "来电号码元数据为私人号码，正文中只体现对方自称抖音官方客服，不要口头念出‘私人号码’。",
    }
    suffix = (
        "来电方最终推进下载非官方软件、共享屏幕、验证码、敏感信息或资金操作，形成明确诈骗闭环。"
        if fraud
        else "整通电话保持合法流程，主动引导用户通过官方App、官网或线下窗口独立核验，不索取验证码、转账或远控。"
    )
    return instructions[scenario] + suffix


def identity_instruction(slot: Mapping[str, Any], children: Mapping[str, Mapping[str, Any]]) -> str:
    """把父类映射依赖的冒充身份显式写入生成约束。"""

    identity = str(slot["claimed_identity_type"])
    if identity in {"douyin_platform_customer_service", "platform_customer_service"}:
        return "来电方全程自称抖音平台、直播会员或平台保障客服，身份保持为平台客服。"
    if identity == "tax_or_market_regulator":
        return "来电方全程自称税务局、市场监管局或政务登记工作人员，身份保持为政府监管人员。"
    if identity == "financial_regulator":
        return "来电方全程自称金融监管总局、银保监会等监管机关工作人员，不能改成平台客服、银行或保险公司客服。"
    if identity == "bank_or_insurance_staff":
        return "来电方全程自称银行或保险公司工作人员，不能改成平台客服或政府监管人员。"
    if identity == "taxonomy_constrained":
        item = children[str(slot["subcategory_code"])]
        return f"来电方身份、关系和诈骗理由必须服务于“{item['subcategory_name']}”定义，不能混入其他诈骗子类的身份主轴。"
    if str(slot["scenario_tag"]) == "private_number_douyin_official":
        return "来电方可自称抖音授权回访人员，但只提供公开业务说明并引导接电人挂机后从官方App独立核验。"
    return "来电方使用可核验的真实业务身份，允许接电人挂机后通过官方App、官网、官方客服电话或线下窗口独立确认。"


def length_instruction(bucket: str) -> str:
    return {
        "wp_385_512": "约14至18轮，每轮保持自然简洁。",
        "wp_513_1024": "约20至30轮，包含连续业务推进和必要追问。",
        "wp_1025_1600": "约32至46轮，增加真实核验、质疑和流程细节。",
        "wp_1601_2600": "约48至70轮，保持情节连续，避免复读和机械填充。",
    }[bucket]


def progression_instruction(bucket: str) -> str:
    stages = {
        "wp_385_512": "至少形成开场、核对诉求、处理分歧、结束四个连续阶段",
        "wp_513_1024": "至少形成开场、背景核对、业务解释、用户追问、流程推进、结束六个连续阶段",
        "wp_1025_1600": "至少形成开场、身份核验、历史事实、规则解释、多轮异议、流程推进、关键动作、结果收束八个连续阶段",
        "wp_1601_2600": "至少形成开场、身份核验、历史事实、业务细节、用户异议、替代方案、流程推进、关键动作、后续影响、结果收束十个连续阶段",
    }[bucket]
    return (
        f"{stages}。每一阶段引入新的事实、问题、决定或操作结果；通过业务进展扩展长度，"
        "每轮承接上一轮并推进下一步。逐轮原创，禁止复制整组问答、循环固定句式、批量追加同一尾句或换少量词重复。"
    )


def build_prompt(task: Mapping[str, Any], children: Mapping[str, Mapping[str, Any]]) -> str:
    evidence = str(task["evidence_position_bin"])
    evidence_rule = (
        f"首次明确风险动作应落在全文{evidence}区间。"
        if task["generation_kind"] == "fraud"
        else "正常样本全程保持合法业务闭环。"
    )
    cross_rule = (
        "身份/理由证据放在较早窗口，最终风险动作放在后续相邻窗口；任何单个320-token窗口都不能同时看到完整组合，语义保持自然连贯。"
        if task["cross_window_evidence"]
        else "关键语义按自然通话节奏分布。"
    )
    return f"""你负责生成一条完整中文电话对话。每条样本从零生成，禁止拼接、改写或引用现有样本。

任务：{task['task_id']}
标签：父类{task['label_13']}，子类{task['subcategory_code']}，类型{task['generation_kind']}
场景要求：{scenario_instruction(task, children)}
身份边界：{identity_instruction(task, children)}
长度要求：去除‘来电人/接电人’标识后，WordPiece必须位于{task['target_wordpiece_min']}至{task['target_wordpiece_max']}；{length_instruction(str(task['length_bucket']))}
情节推进：{progression_instruction(str(task['length_bucket']))}
证据要求：{evidence_rule}{cross_rule}

写作要求：
1. 来电人与接电人自然交替，人物身份、时间线、称呼和业务事实保持一致。
2. 接电人反应可包含配合、追问、质疑或拒绝，避免固定模板。
3. 风险证据必须由来电人实际提出；接电人的否定、复述和反诈提醒不算风险动作。
4. 诈骗样本中，来电人持续推动诈骗目标；拒绝、报警、举报、挂机和安全核验等防守动作只由接电人表达。
5. 正常样本以合法服务闭环收束，来电方主动提供官方独立核验路径。
6. 完成长文本后检查相邻轮次和全文，不保留重复段落或仅用于凑长度的内容。
7. 输出纯JSON对象，不要Markdown，不要解释。

输出结构：
{{"task_id":"{task['task_id']}","turns":[{{"speaker":"caller","text":"..."}},{{"speaker":"callee","text":"..."}}],"context_evidence":["来电人逐字说出的身份或理由短语"],"risk_evidence":["来电人逐字提出的可执行风险动作短语"],"scenario_summary":"不超过40字","generation_notes":"不超过50字"}}
正常样本的context_evidence和risk_evidence都使用空数组；跨窗口诈骗样本至少各提供一条context_evidence和risk_evidence。"""


def make_slots(
    children: Mapping[str, Mapping[str, Any]],
    fraud_codes: list[str],
    source_counts: Counter[str],
) -> list[dict[str, Any]]:
    all_slots: list[dict[str, Any]] = []
    serial = 0
    for split in ("train", "dev", "stress"):
        split_slots = make_content_slots(split, children, fraud_codes, source_counts)
        assign_slot_axes(split_slots, split)
        for split_index, slot in enumerate(split_slots):
            slot.update(
                {
                    "slot_id": f"V361_SLOT_{split}_{split_index:04d}",
                    "split": split,
                    "slot_index": split_index,
                    "global_slot_index": serial,
                }
            )
            all_slots.append(slot)
            serial += 1
    if len(all_slots) != 980:
        raise AssertionError(f"expected 980 final slots, got {len(all_slots)}")
    return all_slots


def make_tasks(
    slots: list[dict[str, Any]], children: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    # 每个slot一个主候选；按split稳定增加155/60/30个备选，共1225。
    alternative_slots: set[str] = set()
    for split, extra in (("train", 155), ("dev", 60), ("stress", 30)):
        split_slots = [slot for slot in slots if slot["split"] == split]
        alternative_slots.update(
            str(slot["slot_id"])
            for slot in sorted(
                split_slots,
                key=lambda item: stable_int(str(item["slot_id"]), "alternative"),
            )[:extra]
        )
    tasks: list[dict[str, Any]] = []
    serial = 0
    for slot in slots:
        variants = 2 if slot["slot_id"] in alternative_slots else 1
        for variant in range(variants):
            task = dict(slot)
            task.update(
                {
                    "schema_version": "v361.longdoc_subagent_task.v1",
                    "task_id": f"V361_TASK_{serial:04d}",
                    "candidate_variant": variant,
                    "generator": "codex_luna_subagent",
                    "generator_model": "gpt-5.6-luna",
                    "reasoning_effort": "medium",
                    "review_status": "pending_generation",
                    "single_complete_dialogue": True,
                    "api_transport": False,
                }
            )
            task["prompt"] = build_prompt(task, children)
            tasks.append(task)
            serial += 1
    if len(tasks) != 1225:
        raise AssertionError(f"expected 1225 tasks, got {len(tasks)}")
    return tasks


def choose_smoke(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_slots: set[str] = set()
    # 8类（四诈骗+四匹配正常）×4长度，另加8条通用样本。
    for generation_kind in ("fraud", "normal"):
        for scenario in SCENARIOS:
            for bucket in LENGTH_BUCKETS:
                candidates = [
                    task
                    for task in tasks
                    if task["generation_kind"] == generation_kind
                    and task["scenario_tag"] == scenario
                    and task["length_bucket"] == bucket
                    and task["slot_id"] not in used_slots
                ]
                if not candidates:
                    raise AssertionError(
                        f"smoke coverage missing: {generation_kind}/{scenario}/{bucket}"
                    )
                if generation_kind == "fraud":
                    desired_cross = bucket == "wp_513_1024"
                    preferred = [
                        task
                        for task in candidates
                        if bool(task["cross_window_evidence"]) == desired_cross
                    ]
                    if preferred:
                        candidates = preferred
                chosen = min(candidates, key=lambda item: stable_int(item["task_id"], "smoke"))
                selected.append(chosen)
                used_slots.add(str(chosen["slot_id"]))
    for group in ("general_fraud", "general_normal"):
        for bucket in LENGTH_BUCKETS:
            candidates = [
                task
                for task in tasks
                if task["content_group"] == group
                and task["length_bucket"] == bucket
                and task["slot_id"] not in used_slots
            ]
            if group == "general_fraud":
                preferred = [
                    task for task in candidates if not task["cross_window_evidence"]
                ]
                if preferred:
                    candidates = preferred
            chosen = min(candidates, key=lambda item: stable_int(item["task_id"], "smoke-general"))
            selected.append(chosen)
            used_slots.add(str(chosen["slot_id"]))
    if len(selected) != 40:
        raise AssertionError(f"smoke must contain 40 tasks, got {len(selected)}")
    return selected


def count_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    axes = (
        "split",
        "content_group",
        "generation_kind",
        "scenario_tag",
        "label_13",
        "subcategory_code",
        "length_bucket",
        "evidence_position_bin",
        "cross_window_evidence",
    )
    return {
        "rows": len(rows),
        **{
            axis: dict(sorted(Counter(str(row.get(axis)) for row in rows).items()))
            for axis in axes
        },
    }


def shard(rows: list[dict[str, Any]], output_dir: Path, prefix: str, size: int = 10) -> list[str]:
    paths: list[str] = []
    for index, start in enumerate(range(0, len(rows), size)):
        path = output_dir / f"{prefix}_{index:03d}.jsonl"
        write_jsonl(path, rows[start : start + size])
        paths.append(str(path))
    return paths


def build(taxonomy_path: Path, train_path: Path, output_dir: Path) -> dict[str, Any]:
    children, fraud_codes = load_taxonomy(taxonomy_path)
    counts = train_child_counts(train_path)
    slots = make_slots(children, fraud_codes, counts)
    tasks = make_tasks(slots, children)
    smoke = choose_smoke(tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "final_slots_980.jsonl", slots)
    write_jsonl(output_dir / "all_tasks_1225.jsonl", tasks)
    write_jsonl(output_dir / "smoke_tasks_40.jsonl", smoke)
    smoke_shards = shard(smoke, output_dir / "smoke_shards", "smoke", 10)
    manifest = {
        "schema_version": "v361.longdoc_subagent_task_manifest.v1",
        "generation_transport": "codex_multi_agent_luna",
        "http_api_used": False,
        "model": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "candidate_tasks": count_summary(tasks),
        "final_slots": count_summary(slots),
        "smoke_tasks": count_summary(smoke),
        "smoke_shards": smoke_shards,
        "full_generation_requires_user_confirmation": True,
        "retry_policy": "one primary plus one prebuilt alternative for 245 slots; unresolved slots are reported",
        "subagent_attempt_limit_per_candidate": 3,
        "output_contract": {
            "format": "JSON object per task",
            "roles": ["caller", "callee"],
            "role_rendering": ["来电人：", "接电人："],
            "role_markers_excluded_from_model_tokens": True,
        },
    }
    write_json(output_dir / "task_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.taxonomy, args.train, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
