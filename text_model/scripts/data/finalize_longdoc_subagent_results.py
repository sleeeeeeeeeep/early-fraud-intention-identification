#!/usr/bin/env python3
"""校验 Luna 子代理结果并构建 V361 长文本数据。

生成、审核、入库分离：自动规则先产出 READY 候选，独立审核决策为 accept 后才可
填充最终slot。正式构建要求980个slot全部填满。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    from text_model.data.longdoc_chunker import make_plain_chunk_token_ranges
    from text_model.data.role_markers import role_contract
    from text_model.data.wordpiece_counter import BertWordPieceCounter
except ModuleNotFoundError:
    from data.longdoc_chunker import make_plain_chunk_token_ranges
    from data.role_markers import role_contract
    from data.wordpiece_counter import BertWordPieceCounter


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_DIR = ROOT / "data/processed/v361_longdoc_subagent_tasks_20260824"
DEFAULT_RESULT_DIR = ROOT / "data/processed/v361_longdoc_subagent_results_20260824"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/v361_longdoc_text_route_20260824"
DEFAULT_TRAIN = ROOT / "data/processed/v340_policy_aligned_coretrain6500/train_v340_policy_aligned_coretrain_6500.jsonl"
DEFAULT_VOCAB = ROOT / "runs/r_260727_042528_V341D1_Policy6500_MegatronBERT_DataCausal_DevOnly_s202_auto_a1/subruns/seed-202/best_model/vocab.txt"

SCENARIO_PATTERNS = {
    "douyin_live_membership": re.compile(r"抖音|直播|会员"),
    "million_protection": re.compile(r"百万保障"),
    "business_license_annual_review": re.compile(r"营业执照|年审|税务|零申报"),
    "private_number_douyin_official": re.compile(r"抖音|直播|会员"),
}
ASSERTED_RISK_RE = re.compile(
    r"(?:请|需要|必须|现在|马上|立即|先|打开|点击|提供|告诉|输入|下载|安装|支付|转账|缴费|扫码|共享).{0,24}"
    r"(?:验证码|校验码|核验码|银行卡|身份证|密码|链接|页面|网页|服务页|软件|App|程序|助手|屏幕|远程|转账|汇款|付款|缴费|保证金|手续费|账户|收款码|二维码)"
)
ADDITIONAL_ASSERTED_RISK_RES = (
    re.compile(
        r"(?:请|需要|必须|现在|马上|立即|先|把|按我).{0,30}"
        r"(?:转|汇|打款|付款|支付|付|垫付|垫|充值|缴费|扫码|下载|安装|点开|打开|开启|共享).{0,30}"
        r"(?:元|块|款|钱|账户|任务账户|收款码|二维码|验证码|校验码|核验码|密码|卡密|截图|屏幕|手机画面|软件|App|程序|助手|链接|页面|给我|给这边)"
    ),
    re.compile(
        r"(?:验证码|校验码|密码|卡密|截图|身份证|银行卡|手机画面|屏幕).{0,24}"
        r"(?:告诉|提供|发给|拍给|念给|读给|报给|输入|共享给)"
    ),
    re.compile(
        r"(?:\d+(?:\.\d+)?元|钱|款项|费用|押金|保证金).{0,24}"
        r"(?:转|汇|打款|付款|支付)"
    ),
    re.compile(
        r"(?:页面|网页|手机画面|屏幕|窗口).{0,24}"
        r"(?:共享|截屏|截图|发给|拍给).{0,12}(?:我|这边|客服|工作人员)?"
    ),
    re.compile(
        r"(?:验证码|校验码|核验码|密码).{0,24}"
        r"(?:需要|由|让).{0,12}(?:客服|工作人员|我|这边).{0,12}"
        r"(?:录入|登记|核验|确认)"
    ),
    re.compile(
        r"(?:收款信息|收款账户|收款账号|收款码).{0,24}"
        r"(?:转|汇|打款|付款|支付)"
    ),
)
NORMAL_UNSAFE_RE = re.compile(
    r"(?:转到|汇入|支付到|打到)(?:个人|指定|安全|陌生)账户|"
    r"(?:告诉|提供|报出)(?:我|这边|工作人员).{0,8}(?:验证码|密码)|"
    r"(?:下载|安装)(?:陌生|指定|非官方).{0,8}(?:软件|App)|"
    r"(?:开启|进行)(?:屏幕共享|远程控制)"
)
PHONE_OR_URL_RE = re.compile(r"https?://|www\.|\b1[3-9]\d{9}\b")
MAX_REPEATED_CLAUSE_RATIO = 0.10
NEGATION_RE = re.compile(r"(?:不|不会|无需|无须|禁止|请勿|切勿|不得|不能|不应|不需要)")
NEGATION_REVERSAL_RE = re.compile(r"(?:但|但是|不过|而是|改为|只需|只要|实际)")
INTERROGATIVE_NEGATION_RE = re.compile(r"(?:能不能|可不可以|要不要|是不是)$")
WARNING_CUE_RE = re.compile(r"(?:如果|若|一旦|有人|遇到|任何要求|陌生人|骗子|诈骗分子)")
WARNING_OUTCOME_RE = re.compile(
    r"(?:不属于|不要|请勿|风险|诈骗|挂断|报警|核实|终止|停止|停下来)"
)
RISK_INQUIRY_RE = re.compile(
    r"(?:确认|核实|询问|了解|回访).{0,24}(?:是否|有无|有没有|发生过|曾经)"
)
REPORTED_RISK_RE = re.compile(
    r"(?:曾经|曾|此前|当时|之前|举报|反映|描述|提到|遇到).{0,40}"
    r"(?:对方|陌生人|骗子|诈骗分子).{0,24}"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def collect_results(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.rglob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def render_turns(raw_turns: Any) -> tuple[list[dict[str, str]], str, str, list[str]]:
    errors: list[str] = []
    if not isinstance(raw_turns, list) or len(raw_turns) < 8:
        return [], "", "", ["turn_count_below_8"]
    turns: list[dict[str, str]] = []
    for index, raw in enumerate(raw_turns):
        if not isinstance(raw, Mapping):
            errors.append(f"turn_{index}_not_object")
            continue
        speaker = str(raw.get("speaker") or "").strip().lower()
        text = str(raw.get("text") or "").strip()
        if speaker not in {"caller", "callee"}:
            errors.append(f"turn_{index}_invalid_speaker")
        if not text:
            errors.append(f"turn_{index}_empty")
        if turns and speaker in {"caller", "callee"} and speaker == turns[-1]["speaker"]:
            turns[-1]["text"] = f"{turns[-1]['text']}\n{text}"
            continue
        turns.append({"speaker": speaker, "text": text})
    if len(turns) < 8:
        errors.append("turn_count_below_8")
    marked = "\n".join(
        ("来电人：" if turn["speaker"] == "caller" else "接电人：") + turn["text"]
        for turn in turns
    )
    model_input = "\n".join(turn["text"] for turn in turns)
    return turns, marked, model_input, errors


def position_bin(start: int, total: int) -> str:
    ratio = start / max(total, 1)
    if ratio < 0.25:
        return "Q1"
    if ratio < 0.50:
        return "Q2"
    if ratio < 0.75:
        return "Q3"
    return "Q4"


def repeated_turn_ratio(turns: list[dict[str, str]]) -> float:
    normalized = [re.sub(r"\W+", "", turn["text"]) for turn in turns]
    return 1.0 - len(set(normalized)) / max(len(normalized), 1)


def near_duplicate_turn_ratio(turns: list[dict[str, str]]) -> float:
    """统计同一说话人机械改写既有话轮的比例。"""

    normalized = [re.sub(r"\W+", "", turn["text"]) for turn in turns]
    repeated = 0
    for index, text in enumerate(normalized):
        if len(text) < 10:
            continue
        previous = [
            normalized[prior]
            for prior in range(index)
            if turns[prior]["speaker"] == turns[index]["speaker"]
            and len(normalized[prior]) >= 10
        ]
        if previous and max(
            SequenceMatcher(None, text, candidate).ratio()
            for candidate in previous
        ) >= 0.78:
            repeated += 1
    return repeated / max(len(turns), 1)


def repeated_clause_ratio(turns: list[dict[str, str]]) -> float:
    """统计跨话轮重复的实质性分句，忽略单个话轮内部的修辞复现。"""

    clause_turns: dict[str, set[int]] = defaultdict(set)
    total = 0
    for turn_index, turn in enumerate(turns):
        seen_in_turn: set[str] = set()
        for clause in re.split(r"[，。！？；,.!?;]", turn["text"]):
            normalized = re.sub(r"\W+", "", clause)
            if len(normalized) < 6 or normalized in seen_in_turn:
                continue
            seen_in_turn.add(normalized)
            clause_turns[normalized].add(turn_index)
            total += 1
    repeated = sum(
        len(turn_indices) - 1
        for turn_indices in clause_turns.values()
        if len(turn_indices) > 1
    )
    return repeated / max(total, 1)


def _risk_match_is_negated_or_warning(text: str, start: int, end: int) -> bool:
    sentence_start = max(
        (text.rfind(mark, 0, start) for mark in "。！？；\n"), default=-1
    ) + 1
    sentence_ends = [text.find(mark, end) for mark in "。！？；\n"]
    sentence_end = min((value for value in sentence_ends if value >= 0), default=len(text))
    sentence = text[sentence_start:sentence_end]
    prefix = text[sentence_start:start]
    negations = list(NEGATION_RE.finditer(prefix))
    if negations and INTERROGATIVE_NEGATION_RE.search(prefix[: negations[-1].end() + 1]):
        negations.pop()
    if negations:
        tail = prefix[negations[-1].end() :]
        if (
            "，" not in tail
            and "," not in tail
            and len(re.sub(r"\W+", "", tail)) <= 30
            and not NEGATION_REVERSAL_RE.search(tail)
        ):
            return True
    if WARNING_CUE_RE.search(prefix) and WARNING_OUTCOME_RE.search(sentence):
        return True
    if RISK_INQUIRY_RE.search(prefix):
        return True
    return bool(REPORTED_RISK_RE.search(prefix))


def caller_asserts_risk_action(caller_text: str) -> bool:
    for pattern in (ASSERTED_RISK_RE, *ADDITIONAL_ASSERTED_RISK_RES):
        for match in pattern.finditer(caller_text):
            if not _risk_match_is_negated_or_warning(
                caller_text, match.start(), match.end()
            ):
                return True
    return False


def normal_caller_advances_unsafe_action(caller_text: str) -> bool:
    return any(
        not _risk_match_is_negated_or_warning(caller_text, match.start(), match.end())
        for match in NORMAL_UNSAFE_RE.finditer(caller_text)
    )


def exact_phrase_ranges(
    phrases: list[str],
    *,
    model_input_text: str,
    caller_text: str,
    counter: BertWordPieceCounter,
    field: str,
    errors: list[str],
) -> list[list[int]]:
    ranges: list[list[int]] = []
    search_cursor = 0
    for phrase in phrases:
        char_start = model_input_text.find(phrase, search_cursor)
        if char_start < 0:
            char_start = model_input_text.find(phrase)
        if char_start < 0:
            errors.append(f"{field}_not_exact_substring")
            continue
        char_end = char_start + len(phrase)
        start, end = counter.char_span_to_token_span(
            model_input_text, char_start, char_end
        )
        ranges.append([start, end])
        search_cursor = char_end
        if phrase not in caller_text:
            errors.append(f"{field}_not_from_caller")
    return ranges


def normalize_evidence_phrases(raw_evidence: Any, field: str, errors: list[str]) -> list[str]:
    if not isinstance(raw_evidence, list):
        errors.append(f"{field}_not_list")
        return []
    phrases = []
    for value in raw_evidence:
        phrase = re.sub(
            r"^\s*(?:来电人|caller)\s*[：:]\s*", "", str(value).strip(), flags=re.I
        ).strip().strip("“”\"")
        if phrase:
            phrases.append(phrase)
    return phrases


def turn_token_ranges(
    turns: list[dict[str, str]], counter: BertWordPieceCounter
) -> list[tuple[int, int, int]]:
    texts: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    for index, turn in enumerate(turns):
        start = counter.count("\n".join(texts)) if texts else 0
        texts.append(turn["text"])
        end = counter.count("\n".join(texts))
        if end > start:
            ranges.append((start, end, index))
    return ranges


def spans_share_window(
    left: list[int], right: list[int], windows: list[tuple[int, int]]
) -> bool:
    return any(
        begin <= left[0]
        and left[1] <= end
        and begin <= right[0]
        and right[1] <= end
        for begin, end in windows
    )


def validate_result(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    counter: BertWordPieceCounter,
) -> dict[str, Any]:
    turns, marked_text, model_input_text, errors = render_turns(result.get("turns"))
    risk_evidence = normalize_evidence_phrases(
        result.get("risk_evidence"), "risk_evidence", errors
    )
    context_evidence = normalize_evidence_phrases(
        result.get("context_evidence"), "context_evidence", errors
    )
    wordpieces = counter.count(model_input_text)
    low = int(task["target_wordpiece_min"])
    high = int(task["target_wordpiece_max"])
    if not low <= wordpieces <= high:
        errors.append(f"wordpiece_out_of_range:{wordpieces}")
    if PHONE_OR_URL_RE.search(model_input_text):
        errors.append("phone_or_url_present")
    exact_repetition = repeated_turn_ratio(turns)
    near_repetition = near_duplicate_turn_ratio(turns)
    clause_repetition = repeated_clause_ratio(turns)
    if exact_repetition > 0.15:
        errors.append("repeated_turn_ratio_high")
    if near_repetition > 0.15:
        errors.append("near_duplicate_turn_ratio_high")
    if clause_repetition > MAX_REPEATED_CLAUSE_RATIO:
        errors.append("repeated_clause_ratio_high")

    scenario = str(task["scenario_tag"])
    pattern = SCENARIO_PATTERNS.get(scenario)
    if pattern is not None and pattern.search(model_input_text) is None:
        errors.append("scenario_cue_missing")
    if scenario == "private_number_douyin_official" and "私人号码" in model_input_text:
        errors.append("private_number_metadata_spoken")

    caller_text = "\n".join(turn["text"] for turn in turns if turn["speaker"] == "caller")
    evidence_ranges = exact_phrase_ranges(
        risk_evidence,
        model_input_text=model_input_text,
        caller_text=caller_text,
        counter=counter,
        field="risk_evidence",
        errors=errors,
    )
    context_ranges = exact_phrase_ranges(
        context_evidence,
        model_input_text=model_input_text,
        caller_text=caller_text,
        counter=counter,
        field="context_evidence",
        errors=errors,
    )
    audited_turn_ranges = turn_token_ranges(turns, counter)
    default_chunk_ranges = (
        make_plain_chunk_token_ranges(
            wordpieces,
            chunk_length=320,
            overlap=64,
            special_tokens=2,
            turn_token_ranges=audited_turn_ranges,
        )
        if wordpieces > 0
        else []
    )
    if task["generation_kind"] == "fraud":
        if not risk_evidence:
            errors.append("fraud_risk_evidence_missing")
        if not caller_asserts_risk_action(caller_text):
            errors.append("caller_asserted_risk_action_missing")
        if evidence_ranges:
            actual_bin = position_bin(min(item[0] for item in evidence_ranges), wordpieces)
            if actual_bin != str(task["evidence_position_bin"]):
                errors.append(
                    f"evidence_position_mismatch:{actual_bin}"
                )
        if bool(task.get("cross_window_evidence")):
            if not context_ranges or not evidence_ranges:
                errors.append("cross_window_context_and_risk_required")
            elif spans_share_window(
                min(context_ranges, key=lambda item: item[0]),
                max(evidence_ranges, key=lambda item: item[0]),
                default_chunk_ranges,
            ):
                errors.append("cross_window_combination_visible_in_one_window")
    else:
        if context_evidence:
            errors.append("normal_context_evidence_must_be_empty")
        if risk_evidence:
            errors.append("normal_risk_evidence_must_be_empty")
        if normal_caller_advances_unsafe_action(caller_text):
            errors.append("normal_caller_advances_unsafe_action")

    contract = role_contract(marked_text)
    output = {
        **{key: value for key, value in task.items() if key != "prompt"},
        "turns": turns,
        "text": marked_text,
        "audio_text": contract["audio_text"],
        "model_input_text": contract["model_input_text"],
        "role_marker_token_policy": contract["role_marker_token_policy"],
        "context_evidence": context_evidence,
        "context_evidence_token_ranges": context_ranges,
        "risk_evidence": risk_evidence,
        "evidence_token_ranges": evidence_ranges,
        "raw_wordpiece_length": wordpieces,
        "chunk_token_ranges": [list(item) for item in default_chunk_ranges],
        "chunking_profile": {
            "input_mode": "plain_sliding_chunks",
            "chunk_length": 320,
            "overlap": 64,
            "special_tokens": 2,
        },
        "full_text_coverage": 1.0,
        "quality_metrics": {
            "exact_repeated_turn_ratio": exact_repetition,
            "near_duplicate_turn_ratio": near_repetition,
            "repeated_clause_ratio": clause_repetition,
        },
        "scenario_summary": str(result.get("scenario_summary") or "").strip(),
        "generation_notes": str(result.get("generation_notes") or "").strip(),
        "automatic_validation_errors": sorted(set(errors)),
        "automatic_validation": "READY" if not errors else "REJECT",
        "review_status": "pending_independent_review" if not errors else "rejected_automatic",
        "longdoc_generated": True,
        "source_type": "luna_subagent_from_scratch",
    }
    return output


def load_review_decisions(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    decisions: dict[str, str] = {}
    paths = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
    for decision_path in paths:
        for row in read_jsonl(decision_path):
            task_id = str(row.get("task_id") or "")
            decision = str(
                row.get("decision") or row.get("review_status") or ""
            ).lower()
            if task_id and decision in {"accept", "accepted", "reject", "rejected"}:
                decisions[task_id] = (
                    "accept" if decision.startswith("accept") else "reject"
                )
    return decisions


def select_slots(
    candidates: list[dict[str, Any]],
    decisions: Mapping[str, str],
    expected_slot_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["automatic_validation"] == "READY":
            by_slot[str(row["slot_id"])].append(row)
    selected: list[dict[str, Any]] = []
    unresolved: list[str] = []
    all_slots = sorted(set(map(str, expected_slot_ids)))
    for slot_id in all_slots:
        accepted = [
            row
            for row in by_slot.get(slot_id, [])
            if decisions.get(str(row["task_id"])) == "accept"
        ]
        if not accepted:
            unresolved.append(slot_id)
            continue
        chosen = min(accepted, key=lambda row: int(row["candidate_variant"]))
        chosen["review_status"] = "accepted"
        chosen["sample_id"] = f"V361_LONG_{chosen['split']}_{int(chosen['slot_index']):04d}"
        chosen["canonical_label_13"] = int(chosen["label_13"])
        chosen["canonical_subcategory_code"] = str(chosen["subcategory_code"])
        chosen["parent_target"] = [
            1.0 if index == int(chosen["label_13"]) else 0.0 for index in range(13)
        ]
        chosen["sample_weight"] = 1.0
        chosen["parent_loss_weight"] = 1.0
        chosen["child_loss_weight"] = 1.0
        selected.append(chosen)
    return selected, unresolved


def donor_priority(row: Mapping[str, Any], child_counts: Counter[str]) -> tuple[Any, ...]:
    code = str(row.get("canonical_subcategory_code") or row.get("subcategory_code") or "")
    rare = child_counts.get(code, 0) < 50
    audited = bool(row.get("v329_audit_status") or row.get("v332_consensus_decision"))
    duplicate = bool(row.get("duplicate_redundant") or row.get("v332_duplicate_redundant"))
    weight = float(row.get("sample_weight", 1.0))
    return (rare, not duplicate, audited, weight, str(row.get("sample_id") or ""))


def build_train6500(
    source_rows: list[dict[str, Any]], long_train: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_parent = Counter(
        int(row.get("canonical_label_13", row.get("label_13", 0)))
        for row in source_rows
    )
    generated_parent = Counter(int(row["label_13"]) for row in long_train)
    child_counts = Counter(
        str(row.get("canonical_subcategory_code") or row.get("subcategory_code") or "")
        for row in source_rows
    )
    donors: list[dict[str, Any]] = []
    for label, required in sorted(generated_parent.items()):
        pool = [
            row
            for row in source_rows
            if int(row.get("canonical_label_13", row.get("label_13", 0))) == label
        ]
        pool.sort(key=lambda row: donor_priority(row, child_counts))
        eligible = [row for row in pool if not donor_priority(row, child_counts)[0]]
        if len(eligible) < required:
            raise ValueError(f"parent {label}: only {len(eligible)} safe donors for {required}")
        donors.extend(eligible[:required])
    donor_ids = {str(row["sample_id"]) for row in donors}
    remaining = [row for row in source_rows if str(row["sample_id"]) not in donor_ids]
    output = remaining + sorted(long_train, key=lambda row: str(row["sample_id"]))
    if len(output) != 6500 or len(donors) != 620:
        raise AssertionError(
            f"train replacement mismatch: rows={len(output)}, donors={len(donors)}"
        )
    output_parent = Counter(
        int(row.get("canonical_label_13", row.get("label_13", 0))) for row in output
    )
    if output_parent != source_parent:
        raise AssertionError("13-class distribution changed after replacement")
    return output, donors


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "split": dict(sorted(Counter(str(row.get("split")) for row in rows).items())),
        "label_13": dict(sorted(Counter(str(row.get("label_13")) for row in rows).items())),
        "subcategory_code": dict(sorted(Counter(str(row.get("subcategory_code")) for row in rows).items())),
        "length_bucket": dict(sorted(Counter(str(row.get("length_bucket")) for row in rows).items())),
        "scenario_tag": dict(sorted(Counter(str(row.get("scenario_tag")) for row in rows).items())),
        "evidence_position_bin": dict(sorted(Counter(str(row.get("evidence_position_bin")) for row in rows).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--source-train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    tasks = read_jsonl(args.task_dir / "all_tasks_1225.jsonl")
    task_by_id = {str(row["task_id"]): row for row in tasks}
    raw_results = collect_results(args.result_dir) if args.result_dir.exists() else []
    counter = BertWordPieceCounter(args.vocab)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    unknown_results: list[str] = []
    for result in raw_results:
        task_id = str(result.get("task_id") or "")
        if task_id not in task_by_id:
            unknown_results.append(task_id)
            continue
        if task_id in seen:
            raise ValueError(f"duplicate generated result: {task_id}")
        seen.add(task_id)
        candidates.append(validate_result(task_by_id[task_id], result, counter))

    decisions = load_review_decisions(args.review_decisions)
    selected, unresolved = select_slots(
        candidates,
        decisions,
        (str(task["slot_id"]) for task in tasks),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "audited_candidates.jsonl", candidates)
    write_jsonl(
        args.output_dir / "automatic_ready.jsonl",
        [row for row in candidates if row["automatic_validation"] == "READY"],
    )
    write_json(args.output_dir / "unresolved_slots.json", unresolved)

    complete = len(selected) == 980 and not unresolved
    if args.require_complete and not complete:
        raise SystemExit(
            f"dataset incomplete: selected={len(selected)}, unresolved={len(unresolved)}"
        )
    manifest: dict[str, Any] = {
        "schema_version": "v361.longdoc_finalize_manifest.v1",
        "generated_results": len(raw_results),
        "validated_candidates": len(candidates),
        "automatic_ready": sum(row["automatic_validation"] == "READY" for row in candidates),
        "review_decisions": len(decisions),
        "selected_slots": len(selected),
        "unresolved_slots": len(unresolved),
        "unknown_result_task_ids": unknown_results,
        "complete": complete,
    }
    if complete:
        long_train = [row for row in selected if row["split"] == "train"]
        long_dev = [row for row in selected if row["split"] == "dev"]
        long_stress = [row for row in selected if row["split"] == "stress"]
        source_train = read_jsonl(args.source_train)
        train6500, donors = build_train6500(source_train, long_train)
        write_jsonl(args.output_dir / "train_v361_longdoc_6500.jsonl", train6500)
        write_jsonl(args.output_dir / "long_dev_v361_240.jsonl", long_dev)
        write_jsonl(args.output_dir / "long_stress_v361_120.jsonl", long_stress)
        write_jsonl(args.output_dir / "replaced_donors_620.jsonl", donors)
        manifest["datasets"] = {
            "train": summarize(train6500),
            "long_train": summarize(long_train),
            "long_dev": summarize(long_dev),
            "long_stress": summarize(long_stress),
            "donors": {"rows": len(donors)},
        }
    write_json(args.output_dir / "finalize_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
