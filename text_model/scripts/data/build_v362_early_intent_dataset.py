#!/usr/bin/env python3
"""构建 V362 诈骗前期意图数据集。

把 979 条严格盲标标注（外部标注包）按 ``sample_id`` 接到 V361 长文本原始通话上，
切出 ``prefix_end_turn`` 之前的前缀，生成可训练 / 可评测的前缀数据集。

设计约束（对应任务书第八节）：

* 只使用 ``turns[0:prefix_end_turn]``，轮次编号从 1 开始，不多取一轮；
* 证据、``tE_turn`` 只用于监督与解释，不拼进 ``prefix_text``；
* 隔离样本不进入任何训练集或主指标，隔离原因一律从 ``_review_summary.json`` 读取，
  不在本脚本里手写，避免与复核结论不一致；
* ``V361_LONG_train_0256`` 无法构造严格盲前缀，进入 ``missing_samples.jsonl``，
  不猜测其前缀位置与标签；
* 不使用随机数，重复运行结果逐字节一致。

本脚本不含任何服务器地址、用户名或口令。

产出（默认 ``data/processed/v362_early_intent/``）::

    prefix_train.jsonl / prefix_dev.jsonl / prefix_stress.jsonl
    quarantine_samples.jsonl
    missing_samples.jsonl
    build_manifest.json
    dataset_statistics.json
    validation_report.json
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "data/processed/v361_longdoc_text_route_20260824/generated_long_all_v361_980.jsonl"
)
DEFAULT_ANNOTATIONS = ROOT / "data/external/v362_gpt_data"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/v362_early_intent"
DEFAULT_VOCAB = (
    ROOT
    / "runs/r_260727_042528_V341D1_Policy6500_MegatronBERT_DataCausal_DevOnly_s202_auto_a1"
    / "subruns/seed-202/best_model/vocab.txt"
)

SPLITS = ("long_train", "long_dev", "long_stress")
SPLIT_TO_SOURCE_SPLIT = {
    "train": "long_train",
    "dev": "long_dev",
    "stress": "long_stress",
}
DECISIONS = ("normal", "ambiguous", "early_fraud", "explicit_fraud")

STAGE_TARGET = {"normal": 0, "ambiguous": 1, "early_fraud": 2, "explicit_fraud": 3}

GENERATION_KIND_TO_BINARY = {"fraud": 1, "normal": 0}

OPERATION_KEYS = (
    "transfer",
    "otp",
    "download",
    "screen_share",
    "remote_control",
    "bank_operation",
    "unofficial_link",
)

# 980 条原始数据中唯一无法构造严格盲前缀的样本：首轮即出现明确风险操作。
MISSING_SAMPLE_ID = "V361_LONG_train_0256"

EVIDENCE_BUCKETS = ("core_evidence", "supporting_evidence")

# 需人工复核的样本不参与 stage（四分类）损失：未经确认不得直接训练四分类头。
REVIEW_REQUIRED_STAGE_MASK = 0

PREFIX_TURN_BUCKETS = (
    ("turns_01_05", 1, 5),
    ("turns_06_10", 6, 10),
    ("turns_11_15", 11, 15),
    ("turns_16_20", 16, 20),
    ("turns_21_30", 21, 30),
    ("turns_31_45", 31, 45),
    ("turns_46_plus", 46, 10 ** 6),
)
PREFIX_WORDPIECE_BUCKETS = (
    ("wp_000_128", 0, 128),
    ("wp_129_256", 129, 256),
    ("wp_257_385", 257, 385),
    ("wp_386_512", 386, 512),
    ("wp_513_1024", 513, 1024),
    ("wp_1025_1600", 1025, 1600),
    ("wp_1601_plus", 1601, 10 ** 9),
)


class BuildError(RuntimeError):
    """数据构建的前置条件不成立。"""


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def normalize_text(value: str) -> str:
    """标准化文本：NFKC 归一 + 去除全部空白。用于哈希与逐字引用比对。"""

    return "".join(unicodedata.normalize("NFKC", str(value or "")).split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def count_nonblank_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BuildError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_source(path: Path) -> "OrderedDict[str, dict[str, Any]]":
    ordered: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for row in load_jsonl(path):
        sample_id = row["sample_id"]
        if sample_id in ordered:
            raise BuildError(f"{path} 出现重复 sample_id: {sample_id}")
        ordered[sample_id] = row
    return ordered


def load_annotations(directory: Path) -> list[dict[str, Any]]:
    """按文件名顺序读取全部 annotations_*.jsonl，保持并记录来源批次。"""

    files = sorted(directory.glob("annotations_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{directory} 下没有 annotations_*.jsonl")
    rows: list[dict[str, Any]] = []
    for path in files:
        for row in load_jsonl(path):
            row = dict(row)
            row["_annotation_file"] = path.name
            rows.append(row)
    return rows


def load_quarantine_reasons(directory: Path) -> "OrderedDict[str, str]":
    """从 _review_summary.json 读取最终复核原因，不在脚本里手写。"""

    path = directory / "_review_summary.json"
    if not path.exists():
        raise BuildError(f"缺少复核汇总文件: {path}")
    payload = load_json(path)
    entries = payload.get("quarantine")
    if not isinstance(entries, list) or not entries:
        raise BuildError(f"{path} 未包含有效的 quarantine 列表")

    reasons: "OrderedDict[str, str]" = OrderedDict()
    for entry in entries:
        sample_id = entry.get("sample_id")
        reason = (entry.get("reason") or "").strip()
        if not sample_id or not reason:
            raise BuildError(f"{path} 的 quarantine 条目缺少 sample_id 或 reason: {entry}")
        if sample_id in reasons:
            raise BuildError(f"{path} 的 quarantine 出现重复 sample_id: {sample_id}")
        reasons[sample_id] = reason
    return reasons


def load_wordpiece_counter(vocab_path: Path):
    """尝试复用项目自带的 WordPiece 审计器；不可用时返回 None。"""

    if not vocab_path.exists():
        return None
    try:
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        try:
            from text_model.data.wordpiece_counter import BertWordPieceCounter
        except ModuleNotFoundError:
            from data.wordpiece_counter import BertWordPieceCounter  # type: ignore

        return BertWordPieceCounter(vocab_path)
    except Exception:
        return None


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "mean": round(sum(ordered) / len(ordered), 2),
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def bucket_of(value: int, buckets: Sequence[tuple[str, int, int]]) -> str:
    for name, low, high in buckets:
        if low <= value <= high:
            return name
    return "out_of_range"


# --------------------------------------------------------------------------- #
# 前置检查（构建前必须通过，否则直接失败）
# --------------------------------------------------------------------------- #
def preflight_annotation_ids(annotations: Sequence[Mapping[str, Any]]) -> None:
    """标注内的 sample_id 必须唯一。发现重复立即失败，不得静默覆盖。"""

    counts = Counter(row.get("sample_id") for row in annotations)
    duplicates = sorted(
        sample_id
        for sample_id, count in counts.items()
        if count > 1 or sample_id in (None, "")
    )
    if duplicates:
        detail = {
            sample_id: sorted(
                row["_annotation_file"]
                for row in annotations
                if row.get("sample_id") == sample_id
            )
            for sample_id in duplicates
        }
        raise BuildError(
            "标注中存在重复或缺失的 sample_id，已中止构建（不允许字典静默覆盖）: "
            f"{json.dumps(detail, ensure_ascii=False)}"
        )


def preflight_split_consistency(
    source: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> None:
    """标注的 source_split 必须与原始数据 split 一致。"""

    mismatches = []
    for annotation in annotations:
        sample_id = annotation["sample_id"]
        source_row = source.get(sample_id)
        if source_row is None:
            continue
        expected = SPLIT_TO_SOURCE_SPLIT.get(source_row.get("split"), source_row.get("split"))
        if annotation.get("source_split") != expected:
            mismatches.append(
                f"{sample_id}: annotation.source_split={annotation.get('source_split')!r} "
                f"!= source.split={source_row.get('split')!r} (期望 {expected!r})"
            )
    if mismatches:
        raise BuildError(
            "标注 source_split 与原始数据 split 不一致，已中止构建: "
            + "; ".join(mismatches[:10])
        )


def preflight_missing_set(
    source: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
) -> set:
    """未被标注的 sample_id 必须恰好是已知缺失样本。"""

    annotated = {row["sample_id"] for row in annotations}
    unannotated = set(source) - annotated
    if unannotated != {MISSING_SAMPLE_ID}:
        raise BuildError(
            "未标注 sample_id 集合与预期不符，已中止构建。"
            f"期望 {{{MISSING_SAMPLE_ID}}}，实际 {sorted(unannotated)}"
        )
    return unannotated


# --------------------------------------------------------------------------- #
# 前缀记录构建
# --------------------------------------------------------------------------- #
def expected_policy(decision: str, full_binary: int, pre_evidence_weight: float):
    """返回 (prefix_relation, binary_target, binary_loss_mask, sample_weight)。"""

    if decision == "normal" and full_binary == 0:
        return ("clean_normal", 0, 1, 1.0)
    if decision == "normal" and full_binary == 1:
        return ("pre_evidence_or_not_yet_detectable", 0, 1, float(pre_evidence_weight))
    if decision == "ambiguous":
        return ("ambiguous_prefix", -1, 0, 0.0)
    if decision == "early_fraud":
        return ("early_fraud_prefix", 1, 1, 1.0)
    if decision == "explicit_fraud":
        return ("explicit_operation_prefix", 1, 1, 1.0)
    raise BuildError(f"未定义的 decision / 完整标签组合: {decision!r} / {full_binary!r}")


def build_prefix_payload(
    source_row: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    counter: Any,
    pre_evidence_weight: float,
) -> dict[str, Any]:
    """把一条标注接到原始通话上，切出前缀并组装训练用字段。"""

    sample_id = annotation["sample_id"]
    prefix_end_turn = annotation["prefix_end_turn"]
    turns = source_row["turns"]

    if not isinstance(prefix_end_turn, int) or isinstance(prefix_end_turn, bool):
        raise BuildError(f"{sample_id}: prefix_end_turn 不是整数")
    if not 1 <= prefix_end_turn <= len(turns):
        raise BuildError(
            f"{sample_id}: prefix_end_turn={prefix_end_turn} 越界（共 {len(turns)} 轮）"
        )

    visible = turns[:prefix_end_turn]
    prefix_turns = [
        {"turn_id": index + 1, "speaker": turn["speaker"], "text": turn["text"]}
        for index, turn in enumerate(visible)
    ]
    # 与项目 model_input_text 约定一致：不带“来电人：/接电人：”角色标识。
    prefix_text = "\n".join(turn["text"] for turn in visible)

    decision = annotation["decision"]
    generation_kind = source_row.get("generation_kind")
    full_binary = GENERATION_KIND_TO_BINARY[generation_kind]

    relation, binary_target, binary_loss_mask, sample_weight = expected_policy(
        decision, full_binary, pre_evidence_weight
    )
    review_required = annotation.get("review_required")
    stage_loss_mask = REVIEW_REQUIRED_STAGE_MASK if review_required else 1

    wordpiece_length = counter.count(prefix_text) if counter is not None else None

    return {
        "sample_id": sample_id,
        "task_id": annotation.get("task_id"),
        "source_split": annotation["source_split"],
        "prefix_end_turn": prefix_end_turn,
        "prefix_turns": prefix_turns,
        "prefix_text": prefix_text,
        "decision": decision,
        "stage_target": STAGE_TARGET[decision],
        "binary_target": binary_target,
        "binary_loss_mask": binary_loss_mask,
        "stage_loss_mask": stage_loss_mask,
        "sample_weight": sample_weight,
        "tE_turn": annotation.get("tE_turn"),
        "early_detectable": annotation.get("early_detectable"),
        "core_evidence": annotation.get("core_evidence") or [],
        "supporting_evidence": annotation.get("supporting_evidence") or [],
        "explicit_operation_present": annotation.get("explicit_operation_present") or {},
        "original_full_binary_label": full_binary,
        "prefix_relation": relation,
        # --- 审计与统计用附加字段 ---
        "scenario_tag": source_row.get("scenario_tag"),
        "claimed_identity_type": source_row.get("claimed_identity_type"),
        "label_13": source_row.get("label_13"),
        "subcategory_code": source_row.get("subcategory_code"),
        "generation_kind": generation_kind,
        "confidence": annotation.get("confidence"),
        "review_required": review_required,
        "future_content_used": annotation.get("future_content_used"),
        "insufficient_reason": annotation.get("insufficient_reason"),
        "annotation_file": annotation.get("_annotation_file"),
        "n_prefix_turns": len(prefix_turns),
        "n_full_turns": len(turns),
        "prefix_char_length": len(prefix_text),
        "prefix_wordpiece_length": wordpiece_length,
        "normalized_text_hash": normalized_text_hash(source_row["text"]),
        "prefix_text_hash": normalized_text_hash(prefix_text),
    }


def build_quarantine_row(
    source_row: Mapping[str, Any],
    annotation: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    sample_id = source_row["sample_id"]
    return {
        "sample_id": sample_id,
        "source_split": SPLIT_TO_SOURCE_SPLIT.get(
            source_row.get("split"), source_row.get("split")
        ),
        "source_split_raw": source_row.get("split"),
        "scenario_tag": source_row.get("scenario_tag"),
        "generation_kind": source_row.get("generation_kind"),
        "original_full_binary_label": GENERATION_KIND_TO_BINARY.get(
            source_row.get("generation_kind")
        ),
        "quarantine_reason": reason,
        "quarantine_reason_source": "_review_summary.json",
        "annotation_decision": annotation.get("decision") if annotation else None,
        "annotation_tE_turn": annotation.get("tE_turn") if annotation else None,
        "annotation_prefix_end_turn": (
            annotation.get("prefix_end_turn") if annotation else None
        ),
        "enters_training": False,
        "enters_main_metrics": False,
        "usable_for": "污染溯源与生成质量分析",
    }


def build_missing_row(source_row: Mapping[str, Any] | None) -> dict[str, Any]:
    """缺失样本记录。明确不猜测前缀位置与标签。"""

    row: dict[str, Any] = {
        "sample_id": MISSING_SAMPLE_ID,
        "source_split": "long_train",
        "in_source_980": source_row is not None,
        "has_annotation": False,
        "prefix_end_turn": None,
        "decision": None,
        "tE_turn": None,
        "exclusion_reason": (
            "首轮即出现明确风险操作，无法构造严格盲前缀，"
            "因此未生成盲标任务；前缀位置与标签均不可恢复。"
        ),
        "label_guessed": False,
        "action": "写入排除清单，不进入训练、开发、压力集与主指标",
    }
    if source_row is not None:
        turns = source_row.get("turns") or []
        row.update(
            {
                "scenario_tag": source_row.get("scenario_tag"),
                "generation_kind": source_row.get("generation_kind"),
                "n_full_turns": len(turns),
                "first_turn_speaker": turns[0]["speaker"] if turns else None,
                "first_turn_text": turns[0]["text"] if turns else None,
                "source_risk_evidence": source_row.get("risk_evidence"),
                "source_context_evidence": source_row.get("context_evidence"),
            }
        )
    return row


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #
class Validator:
    def __init__(self) -> None:
        self.checks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def check(
        self, name: str, failures: Sequence[Any], description: str, max_examples: int = 5
    ) -> None:
        self.checks[name] = {
            "description": description,
            "passed": len(failures) == 0,
            "failure_count": len(failures),
            "examples": [str(item) for item in list(failures)[:max_examples]],
        }

    def report(self) -> dict[str, Any]:
        failed = [name for name, item in self.checks.items() if not item["passed"]]
        return {
            "all_passed": not failed,
            "failed_checks": failed,
            "n_checks": len(self.checks),
            "checks": self.checks,
        }


def run_validations(
    *,
    source: Mapping[str, Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
    quarantine_reasons: Mapping[str, str],
    pre_evidence_weight: float,
) -> dict[str, Any]:
    validator = Validator()

    # 1. 覆盖：所有标注都能按 sample_id 定位。
    unmatched = [a["sample_id"] for a in annotations if a["sample_id"] not in source]
    validator.check(
        "annotation_sample_ids_resolve",
        unmatched,
        "979 条标注全部能在 980 条原始数据中按 sample_id 定位。",
    )

    # 2. 标注 sample_id 唯一。
    duplicates = [
        sample_id
        for sample_id, count in Counter(a["sample_id"] for a in annotations).items()
        if count > 1
    ]
    validator.check(
        "annotation_sample_id_unique", duplicates, "标注内 sample_id 唯一，无静默覆盖。"
    )

    # 3. 标注 source_split 与原始 split 一致。
    split_mismatch = []
    for annotation in annotations:
        source_row = source.get(annotation["sample_id"])
        if source_row is None:
            continue
        expected = SPLIT_TO_SOURCE_SPLIT.get(
            source_row.get("split"), source_row.get("split")
        )
        if annotation.get("source_split") != expected:
            split_mismatch.append(
                f"{annotation['sample_id']}: {annotation.get('source_split')} != {expected}"
            )
    validator.check(
        "annotation_split_matches_source",
        split_mismatch,
        "每条标注的 source_split 与原始数据 split 一致。",
    )

    # 4. 未标注 ID 集合恰好等于已知缺失样本。
    unannotated = sorted(set(source) - {a["sample_id"] for a in annotations})
    validator.check(
        "unannotated_equals_known_missing",
        [item for item in unannotated if item != MISSING_SAMPLE_ID],
        f"未标注 ID 集合恰好等于 {{{MISSING_SAMPLE_ID}}}。",
    )
    validator.check(
        "known_missing_present",
        [] if MISSING_SAMPLE_ID in unannotated else [f"{MISSING_SAMPLE_ID} 已被标注"],
        "已知缺失样本确实未出现在标注中。",
    )
    validator.check(
        "missing_sample_recorded",
        []
        if [row["sample_id"] for row in missing_rows] == [MISSING_SAMPLE_ID]
        else [f"missing_samples.jsonl 内容异常: {[r['sample_id'] for r in missing_rows]}"],
        "缺失样本写入 missing_samples.jsonl 且不猜测前缀与标签。",
    )

    # 5. 前缀边界。
    boundary_failures = []
    for row in prefix_rows:
        if len(row["prefix_turns"]) != row["prefix_end_turn"]:
            boundary_failures.append(
                f"{row['sample_id']}: prefix_turns={len(row['prefix_turns'])} "
                f"!= prefix_end_turn={row['prefix_end_turn']}"
            )
        ids = [turn["turn_id"] for turn in row["prefix_turns"]]
        if ids != list(range(1, row["prefix_end_turn"] + 1)):
            boundary_failures.append(f"{row['sample_id']}: 轮次编号不是 1..prefix_end_turn")
        if row["prefix_end_turn"] > row["n_full_turns"]:
            boundary_failures.append(f"{row['sample_id']}: prefix_end_turn 超过完整通话轮数")
    validator.check(
        "prefix_boundary_exact",
        boundary_failures,
        "每条前缀的最后轮次与 prefix_end_turn 一致，轮次编号从 1 开始，不多取一轮。",
    )

    # 6. 证据轮次不超过 prefix_end_turn。
    evidence_over = []
    for row in prefix_rows:
        for bucket in EVIDENCE_BUCKETS:
            for evidence in row[bucket]:
                turn_id = evidence.get("turn_id")
                if not isinstance(turn_id, int) or not 1 <= turn_id <= row["prefix_end_turn"]:
                    evidence_over.append(
                        f"{row['sample_id']}:{bucket}: turn_id={turn_id} "
                        f"> prefix_end_turn={row['prefix_end_turn']}"
                    )
    validator.check(
        "evidence_turn_within_prefix",
        evidence_over,
        "所有证据轮次均不超过 prefix_end_turn。",
    )

    # 7. 证据原文逐字（标准化后）命中。
    quote_failures = []
    for row in prefix_rows:
        turns = {turn["turn_id"]: normalize_text(turn["text"]) for turn in row["prefix_turns"]}
        for bucket in EVIDENCE_BUCKETS:
            for evidence in row[bucket]:
                turn_id = evidence.get("turn_id")
                text = turns.get(turn_id)
                if text is None:
                    continue
                if normalize_text(evidence.get("quote") or "") not in text:
                    quote_failures.append(
                        f"{row['sample_id']}:{bucket}: turn {turn_id} 未命中 quote"
                    )
    validator.check(
        "evidence_quote_verbatim",
        quote_failures,
        "证据原文能在对应可见轮次中逐字（标准化后）匹配。",
    )

    # 8. tE_turn 不超过 prefix_end_turn。
    te_over = []
    for row in prefix_rows:
        te = row["tE_turn"]
        if te is None:
            continue
        if not isinstance(te, int) or isinstance(te, bool):
            te_over.append(f"{row['sample_id']}: tE_turn 非整数 ({te!r})")
        elif not 1 <= te <= row["prefix_end_turn"]:
            te_over.append(
                f"{row['sample_id']}: tE_turn={te} > prefix_end_turn={row['prefix_end_turn']}"
            )
    validator.check("tE_within_prefix", te_over, "tE_turn 不超过 prefix_end_turn。")

    # 9. early_fraud / explicit_fraud 必须有有效 tE_turn。
    validator.check(
        "fraud_decisions_have_tE",
        [
            f"{row['sample_id']}: decision={row['decision']} 缺少有效 tE_turn"
            for row in prefix_rows
            if row["decision"] in ("early_fraud", "explicit_fraud")
            and (
                not isinstance(row["tE_turn"], int)
                or isinstance(row["tE_turn"], bool)
                or row["tE_turn"] < 1
            )
        ],
        "early_fraud 与 explicit_fraud 必须有有效 tE_turn。",
    )

    # 10. normal / ambiguous 通常不应有 tE_turn。
    validator.check(
        "non_fraud_decisions_without_tE",
        [
            f"{row['sample_id']}: decision={row['decision']} 却带 tE_turn"
            for row in prefix_rows
            if row["decision"] in ("normal", "ambiguous") and row["tE_turn"] is not None
        ],
        "normal 与 ambiguous 的 tE_turn 必须为空。",
    )
    validator.check(
        "early_detectable_flag_consistent",
        [
            row["sample_id"]
            for row in prefix_rows
            if bool(row["early_detectable"]) != bool(row["tE_turn"] is not None)
        ],
        "early_detectable 与 tE_turn 是否为空保持一致。",
    )

    # 11. explicit_fraud 操作证据。
    validator.check(
        "explicit_fraud_has_operation",
        [
            f"{row['sample_id']}: explicit_fraud 但无操作位为真"
            for row in prefix_rows
            if row["decision"] == "explicit_fraud"
            and not any(
                bool((row["explicit_operation_present"] or {}).get(key))
                for key in OPERATION_KEYS
            )
        ],
        "explicit_fraud 必须在当前前缀中引用明确操作证据。",
    )
    validator.check(
        "operation_flags_only_for_explicit",
        [
            row["sample_id"]
            for row in prefix_rows
            if row["decision"] != "explicit_fraud"
            and any(
                bool((row["explicit_operation_present"] or {}).get(key))
                for key in OPERATION_KEYS
            )
        ],
        "非 explicit_fraud 不应标记任何明确操作位。",
    )

    # 12. future_content_used 始终为 false。
    validator.check(
        "future_content_unused",
        [row["sample_id"] for row in prefix_rows if row["future_content_used"] is not False],
        "future_content_used 必须始终为 false。",
    )

    # 13. sample_id 唯一。
    validator.check(
        "prefix_sample_id_unique",
        [
            item
            for item, count in Counter(r["sample_id"] for r in prefix_rows).items()
            if count > 1
        ],
        "前缀数据集内 sample_id 唯一。",
    )

    # 14. 训练 / 开发 / 压力集按 sample_id 无交叉。
    id_sets: dict[str, set] = defaultdict(set)
    for row in prefix_rows:
        id_sets[row["source_split"]].add(row["sample_id"])
    validator.check(
        "split_disjoint_by_sample_id",
        [
            f"{left}∩{right}: {sorted(id_sets[left] & id_sets[right])[:3]}"
            for left in SPLITS
            for right in SPLITS
            if left < right and (id_sets[left] & id_sets[right])
        ],
        "训练、开发、压力集按 sample_id 无交叉。",
    )

    # 15. 训练 / 开发 / 压力集按完整文本标准化哈希无交叉。
    full_hash_sets: dict[str, set] = defaultdict(set)
    for row in prefix_rows:
        full_hash_sets[row["source_split"]].add(row["normalized_text_hash"])
    validator.check(
        "split_disjoint_by_full_text_hash",
        [
            f"{left}∩{right}: {len(full_hash_sets[left] & full_hash_sets[right])} 组"
            for left in SPLITS
            for right in SPLITS
            if left < right and (full_hash_sets[left] & full_hash_sets[right])
        ],
        "训练、开发、压力集按完整文本标准化哈希无交叉。",
    )
    validator.check(
        "source_text_hash_unique",
        [
            hashes
            for hashes, count in Counter(
                normalized_text_hash(row["text"]) for row in source.values()
            ).items()
            if count > 1
        ],
        "原始 980 条通话的标准化文本哈希互不重复。",
    )

    # 16. 训练 / 开发 / 压力集按前缀文本标准化哈希无交叉。
    prefix_hash_sets: dict[str, set] = defaultdict(set)
    for row in prefix_rows:
        prefix_hash_sets[row["source_split"]].add(row["prefix_text_hash"])
    validator.check(
        "split_disjoint_by_prefix_text_hash",
        [
            f"{left}∩{right}: {len(prefix_hash_sets[left] & prefix_hash_sets[right])} 组"
            for left in SPLITS
            for right in SPLITS
            if left < right and (prefix_hash_sets[left] & prefix_hash_sets[right])
        ],
        "训练、开发、压力集按前缀文本标准化哈希无交叉。",
    )
    prefix_hash_owners: dict[str, list] = defaultdict(list)
    for row in prefix_rows:
        prefix_hash_owners[row["prefix_text_hash"]].append(row["sample_id"])
    validator.check(
        "prefix_text_hash_unique",
        [
            owners
            for owners in prefix_hash_owners.values()
            if len(owners) > 1
        ],
        "前缀文本标准化哈希在数据集内互不重复。",
    )

    # 17. 隔离样本不进入任何训练 / 开发 / 压力集。
    used_ids = {row["sample_id"] for row in prefix_rows}
    validator.check(
        "quarantine_excluded_from_prefix_sets",
        sorted(set(quarantine_reasons) & used_ids),
        "隔离样本不进入任何训练、开发或压力集。",
    )
    validator.check(
        "quarantine_recorded",
        []
        if sorted(row["sample_id"] for row in quarantine_rows)
        == sorted(quarantine_reasons)
        else ["quarantine_samples.jsonl 记录与 _review_summary.json 不一致"],
        "隔离样本写入 quarantine_samples.jsonl，且与复核汇总一致。",
    )
    validator.check(
        "quarantine_reason_from_summary",
        [
            row["sample_id"]
            for row in quarantine_rows
            if row["quarantine_reason"] != quarantine_reasons.get(row["sample_id"])
        ],
        "隔离原因逐字来自 _review_summary.json，未在脚本中手写。",
    )

    # 18. stage_target 映射。
    validator.check(
        "stage_target_mapping",
        [
            row["sample_id"]
            for row in prefix_rows
            if row["stage_target"] != STAGE_TARGET[row["decision"]]
        ],
        "stage_target 按 normal=0 / ambiguous=1 / early_fraud=2 / explicit_fraud=3 映射。",
    )

    # 19. binary_target / binary_loss_mask / sample_weight / prefix_relation 一致性。
    policy_failures: dict[str, list] = {
        "binary_target": [],
        "binary_loss_mask": [],
        "sample_weight": [],
        "prefix_relation": [],
    }
    for row in prefix_rows:
        relation, binary_target, loss_mask, weight = expected_policy(
            row["decision"], row["original_full_binary_label"], pre_evidence_weight
        )
        if row["prefix_relation"] != relation:
            policy_failures["prefix_relation"].append(
                f"{row['sample_id']}: {row['prefix_relation']} != {relation}"
            )
        if row["binary_target"] != binary_target:
            policy_failures["binary_target"].append(
                f"{row['sample_id']}: {row['binary_target']} != {binary_target}"
            )
        if row["binary_loss_mask"] != loss_mask:
            policy_failures["binary_loss_mask"].append(
                f"{row['sample_id']}: {row['binary_loss_mask']} != {loss_mask}"
            )
        if abs(float(row["sample_weight"]) - weight) > 1e-9:
            policy_failures["sample_weight"].append(
                f"{row['sample_id']}: {row['sample_weight']} != {weight}"
            )
    validator.check(
        "binary_target_consistent",
        policy_failures["binary_target"],
        "binary_target 与 decision + 完整标签的既定策略一致。",
    )
    validator.check(
        "binary_loss_mask_consistent",
        policy_failures["binary_loss_mask"],
        "binary_loss_mask 与既定策略一致：仅 ambiguous 不参与二分类交叉熵。",
    )
    validator.check(
        "sample_weight_consistent",
        policy_failures["sample_weight"],
        "sample_weight 与既定策略一致：clean_normal=1.0，pre_evidence=%.2f，ambiguous=0.0，诈骗=1.0。"
        % pre_evidence_weight,
    )
    validator.check(
        "prefix_relation_consistent",
        policy_failures["prefix_relation"],
        "prefix_relation 与 decision + 完整标签的既定分类一致。",
    )

    # 20. stage_loss_mask：需人工复核的样本不参与 stage 损失。
    validator.check(
        "stage_loss_mask_for_review_required",
        [
            row["sample_id"]
            for row in prefix_rows
            if bool(row["review_required"]) != (row["stage_loss_mask"] == 0)
        ],
        "review_required 样本的 stage_loss_mask 必须为 0，其余为 1。",
    )
    validator.check(
        "review_required_all_ambiguous",
        [
            f"{row['sample_id']}: decision={row['decision']}"
            for row in prefix_rows
            if row["review_required"] and row["decision"] != "ambiguous"
        ],
        "当前所有 review_required 样本均为 ambiguous（若后续变化需重新确认四分类头策略）。",
    )

    return validator.report()


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def tabulate(rows: Sequence[Mapping[str, Any]], key) -> dict[str, Any]:
    counter = Counter(key(row) for row in rows)
    return {str(name): counter[name] for name in sorted(counter, key=lambda item: str(item))}


def cross_tab(rows: Sequence[Mapping[str, Any]], left_key, right_key) -> dict[str, Any]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        table[str(left_key(row))][str(right_key(row))] += 1
    return {
        left: {right: counts[right] for right in sorted(counts)}
        for left, counts in sorted(table.items())
    }


def length_block(rows: Sequence[Mapping[str, Any]], counter_used: bool) -> dict[str, Any]:
    block: dict[str, Any] = {
        "prefix_turns": describe([row["n_prefix_turns"] for row in rows]),
        "prefix_chars": describe([row["prefix_char_length"] for row in rows]),
        "by_turn_bucket": tabulate(
            rows, lambda row: bucket_of(row["n_prefix_turns"], PREFIX_TURN_BUCKETS)
        ),
    }
    if counter_used:
        block["prefix_wordpieces"] = describe(
            [
                row["prefix_wordpiece_length"]
                for row in rows
                if row["prefix_wordpiece_length"] is not None
            ]
        )
        block["by_wordpiece_bucket"] = tabulate(
            rows,
            lambda row: bucket_of(
                row["prefix_wordpiece_length"] or 0, PREFIX_WORDPIECE_BUCKETS
            ),
        )
    else:
        block["prefix_wordpieces"] = {"available": False}
        block["by_wordpiece_bucket"] = {"available": False}
    return block


def build_statistics(
    *,
    prefix_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
    counter_used: bool,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "scope": "V362 前缀数据集（不含隔离与缺失样本）",
        "total_prefix_rows": len(prefix_rows),
        "split_totals": tabulate(prefix_rows, lambda row: row["source_split"]),
        "decision_totals": tabulate(prefix_rows, lambda row: row["decision"]),
        "original_full_binary_label_totals": tabulate(
            prefix_rows, lambda row: row["original_full_binary_label"]
        ),
        "prefix_relation_totals": tabulate(prefix_rows, lambda row: row["prefix_relation"]),
        "scenario_totals": tabulate(prefix_rows, lambda row: row["scenario_tag"]),
        "confidence_totals": tabulate(prefix_rows, lambda row: row["confidence"]),
        "review_required_totals": tabulate(prefix_rows, lambda row: row["review_required"]),
        "sample_weight_totals": tabulate(prefix_rows, lambda row: row["sample_weight"]),
        "stage_loss_mask_totals": tabulate(prefix_rows, lambda row: row["stage_loss_mask"]),
        "cross_tabs": {
            "split_x_decision": cross_tab(
                prefix_rows, lambda row: row["source_split"], lambda row: row["decision"]
            ),
            "split_x_original_full_binary_label": cross_tab(
                prefix_rows,
                lambda row: row["source_split"],
                lambda row: row["original_full_binary_label"],
            ),
            "split_x_prefix_relation": cross_tab(
                prefix_rows,
                lambda row: row["source_split"],
                lambda row: row["prefix_relation"],
            ),
            "decision_x_original_full_binary_label": cross_tab(
                prefix_rows,
                lambda row: row["decision"],
                lambda row: row["original_full_binary_label"],
            ),
            "scenario_x_decision": cross_tab(
                prefix_rows, lambda row: row["scenario_tag"], lambda row: row["decision"]
            ),
        },
        "tE_turn": describe(
            [row["tE_turn"] for row in prefix_rows if isinstance(row["tE_turn"], int)]
        ),
        "review_required_samples": {
            "n": sum(1 for row in prefix_rows if row["review_required"]),
            "sample_ids": sorted(
                row["sample_id"] for row in prefix_rows if row["review_required"]
            ),
            "all_ambiguous": all(
                row["decision"] == "ambiguous"
                for row in prefix_rows
                if row["review_required"]
            ),
            "stage_loss_mask": 0,
        },
        "per_split": {},
        "evidence_type_totals": {},
        "excluded": {
            "quarantine": {
                "n": len(quarantine_rows),
                "sample_ids": [row["sample_id"] for row in quarantine_rows],
                "reasons": {
                    row["sample_id"]: row["quarantine_reason"] for row in quarantine_rows
                },
                "enters_training": False,
                "enters_main_metrics": False,
            },
            "missing": {
                "n": len(missing_rows),
                "sample_ids": [row["sample_id"] for row in missing_rows],
                "label_guessed": False,
            },
        },
    }

    evidence_counts: Counter = Counter()
    for row in prefix_rows:
        for bucket in EVIDENCE_BUCKETS:
            for evidence in row[bucket]:
                evidence_counts[str(evidence.get("evidence_type"))] += 1
    stats["evidence_type_totals"] = {
        name: evidence_counts[name] for name in sorted(evidence_counts)
    }

    for split in SPLITS:
        subset = [row for row in prefix_rows if row["source_split"] == split]
        if not subset:
            stats["per_split"][split] = {"n": 0}
            continue
        stats["per_split"][split] = {
            "n": len(subset),
            "decision_counts": tabulate(subset, lambda row: row["decision"]),
            "original_full_binary_label_counts": tabulate(
                subset, lambda row: row["original_full_binary_label"]
            ),
            "scenario_counts": tabulate(subset, lambda row: row["scenario_tag"]),
            "prefix_relation_counts": tabulate(subset, lambda row: row["prefix_relation"]),
            "length": length_block(subset, counter_used),
        }

    stats["length_all"] = length_block(prefix_rows, counter_used)
    stats["length_by_decision"] = {
        decision: length_block(
            [row for row in prefix_rows if row["decision"] == decision], counter_used
        )
        for decision in DECISIONS
    }
    return stats


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="V361 980 条原始长文本 JSONL")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS, help="annotations_*.jsonl 与 _review_summary.json 所在目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB, help="WordPiece 计数用 vocab.txt（缺失则跳过长度统计）")
    parser.add_argument(
        "--pre-evidence-weight",
        type=float,
        default=0.4,
        help="pre_evidence_or_not_yet_detectable 类 normal 的二分类权重，取值区间 0.3~0.5",
    )
    parser.add_argument("--quiet", action="store_true", help="只输出最终摘要")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.3 <= args.pre_evidence_weight <= 0.5:
        raise SystemExit("--pre-evidence-weight 必须在 0.3 到 0.5 之间（任务书建议区间）")

    source = load_source(args.source)
    annotations = load_annotations(args.annotations_dir)
    quarantine_reasons = load_quarantine_reasons(args.annotations_dir)

    # --- 构建前显式前置检查：任一项不成立立即失败 ---
    preflight_annotation_ids(annotations)
    preflight_split_consistency(source, annotations)
    preflight_missing_set(source, annotations)

    counter = load_wordpiece_counter(args.vocab)
    annotation_by_id = {row["sample_id"]: row for row in annotations}

    prefix_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    for sample_id in sorted(annotation_by_id):
        annotation = annotation_by_id[sample_id]
        source_row = source[sample_id]
        if sample_id in quarantine_reasons:
            quarantine_rows.append(
                build_quarantine_row(source_row, annotation, quarantine_reasons[sample_id])
            )
            continue
        prefix_rows.append(
            build_prefix_payload(
                source_row,
                annotation,
                counter=counter,
                pre_evidence_weight=args.pre_evidence_weight,
            )
        )

    missing_rows = [build_missing_row(source.get(MISSING_SAMPLE_ID))]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    split_filename = {
        "long_train": "prefix_train.jsonl",
        "long_dev": "prefix_dev.jsonl",
        "long_stress": "prefix_stress.jsonl",
    }

    written_files: list[Path] = []
    outputs: dict[str, Any] = {}
    for split in SPLITS:
        subset = [row for row in prefix_rows if row["source_split"] == split]
        path = out / split_filename[split]
        write_jsonl(path, subset)
        written_files.append(path)
        outputs[split] = {"file": path.name, "rows": len(subset)}

    quarantine_path = out / "quarantine_samples.jsonl"
    write_jsonl(quarantine_path, quarantine_rows)
    written_files.append(quarantine_path)
    outputs["quarantine"] = {
        "file": quarantine_path.name,
        "rows": len(quarantine_rows),
    }

    missing_path = out / "missing_samples.jsonl"
    write_jsonl(missing_path, missing_rows)
    written_files.append(missing_path)
    outputs["missing"] = {"file": missing_path.name, "rows": len(missing_rows)}

    validation = run_validations(
        source=source,
        annotations=annotations,
        prefix_rows=prefix_rows,
        quarantine_rows=quarantine_rows,
        missing_rows=missing_rows,
        quarantine_reasons=quarantine_reasons,
        pre_evidence_weight=args.pre_evidence_weight,
    )
    statistics = build_statistics(
        prefix_rows=prefix_rows,
        quarantine_rows=quarantine_rows,
        missing_rows=missing_rows,
        counter_used=counter is not None,
    )

    stats_path = out / "dataset_statistics.json"
    write_json(stats_path, statistics)
    written_files.append(stats_path)
    outputs["dataset_statistics"] = {"file": stats_path.name}

    validation_path = out / "validation_report.json"
    write_json(validation_path, validation)
    written_files.append(validation_path)
    outputs["validation_report"] = {"file": validation_path.name}

    # --- SHA256 清单：输入、脚本与全部产出 ---
    script_path = Path(__file__).resolve()
    annotation_files = sorted(args.annotations_dir.glob("annotations_*.jsonl"))
    integrity: dict[str, Any] = {
        "algorithm": "sha256",
        "inputs": {
            "source_980": {
                "path": relative(args.source),
                "rows": len(source),
                "sha256": sha256_file(args.source),
            },
            "annotations": {
                "directory": relative(args.annotations_dir),
                "file_count": len(annotation_files),
                "rows": len(annotations),
                "files": [
                    {
                        "file": path.name,
                        "rows": count_nonblank_lines(path),
                        "sha256": sha256_file(path),
                    }
                    for path in annotation_files
                ],
            },
            "review_summary": {
                "path": relative(args.annotations_dir / "_review_summary.json"),
                "sha256": sha256_file(args.annotations_dir / "_review_summary.json"),
                "quarantine_count": len(quarantine_reasons),
            },
            "vocab": {
                "path": relative(args.vocab),
                "sha256": sha256_file(args.vocab) if args.vocab.exists() else None,
                "available": counter is not None,
            },
        },
        "script": {
            "path": relative(script_path),
            "sha256": sha256_file(script_path),
        },
        "outputs": {},
    }
    for name, path in [
        ("long_train", out / split_filename["long_train"]),
        ("long_dev", out / split_filename["long_dev"]),
        ("long_stress", out / split_filename["long_stress"]),
        ("quarantine", quarantine_path),
        ("missing", missing_path),
        ("dataset_statistics", stats_path),
        ("validation_report", validation_path),
    ]:
        outputs[name]["sha256"] = sha256_file(path)
    integrity["outputs"] = {
        name: {"file": payload["file"], "sha256": payload["sha256"]}
        for name, payload in outputs.items()
    }

    manifest = {
        "dataset": "v362_early_intent",
        "purpose": "诈骗前期意图前缀数据集：在明确转账/验证码/下载等操作出现前判断是否已具备可观察诈骗意图",
        "inputs": {
            "source_980": relative(args.source),
            "annotations_dir": relative(args.annotations_dir),
            "annotation_file_count": len(annotation_files),
            "annotation_rows": len(annotations),
            "vocab": relative(args.vocab),
            "wordpiece_counter_available": counter is not None,
        },
        "outputs": {
            name: {"file": payload["file"], "rows": payload.get("rows")}
            for name, payload in outputs.items()
        },
        "integrity": integrity,
        "label_mapping": {
            "stage_target": STAGE_TARGET,
            "binary_target_policy": {
                "early_fraud": 1,
                "explicit_fraud": 1,
                "normal": 0,
                "ambiguous": -1,
            },
            "binary_loss_mask_policy": {
                "early_fraud": 1,
                "explicit_fraud": 1,
                "normal": 1,
                "ambiguous": 0,
            },
            "original_full_binary_label": GENERATION_KIND_TO_BINARY,
        },
        "sample_weight_policy": {
            "clean_normal": 1.0,
            "pre_evidence_or_not_yet_detectable": args.pre_evidence_weight,
            "ambiguous_prefix": 0.0,
            "early_fraud_prefix": 1.0,
            "explicit_operation_prefix": 1.0,
        },
        "stage_loss_policy": {
            "stage_loss_mask_default": 1,
            "stage_loss_mask_review_required": REVIEW_REQUIRED_STAGE_MASK,
            "note": (
                "review_required 样本的 stage_loss_mask=0。"
                "在未获确认前不得用本数据集训练四分类（stage）头；"
                "当前 review_required 全部为 ambiguous。"
            ),
        },
        "prefix_construction": {
            "slice": "turns[0:prefix_end_turn]",
            "turn_indexing": "1-based，prefix_end_turn 为可见的最后一轮，不多取",
            "role_markers": "prefix_text 不带角色标识，与项目 model_input_text 约定一致",
            "evidence_excluded_from_text": True,
        },
        "text_normalization": "NFKC 归一 + 去除全部空白，再取 sha256",
        "excluded_from_training": {
            "quarantine": [row["sample_id"] for row in quarantine_rows],
            "missing": [row["sample_id"] for row in missing_rows],
        },
        "counts": {
            "prefix_rows": len(prefix_rows),
            "quarantine_rows": len(quarantine_rows),
            "missing_rows": len(missing_rows),
        },
        "reproducibility": "不使用随机数；相同输入重复运行结果一致",
    }

    write_json(out / "build_manifest.json", manifest)

    if not args.quiet:
        print("== V362 前缀数据集构建完成 ==")
        print("输出目录: %s" % out)
        for split in SPLITS:
            print("  %-28s %d 条" % (outputs[split]["file"], outputs[split]["rows"]))
        print("  %-28s %d 条" % (quarantine_path.name, len(quarantine_rows)))
        print("  %-28s %d 条" % (missing_path.name, len(missing_rows)))
        print("decision 分布: %s" % statistics["decision_totals"])
        print(
            "校验: %d 项，%s"
            % (
                validation["n_checks"],
                "全部通过"
                if validation["all_passed"]
                else "失败 -> %s" % validation["failed_checks"],
            )
        )

    return 0 if validation["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
