#!/usr/bin/env python3
"""Train a 13-class classifier on CE-supervised HF/HN data and eval external CSVs.

The external CSVs are evaluation-only. This script derives binary metrics from
the 13-class prediction via `pred_label_13 != 0` and reports subtype metrics on
fraud rows separately.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from .fraudce_agent.data_contract import strict_label13_of
except ImportError:
    try:
        from ..cascade.fraudce_agent.data_contract import strict_label13_of
    except ImportError:
        from fraudce_agent.data_contract import strict_label13_of

try:
    from lib.history_writer import append_epoch_history
    from lib.progress_writer import update_eval_progress, update_progress, update_train_progress
except ImportError:
    try:
        from ..lib.history_writer import append_epoch_history
        from ..lib.progress_writer import update_eval_progress, update_progress, update_train_progress
    except ImportError:
        from scripts.lib.history_writer import append_epoch_history
        from scripts.lib.progress_writer import update_eval_progress, update_progress, update_train_progress

try:
    from .official_subcategories import (
        NUM_OFFICIAL_SUBCATEGORIES,
        OFFICIAL_SUBCATEGORY_CODES,
        OFFICIAL_PARENT_TO_INDICES,
        official_subcategory_code_of as _official_subcategory_code_of,
        official_subcategory_index_of as _official_subcategory_index_of,
    )
except ImportError:
    from official_subcategories import (
        NUM_OFFICIAL_SUBCATEGORIES,
        OFFICIAL_SUBCATEGORY_CODES,
        OFFICIAL_PARENT_TO_INDICES,
        official_subcategory_code_of as _official_subcategory_code_of,
        official_subcategory_index_of as _official_subcategory_index_of,
    )

try:
    from .hierarchical_protocol import (
        fpr_constrained_checkpoint_decision,
        h2_binary_mass_consistency_loss,
    )
except ImportError:
    from hierarchical_protocol import (
        fpr_constrained_checkpoint_decision,
        h2_binary_mass_consistency_loss,
    )


NUM_CLASSES = 13
DUAL_HEAD_ARCHITECTURES = {
    "ce_dual_head",
    "ce_dual_head_hier",
    "ce_dual_head_h2",
    "h2_cascade",
    "v24_clear11_shadow",
    "v24_clear11_residual12_shadow",
    "ce_selective_pair_expert",
    "ce_pairwise_boundary_expert",
}
SUBTYPE_AUX_ARCHITECTURES = DUAL_HEAD_ARCHITECTURES | {
    "ce_subtype_aux",
    "ce_protected_subtype",
    "ce_protected_top_layers",
}
CUSTOM_ARCHITECTURES = {
    "ce_gated_neck",
    "ce_dual_head",
    "ce_dual_head_hier",
    "ce_dual_head_h2",
    "v24_clear11_shadow",
    "v24_clear11_residual12_shadow",
    "ce_subtype_aux",
    "ce_mid_adapter",
    "h2_cascade",
    "ce_protected_subtype",
    "ce_protected_top_layers",
    "ce_protected_official_prototypes",
    "ce_protected_official_postgate",
    "ce_pbal_frozen_official_postgate",
    "ce_local_expert",
    "ce_selective_pair_expert",
    "ce_pairwise_boundary_expert",
}

DEFAULT_LOCAL_EXPERT_GROUPS = "0,1,7,12;0,3,6,12;0,4,9,12"
LOCAL_EXPERT_GROUP_PRESETS = {
    "g1": "0,1,7,12",
    "g12": "0,1,7,12;0,3,6,12",
    "g123": DEFAULT_LOCAL_EXPERT_GROUPS,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            if "_note" in obj and len(obj) == 1:
                continue
            rows.append(obj)
    if not rows:
        raise SystemExit(f"empty JSONL: {path}")
    return rows


def read_eval_csv(path: Path) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                rows = list(csv.DictReader(file))
            if not rows:
                raise ValueError("empty CSV")
            if "label_13" not in rows[0]:
                raise ValueError("missing label_13 column")
            for idx, row in enumerate(rows):
                row["_row_index"] = idx
                row["_csv_encoding"] = encoding
            return rows
        except Exception as exc:
            last_error = exc
    raise SystemExit(f"failed to read eval CSV {path}: {last_error}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def mirror_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def distributed_env() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    return world_size > 1, world_size, rank, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def distributed_barrier() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        return


def distributed_broadcast_object(payload: Any, rank: int) -> Any:
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return payload
        box = [payload if rank == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]
    except Exception:
        return payload


def text_of(row: dict[str, Any]) -> str:
    value = (
        row.get("model_input_text")
        or row.get("transcript")
        or row.get("text")
        or row.get("script_text")
        or ""
    )
    return " ".join(str(value).split())


def label13_of(row: dict[str, Any]) -> int:
    return strict_label13_of(row)


def official_subcategory_code_of(row: dict[str, Any]) -> str:
    return _official_subcategory_code_of(row, label13_of)


def official_subcategory_index_of(row: dict[str, Any]) -> int:
    return _official_subcategory_index_of(row, label13_of)


def confusable_label13_of(row: dict[str, Any]) -> int:
    value = (
        row.get("boundary_confusable_label_13")
        if row.get("boundary_confusable_label_13") not in (None, "")
        else row.get("confusable_label_13")
    )
    try:
        label = int(float(value))
    except (TypeError, ValueError):
        return -1
    return label if 0 <= label < NUM_CLASSES else -1


def sample_id_of(row: dict[str, Any], prefix: str, idx: int) -> str:
    return str(row.get("sample_id") or row.get("id") or row.get("public_id") or f"{prefix}_{idx:06d}")


def base_sample_id_of(row: dict[str, Any], prefix: str, idx: int) -> str:
    return str(row.get("base_sample_id") or sample_id_of(row, prefix, idx))


def stratified_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[label13_of(row)].append(row)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        if len(items) - n_val <= 0:
            n_val = max(0, len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def scenario_key_of(row: dict[str, Any]) -> str:
    for key in ("target_subdomain", "source_task_id", "scenario_family", "source_quota", "source_axis", "axis"):
        value = row.get(key)
        if value not in (None, "", []):
            return f"{key}:{value}"
    return ""


def split_group_key(row: dict[str, Any], mode: str, idx: int) -> str:
    if mode == "random":
        return f"row:{idx}"
    pair_id = str(row.get("pair_id") or "")
    scenario = scenario_key_of(row)
    if mode == "pair":
        return f"pair:{pair_id}" if pair_id else f"row:{idx}"
    if mode == "scenario":
        return f"scenario:{scenario}" if scenario else f"pair:{pair_id}" if pair_id else f"row:{idx}"
    if mode == "pair_scenario":
        return f"scenario:{scenario}" if scenario else f"pair:{pair_id}" if pair_id else f"row:{idx}"
    raise ValueError(f"unsupported val split mode: {mode}")


def stratified_group_split(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if mode == "random":
        train, val = stratified_split(rows, val_ratio, seed)
    else:
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for idx, row in enumerate(rows):
            by_group[split_group_key(row, mode, idx)].append(row)
        groups = list(by_group.items())
        rng = random.Random(seed)
        rng.shuffle(groups)
        target_val = max(1, int(round(len(rows) * val_ratio)))
        val_group_keys: set[str] = set()
        val_counts: Counter[str] = Counter()
        total_val = 0
        for group_key, group_rows in groups:
            if total_val >= target_val:
                break
            labels = {label13_of(row) for row in group_rows}
            if any(val_counts[str(label)] == 0 for label in labels) or total_val + len(group_rows) <= target_val:
                val_group_keys.add(group_key)
                total_val += len(group_rows)
                val_counts.update(str(label13_of(row)) for row in group_rows)
        missing_labels = sorted({label13_of(row) for row in rows} - {int(label) for label in val_counts})
        for missing in missing_labels:
            candidates = [
                (group_key, group_rows)
                for group_key, group_rows in groups
                if group_key not in val_group_keys and any(label13_of(row) == missing for row in group_rows)
            ]
            if candidates:
                group_key, group_rows = candidates[0]
                val_group_keys.add(group_key)
                total_val += len(group_rows)
                val_counts.update(str(label13_of(row)) for row in group_rows)
        train, val = [], []
        for idx, row in enumerate(rows):
            key = split_group_key(row, mode, idx)
            (val if key in val_group_keys else train).append(row)
        rng.shuffle(train)
        rng.shuffle(val)

    train_pairs = {str(row.get("pair_id") or "") for row in train if row.get("pair_id")}
    val_pairs = {str(row.get("pair_id") or "") for row in val if row.get("pair_id")}
    train_scenarios = {scenario_key_of(row) for row in train if scenario_key_of(row)}
    val_scenarios = {scenario_key_of(row) for row in val if scenario_key_of(row)}
    manifest = {
        "val_split_mode": mode,
        "val_ratio": val_ratio,
        "seed": seed,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_label_13_counts": dict(sorted(Counter(str(label13_of(row)) for row in train).items(), key=lambda item: int(item[0]))),
        "val_label_13_counts": dict(sorted(Counter(str(label13_of(row)) for row in val).items(), key=lambda item: int(item[0]))),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "pair_overlap_count": len(train_pairs & val_pairs),
        "train_scenario_count": len(train_scenarios),
        "val_scenario_count": len(val_scenarios),
        "scenario_overlap_count": len(train_scenarios & val_scenarios),
    }
    return train, val, manifest


def fixed_id_split(
    rows: list[dict[str, Any]],
    val_id_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    val_ids = {line.strip() for line in val_id_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not val_ids:
        raise SystemExit(f"empty val id file: {val_id_file}")
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    seen_val_ids: set[str] = set()
    for idx, row in enumerate(rows):
        sample_id = sample_id_of(row, "train", idx)
        if sample_id in val_ids:
            val.append(row)
            seen_val_ids.add(sample_id)
        else:
            train.append(row)
    missing = sorted(val_ids - seen_val_ids)
    if missing:
        raise SystemExit(f"val id file contains ids not found in train_jsonl: {missing[:10]} total={len(missing)}")
    if not train or not val:
        raise SystemExit(f"invalid fixed split: train={len(train)} val={len(val)}")
    train_pairs = {str(row.get("pair_id") or "") for row in train if row.get("pair_id")}
    val_pairs = {str(row.get("pair_id") or "") for row in val if row.get("pair_id")}
    train_scenarios = {scenario_key_of(row) for row in train if scenario_key_of(row)}
    val_scenarios = {scenario_key_of(row) for row in val if scenario_key_of(row)}
    manifest = {
        "val_split_mode": "fixed_id",
        "val_id_file": str(val_id_file),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_label_13_counts": dict(sorted(Counter(str(label13_of(row)) for row in train).items(), key=lambda item: int(item[0]))),
        "val_label_13_counts": dict(sorted(Counter(str(label13_of(row)) for row in val).items(), key=lambda item: int(item[0]))),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "pair_overlap_count": len(train_pairs & val_pairs),
        "train_scenario_count": len(train_scenarios),
        "val_scenario_count": len(val_scenarios),
        "scenario_overlap_count": len(train_scenarios & val_scenarios),
    }
    return train, val, manifest


def independent_val_manifest(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    val_jsonl: Path,
) -> dict[str, Any]:
    train_sample_ids = {sample_id_of(row, "train", idx) for idx, row in enumerate(train_rows)}
    val_sample_ids = {sample_id_of(row, "val", idx) for idx, row in enumerate(val_rows)}
    train_base_ids = {base_sample_id_of(row, "train", idx) for idx, row in enumerate(train_rows)}
    val_base_ids = {base_sample_id_of(row, "val", idx) for idx, row in enumerate(val_rows)}
    train_pairs = {str(row.get("pair_id") or "") for row in train_rows if row.get("pair_id")}
    val_pairs = {str(row.get("pair_id") or "") for row in val_rows if row.get("pair_id")}
    train_scenarios = {scenario_key_of(row) for row in train_rows if scenario_key_of(row)}
    val_scenarios = {scenario_key_of(row) for row in val_rows if scenario_key_of(row)}
    return {
        "val_split_mode": "independent_jsonl",
        "val_jsonl": str(val_jsonl),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_label_13_counts": dict(sorted(Counter(str(label13_of(row)) for row in train_rows).items(), key=lambda item: int(item[0]))),
        "val_label_13_counts": dict(sorted(Counter(str(label13_of(row)) for row in val_rows).items(), key=lambda item: int(item[0]))),
        "sample_id_overlap_count": len(train_sample_ids & val_sample_ids),
        "base_sample_id_overlap_count": len(train_base_ids & val_base_ids),
        "train_pair_count": len(train_pairs),
        "val_pair_count": len(val_pairs),
        "pair_overlap_count": len(train_pairs & val_pairs),
        "train_scenario_count": len(train_scenarios),
        "val_scenario_count": len(val_scenarios),
        "scenario_overlap_count": len(train_scenarios & val_scenarios),
        "val_group_counts": dict(sorted(Counter(str(row.get("val_group") or "unknown") for row in val_rows).items())),
    }


@dataclass
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    typed_evidence_input_ids: list[int]
    typed_evidence_attention_mask: list[int]
    has_typed_evidence: int
    span_region_mask: list[int]
    has_span_region: int
    label13: int
    official_subcategory_index: int
    sample_id: str
    base_sample_id: str
    view_id: str
    split: str
    source: str
    pair_id: str
    pair_role: str
    confusable_label13: int
    is_fraud: int
    sample_weight: float
    distill_probs: list[float]
    has_distill_probs: int
    coarse_label_mode: str
    text: str
    source_group: str
    val_group: str


def span_texts(row: dict[str, Any], field: str) -> list[str]:
    raw = row.get(field)
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, dict):
            value = item.get("span") or item.get("text") or ""
        else:
            value = item
        text = " ".join(str(value).split())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def span_region_texts_of(row: dict[str, Any]) -> list[str]:
    label = label13_of(row)
    if label == 0:
        return span_texts(row, "counter_evidence_spans") or ce_span_texts(row, "counter_evidence")
    decisive = span_texts(row, "decisive_fraud_action_spans") or ce_span_texts(row, "decisive_fraud_action")
    subtype = span_texts(row, "subtype_distinctive_spans") or span_texts(row, "subtype_evidence_spans") or ce_span_texts(
        row,
        "subtype_distinctive",
        "subtype_evidence",
    )
    out: list[str] = []
    seen: set[str] = set()
    for span in [*decisive, *subtype]:
        if span and span not in seen:
            seen.add(span)
            out.append(span)
    return out


def build_span_region_mask(text: str, spans: list[str], offset_mapping: list[tuple[int, int]], max_length: int) -> tuple[list[int], int]:
    mask = [0] * max_length
    if not spans:
        return mask, 0
    matched = 0
    for span in spans:
        value = " ".join(str(span or "").split())
        if not value:
            continue
        start = text.find(value)
        if start < 0:
            continue
        end = start + len(value)
        span_tokens = 0
        for idx, offset in enumerate(offset_mapping[:max_length]):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                continue
            token_start, token_end = int(offset[0]), int(offset[1])
            if token_start == token_end == 0:
                continue
            if token_end > start and token_start < end:
                mask[idx] = 1
                span_tokens += 1
        if span_tokens > 0:
            matched += 1
    return mask, matched


def coarse_label_mode_of(row: dict[str, Any]) -> str:
    mode = str(row.get("coarse_label_mode") or row.get("partial_label_mode") or "").strip()
    if mode in {"", "none", "label13", "full13", "normal", "normal_any"}:
        return "none"
    if mode in {"fraud_any", "fraud_1_12"}:
        return "fraud_any"
    raise ValueError(f"unsupported coarse_label_mode: {mode!r}")


def distill_probs_of(row: dict[str, Any], field: str) -> list[float]:
    if not field:
        return [0.0] * NUM_CLASSES
    raw = row.get(field)
    if raw is None:
        values = [row.get(f"{field}_{idx}") for idx in range(NUM_CLASSES)]
    else:
        values = raw
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = [part.strip() for part in values.split(",")]
    if not isinstance(values, list) or len(values) != NUM_CLASSES:
        return [0.0] * NUM_CLASSES
    out: list[float] = []
    for item in values:
        try:
            out.append(float(item))
        except Exception:
            return [0.0] * NUM_CLASSES
    total = sum(max(0.0, value) for value in out)
    if total <= 0:
        return [0.0] * NUM_CLASSES
    return [max(0.0, value) / total for value in out]


def has_coarse_fraud_label(row: dict[str, Any]) -> bool:
    return coarse_label_mode_of(row) == "fraud_any"


def ce_span_texts(row: dict[str, Any], *roles: str) -> list[str]:
    raw = row.get("ce_spans")
    if not isinstance(raw, list):
        return []
    role_set = set(roles)
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("ce_role") or item.get("type") or "")
        if role_set and role not in role_set:
            continue
        text = " ".join(str(item.get("span") or item.get("text") or "").split())
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def typed_evidence_text_of(row: dict[str, Any]) -> str:
    label = label13_of(row)
    confusable = confusable_label13_of(row)
    if label == 0:
        counter = span_texts(row, "counter_evidence_spans") or ce_span_texts(row, "counter_evidence")
        surface = span_texts(row, "surface_risk_spans")[:2]
        non_decisive = span_texts(row, "non_decisive_risk_spans")[:2]
        reference = str(row.get("normal_counter_evidence_reference") or "").strip()
        if not counter:
            counter = [reference] if reference else []
        if not counter:
            return ""
        excluded = confusable if confusable >= 0 else row.get("paired_fraud_label_13", "")
        parts = [f"正常反证 排除类型{excluded}", "合法反证：" + "；".join(counter[:4])]
        if surface:
            parts.append("表面风险：" + "；".join(surface))
        if non_decisive:
            parts.append("非决定风险：" + "；".join(non_decisive))
        return "。".join(parts)

    evidence = span_texts(row, "evidence_spans") or ce_span_texts(row, "decisive_fraud_action", "pressure_or_isolation")
    decisive = span_texts(row, "decisive_fraud_action_spans") or ce_span_texts(row, "decisive_fraud_action")
    fake = (span_texts(row, "fake_legitimacy_spans") or ce_span_texts(row, "fake_legitimacy"))[:3]
    surface = span_texts(row, "surface_risk_spans")[:2]
    action_ref = str(row.get("decisive_fraud_action_reference") or "").strip()
    support = evidence or decisive
    if not support and action_ref:
        decisive = [action_ref]
        support = decisive
    if not support:
        return ""
    parts = [f"诈骗类型{label}证据"]
    subtype = str(row.get("label12_subtype") or "").strip()
    bucket = str(row.get("label12_boundary_bucket") or "").strip()
    axis = str(row.get("boundary_axis") or "").strip()
    if label == 12 and subtype:
        parts.append("label12子域：" + subtype)
    if bucket:
        parts.append("label12边界桶：" + bucket)
    if axis:
        parts.append("边界轴：" + axis)
    if evidence:
        parts.append("风险证据：" + "；".join(evidence[:4]))
    if decisive:
        parts.append("决定动作：" + "；".join(decisive[:3]))
    if fake:
        parts.append("伪正规：" + "；".join(fake))
    if surface:
        parts.append("表面风险：" + "；".join(surface))
    if confusable >= 0 and confusable != label:
        parts.append(f"不是类型{confusable}")
    why = str(row.get("why_this_label") or "").strip()
    why_not = str(row.get("why_not_confusable_label") or "").strip()
    if why:
        parts.append("判别理由：" + why[:180])
    if why_not:
        parts.append("排除理由：" + why_not[:180])
    return "。".join(parts)


class TextDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        split: str,
        source: str,
        typed_evidence_max_length: int = 0,
        span_region_ce_enabled: bool = False,
        distill_prob_field: str = "",
    ) -> None:
        self.examples: list[EncodedExample] = []
        for idx, row in enumerate(rows):
            text = text_of(row)
            distill_probs = distill_probs_of(row, distill_prob_field)
            has_distill_probs = int(sum(distill_probs) > 0)
            if span_region_ce_enabled:
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    return_offsets_mapping=True,
                )
                offset_mapping = list(encoded.pop("offset_mapping"))
                span_region_mask, matched_spans = build_span_region_mask(
                    text,
                    span_region_texts_of(row),
                    offset_mapping,
                    max_length,
                )
            else:
                encoded = tokenizer(text, max_length=max_length, truncation=True, padding="max_length")
                span_region_mask = [0] * max_length
                matched_spans = 0
            typed_text = typed_evidence_text_of(row) if typed_evidence_max_length > 0 else ""
            if typed_evidence_max_length > 0:
                typed_encoded = tokenizer(
                    typed_text if typed_text else text,
                    max_length=typed_evidence_max_length,
                    truncation=True,
                    padding="max_length",
                )
            else:
                typed_encoded = {"input_ids": [0], "attention_mask": [0]}
            self.examples.append(
                EncodedExample(
                    input_ids=list(encoded["input_ids"]),
                    attention_mask=list(encoded["attention_mask"]),
                    typed_evidence_input_ids=list(typed_encoded["input_ids"]),
                    typed_evidence_attention_mask=list(typed_encoded["attention_mask"]),
                    has_typed_evidence=1 if typed_text else 0,
                    span_region_mask=span_region_mask,
                    has_span_region=1 if matched_spans > 0 else 0,
                    label13=label13_of(row),
                    official_subcategory_index=official_subcategory_index_of(row),
                    sample_id=sample_id_of(row, source, idx),
                    base_sample_id=base_sample_id_of(row, source, idx),
                    view_id=str(row.get("view_id") or ""),
                    split=split,
                    source=source,
                    pair_id=str(row.get("pair_id") or ""),
                    pair_role=str(row.get("pair_role") or ""),
                    confusable_label13=confusable_label13_of(row),
                    is_fraud=1 if bool(row.get("is_fraud", label13_of(row) != 0)) else 0,
                    sample_weight=float(row.get("sample_weight", row.get("field_quality_weight", 1.0)) or 1.0),
                    distill_probs=distill_probs,
                    has_distill_probs=has_distill_probs,
                    coarse_label_mode=coarse_label_mode_of(row),
                    text=str(row.get("text") or row.get("model_input_text") or ""),
                    source_group=str(row.get("source_group") or ""),
                    val_group=str(row.get("val_group") or ""),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {
            "input_ids": ex.input_ids,
            "attention_mask": ex.attention_mask,
            "typed_evidence_input_ids": ex.typed_evidence_input_ids,
            "typed_evidence_attention_mask": ex.typed_evidence_attention_mask,
            "has_typed_evidence": ex.has_typed_evidence,
            "span_region_mask": ex.span_region_mask,
            "has_span_region": ex.has_span_region,
            "label13": ex.label13,
            "official_subcategory_index": ex.official_subcategory_index,
            "sample_id": ex.sample_id,
            "base_sample_id": ex.base_sample_id,
            "view_id": ex.view_id,
            "split": ex.split,
            "source": ex.source,
            "pair_id": ex.pair_id,
            "pair_role": ex.pair_role,
            "confusable_label13": ex.confusable_label13,
            "is_fraud": ex.is_fraud,
            "sample_weight": ex.sample_weight,
            "distill_probs": ex.distill_probs,
            "has_distill_probs": ex.has_distill_probs,
            "coarse_label_mode": ex.coarse_label_mode,
            "text": ex.text,
            "source_group": ex.source_group,
            "val_group": ex.val_group,
        }


class RankShardSampler:
    def __init__(self, size: int, world_size: int, rank: int) -> None:
        self.indices = list(range(rank, size, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        "typed_evidence_input_ids": torch.tensor([item["typed_evidence_input_ids"] for item in batch], dtype=torch.long),
        "typed_evidence_attention_mask": torch.tensor([item["typed_evidence_attention_mask"] for item in batch], dtype=torch.long),
        "has_typed_evidence": torch.tensor([item["has_typed_evidence"] for item in batch], dtype=torch.bool),
        "span_region_mask": torch.tensor([item["span_region_mask"] for item in batch], dtype=torch.float32),
        "has_span_region": torch.tensor([item["has_span_region"] for item in batch], dtype=torch.bool),
        "label13": torch.tensor([item["label13"] for item in batch], dtype=torch.long),
        "official_subcategory_index": torch.tensor(
            [item["official_subcategory_index"] for item in batch], dtype=torch.long
        ),
        "sample_id": [item["sample_id"] for item in batch],
        "base_sample_id": [item["base_sample_id"] for item in batch],
        "view_id": [item["view_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "source": [item["source"] for item in batch],
        "pair_id": [item["pair_id"] for item in batch],
        "pair_role": [item["pair_role"] for item in batch],
        "confusable_label13": torch.tensor([item["confusable_label13"] for item in batch], dtype=torch.long),
        "is_fraud": torch.tensor([item["is_fraud"] for item in batch], dtype=torch.float32),
        "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
        "distill_probs": torch.tensor([item["distill_probs"] for item in batch], dtype=torch.float32),
        "has_distill_probs": torch.tensor([item["has_distill_probs"] for item in batch], dtype=torch.bool),
        "coarse_label_mode": [item["coarse_label_mode"] for item in batch],
        "text": [item["text"] for item in batch],
        "source_group": [item["source_group"] for item in batch],
        "val_group": [item["val_group"] for item in batch],
    }


def build_pair_aware_batches(dataset: TextDataset, batch_size: int, seed: int) -> list[list[int]]:
    by_pair: dict[str, list[int]] = defaultdict(list)
    unpaired: list[int] = []
    for idx, example in enumerate(dataset.examples):
        if example.pair_id:
            by_pair[example.pair_id].append(idx)
        else:
            unpaired.append(idx)
    groups = [indices for indices in by_pair.values() if indices]
    groups.extend([idx] for idx in unpaired)
    rng = random.Random(seed)
    rng.shuffle(groups)
    batches: list[list[int]] = []
    current: list[int] = []
    for group in groups:
        group = list(group)
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        if len(group) > batch_size:
            for start in range(0, len(group), batch_size):
                chunk = group[start : start + batch_size]
                if chunk:
                    batches.append(chunk)
            continue
        current.extend(group)
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def same_pair_margin_loss(logits: Any, pair_ids: list[str], pair_roles: list[str], margin: float) -> Any:
    import torch

    fraud_score = torch.logsumexp(logits[:, 1:], dim=1)
    by_pair: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"fraud": [], "normal": []})
    for idx, (pair_id, role) in enumerate(zip(pair_ids, pair_roles)):
        if not pair_id:
            continue
        role_key = "fraud" if role == "fraud" else "normal" if role == "normal" else ""
        if role_key:
            by_pair[pair_id][role_key].append(idx)
    losses = []
    for item in by_pair.values():
        if not item["fraud"] or not item["normal"]:
            continue
        fraud_value = fraud_score[item["fraud"]].mean()
        normal_value = fraud_score[item["normal"]].mean()
        losses.append(torch.relu(torch.tensor(float(margin), device=logits.device) - (fraud_value - normal_value)))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def confusable_margin_loss(logits: Any, labels: Any, confusable: Any, margin: float) -> Any:
    import torch

    mask = (confusable >= 0) & (confusable < NUM_CLASSES) & (confusable != labels)
    if int(mask.sum().detach().cpu()) <= 0:
        return logits.sum() * 0.0
    row_idx = torch.arange(logits.shape[0], device=logits.device)[mask]
    true_logits = logits[row_idx, labels[mask]]
    confusable_logits = logits[row_idx, confusable[mask]]
    return torch.relu(torch.tensor(float(margin), device=logits.device) - (true_logits - confusable_logits)).mean()


def cls_embedding(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    base_model = getattr(model, model.base_model_prefix)
    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is not None:
        return pooled
    return outputs.last_hidden_state[:, 0]


def classifier_logits_from_embedding(model: Any, embedding: Any) -> Any:
    if not hasattr(model, "classifier"):
        raise ValueError("typed evidence head_stopgrad mode requires model.classifier")
    hidden = model.dropout(embedding) if hasattr(model, "dropout") else embedding
    return model.classifier(hidden)


def attach_binary_aux_head(model: Any, device: str) -> None:
    import torch

    if hasattr(model, "binary_aux_classifier"):
        return
    hidden_size = int(getattr(model.config, "hidden_size"))
    model.binary_aux_classifier = torch.nn.Linear(hidden_size, 2).to(device)


def binary_aux_logits(model: Any, batch: dict[str, Any], device: str) -> Any:
    if not hasattr(model, "binary_aux_classifier"):
        raise ValueError("binary aux loss requires model.binary_aux_classifier")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if hasattr(model, "pooled_embedding"):
        pooled = model.pooled_embedding(input_ids, attention_mask)
    else:
        pooled = cls_embedding(model, input_ids, attention_mask)
    hidden = model.dropout(pooled) if hasattr(model, "dropout") else pooled
    return model.binary_aux_classifier(hidden)


def build_ce_gated_neck_model(
    model_name_or_path: str,
    num_labels: int,
    dropout_prob: float,
    gate_bias: float,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout

    class CEGatedNeckForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.num_labels = num_labels
            self.dropout = torch.nn.Dropout(dropout)
            self.gate = torch.nn.Linear(hidden_size, hidden_size)
            self.adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size),
            )
            self.neck_norm = torch.nn.LayerNorm(hidden_size)
            self.classifier = torch.nn.Linear(hidden_size, num_labels)
            torch.nn.init.zeros_(self.adapter[-1].weight)
            torch.nn.init.zeros_(self.adapter[-1].bias)
            torch.nn.init.constant_(self.gate.bias, gate_bias)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def neck_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            delta = self.adapter(pooled)
            gate = torch.sigmoid(self.gate(pooled))
            return self.neck_norm(pooled + gate * delta)

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            neck = self.neck_embedding(input_ids, attention_mask)
            logits = self.classifier(self.dropout(neck))
            return SimpleNamespace(logits=logits)

    return CEGatedNeckForSequenceClassification()


def build_ce_dual_head_model(
    model_name_or_path: str,
    dropout_prob: float,
    joint_mode: str = "additive",
    official_subcategory_aux: bool = False,
    flat_auxiliary: bool = False,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    if joint_mode not in {"additive", "hierarchical"}:
        raise ValueError(f"unsupported dual-head joint mode: {joint_mode}")

    class CEDualHeadForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.joint_mode = joint_mode
            self.dropout = torch.nn.Dropout(dropout)
            self.binary_classifier = torch.nn.Linear(hidden_size, 2)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.flat_aux_classifier = (
                torch.nn.Linear(hidden_size, NUM_CLASSES)
                if flat_auxiliary
                else None
            )
            self.official_subcategory_classifier = (
                torch.nn.Linear(hidden_size, NUM_OFFICIAL_SUBCATEGORIES)
                if official_subcategory_aux
                else None
            )

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def subtype_logits(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            return self.subtype_classifier(self.dropout(pooled))

        def joint_logits(self, binary_logits: Any, subtype_logits: Any) -> Any:
            if self.joint_mode == "hierarchical":
                binary_scores = torch.log_softmax(binary_logits, dim=-1)
                subtype_scores = torch.log_softmax(subtype_logits, dim=-1)
            else:
                binary_scores = binary_logits
                subtype_scores = subtype_logits
            normal_scores = binary_scores[:, :1]
            fraud_scores = binary_scores[:, 1:2] + subtype_scores
            return torch.cat([normal_scores, fraud_scores], dim=-1)

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            hidden = self.dropout(pooled)
            binary_logits = self.binary_classifier(hidden)
            subtype_logits = self.subtype_classifier(hidden)
            logits = self.joint_logits(binary_logits, subtype_logits)
            flat_aux_logits = (
                self.flat_aux_classifier(hidden)
                if self.flat_aux_classifier is not None
                else None
            )
            official_subcategory_logits = (
                self.official_subcategory_classifier(hidden)
                if self.official_subcategory_classifier is not None
                else None
            )
            return SimpleNamespace(
                logits=logits,
                binary_logits=binary_logits,
                subtype_logits=subtype_logits,
                flat_aux_logits=flat_aux_logits,
                official_subcategory_logits=official_subcategory_logits,
            )

    return CEDualHeadForSequenceClassification()


def build_v24_clear11_shadow_model(
    model_name_or_path: str,
    dropout_prob: float,
    joint_mode: str = "additive",
    clear11_detach: bool = False,
    residual12_detach: bool = False,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    if joint_mode not in {"additive", "hierarchical"}:
        raise ValueError(f"unsupported dual-head joint mode: {joint_mode}")

    class V24Clear11ShadowForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.joint_mode = joint_mode
            self.clear11_detach = bool(clear11_detach)
            self.residual12_detach = bool(residual12_detach)
            self.dropout = torch.nn.Dropout(dropout)
            self.binary_classifier = torch.nn.Linear(hidden_size, 2)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.clear11_classifier = torch.nn.Linear(hidden_size, 11)
            self.residual12_classifier = torch.nn.Linear(hidden_size, 1)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def subtype_logits(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            return self.subtype_classifier(self.dropout(pooled))

        def clear11_logits(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            return self.clear11_classifier(self.dropout(pooled))

        def joint_logits(self, binary_logits: Any, subtype_logits: Any) -> Any:
            if self.joint_mode == "hierarchical":
                binary_scores = torch.log_softmax(binary_logits, dim=-1)
                subtype_scores = torch.log_softmax(subtype_logits, dim=-1)
            else:
                binary_scores = binary_logits
                subtype_scores = subtype_logits
            normal_scores = binary_scores[:, :1]
            fraud_scores = binary_scores[:, 1:2] + subtype_scores
            return torch.cat([normal_scores, fraud_scores], dim=-1)

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            hidden = self.dropout(pooled)
            binary_logits = self.binary_classifier(hidden)
            subtype_logits = self.subtype_classifier(hidden)
            clear11_hidden = hidden.detach() if self.clear11_detach else hidden
            clear11_logits = self.clear11_classifier(clear11_hidden)
            residual12_hidden = hidden.detach() if self.residual12_detach else hidden
            residual12_logits = self.residual12_classifier(residual12_hidden).squeeze(-1)
            logits = self.joint_logits(binary_logits, subtype_logits)
            return SimpleNamespace(
                logits=logits,
                binary_logits=binary_logits,
                subtype_logits=subtype_logits,
                clear11_logits=clear11_logits,
                residual12_logits=residual12_logits,
            )

    return V24Clear11ShadowForSequenceClassification()


def build_ce_subtype_aux_model(
    model_name_or_path: str,
    dropout_prob: float,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout

    class CESubtypeAuxForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.binary_classifier = torch.nn.Linear(hidden_size, 2)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def subtype_logits(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            return self.subtype_classifier(self.dropout(pooled))

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            hidden = self.dropout(pooled)
            logits = self.classifier(hidden)
            subtype_logits = self.subtype_classifier(hidden)
            return SimpleNamespace(logits=logits, subtype_logits=subtype_logits)

    return CESubtypeAuxForSequenceClassification()


def build_ce_mid_adapter_model(
    model_name_or_path: str,
    num_labels: int,
    dropout_prob: float,
    gate_bias: float,
    adapter_layer: int,
    adapter_bottleneck: int,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    bottleneck = max(8, int(adapter_bottleneck))
    if adapter_layer < 0:
        layer_index = max(1, num_hidden_layers // 2) if num_hidden_layers else 1
    else:
        layer_index = int(adapter_layer)
    if num_hidden_layers:
        layer_index = min(max(1, layer_index), num_hidden_layers)
    initial_gate_bias = gate_bias if gate_bias >= 0 else -2.0

    class CEMidAdapterForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.adapter_layer = layer_index
            self.dropout = torch.nn.Dropout(dropout)
            self.mid_adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, bottleneck),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck, hidden_size),
            )
            self.mid_gate = torch.nn.Linear(hidden_size, hidden_size)
            self.classifier = torch.nn.Linear(hidden_size, num_labels)
            torch.nn.init.zeros_(self.mid_adapter[-1].weight)
            torch.nn.init.zeros_(self.mid_adapter[-1].bias)
            torch.nn.init.zeros_(self.mid_gate.weight)
            torch.nn.init.constant_(self.mid_gate.bias, initial_gate_bias)

        def pooled_and_mid(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states:
                mid_index = min(self.adapter_layer, len(hidden_states) - 1)
                mid = hidden_states[mid_index][:, 0]
            else:
                mid = outputs.last_hidden_state[:, 0]
            return pooled, mid

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled, mid = self.pooled_and_mid(input_ids, attention_mask)
            delta = self.mid_adapter(mid)
            gate = torch.sigmoid(self.mid_gate(mid))
            return pooled + gate * delta

        def neck_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.pooled_embedding(input_ids, attention_mask)

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            embedding = self.pooled_embedding(input_ids, attention_mask)
            logits = self.classifier(self.dropout(embedding))
            return SimpleNamespace(logits=logits)

    return CEMidAdapterForSequenceClassification()


def build_h2_cascade_model(
    model_name_or_path: str,
    dropout_prob: float,
    adapter_bottleneck: int,
    gate_bias: float,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    bottleneck = max(8, int(adapter_bottleneck))
    initial_gate_bias = gate_bias if gate_bias >= 0 else -1.0

    class H2CascadeForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.binary_classifier = torch.nn.Linear(hidden_size, 2)
            self.fraud_adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, bottleneck),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck, hidden_size),
            )
            self.fraud_gate = torch.nn.Linear(hidden_size, hidden_size)
            self.fraud_norm = torch.nn.LayerNorm(hidden_size)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            torch.nn.init.zeros_(self.fraud_adapter[-1].weight)
            torch.nn.init.zeros_(self.fraud_adapter[-1].bias)
            torch.nn.init.zeros_(self.fraud_gate.weight)
            torch.nn.init.constant_(self.fraud_gate.bias, initial_gate_bias)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def fraud_embedding_from_pooled(self, pooled: Any) -> Any:
            delta = self.fraud_adapter(pooled)
            gate = torch.sigmoid(self.fraud_gate(pooled))
            return self.fraud_norm(pooled + gate * delta)

        def subtype_logits(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            fraud_hidden = self.fraud_embedding_from_pooled(pooled)
            return self.subtype_classifier(self.dropout(fraud_hidden))

        def joint_logits(self, binary_logits: Any, subtype_logits: Any) -> Any:
            binary_scores = torch.log_softmax(binary_logits, dim=-1)
            subtype_scores = torch.log_softmax(subtype_logits, dim=-1)
            normal_scores = binary_scores[:, :1]
            fraud_scores = binary_scores[:, 1:2] + subtype_scores
            return torch.cat([normal_scores, fraud_scores], dim=-1)

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            binary_logits = self.binary_classifier(self.dropout(pooled))
            fraud_hidden = self.fraud_embedding_from_pooled(pooled)
            subtype_logits = self.subtype_classifier(self.dropout(fraud_hidden))
            logits = self.joint_logits(binary_logits, subtype_logits)
            return SimpleNamespace(logits=logits, binary_logits=binary_logits, subtype_logits=subtype_logits)

    return H2CascadeForSequenceClassification()


def protected_two_stage_logits(
    main_logits: Any,
    subtype_logits: Any,
    subtype_weight: float,
    gate_threshold: float,
) -> tuple[Any, Any, Any]:
    """Decode an explicit normal/fraud gate followed by a fraud-only subtype head."""
    import torch

    weight = min(max(float(subtype_weight), 0.0), 1.0)
    threshold = min(max(float(gate_threshold), 0.0), 1.0)
    main_log_probs = torch.log_softmax(main_logits, dim=-1)
    gate_logits = torch.stack(
        [main_log_probs[:, 0], torch.logsumexp(main_log_probs[:, 1:], dim=-1)],
        dim=-1,
    )
    gate_probs = torch.softmax(gate_logits, dim=-1)
    main_subtype_probs = torch.softmax(main_logits[:, 1:], dim=-1)
    adapter_subtype_probs = torch.softmax(subtype_logits, dim=-1)
    blended_subtype_probs = (
        (1.0 - weight) * main_subtype_probs + weight * adapter_subtype_probs
    ).clamp_min(1e-12)
    fraud_mask = gate_probs[:, 1] >= threshold
    decoded_probs = torch.zeros(
        (main_logits.shape[0], NUM_CLASSES),
        dtype=main_logits.dtype,
        device=main_logits.device,
    )
    decoded_probs[:, 0] = (~fraud_mask).to(main_logits.dtype)
    decoded_probs[:, 1:] = fraud_mask[:, None].to(main_logits.dtype) * blended_subtype_probs
    decoded_logits = decoded_probs.clamp_min(1e-12).log()
    return decoded_logits, gate_logits, blended_subtype_probs.log()


def pbal_official_postgate_logits(
    pbal_logits: Any,
    pbal_subtype_log_probs: Any,
    official_subcategory_logits: Any,
    official_weight: float,
) -> tuple[Any, Any]:
    """Fuse official evidence only inside PBAL's already-frozen fraud decision.

    ``official_weight=0`` deliberately returns the original PBAL logits without
    reconstructing them.  This makes the alpha-zero control an exact numerical
    replay of PBAL, rather than merely an equivalent reimplementation.
    """
    import torch

    weight = min(max(float(official_weight), 0.0), 1.0)
    official_parent_probs = official_parent_probabilities(official_subcategory_logits)
    if weight == 0.0:
        return pbal_logits, official_parent_probs

    pbal_subtype_probs = pbal_subtype_log_probs.exp()
    official_fraud_probs = official_parent_probs[:, 1:]
    official_subtype_probs = official_fraud_probs / official_fraud_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    blended_subtype_probs = (
        (1.0 - weight) * pbal_subtype_probs + weight * official_subtype_probs
    ).clamp_min(1e-12)
    blended_subtype_probs = blended_subtype_probs / blended_subtype_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    fraud_mask = pbal_logits.argmax(dim=-1) != 0
    decoded_probs = torch.zeros_like(pbal_logits)
    decoded_probs[:, 0] = (~fraud_mask).to(decoded_probs.dtype)
    decoded_probs[:, 1:] = fraud_mask[:, None].to(decoded_probs.dtype) * blended_subtype_probs
    return decoded_probs.clamp_min(1e-12).log(), official_parent_probs


def build_ce_protected_subtype_model(
    model_name_or_path: str,
    dropout_prob: float,
    adapter_bottleneck: int,
    subtype_weight: float,
    gate_threshold: float,
) -> Any:
    """One encoder, a stable 13-class gate path, and a detached fraud adapter."""
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    bottleneck = max(8, int(adapter_bottleneck))

    class CEProtectedSubtypeForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden_size, NUM_CLASSES)
            self.fraud_adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, bottleneck),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck, hidden_size),
            )
            self.fraud_norm = torch.nn.LayerNorm(hidden_size)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.protected_subtype_weight = float(subtype_weight)
            self.protected_gate_threshold = float(gate_threshold)
            self.protected_main_only_mode = False
            self.protected_main_frozen = False
            torch.nn.init.zeros_(self.fraud_adapter[-1].weight)
            torch.nn.init.zeros_(self.fraud_adapter[-1].bias)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            if self.protected_main_frozen:
                with torch.no_grad():
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
            else:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def fraud_embedding_from_pooled(self, pooled: Any) -> Any:
            # The subtype objective trains only this adapter/head. It cannot move
            # the shared encoder or the main 13-class gate representation.
            protected = pooled.detach()
            return self.fraud_norm(protected + self.fraud_adapter(protected))

        def subtype_logits(self, input_ids: Any, attention_mask: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            fraud_hidden = self.fraud_embedding_from_pooled(pooled)
            return self.subtype_classifier(self.dropout(fraud_hidden))

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            main_logits = self.classifier(
                pooled if self.protected_main_frozen else self.dropout(pooled)
            )
            if self.protected_main_only_mode:
                subtype_logits = torch.zeros(
                    (main_logits.shape[0], NUM_CLASSES - 1),
                    dtype=main_logits.dtype,
                    device=main_logits.device,
                )
            else:
                fraud_hidden = self.fraud_embedding_from_pooled(pooled)
                subtype_logits = self.subtype_classifier(self.dropout(fraud_hidden))
            if self.training:
                # Main CE trains the stable 13-class path; fraud-only CE trains
                # the detached adapter/head through outputs.subtype_logits.
                logits = main_logits
                gate_logits = torch.stack(
                    [main_logits[:, 0], torch.logsumexp(main_logits[:, 1:], dim=-1)],
                    dim=-1,
                )
                protected_subtype_logits = subtype_logits
            else:
                logits, gate_logits, protected_subtype_logits = protected_two_stage_logits(
                    main_logits,
                    subtype_logits,
                    0.0 if self.protected_main_only_mode else self.protected_subtype_weight,
                    self.protected_gate_threshold,
                )
            return SimpleNamespace(
                logits=logits,
                main_logits=main_logits,
                binary_logits=gate_logits,
                subtype_logits=subtype_logits,
                protected_subtype_logits=protected_subtype_logits,
            )

    return CEProtectedSubtypeForSequenceClassification()


def build_ce_pbal_frozen_official_postgate_model(
    model_name_or_path: str,
    dropout_prob: float,
    adapter_bottleneck: int,
    subtype_weight: float,
    official_weight: float,
    gate_threshold: float,
) -> Any:
    """PBAL plus a separately trainable official-child post-gate branch.

    The PBAL modules use exactly the same state-dict names as
    ``ce_protected_subtype`` so a selected PBAL checkpoint can be loaded
    directly.  The official branch is intentionally separate, and its input is
    detached, so it cannot move any parameter used by the PBAL decision.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(
        getattr(config, "classifier_dropout", 0.0)
        or getattr(config, "hidden_dropout_prob", 0.1)
    )
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    bottleneck = max(8, int(adapter_bottleneck))

    class CEPBALFrozenOfficialPostgateForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"
        pbal_frozen_core = True

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            # These five modules are the complete original PBAL path.  Keep
            # their names aligned with ce_protected_subtype for strict loading.
            self.classifier = torch.nn.Linear(hidden_size, NUM_CLASSES)
            self.fraud_adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, bottleneck),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck, hidden_size),
            )
            self.fraud_norm = torch.nn.LayerNorm(hidden_size)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.protected_subtype_weight = float(subtype_weight)
            self.protected_gate_threshold = float(gate_threshold)
            self.protected_official_weight = float(official_weight)
            self.protected_main_frozen = False

            self.official_adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, bottleneck),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck, hidden_size),
            )
            self.official_norm = torch.nn.LayerNorm(hidden_size)
            self.official_subcategory_classifier = torch.nn.Linear(
                hidden_size, NUM_OFFICIAL_SUBCATEGORIES
            )
            torch.nn.init.zeros_(self.fraud_adapter[-1].weight)
            torch.nn.init.zeros_(self.fraud_adapter[-1].bias)
            torch.nn.init.zeros_(self.official_adapter[-1].weight)
            torch.nn.init.zeros_(self.official_adapter[-1].bias)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            if self.protected_main_frozen:
                with torch.no_grad():
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
            else:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            pooled = getattr(outputs, "pooler_output", None)
            return pooled if pooled is not None else outputs.last_hidden_state[:, 0]

        def pbal_subtype_from_pooled(self, pooled: Any) -> Any:
            protected = pooled.detach()
            if self.protected_main_frozen:
                with torch.no_grad():
                    fraud_hidden = self.fraud_norm(protected + self.fraud_adapter(protected))
                    return self.subtype_classifier(fraud_hidden)
            fraud_hidden = self.fraud_norm(protected + self.fraud_adapter(protected))
            return self.subtype_classifier(self.dropout(fraud_hidden))

        def official_embedding_from_pooled(self, pooled: Any) -> Any:
            protected = pooled.detach()
            return self.official_norm(protected + self.official_adapter(protected))

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            main_logits = self.classifier(
                pooled if self.protected_main_frozen else self.dropout(pooled)
            )
            subtype_logits = self.pbal_subtype_from_pooled(pooled)
            official_hidden = self.official_embedding_from_pooled(pooled)
            official_subcategory_logits = self.official_subcategory_classifier(
                self.dropout(official_hidden)
            )
            pbal_logits, gate_logits, pbal_subtype_log_probs = protected_two_stage_logits(
                main_logits,
                subtype_logits,
                self.protected_subtype_weight,
                self.protected_gate_threshold,
            )
            if self.training:
                logits = pbal_logits
                official_parent_probs = official_parent_probabilities(official_subcategory_logits)
            else:
                logits, official_parent_probs = pbal_official_postgate_logits(
                    pbal_logits,
                    pbal_subtype_log_probs,
                    official_subcategory_logits,
                    self.protected_official_weight,
                )
            return SimpleNamespace(
                logits=logits,
                main_logits=main_logits,
                binary_logits=gate_logits,
                subtype_logits=subtype_logits,
                protected_subtype_logits=pbal_subtype_log_probs,
                official_subcategory_logits=official_subcategory_logits,
                prototype_parent_probs=official_parent_probs,
            )

    return CEPBALFrozenOfficialPostgateForSequenceClassification()


def build_ce_protected_top_layers_model(
    model_name_or_path: str,
    dropout_prob: float,
    subtype_weight: float,
    gate_threshold: float,
    fraud_top_layer_count: int,
) -> Any:
    """Frozen main gate plus trainable fraud-only copies of the top encoder layers."""
    import copy
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        local_files_only=True,
    )
    hidden_size = int(config.hidden_size)
    default_dropout = float(
        getattr(config, "classifier_dropout", 0.0)
        or getattr(config, "hidden_dropout_prob", 0.1)
    )
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    encoder_layers = getattr(getattr(base_model, "encoder", None), "layer", None)
    if encoder_layers is None:
        raise ValueError("ce_protected_top_layers requires an encoder.layer stack")
    layer_count = int(fraud_top_layer_count)
    if not 1 <= layer_count <= len(encoder_layers):
        raise ValueError(
            f"fraud_top_layer_count must be in [1, {len(encoder_layers)}], got {layer_count}"
        )

    class CEProtectedTopLayersForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden_size, NUM_CLASSES)
            self.fraud_top_layer_count = layer_count
            self.fraud_top_layers = torch.nn.ModuleList(
                [copy.deepcopy(layer) for layer in encoder_layers[-layer_count:]]
            )
            main_pooler = getattr(base_model, "pooler", None)
            self.fraud_pooler = copy.deepcopy(main_pooler) if main_pooler is not None else None
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.protected_subtype_weight = float(subtype_weight)
            self.protected_gate_threshold = float(gate_threshold)
            self.protected_main_frozen = False

        def initialize_fraud_branch_from_main(self) -> None:
            source_layers = self.base_model.encoder.layer[-self.fraud_top_layer_count :]
            for target, source in zip(self.fraud_top_layers, source_layers):
                target.load_state_dict(source.state_dict())
            main_pooler = getattr(self.base_model, "pooler", None)
            if self.fraud_pooler is not None and main_pooler is not None:
                self.fraud_pooler.load_state_dict(main_pooler.state_dict())

        def base_outputs(self, input_ids: Any, attention_mask: Any) -> Any:
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if self.protected_main_frozen:
                with torch.no_grad():
                    return self.base_model(**kwargs)
            return self.base_model(**kwargs)

        def fraud_logits_from_outputs(self, outputs: Any, attention_mask: Any) -> Any:
            hidden = outputs.hidden_states[-(self.fraud_top_layer_count + 1)].detach()
            extended_attention_mask = self.base_model.get_extended_attention_mask(
                attention_mask,
                attention_mask.shape,
                hidden.device,
            )
            for layer in self.fraud_top_layers:
                hidden = layer(hidden, attention_mask=extended_attention_mask)[0]
            fraud_pooled = (
                self.fraud_pooler(hidden)
                if self.fraud_pooler is not None
                else hidden[:, 0]
            )
            return self.subtype_classifier(self.dropout(fraud_pooled))

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            outputs = self.base_outputs(input_ids, attention_mask)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            main_logits = self.classifier(
                pooled if self.protected_main_frozen else self.dropout(pooled)
            )
            subtype_logits = self.fraud_logits_from_outputs(outputs, attention_mask)
            if self.training:
                logits = main_logits
                gate_logits = torch.stack(
                    [main_logits[:, 0], torch.logsumexp(main_logits[:, 1:], dim=-1)],
                    dim=-1,
                )
                protected_subtype_logits = subtype_logits
            else:
                logits, gate_logits, protected_subtype_logits = protected_two_stage_logits(
                    main_logits,
                    subtype_logits,
                    self.protected_subtype_weight,
                    self.protected_gate_threshold,
                )
            return SimpleNamespace(
                logits=logits,
                main_logits=main_logits,
                binary_logits=gate_logits,
                subtype_logits=subtype_logits,
                protected_subtype_logits=protected_subtype_logits,
            )

    return CEProtectedTopLayersForSequenceClassification()


def official_parent_probabilities(official_subcategory_logits: Any) -> Any:
    """Aggregate normal + 38 official subcategory probabilities into 13 parents."""
    import torch

    official_probs = torch.softmax(official_subcategory_logits, dim=-1)
    parent_probs = torch.zeros(
        (official_subcategory_logits.shape[0], NUM_CLASSES),
        dtype=official_subcategory_logits.dtype,
        device=official_subcategory_logits.device,
    )
    for parent, indices in OFFICIAL_PARENT_TO_INDICES.items():
        parent_probs[:, parent] = official_probs[:, list(indices)].sum(dim=-1)
    return parent_probs / parent_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def protected_official_prototype_logits(
    main_logits: Any,
    official_subcategory_logits: Any,
    official_weight: float,
    gate_threshold: float,
) -> tuple[Any, Any, Any]:
    """Blend a protected 13-class path with child prototypes, then decode explicitly."""
    import torch

    weight = min(max(float(official_weight), 0.0), 1.0)
    threshold = min(max(float(gate_threshold), 0.0), 1.0)
    main_probs = torch.softmax(main_logits, dim=-1)
    prototype_parent_probs = official_parent_probabilities(official_subcategory_logits)
    blended_parent_probs = (
        (1.0 - weight) * main_probs + weight * prototype_parent_probs
    ).clamp_min(1e-12)
    blended_parent_probs = blended_parent_probs / blended_parent_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    fraud_prob = blended_parent_probs[:, 1:].sum(dim=-1)
    gate_probs = torch.stack([1.0 - fraud_prob, fraud_prob], dim=-1).clamp_min(1e-12)
    fraud_parent_probs = blended_parent_probs[:, 1:] / fraud_prob[:, None].clamp_min(1e-12)
    fraud_mask = fraud_prob >= threshold
    decoded_probs = torch.zeros_like(blended_parent_probs)
    decoded_probs[:, 0] = (~fraud_mask).to(decoded_probs.dtype)
    decoded_probs[:, 1:] = fraud_mask[:, None].to(decoded_probs.dtype) * fraud_parent_probs
    return decoded_probs.clamp_min(1e-12).log(), gate_probs.log(), prototype_parent_probs


def protected_official_postgate_logits(
    main_logits: Any,
    official_subcategory_logits: Any,
    official_weight: float,
    gate_threshold: float,
) -> tuple[Any, Any, Any]:
    """Keep the main normal/fraud gate fixed and blend official evidence post-gate."""
    import torch

    weight = min(max(float(official_weight), 0.0), 1.0)
    threshold = min(max(float(gate_threshold), 0.0), 1.0)
    main_log_probs = torch.log_softmax(main_logits, dim=-1)
    gate_logits = torch.stack(
        [main_log_probs[:, 0], torch.logsumexp(main_log_probs[:, 1:], dim=-1)],
        dim=-1,
    )
    gate_probs = torch.softmax(gate_logits, dim=-1)

    main_subtype_probs = torch.softmax(main_logits[:, 1:], dim=-1)
    prototype_parent_probs = official_parent_probabilities(official_subcategory_logits)
    prototype_fraud_probs = prototype_parent_probs[:, 1:]
    prototype_subtype_probs = prototype_fraud_probs / prototype_fraud_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    blended_subtype_probs = (
        (1.0 - weight) * main_subtype_probs + weight * prototype_subtype_probs
    ).clamp_min(1e-12)
    blended_subtype_probs = blended_subtype_probs / blended_subtype_probs.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)

    fraud_mask = gate_probs[:, 1] >= threshold
    decoded_probs = torch.zeros_like(main_logits)
    decoded_probs[:, 0] = (~fraud_mask).to(decoded_probs.dtype)
    decoded_probs[:, 1:] = fraud_mask[:, None].to(decoded_probs.dtype) * blended_subtype_probs
    return decoded_probs.clamp_min(1e-12).log(), gate_logits, prototype_parent_probs


def linear_with_reference_rng_advance(
    in_features: int,
    out_features: int,
    reference_out_features: int,
) -> Any:
    """Build a linear head while advancing global RNG like a reference-sized head."""
    import torch

    if int(reference_out_features) <= 0:
        return torch.nn.Linear(in_features, out_features)
    cpu_rng_state = torch.get_rng_state()
    layer = torch.nn.Linear(in_features, out_features)
    torch.set_rng_state(cpu_rng_state)
    # This unregistered temporary layer consumes exactly the initialization RNG
    # of the reference architecture without changing the returned head.
    torch.nn.Linear(in_features, int(reference_out_features))
    return layer


def build_ce_protected_official_prototypes_model(
    model_name_or_path: str,
    dropout_prob: float,
    adapter_bottleneck: int,
    official_weight: float,
    gate_threshold: float,
    postgate_only: bool = False,
    rng_reference_head_classes: int = 0,
) -> Any:
    """One encoder with a stable 13-class path and detached official child prototypes."""
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    bottleneck = max(8, int(adapter_bottleneck))

    class CEProtectedOfficialPrototypesForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden_size, NUM_CLASSES)
            self.official_adapter = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, bottleneck),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(bottleneck, hidden_size),
            )
            self.official_norm = torch.nn.LayerNorm(hidden_size)
            self.official_subcategory_classifier = linear_with_reference_rng_advance(
                hidden_size,
                NUM_OFFICIAL_SUBCATEGORIES,
                rng_reference_head_classes,
            )
            self.protected_official_weight = float(official_weight)
            self.protected_gate_threshold = float(gate_threshold)
            self.protected_official_postgate_only = bool(postgate_only)
            self.protected_official_rng_reference_head_classes = int(
                rng_reference_head_classes
            )
            torch.nn.init.zeros_(self.official_adapter[-1].weight)
            torch.nn.init.zeros_(self.official_adapter[-1].bias)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def official_embedding_from_pooled(self, pooled: Any) -> Any:
            # The official-child objective trains only this adapter/head. It
            # cannot pull the shared encoder away from the stable 13-class path.
            protected = pooled.detach()
            return self.official_norm(protected + self.official_adapter(protected))

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            main_logits = self.classifier(self.dropout(pooled))
            official_hidden = self.official_embedding_from_pooled(pooled)
            official_subcategory_logits = self.official_subcategory_classifier(
                self.dropout(official_hidden)
            )
            if self.training:
                logits = main_logits
                main_probs = torch.softmax(main_logits, dim=-1)
                fraud_prob = main_probs[:, 1:].sum(dim=-1)
                gate_logits = torch.stack([1.0 - fraud_prob, fraud_prob], dim=-1).clamp_min(1e-12).log()
                prototype_parent_probs = official_parent_probabilities(
                    official_subcategory_logits
                )
            else:
                decode = (
                    protected_official_postgate_logits
                    if self.protected_official_postgate_only
                    else protected_official_prototype_logits
                )
                logits, gate_logits, prototype_parent_probs = (
                    decode(
                        main_logits,
                        official_subcategory_logits,
                        self.protected_official_weight,
                        self.protected_gate_threshold,
                    )
                )
            return SimpleNamespace(
                logits=logits,
                main_logits=main_logits,
                binary_logits=gate_logits,
                official_subcategory_logits=official_subcategory_logits,
                prototype_parent_probs=prototype_parent_probs,
            )

    return CEProtectedOfficialPrototypesForSequenceClassification()


def build_ce_local_expert_model(
    model_name_or_path: str,
    dropout_prob: float,
    groups: list[list[int]],
    residual_scale: float,
    gate_bias: float,
) -> Any:
    import torch
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(model_name_or_path, config=config, local_files_only=True)
    hidden_size = int(config.hidden_size)
    default_dropout = float(getattr(config, "classifier_dropout", 0.0) or getattr(config, "hidden_dropout_prob", 0.1))
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout
    initial_gate_bias = gate_bias if gate_bias >= -20 else -2.0

    class CELocalExpertForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.classifier = torch.nn.Linear(hidden_size, NUM_CLASSES)
            self.local_expert_groups = [list(group) for group in groups]
            self.local_expert_heads = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_size, len(group)) for group in self.local_expert_groups]
            )
            self.local_expert_gates = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_size, 1) for _ in self.local_expert_groups]
            )
            self.local_expert_residual_scale = float(residual_scale)
            for gate in self.local_expert_gates:
                torch.nn.init.zeros_(gate.weight)
                torch.nn.init.constant_(gate.bias, initial_gate_bias)

        def pooled_embedding(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled
            return outputs.last_hidden_state[:, 0]

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            pooled = self.pooled_embedding(input_ids, attention_mask)
            hidden = self.dropout(pooled)
            logits = self.classifier(hidden)
            expert_logits = []
            expert_gates = []
            adjusted_logits = logits.clone()
            for group, head, gate_layer in zip(self.local_expert_groups, self.local_expert_heads, self.local_expert_gates):
                local_logits = head(hidden)
                gate = torch.sigmoid(gate_layer(hidden)) * self.local_expert_residual_scale
                expert_logits.append(local_logits)
                expert_gates.append(gate)
                for local_idx, label in enumerate(group):
                    adjusted_logits[:, label] = adjusted_logits[:, label] + gate[:, 0] * local_logits[:, local_idx]
            return SimpleNamespace(
                logits=adjusted_logits,
                base_logits=logits,
                expert_logits=expert_logits,
                expert_gates=expert_gates,
                expert_groups=self.local_expert_groups,
            )

    return CELocalExpertForSequenceClassification()


def build_ce_selective_pair_expert_model(
    model_name_or_path: str,
    dropout_prob: float,
    groups: list[list[int]],
) -> Any:
    """Build sparse pair specialists with an explicit KEEP-main-model action."""
    import torch
    from transformers import AutoConfig, AutoModel

    if any(len(group) != 2 for group in groups):
        raise ValueError("ce_selective_pair_expert requires two-label groups")
    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        local_files_only=True,
    )
    hidden_size = int(config.hidden_size)
    default_dropout = float(
        getattr(config, "classifier_dropout", 0.0)
        or getattr(config, "hidden_dropout_prob", 0.1)
    )
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout

    class CESelectivePairExpertForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.binary_classifier = torch.nn.Linear(hidden_size, 2)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.selective_pair_groups = [list(group) for group in groups]
            # Per pair: choose left, choose right, or keep the main prediction.
            # Four detached main-head features expose local confidence without
            # letting the specialist loss rewrite the main classifier.
            self.selective_pair_heads = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_size + 4, 3) for _ in groups]
            )

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            hidden = self.dropout(pooled)
            binary_logits = self.binary_classifier(hidden)
            subtype_logits = self.subtype_classifier(hidden)
            logits = torch.cat(
                [binary_logits[:, :1], binary_logits[:, 1:2] + subtype_logits],
                dim=-1,
            )
            detached_logits = logits.detach()
            top2 = torch.topk(detached_logits, k=2, dim=-1).indices
            sorted_top2 = torch.sort(top2, dim=-1).values
            global_margin = (
                torch.topk(detached_logits, k=2, dim=-1).values[:, 0]
                - torch.topk(detached_logits, k=2, dim=-1).values[:, 1]
            )
            pair_logits = []
            pair_routes = []
            for group, head in zip(self.selective_pair_groups, self.selective_pair_heads):
                left, right = group
                local_features = torch.stack(
                    [
                        detached_logits[:, left],
                        detached_logits[:, right],
                        detached_logits[:, right] - detached_logits[:, left],
                        global_margin,
                    ],
                    dim=-1,
                )
                pair_logits.append(head(torch.cat([hidden, local_features], dim=-1)))
                pair_routes.append(
                    (sorted_top2[:, 0] == min(left, right))
                    & (sorted_top2[:, 1] == max(left, right))
                )
            return SimpleNamespace(
                logits=logits,
                base_logits=logits,
                binary_logits=binary_logits,
                subtype_logits=subtype_logits,
                selective_pair_logits=pair_logits,
                selective_pair_routes=pair_routes,
                selective_pair_groups=self.selective_pair_groups,
            )

    return CESelectivePairExpertForSequenceClassification()


def selective_pair_expert_loss(
    outputs: Any,
    labels: Any,
    sample_weight: Any | None,
    loss_fn: Any,
    rescue_weight: float,
) -> Any:
    """Teach each expert to rescue a local error or explicitly keep the main output."""
    import torch

    if not hasattr(outputs, "selective_pair_logits"):
        return outputs.logits.sum() * 0.0
    base_prediction = outputs.base_logits.detach().argmax(dim=-1)
    losses = []
    for group, action_logits, routed in zip(
        outputs.selective_pair_groups,
        outputs.selective_pair_logits,
        outputs.selective_pair_routes,
    ):
        if int(routed.sum().detach().cpu()) <= 0:
            continue
        left, right = group
        routed_labels = labels[routed]
        routed_base = base_prediction[routed]
        targets = torch.full_like(routed_labels, 2)  # action 2 = KEEP
        needs_rescue = routed_base != routed_labels
        targets = torch.where(
            needs_rescue & (routed_labels == left),
            torch.zeros_like(targets),
            targets,
        )
        targets = torch.where(
            needs_rescue & (routed_labels == right),
            torch.ones_like(targets),
            targets,
        )
        per_sample = loss_fn(action_logits[routed], targets)
        action_weight = torch.where(
            targets == 2,
            torch.ones_like(per_sample),
            torch.full_like(per_sample, float(rescue_weight)),
        )
        if sample_weight is not None:
            action_weight = action_weight * sample_weight[routed]
        losses.append(
            (per_sample * action_weight).sum() / action_weight.sum().clamp_min(1e-6)
        )
    if not losses:
        return outputs.logits.sum() * 0.0
    return torch.stack(losses).mean()


def build_ce_pairwise_boundary_expert_model(
    model_name_or_path: str,
    dropout_prob: float,
    groups: list[list[int]],
) -> Any:
    """Build pair classifiers whose gradients cannot rewrite the main encoder."""
    import torch
    from transformers import AutoConfig, AutoModel

    if any(len(group) != 2 for group in groups):
        raise ValueError("ce_pairwise_boundary_expert requires two-label groups")
    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    base_model = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        local_files_only=True,
    )
    hidden_size = int(config.hidden_size)
    default_dropout = float(
        getattr(config, "classifier_dropout", 0.0)
        or getattr(config, "hidden_dropout_prob", 0.1)
    )
    dropout = dropout_prob if dropout_prob >= 0 else default_dropout

    class CEPairwiseBoundaryExpertForSequenceClassification(torch.nn.Module):
        base_model_prefix = "base_model"

        def __init__(self) -> None:
            super().__init__()
            self.base_model = base_model
            self.config = config
            self.dropout = torch.nn.Dropout(dropout)
            self.binary_classifier = torch.nn.Linear(hidden_size, 2)
            self.subtype_classifier = torch.nn.Linear(hidden_size, NUM_CLASSES - 1)
            self.pairwise_boundary_groups = [list(group) for group in groups]
            self.pairwise_boundary_heads = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_size + 4, 2) for _ in groups]
            )

        def forward(self, input_ids: Any, attention_mask: Any, **_: Any) -> Any:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                pooled = outputs.last_hidden_state[:, 0]
            hidden = self.dropout(pooled)
            binary_logits = self.binary_classifier(hidden)
            subtype_logits = self.subtype_classifier(hidden)
            logits = torch.cat(
                [binary_logits[:, :1], binary_logits[:, 1:2] + subtype_logits],
                dim=-1,
            )
            detached_logits = logits.detach()
            top_values, top_indices = torch.topk(detached_logits, k=2, dim=-1)
            sorted_top2 = torch.sort(top_indices, dim=-1).values
            global_margin = top_values[:, 0] - top_values[:, 1]
            boundary_logits = []
            boundary_routes = []
            for group, head in zip(
                self.pairwise_boundary_groups,
                self.pairwise_boundary_heads,
            ):
                left, right = group
                local_features = torch.stack(
                    [
                        detached_logits[:, left],
                        detached_logits[:, right],
                        detached_logits[:, right] - detached_logits[:, left],
                        global_margin,
                    ],
                    dim=-1,
                )
                # The specialist learns on a detached shared representation;
                # only the ordinary dual-head losses update the main path.
                boundary_logits.append(
                    head(torch.cat([hidden.detach(), local_features], dim=-1))
                )
                boundary_routes.append(
                    (sorted_top2[:, 0] == min(left, right))
                    & (sorted_top2[:, 1] == max(left, right))
                )
            return SimpleNamespace(
                logits=logits,
                base_logits=logits,
                binary_logits=binary_logits,
                subtype_logits=subtype_logits,
                pairwise_boundary_logits=boundary_logits,
                pairwise_boundary_routes=boundary_routes,
                pairwise_boundary_groups=self.pairwise_boundary_groups,
            )

    return CEPairwiseBoundaryExpertForSequenceClassification()


def pairwise_boundary_expert_loss(
    outputs: Any,
    labels: Any,
    sample_weight: Any | None,
    loss_fn: Any,
) -> Any:
    """Train each A/B specialist on every audited row belonging to that pair."""
    import torch

    if not hasattr(outputs, "pairwise_boundary_logits"):
        return outputs.logits.sum() * 0.0
    losses = []
    for group, logits in zip(
        outputs.pairwise_boundary_groups,
        outputs.pairwise_boundary_logits,
    ):
        left, right = group
        mask = (labels == left) | (labels == right)
        if int(mask.sum().detach().cpu()) <= 0:
            continue
        targets = (labels[mask] == right).long()
        per_sample = loss_fn(logits[mask], targets)
        if sample_weight is not None:
            weights = sample_weight[mask]
            losses.append(
                (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
            )
        else:
            losses.append(per_sample.mean())
    if not losses:
        return outputs.logits.sum() * 0.0
    return torch.stack(losses).mean()


def local_expert_loss(
    outputs: Any,
    labels: Any,
    confusable: Any,
    sample_weight: Any | None,
    loss_fn: Any,
) -> Any:
    import torch

    if not hasattr(outputs, "expert_logits"):
        return outputs.logits.sum() * 0.0
    losses = []
    for group, logits in zip(outputs.expert_groups, outputs.expert_logits):
        mask = torch.zeros_like(labels, dtype=torch.bool)
        label_in_group = torch.zeros_like(labels, dtype=torch.bool)
        for label in group:
            mask = mask | (labels == int(label)) | (confusable == int(label))
            label_in_group = label_in_group | (labels == int(label))
        # Keep each expert local: only rows whose true class belongs to the group
        # can define an in-group target. Confusable-only rows still reach the
        # global/confusable-margin losses.
        target_mask = mask & label_in_group
        if int(target_mask.sum().detach().cpu()) <= 0:
            continue
        local_labels = torch.zeros_like(labels[target_mask])
        for local_idx, label in enumerate(group):
            local_labels = torch.where(labels[target_mask] == int(label), torch.tensor(local_idx, device=labels.device), local_labels)
        per_sample = loss_fn(logits[target_mask], local_labels)
        if sample_weight is not None:
            weights = sample_weight[target_mask]
            losses.append((per_sample * weights).sum() / weights.sum().clamp_min(1e-6))
        else:
            losses.append(per_sample.mean())
    if not losses:
        return outputs.logits.sum() * 0.0
    return torch.stack(losses).mean()


def typed_evidence_contrastive_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    temperature: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    mask = batch["has_typed_evidence"].to(device)
    if int(mask.sum().detach().cpu()) <= 1:
        return torch.zeros((), device=device)
    text_cls = cls_embedding(model, batch["input_ids"].to(device), batch["attention_mask"].to(device))[mask]
    evidence_cls = cls_embedding(
        model,
        batch["typed_evidence_input_ids"].to(device),
        batch["typed_evidence_attention_mask"].to(device),
    )[mask]
    text_cls = F.normalize(text_cls, dim=-1)
    evidence_cls = F.normalize(evidence_cls, dim=-1)
    logits = text_cls @ evidence_cls.T / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def typed_evidence_axis_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    loss_fn: Any,
    confusable_margin_weight: float,
    confusable_margin: float,
    mode: str,
    allowed_labels: set[int] | None = None,
    target: str = "label13",
) -> Any:
    import torch
    import torch.nn.functional as F

    mask = batch["has_typed_evidence"].to(device)
    if allowed_labels is not None:
        label_mask = torch.zeros_like(mask)
        batch_labels = batch["label13"].to(device)
        for label in allowed_labels:
            label_mask = label_mask | (batch_labels == int(label))
        mask = mask & label_mask
    if target == "fraud_subtype":
        mask = mask & (batch["label13"].to(device) != 0)
    if int(mask.sum().detach().cpu()) <= 0:
        return torch.zeros((), device=device)
    input_ids = batch["typed_evidence_input_ids"].to(device)[mask]
    attention_mask = batch["typed_evidence_attention_mask"].to(device)[mask]
    if target == "fraud_subtype":
        if not hasattr(model, "subtype_logits"):
            raise ValueError("typed evidence fraud_subtype target requires model.subtype_logits")
        labels = batch["label13"].to(device)[mask] - 1
        logits = model.subtype_logits(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        if confusable_margin_weight > 0:
            confusable = batch["confusable_label13"].to(device)[mask] - 1
            valid = (confusable >= 0) & (confusable < (NUM_CLASSES - 1)) & (confusable != labels)
            if int(valid.sum().detach().cpu()) > 0:
                row_idx = torch.arange(logits.shape[0], device=logits.device)[valid]
                true_logits = logits[row_idx, labels[valid]]
                confusable_logits = logits[row_idx, confusable[valid]]
                confusable_loss = torch.relu(
                    torch.tensor(float(confusable_margin), device=logits.device) - (true_logits - confusable_logits)
                ).mean()
                loss = loss + confusable_margin_weight * confusable_loss
        return loss

    labels = batch["label13"].to(device)[mask]
    if mode == "full":
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    elif mode == "head_stopgrad":
        embedding = cls_embedding(model, input_ids, attention_mask).detach()
        logits = classifier_logits_from_embedding(model, embedding)
    else:
        raise ValueError(f"unsupported typed evidence axis mode: {mode}")
    per_sample_loss = loss_fn(logits, labels)
    loss = per_sample_loss.mean()
    if confusable_margin_weight > 0:
        confusable_loss = confusable_margin_loss(
            logits,
            labels,
            batch["confusable_label13"].to(device)[mask],
            confusable_margin,
        )
        loss = loss + confusable_margin_weight * confusable_loss
    return loss


def typed_evidence_consistency_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    text_logits: Any,
    temperature: float,
    mode: str,
    allowed_labels: set[int] | None = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    mask = batch["has_typed_evidence"].to(device)
    if allowed_labels is not None:
        label_mask = torch.zeros_like(mask)
        batch_labels = batch["label13"].to(device)
        for label in allowed_labels:
            label_mask = label_mask | (batch_labels == int(label))
        mask = mask & label_mask
    if int(mask.sum().detach().cpu()) <= 0:
        return torch.zeros((), device=device)
    input_ids = batch["typed_evidence_input_ids"].to(device)[mask]
    attention_mask = batch["typed_evidence_attention_mask"].to(device)[mask]
    if mode == "full":
        evidence_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    elif mode == "head_stopgrad":
        embedding = cls_embedding(model, input_ids, attention_mask).detach()
        evidence_logits = classifier_logits_from_embedding(model, embedding)
    else:
        raise ValueError(f"unsupported typed evidence consistency mode: {mode}")
    temperature = max(float(temperature), 1e-6)
    text_log_probs = F.log_softmax(text_logits[mask] / temperature, dim=-1)
    evidence_probs = F.softmax(evidence_logits.detach() / temperature, dim=-1)
    return F.kl_div(text_log_probs, evidence_probs, reduction="batchmean") * (temperature**2)


def ce_gate_alignment_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    allowed_labels: set[int] | None = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    if not hasattr(model, "neck_embedding"):
        return torch.zeros((), device=device)
    mask = batch["has_typed_evidence"].to(device)
    if allowed_labels is not None:
        label_mask = torch.zeros_like(mask)
        batch_labels = batch["label13"].to(device)
        for label in allowed_labels:
            label_mask = label_mask | (batch_labels == int(label))
        mask = mask & label_mask
    if int(mask.sum().detach().cpu()) <= 0:
        return torch.zeros((), device=device)
    text_neck = model.neck_embedding(batch["input_ids"].to(device), batch["attention_mask"].to(device))[mask]
    evidence_neck = model.neck_embedding(
        batch["typed_evidence_input_ids"].to(device),
        batch["typed_evidence_attention_mask"].to(device),
    )[mask].detach()
    return (1.0 - F.cosine_similarity(text_neck, evidence_neck, dim=-1)).mean()


def span_region_embeddings(
    model: Any,
    batch: dict[str, Any],
    device: str,
    allowed_labels: set[int] | None = None,
) -> tuple[Any, Any, Any, Any]:
    import torch

    mask = batch["has_span_region"].to(device)
    labels = batch["label13"].to(device)
    if allowed_labels is not None:
        label_mask = torch.zeros_like(mask)
        for label in allowed_labels:
            label_mask = label_mask | (labels == int(label))
        mask = mask & label_mask
    if int(mask.sum().detach().cpu()) <= 0:
        empty = torch.empty(0, 1, device=device)
        return empty, empty, labels[mask], mask

    if not hasattr(model, "base_model_prefix"):
        raise ValueError("span-region CE requires model.base_model_prefix")
    base_model = getattr(model, model.base_model_prefix)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden = outputs.last_hidden_state
    token_mask = (batch["span_region_mask"].to(device) > 0) & (attention_mask > 0)
    valid_token_count = token_mask.sum(dim=1)
    mask = mask & (valid_token_count > 0)
    if int(mask.sum().detach().cpu()) <= 0:
        empty = torch.empty(0, hidden.shape[-1], device=device)
        return empty, empty, labels[mask], mask
    weights = token_mask.float()
    region = (hidden * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = hidden[:, 0]
    return pooled[mask], region[mask], labels[mask], mask


def span_region_axis_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    loss_fn: Any,
    allowed_labels: set[int] | None = None,
) -> Any:
    import torch

    _, region, labels, _ = span_region_embeddings(model, batch, device, allowed_labels)
    if region.numel() == 0:
        return torch.zeros((), device=device)
    logits = classifier_logits_from_embedding(model, region)
    per_sample_loss = loss_fn(logits, labels)
    return per_sample_loss.mean()


def span_region_contrastive_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    temperature: float,
    allowed_labels: set[int] | None = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    text_cls, region, _, _ = span_region_embeddings(model, batch, device, allowed_labels)
    if region.shape[0] <= 1:
        return torch.zeros((), device=device)
    text_cls = F.normalize(text_cls, dim=-1)
    region = F.normalize(region, dim=-1)
    logits = text_cls @ region.T / max(float(temperature), 1e-6)
    labels = torch.arange(logits.shape[0], device=device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def span_region_consistency_loss(
    model: Any,
    batch: dict[str, Any],
    device: str,
    text_logits: Any,
    temperature: float,
    allowed_labels: set[int] | None = None,
) -> Any:
    import torch
    import torch.nn.functional as F

    _, region, _, mask = span_region_embeddings(model, batch, device, allowed_labels)
    if region.numel() == 0:
        return torch.zeros((), device=device)
    region_logits = classifier_logits_from_embedding(model, region)
    temperature = max(float(temperature), 1e-6)
    text_log_probs = F.log_softmax(text_logits[mask] / temperature, dim=-1)
    region_probs = F.softmax(region_logits.detach() / temperature, dim=-1)
    return F.kl_div(text_log_probs, region_probs, reduction="batchmean") * (temperature**2)


def prf_by_label(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == label and p == label)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != label and p == label)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == label and p != label)
        support = sum(1 for y in y_true if y == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(y_true),
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normal_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def metrics_from_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    observed = sorted(set(y_true))
    all_prf = prf_by_label(y_true, y_pred, list(range(NUM_CLASSES)))
    observed_prf = prf_by_label(y_true, y_pred, observed)
    observed_macro_f1 = (
        sum(item["f1"] for item in observed_prf.values()) / len(observed_prf) if observed_prf else 0.0
    )
    y_binary = [1 if y != 0 else 0 for y in y_true]
    pred_binary = [1 if p != 0 else 0 for p in y_pred]
    fraud_indices = [idx for idx, y in enumerate(y_true) if y != 0]
    fraud_true = [y_true[idx] for idx in fraud_indices]
    fraud_pred = [y_pred[idx] for idx in fraud_indices]
    fraud_observed = sorted(set(fraud_true))
    fraud_prf = prf_by_label(fraud_true, fraud_pred, fraud_observed)
    fraud_macro_f1 = sum(item["f1"] for item in fraud_prf.values()) / len(fraud_prf) if fraud_prf else 0.0
    target_prior_weighted_accuracy = 0.0
    for label in range(NUM_CLASSES):
        label_indices = [idx for idx, y in enumerate(y_true) if y == label]
        if not label_indices:
            continue
        prior = 0.5 if label == 0 else 1.0 / 24.0
        label_recall = sum(1 for idx in label_indices if y_pred[idx] == label) / len(label_indices)
        target_prior_weighted_accuracy += prior * label_recall
    return {
        "n": len(y_true),
        "label_13_accuracy": sum(1 for y, p in zip(y_true, y_pred) if y == p) / len(y_true) if y_true else 0.0,
        "target_prior_weighted_accuracy": target_prior_weighted_accuracy,
        "observed_label_macro_f1": observed_macro_f1,
        "observed_labels": observed,
        "binary": binary_metrics(y_binary, pred_binary),
        "fraud_only_subtype_accuracy": (
            sum(1 for y, p in zip(fraud_true, fraud_pred) if y == p) / len(fraud_true) if fraud_true else 0.0
        ),
        "fraud_only_subtype_macro_f1": fraud_macro_f1,
        "fraud_observed_labels": fraud_observed,
        "per_class_prf": all_prf,
        "label_13_counts": dict(sorted(Counter(str(y) for y in y_true).items(), key=lambda item: int(item[0]))),
        "pred_label_13_counts": dict(sorted(Counter(str(p) for p in y_pred).items(), key=lambda item: int(item[0]))),
    }


def metrics_from_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["true_label_13"]) for row in rows]
    y_pred = [int(row["pred_label_13"]) for row in rows]
    metrics = metrics_from_predictions(y_true, y_pred)
    forced_fraud_rows = [
        row
        for row in rows
        if int(row["true_label_13"]) != 0
        and row.get("pred_fraud_head_forced_12class") is not None
    ]
    if forced_fraud_rows:
        forced_true = [int(row["true_label_13"]) for row in forced_fraud_rows]
        forced_pred = [
            int(row["pred_fraud_head_forced_12class"])
            for row in forced_fraud_rows
        ]
        observed = sorted(set(forced_true))
        per_class = prf_by_label(forced_true, forced_pred, list(range(1, NUM_CLASSES)))
        metrics["fraud_head_forced_12class"] = {
            "n": len(forced_fraud_rows),
            "accuracy": sum(
                true_label == pred_label
                for true_label, pred_label in zip(forced_true, forced_pred)
            )
            / len(forced_fraud_rows),
            "macro_f1_observed": sum(per_class[str(label)]["f1"] for label in observed)
            / len(observed),
            "observed_labels": observed,
            "per_class_prf": {
                str(label): per_class[str(label)] for label in range(1, NUM_CLASSES)
            },
        }
    official_rows = [
        row
        for row in rows
        if int(row.get("true_official_subcategory_index", -100)) >= 0
        and int(row.get("pred_official_subcategory_index", -100)) >= 0
    ]
    if official_rows:
        official_true = [int(row["true_official_subcategory_index"]) for row in official_rows]
        official_pred = [int(row["pred_official_subcategory_index"]) for row in official_rows]
        observed = sorted(set(official_true))
        per_class = prf_by_label(
            official_true,
            official_pred,
            list(range(NUM_OFFICIAL_SUBCATEGORIES)),
        )
        metrics["official_subcategory"] = {
            "n": len(official_rows),
            "accuracy": sum(y == p for y, p in zip(official_true, official_pred)) / len(official_rows),
            "observed_codes": [OFFICIAL_SUBCATEGORY_CODES[idx] for idx in observed],
            "macro_f1_observed": sum(per_class[str(idx)]["f1"] for idx in observed) / len(observed),
            "per_code_prf": {
                OFFICIAL_SUBCATEGORY_CODES[idx]: per_class[str(idx)]
                for idx in range(NUM_OFFICIAL_SUBCATEGORIES)
            },
        }
        true_parent_rows = [
            row
            for row in official_rows
            if int(row.get("pred_official_subcategory_true_parent_index", -100)) >= 0
        ]
        if true_parent_rows:
            metrics["official_subcategory"]["true_parent_conditioned_accuracy"] = sum(
                int(row["true_official_subcategory_index"])
                == int(row["pred_official_subcategory_true_parent_index"])
                for row in true_parent_rows
            ) / len(true_parent_rows)
    return metrics


def prediction_error_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = [[0 for _ in range(NUM_CLASSES)] for _ in range(NUM_CLASSES)]
    pair_counts: Counter[str] = Counter()
    per_class: dict[str, dict[str, Any]] = {
        str(label): {"support": 0, "correct": 0, "errors": 0, "recall": 0.0}
        for label in range(NUM_CLASSES)
    }
    for row in rows:
        true_label = int(row["true_label_13"])
        pred_label = int(row["pred_label_13"])
        confusion[true_label][pred_label] += 1
        per_class[str(true_label)]["support"] += 1
        if true_label == pred_label:
            per_class[str(true_label)]["correct"] += 1
        else:
            per_class[str(true_label)]["errors"] += 1
            pair_counts[f"{true_label}->{pred_label}"] += 1
    for label in range(NUM_CLASSES):
        item = per_class[str(label)]
        item["recall"] = item["correct"] / item["support"] if item["support"] else 0.0
    weak_labels = [1, 4, 6, 7, 10, 12]
    fraud_rows = [row for row in rows if int(row["true_label_13"]) != 0]
    fraud_errors = [row for row in fraud_rows if int(row["true_label_13"]) != int(row["pred_label_13"])]
    fraud_false_negative_rows = [
        row for row in rows if int(row["true_label_13"]) != 0 and int(row["pred_label_13"]) == 0
    ]
    return {
        "n": len(rows),
        "error_count": sum(1 for row in rows if int(row["true_label_13"]) != int(row["pred_label_13"])),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "top_error_pairs": [
            {"pair": pair, "count": count}
            for pair, count in pair_counts.most_common(30)
        ],
        "weak_label_recalls": {
            str(label): per_class[str(label)]["recall"]
            for label in weak_labels
        },
        "fraud_error_count": len(fraud_errors),
        "fraud_false_negative_count": len(fraud_false_negative_rows),
        "normal_false_positive_count": sum(
            1 for row in rows if int(row["true_label_13"]) == 0 and int(row["pred_label_13"]) != 0
        ),
    }


def write_prediction_artifacts(output: Path, stem: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    predictions_dir = output / "predictions"
    analysis_dir = output / "error_analysis"
    csv_path = predictions_dir / f"{stem}_predictions.csv"
    jsonl_path = predictions_dir / f"{stem}_predictions.jsonl"
    report_path = analysis_dir / f"{stem}_error_report.json"
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    write_json(report_path, prediction_error_report(rows))
    return {
        "predictions_csv": str(csv_path),
        "predictions_jsonl": str(jsonl_path),
        "error_report_json": str(report_path),
    }


def _safe_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(int(row["true_label_13"]) == int(row["pred_label_13"]) for row in rows)
    return correct / len(rows)


def _safe_recall(rows: list[dict[str, Any]], label: int) -> float:
    subset = [row for row in rows if int(row["true_label_13"]) == label]
    if not subset:
        return 0.0
    hits = sum(int(row["pred_label_13"]) == label for row in subset)
    return hits / len(subset)


def _safe_precision(rows: list[dict[str, Any]], label: int) -> float:
    subset = [row for row in rows if int(row["pred_label_13"]) == label]
    if not subset:
        return 0.0
    return sum(int(row["true_label_13"]) == label for row in subset) / len(subset)


def compute_task_v21_proxy_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[str(row.get("val_group") or "unknown")].append(row)
    hard_near_domain_rows = [
        *group_rows.get("near_domain_normal", []),
        *group_rows.get("high_risk_normal_0_to_3_7_12", []),
    ]
    thchs_like_rows = group_rows.get("thchs_like_ood_normal", [])
    fraud_rows = [row for row in rows if int(row["true_label_13"]) != 0]
    acc13 = _safe_accuracy(rows)
    hard_near_domain_normal_acc = _safe_accuracy(hard_near_domain_rows)
    fraud_subtype_acc = _safe_accuracy(fraud_rows)
    label1_recall = _safe_recall(rows, 1)
    label6_recall = _safe_recall(rows, 6)
    label12_recall = _safe_recall(rows, 12)
    thchs_like_ood_normal_acc = _safe_accuracy(thchs_like_rows)
    weak_label_recall_min = min(label1_recall, label6_recall, label12_recall)
    selection_score = (
        0.30 * acc13
        + 0.25 * hard_near_domain_normal_acc
        + 0.20 * fraud_subtype_acc
        + 0.15 * weak_label_recall_min
        + 0.10 * thchs_like_ood_normal_acc
    )
    gate = {
        "hard_near_domain_normal_acc": hard_near_domain_normal_acc >= 0.94,
        "label1_recall": label1_recall >= 0.70,
        "label6_recall": label6_recall >= 0.78,
        "label12_recall": label12_recall >= 0.75,
        "thchs_like_ood_normal_acc": thchs_like_ood_normal_acc >= 0.96,
    }
    return {
        "selection_score": selection_score,
        "acc13": acc13,
        "hard_near_domain_normal_acc": hard_near_domain_normal_acc,
        "fraud_subtype_acc": fraud_subtype_acc,
        "label1_recall": label1_recall,
        "label6_recall": label6_recall,
        "label12_recall": label12_recall,
        "weak_label_recall_min": weak_label_recall_min,
        "thchs_like_ood_normal_acc": thchs_like_ood_normal_acc,
        "gate": gate,
        "gate_pass": all(gate.values()),
        "group_counts": {key: len(value) for key, value in sorted(group_rows.items())},
        "hard_near_domain_count": len(hard_near_domain_rows),
        "fraud_count": len(fraud_rows),
        "thchs_like_ood_normal_count": len(thchs_like_rows),
    }


def compute_task_v23_proxy_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[str(row.get("val_group") or "unknown")].append(row)
    hard_normal_rows = group_rows.get("hard_near_domain_normal", [])
    thchs_proxy_rows = group_rows.get("THCHS_proxy", [])
    fraud_rows = [row for row in rows if int(row["true_label_13"]) != 0]
    acc13 = _safe_accuracy(rows)
    hard_normal_acc = _safe_accuracy(hard_normal_rows)
    fraud_subtype_acc = _safe_accuracy(fraud_rows)
    label1_recall = _safe_recall(rows, 1)
    label6_recall = _safe_recall(rows, 6)
    label12_recall = _safe_recall(rows, 12)
    thchs_proxy_acc = _safe_accuracy(thchs_proxy_rows)
    weak_label_recall_min = min(label1_recall, label6_recall, label12_recall)
    selection_score = (
        0.30 * acc13
        + 0.20 * hard_normal_acc
        + 0.20 * fraud_subtype_acc
        + 0.15 * weak_label_recall_min
        + 0.10 * label12_recall
        + 0.05 * thchs_proxy_acc
    )
    gate = {
        "hard_normal_acc": hard_normal_acc >= 0.94,
        "fraud_subtype_acc": fraud_subtype_acc >= 0.87,
        "label12_recall": label12_recall >= 0.82,
        "label6_recall": label6_recall >= 0.75,
        "label1_recall": label1_recall >= 0.70,
    }
    return {
        "selection_score": selection_score,
        "acc13": acc13,
        "hard_normal_acc": hard_normal_acc,
        "fraud_subtype_acc": fraud_subtype_acc,
        "label1_recall": label1_recall,
        "label6_recall": label6_recall,
        "label12_recall": label12_recall,
        "weak_label_recall_min": weak_label_recall_min,
        "THCHS_proxy_acc": thchs_proxy_acc,
        "gate": gate,
        "gate_pass": all(gate.values()),
        "group_counts": {key: len(value) for key, value in sorted(group_rows.items())},
    }


def compute_task_v22_repair_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select task-22 repair checkpoints using validated guard signals only.

    The generated label-8/label-12 boundary proxy is intentionally excluded
    from hard gates because its epoch ordering was empirically anti-correlated
    with the fixed external evaluation. External rows remain report-only.
    """
    group_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_rows[str(row.get("val_group") or "unknown")].append(row)
    hard_normal_rows = group_rows.get("hard_near_domain_normal", [])
    thchs_proxy_rows = group_rows.get("THCHS_proxy", [])
    fraud_rows = [row for row in rows if int(row["true_label_13"]) != 0]
    values = {
        "acc13": _safe_accuracy(rows),
        "hard_normal_acc": _safe_accuracy(hard_normal_rows),
        "THCHS_proxy_acc": _safe_accuracy(thchs_proxy_rows),
        "fraud_subtype_acc": _safe_accuracy(fraud_rows),
        "fraud_recall": sum(int(row["pred_label_13"]) != 0 for row in fraud_rows) / len(fraud_rows) if fraud_rows else 0.0,
        "label1_recall": _safe_recall(rows, 1),
        "label6_recall": _safe_recall(rows, 6),
    }
    selection_score = (
        0.30 * values["hard_normal_acc"]
        + 0.20 * values["acc13"]
        + 0.20 * values["fraud_subtype_acc"]
        + 0.10 * values["fraud_recall"]
        + 0.10 * values["label1_recall"]
        + 0.05 * values["label6_recall"]
        + 0.05 * values["THCHS_proxy_acc"]
    )
    gate = {
        "hard_normal_acc": values["hard_normal_acc"] >= 0.94,
        "THCHS_proxy_acc": values["THCHS_proxy_acc"] >= 0.90,
        "fraud_recall": values["fraud_recall"] >= 0.90,
        "label1_recall": values["label1_recall"] >= 0.70,
        "label6_recall": values["label6_recall"] >= 0.75,
    }
    return {
        "selection_score": selection_score,
        **values,
        "gate": gate,
        "gate_pass": all(gate.values()),
        "group_counts": {key: len(value) for key, value in sorted(group_rows.items())},
    }


def compute_task_v24_proxy_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: groups[str(row.get("val_group") or "unknown")].append(row)
    hard=groups.get("hard_near_domain_normal",[]); thchs=groups.get("THCHS_proxy",[]); fraud=[r for r in rows if int(r["true_label_13"])!=0]
    values={"acc13":_safe_accuracy(rows),"hard_normal_acc":_safe_accuracy(hard),"fraud_subtype_acc":_safe_accuracy(fraud),"label1_recall":_safe_recall(rows,1),"label3_precision":_safe_precision(rows,3),"label6_recall":_safe_recall(rows,6),"label7_precision":_safe_precision(rows,7),"label8_recall":_safe_recall(rows,8),"label12_recall":_safe_recall(rows,12),"THCHS_proxy_acc":_safe_accuracy(thchs)}
    floor=min(values[k] for k in ("label1_recall","label3_precision","label6_recall","label12_recall"))
    score=.25*values["acc13"]+.20*values["hard_normal_acc"]+.20*values["fraud_subtype_acc"]+.15*floor+.10*values["label7_precision"]+.05*values["label8_recall"]+.05*values["THCHS_proxy_acc"]
    gates={"hard_normal_acc":values["hard_normal_acc"]>=.94,"fraud_subtype_acc":values["fraud_subtype_acc"]>=.88,"label1_recall":values["label1_recall"]>=.72,"label3_precision":values["label3_precision"]>=.82,"label6_recall":values["label6_recall"]>=.80,"label7_precision":values["label7_precision"]>=.85,"label8_recall":values["label8_recall"]>=.90,"label12_recall":values["label12_recall"]>=.82,"THCHS_proxy_acc":values["THCHS_proxy_acc"]>=.96}
    return {"selection_score":score,**values,"weak_boundary_floor":floor,"gate":gates,"gate_pass":all(gates.values()),"group_counts":{k:len(v) for k,v in sorted(groups.items())}}


def compute_binary_robust_proxy_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a binary gate only from internal, pre-declared validation groups."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("val_group") or "unknown")].append(row)
    hard_normal = groups.get("hard_near_domain_normal", [])
    thchs_proxy = groups.get("THCHS_proxy", [])
    fraud = [row for row in rows if int(row["true_label_13"]) != 0]

    def normal_accuracy(group: list[dict[str, Any]]) -> float:
        return (
            sum(int(row["pred_label_13"]) == 0 for row in group) / len(group)
            if group
            else 0.0
        )

    fraud_recall = (
        sum(int(row["pred_label_13"]) != 0 for row in fraud) / len(fraud)
        if fraud
        else 0.0
    )
    hard_normal_accuracy = normal_accuracy(hard_normal)
    thchs_proxy_accuracy = normal_accuracy(thchs_proxy)
    binary_accuracy = metrics_from_prediction_rows(rows)["binary"]["accuracy"]
    score = (
        0.40 * fraud_recall
        + 0.30 * hard_normal_accuracy
        + 0.20 * thchs_proxy_accuracy
        + 0.10 * binary_accuracy
    )
    gate = {
        "fraud_recall": fraud_recall >= 0.95,
        "hard_normal_accuracy": hard_normal_accuracy >= 0.95,
        "THCHS_proxy_accuracy": thchs_proxy_accuracy >= 0.95,
    }
    return {
        "selection_score": score,
        "fraud_recall": fraud_recall,
        "hard_normal_accuracy": hard_normal_accuracy,
        "THCHS_proxy_accuracy": thchs_proxy_accuracy,
        "binary_accuracy": binary_accuracy,
        "gate": gate,
        "gate_pass": all(gate.values()),
        "group_counts": {key: len(value) for key, value in sorted(groups.items())},
    }


def predict_rows(
    model: Any,
    dataset: TextDataset,
    batch_size: int,
    device: str,
    sampler: Any | None = None,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, collate_fn=collate)
    rows: list[dict[str, Any]] = []
    attribute_model = model.module if hasattr(model, "module") else model
    official_subcategory_loss_mode = getattr(
        attribute_model, "official_subcategory_loss_mode", "flat"
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            logits = outputs.logits.detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).tolist()
            output_classes = int(probs.shape[-1])
            top_probs, top_labels = torch.topk(probs, k=min(5, output_classes), dim=-1)
            clear11_probs = None
            clear11_top_probs = None
            clear11_top_indices = None
            if hasattr(outputs, "clear11_logits"):
                clear11_probs = torch.softmax(outputs.clear11_logits.detach().cpu(), dim=-1)
                clear11_top_probs, clear11_top_indices = torch.topk(clear11_probs, k=2, dim=-1)
            residual12_probs = None
            if hasattr(outputs, "residual12_logits"):
                residual12_probs = torch.sigmoid(outputs.residual12_logits.detach().cpu())
            forced_fraud_head_logits = None
            forced_fraud_head_probs = None
            forced_fraud_head_preds = None
            if getattr(outputs, "subtype_logits", None) is not None:
                forced_fraud_head_logits = outputs.subtype_logits.detach().cpu()
                forced_fraud_head_probs = torch.softmax(
                    forced_fraud_head_logits, dim=-1
                )
                forced_fraud_head_preds = (
                    forced_fraud_head_probs.argmax(dim=-1) + 1
                ).tolist()
            binary_head_logits = None
            binary_head_probs = None
            if getattr(outputs, "binary_logits", None) is not None:
                binary_head_logits = outputs.binary_logits.detach().cpu()
                binary_head_probs = torch.softmax(binary_head_logits, dim=-1)
            official_subcategory_preds = None
            official_subcategory_probs = None
            official_subcategory_logits = None
            if getattr(outputs, "official_subcategory_logits", None) is not None:
                official_subcategory_logits = outputs.official_subcategory_logits.detach().cpu()
                official_subcategory_probs = torch.softmax(
                    official_subcategory_logits, dim=-1
                )
                official_subcategory_preds = official_subcategory_probs.argmax(dim=-1).tolist()
            selective_pair_groups = getattr(outputs, "selective_pair_groups", None)
            selective_pair_probs = None
            selective_pair_routes = None
            if getattr(outputs, "selective_pair_logits", None) is not None:
                selective_pair_probs = [
                    torch.softmax(value.detach().cpu(), dim=-1)
                    for value in outputs.selective_pair_logits
                ]
                selective_pair_routes = [
                    value.detach().cpu() for value in outputs.selective_pair_routes
                ]
            pairwise_boundary_groups = getattr(outputs, "pairwise_boundary_groups", None)
            pairwise_boundary_probs = None
            pairwise_boundary_routes = None
            if getattr(outputs, "pairwise_boundary_logits", None) is not None:
                pairwise_boundary_probs = [
                    torch.softmax(value.detach().cpu(), dim=-1)
                    for value in outputs.pairwise_boundary_logits
                ]
                pairwise_boundary_routes = [
                    value.detach().cpu() for value in outputs.pairwise_boundary_routes
                ]
            labels = batch["label13"].tolist()
            for idx, (label, pred) in enumerate(zip(labels, preds)):
                label_int = int(label)
                pred_int = int(pred)
                row = {
                    "sample_id": batch["sample_id"][idx],
                    "base_sample_id": batch["base_sample_id"][idx],
                    "view_id": batch["view_id"][idx],
                    "split": batch["split"][idx],
                    "source": batch["source"][idx],
                    "text": batch["text"][idx],
                    "source_group": batch["source_group"][idx],
                    "val_group": batch["val_group"][idx],
                    "true_label_13": label_int,
                    "pred_label_13": pred_int,
                    "true_binary": 1 if label_int != 0 else 0,
                    "pred_binary_from_label_13": 1 if pred_int != 0 else 0,
                    "score_fraud": float(1.0 - probs[idx, 0]),
                    "p_pred": float(probs[idx, pred_int]),
                }
                for label_id in range(output_classes):
                    row[f"logit_{label_id}"] = float(logits[idx, label_id])
                    row[f"prob_{label_id}"] = float(probs[idx, label_id])
                if (
                    selective_pair_groups is not None
                    and selective_pair_probs is not None
                    and selective_pair_routes is not None
                ):
                    for group, action_probs, routed in zip(
                        selective_pair_groups,
                        selective_pair_probs,
                        selective_pair_routes,
                    ):
                        left, right = (int(group[0]), int(group[1]))
                        prefix = f"pair_expert_{left}_{right}"
                        action_index = int(action_probs[idx].argmax())
                        row[f"{prefix}_routed"] = bool(routed[idx])
                        row[f"{prefix}_prob_left"] = float(action_probs[idx, 0])
                        row[f"{prefix}_prob_right"] = float(action_probs[idx, 1])
                        row[f"{prefix}_prob_keep"] = float(action_probs[idx, 2])
                        row[f"{prefix}_action"] = (
                            str(left)
                            if action_index == 0
                            else str(right)
                            if action_index == 1
                            else "keep"
                        )
                if (
                    pairwise_boundary_groups is not None
                    and pairwise_boundary_probs is not None
                    and pairwise_boundary_routes is not None
                ):
                    for group, pair_probs, routed in zip(
                        pairwise_boundary_groups,
                        pairwise_boundary_probs,
                        pairwise_boundary_routes,
                    ):
                        left, right = (int(group[0]), int(group[1]))
                        prefix = f"pair_boundary_{left}_{right}"
                        row[f"{prefix}_routed"] = bool(routed[idx])
                        row[f"{prefix}_prob_left"] = float(pair_probs[idx, 0])
                        row[f"{prefix}_prob_right"] = float(pair_probs[idx, 1])
                        row[f"{prefix}_candidate"] = int(
                            left if pair_probs[idx, 0] >= pair_probs[idx, 1] else right
                        )
                if binary_head_logits is not None and binary_head_probs is not None:
                    for binary_label in range(int(binary_head_probs.shape[-1])):
                        row[f"binary_logit_{binary_label}"] = float(
                            binary_head_logits[idx, binary_label]
                        )
                        row[f"binary_prob_{binary_label}"] = float(
                            binary_head_probs[idx, binary_label]
                        )
                    row["pred_binary_head"] = int(binary_head_probs[idx].argmax())
                    if int(binary_head_probs.shape[-1]) == 2:
                        row["p_binary_fraud"] = float(binary_head_probs[idx, 1])
                for rank, (top_label, top_prob) in enumerate(zip(top_labels[idx].tolist(), top_probs[idx].tolist()), start=1):
                    clipped = min(max(float(top_prob), 1e-12), 1.0)
                    row[f"top{rank}_label"] = int(top_label)
                    row[f"top{rank}_prob"] = float(top_prob)
                    row[f"top{rank}_logprob"] = float(torch.log(torch.tensor(clipped)).item())
                if output_classes >= 2:
                    row["top1_top2_margin"] = float(top_probs[idx, 0] - top_probs[idx, 1])
                if clear11_probs is not None and clear11_top_probs is not None and clear11_top_indices is not None:
                    clear11_top_label = int(clear11_top_indices[idx, 0]) + 1
                    clear11_second_label = int(clear11_top_indices[idx, 1]) + 1
                    clear11_top_score = float(clear11_top_probs[idx, 0])
                    clear11_second_score = float(clear11_top_probs[idx, 1])
                    row["clear11_top_label"] = clear11_top_label
                    row["clear11_top_score"] = clear11_top_score
                    row["clear11_second_label"] = clear11_second_label
                    row["clear11_second_score"] = clear11_second_score
                    row["clear11_margin"] = float(clear11_top_score - clear11_second_score)
                    row["clear11_shadow_pred"] = clear11_top_label
                    entropy = -(clear11_probs[idx] * torch.log(clear11_probs[idx].clamp_min(1e-12))).sum()
                    row["clear11_entropy"] = float(entropy)
                    for offset in range(11):
                        label_id = offset + 1
                        row[f"clear11_prob_{label_id}"] = float(clear11_probs[idx, offset])
                if residual12_probs is not None:
                    p_residual12 = float(residual12_probs[idx])
                    row["p_residual12"] = p_residual12
                    row["residual12_candidate"] = bool(p_residual12 >= 0.60)
                    row["residual12_shadow_pred"] = 12 if p_residual12 >= 0.60 else 0
                    row["residual12_reason"] = "p_residual12_ge_0.60" if p_residual12 >= 0.60 else "p_residual12_lt_0.60"
                if (
                    forced_fraud_head_logits is not None
                    and forced_fraud_head_probs is not None
                    and forced_fraud_head_preds is not None
                ):
                    forced_pred = int(forced_fraud_head_preds[idx])
                    row["pred_fraud_head_forced_12class"] = forced_pred
                    row["p_fraud_head_forced_top1"] = float(
                        forced_fraud_head_probs[idx, forced_pred - 1]
                    )
                    for subtype_offset in range(int(forced_fraud_head_probs.shape[-1])):
                        subtype_label = subtype_offset + 1
                        row[f"subtype_logit_{subtype_label}"] = float(
                            forced_fraud_head_logits[idx, subtype_offset]
                        )
                        row[f"subtype_prob_{subtype_label}"] = float(
                            forced_fraud_head_probs[idx, subtype_offset]
                        )
                if (
                    official_subcategory_preds is not None
                    and official_subcategory_probs is not None
                    and official_subcategory_logits is not None
                ):
                    true_official_idx = int(batch["official_subcategory_index"][idx])
                    true_parent_pred_official_idx = int(official_subcategory_preds[idx])
                    if official_subcategory_loss_mode == "sibling_masked":
                        predicted_siblings = OFFICIAL_PARENT_TO_INDICES[pred_int]
                        predicted_sibling_logits = official_subcategory_logits[
                            idx, list(predicted_siblings)
                        ]
                        predicted_local_index = int(predicted_sibling_logits.argmax())
                        pred_official_idx = int(predicted_siblings[predicted_local_index])
                        pred_official_prob = float(
                            torch.softmax(predicted_sibling_logits, dim=-1)[predicted_local_index]
                        )
                        true_siblings = OFFICIAL_PARENT_TO_INDICES[label_int]
                        true_sibling_logits = official_subcategory_logits[idx, list(true_siblings)]
                        true_parent_local_index = int(true_sibling_logits.argmax())
                        true_parent_pred_official_idx = int(
                            true_siblings[true_parent_local_index]
                        )
                    else:
                        pred_official_idx = int(official_subcategory_preds[idx])
                        pred_official_prob = float(
                            official_subcategory_probs[idx, pred_official_idx]
                        )
                    row["true_official_subcategory_index"] = true_official_idx
                    row["true_official_subcategory_code"] = (
                        OFFICIAL_SUBCATEGORY_CODES[true_official_idx]
                        if true_official_idx >= 0
                        else ""
                    )
                    row["pred_official_subcategory_index"] = pred_official_idx
                    row["pred_official_subcategory_code"] = OFFICIAL_SUBCATEGORY_CODES[pred_official_idx]
                    row["pred_official_subcategory_prob"] = pred_official_prob
                    row["pred_official_subcategory_true_parent_index"] = true_parent_pred_official_idx
                    row["pred_official_subcategory_true_parent_code"] = OFFICIAL_SUBCATEGORY_CODES[
                        true_parent_pred_official_idx
                    ]
                rows.append(row)
    return rows


def predict(model: Any, dataset: TextDataset, batch_size: int, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = predict_rows(model, dataset, batch_size, device)
    return metrics_from_prediction_rows(rows), rows


def parse_eval_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--eval-csv must be name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"empty eval name in {spec!r}")
        parsed.append((name, Path(path)))
    return parsed


def parse_label_filter(value: str) -> set[int] | None:
    if not value.strip():
        return None
    labels: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        label = int(item)
        if not 0 <= label < NUM_CLASSES:
            raise SystemExit(f"--typed-evidence-labels out of range: {label}")
        labels.add(label)
    if not labels:
        return None
    return labels


def parse_probability_grid(value: str) -> list[float]:
    """Parse a deterministic, de-duplicated [0, 1] fusion-weight grid."""
    if not str(value).strip():
        return []
    parsed: list[float] = []
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        probability = float(item)
        if not 0.0 <= probability <= 1.0:
            raise SystemExit("--protected-official-alpha-grid values must be in [0, 1]")
        if probability not in parsed:
            parsed.append(probability)
    if not parsed:
        raise SystemExit("--protected-official-alpha-grid contained no usable values")
    return parsed


def parse_local_expert_groups(value: str) -> list[list[int]]:
    groups: list[list[int]] = []
    normalized = str(value or "").replace("|", ";").replace("/", ";")
    for raw_group in normalized.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        group: list[int] = []
        for item in raw_group.split(","):
            item = item.strip()
            if not item:
                continue
            label = int(item)
            if not 0 <= label < NUM_CLASSES:
                raise SystemExit(f"local expert group label out of range: {label}")
            if label not in group:
                group.append(label)
        if len(group) < 2:
            raise SystemExit(f"local expert group must contain at least two labels: {raw_group!r}")
        if group not in groups:
            groups.append(group)
    if not groups:
        raise SystemExit("--local-expert-groups produced no groups")
    return groups


def build_model_from_args(
    args: Any,
    local_expert_groups: list[list[int]],
    device: str,
) -> Any:
    from transformers import AutoModelForSequenceClassification

    if args.architecture == "ce_gated_neck":
        model = build_ce_gated_neck_model(
            args.model,
            NUM_CLASSES,
            args.ce_neck_dropout,
            args.ce_neck_gate_bias,
        )
    elif args.architecture in {"ce_dual_head", "ce_dual_head_hier", "ce_dual_head_h2"}:
        joint_mode = (
            "hierarchical"
            if args.architecture in {"ce_dual_head_hier", "ce_dual_head_h2"}
            else "additive"
        )
        model = build_ce_dual_head_model(
            args.model,
            args.ce_neck_dropout,
            joint_mode=joint_mode,
            official_subcategory_aux=args.official_subcategory_loss_weight > 0,
            flat_auxiliary=args.architecture == "ce_dual_head_h2",
        )
    elif args.architecture == "v24_clear11_shadow":
        model = build_v24_clear11_shadow_model(
            args.model,
            args.ce_neck_dropout,
            joint_mode="additive",
            clear11_detach=args.clear11_detach,
        )
    elif args.architecture == "v24_clear11_residual12_shadow":
        model = build_v24_clear11_shadow_model(
            args.model,
            args.ce_neck_dropout,
            joint_mode="additive",
            clear11_detach=args.clear11_detach,
            residual12_detach=args.residual12_detach,
        )
    elif args.architecture == "ce_subtype_aux":
        model = build_ce_subtype_aux_model(args.model, args.ce_neck_dropout)
    elif args.architecture == "ce_mid_adapter":
        model = build_ce_mid_adapter_model(
            args.model,
            NUM_CLASSES,
            args.ce_neck_dropout,
            args.ce_neck_gate_bias,
            args.ce_mid_adapter_layer,
            args.ce_mid_adapter_bottleneck,
        )
    elif args.architecture == "h2_cascade":
        model = build_h2_cascade_model(
            args.model,
            args.ce_neck_dropout,
            args.ce_mid_adapter_bottleneck,
            args.ce_neck_gate_bias,
        )
    elif args.architecture == "ce_protected_subtype":
        model = build_ce_protected_subtype_model(
            args.model,
            args.ce_neck_dropout,
            args.ce_mid_adapter_bottleneck,
            args.protected_subtype_weight,
            args.protected_gate_threshold,
        )
    elif args.architecture == "ce_pbal_frozen_official_postgate":
        model = build_ce_pbal_frozen_official_postgate_model(
            args.model,
            args.ce_neck_dropout,
            args.ce_mid_adapter_bottleneck,
            args.protected_subtype_weight,
            args.protected_official_weight,
            args.protected_gate_threshold,
        )
    elif args.architecture == "ce_protected_top_layers":
        model = build_ce_protected_top_layers_model(
            args.model,
            args.ce_neck_dropout,
            args.protected_subtype_weight,
            args.protected_gate_threshold,
            args.protected_fraud_top_layers,
        )
    elif args.architecture in {
        "ce_protected_official_prototypes",
        "ce_protected_official_postgate",
    }:
        model = build_ce_protected_official_prototypes_model(
            args.model,
            args.ce_neck_dropout,
            args.ce_mid_adapter_bottleneck,
            args.protected_official_weight,
            args.protected_gate_threshold,
            postgate_only=args.architecture == "ce_protected_official_postgate",
            rng_reference_head_classes=args.protected_official_rng_reference_head_classes,
        )
    elif args.architecture == "ce_local_expert":
        model = build_ce_local_expert_model(
            args.model,
            args.ce_neck_dropout,
            local_expert_groups,
            args.local_expert_residual_scale,
            args.local_expert_gate_bias,
        )
    elif args.architecture == "ce_selective_pair_expert":
        model = build_ce_selective_pair_expert_model(
            args.model,
            args.ce_neck_dropout,
            local_expert_groups,
        )
    elif args.architecture == "ce_pairwise_boundary_expert":
        model = build_ce_pairwise_boundary_expert_model(
            args.model,
            args.ce_neck_dropout,
            local_expert_groups,
        )
    elif args.architecture == "true_binary_sequence_classifier":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2,
            local_files_only=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=NUM_CLASSES,
            local_files_only=True,
        )
    setattr(model, "official_subcategory_loss_mode", args.official_subcategory_loss_mode)
    return model.to(device)


def maybe_enable_gradient_checkpointing(model: Any) -> None:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "config"):
            setattr(base_model.config, "use_cache", False)
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            setattr(model.config, "use_cache", False)


def maybe_freeze_base_model(model: Any) -> None:
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        base_model = getattr(model, getattr(model, "base_model_prefix", ""), None)
    if base_model is None:
        return
    for parameter in base_model.parameters():
        parameter.requires_grad = False


def freeze_base_bottom_encoder_layers(model: Any, layer_count: int) -> dict[str, int]:
    """Freeze embeddings and the first ``layer_count`` encoder blocks.

    The pooler and upper blocks remain trainable.  The returned layout is
    recorded in run metrics so partial-finetuning runs are auditable.
    """
    attribute_model = model.module if hasattr(model, "module") else model
    base_model = getattr(attribute_model, "base_model", None)
    if base_model is None:
        base_model = getattr(
            attribute_model,
            getattr(attribute_model, "base_model_prefix", ""),
            None,
        )
    if base_model is None:
        raise ValueError("partial base freezing requires a base_model")
    embeddings = getattr(base_model, "embeddings", None)
    encoder_layers = getattr(getattr(base_model, "encoder", None), "layer", None)
    if embeddings is None or encoder_layers is None:
        raise ValueError(
            "partial base freezing requires base_model.embeddings and base_model.encoder.layer"
        )
    total_layers = len(encoder_layers)
    if not 1 <= layer_count < total_layers:
        raise ValueError(
            f"freeze_base_bottom_layers must be in [1, {total_layers - 1}], got {layer_count}"
        )
    frozen_parameter_count = 0
    for module in [embeddings, *encoder_layers[:layer_count]]:
        for parameter in module.parameters():
            parameter.requires_grad = False
            frozen_parameter_count += int(parameter.numel())
        module.eval()
    setattr(attribute_model, "frozen_base_bottom_layers", int(layer_count))
    return {
        "encoder_layers_total": total_layers,
        "encoder_layers_frozen": int(layer_count),
        "frozen_parameter_count": frozen_parameter_count,
    }


def enforce_frozen_base_bottom_layers_eval(model: Any) -> None:
    """Keep partial-frozen blocks deterministic after ``model.train()``."""
    attribute_model = model.module if hasattr(model, "module") else model
    layer_count = int(getattr(attribute_model, "frozen_base_bottom_layers", 0))
    if layer_count <= 0:
        return
    base_model = getattr(attribute_model, "base_model", None)
    embeddings = getattr(base_model, "embeddings", None)
    encoder_layers = getattr(getattr(base_model, "encoder", None), "layer", None)
    if embeddings is None or encoder_layers is None or layer_count > len(encoder_layers):
        raise RuntimeError("partial-frozen encoder structure changed after initialization")
    embeddings.eval()
    for layer in encoder_layers[:layer_count]:
        layer.eval()


def protected_main_parameters(model: Any) -> list[tuple[str, Any]]:
    """Return the immutable inference path for a protected-stage model."""
    attribute_model = model.module if hasattr(model, "module") else model
    prefixes = ("base_model.", "classifier.")
    if bool(getattr(attribute_model, "pbal_frozen_core", False)):
        prefixes = prefixes + (
            "fraud_adapter.",
            "fraud_norm.",
            "subtype_classifier.",
        )
    return [
        (name, parameter)
        for name, parameter in attribute_model.named_parameters()
        if name.startswith(prefixes)
    ]


def protected_main_state_sha256(model: Any) -> str:
    """Stable content hash used to prove that stage B did not move the gate."""
    import hashlib
    import torch

    digest = hashlib.sha256()
    for name, parameter in protected_main_parameters(model):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def should_track_protected_main_hash(architecture: str) -> bool:
    """Whether the architecture exposes the protected binary-gate path."""
    return architecture in SUBTYPE_AUX_ARCHITECTURES | {
        "ce_pbal_frozen_official_postgate"
    }


def freeze_protected_main_path(model: Any) -> None:
    """Freeze every parameter and stochastic module in the protected path."""
    attribute_model = model.module if hasattr(model, "module") else model
    if not hasattr(attribute_model, "base_model") or not hasattr(attribute_model, "classifier"):
        raise ValueError("protected main-path freezing requires base_model and classifier")
    for _name, parameter in protected_main_parameters(attribute_model):
        parameter.requires_grad = False
    setattr(attribute_model, "protected_main_frozen", True)
    attribute_model.base_model.eval()
    attribute_model.classifier.eval()
    if bool(getattr(attribute_model, "pbal_frozen_core", False)):
        attribute_model.fraud_adapter.eval()
        attribute_model.fraud_norm.eval()
        attribute_model.subtype_classifier.eval()


def enforce_protected_main_eval(model: Any) -> None:
    """Keep the frozen encoder/head in eval mode after a surrounding model.train()."""
    attribute_model = model.module if hasattr(model, "module") else model
    if not bool(getattr(attribute_model, "protected_main_frozen", False)):
        return
    attribute_model.base_model.eval()
    attribute_model.classifier.eval()
    if bool(getattr(attribute_model, "pbal_frozen_core", False)):
        attribute_model.fraud_adapter.eval()
        attribute_model.fraud_norm.eval()
        attribute_model.subtype_classifier.eval()


def distributed_predict(
    model: Any,
    dataset: TextDataset,
    batch_size: int,
    device: str,
    world_size: int,
    rank: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch.distributed as dist

    sampler = RankShardSampler(len(dataset), world_size, rank)
    local_rows = predict_rows(model, dataset, batch_size, device, sampler=sampler)
    gathered: list[list[dict[str, Any]] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_rows)
    all_rows: list[dict[str, Any]] = []
    if rank == 0:
        for shard in gathered:
            if shard:
                all_rows.extend(shard)
        metrics = metrics_from_prediction_rows(all_rows)
    else:
        metrics = None
    metrics = distributed_broadcast_object(metrics, rank)
    return metrics, (all_rows if rank == 0 else [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-csv", action="append", default=[], help="Evaluation CSV as name=path.")
    parser.add_argument(
        "--selection-eval-csv",
        default="",
        help="Optional frozen target-domain dev CSV used only for label-13 ACC checkpoint selection.",
    )
    parser.add_argument(
        "--selection-eval-name",
        default="checkpoint_selection_dev",
        help="Artifact name recorded for --selection-eval-csv.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--artifact-output",
        default="",
        help="Optional directory for large artifacts; metrics are mirrored back to --output.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--architecture",
        choices=[
            "sequence_classifier",
            "true_binary_sequence_classifier",
            "ce_gated_neck",
            "ce_dual_head",
            "ce_dual_head_hier",
            "ce_dual_head_h2",
            "v24_clear11_shadow",
            "v24_clear11_residual12_shadow",
            "ce_subtype_aux",
            "ce_mid_adapter",
            "h2_cascade",
            "ce_protected_subtype",
            "ce_protected_top_layers",
            "ce_protected_official_prototypes",
            "ce_protected_official_postgate",
            "ce_pbal_frozen_official_postgate",
            "ce_local_expert",
            "ce_selective_pair_expert",
            "ce_pairwise_boundary_expert",
        ],
        default="sequence_classifier",
    )
    parser.add_argument("--ce-neck-dropout", type=float, default=-1.0, help="Negative means use the pretrained config dropout.")
    parser.add_argument("--ce-neck-gate-bias", type=float, default=-1.0)
    parser.add_argument("--ce-mid-adapter-layer", type=int, default=-1, help="Negative selects the middle encoder layer.")
    parser.add_argument("--ce-mid-adapter-bottleneck", type=int, default=256)
    parser.add_argument("--protected-subtype-weight", type=float, default=0.45)
    parser.add_argument("--protected-fraud-top-layers", type=int, default=2)
    parser.add_argument("--protected-official-weight", type=float, default=0.30)
    parser.add_argument(
        "--protected-official-alpha-grid",
        default="",
        help="Comma-separated alpha candidates selected only by final internal label-13 ACC.",
    )
    parser.add_argument("--protected-official-rng-reference-head-classes", type=int, default=0)
    parser.add_argument("--protected-gate-threshold", type=float, default=0.5)
    parser.add_argument("--ce-gate-alignment-weight", type=float, default=0.0)
    parser.add_argument("--local-expert-groups", default=DEFAULT_LOCAL_EXPERT_GROUPS)
    parser.add_argument("--local-expert-group-preset", choices=["custom", "g1", "g12", "g123"], default="g123")
    parser.add_argument("--local-expert-loss-weight", type=float, default=0.0)
    parser.add_argument("--local-expert-residual-scale", type=float, default=0.5)
    parser.add_argument("--local-expert-gate-bias", type=float, default=-2.0)
    parser.add_argument("--local-expert-rescue-weight", type=float, default=3.0)
    parser.add_argument("--local-expert-warmup-epochs", type=int, default=1)
    parser.add_argument("--dual-binary-loss-weight", type=float, default=0.0)
    parser.add_argument("--dual-subtype-loss-weight", type=float, default=0.0)
    parser.add_argument("--dual-subtype-class-weight", choices=["none", "balanced", "sqrt_balanced"], default="none")
    parser.add_argument("--dual-subtype-logit-adjust-tau", type=float, default=0.0)
    parser.add_argument(
        "--h2-flat-aux-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary flat 13-class CE weight for ce_dual_head_h2 only.",
    )
    parser.add_argument(
        "--h2-binary-consistency-weight",
        type=float,
        default=0.0,
        help="Normal/fraud mass consistency weight between H2 hierarchy and flat auxiliary head.",
    )
    parser.add_argument("--official-subcategory-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--official-subcategory-loss-mode",
        choices=["flat", "sibling_masked"],
        default="flat",
    )
    parser.add_argument(
        "--official-subcategory-class-weight",
        choices=["none", "sqrt_balanced"],
        default="none",
    )
    parser.add_argument("--official-subcategory-class-weight-cap", type=float, default=3.0)
    parser.add_argument("--clear11-loss-weight", type=float, default=0.0)
    parser.add_argument("--clear11-class-weight", choices=["none", "balanced", "sqrt_balanced"], default="none")
    parser.add_argument("--clear11-detach", action="store_true", help="Detach shared representation before clear11 head/loss.")
    parser.add_argument("--residual12-loss-weight", type=float, default=0.0)
    parser.add_argument("--residual12-pos-weight", type=float, default=0.0, help="Optional BCE pos_weight for label12 residual head; <=0 disables.")
    parser.add_argument("--residual12-detach", action="store_true", help="Detach shared representation before residual12 head/loss.")
    parser.add_argument(
        "--binary-aux-jsonl",
        default="",
        help="Optional binary-only auxiliary training JSONL. It does not enter validation/evaluation.",
    )
    parser.add_argument("--binary-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--binary-aux-batch-size", type=int, default=0, help="Defaults to --batch-size when <= 0.")
    parser.add_argument("--distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--distill-prob-field", default="teacher_probs")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-split-mode", choices=["random", "pair", "scenario", "pair_scenario"], default="random")
    parser.add_argument("--val-id-file", default="", help="Optional training-side sample-id file for a fixed validation split.")
    parser.add_argument(
        "--val-jsonl",
        default="",
        help="Optional independent validation JSONL. Its rows never enter training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none", "sqrt_balanced", "balanced_cap_1.5"],
        default="balanced",
    )
    parser.add_argument("--class-weight-cap", type=float, default=0.0, help="Optional positive cap applied after class-weight construction.")
    parser.add_argument("--label12-weight-cap", type=float, default=0.0, help="Optional positive cap applied to label-12 weight.")
    parser.add_argument("--normal-weight-scale", type=float, default=1.0)
    parser.add_argument("--sample-weight-field", default="", help="Optional JSONL field used as per-sample loss weight.")
    parser.add_argument("--pair-margin-weight", type=float, default=0.0)
    parser.add_argument("--pair-margin", type=float, default=1.0)
    parser.add_argument("--confusable-margin-weight", type=float, default=0.0)
    parser.add_argument("--confusable-margin", type=float, default=0.35)
    parser.add_argument("--typed-evidence-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--typed-evidence-axis-weight", type=float, default=0.0)
    parser.add_argument("--typed-evidence-axis-mode", choices=["full", "head_stopgrad"], default="full")
    parser.add_argument("--typed-evidence-axis-target", choices=["label13", "fraud_subtype"], default="label13")
    parser.add_argument("--typed-evidence-axis-class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--typed-evidence-axis-confusable-margin-weight", type=float, default=0.0)
    parser.add_argument("--typed-evidence-axis-confusable-margin", type=float, default=0.35)
    parser.add_argument("--typed-evidence-consistency-weight", type=float, default=0.0)
    parser.add_argument("--typed-evidence-consistency-mode", choices=["full", "head_stopgrad"], default="head_stopgrad")
    parser.add_argument("--typed-evidence-labels", default="", help="Optional comma-separated label_13 filter for typed-evidence losses.")
    parser.add_argument("--typed-evidence-temperature", type=float, default=0.07)
    parser.add_argument("--typed-evidence-max-length", type=int, default=128)
    parser.add_argument("--typed-evidence-warmup-epochs", type=int, default=1)
    parser.add_argument("--span-region-ce-axis-weight", type=float, default=0.0)
    parser.add_argument("--span-region-ce-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--span-region-ce-consistency-weight", type=float, default=0.0)
    parser.add_argument("--span-region-ce-labels", default="", help="Optional comma-separated label_13 filter for span-region CE losses.")
    parser.add_argument("--span-region-ce-temperature", type=float, default=0.07)
    parser.add_argument("--span-region-ce-warmup-epochs", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--freeze-base-model", action="store_true")
    parser.add_argument(
        "--freeze-base-bottom-layers",
        type=int,
        default=0,
        help="Freeze embeddings plus this many bottom encoder layers while tuning upper layers.",
    )
    parser.add_argument(
        "--init-protected-checkpoint",
        default="",
        help="Optional ce_protected_subtype state_dict used to initialize a frozen stage-B run.",
    )
    parser.add_argument(
        "--freeze-protected-main-path",
        action="store_true",
        help="Freeze encoder and 13-class classifier and keep them in eval mode.",
    )
    parser.add_argument(
        "--reinitialize-protected-fraud-branch",
        action="store_true",
        help="After loading a protected checkpoint, restore a fresh fraud adapter/norm/head initialization.",
    )
    parser.add_argument(
        "--protected-main-only-mode",
        action="store_true",
        help="Disable the fraud adapter/head so stage A trains and selects only the protected main path.",
    )
    parser.add_argument(
        "--train-fraud-only",
        action="store_true",
        help="Filter the training view to fully labeled fraud rows while preserving validation/evaluation sets.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip-save-best-model", action="store_true", help="Keep the best checkpoint in memory only.")
    parser.add_argument(
        "--fixed-final-epoch",
        action="store_true",
        help="Use the final scheduled epoch without validation-based checkpoint selection.",
    )
    parser.add_argument(
        "--rdrop-kl-weight",
        type=float,
        default=0.0,
        help="Symmetric KL weight between two dropout forwards of the same batch.",
    )
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument(
        "--checkpoint-bank-only",
        action="store_true",
        help=(
            "Train a fixed epoch schedule and save every epoch checkpoint without "
            "running checkpoint selection or final evaluation."
        ),
    )
    parser.add_argument("--weight-averaging-mode", choices=["none", "ema", "swa"], default="none")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--weight-averaging-start-epoch", type=int, default=2)
    parser.add_argument(
        "--selection-score",
        choices=[
            "legacy_macro_binary",
            "target_prior_weighted_accuracy",
            "label_13_accuracy",
            "fpr_constrained_label13_accuracy",
            "task_v21_proxy_selection",
            "task_v23_proxy_selection",
            "task_v24_proxy_selection",
            "task_v22_repair_selection",
            "binary_robust_proxy_selection",
            "fraud_head_forced_macro_f1",
        ],
        default="legacy_macro_binary",
    )
    parser.add_argument(
        "--selection-normal-fpr-max",
        type=float,
        default=0.12,
        help="Hard normal-FPR ceiling used by fpr_constrained_label13_accuracy.",
    )
    args = parser.parse_args()
    if args.fixed_final_epoch and args.early_stopping_patience != 0:
        raise SystemExit("--fixed-final-epoch requires --early-stopping-patience 0")
    if args.val_jsonl and args.val_id_file:
        raise SystemExit("--val-jsonl and --val-id-file are mutually exclusive")
    if args.selection_eval_csv and args.selection_score not in {
        "label_13_accuracy",
        "fpr_constrained_label13_accuracy",
    }:
        raise SystemExit(
            "--selection-eval-csv requires label_13_accuracy or fpr_constrained_label13_accuracy"
        )
    if not 0.0 <= args.selection_normal_fpr_max <= 1.0:
        raise SystemExit("--selection-normal-fpr-max must be in [0, 1]")
    if args.h2_flat_aux_loss_weight < 0 or args.h2_binary_consistency_weight < 0:
        raise SystemExit("H2 loss weights must be non-negative")
    if args.architecture == "ce_dual_head_h2" and args.h2_flat_aux_loss_weight <= 0:
        raise SystemExit("ce_dual_head_h2 requires --h2-flat-aux-loss-weight > 0")
    if args.architecture != "ce_dual_head_h2" and (
        args.h2_flat_aux_loss_weight > 0 or args.h2_binary_consistency_weight > 0
    ):
        raise SystemExit("H2 loss weights require --architecture ce_dual_head_h2")
    if args.checkpoint_bank_only:
        if not args.save_epoch_checkpoints:
            raise SystemExit("--checkpoint-bank-only requires --save-epoch-checkpoints")
        if args.selection_eval_csv:
            raise SystemExit(
                "--checkpoint-bank-only must not receive --selection-eval-csv; "
                "selection is deferred to a future frozen dev set"
            )
        if args.weight_averaging_mode != "none":
            raise SystemExit(
                "--checkpoint-bank-only currently requires --weight-averaging-mode none"
            )
    if args.selection_score == "task_v21_proxy_selection" and not args.val_jsonl:
        raise SystemExit("task_v21_proxy_selection requires --val-jsonl")
    if args.selection_score == "task_v23_proxy_selection" and not (args.val_jsonl or args.val_id_file):
        raise SystemExit("task_v23_proxy_selection requires an explicit internal validation split")
    if args.selection_score == "task_v24_proxy_selection" and not (args.val_jsonl or args.val_id_file):
        raise SystemExit("task_v24_proxy_selection requires an explicit internal validation split")
    if args.selection_score == "task_v22_repair_selection" and not (args.val_jsonl or args.val_id_file):
        raise SystemExit("task_v22_repair_selection requires an explicit internal validation split")
    if args.selection_score == "binary_robust_proxy_selection" and not args.val_jsonl:
        raise SystemExit("binary_robust_proxy_selection requires --val-jsonl")
    if args.architecture == "true_binary_sequence_classifier":
        if args.class_weight != "none":
            raise SystemExit("true_binary_sequence_classifier currently requires --class-weight none")
        if args.rdrop_kl_weight < 0:
            raise SystemExit("--rdrop-kl-weight must be non-negative")
    if args.freeze_base_bottom_layers < 0:
        raise SystemExit("--freeze-base-bottom-layers must be non-negative")
    if args.freeze_base_model and args.freeze_base_bottom_layers > 0:
        raise SystemExit(
            "--freeze-base-model and --freeze-base-bottom-layers are mutually exclusive"
        )
    if args.freeze_protected_main_path and args.freeze_base_bottom_layers > 0:
        raise SystemExit(
            "--freeze-protected-main-path cannot be combined with partial base freezing"
        )
    if not 0.0 <= args.protected_subtype_weight <= 1.0:
        raise SystemExit("--protected-subtype-weight must be in [0, 1]")
    if not 0.0 <= args.protected_official_weight <= 1.0:
        raise SystemExit("--protected-official-weight must be in [0, 1]")
    protected_official_alpha_grid = parse_probability_grid(
        args.protected_official_alpha_grid
    )
    if args.protected_official_rng_reference_head_classes < 0:
        raise SystemExit("--protected-official-rng-reference-head-classes must be non-negative")
    if not 0.0 <= args.protected_gate_threshold <= 1.0:
        raise SystemExit("--protected-gate-threshold must be in [0, 1]")
    protected_stage_flags = (
        bool(args.init_protected_checkpoint)
        or args.freeze_protected_main_path
        or args.reinitialize_protected_fraud_branch
        or args.protected_main_only_mode
        or args.train_fraud_only
        or args.selection_score == "fraud_head_forced_macro_f1"
    )
    protected_stage_architectures = {
        "ce_protected_subtype",
        "ce_protected_top_layers",
        "ce_pbal_frozen_official_postgate",
    }
    if protected_stage_flags and args.architecture not in protected_stage_architectures:
        raise SystemExit(
            "E1 protected-stage flags require a protected subtype architecture"
        )
    if args.protected_main_only_mode and args.architecture != "ce_protected_subtype":
        raise SystemExit("--protected-main-only-mode requires ce_protected_subtype")
    if (
        args.reinitialize_protected_fraud_branch
        and args.architecture != "ce_protected_subtype"
    ):
        raise SystemExit(
            "--reinitialize-protected-fraud-branch requires ce_protected_subtype"
        )
    if args.protected_fraud_top_layers <= 0:
        raise SystemExit("--protected-fraud-top-layers must be positive")
    if args.freeze_protected_main_path and not args.init_protected_checkpoint:
        raise SystemExit("--freeze-protected-main-path requires --init-protected-checkpoint")
    if args.reinitialize_protected_fraud_branch and not args.init_protected_checkpoint:
        raise SystemExit(
            "--reinitialize-protected-fraud-branch requires --init-protected-checkpoint"
        )
    if args.train_fraud_only and args.dual_subtype_loss_weight <= 0:
        raise SystemExit("--train-fraud-only requires a positive subtype loss weight")
    if args.architecture == "ce_pbal_frozen_official_postgate":
        if not args.init_protected_checkpoint or not args.freeze_protected_main_path:
            raise SystemExit(
                "ce_pbal_frozen_official_postgate requires an initialized and frozen PBAL checkpoint"
            )
        if args.reinitialize_protected_fraud_branch or args.protected_main_only_mode:
            raise SystemExit(
                "ce_pbal_frozen_official_postgate cannot alter its frozen PBAL branch"
            )
        if args.dual_subtype_loss_weight != 0 or args.train_fraud_only:
            raise SystemExit(
                "ce_pbal_frozen_official_postgate trains only the official subcategory branch"
            )
        if args.official_subcategory_loss_weight <= 0:
            raise SystemExit(
                "ce_pbal_frozen_official_postgate requires a positive official subcategory loss"
            )
    elif protected_official_alpha_grid:
        raise SystemExit(
            "--protected-official-alpha-grid requires ce_pbal_frozen_official_postgate"
        )
    if args.official_subcategory_loss_weight > 0 and args.architecture not in {
        "ce_dual_head",
        "ce_dual_head_hier",
        "ce_protected_official_prototypes",
        "ce_protected_official_postgate",
        "ce_pbal_frozen_official_postgate",
    }:
        raise SystemExit(
            "official subcategory loss requires ce_dual_head or "
            "ce_protected_official_prototypes/ce_protected_official_postgate"
        )
    typed_evidence_allowed_labels = parse_label_filter(args.typed_evidence_labels)
    span_region_ce_allowed_labels = parse_label_filter(args.span_region_ce_labels)
    local_expert_group_spec = (
        LOCAL_EXPERT_GROUP_PRESETS[args.local_expert_group_preset]
        if args.local_expert_group_preset != "custom"
        else args.local_expert_groups
    )
    local_expert_groups = parse_local_expert_groups(local_expert_group_spec)
    if args.architecture in {
        "ce_selective_pair_expert",
        "ce_pairwise_boundary_expert",
    } and any(
        len(group) != 2 for group in local_expert_groups
    ):
        raise SystemExit(
            "pair expert architectures require every --local-expert-groups item to contain exactly two labels"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    distributed, world_size, rank, local_rank = distributed_env()
    if distributed and not torch.cuda.is_available():
        raise SystemExit("distributed training requires CUDA")
    if distributed and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    is_main = is_main_process(rank)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.artifact_output) if args.artifact_output else Path(args.output)
    mirror_output = Path(args.output)
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", vars(args))

    train_all = read_jsonl(Path(args.train_jsonl))
    if args.val_jsonl:
        train_rows = train_all
        val_rows = read_jsonl(Path(args.val_jsonl))
        split_manifest = independent_val_manifest(train_rows, val_rows, Path(args.val_jsonl))
    elif args.val_id_file:
        train_rows, val_rows, split_manifest = fixed_id_split(train_all, Path(args.val_id_file))
    else:
        train_rows, val_rows, split_manifest = stratified_group_split(
            train_all,
            args.val_ratio,
            args.seed,
            args.val_split_mode,
        )
    train_rows_before_stage_filter = len(train_rows)
    if args.train_fraud_only:
        train_rows = [
            row
            for row in train_rows
            if label13_of(row) != 0 and not has_coarse_fraud_label(row)
        ]
        if not train_rows:
            raise SystemExit("--train-fraud-only selected zero fully labeled fraud rows")
    if any(has_coarse_fraud_label(row) for row in val_rows):
        raise SystemExit("coarse fraud rows cannot be in validation; provide --val-id-file that selects only fully labeled rows")
    binary_aux_rows = read_jsonl(Path(args.binary_aux_jsonl)) if args.binary_aux_jsonl else []
    if args.binary_aux_loss_weight > 0 and not binary_aux_rows:
        raise SystemExit("--binary-aux-loss-weight > 0 requires --binary-aux-jsonl")
    if is_main:
        write_json(output / "split_manifest.json", split_manifest)
    eval_specs = parse_eval_specs(args.eval_csv)
    eval_rows = [(name, read_eval_csv(path), path) for name, path in eval_specs]
    selection_eval_path = Path(args.selection_eval_csv) if args.selection_eval_csv else None
    selection_eval_rows = (
        read_eval_csv(selection_eval_path) if selection_eval_path is not None else None
    )
    if selection_eval_path is not None:
        selection_path_resolved = selection_eval_path.resolve()
        if any(path.resolve() == selection_path_resolved for _, _, path in eval_rows):
            raise SystemExit(
                "--selection-eval-csv must not also be passed through --eval-csv; "
                "reserve the target-domain test CSV for final evaluation only"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if args.sample_weight_field:
        for row in train_rows:
            row["sample_weight"] = float(row.get(args.sample_weight_field, 1.0) or 1.0)
        for row in val_rows:
            row["sample_weight"] = float(row.get(args.sample_weight_field, 1.0) or 1.0)
    typed_evidence_max_length = (
        args.typed_evidence_max_length
        if (
            args.typed_evidence_contrastive_weight > 0
            or args.typed_evidence_axis_weight > 0
            or args.typed_evidence_consistency_weight > 0
            or args.ce_gate_alignment_weight > 0
        )
        else 0
    )
    span_region_ce_enabled = (
        args.span_region_ce_axis_weight > 0
        or args.span_region_ce_contrastive_weight > 0
        or args.span_region_ce_consistency_weight > 0
    )
    train_ds = TextDataset(
        train_rows,
        tokenizer,
        args.max_length,
        "train",
        "train",
        typed_evidence_max_length,
        span_region_ce_enabled,
        args.distill_prob_field if args.distill_loss_weight > 0 else "",
    )
    val_ds = TextDataset(
        val_rows,
        tokenizer,
        args.max_length,
        "val",
        "train",
        0,
        span_region_ce_enabled,
        "",
    )
    binary_aux_ds = (
        TextDataset(binary_aux_rows, tokenizer, args.max_length, "binary_aux", "binary_aux")
        if binary_aux_rows
        else None
    )
    eval_datasets = [
        (name, TextDataset(rows, tokenizer, args.max_length, name, name), path, rows)
        for name, rows, path in eval_rows
    ]
    selection_eval_ds = (
        TextDataset(
            selection_eval_rows,
            tokenizer,
            args.max_length,
            args.selection_eval_name,
            args.selection_eval_name,
        )
        if selection_eval_rows is not None
        else None
    )

    train_counts = Counter(str(label13_of(row)) for row in train_rows)
    val_counts = Counter(str(label13_of(row)) for row in val_rows)
    train_base_ids = {base_sample_id_of(row, "train", idx) for idx, row in enumerate(train_rows)}
    val_base_ids = {base_sample_id_of(row, "val", idx) for idx, row in enumerate(val_rows)}
    all_base_ids = {base_sample_id_of(row, "train_all", idx) for idx, row in enumerate(train_all)}
    train_view_counts = Counter(str(row.get("view_id") or "default") for row in train_rows)
    val_view_counts = Counter(str(row.get("view_id") or "default") for row in val_rows)
    train_full_label_counts = Counter(str(label13_of(row)) for row in train_rows if not has_coarse_fraud_label(row))
    train_official_subcategory_counts = Counter(
        official_subcategory_code_of(row)
        for row in train_rows
        if official_subcategory_code_of(row)
    )
    val_official_subcategory_counts = Counter(
        official_subcategory_code_of(row)
        for row in val_rows
        if official_subcategory_code_of(row)
    )
    train_coarse_counts = Counter(coarse_label_mode_of(row) for row in train_rows)
    val_coarse_counts = Counter(coarse_label_mode_of(row) for row in val_rows)
    binary_aux_binary_counts = Counter(str(1 if bool(row.get("is_fraud", label13_of(row) != 0)) else 0) for row in binary_aux_rows)
    binary_aux_source_counts = Counter(str(row.get("source") or row.get("dataset_version") or "unknown") for row in binary_aux_rows)
    if is_main:
        write_json(
            output / "data_manifest.json",
            {
            "train_jsonl": args.train_jsonl,
            "train_total_rows": len(train_all),
            "train_rows_before_stage_filter": train_rows_before_stage_filter,
            "train_fraud_only": bool(args.train_fraud_only),
            "train_total_unique_base_sample_ids": len(all_base_ids),
            "train_rows": len(train_rows),
            "train_unique_base_sample_ids": len(train_base_ids),
            "val_rows": len(val_rows),
            "val_unique_base_sample_ids": len(val_base_ids),
            "train_val_base_sample_overlap_count": len(train_base_ids & val_base_ids),
            "split_manifest": split_manifest,
            "train_label_13_counts": dict(sorted(train_counts.items(), key=lambda item: int(item[0]))),
            "train_full_label_13_counts_for_class_weight": dict(
                sorted(train_full_label_counts.items(), key=lambda item: int(item[0]))
            ),
            "train_official_subcategory_counts": dict(sorted(train_official_subcategory_counts.items())),
            "val_official_subcategory_counts": dict(sorted(val_official_subcategory_counts.items())),
            "val_label_13_counts": dict(sorted(val_counts.items(), key=lambda item: int(item[0]))),
            "checkpoint_selection": {
                "source": args.selection_eval_name if selection_eval_rows is not None else "internal_val",
                "path": str(selection_eval_path) if selection_eval_path is not None else None,
                "rows": len(selection_eval_rows) if selection_eval_rows is not None else len(val_rows),
                "label_13_counts": dict(
                    sorted(
                        Counter(
                            str(label13_of(row))
                            for row in (selection_eval_rows if selection_eval_rows is not None else val_rows)
                        ).items(),
                        key=lambda item: int(item[0]),
                    )
                ),
                "metric": "label_13_accuracy",
            },
            "train_view_counts": dict(sorted(train_view_counts.items())),
            "val_view_counts": dict(sorted(val_view_counts.items())),
            "train_coarse_label_mode_counts": dict(sorted(train_coarse_counts.items())),
            "val_coarse_label_mode_counts": dict(sorted(val_coarse_counts.items())),
            "binary_aux_jsonl": args.binary_aux_jsonl,
            "binary_aux_rows": len(binary_aux_rows),
            "binary_aux_binary_counts": dict(sorted(binary_aux_binary_counts.items())),
            "binary_aux_source_counts": dict(sorted(binary_aux_source_counts.items())),
            "span_region_ce_enabled": span_region_ce_enabled,
            "train_span_region_rows": sum(int(ex.has_span_region) for ex in train_ds.examples),
            "val_span_region_rows": sum(int(ex.has_span_region) for ex in val_ds.examples),
            "span_region_ce_axis_weight": args.span_region_ce_axis_weight,
            "span_region_ce_contrastive_weight": args.span_region_ce_contrastive_weight,
            "span_region_ce_consistency_weight": args.span_region_ce_consistency_weight,
            "span_region_ce_labels": sorted(span_region_ce_allowed_labels) if span_region_ce_allowed_labels is not None else [],
            "distill_loss_weight": args.distill_loss_weight,
            "distill_temperature": args.distill_temperature,
            "distill_prob_field": args.distill_prob_field,
            "train_distill_rows": sum(int(ex.has_distill_probs) for ex in train_ds.examples),
            "eval_csvs": {
                name: {
                    "path": str(path),
                    "rows": len(rows),
                    "label_13_counts": dict(
                        sorted(Counter(str(label13_of(row)) for row in rows).items(), key=lambda item: int(item[0]))
                    ),
                }
                for name, _, path, rows in eval_datasets
            },
            },
        )

    distributed_barrier()

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    model = build_model_from_args(args, local_expert_groups, device)
    fresh_fraud_branch_state = None
    if args.reinitialize_protected_fraud_branch:
        fraud_prefixes = ("fraud_adapter.", "fraud_norm.", "subtype_classifier.")
        fresh_fraud_branch_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith(fraud_prefixes)
        }
        if not fresh_fraud_branch_state:
            raise RuntimeError("protected fraud branch parameters were not found")
    if args.init_protected_checkpoint:
        checkpoint_state = torch.load(
            Path(args.init_protected_checkpoint),
            map_location=device,
        )
        if args.architecture == "ce_protected_top_layers":
            main_prefixes = ("base_model.", "classifier.")
            main_state = {
                name: value
                for name, value in checkpoint_state.items()
                if name.startswith(main_prefixes)
            }
            expected_main_names = {
                name
                for name in model.state_dict()
                if name.startswith(main_prefixes)
            }
            missing_main_names = sorted(expected_main_names - set(main_state))
            if missing_main_names:
                raise RuntimeError(
                    "protected main checkpoint is missing parameters: "
                    + ", ".join(missing_main_names[:10])
                )
            model.load_state_dict(main_state, strict=False)
            model.initialize_fraud_branch_from_main()
        elif args.architecture == "ce_pbal_frozen_official_postgate":
            incompatible = model.load_state_dict(checkpoint_state, strict=False)
            missing = sorted(incompatible.missing_keys)
            unexpected = sorted(incompatible.unexpected_keys)
            allowed_missing_prefixes = (
                "official_adapter.",
                "official_norm.",
                "official_subcategory_classifier.",
            )
            missing_pbal_core = [
                name for name in missing if not name.startswith(allowed_missing_prefixes)
            ]
            if missing_pbal_core or unexpected:
                raise RuntimeError(
                    "PBAL checkpoint did not load exactly into the frozen PBAL path; "
                    f"missing={missing_pbal_core[:10]}, unexpected={unexpected[:10]}"
                )
        else:
            model.load_state_dict(checkpoint_state)
    if fresh_fraud_branch_state is not None:
        model.load_state_dict(fresh_fraud_branch_state, strict=False)
    if args.protected_main_only_mode:
        setattr(model, "protected_main_only_mode", True)
    if args.freeze_protected_main_path:
        freeze_protected_main_path(model)
    protected_main_sha256_before_training = (
        protected_main_state_sha256(model)
        if should_track_protected_main_hash(args.architecture)
        else None
    )
    if args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model)
    if args.freeze_base_model:
        maybe_freeze_base_model(model)
        partial_base_freeze = {
            "encoder_layers_total": 0,
            "encoder_layers_frozen": 0,
            "frozen_parameter_count": 0,
        }
    elif args.freeze_base_bottom_layers > 0:
        partial_base_freeze = freeze_base_bottom_encoder_layers(
            model,
            args.freeze_base_bottom_layers,
        )
    else:
        partial_base_freeze = {
            "encoder_layers_total": 0,
            "encoder_layers_frozen": 0,
            "frozen_parameter_count": 0,
        }
    if binary_aux_ds is not None and args.binary_aux_loss_weight > 0:
        attach_binary_aux_head(model, device)
    if distributed and args.pair_margin_weight > 0:
        raise SystemExit("distributed mode does not yet support pair-margin batch sampler")
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed and args.pair_margin_weight <= 0
        else None
    )
    binary_aux_sampler = (
        DistributedSampler(binary_aux_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed and binary_aux_ds is not None and args.binary_aux_loss_weight > 0
        else None
    )
    if args.pair_margin_weight > 0:
        loader = DataLoader(
            train_ds,
            batch_sampler=build_pair_aware_batches(train_ds, args.batch_size, args.seed),
            collate_fn=collate,
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=bool(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate,
        )
    binary_aux_loader = (
        DataLoader(
            binary_aux_ds,
            batch_size=args.binary_aux_batch_size if args.binary_aux_batch_size > 0 else args.batch_size,
            shuffle=bool(binary_aux_sampler is None),
            sampler=binary_aux_sampler,
            collate_fn=collate,
        )
        if binary_aux_ds is not None and args.binary_aux_loss_weight > 0
        else None
    )
    if args.class_weight != "none":
        label_counts = [train_full_label_counts.get(str(label), 0) for label in range(NUM_CLASSES)]
        total = sum(label_counts)
        weights = [total / max(1, NUM_CLASSES * count) for count in label_counts]
        if args.class_weight == "sqrt_balanced":
            weights = [weight ** 0.5 for weight in weights]
        effective_cap = 1.5 if args.class_weight == "balanced_cap_1.5" else args.class_weight_cap
        if effective_cap > 0:
            weights = [min(weight, effective_cap) for weight in weights]
        weights[0] *= args.normal_weight_scale
        if args.label12_weight_cap > 0:
            weights[12] = min(weights[12], args.label12_weight_cap)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), reduction="none")
    else:
        weights = None
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    binary_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    subtype_label_counts = [train_full_label_counts.get(str(label), 0) for label in range(1, NUM_CLASSES)]
    subtype_total = sum(subtype_label_counts)
    subtype_weights: list[float] | None = None
    if args.dual_subtype_class_weight in {"balanced", "sqrt_balanced"} and subtype_total > 0:
        subtype_weights = [
            subtype_total / max(1, (NUM_CLASSES - 1) * count)
            for count in subtype_label_counts
        ]
        if args.dual_subtype_class_weight == "sqrt_balanced":
            subtype_weights = [float(weight**0.5) for weight in subtype_weights]
    subtype_loss_fn = torch.nn.CrossEntropyLoss(
        weight=(
            torch.tensor(subtype_weights, dtype=torch.float32, device=device)
            if subtype_weights is not None
            else None
        ),
        reduction="none",
    )
    official_subcategory_label_counts = [
        train_official_subcategory_counts.get(code, 0)
        for code in OFFICIAL_SUBCATEGORY_CODES
    ]
    official_subcategory_weights: list[float] | None = None
    if args.official_subcategory_class_weight == "sqrt_balanced":
        observed_classes = sum(count > 0 for count in official_subcategory_label_counts)
        observed_total = sum(official_subcategory_label_counts)
        cap = max(0.0, float(args.official_subcategory_class_weight_cap))
        official_subcategory_weights = []
        for count in official_subcategory_label_counts:
            if count <= 0:
                weight = 0.0
            else:
                weight = (observed_total / max(1, observed_classes * count)) ** 0.5
                if cap > 0:
                    weight = min(weight, cap)
            official_subcategory_weights.append(float(weight))
    official_subcategory_weight_tensor = (
        torch.tensor(official_subcategory_weights, dtype=torch.float32, device=device)
        if official_subcategory_weights is not None
        else None
    )
    official_subcategory_loss_fn = torch.nn.CrossEntropyLoss(
        weight=official_subcategory_weight_tensor,
        ignore_index=-100,
        reduction="none",
    )
    clear11_label_counts = [train_full_label_counts.get(str(label), 0) for label in range(1, 12)]
    clear11_total = sum(clear11_label_counts)
    clear11_weights: list[float] | None = None
    if args.clear11_class_weight in {"balanced", "sqrt_balanced"} and clear11_total > 0:
        clear11_weights = [clear11_total / max(1, 11 * count) for count in clear11_label_counts]
        if args.clear11_class_weight == "sqrt_balanced":
            clear11_weights = [float(weight**0.5) for weight in clear11_weights]
    clear11_loss_fn = torch.nn.CrossEntropyLoss(
        weight=(
            torch.tensor(clear11_weights, dtype=torch.float32, device=device)
            if clear11_weights is not None
            else None
        ),
        reduction="none",
    )
    residual12_pos_weight = None
    if args.residual12_pos_weight > 0:
        residual12_pos_weight = torch.tensor(args.residual12_pos_weight, dtype=torch.float32, device=device)
    residual12_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=residual12_pos_weight, reduction="none")
    local_expert_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    subtype_log_prior = None
    if args.dual_subtype_logit_adjust_tau > 0 and subtype_total > 0:
        subtype_priors = [
            max(count, 1) / subtype_total
            for count in subtype_label_counts
        ]
        subtype_log_prior = torch.log(torch.tensor(subtype_priors, dtype=torch.float32, device=device).clamp_min(1e-8))
    if args.typed_evidence_axis_class_weight == "balanced":
        label_counts = [train_full_label_counts.get(str(label), 0) for label in range(NUM_CLASSES)]
        total = sum(label_counts)
        axis_weights = [total / max(1, NUM_CLASSES * count) for count in label_counts]
        axis_weights[0] *= args.normal_weight_scale
        typed_evidence_axis_loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(axis_weights, dtype=torch.float32, device=device),
            reduction="none",
        )
    else:
        axis_weights = None
        typed_evidence_axis_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    gradient_accumulation_steps = max(1, int(args.gradient_accumulation_steps))
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        optimizer = ZeroRedundancyOptimizer(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            optimizer_class=torch.optim.AdamW,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    total_steps = max(1, ((len(loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and str(device).startswith("cuda")))
    trainable_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_selection_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_checkpoint_gate_pass = False
    best_state_dict: dict[str, Any] | None = None
    epochs_without_improvement = 0
    start = time.time()
    optimizer_step = 0
    averaging_state: dict[str, Any] | None = None
    averaging_updates = 0
    if is_main:
        update_train_progress(
            epoch_current=0,
            epoch_total=args.epochs,
            step_current=0,
            step_total=total_steps,
            force=True,
        )
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if binary_aux_sampler is not None:
            binary_aux_sampler.set_epoch(epoch)
        model.train()
        enforce_protected_main_eval(model)
        enforce_frozen_base_bottom_layers_eval(model)
        optimizer.zero_grad(set_to_none=True)
        binary_aux_iter = iter(binary_aux_loader) if binary_aux_loader is not None else None
        total_loss = 0.0
        total_typed_evidence_loss = 0.0
        total_typed_evidence_axis_loss = 0.0
        total_typed_evidence_consistency_loss = 0.0
        total_ce_gate_alignment_loss = 0.0
        total_span_region_ce_axis_loss = 0.0
        total_span_region_ce_contrastive_loss = 0.0
        total_span_region_ce_consistency_loss = 0.0
        total_dual_binary_loss = 0.0
        total_dual_subtype_loss = 0.0
        total_h2_flat_aux_loss = 0.0
        total_h2_binary_consistency_loss = 0.0
        total_official_subcategory_loss = 0.0
        total_clear11_loss = 0.0
        total_residual12_loss = 0.0
        total_local_expert_loss = 0.0
        total_binary_aux_loss = 0.0
        total_distill_loss = 0.0
        total_rdrop_kl_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            with torch.cuda.amp.autocast(enabled=bool(args.fp16 and str(device).startswith("cuda"))):
                outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                labels = batch["label13"].to(device)
                frozen_pbal_official_only = (
                    args.architecture == "ce_pbal_frozen_official_postgate"
                )
                per_sample_loss = (
                    torch.zeros_like(labels, dtype=outputs.logits.dtype)
                    if frozen_pbal_official_only
                    else loss_fn(outputs.logits, labels)
                )
                rdrop_kl_loss_value = torch.zeros((), device=device)
                if args.rdrop_kl_weight > 0:
                    outputs_second = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                    per_sample_loss = 0.5 * (
                        per_sample_loss + loss_fn(outputs_second.logits, labels)
                    )
                    first_log_probs = F.log_softmax(outputs.logits, dim=-1)
                    second_log_probs = F.log_softmax(outputs_second.logits, dim=-1)
                    first_probs = first_log_probs.exp()
                    second_probs = second_log_probs.exp()
                    rdrop_kl_loss_value = 0.5 * (
                        F.kl_div(first_log_probs, second_probs, reduction="batchmean")
                        + F.kl_div(second_log_probs, first_probs, reduction="batchmean")
                    )
                coarse_fraud_mask = torch.tensor(
                    [mode == "fraud_any" for mode in batch["coarse_label_mode"]],
                    dtype=torch.bool,
                    device=device,
                )
                if (not frozen_pbal_official_only) and int(coarse_fraud_mask.sum().detach().cpu()) > 0:
                    log_probs = F.log_softmax(outputs.logits, dim=-1)
                    per_sample_loss = per_sample_loss.clone()
                    per_sample_loss[coarse_fraud_mask] = -torch.logsumexp(
                        log_probs[coarse_fraud_mask, 1:],
                        dim=-1,
                    )
                if frozen_pbal_official_only:
                    # The ordinary 13-class CE must not create an update: this
                    # run trains the official branch only.  The differentiable
                    # zero keeps batches containing only masked labels valid.
                    loss = outputs.official_subcategory_logits.sum() * 0.0
                elif args.sample_weight_field:
                    sample_weight = batch["sample_weight"].to(device)
                    loss = (per_sample_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
                else:
                    loss = per_sample_loss.mean()
                if args.rdrop_kl_weight > 0:
                    loss = loss + args.rdrop_kl_weight * rdrop_kl_loss_value
                distill_loss_value = torch.zeros((), device=device)
                if args.distill_loss_weight > 0:
                    distill_mask = batch["has_distill_probs"].to(device)
                    if int(distill_mask.sum().detach().cpu()) > 0:
                        temperature = max(float(args.distill_temperature), 1e-6)
                        teacher_probs = batch["distill_probs"].to(device)[distill_mask].clamp_min(1e-8)
                        teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        student_log_probs = F.log_softmax(outputs.logits[distill_mask] / temperature, dim=-1)
                        distill_loss_value = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
                            temperature**2
                        )
                        loss = loss + args.distill_loss_weight * distill_loss_value
                dual_binary_loss_value = torch.zeros((), device=device)
                dual_subtype_loss_value = torch.zeros((), device=device)
                if args.architecture in DUAL_HEAD_ARCHITECTURES and args.dual_binary_loss_weight > 0:
                    binary_labels = (labels != 0).long()
                    dual_binary_per_sample = binary_loss_fn(outputs.binary_logits, binary_labels)
                    if args.sample_weight_field:
                        sample_weight = batch["sample_weight"].to(device)
                        dual_binary_loss_value = (
                            dual_binary_per_sample * sample_weight
                        ).sum() / sample_weight.sum().clamp_min(1e-6)
                    else:
                        dual_binary_loss_value = dual_binary_per_sample.mean()
                    loss = loss + args.dual_binary_loss_weight * dual_binary_loss_value
                if args.architecture in SUBTYPE_AUX_ARCHITECTURES and args.dual_subtype_loss_weight > 0:
                    fraud_mask = (labels != 0) & (~coarse_fraud_mask)
                    if int(fraud_mask.sum().detach().cpu()) > 0:
                        subtype_labels = labels[fraud_mask] - 1
                        subtype_logits_for_loss = outputs.subtype_logits[fraud_mask]
                        if subtype_log_prior is not None:
                            subtype_logits_for_loss = (
                                subtype_logits_for_loss
                                + args.dual_subtype_logit_adjust_tau * subtype_log_prior
                            )
                        dual_subtype_per_sample = subtype_loss_fn(subtype_logits_for_loss, subtype_labels)
                        if args.sample_weight_field:
                            fraud_sample_weight = batch["sample_weight"].to(device)[fraud_mask]
                            dual_subtype_loss_value = (
                                dual_subtype_per_sample * fraud_sample_weight
                            ).sum() / fraud_sample_weight.sum().clamp_min(1e-6)
                        else:
                            dual_subtype_loss_value = dual_subtype_per_sample.mean()
                        loss = loss + args.dual_subtype_loss_weight * dual_subtype_loss_value
                h2_flat_aux_loss_value = torch.zeros((), device=device)
                h2_binary_consistency_loss_value = torch.zeros((), device=device)
                if args.architecture == "ce_dual_head_h2":
                    flat_aux_logits = getattr(outputs, "flat_aux_logits", None)
                    if flat_aux_logits is None:
                        raise RuntimeError("ce_dual_head_h2 did not return flat_aux_logits")
                    if args.h2_flat_aux_loss_weight > 0:
                        h2_flat_per_sample = loss_fn(flat_aux_logits, labels)
                        if int(coarse_fraud_mask.sum().detach().cpu()) > 0:
                            flat_log_probs = F.log_softmax(flat_aux_logits, dim=-1)
                            h2_flat_per_sample = h2_flat_per_sample.clone()
                            h2_flat_per_sample[coarse_fraud_mask] = -torch.logsumexp(
                                flat_log_probs[coarse_fraud_mask, 1:],
                                dim=-1,
                            )
                        if args.sample_weight_field:
                            sample_weight = batch["sample_weight"].to(device)
                            h2_flat_aux_loss_value = (
                                h2_flat_per_sample * sample_weight
                            ).sum() / sample_weight.sum().clamp_min(1e-6)
                        else:
                            h2_flat_aux_loss_value = h2_flat_per_sample.mean()
                        loss = loss + args.h2_flat_aux_loss_weight * h2_flat_aux_loss_value
                    if args.h2_binary_consistency_weight > 0:
                        h2_binary_consistency_loss_value = h2_binary_mass_consistency_loss(
                            flat_aux_logits,
                            outputs.logits,
                        )
                        loss = (
                            loss
                            + args.h2_binary_consistency_weight
                            * h2_binary_consistency_loss_value
                        )
                official_subcategory_loss_value = torch.zeros((), device=device)
                if args.official_subcategory_loss_weight > 0:
                    official_targets = batch["official_subcategory_index"].to(device)
                    official_mask = official_targets != -100
                    official_logits = getattr(outputs, "official_subcategory_logits", None)
                    if official_logits is None:
                        raise RuntimeError("official subcategory loss enabled but model returned no auxiliary logits")
                    if args.official_subcategory_loss_mode == "sibling_masked":
                        parent_losses = []
                        for parent, sibling_indices in OFFICIAL_PARENT_TO_INDICES.items():
                            if parent == 0 or len(sibling_indices) <= 1:
                                continue
                            parent_mask = official_mask & (labels == parent)
                            if int(parent_mask.sum().detach().cpu()) <= 0:
                                continue
                            sibling_index_tensor = torch.tensor(
                                sibling_indices, dtype=torch.long, device=device
                            )
                            local_logits = official_logits[parent_mask].index_select(
                                1, sibling_index_tensor
                            )
                            local_lookup = {
                                global_index: local_index
                                for local_index, global_index in enumerate(sibling_indices)
                            }
                            local_targets = torch.tensor(
                                [
                                    local_lookup[int(global_index)]
                                    for global_index in official_targets[parent_mask].detach().cpu().tolist()
                                ],
                                dtype=torch.long,
                                device=device,
                            )
                            local_weights = (
                                official_subcategory_weight_tensor.index_select(0, sibling_index_tensor)
                                if official_subcategory_weight_tensor is not None
                                else None
                            )
                            local_per_sample = F.cross_entropy(
                                local_logits,
                                local_targets,
                                weight=local_weights,
                                reduction="none",
                            )
                            if args.sample_weight_field:
                                local_sample_weight = batch["sample_weight"].to(device)[parent_mask]
                                parent_loss = (
                                    local_per_sample * local_sample_weight
                                ).sum() / local_sample_weight.sum().clamp_min(1e-6)
                            else:
                                parent_loss = local_per_sample.mean()
                            parent_losses.append(parent_loss)
                        if parent_losses:
                            official_subcategory_loss_value = torch.stack(parent_losses).mean()
                    elif int(official_mask.sum().detach().cpu()) > 0:
                        official_per_sample = official_subcategory_loss_fn(
                            official_logits[official_mask], official_targets[official_mask]
                        )
                        if args.sample_weight_field:
                            official_sample_weight = batch["sample_weight"].to(device)[official_mask]
                            official_subcategory_loss_value = (
                                official_per_sample * official_sample_weight
                            ).sum() / official_sample_weight.sum().clamp_min(1e-6)
                        else:
                            official_subcategory_loss_value = official_per_sample.mean()
                    if official_subcategory_loss_value.requires_grad:
                        loss = loss + args.official_subcategory_loss_weight * official_subcategory_loss_value
                clear11_loss_value = torch.zeros((), device=device)
                if args.clear11_loss_weight > 0 and hasattr(outputs, "clear11_logits"):
                    clear11_mask = (labels >= 1) & (labels <= 11) & (~coarse_fraud_mask)
                    if int(clear11_mask.sum().detach().cpu()) > 0:
                        clear11_labels = labels[clear11_mask] - 1
                        clear11_per_sample = clear11_loss_fn(outputs.clear11_logits[clear11_mask], clear11_labels)
                        if args.sample_weight_field:
                            clear11_sample_weight = batch["sample_weight"].to(device)[clear11_mask]
                            clear11_loss_value = (
                                clear11_per_sample * clear11_sample_weight
                            ).sum() / clear11_sample_weight.sum().clamp_min(1e-6)
                        else:
                            clear11_loss_value = clear11_per_sample.mean()
                        loss = loss + args.clear11_loss_weight * clear11_loss_value
                residual12_loss_value = torch.zeros((), device=device)
                if args.residual12_loss_weight > 0 and hasattr(outputs, "residual12_logits"):
                    residual12_targets = (labels == 12).float()
                    residual12_per_sample = residual12_loss_fn(outputs.residual12_logits, residual12_targets)
                    if args.sample_weight_field:
                        sample_weight = batch["sample_weight"].to(device)
                        residual12_loss_value = (
                            residual12_per_sample * sample_weight
                        ).sum() / sample_weight.sum().clamp_min(1e-6)
                    else:
                        residual12_loss_value = residual12_per_sample.mean()
                    loss = loss + args.residual12_loss_weight * residual12_loss_value
                local_expert_loss_value = torch.zeros((), device=device)
                if args.architecture == "ce_local_expert" and args.local_expert_loss_weight > 0:
                    local_expert_loss_value = local_expert_loss(
                        outputs,
                        labels,
                        batch["confusable_label13"].to(device),
                        batch["sample_weight"].to(device) if args.sample_weight_field else None,
                        local_expert_loss_fn,
                    )
                    loss = loss + args.local_expert_loss_weight * local_expert_loss_value
                if (
                    args.architecture == "ce_selective_pair_expert"
                    and args.local_expert_loss_weight > 0
                    and epoch > args.local_expert_warmup_epochs
                ):
                    local_expert_loss_value = selective_pair_expert_loss(
                        outputs,
                        labels,
                        batch["sample_weight"].to(device)
                        if args.sample_weight_field
                        else None,
                        local_expert_loss_fn,
                        args.local_expert_rescue_weight,
                    )
                    loss = loss + args.local_expert_loss_weight * local_expert_loss_value
                if (
                    args.architecture == "ce_pairwise_boundary_expert"
                    and args.local_expert_loss_weight > 0
                    and epoch > args.local_expert_warmup_epochs
                ):
                    local_expert_loss_value = pairwise_boundary_expert_loss(
                        outputs,
                        labels,
                        batch["sample_weight"].to(device)
                        if args.sample_weight_field
                        else None,
                        local_expert_loss_fn,
                    )
                    loss = loss + args.local_expert_loss_weight * local_expert_loss_value
                binary_aux_loss_value = torch.zeros((), device=device)
                if binary_aux_iter is not None and args.binary_aux_loss_weight > 0:
                    try:
                        aux_batch = next(binary_aux_iter)
                    except StopIteration:
                        binary_aux_iter = iter(binary_aux_loader)
                        aux_batch = next(binary_aux_iter)
                    aux_labels = aux_batch["is_fraud"].to(device).long()
                    aux_logits = binary_aux_logits(model, aux_batch, device)
                    binary_aux_loss_value = binary_loss_fn(aux_logits, aux_labels).mean()
                    loss = loss + args.binary_aux_loss_weight * binary_aux_loss_value
                pair_loss_value = torch.zeros((), device=device)
                confusable_loss_value = torch.zeros((), device=device)
                if args.pair_margin_weight > 0:
                    pair_loss_value = same_pair_margin_loss(outputs.logits, batch["pair_id"], batch["pair_role"], args.pair_margin)
                    loss = loss + args.pair_margin_weight * pair_loss_value
                if args.confusable_margin_weight > 0:
                    confusable_loss_value = confusable_margin_loss(
                        outputs.logits,
                        labels,
                        batch["confusable_label13"].to(device),
                        args.confusable_margin,
                    )
                    loss = loss + args.confusable_margin_weight * confusable_loss_value
                typed_evidence_loss_value = torch.zeros((), device=device)
                effective_typed_weight = (
                    args.typed_evidence_contrastive_weight
                    if epoch > args.typed_evidence_warmup_epochs
                    else 0.0
                )
                if effective_typed_weight > 0:
                    typed_evidence_loss_value = typed_evidence_contrastive_loss(
                        model,
                        batch,
                        device,
                        args.typed_evidence_temperature,
                    )
                    loss = loss + effective_typed_weight * typed_evidence_loss_value
                typed_evidence_axis_loss_value = torch.zeros((), device=device)
                effective_axis_weight = (
                    args.typed_evidence_axis_weight
                    if epoch > args.typed_evidence_warmup_epochs
                    else 0.0
                )
                if effective_axis_weight > 0:
                    typed_evidence_axis_loss_value = typed_evidence_axis_loss(
                        model,
                        batch,
                        device,
                        typed_evidence_axis_loss_fn,
                        args.typed_evidence_axis_confusable_margin_weight,
                        args.typed_evidence_axis_confusable_margin,
                        args.typed_evidence_axis_mode,
                        typed_evidence_allowed_labels,
                        args.typed_evidence_axis_target,
                    )
                    loss = loss + effective_axis_weight * typed_evidence_axis_loss_value
                typed_evidence_consistency_loss_value = torch.zeros((), device=device)
                effective_consistency_weight = (
                    args.typed_evidence_consistency_weight
                    if epoch > args.typed_evidence_warmup_epochs
                    else 0.0
                )
                if effective_consistency_weight > 0:
                    typed_evidence_consistency_loss_value = typed_evidence_consistency_loss(
                        model,
                        batch,
                        device,
                        outputs.logits,
                        args.typed_evidence_temperature,
                        args.typed_evidence_consistency_mode,
                        typed_evidence_allowed_labels,
                    )
                    loss = loss + effective_consistency_weight * typed_evidence_consistency_loss_value
                ce_gate_alignment_loss_value = torch.zeros((), device=device)
                if args.ce_gate_alignment_weight > 0:
                    ce_gate_alignment_loss_value = ce_gate_alignment_loss(
                        model,
                        batch,
                        device,
                        typed_evidence_allowed_labels,
                    )
                    loss = loss + args.ce_gate_alignment_weight * ce_gate_alignment_loss_value
                span_region_ce_axis_loss_value = torch.zeros((), device=device)
                span_region_axis_weight = (
                    args.span_region_ce_axis_weight
                    if epoch > args.span_region_ce_warmup_epochs
                    else 0.0
                )
                if span_region_axis_weight > 0:
                    span_region_ce_axis_loss_value = span_region_axis_loss(
                        model,
                        batch,
                        device,
                        typed_evidence_axis_loss_fn,
                        span_region_ce_allowed_labels,
                    )
                    loss = loss + span_region_axis_weight * span_region_ce_axis_loss_value
                span_region_ce_contrastive_loss_value = torch.zeros((), device=device)
                span_region_contrastive_weight = (
                    args.span_region_ce_contrastive_weight
                    if epoch > args.span_region_ce_warmup_epochs
                    else 0.0
                )
                if span_region_contrastive_weight > 0:
                    span_region_ce_contrastive_loss_value = span_region_contrastive_loss(
                        model,
                        batch,
                        device,
                        args.span_region_ce_temperature,
                        span_region_ce_allowed_labels,
                    )
                    loss = loss + span_region_contrastive_weight * span_region_ce_contrastive_loss_value
                span_region_ce_consistency_loss_value = torch.zeros((), device=device)
                span_region_consistency_weight = (
                    args.span_region_ce_consistency_weight
                    if epoch > args.span_region_ce_warmup_epochs
                    else 0.0
                )
                if span_region_consistency_weight > 0:
                    span_region_ce_consistency_loss_value = span_region_consistency_loss(
                        model,
                        batch,
                        device,
                        outputs.logits,
                        args.span_region_ce_temperature,
                        span_region_ce_allowed_labels,
                    )
                    loss = loss + span_region_consistency_weight * span_region_ce_consistency_loss_value
            scaler.scale(loss / gradient_accumulation_steps).backward()
            should_step = step % gradient_accumulation_steps == 0 or step == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if args.weight_averaging_mode != "none" and epoch >= args.weight_averaging_start_epoch:
                    current = dict(train_module.named_parameters())
                    if averaging_state is None:
                        averaging_state = {name: value.detach().clone() for name, value in current.items()}
                        averaging_updates = 1
                    else:
                        averaging_updates += 1
                        if args.weight_averaging_mode == "ema":
                            decay = float(args.ema_decay)
                            for name, value in current.items():
                                averaging_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
                        else:
                            alpha = 1.0 / float(averaging_updates)
                            for name, value in current.items():
                                averaging_state[name].add_(value.detach() - averaging_state[name], alpha=alpha)
            total_loss += float(loss.detach().cpu())
            total_typed_evidence_loss += float(typed_evidence_loss_value.detach().cpu())
            total_typed_evidence_axis_loss += float(typed_evidence_axis_loss_value.detach().cpu())
            total_typed_evidence_consistency_loss += float(typed_evidence_consistency_loss_value.detach().cpu())
            total_ce_gate_alignment_loss += float(ce_gate_alignment_loss_value.detach().cpu())
            total_span_region_ce_axis_loss += float(span_region_ce_axis_loss_value.detach().cpu())
            total_span_region_ce_contrastive_loss += float(span_region_ce_contrastive_loss_value.detach().cpu())
            total_span_region_ce_consistency_loss += float(span_region_ce_consistency_loss_value.detach().cpu())
            total_dual_binary_loss += float(dual_binary_loss_value.detach().cpu())
            total_dual_subtype_loss += float(dual_subtype_loss_value.detach().cpu())
            total_h2_flat_aux_loss += float(h2_flat_aux_loss_value.detach().cpu())
            total_h2_binary_consistency_loss += float(
                h2_binary_consistency_loss_value.detach().cpu()
            )
            total_official_subcategory_loss += float(official_subcategory_loss_value.detach().cpu())
            total_clear11_loss += float(clear11_loss_value.detach().cpu())
            total_residual12_loss += float(residual12_loss_value.detach().cpu())
            total_local_expert_loss += float(local_expert_loss_value.detach().cpu())
            total_binary_aux_loss += float(binary_aux_loss_value.detach().cpu())
            total_distill_loss += float(distill_loss_value.detach().cpu())
            total_rdrop_kl_loss += float(rdrop_kl_loss_value.detach().cpu())
            if is_main:
                update_train_progress(
                    epoch_current=(epoch - 1) + min(1.0, step / max(1, len(loader))),
                    epoch_total=args.epochs,
                    step_current=min(total_steps, optimizer_step),
                    step_total=total_steps,
                )

        if args.checkpoint_bank_only:
            bank_row = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, len(loader)),
                "architecture": args.architecture,
                "selection_deferred": True,
                "checkpoint_selection": {
                    "source": "new_internal_dev_pending",
                    "metric": "label_13_accuracy",
                    "metrics": None,
                },
            }
            history.append(bank_row)
            bank_module = model.module if distributed else model
            if is_main:
                checkpoint_dir = output / "checkpoints" / f"epoch-{epoch}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                if args.architecture in CUSTOM_ARCHITECTURES:
                    torch.save(bank_module.state_dict(), checkpoint_dir / "pytorch_model.bin")
                    write_json(
                        checkpoint_dir / "custom_model.json",
                        {
                            "architecture": args.architecture,
                            "joint_mode": (
                                "hierarchical"
                                if args.architecture
                                in {"ce_dual_head_hier", "ce_dual_head_h2", "h2_cascade"}
                                else "additive"
                            ),
                            "ce_mid_adapter_layer": args.ce_mid_adapter_layer,
                            "ce_mid_adapter_bottleneck": args.ce_mid_adapter_bottleneck,
                            "local_expert_groups": local_expert_groups,
                            "local_expert_residual_scale": args.local_expert_residual_scale,
                            "local_expert_gate_bias": args.local_expert_gate_bias,
                            "local_expert_rescue_weight": args.local_expert_rescue_weight,
                            "local_expert_warmup_epochs": args.local_expert_warmup_epochs,
                        },
                    )
                else:
                    bank_module.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                write_json(output / "history.json", history)
                if output != mirror_output:
                    mirror_file(output / "history.json", mirror_output / "history.json")
                update_train_progress(
                    epoch_current=epoch,
                    epoch_total=args.epochs,
                    step_current=min(total_steps, optimizer_step),
                    step_total=total_steps,
                    force=True,
                )
            distributed_barrier()
            continue

        if is_main:
            update_eval_progress(dataset=f"internal_val_epoch_{epoch}", done=0, total=1, force=True)
        if distributed:
            val_metrics, val_prediction_rows = distributed_predict(
                model,
                val_ds,
                args.eval_batch_size,
                device,
                world_size,
                rank,
            )
        else:
            val_metrics, val_prediction_rows = predict(model, val_ds, args.eval_batch_size, device)
        if is_main:
            update_eval_progress(dataset=f"internal_val_epoch_{epoch}", done=1, total=1, force=True)
        checkpoint_selection_metrics = val_metrics
        checkpoint_selection_prediction_rows = val_prediction_rows
        checkpoint_selection_source = "internal_val"
        if selection_eval_ds is not None:
            checkpoint_selection_source = args.selection_eval_name
            if is_main:
                update_eval_progress(
                    dataset=f"{args.selection_eval_name}_epoch_{epoch}",
                    done=0,
                    total=1,
                    force=True,
                )
            if distributed:
                checkpoint_selection_metrics, checkpoint_selection_prediction_rows = distributed_predict(
                    model,
                    selection_eval_ds,
                    args.eval_batch_size,
                    device,
                    world_size,
                    rank,
                )
            else:
                checkpoint_selection_metrics, checkpoint_selection_prediction_rows = predict(
                    model, selection_eval_ds, args.eval_batch_size, device
                )
            if is_main:
                update_eval_progress(
                    dataset=f"{args.selection_eval_name}_epoch_{epoch}",
                    done=1,
                    total=1,
                    force=True,
                )
        task_proxy_selection: dict[str, Any] | None = None
        fpr_constrained_selection: dict[str, Any] | None = None
        checkpoint_gate_pass = True
        if args.selection_score == "task_v21_proxy_selection":
            task_proxy_selection = compute_task_v21_proxy_selection(val_prediction_rows) if is_main else None
            task_proxy_selection = distributed_broadcast_object(task_proxy_selection, rank)
            score = float(task_proxy_selection["selection_score"])
            checkpoint_gate_pass = bool(task_proxy_selection["gate_pass"])
        elif args.selection_score == "task_v23_proxy_selection":
            task_proxy_selection = compute_task_v23_proxy_selection(val_prediction_rows) if is_main else None
            task_proxy_selection = distributed_broadcast_object(task_proxy_selection, rank)
            score = float(task_proxy_selection["selection_score"])
            checkpoint_gate_pass = bool(task_proxy_selection["gate_pass"])
        elif args.selection_score == "task_v24_proxy_selection":
            task_proxy_selection = compute_task_v24_proxy_selection(val_prediction_rows) if is_main else None
            task_proxy_selection = distributed_broadcast_object(task_proxy_selection, rank)
            score = float(task_proxy_selection["selection_score"])
            checkpoint_gate_pass = bool(task_proxy_selection["gate_pass"])
        elif args.selection_score == "task_v22_repair_selection":
            task_proxy_selection = compute_task_v22_repair_selection(val_prediction_rows) if is_main else None
            task_proxy_selection = distributed_broadcast_object(task_proxy_selection, rank)
            score = float(task_proxy_selection["selection_score"])
            checkpoint_gate_pass = bool(task_proxy_selection["gate_pass"])
        elif args.selection_score == "binary_robust_proxy_selection":
            task_proxy_selection = compute_binary_robust_proxy_selection(val_prediction_rows) if is_main else None
            task_proxy_selection = distributed_broadcast_object(task_proxy_selection, rank)
            score = float(task_proxy_selection["selection_score"])
            checkpoint_gate_pass = bool(task_proxy_selection["gate_pass"])
        elif args.selection_score == "fpr_constrained_label13_accuracy":
            fpr_constrained_selection = (
                fpr_constrained_checkpoint_decision(
                    checkpoint_selection_metrics,
                    args.selection_normal_fpr_max,
                )
                if is_main
                else None
            )
            fpr_constrained_selection = distributed_broadcast_object(
                fpr_constrained_selection,
                rank,
            )
            score = float(fpr_constrained_selection["label_13_accuracy"])
            checkpoint_gate_pass = bool(fpr_constrained_selection["normal_fpr_pass"])
        elif args.selection_score == "target_prior_weighted_accuracy":
            score = checkpoint_selection_metrics["target_prior_weighted_accuracy"]
        elif args.selection_score == "label_13_accuracy":
            score = checkpoint_selection_metrics["label_13_accuracy"]
        elif args.selection_score == "fraud_head_forced_macro_f1":
            forced_metrics = checkpoint_selection_metrics.get("fraud_head_forced_12class")
            if forced_metrics is None:
                raise RuntimeError(
                    "fraud_head_forced_macro_f1 selection requires subtype logits"
                )
            score = forced_metrics["macro_f1_observed"]
        else:
            score = val_metrics["observed_label_macro_f1"] + val_metrics["binary"]["f1"]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(loader)),
            "typed_evidence_loss": total_typed_evidence_loss / max(1, len(loader)),
            "typed_evidence_axis_loss": total_typed_evidence_axis_loss / max(1, len(loader)),
            "typed_evidence_contrastive_weight": args.typed_evidence_contrastive_weight,
            "typed_evidence_effective_weight": (
                args.typed_evidence_contrastive_weight
                if epoch > args.typed_evidence_warmup_epochs
                else 0.0
            ),
            "typed_evidence_axis_weight": args.typed_evidence_axis_weight,
            "typed_evidence_axis_mode": args.typed_evidence_axis_mode,
            "typed_evidence_axis_effective_weight": (
                args.typed_evidence_axis_weight
                if epoch > args.typed_evidence_warmup_epochs
                else 0.0
            ),
            "typed_evidence_axis_class_weight": args.typed_evidence_axis_class_weight,
            "typed_evidence_axis_confusable_margin_weight": args.typed_evidence_axis_confusable_margin_weight,
            "typed_evidence_labels": sorted(typed_evidence_allowed_labels) if typed_evidence_allowed_labels is not None else [],
            "typed_evidence_consistency_loss": total_typed_evidence_consistency_loss / max(1, len(loader)),
            "typed_evidence_consistency_weight": args.typed_evidence_consistency_weight,
            "typed_evidence_consistency_mode": args.typed_evidence_consistency_mode,
            "typed_evidence_consistency_effective_weight": (
                args.typed_evidence_consistency_weight
                if epoch > args.typed_evidence_warmup_epochs
                else 0.0
            ),
            "architecture": args.architecture,
            "ce_gate_alignment_loss": total_ce_gate_alignment_loss / max(1, len(loader)),
            "ce_gate_alignment_weight": args.ce_gate_alignment_weight,
            "span_region_ce_axis_loss": total_span_region_ce_axis_loss / max(1, len(loader)),
            "span_region_ce_axis_weight": args.span_region_ce_axis_weight,
            "span_region_ce_axis_effective_weight": (
                args.span_region_ce_axis_weight if epoch > args.span_region_ce_warmup_epochs else 0.0
            ),
            "span_region_ce_contrastive_loss": total_span_region_ce_contrastive_loss / max(1, len(loader)),
            "span_region_ce_contrastive_weight": args.span_region_ce_contrastive_weight,
            "span_region_ce_contrastive_effective_weight": (
                args.span_region_ce_contrastive_weight if epoch > args.span_region_ce_warmup_epochs else 0.0
            ),
            "span_region_ce_consistency_loss": total_span_region_ce_consistency_loss / max(1, len(loader)),
            "span_region_ce_consistency_weight": args.span_region_ce_consistency_weight,
            "span_region_ce_consistency_effective_weight": (
                args.span_region_ce_consistency_weight if epoch > args.span_region_ce_warmup_epochs else 0.0
            ),
            "span_region_ce_labels": sorted(span_region_ce_allowed_labels) if span_region_ce_allowed_labels is not None else [],
            "ce_neck_gate_bias": args.ce_neck_gate_bias,
            "dual_binary_loss": total_dual_binary_loss / max(1, len(loader)),
            "dual_binary_loss_weight": args.dual_binary_loss_weight,
            "dual_subtype_loss": total_dual_subtype_loss / max(1, len(loader)),
            "dual_subtype_loss_weight": args.dual_subtype_loss_weight,
            "dual_subtype_class_weight": args.dual_subtype_class_weight,
            "dual_subtype_logit_adjust_tau": args.dual_subtype_logit_adjust_tau,
            "h2_flat_aux_loss": total_h2_flat_aux_loss / max(1, len(loader)),
            "h2_flat_aux_loss_weight": args.h2_flat_aux_loss_weight,
            "h2_binary_consistency_loss": total_h2_binary_consistency_loss
            / max(1, len(loader)),
            "h2_binary_consistency_weight": args.h2_binary_consistency_weight,
            "official_subcategory_loss": total_official_subcategory_loss / max(1, len(loader)),
            "official_subcategory_loss_weight": args.official_subcategory_loss_weight,
            "official_subcategory_loss_mode": args.official_subcategory_loss_mode,
            "official_subcategory_class_weight": args.official_subcategory_class_weight,
            "clear11_loss": total_clear11_loss / max(1, len(loader)),
            "clear11_loss_weight": args.clear11_loss_weight,
            "clear11_class_weight": args.clear11_class_weight,
            "clear11_detach": bool(args.clear11_detach),
            "clear11_label_counts": clear11_label_counts,
            "residual12_loss": total_residual12_loss / max(1, len(loader)),
            "residual12_loss_weight": args.residual12_loss_weight,
            "residual12_pos_weight": args.residual12_pos_weight,
            "residual12_detach": bool(args.residual12_detach),
            "local_expert_loss": total_local_expert_loss / max(1, len(loader)),
            "local_expert_loss_weight": args.local_expert_loss_weight,
            "local_expert_groups": local_expert_groups,
            "local_expert_residual_scale": args.local_expert_residual_scale,
            "local_expert_gate_bias": args.local_expert_gate_bias,
            "local_expert_rescue_weight": args.local_expert_rescue_weight,
            "local_expert_warmup_epochs": args.local_expert_warmup_epochs,
            "binary_aux_loss": total_binary_aux_loss / max(1, len(loader)),
            "binary_aux_loss_weight": args.binary_aux_loss_weight,
            "binary_aux_rows": len(binary_aux_rows),
            "distill_loss": total_distill_loss / max(1, len(loader)),
            "distill_loss_weight": args.distill_loss_weight,
            "distill_temperature": args.distill_temperature,
            "rdrop_kl_loss": total_rdrop_kl_loss / max(1, len(loader)),
            "rdrop_kl_weight": args.rdrop_kl_weight,
            "train_distill_rows": sum(int(ex.has_distill_probs) for ex in train_ds.examples),
            "typed_evidence_axis_target": args.typed_evidence_axis_target,
            "pair_margin_weight": args.pair_margin_weight,
            "confusable_margin_weight": args.confusable_margin_weight,
            "val": val_metrics,
            "checkpoint_selection": {
                "source": checkpoint_selection_source,
                "metrics": checkpoint_selection_metrics,
            },
            "selection_score": score,
            "selection_score_mode": args.selection_score,
            "checkpoint_gate_pass": checkpoint_gate_pass,
            "task_v21_proxy_selection": task_proxy_selection,
            "fpr_constrained_selection": fpr_constrained_selection,
        }
        history.append(row)
        train_module = model.module if distributed else model
        if is_main:
            append_epoch_history(
                output,
                index=epoch,
                train_loss=total_loss / max(1, len(loader)),
                selection_score=score,
                val=val_metrics,
                extras={key: value for key, value in row.items() if key not in {"epoch", "train_loss", "val", "selection_score"}},
            )
            if output != mirror_output:
                mirror_file(output / "history.json", mirror_output / "history.json")
            update_train_progress(
                epoch_current=epoch,
                epoch_total=args.epochs,
                step_current=min(total_steps, optimizer_step),
                step_total=total_steps,
                force=True,
            )
        if args.save_epoch_checkpoints and is_main:
            checkpoint_dir = output / "checkpoints" / f"epoch-{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if args.architecture in CUSTOM_ARCHITECTURES:
                torch.save(train_module.state_dict(), checkpoint_dir / "pytorch_model.bin")
                write_json(
                    checkpoint_dir / "custom_model.json",
                    {
                        "architecture": args.architecture,
                        "joint_mode": (
                            "hierarchical"
                            if args.architecture
                            in {"ce_dual_head_hier", "ce_dual_head_h2", "h2_cascade"}
                            else "additive"
                        ),
                        "ce_mid_adapter_layer": args.ce_mid_adapter_layer,
                        "ce_mid_adapter_bottleneck": args.ce_mid_adapter_bottleneck,
                        "local_expert_groups": local_expert_groups,
                        "local_expert_residual_scale": args.local_expert_residual_scale,
                        "local_expert_gate_bias": args.local_expert_gate_bias,
                        "local_expert_rescue_weight": args.local_expert_rescue_weight,
                        "local_expert_warmup_epochs": args.local_expert_warmup_epochs,
                    },
                )
            else:
                train_module.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        # The FPR-constrained selector retains a diagnostic fallback when no
        # epoch passes the ceiling, but marks it non-promotable in metrics.
        if args.fixed_final_epoch:
            checkpoint_is_better = epoch == args.epochs
        elif args.selection_score == "fpr_constrained_label13_accuracy":
            if fpr_constrained_selection is None:
                raise RuntimeError("missing FPR-constrained checkpoint decision")
            selection_key = tuple(
                float(value) for value in fpr_constrained_selection["ranking_key"]
            )
            checkpoint_is_better = (
                best_selection_key is None or selection_key > best_selection_key
            )
        # V23 prefers the best promotable checkpoint when one exists; until then
        # it retains the best diagnostic checkpoint so export can never be gated.
        elif args.selection_score in {"task_v23_proxy_selection", "task_v24_proxy_selection", "task_v22_repair_selection"}:
            checkpoint_is_better = (
                (checkpoint_gate_pass and not best_checkpoint_gate_pass)
                or (checkpoint_gate_pass == best_checkpoint_gate_pass and score > best_score + args.early_stopping_min_delta)
            )
        else:
            checkpoint_is_better = checkpoint_gate_pass and score > best_score + args.early_stopping_min_delta
        if checkpoint_is_better:
            best_score = score
            if args.selection_score == "fpr_constrained_label13_accuracy":
                best_selection_key = selection_key
            best_epoch = epoch
            best_checkpoint_gate_pass = checkpoint_gate_pass
            epochs_without_improvement = 0
            if args.skip_save_best_model:
                if is_main:
                    best_state_dict = {key: value.detach().cpu().clone() for key, value in train_module.state_dict().items()}
            elif args.architecture in CUSTOM_ARCHITECTURES:
                if is_main:
                    best_state_dict = {key: value.detach().cpu().clone() for key, value in train_module.state_dict().items()}
                    (output / "best_model").mkdir(parents=True, exist_ok=True)
                    torch.save(best_state_dict, output / "best_model" / "pytorch_model.bin")
                    write_json(
                        output / "best_model" / "custom_model.json",
                        {
                            "architecture": args.architecture,
                            "joint_mode": (
                                "hierarchical"
                                if args.architecture
                                in {"ce_dual_head_hier", "ce_dual_head_h2", "h2_cascade"}
                                else "additive"
                            ),
                            "ce_mid_adapter_layer": args.ce_mid_adapter_layer,
                            "ce_mid_adapter_bottleneck": args.ce_mid_adapter_bottleneck,
                            "local_expert_groups": local_expert_groups,
                            "local_expert_residual_scale": args.local_expert_residual_scale,
                            "local_expert_gate_bias": args.local_expert_gate_bias,
                            "local_expert_rescue_weight": args.local_expert_rescue_weight,
                            "local_expert_warmup_epochs": args.local_expert_warmup_epochs,
                            "official_subcategory_aux": args.official_subcategory_loss_weight > 0,
                            "official_subcategory_loss_mode": args.official_subcategory_loss_mode,
                        },
                    )
            else:
                if is_main:
                    train_module.save_pretrained(output / "best_model")
            if not args.skip_save_best_model and is_main:
                tokenizer.save_pretrained(output / "best_model")
            if is_main:
                write_json(output / "best_epoch.json", row)
                if output != mirror_output:
                    mirror_file(output / "best_epoch.json", mirror_output / "best_epoch.json")
        elif best_epoch > 0:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                break
        distributed_barrier()

    distributed_barrier()
    if args.checkpoint_bank_only:
        protected_main_sha256_after_training = (
            protected_main_state_sha256(model)
            if should_track_protected_main_hash(args.architecture)
            else None
        )
        protected_main_unchanged = (
            protected_main_sha256_before_training == protected_main_sha256_after_training
            if args.freeze_protected_main_path
            else None
        )
        if args.freeze_protected_main_path and not protected_main_unchanged:
            raise RuntimeError("protected main-path parameters changed during checkpoint-bank training")
        if is_main:
            bank_metrics = {
                "mode": "checkpoint_bank_only",
                "epochs_completed": len(history),
                "selection_deferred": True,
                "checkpoint_selection": {
                    "source": "new_internal_dev_pending",
                    "metric": "label_13_accuracy",
                    "metrics": None,
                },
                "external": {},
                "class_weights": weights,
                "total_parameter_count": total_parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "freeze_base_model": bool(args.freeze_base_model),
                "freeze_base_bottom_layers": int(args.freeze_base_bottom_layers),
                "partial_base_freeze": partial_base_freeze,
                "protected_main_sha256_before_training": protected_main_sha256_before_training,
                "protected_main_sha256_after_training": protected_main_sha256_after_training,
                "protected_main_unchanged": protected_main_unchanged,
            }
            write_json(output / "metrics.json", bank_metrics)
            if output != mirror_output:
                write_json(mirror_output / "metrics.json", bank_metrics)
            update_progress(
                phase="completed",
                phase_current="checkpoint_bank_ready",
                phase_done=args.epochs,
                phase_total=args.epochs,
                epoch_current=len(history),
                epoch_total=args.epochs,
                force=True,
            )
        distributed_barrier()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return
    if best_epoch == 0:
        if is_main:
            write_json(
                output / "checkpoint_gate_failure.json",
                {
                    "selection_score_mode": args.selection_score,
                    "reason": "no checkpoint satisfied the mandatory Task v21 selection gate",
                    "epochs_completed": len(history),
                    "history": history,
                },
            )
            if output != mirror_output:
                mirror_file(output / "checkpoint_gate_failure.json", mirror_output / "checkpoint_gate_failure.json")
        distributed_barrier()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        raise SystemExit("no checkpoint satisfied the mandatory Task v21 selection gate")
    if (
        args.selection_score == "fpr_constrained_label13_accuracy"
        and not best_checkpoint_gate_pass
        and is_main
    ):
        write_json(
            output / "checkpoint_gate_failure.json",
            {
                "selection_score_mode": args.selection_score,
                "reason": "no epoch met the normal-FPR ceiling; selected the diagnostic fallback only",
                "normal_fpr_max": args.selection_normal_fpr_max,
                "selected_epoch": best_epoch,
                "history": history,
            },
        )
        if output != mirror_output:
            mirror_file(
                output / "checkpoint_gate_failure.json",
                mirror_output / "checkpoint_gate_failure.json",
            )
    protected_main_sha256_after_training = (
        protected_main_state_sha256(model)
        if should_track_protected_main_hash(args.architecture)
        else None
    )
    protected_main_unchanged = (
        protected_main_sha256_before_training == protected_main_sha256_after_training
        if args.freeze_protected_main_path
        else None
    )
    if args.freeze_protected_main_path and not protected_main_unchanged:
        raise RuntimeError("protected main-path parameters changed during frozen stage B")
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    metrics: dict[str, Any] | None = None
    if is_main:
        eval_model = build_model_from_args(args, local_expert_groups, device)
        if args.protected_main_only_mode:
            setattr(eval_model, "protected_main_only_mode", True)
        if args.skip_save_best_model:
            if best_state_dict is not None:
                eval_model.load_state_dict({key: value.to(device) for key, value in best_state_dict.items()})
        elif args.architecture in CUSTOM_ARCHITECTURES:
            state_dict = torch.load(output / "best_model" / "pytorch_model.bin", map_location=device)
            eval_model.load_state_dict(state_dict)
        else:
            eval_model = AutoModelForSequenceClassification.from_pretrained(output / "best_model", local_files_only=True).to(device)
        if args.weight_averaging_mode != "none" and averaging_state is not None:
            with torch.no_grad():
                for name, value in eval_model.named_parameters():
                    value.copy_(averaging_state[name].to(device=device, dtype=value.dtype))
        protected_main_sha256_selected_checkpoint = (
            protected_main_state_sha256(eval_model)
            if should_track_protected_main_hash(args.architecture)
            else None
        )

        protected_official_alpha_selection: list[dict[str, float]] = []
        selected_protected_official_alpha: float | None = None
        checkpoint_selection_dataset = selection_eval_ds or val_ds
        checkpoint_selection_source = (
            args.selection_eval_name if selection_eval_ds is not None else "internal_val"
        )
        update_eval_progress(dataset="internal_val_final", done=0, total=1, force=True)
        val_metrics, val_predictions = predict(
            eval_model, val_ds, args.eval_batch_size, device
        )
        update_eval_progress(dataset="internal_val_final", done=1, total=1, force=True)
        if protected_official_alpha_grid:
            update_eval_progress(
                dataset=f"{checkpoint_selection_source}_alpha_selection",
                done=0,
                total=len(protected_official_alpha_grid),
                force=True,
            )
            alpha_candidates: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
            for index, alpha in enumerate(protected_official_alpha_grid, start=1):
                setattr(eval_model, "protected_official_weight", float(alpha))
                candidate_metrics, candidate_predictions = predict(
                    eval_model, checkpoint_selection_dataset, args.eval_batch_size, device
                )
                candidate_acc = float(candidate_metrics["label_13_accuracy"])
                protected_official_alpha_selection.append(
                    {"alpha": float(alpha), "label_13_accuracy": candidate_acc}
                )
                alpha_candidates.append((float(alpha), candidate_metrics, candidate_predictions))
                update_eval_progress(
                    dataset=f"{checkpoint_selection_source}_alpha_selection",
                    done=index,
                    total=len(protected_official_alpha_grid),
                    force=True,
                )
            # Prefer the smaller alpha on an exact ACC tie, so alpha=0 remains
            # the deterministic no-change control rather than a hidden tie-break.
            (
                selected_protected_official_alpha,
                checkpoint_selection_metrics_final,
                checkpoint_selection_predictions_final,
            ) = max(
                alpha_candidates,
                key=lambda item: (float(item[1]["label_13_accuracy"]), -item[0]),
            )
            setattr(
                eval_model,
                "protected_official_weight",
                selected_protected_official_alpha,
            )
            if not args.skip_save_best_model:
                write_json(
                    output / "best_model" / "decode_config.json",
                    {
                        "architecture": args.architecture,
                        "protected_official_selected_alpha": selected_protected_official_alpha,
                        "selection_metric": "label_13_accuracy",
                        "selection_source": checkpoint_selection_source,
                    },
                )
        else:
            if selection_eval_ds is None:
                checkpoint_selection_metrics_final = val_metrics
                checkpoint_selection_predictions_final = val_predictions
            else:
                update_eval_progress(
                    dataset=f"{checkpoint_selection_source}_final",
                    done=0,
                    total=1,
                    force=True,
                )
                (
                    checkpoint_selection_metrics_final,
                    checkpoint_selection_predictions_final,
                ) = predict(
                    eval_model,
                    selection_eval_ds,
                    args.eval_batch_size,
                    device,
                )
                update_eval_progress(
                    dataset=f"{checkpoint_selection_source}_final",
                    done=1,
                    total=1,
                    force=True,
                )
        prediction_artifacts: dict[str, dict[str, str]] = {
            "internal_val": write_prediction_artifacts(output, "val", val_predictions)
        }
        if selection_eval_ds is not None:
            prediction_artifacts[checkpoint_selection_source] = write_prediction_artifacts(
                output,
                checkpoint_selection_source,
                checkpoint_selection_predictions_final,
            )
        external: dict[str, Any] = {}
        for idx, (name, dataset, path, _rows) in enumerate(eval_datasets, start=1):
            update_eval_progress(dataset=name, done=idx - 1, total=len(eval_datasets), force=True)
            dataset_metrics, predictions = predict(eval_model, dataset, args.eval_batch_size, device)
            external[name] = dataset_metrics
            prediction_artifacts[name] = write_prediction_artifacts(output, name, predictions)
            update_eval_progress(dataset=name, done=idx, total=len(eval_datasets), force=True)

        metrics = {
            "best_epoch": best_epoch,
            "fixed_final_epoch": bool(args.fixed_final_epoch),
            "best_selection_score": best_score,
            "best_checkpoint_gate_pass": best_checkpoint_gate_pass,
            "selection_score_mode": args.selection_score,
            "selection_normal_fpr_max": args.selection_normal_fpr_max,
            "elapsed_sec": time.time() - start,
            "device": device,
            "distributed": distributed,
            "world_size": world_size,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "freeze_base_model": bool(args.freeze_base_model),
            "freeze_base_bottom_layers": int(args.freeze_base_bottom_layers),
            "partial_base_freeze": partial_base_freeze,
            "init_protected_checkpoint": args.init_protected_checkpoint,
            "freeze_protected_main_path": bool(args.freeze_protected_main_path),
            "reinitialize_protected_fraud_branch": bool(
                args.reinitialize_protected_fraud_branch
            ),
            "protected_main_only_mode": bool(args.protected_main_only_mode),
            "protected_fraud_top_layers": (
                args.protected_fraud_top_layers
                if args.architecture == "ce_protected_top_layers"
                else None
            ),
            "train_fraud_only": bool(args.train_fraud_only),
            "protected_main_sha256_before_training": protected_main_sha256_before_training,
            "protected_main_sha256_after_training": protected_main_sha256_after_training,
            "protected_main_sha256_selected_checkpoint": protected_main_sha256_selected_checkpoint,
            "protected_main_unchanged": protected_main_unchanged,
            "weight_averaging_mode": args.weight_averaging_mode,
            "ema_decay": args.ema_decay if args.weight_averaging_mode == "ema" else None,
            "weight_averaging_start_epoch": args.weight_averaging_start_epoch,
            "weight_averaging_updates": averaging_updates,
            "total_parameter_count": total_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "class_weights": weights,
            "dual_subtype_class_weight": args.dual_subtype_class_weight,
            "dual_subtype_class_weights": subtype_weights,
            "dual_subtype_logit_adjust_tau": args.dual_subtype_logit_adjust_tau,
            "dual_subtype_label_counts": subtype_label_counts,
            "h2_flat_aux_loss_weight": args.h2_flat_aux_loss_weight,
            "h2_binary_consistency_weight": args.h2_binary_consistency_weight,
            "official_subcategory_loss_weight": args.official_subcategory_loss_weight,
            "official_subcategory_loss_mode": args.official_subcategory_loss_mode,
            "official_subcategory_class_weight": args.official_subcategory_class_weight,
            "official_subcategory_class_weight_cap": args.official_subcategory_class_weight_cap,
            "official_subcategory_class_weights": official_subcategory_weights,
            "official_subcategory_codes": list(OFFICIAL_SUBCATEGORY_CODES),
            "official_subcategory_label_counts": official_subcategory_label_counts,
            "protected_official_alpha_grid": protected_official_alpha_selection,
            "protected_official_selected_alpha": selected_protected_official_alpha,
            "clear11_loss_weight": args.clear11_loss_weight,
            "clear11_class_weight": args.clear11_class_weight,
            "clear11_class_weights": clear11_weights,
            "clear11_detach": bool(args.clear11_detach),
            "clear11_label_counts": clear11_label_counts,
            "residual12_loss_weight": args.residual12_loss_weight,
            "residual12_pos_weight": args.residual12_pos_weight,
            "residual12_detach": bool(args.residual12_detach),
            "local_expert_groups": local_expert_groups,
            "local_expert_loss_weight": args.local_expert_loss_weight,
            "local_expert_residual_scale": args.local_expert_residual_scale,
            "local_expert_gate_bias": args.local_expert_gate_bias,
            "local_expert_rescue_weight": args.local_expert_rescue_weight,
            "local_expert_warmup_epochs": args.local_expert_warmup_epochs,
            "typed_evidence_axis_class_weights": axis_weights,
            "binary_aux_jsonl": args.binary_aux_jsonl,
            "binary_aux_loss_weight": args.binary_aux_loss_weight,
            "binary_aux_rows": len(binary_aux_rows),
            "distill_loss_weight": args.distill_loss_weight,
            "distill_temperature": args.distill_temperature,
            "distill_prob_field": args.distill_prob_field,
            "train_distill_rows": sum(int(ex.has_distill_probs) for ex in train_ds.examples),
            "val": val_metrics,
            "checkpoint_selection": {
                "source": checkpoint_selection_source,
                "path": args.selection_eval_csv or None,
                "metric": args.selection_score,
                "normal_fpr_max": (
                    args.selection_normal_fpr_max
                    if args.selection_score == "fpr_constrained_label13_accuracy"
                    else None
                ),
                "metrics": checkpoint_selection_metrics_final,
            },
            "external": external,
            "prediction_artifacts": prediction_artifacts,
        }
        write_json(output / "metrics.json", metrics)
        if output != mirror_output:
            write_json(
                mirror_output / "artifact_manifest.json",
                {
                    "artifact_output": str(output),
                    "mirrored_from_artifact_output": True,
                    "skip_save_best_model": bool(args.skip_save_best_model),
                },
            )
            for relative in [
                "config.json",
                "split_manifest.json",
                "data_manifest.json",
                "history.json",
                "best_epoch.json",
                "metrics.json",
            ]:
                src = output / relative
                if src.exists():
                    mirror_file(src, mirror_output / relative)
            predictions_dir = output / "predictions"
            if predictions_dir.exists():
                for src in sorted([*predictions_dir.glob("*.csv"), *predictions_dir.glob("*.jsonl")]):
                    mirror_file(src, mirror_output / "predictions" / src.name)
            error_analysis_dir = output / "error_analysis"
            if error_analysis_dir.exists():
                for src in sorted(error_analysis_dir.glob("*.json")):
                    mirror_file(src, mirror_output / "error_analysis" / src.name)
        update_progress(
            phase="completed",
            phase_current="done",
            phase_done=args.epochs,
            phase_total=args.epochs,
            epoch_current=best_epoch or len(history),
            epoch_total=args.epochs,
            force=True,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    distributed_barrier()


if __name__ == "__main__":
    main()
