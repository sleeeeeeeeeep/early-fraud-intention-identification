"""长文本模型的动态整通 collator。

batch 的第一维是整通电话，不是 chunk。不同电话的 chunk 数在 batch 内做
padding，``chunk_mask`` 保证所有聚合器都不会把 padding 当成证据。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .longdoc_chunker import ChunkConfig, ChunkedDocument, chunk_row


NUM_PARENT_CLASSES = 13
NUM_CHILD_CLASSES = 39


def _int_value(row: Mapping[str, Any], *keys: str, default: int | None = None) -> int:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return int(value)
    if default is None:
        raise ValueError(f"missing integer field: {keys}")
    return int(default)


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _target_distribution(
    row: Mapping[str, Any],
    field: str,
    fallback: int,
    size: int,
    code_to_index: Mapping[str, int] | None = None,
) -> list[float]:
    raw = row.get(field)
    if raw in (None, ""):
        return _one_hot(fallback, size)
    if isinstance(raw, Mapping) and "probabilities" in raw:
        raw = raw["probabilities"]
    if isinstance(raw, Mapping):
        values = [0.0] * size
        for key, value in raw.items():
            token = str(key)
            if token.isdigit():
                index = int(token)
            elif code_to_index is not None and token in code_to_index:
                index = int(code_to_index[token])
            else:
                raise ValueError(f"{field} code is unknown: {token}")
            if not 0 <= index < size:
                raise ValueError(f"{field} index out of range: {index}")
            values[index] = float(value)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(value) for value in raw]
        if len(values) != size:
            raise ValueError(f"{field} must contain {size} values")
    else:
        raise ValueError(f"unsupported {field} format")
    if any(value < 0 for value in values):
        raise ValueError(f"{field} contains a negative value")
    total = sum(values)
    if total <= 0:
        return _one_hot(fallback, size)
    return [value / total for value in values]


def _child_index(row: Mapping[str, Any], child_code_to_index: Mapping[str, int]) -> int:
    raw = row.get("canonical_child39_index", row.get("child39_index"))
    if raw not in (None, ""):
        return int(raw)
    code = str(
        row.get("canonical_subcategory_code")
        or row.get("subcategory_code")
        or ""
    ).strip()
    if not code:
        return -1
    if code not in child_code_to_index:
        raise ValueError(f"unknown official subcategory code: {code}")
    return int(child_code_to_index[code])


def _length_features(doc: ChunkedDocument) -> list[float]:
    raw = float(doc.raw_wordpiece_length)
    chars = float(doc.raw_char_length)
    chunks = float(doc.num_chunks)
    return [
        min(raw / 2048.0, 4.0),
        min(chars / 10000.0, 4.0),
        min(chunks / 16.0, 4.0),
        float(doc.was_chunked),
    ]


def _evidence_ranges(row: Mapping[str, Any]) -> list[tuple[int, int]]:
    raw = row.get("evidence_token_ranges") or []
    ranges: list[tuple[int, int]] = []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ranges
    for item in raw:
        if isinstance(item, Mapping):
            start = item.get("start", item.get("token_start"))
            end = item.get("end", item.get("token_end"))
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        if start is None or end is None:
            continue
        start, end = int(start), int(end)
        if 0 <= start < end:
            ranges.append((start, end))
    return ranges


class LongDocCollator:
    """把原始 JSONL 行编码为路线 A/B 共用的 batch。"""

    def __init__(
        self,
        tokenizer: Any,
        config: ChunkConfig,
        *,
        child_code_to_index: Mapping[str, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.child_code_to_index = dict(child_code_to_index or {})

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not rows:
            raise ValueError("cannot collate an empty batch")
        documents = [chunk_row(row, self.tokenizer, self.config) for row in rows]
        batch_size = len(documents)
        max_chunks = max(document.num_chunks for document in documents)
        chunk_length = self.config.chunk_length
        input_ids = torch.zeros(
            (batch_size, max_chunks, chunk_length), dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
        chunk_positions = torch.zeros((batch_size, max_chunks, 6), dtype=torch.float32)
        window_risk_target = torch.zeros((batch_size, max_chunks), dtype=torch.float32)
        window_risk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool)
        raw_char_length = torch.zeros(batch_size, dtype=torch.long)
        raw_wordpiece_length = torch.zeros(batch_size, dtype=torch.long)
        num_chunks = torch.zeros(batch_size, dtype=torch.long)
        global_length_features = torch.zeros((batch_size, 4), dtype=torch.float32)

        parent_labels: list[int] = []
        binary_targets: list[int] = []
        subtype_labels: list[int] = []
        child_labels: list[int] = []
        parent_targets: list[list[float]] = []
        child_targets: list[list[float]] = []
        sample_weights: list[float] = []
        binary_weights: list[float] = []
        parent_weights: list[float] = []
        child_weights: list[float] = []
        metadata: list[dict[str, Any]] = []

        for batch_index, (row, document) in enumerate(zip(rows, documents)):
            raw_parent = _int_value(
                row, "canonical_label_13", "label_13", default=0
            )
            if not 0 <= raw_parent < NUM_PARENT_CLASSES:
                raise ValueError(f"parent label out of range: {raw_parent}")
            child_index = (
                -1
                if raw_parent == 0
                else _child_index(row, self.child_code_to_index)
            )
            if raw_parent != 0 and not 0 <= child_index < NUM_CHILD_CLASSES:
                raise ValueError(
                    f"fraud row {document.sample_id} is missing child39 index"
                )
            for chunk_index, chunk in enumerate(document.chunks):
                input_ids[batch_index, chunk_index] = torch.tensor(
                    chunk.input_ids, dtype=torch.long
                )
                attention_mask[batch_index, chunk_index] = torch.tensor(
                    chunk.attention_mask, dtype=torch.long
                )
                if chunk.token_type_ids is not None:
                    token_type_ids[batch_index, chunk_index] = torch.tensor(
                        chunk.token_type_ids, dtype=torch.long
                    )
                chunk_mask[batch_index, chunk_index] = True
                chunk_positions[batch_index, chunk_index] = torch.tensor(
                    chunk.position_features, dtype=torch.float32
                )

            evidence_ranges = _evidence_ranges(row)
            if row.get("disable_window_evidence"):
                pass
            elif raw_parent == 0:
                window_risk_mask[batch_index, : document.num_chunks] = True
            elif evidence_ranges:
                head_has_evidence = (
                    document.head_context_range is not None
                    and any(
                        max(document.head_context_range[0], start)
                        < min(document.head_context_range[1], end)
                        for start, end in evidence_ranges
                    )
                )
                for chunk_index, (chunk_start, chunk_end) in enumerate(
                    document.chunk_token_ranges
                ):
                    has_evidence = head_has_evidence or any(
                        max(chunk_start, start) < min(chunk_end, end)
                        for start, end in evidence_ranges
                    )
                    if has_evidence:
                        window_risk_target[batch_index, chunk_index] = 1.0
                        window_risk_mask[batch_index, chunk_index] = True
            raw_char_length[batch_index] = document.raw_char_length
            raw_wordpiece_length[batch_index] = document.raw_wordpiece_length
            num_chunks[batch_index] = document.num_chunks
            global_length_features[batch_index] = torch.tensor(
                _length_features(document), dtype=torch.float32
            )

            parent_labels.append(raw_parent)
            raw_binary_target = int(row.get("binary_target", int(raw_parent != 0)))
            binary_loss_mask = float(row.get("binary_loss_mask", 1.0))
            if binary_loss_mask < 0.0:
                raise ValueError("binary_loss_mask must be non-negative")
            # V362 uses -1 for ambiguous rows.  They need a valid placeholder
            # target for tensor construction, while binary_weight=0 guarantees
            # that they contribute no supervised binary gradient.
            if raw_binary_target not in (0, 1):
                if binary_loss_mask > 0.0:
                    raise ValueError(
                        f"invalid supervised binary_target: {raw_binary_target}"
                    )
                raw_binary_target = 0
            binary_targets.append(raw_binary_target)
            subtype_labels.append(raw_parent - 1 if raw_parent != 0 else -1)
            child_labels.append(child_index)
            parent_targets.append(
                _target_distribution(row, "parent_target", raw_parent, NUM_PARENT_CLASSES)
            )
            child_targets.append(
                _target_distribution(
                    row,
                    "child_target",
                    child_index if child_index >= 0 else 0,
                    NUM_CHILD_CLASSES,
                    self.child_code_to_index,
                )
                if child_index >= 0
                else [0.0] * NUM_CHILD_CLASSES
            )
            sample_weight = float(row.get("sample_weight", 1.0))
            binary_weights.append(
                float(row.get("binary_loss_weight", sample_weight))
                * binary_loss_mask
            )
            parent_weights.append(float(row.get("parent_loss_weight", sample_weight)))
            child_weights.append(float(row.get("child_loss_weight", parent_weights[-1])))
            sample_weights.append(sample_weight)
            metadata.append(
                {
                    **document.to_metadata(),
                    "label_13": raw_parent,
                    "subcategory_code": str(
                        row.get("canonical_subcategory_code")
                        or row.get("subcategory_code")
                        or ""
                    ),
                    "scenario_tag": str(row.get("scenario_tag") or "general"),
                    "evidence_position_bin": str(
                        row.get("evidence_position_bin") or "unknown"
                    ),
                    "length_bucket": str(row.get("length_bucket") or "unknown"),
                    "generation_kind": str(
                        row.get("generation_kind") or ("normal" if raw_parent == 0 else "fraud")
                    ),
                    "content_group": str(row.get("content_group") or "source"),
                    "cross_window_evidence": bool(
                        row.get("cross_window_evidence", False)
                    ),
                    "caller_number_type": str(
                        row.get("caller_number_type") or "unknown"
                    ),
                    "review_status": str(row.get("review_status") or "source"),
                    "longdoc_generated": bool(row.get("longdoc_generated", False)),
                    "input_view": str(row.get("input_view") or "source"),
                }
            )

        return {
            "sample_id": [document.sample_id for document in documents],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "chunk_mask": chunk_mask,
            "chunk_positions": chunk_positions,
            "window_risk_target": window_risk_target,
            "window_risk_mask": window_risk_mask,
            "global_length_features": global_length_features,
            "raw_char_length": raw_char_length,
            "raw_wordpiece_length": raw_wordpiece_length,
            "num_chunks": num_chunks,
            "label_13": torch.tensor(parent_labels, dtype=torch.long),
            "binary_target": torch.tensor(binary_targets, dtype=torch.long),
            "subtype_label": torch.tensor(subtype_labels, dtype=torch.long),
            "child_label": torch.tensor(child_labels, dtype=torch.long),
            "parent_target": torch.tensor(parent_targets, dtype=torch.float32),
            "child_target": torch.tensor(child_targets, dtype=torch.float32),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
            "binary_weight": torch.tensor(binary_weights, dtype=torch.float32),
            "parent_weight": torch.tensor(parent_weights, dtype=torch.float32),
            "child_weight": torch.tensor(child_weights, dtype=torch.float32),
            "metadata": metadata,
        }
