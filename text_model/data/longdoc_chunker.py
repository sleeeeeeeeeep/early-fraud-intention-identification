"""确定性长文本分块器。

实现任务书中的两种输入模式：

* ``plain_sliding_chunks``：只使用连续局部滑窗；
* ``head_context_enhanced_chunks``：在长文本的每个局部块前携带开头上下文。

分块器永远不截掉尾部文本。``max_chunks_per_call`` 只是诊断预算，超过时
仍然返回全部块，并在结果中显式标记，交给动态 batching 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, Mapping, Sequence

from .role_markers import strip_role_markers


PLAIN_MODE = "plain_sliding_chunks"
HEAD_MODE = "head_context_enhanced_chunks"
SUPPORTED_INPUT_MODES = {PLAIN_MODE, HEAD_MODE}


@dataclass(frozen=True)
class ChunkConfig:
    chunk_length: int = 320
    overlap: int = 64
    input_mode: str = PLAIN_MODE
    head_context_length: int = 64
    max_chunks_per_call: int = 12

    def __post_init__(self) -> None:
        if self.chunk_length < 8:
            raise ValueError("chunk_length must leave room for special tokens")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.input_mode not in SUPPORTED_INPUT_MODES:
            raise ValueError(f"unsupported input_mode: {self.input_mode}")
        if self.head_context_length < 0:
            raise ValueError("head_context_length must be non-negative")
        if self.max_chunks_per_call < 1:
            raise ValueError("max_chunks_per_call must be positive")


@dataclass(frozen=True)
class TurnSpan:
    char_start: int
    char_end: int
    turn_id: int


@dataclass
class DocumentChunk:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    token_start: int
    token_end: int
    speaker_turn_ids: list[int]
    chunk_index: int
    position_features: list[float]
    head_token_start: int | None = None
    head_token_end: int | None = None

    @property
    def token_range(self) -> tuple[int, int]:
        return self.token_start, self.token_end


@dataclass
class ChunkedDocument:
    sample_id: str
    raw_char_length: int
    raw_wordpiece_length: int
    chunk_length: int
    chunk_stride: int
    num_chunks: int
    chunk_token_ranges: list[tuple[int, int]]
    chunks: list[DocumentChunk]
    was_chunked: bool
    input_mode: str
    head_context_range: tuple[int, int] | None
    max_chunks_exceeded: bool
    coverage_ratio: float
    turn_token_ranges: list[tuple[int, int, int]] = field(default_factory=list)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "raw_char_length": self.raw_char_length,
            "raw_wordpiece_length": self.raw_wordpiece_length,
            "chunk_length": self.chunk_length,
            "chunk_stride": self.chunk_stride,
            "num_chunks": self.num_chunks,
            "chunk_token_ranges": [list(item) for item in self.chunk_token_ranges],
            "was_chunked": self.was_chunked,
            "input_mode": self.input_mode,
            "head_context_range": (
                list(self.head_context_range)
                if self.head_context_range is not None
                else None
            ),
            "max_chunks_exceeded": self.max_chunks_exceeded,
            "coverage_ratio": self.coverage_ratio,
            "turn_token_ranges": [list(item) for item in self.turn_token_ranges],
        }


_SPEAKER_MARKER = re.compile(
    r"(?m)^[ \t]*(?:来电方|来电人|接听方|接听人|接电人|对方|用户|客服|客户|诈骗分子|受害人|"
    r"说话人[一二三四五六七八九十]?|甲|乙|A|B)\s*[:：]"
)


def extract_turn_spans(text: str) -> list[TurnSpan]:
    """提取对话轮次；没有说话人标记时退化为非空行。"""

    if not text:
        return []
    matches = list(_SPEAKER_MARKER.finditer(text))
    if matches:
        spans: list[TurnSpan] = []
        if matches[0].start() > 0 and text[: matches[0].start()].strip():
            spans.append(TurnSpan(0, matches[0].start(), 0))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            if text[match.start() : end].strip():
                spans.append(TurnSpan(match.start(), end, len(spans)))
        return spans

    spans = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        if line.strip():
            spans.append(TurnSpan(cursor, end, len(spans)))
        cursor = end
    if not spans:
        spans.append(TurnSpan(0, len(text), 0))
    return spans


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _flatten_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [int(item) for item in value]


def _tokenize_without_specials(
    tokenizer: Any, text: str
) -> tuple[list[int], list[tuple[int, int]] | None]:
    offsets: list[tuple[int, int]] | None = None
    try:
        result = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        raw_offsets = _result_value(result, "offset_mapping")
        if raw_offsets is not None:
            if hasattr(raw_offsets, "tolist"):
                raw_offsets = raw_offsets.tolist()
            if raw_offsets and isinstance(raw_offsets[0], (list, tuple)) and raw_offsets[0] and isinstance(raw_offsets[0][0], (list, tuple)):
                raw_offsets = raw_offsets[0]
            offsets = [tuple(map(int, pair)) for pair in raw_offsets]
    except (TypeError, ValueError, KeyError, NotImplementedError):
        result = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
    ids = _flatten_ids(_result_value(result, "input_ids", result))
    if offsets is not None and len(offsets) != len(ids):
        offsets = None
    return ids, offsets


def _turn_token_ranges(
    offsets: Sequence[tuple[int, int]] | None,
    spans: Sequence[TurnSpan],
) -> list[tuple[int, int, int]]:
    if not offsets or not spans:
        return []
    token_turns: list[int] = []
    for start, end in offsets:
        if end <= start:
            token_turns.append(-1)
            continue
        best_turn = -1
        best_overlap = 0
        for span in spans:
            overlap = max(0, min(end, span.char_end) - max(start, span.char_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_turn = span.turn_id
        token_turns.append(best_turn)
    ranges: list[tuple[int, int, int]] = []
    start = None
    current = None
    for index, turn_id in enumerate(token_turns + [-2]):
        if turn_id != current:
            if start is not None and current not in (None, -1):
                ranges.append((start, index, int(current)))
            start = index if turn_id not in (-1, -2) else None
            current = turn_id
    return ranges


def _special_count(tokenizer: Any, pair: bool) -> int:
    method = getattr(tokenizer, "num_special_tokens_to_add", None)
    if callable(method):
        try:
            return int(method(pair=pair))
        except TypeError:
            return int(method(pair))
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        return len(builder([], [] if pair else None))
    return 3 if pair else 2


def _build_chunk_encoding(
    tokenizer: Any,
    local_ids: Sequence[int],
    chunk_length: int,
    head_ids: Sequence[int] | None = None,
) -> tuple[list[int], list[int], list[int] | None]:
    local_ids = list(map(int, local_ids))
    head_ids = None if head_ids is None else list(map(int, head_ids))
    builder = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(builder):
        ids = (
            builder(head_ids, local_ids)
            if head_ids is not None
            else builder(local_ids)
        )
    else:
        cls_id = getattr(tokenizer, "cls_token_id", None)
        sep_id = getattr(tokenizer, "sep_token_id", None)
        if cls_id is None or sep_id is None:
            ids = local_ids
        elif head_ids is None:
            ids = [int(cls_id), *local_ids, int(sep_id)]
        else:
            ids = [int(cls_id), *head_ids, int(sep_id), *local_ids, int(sep_id)]
    ids = [int(item) for item in ids]
    if len(ids) > chunk_length:
        raise ValueError(
            f"constructed chunk has {len(ids)} tokens, exceeds chunk_length={chunk_length}"
        )
    type_ids: list[int] | None = None
    type_builder = getattr(tokenizer, "create_token_type_ids_from_sequences", None)
    if callable(type_builder):
        type_ids = (
            type_builder(head_ids, local_ids)
            if head_ids is not None
            else type_builder(local_ids)
        )
        type_ids = [int(item) for item in type_ids]
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0
    padding = chunk_length - len(ids)
    ids = ids + [int(pad_id)] * padding
    attention = [1] * (chunk_length - padding) + [0] * padding
    if type_ids is not None:
        type_ids = type_ids + [0] * padding
    return ids, attention, type_ids


def _choose_end(
    start: int,
    target_end: int,
    total: int,
    turn_ranges: Sequence[tuple[int, int, int]],
) -> int:
    if target_end >= total:
        return total
    candidates = [end for begin, end, _ in turn_ranges if start < end <= target_end]
    return max(candidates) if candidates else target_end


def _choose_next_start(
    end: int,
    requested: int,
    lower_bound: int,
    turn_ranges: Sequence[tuple[int, int, int]],
) -> int:
    requested = max(lower_bound, min(requested, end - 1))
    candidates = [
        begin
        for begin, _, _ in turn_ranges
        if lower_bound <= begin < end
    ]
    if not candidates:
        return requested
    return min(candidates, key=lambda value: abs(value - requested))


def _make_ranges(
    total: int,
    capacity: int,
    overlap: int,
    turn_ranges: Sequence[tuple[int, int, int]],
    *,
    start_at: int = 0,
) -> list[tuple[int, int]]:
    if total <= 0:
        return [(0, 0)]
    capacity = max(1, min(capacity, total))
    overlap = min(max(0, overlap), max(0, capacity - 1))
    ranges: list[tuple[int, int]] = []
    start = max(0, min(start_at, total - 1))
    while start < total:
        end = _choose_end(start, min(total, start + capacity), total, turn_ranges)
        if end <= start:
            end = min(total, start + capacity)
        ranges.append((start, end))
        if end >= total:
            break
        next_start = end - overlap
        next_start = _choose_next_start(end, next_start, start + 1, turn_ranges)
        if next_start <= start:
            next_start = min(end, start + 1)
        start = next_start
    return ranges


def make_plain_chunk_token_ranges(
    total_wordpieces: int,
    *,
    chunk_length: int = 320,
    overlap: int = 64,
    special_tokens: int = 2,
    turn_token_ranges: Sequence[tuple[int, int, int]] = (),
) -> list[tuple[int, int]]:
    """生成与普通滑窗一致的纯 token 范围，供数据入库审计使用。"""

    if total_wordpieces < 1:
        raise ValueError("total_wordpieces must be positive")
    capacity = int(chunk_length) - int(special_tokens)
    if capacity < 1:
        raise ValueError("chunk_length must leave room for special tokens")
    return _make_ranges(
        int(total_wordpieces),
        capacity,
        int(overlap),
        turn_token_ranges,
    )


def _select_head_end(
    total: int,
    max_tokens: int,
    turn_ranges: Sequence[tuple[int, int, int]],
) -> int:
    if total <= 0 or max_tokens <= 0:
        return 0
    candidates = [end for begin, end, _ in turn_ranges if begin == 0 and end <= max_tokens]
    if candidates:
        return max(candidates)
    for begin, end, _ in turn_ranges:
        if begin == 0:
            return min(end, max_tokens)
    return min(total, max_tokens)


class LongDocChunker:
    """对单条通话做全文覆盖的确定性分块。"""

    def __init__(self, tokenizer: Any, config: ChunkConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def chunk(self, sample_id: str, text: str) -> ChunkedDocument:
        # 角色标识仅服务于音频侧轮次恢复，不是电话内容；即使调用方误传了
        # 带标识文本，这里也再次剥离，确保 token 长度和分块位置不包含标识。
        text = strip_role_markers(str(text or ""))
        if not text.strip():
            raise ValueError(f"{sample_id}: empty text")
        token_ids, offsets = _tokenize_without_specials(self.tokenizer, text)
        if not token_ids:
            raise ValueError(f"{sample_id}: tokenizer produced no tokens")
        turns = extract_turn_spans(text)
        turn_ranges = _turn_token_ranges(offsets, turns)
        pair_specials = _special_count(self.tokenizer, pair=True)
        single_specials = _special_count(self.tokenizer, pair=False)
        raw_length = len(token_ids)
        requires_chunks = raw_length + single_specials > self.config.chunk_length

        head_end = 0
        head_ids: list[int] | None = None
        if self.config.input_mode == HEAD_MODE and requires_chunks:
            head_end = _select_head_end(
                raw_length,
                self.config.head_context_length,
                turn_ranges,
            )
            head_ids = token_ids[:head_end] or None

        if head_ids is None:
            capacity = self.config.chunk_length - single_specials
            ranges = _make_ranges(
                raw_length,
                capacity,
                self.config.overlap,
                turn_ranges,
            )
            chunk_stride = max(1, capacity - min(self.config.overlap, capacity - 1))
        else:
            capacity = self.config.chunk_length - pair_specials - len(head_ids)
            if capacity < 1:
                raise ValueError(
                    f"{sample_id}: head context leaves no local token capacity"
                )
            ranges = _make_ranges(
                raw_length,
                capacity,
                self.config.overlap,
                turn_ranges,
                start_at=head_end,
            )
            chunk_stride = max(1, capacity - min(self.config.overlap, capacity - 1))

        # A short document is deliberately represented by one plain chunk for
        # both modes; no duplicated head context is allowed.
        if not requires_chunks:
            ranges = [(0, raw_length)]
            head_ids = None
            head_end = 0
            chunk_stride = max(1, raw_length)

        chunks: list[DocumentChunk] = []
        for index, (start, end) in enumerate(ranges):
            ids, attention, type_ids = _build_chunk_encoding(
                self.tokenizer,
                token_ids[start:end],
                self.config.chunk_length,
                head_ids=head_ids,
            )
            position_features = [
                float(index / max(len(ranges) - 1, 1)),
                float(start / max(raw_length, 1)),
                float(end / max(raw_length, 1)),
                float(index == 0),
                float(index == len(ranges) - 1),
                float(1.0 - end / max(raw_length, 1)),
            ]
            speaker_ids = [
                turn_id
                for turn_start, turn_end, turn_id in turn_ranges
                if max(start, turn_start) < min(end, turn_end)
            ]
            chunks.append(
                DocumentChunk(
                    input_ids=ids,
                    attention_mask=attention,
                    token_type_ids=type_ids,
                    token_start=start,
                    token_end=end,
                    speaker_turn_ids=speaker_ids,
                    chunk_index=index,
                    position_features=position_features,
                    head_token_start=0 if head_ids is not None else None,
                    head_token_end=head_end if head_ids is not None else None,
                )
            )

        covered = [False] * raw_length
        if head_ids is not None:
            for index in range(head_end):
                covered[index] = True
        for start, end in ranges:
            for index in range(start, min(end, raw_length)):
                covered[index] = True
        coverage = sum(covered) / max(raw_length, 1)
        if coverage != 1.0:
            raise AssertionError(
                f"{sample_id}: chunk coverage is {coverage:.6f}, expected 1.0"
            )
        return ChunkedDocument(
            sample_id=str(sample_id),
            raw_char_length=len(text),
            raw_wordpiece_length=raw_length,
            chunk_length=self.config.chunk_length,
            chunk_stride=chunk_stride,
            num_chunks=len(chunks),
            chunk_token_ranges=[chunk.token_range for chunk in chunks],
            chunks=chunks,
            was_chunked=requires_chunks,
            input_mode=self.config.input_mode,
            head_context_range=(0, head_end) if head_ids is not None else None,
            max_chunks_exceeded=len(chunks) > self.config.max_chunks_per_call,
            coverage_ratio=coverage,
            turn_token_ranges=list(turn_ranges),
        )


def chunk_row(
    row: Mapping[str, Any], tokenizer: Any, config: ChunkConfig
) -> ChunkedDocument:
    sample_id = str(row.get("sample_id") or row.get("base_sample_id") or "").strip()
    text = str(
        row.get("model_input_text")
        or row.get("transcript")
        or row.get("text")
        or ""
    )
    if not sample_id:
        raise ValueError("row is missing sample_id/base_sample_id")
    return LongDocChunker(tokenizer, config).chunk(sample_id, text)
