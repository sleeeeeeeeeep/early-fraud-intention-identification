from __future__ import annotations

from text_model.data.longdoc_chunker import HEAD_MODE, PLAIN_MODE, ChunkConfig, LongDocChunker
from text_model.data.role_markers import strip_role_markers


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_):
        result = {"input_ids": [10 + (ord(char) % 80) for char in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def build_inputs_with_special_tokens(self, first, second=None):
        if second is None:
            return [self.cls_token_id, *first, self.sep_token_id]
        return [self.cls_token_id, *first, self.sep_token_id, *second, self.sep_token_id]


def test_short_text_is_one_chunk_and_modes_are_identical():
    tokenizer = FakeTokenizer()
    text = "来电方：短文本"
    plain = LongDocChunker(
        tokenizer, ChunkConfig(chunk_length=64, overlap=8, input_mode=PLAIN_MODE)
    ).chunk("short", text)
    head = LongDocChunker(
        tokenizer,
        ChunkConfig(
            chunk_length=64,
            overlap=8,
            input_mode=HEAD_MODE,
            head_context_length=16,
        ),
    ).chunk("short", text)
    assert plain.num_chunks == 1
    assert head.num_chunks == 1
    assert head.head_context_range is None
    assert plain.chunks[0].input_ids == head.chunks[0].input_ids
    assert plain.coverage_ratio == 1.0


def test_long_text_covers_tail_without_silent_truncation():
    tokenizer = FakeTokenizer()
    text = "\n".join(
        [f"来电方：第{i}轮要求核验并说明。" + "风险" * 8 for i in range(10)]
    )
    for mode in (PLAIN_MODE, HEAD_MODE):
        document = LongDocChunker(
            tokenizer,
            ChunkConfig(
                chunk_length=48,
                overlap=8,
                input_mode=mode,
                head_context_length=12,
                max_chunks_per_call=2,
            ),
        ).chunk("long", text)
        assert document.coverage_ratio == 1.0
        assert document.chunks[-1].token_end == document.raw_wordpiece_length
        assert document.max_chunks_exceeded


def test_audio_role_markers_are_excluded_without_touching_body_words():
    text = "来电人：您好，我是平台客服。\n接电人：请说明。\n备注：请联系官方客服：95000。"
    stripped = strip_role_markers(text)
    assert "来电人：" not in stripped
    assert "接电人：" not in stripped
    assert "平台客服" in stripped
    assert "官方客服：95000" in stripped
    tokenizer = FakeTokenizer()
    marked = LongDocChunker(
        tokenizer, ChunkConfig(chunk_length=64, overlap=8, input_mode=PLAIN_MODE)
    ).chunk("marked", text)
    plain = LongDocChunker(
        tokenizer, ChunkConfig(chunk_length=64, overlap=8, input_mode=PLAIN_MODE)
    ).chunk("plain", stripped)
    assert marked.raw_wordpiece_length == plain.raw_wordpiece_length
