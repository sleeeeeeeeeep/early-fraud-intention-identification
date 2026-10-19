from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from text_model.data.longdoc_chunker import (
    HEAD_MODE,
    PLAIN_MODE,
    ChunkConfig,
    LongDocChunker,
)
from text_model.data.longdoc_collator import LongDocCollator
from text_model.models.longdoc_multitask_classifier import LongDocMultiTaskClassifier
from text_model.models.fixed_logit_aggregator import aggregate_window_logits
from text_model.inference.run_longdoc_inference import (
    FIXED_CONTROL_MODES,
    read_rows,
    resolve_fixed_pooling_modes,
)
from text_model.training.longdoc_losses import longdoc_multitask_loss
from text_model.training.longdoc_sampler import LongExposureSampler
from text_model.inference.train_longdoc import RowsDataset


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=False,
        **_,
    ):
        ids = [10 + (ord(char) % 80) for char in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def build_inputs_with_special_tokens(self, first, second=None):
        if second is None:
            return [self.cls_token_id, *first, self.sep_token_id]
        return [self.cls_token_id, *first, self.sep_token_id, *second, self.sep_token_id]

    def create_token_type_ids_from_sequences(self, first, second=None):
        if second is None:
            return [0] * (len(first) + 2)
        return [0] * (len(first) + 2) + [1] * (len(second) + 1)


class FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(256, hidden_size)

    def forward(self, input_ids, attention_mask=None, return_dict=True, **_):
        hidden = self.embedding(input_ids.clamp(0, 255))
        return type("Output", (), {"last_hidden_state": hidden})()


def _fraud_row(sample_id="fraud"):
    return {
        "sample_id": sample_id,
        "model_input_text": "来电方：" + "风险动作要求转账。" * 20,
        "label_13": 4,
        "subcategory_code": "04_0403",
        "sample_weight": 1.0,
    }


def _normal_row(sample_id="normal"):
    return {
        "sample_id": sample_id,
        "model_input_text": "来电方：" + "正常业务确认。" * 20,
        "label_13": 0,
        "subcategory_code": "00_0000",
        "sample_weight": 1.0,
    }


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


def test_routes_share_collator_and_return_finite_outputs():
    tokenizer = FakeTokenizer()
    collator = LongDocCollator(
        tokenizer,
        ChunkConfig(chunk_length=48, overlap=8, input_mode=PLAIN_MODE),
        child_code_to_index={"04_0403": 1},
    )
    batch = collator([_normal_row(), _fraud_row()])
    for route in ("A", "B"):
        model = LongDocMultiTaskClassifier(
            FakeBackbone(),
            route=route,
            projection_dim=32,
            fusion_hidden=24,
            document_heads=4,
            document_intermediate=64,
        )
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            chunk_mask=batch["chunk_mask"],
            chunk_positions=batch["chunk_positions"],
            global_length_features=batch["global_length_features"],
        )
        assert outputs.logits.shape == (2, 13)
        assert outputs.binary_logits.shape == (2, 2)
        assert outputs.subtype_logits.shape == (2, 12)
        assert outputs.official_subcategory_logits.shape == (2, 39)
        assert torch.isfinite(outputs.logits).all()
        loss, details = longdoc_multitask_loss(outputs, batch)
        assert torch.isfinite(loss)
        assert details["total"] >= 0
        loss.backward()
        assert model.chunk_encoder.backbone.embedding.weight.grad is not None
        assert any(
            parameter.grad is not None
            for parameter in model.aggregator.parameters()
            if parameter.requires_grad
        )


def test_single_chunk_uses_exact_v341_hidden_and_heads():
    tokenizer = FakeTokenizer()
    collator = LongDocCollator(
        tokenizer,
        ChunkConfig(chunk_length=128, overlap=8, input_mode=PLAIN_MODE),
        child_code_to_index={"04_0403": 1},
    )
    batch = collator([{"sample_id": "short", "model_input_text": "简短贷款核验", "label_13": 4, "subcategory_code": "04_0403"}])
    model = LongDocMultiTaskClassifier(
        FakeBackbone(), route="B", projection_dim=32, fusion_hidden=24
    ).eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            chunk_mask=batch["chunk_mask"],
            chunk_positions=batch["chunk_positions"],
            global_length_features=batch["global_length_features"],
        )
        direct = model.logits_from_hidden(outputs.base_chunk_repr[:, 0]).joint_logits
    assert outputs.single_chunk_bypass.tolist() == [True]
    assert torch.equal(outputs.logits, direct)


def test_chunk_encoder_has_no_64_chunk_limit():
    model = LongDocMultiTaskClassifier(
        FakeBackbone(),
        route="B",
        projection_dim=32,
        fusion_hidden=24,
        encoder_micro_batch_size=7,
    ).eval()
    chunks = 65
    input_ids = torch.ones((1, chunks, 8), dtype=torch.long)
    mask = torch.ones_like(input_ids)
    chunk_mask = torch.ones((1, chunks), dtype=torch.bool)
    positions = torch.zeros((1, chunks, 6), dtype=torch.float32)
    positions[0, :, 0] = torch.arange(chunks)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=mask,
            chunk_mask=chunk_mask,
            chunk_positions=positions,
            global_length_features=torch.zeros((1, 4)),
        )
    assert outputs.logits.shape == (1, 13)


def test_fixed_logit_aggregators_ignore_padding():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(aggregate_window_logits(logits, mask, "max"), torch.tensor([[3.0, 2.0]]))
    assert torch.equal(
        aggregate_window_logits(logits, mask, "first"),
        torch.tensor([[1.0, 2.0]]),
    )
    assert torch.allclose(
        aggregate_window_logits(logits, mask, "top2_mean"),
        torch.tensor([[2.0, 1.5]]),
    )


def test_fixed_control_all_reuses_the_three_declared_pooling_modes():
    assert resolve_fixed_pooling_modes("all") == FIXED_CONTROL_MODES
    assert resolve_fixed_pooling_modes("max") == ("max",)
    assert resolve_fixed_pooling_modes(None) == ()


def test_longdoc_inference_reads_test_b_csv(tmp_path):
    path = tmp_path / "test_b.csv"
    path.write_text("id,transcript,label_13\n7,测试通话,4\n", encoding="utf-8")
    rows = read_rows(path)
    assert rows == [{"id": "7", "transcript": "测试通话", "label_13": "4", "sample_id": "7"}]


def test_window_supervision_uses_normal_negatives_and_exact_fraud_positives():
    tokenizer = FakeTokenizer()
    collator = LongDocCollator(
        tokenizer,
        ChunkConfig(chunk_length=48, overlap=8, input_mode=PLAIN_MODE),
        child_code_to_index={"04_0403": 1},
    )
    normal = _normal_row()
    fraud = _fraud_row()
    fraud["evidence_token_ranges"] = [[70, 78]]
    batch = collator([normal, fraud])
    normal_chunks = int(batch["chunk_mask"][0].sum())
    assert batch["window_risk_mask"][0, :normal_chunks].all()
    assert not batch["window_risk_target"][0, :normal_chunks].any()
    fraud_supervised = batch["window_risk_mask"][1]
    assert fraud_supervised.any()
    assert batch["window_risk_target"][1][fraud_supervised].eq(1).all()


def test_head_context_evidence_supervises_every_window_that_repeats_the_head():
    tokenizer = FakeTokenizer()
    collator = LongDocCollator(
        tokenizer,
        ChunkConfig(
            chunk_length=48,
            overlap=8,
            input_mode=HEAD_MODE,
            head_context_length=12,
        ),
        child_code_to_index={"04_0403": 1},
    )
    fraud = _fraud_row()
    fraud["evidence_token_ranges"] = [[0, 6]]
    batch = collator([fraud])
    count = int(batch["chunk_mask"][0].sum())
    assert count > 1
    assert batch["window_risk_mask"][0, :count].all()
    assert batch["window_risk_target"][0, :count].eq(1).all()


def test_long_exposure_sampler_reaches_21_percent_without_dropping_short_rows():
    rows = [
        {"sample_id": f"short-{index}"} for index in range(5880)
    ] + [
        {"sample_id": f"long-{index}", "longdoc_generated": True}
        for index in range(620)
    ]
    sampler = LongExposureSampler(rows, target_fraction=0.21, seed=202)
    indices = list(iter(sampler))
    assert len(indices) == len(sampler)
    assert set(range(5880)).issubset(indices)
    long_draws = sum(index >= 5880 for index in indices)
    assert long_draws == sampler.long_draws
    assert abs(long_draws / len(indices) - 0.21) < 0.0001


def test_paired_dataset_selects_exact_asr_fraction_without_duplicating_rows():
    rows = [
        {
            "sample_id": f"paired-{index}",
            "model_input_text": f"clean-{index}",
            "clean_model_input_text": f"clean-{index}",
            "asr_model_input_text": f"asr-{index}",
            "evidence_token_ranges": [[0, 2]],
        }
        for index in range(10)
    ]
    dataset = RowsDataset(rows, asr_view_fraction=0.5, seed=202)
    dataset.set_epoch(1)
    selected = [dataset[index] for index in range(len(dataset))]
    assert len(selected) == len(rows) == 10
    assert sum(row["input_view"] == "asr" for row in selected) == 5
    assert all(row["disable_window_evidence"] for row in selected if row["input_view"] == "asr")
    assert all(row["model_input_text"].startswith(row["input_view"]) for row in selected)
    assert all(row["model_input_text"].startswith("clean") for row in rows)


def test_asr_view_disables_unaligned_window_evidence_targets():
    row = _normal_row("asr-normal")
    row["input_view"] = "asr"
    row["disable_window_evidence"] = True
    collator = LongDocCollator(
        FakeTokenizer(),
        ChunkConfig(chunk_length=64, overlap=8, input_mode=PLAIN_MODE),
        child_code_to_index={"04_0403": 1},
    )
    batch = collator([row])
    assert not batch["window_risk_mask"].any()
    assert batch["metadata"][0]["input_view"] == "asr"
