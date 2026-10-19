"""V341 长文本路线 A/B 的统一多任务分类器。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import nn

from .gated_attention_aggregator import GatedAttentionAggregator
from .hierarchical_transformer_aggregator import HierarchicalTransformerAggregator
from .shared_chunk_encoder import SharedChunkEncoder


NUM_PARENT_CLASSES = 13
NUM_FRAUD_PARENT_CLASSES = 12
NUM_CHILD_CLASSES = 39


def _gather_first_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    indices = mask.to(torch.int64).argmax(dim=1)
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, indices]


class LongDocMultiTaskClassifier(nn.Module):
    """共享 V341 编码器和分类头，只替换整通级聚合器。

    单 chunk 样本直接使用 V341 的 pooler 表示和原分类头。模型处于 eval 模式时，
    该分支与原 V341-D1 数值路径一致。
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        route: str,
        projection_dim: int = 512,
        position_dim: int = 6,
        dropout: float = 0.1,
        fusion_hidden: int = 512,
        document_layers: int = 2,
        document_heads: int = 8,
        document_intermediate: int = 2048,
        encoder_micro_batch_size: int = 12,
        evidence_components: Sequence[str] | str | None = None,
    ) -> None:
        super().__init__()
        self.route = str(route)
        self.chunk_encoder = SharedChunkEncoder(
            backbone,
            projection_dim=projection_dim,
            position_dim=position_dim,
            dropout=dropout,
            encoder_micro_batch_size=encoder_micro_batch_size,
        )
        if self.route in {"B", "route_b", "route_b_gated_attention"}:
            self.aggregator = GatedAttentionAggregator(
                projection_dim,
                position_dim=position_dim,
                fusion_hidden=fusion_hidden,
                dropout=dropout,
                evidence_components=evidence_components,
            )
            self.route = "B"
        elif self.route in {"A", "route_a", "route_a_hierarchical_transformer"}:
            self.aggregator = HierarchicalTransformerAggregator(
                projection_dim,
                position_dim=position_dim,
                num_layers=document_layers,
                num_heads=document_heads,
                intermediate_size=document_intermediate,
                fusion_hidden=fusion_hidden,
                dropout=dropout,
                evidence_components=evidence_components,
            )
            self.route = "A"
        else:
            raise ValueError(f"unsupported longdoc route: {route}")

        hidden_size = self.chunk_encoder.hidden_size
        self.document_residual = nn.Linear(self.aggregator.output_dim, hidden_size)
        self.document_gate = nn.Linear(self.aggregator.output_dim, 1)
        nn.init.zeros_(self.document_gate.weight)
        nn.init.constant_(self.document_gate.bias, -4.0)
        self.dropout = nn.Dropout(dropout)

        # 名称和形状与 V341-D1 checkpoint 保持一致。
        self.binary_classifier = nn.Linear(hidden_size, 2)
        self.subtype_classifier = nn.Linear(hidden_size, NUM_FRAUD_PARENT_CLASSES)
        self.official_subcategory_classifier = nn.Linear(hidden_size, NUM_CHILD_CLASSES)
        self.v341_load_report: dict[str, Any] | None = None

    @property
    def backbone(self) -> nn.Module:
        return self.chunk_encoder.backbone

    @staticmethod
    def joint_logits(
        binary_logits: torch.Tensor, subtype_logits: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (binary_logits[:, :1], binary_logits[:, 1:2] + subtype_logits), dim=-1
        )

    def logits_from_hidden(self, hidden: torch.Tensor) -> SimpleNamespace:
        dropped = self.dropout(hidden)
        binary_logits = self.binary_classifier(dropped)
        subtype_logits = self.subtype_classifier(dropped)
        official_logits = self.official_subcategory_classifier(dropped)
        return SimpleNamespace(
            binary_logits=binary_logits,
            subtype_logits=subtype_logits,
            official_subcategory_logits=official_logits,
            joint_logits=self.joint_logits(binary_logits, subtype_logits),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        chunk_positions: torch.Tensor,
        global_length_features: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        *,
        hard_last: bool = False,
    ) -> Any:
        encoded = self.chunk_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            chunk_mask=chunk_mask,
            chunk_positions=chunk_positions,
            token_type_ids=token_type_ids,
        )
        evidence = self.aggregator(
            encoded["chunk_repr"],
            encoded["chunk_mask"],
            encoded["chunk_positions"],
            global_length_features,
            hard_last=hard_last,
        )

        attention_weights = evidence["attention_weights"].to(
            encoded["base_chunk_repr"].dtype
        )
        anchor_hidden = (
            encoded["base_chunk_repr"] * attention_weights.unsqueeze(-1)
        ).sum(dim=1)
        residual = self.document_residual(evidence["doc_repr"])
        residual_gate = torch.sigmoid(self.document_gate(evidence["doc_repr"]))
        long_hidden = anchor_hidden + residual_gate * residual

        single_chunk = chunk_mask.sum(dim=1) == 1
        baseline_hidden = _gather_first_valid(encoded["base_chunk_repr"], chunk_mask)
        document_hidden = torch.where(
            single_chunk.unsqueeze(-1), baseline_hidden, long_hidden
        )
        logits = self.logits_from_hidden(document_hidden)
        return SimpleNamespace(
            logits=logits.joint_logits,
            joint_logits=logits.joint_logits,
            binary_logits=logits.binary_logits,
            subtype_logits=logits.subtype_logits,
            official_subcategory_logits=logits.official_subcategory_logits,
            risk_logits=evidence["risk_logits"],
            risk_scores=evidence["risk_scores"],
            evidence=evidence,
            document_repr=document_hidden,
            chunk_repr=encoded["chunk_repr"],
            base_chunk_repr=encoded["base_chunk_repr"],
            single_chunk_bypass=single_chunk,
            document_residual_gate=residual_gate,
        )


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if all(isinstance(key, str) for key in payload):
            return payload
    raise ValueError("unsupported V341 checkpoint payload")


def load_v341_checkpoint(
    model: LongDocMultiTaskClassifier, checkpoint_path: str | Path
) -> dict[str, Any]:
    """加载 V341 backbone、binary、12类和39类头并校验关键权重。"""

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu")
    source = _checkpoint_state(payload)
    mapped: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    for raw_key, value in source.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("base_model."):
            key = "chunk_encoder.backbone." + key[len("base_model.") :]
        elif key.startswith("backbone."):
            key = "chunk_encoder.backbone." + key[len("backbone.") :]
        if key.startswith(
            (
                "chunk_encoder.backbone.",
                "binary_classifier.",
                "subtype_classifier.",
                "official_subcategory_classifier.",
            )
        ):
            mapped[key] = value
        else:
            ignored.append(str(raw_key))

    incompatible = model.load_state_dict(mapped, strict=False)
    loaded = set(mapped) - set(incompatible.unexpected_keys)
    required = {
        "binary_classifier.weight",
        "binary_classifier.bias",
        "subtype_classifier.weight",
        "subtype_classifier.bias",
        "official_subcategory_classifier.weight",
        "official_subcategory_classifier.bias",
    }
    missing_required = sorted(required - loaded)
    backbone_loaded = sum(key.startswith("chunk_encoder.backbone.") for key in loaded)
    if missing_required:
        raise ValueError(
            "V341 checkpoint is missing required classification heads: "
            + ", ".join(missing_required)
        )
    if backbone_loaded == 0:
        raise ValueError("V341 checkpoint did not load any backbone parameters")
    report = {
        "checkpoint": str(path),
        "loaded_parameters": len(loaded),
        "loaded_backbone_parameters": backbone_loaded,
        "ignored_parameters": sorted(ignored),
        "missing_new_parameters": sorted(incompatible.missing_keys),
        "unexpected_parameters": sorted(incompatible.unexpected_keys),
        "single_chunk_path": "exact_v341_pooler_and_heads",
    }
    model.v341_load_report = report
    return report


def build_longdoc_model(
    model_name_or_path: str,
    *,
    route: str,
    projection_dim: int = 512,
    dropout: float = 0.1,
    fusion_hidden: int = 512,
    document_layers: int = 2,
    document_heads: int = 8,
    document_intermediate: int = 2048,
    encoder_micro_batch_size: int = 12,
    evidence_components: Sequence[str] | str | None = None,
    v341_checkpoint: str | Path | None = None,
) -> LongDocMultiTaskClassifier:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
    backbone = AutoModel.from_pretrained(
        model_name_or_path, config=config, local_files_only=True
    )
    model = LongDocMultiTaskClassifier(
        backbone,
        route=route,
        projection_dim=projection_dim,
        dropout=dropout,
        fusion_hidden=fusion_hidden,
        document_layers=document_layers,
        document_heads=document_heads,
        document_intermediate=document_intermediate,
        encoder_micro_batch_size=encoder_micro_batch_size,
        evidence_components=evidence_components,
    )
    if v341_checkpoint is not None:
        load_v341_checkpoint(model, v341_checkpoint)
    return model
