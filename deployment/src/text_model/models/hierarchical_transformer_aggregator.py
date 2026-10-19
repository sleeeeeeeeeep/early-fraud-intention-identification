"""路线 A：文档级层级 Transformer + 证据池化。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .evidence_pooling import (
    evidence_pool,
    masked_softmax,
    normalize_evidence_components,
    selected_evidence,
)
from .gated_attention_aggregator import WindowRiskHead


class HierarchicalTransformerAggregator(nn.Module):
    """在 chunk 表示之间建模关系，并保留稀疏证据残差通道。"""

    route_name = "route_a_hierarchical_transformer"

    def __init__(
        self,
        input_dim: int,
        *,
        position_dim: int = 6,
        num_layers: int = 2,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        fusion_hidden: int = 512,
        dropout: float = 0.1,
        last_threshold: float = 0.5,
        last_position_bias: float = 1.5,
        evidence_components: Sequence[str] | str | None = None,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        self.doc_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.normal_(self.doc_token, mean=0.0, std=0.02)
        self.position_projection = nn.Sequential(
            nn.Linear(position_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_score = nn.Linear(input_dim, 1)
        self.position_bias = nn.Linear(position_dim, 1)
        self.risk_head = WindowRiskHead(input_dim, dropout=dropout)
        self.last_threshold = float(last_threshold)
        self.last_position_bias = float(last_position_bias)
        self.evidence_components = normalize_evidence_components(
            evidence_components
        )
        fusion_input = input_dim * 6 + 4
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, fusion_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden * 2, fusion_hidden),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden),
            nn.Dropout(dropout),
        )
        self.output_dim = fusion_hidden

    def forward(
        self,
        values: torch.Tensor,
        chunk_mask: torch.Tensor,
        chunk_positions: torch.Tensor,
        global_length_features: torch.Tensor,
        *,
        hard_last: bool = False,
    ) -> dict[str, Any]:
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, chunks, dim]")
        valid = chunk_mask.bool()
        position_values = self.position_projection(chunk_positions)
        contextual_input = values + position_values
        doc = self.doc_token.expand(values.shape[0], -1, -1)
        sequence = torch.cat((doc, contextual_input), dim=1)
        key_padding_mask = torch.cat(
            (
                torch.zeros(
                    (values.shape[0], 1), dtype=torch.bool, device=values.device
                ),
                ~valid,
            ),
            dim=1,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        doc_repr = encoded[:, 0]
        contextual_chunks = encoded[:, 1:]
        attention_logits = self.pool_score(contextual_chunks).squeeze(-1)
        attention_logits = attention_logits + self.position_bias(
            chunk_positions
        ).squeeze(-1)
        attention_weights = masked_softmax(attention_logits, valid)
        risk_logits = self.risk_head(values)
        risk_scores = torch.sigmoid(risk_logits)
        evidence = evidence_pool(
            contextual_chunks,
            valid,
            risk_scores,
            attention_weights,
            last_threshold=self.last_threshold,
            soft_last=not hard_last,
            last_position_bias=self.last_position_bias,
        )
        fusion_input = torch.cat(
            (
                doc_repr,
                selected_evidence(
                    evidence["attention"], "attention", self.evidence_components
                ),
                selected_evidence(evidence["top1"], "top1", self.evidence_components),
                selected_evidence(evidence["top2"], "top2", self.evidence_components),
                selected_evidence(evidence["last"], "last", self.evidence_components),
                selected_evidence(evidence["mean"], "mean", self.evidence_components),
                global_length_features,
            ),
            dim=-1,
        )
        output = self.fusion(fusion_input)
        evidence.update(
            {
                "doc_token": doc_repr,
                "doc_repr": output,
                "risk_logits": risk_logits,
                "attention_logits": attention_logits,
                "route": self.route_name,
                "evidence_components": tuple(sorted(self.evidence_components)),
            }
        )
        return evidence
