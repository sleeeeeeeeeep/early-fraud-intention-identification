"""路线 B：门控 Attention + Top-k/Last-risk 证据聚合。"""

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


class WindowRiskHead(nn.Module):
    """窗口风险评分；只表示局部可执行风险证据，不等同于整通诈骗概率。"""

    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(64, input_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class GatedAttentionAggregator(nn.Module):
    """不做完整块间交互，主要学习哪些窗口是证据。"""

    route_name = "route_b_gated_attention"

    def __init__(
        self,
        input_dim: int,
        *,
        position_dim: int = 6,
        fusion_hidden: int = 512,
        dropout: float = 0.1,
        last_threshold: float = 0.5,
        last_position_bias: float = 1.5,
        evidence_components: Sequence[str] | str | None = None,
    ) -> None:
        super().__init__()
        gate_dim = max(128, input_dim // 2)
        self.value_projection = nn.Linear(input_dim, gate_dim)
        self.gate_projection = nn.Linear(input_dim, gate_dim)
        self.attention_score = nn.Linear(gate_dim, 1)
        self.position_bias = nn.Linear(position_dim, 1)
        self.risk_head = WindowRiskHead(input_dim, dropout=dropout)
        self.last_threshold = float(last_threshold)
        self.last_position_bias = float(last_position_bias)
        self.evidence_components = normalize_evidence_components(
            evidence_components
        )
        fusion_input = input_dim * 5 + 4
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
        if chunk_mask.shape != values.shape[:2]:
            raise ValueError("chunk_mask shape mismatch")
        if chunk_positions.shape[:2] != values.shape[:2]:
            raise ValueError("chunk_positions shape mismatch")
        gated = torch.tanh(self.value_projection(values)) * torch.sigmoid(
            self.gate_projection(values)
        )
        raw_scores = self.attention_score(gated).squeeze(-1)
        raw_scores = raw_scores + self.position_bias(chunk_positions).squeeze(-1)
        attention_weights = masked_softmax(raw_scores, chunk_mask.bool())
        risk_logits = self.risk_head(values)
        risk_scores = torch.sigmoid(risk_logits)
        evidence = evidence_pool(
            values,
            chunk_mask,
            risk_scores,
            attention_weights,
            last_threshold=self.last_threshold,
            soft_last=not hard_last,
            last_position_bias=self.last_position_bias,
        )
        fusion_input = torch.cat(
            (
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
                "doc_repr": output,
                "risk_logits": risk_logits,
                "attention_logits": raw_scores,
                "route": self.route_name,
                "evidence_components": tuple(sorted(self.evidence_components)),
            }
        )
        return evidence
