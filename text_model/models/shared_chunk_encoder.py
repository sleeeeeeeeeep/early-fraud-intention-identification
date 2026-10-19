"""路线 A/B 共用的窗口编码器。

窗口编码采用微批执行，保证整通文本的全部窗口都能进入文档聚合器。位置编码
使用连续数值特征，窗口数量没有固定上限。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SharedChunkEncoder(nn.Module):
    """共享 V341 backbone 编码各 chunk，并返回原始与投影表示。"""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int | None = None,
        projection_dim: int = 512,
        position_dim: int = 6,
        dropout: float = 0.1,
        encoder_micro_batch_size: int = 12,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, "config", None)
        hidden_size = hidden_size or int(getattr(config, "hidden_size", 0))
        if hidden_size <= 0:
            raise ValueError("hidden_size is required when backbone.config is absent")
        if encoder_micro_batch_size < 1:
            raise ValueError("encoder_micro_batch_size must be positive")
        self.hidden_size = int(hidden_size)
        self.projection_dim = int(projection_dim)
        self.encoder_micro_batch_size = int(encoder_micro_batch_size)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.projection_dim),
            nn.GELU(),
            nn.LayerNorm(self.projection_dim),
            nn.Dropout(dropout),
        )
        self.position_mlp = nn.Sequential(
            nn.Linear(position_dim, self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )
        self.output_dim = self.projection_dim

    @staticmethod
    def _pool(encoded: Any) -> torch.Tensor:
        pooled = getattr(encoded, "pooler_output", None)
        if pooled is not None:
            return pooled
        last_hidden = getattr(encoded, "last_hidden_state", None)
        if last_hidden is None:
            raise ValueError("backbone output has neither pooler_output nor last_hidden_state")
        return last_hidden[:, 0]

    def _encode_valid(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, input_ids.shape[0], self.encoder_micro_batch_size):
            end = min(input_ids.shape[0], start + self.encoder_micro_batch_size)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids[start:end],
                "attention_mask": attention_mask[start:end],
                "return_dict": True,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids[start:end]
            try:
                encoded = self.backbone(**kwargs)
            except TypeError:
                kwargs.pop("token_type_ids", None)
                encoded = self.backbone(**kwargs)
            outputs.append(self._pool(encoded))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        chunk_positions: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, chunks, length]")
        batch_size, num_chunks, length = input_ids.shape
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape mismatch")
        if chunk_mask.shape != (batch_size, num_chunks):
            raise ValueError("chunk_mask shape mismatch")
        if chunk_positions.shape[:2] != (batch_size, num_chunks):
            raise ValueError("chunk_positions shape mismatch")

        valid = chunk_mask.bool().reshape(-1)
        if not bool(valid.any()):
            raise ValueError("batch contains no valid chunks")
        flat_ids = input_ids.reshape(batch_size * num_chunks, length)[valid]
        flat_mask = attention_mask.reshape(batch_size * num_chunks, length)[valid]
        flat_types = None
        if token_type_ids is not None:
            flat_types = token_type_ids.reshape(batch_size * num_chunks, length)[valid]
        pooled_valid = self._encode_valid(flat_ids, flat_mask, flat_types)

        flat_base = pooled_valid.new_zeros(
            (batch_size * num_chunks, self.hidden_size)
        )
        flat_base[valid] = pooled_valid
        base_values = flat_base.reshape(batch_size, num_chunks, self.hidden_size)

        projected_valid = self.projection(pooled_valid)
        flat_projected = projected_valid.new_zeros(
            (batch_size * num_chunks, self.projection_dim)
        )
        flat_projected[valid] = projected_valid
        values = flat_projected.reshape(batch_size, num_chunks, self.projection_dim)
        values = values + self.position_mlp(chunk_positions)
        values = values * chunk_mask.unsqueeze(-1).to(values.dtype)
        base_values = base_values * chunk_mask.unsqueeze(-1).to(base_values.dtype)
        return {
            "base_chunk_repr": base_values,
            "chunk_repr": values,
            "chunk_mask": chunk_mask,
            "chunk_positions": chunk_positions,
        }
