"""无需训练的窗口 logit 聚合对照。"""

from __future__ import annotations

import torch


SUPPORTED_FIXED_POOLING = {"first", "max", "top2_mean", "logsumexp"}


def aggregate_window_logits(
    window_logits: torch.Tensor,
    chunk_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """把 ``[batch, chunks, classes]`` 聚合为整通分类分数。"""

    if mode not in SUPPORTED_FIXED_POOLING:
        raise ValueError(f"unsupported fixed pooling mode: {mode}")
    if window_logits.ndim != 3:
        raise ValueError("window_logits must have shape [batch, chunks, classes]")
    if chunk_mask.shape != window_logits.shape[:2]:
        raise ValueError("chunk_mask shape mismatch")
    valid = chunk_mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("every document must contain at least one valid chunk")
    masked = window_logits.float().masked_fill(~valid.unsqueeze(-1), -torch.inf)
    if mode == "first":
        first_indices = valid.to(torch.int64).argmax(dim=1)
        batch = torch.arange(window_logits.shape[0], device=window_logits.device)
        return window_logits.float()[batch, first_indices]
    if mode == "max":
        return masked.max(dim=1).values
    if mode == "logsumexp":
        counts = valid.sum(dim=1).clamp_min(1).float()
        return torch.logsumexp(masked, dim=1) - counts.log().unsqueeze(-1)

    topk = min(2, window_logits.shape[1])
    values = masked.topk(k=topk, dim=1).values
    finite = torch.isfinite(values)
    return values.masked_fill(~finite, 0.0).sum(dim=1) / finite.sum(
        dim=1
    ).clamp_min(1)
