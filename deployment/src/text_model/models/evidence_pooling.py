"""路线 A/B 共用的证据池化。"""

from __future__ import annotations

from typing import Any


EVIDENCE_COMPONENTS = ("attention", "top1", "top2", "last", "mean")


def normalize_evidence_components(values: Any = None) -> frozenset[str]:
    """规范化证据池化开关，供路线 A/B 的消融实验共用。"""

    if values is None:
        return frozenset(EVIDENCE_COMPONENTS)
    if isinstance(values, str):
        values = [part.strip() for part in values.split(",") if part.strip()]
    selected = frozenset(str(value).strip() for value in values)
    unknown = selected - set(EVIDENCE_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown evidence components: {sorted(unknown)}")
    if not selected:
        raise ValueError("at least one evidence component is required")
    return selected


def selected_evidence(value: Any, name: str, selected: frozenset[str]) -> Any:
    """保持融合层形状不变，用零向量关闭指定证据分支。"""

    return value if name in selected else value.new_zeros(value.shape)


def masked_softmax(logits: Any, mask: Any, dim: int = -1) -> Any:
    import torch

    if logits.shape != mask.shape:
        raise ValueError("logits and mask must have the same shape")
    masked = logits.float().masked_fill(~mask.bool(), torch.finfo(torch.float32).min)
    weights = masked.softmax(dim=dim)
    weights = weights * mask.float()
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


def masked_mean(values: Any, mask: Any) -> Any:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _gather_rows(values: Any, indices: Any) -> Any:
    """按行取块，支持一维与多维索引。

    ``indices`` 既可能是一维的 ``[batch]``（Top-1），也可能是二维的
    ``[batch, k]``（Top-k）。直接写 ``values[batch, indices]`` 时，一维的
    ``batch`` 会被广播到 ``indices`` 的形状，于是第 ``i`` 行的第 2 个索引会落到
    第 ``i+1`` 行上，从别的样本里取了块。这里显式把 ``batch`` 补成列向量再广播，
    保证每一行的索引都只在本行内取值。
    """

    import torch

    batch = torch.arange(values.shape[0], device=values.device)
    if indices.dim() == 1:
        return values[batch, indices]
    batch = batch.reshape([values.shape[0]] + [1] * (indices.dim() - 1))
    return values[batch, indices]


def evidence_pool(
    values: Any,
    chunk_mask: Any,
    risk_scores: Any,
    attention_weights: Any,
    *,
    last_threshold: float = 0.5,
    soft_last: bool = True,
    last_position_bias: float = 1.5,
) -> dict[str, Any]:
    """生成 attention、Top-1/2、最后高风险块和均值证据。"""

    import torch

    if values.ndim != 3:
        raise ValueError("values must have shape [batch, chunks, dim]")
    if chunk_mask.shape != values.shape[:2]:
        raise ValueError("chunk_mask shape mismatch")
    if risk_scores.shape != chunk_mask.shape:
        raise ValueError("risk_scores shape mismatch")
    if attention_weights.shape != chunk_mask.shape:
        raise ValueError("attention_weights shape mismatch")
    valid = chunk_mask.bool()
    scores = risk_scores.float().masked_fill(~valid, -torch.inf)
    top1_indices = scores.argmax(dim=-1)
    topk = min(2, values.shape[1])
    top_indices = scores.topk(k=topk, dim=-1).indices
    top1 = _gather_rows(values, top_indices[:, 0])
    top2 = _gather_rows(values, top_indices).mean(dim=1)
    attn = (values * attention_weights.unsqueeze(-1)).sum(dim=1)
    mean = masked_mean(values, valid)

    relative_end = torch.arange(
        values.shape[1], device=values.device, dtype=torch.float32
    )
    relative_end = relative_end / max(values.shape[1] - 1, 1)
    risk_logits = torch.logit(risk_scores.float().clamp(1e-5, 1 - 1e-5))
    if soft_last:
        soft_gate = torch.sigmoid((risk_scores.float() - last_threshold) * 12.0)
        last_logits = risk_logits + last_position_bias * relative_end.unsqueeze(0)
        last_weights = masked_softmax(last_logits, valid)
        last_weights = last_weights * soft_gate
        last_found = last_weights.sum(dim=-1).clamp(0.0, 1.0)
        last_repr = (values * last_weights.unsqueeze(-1)).sum(dim=1)
        hard_last_indices = torch.where(
            last_found > 1e-6,
            (last_weights * torch.arange(values.shape[1], device=values.device)).max(dim=-1).indices,
            torch.full_like(top1_indices, -1),
        )
    else:
        high = valid & (risk_scores.float() >= float(last_threshold))
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        last_indices = torch.where(high, positions, torch.full_like(positions, -1)).max(dim=-1).values
        found = last_indices >= 0
        safe_indices = last_indices.clamp_min(0)
        last_repr = _gather_rows(values, safe_indices) * found.unsqueeze(-1).float()
        last_found = found.float()
        hard_last_indices = last_indices
        last_weights = torch.zeros_like(risk_scores, dtype=torch.float32)
        if bool(found.any()):
            batch = torch.arange(values.shape[0], device=values.device)[found]
            last_weights[batch, safe_indices[found]] = 1.0

    return {
        "attention": attn,
        "attention_weights": attention_weights,
        "top1": top1,
        "top2": top2,
        "top1_indices": top_indices[:, 0],
        "top2_indices": top_indices,
        "last": last_repr,
        "last_weights": last_weights,
        "last_indices": hard_last_indices,
        "last_found": last_found,
        "mean": mean,
        "risk_scores": risk_scores,
    }
