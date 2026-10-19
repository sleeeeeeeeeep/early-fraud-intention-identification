"""路线 A/B 共用的文档级多任务与 MIL 损失。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(values.device).float()
    return (values.float() * weights).sum() / weights.sum().clamp_min(1e-8)


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("logits and soft targets must have the same shape")
    log_probs = logits.float().log_softmax(dim=-1)
    factors = (
        class_weights.to(logits.device).float()
        if class_weights is not None
        else torch.ones(logits.shape[-1], device=logits.device)
    )
    per_row = -(targets.float() * log_probs * factors.unsqueeze(0)).sum(dim=-1)
    return weighted_mean(per_row, weights)


def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_row = F.cross_entropy(
        logits,
        labels,
        weight=class_weights.to(logits.device).float() if class_weights is not None else None,
        reduction="none",
    )
    return weighted_mean(per_row, weights)


def mil_window_loss(
    risk_logits: torch.Tensor,
    chunk_mask: torch.Tensor,
    binary_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """整通 MIL 监督，不把诈骗 Gold 复制给所有窗口。"""

    valid = chunk_mask.bool()
    masked_log_no_risk = F.logsigmoid(-risk_logits.float()).masked_fill(~valid, 0.0)
    p_call = -torch.expm1(masked_log_no_risk.sum(dim=-1))
    call_logits = torch.logit(p_call.clamp(1e-5, 1.0 - 1e-5))
    call_loss = F.binary_cross_entropy_with_logits(
        call_logits, binary_target.float(), reduction="none"
    ).mean()
    normal = binary_target == 0
    if bool(normal.any()):
        normal_logits = risk_logits[normal][valid[normal]]
        normal_window = F.binary_cross_entropy_with_logits(
            normal_logits.float(), torch.zeros_like(normal_logits.float())
        )
    else:
        normal_window = risk_logits.sum() * 0.0
    return call_loss + 0.5 * normal_window, p_call.detach()


def trusted_window_evidence_loss(
    risk_logits: torch.Tensor,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    """只在正常窗口或可精确定位的风险证据窗口上计算监督。"""

    mask = supervision_mask.bool()
    if not bool(mask.any()):
        return risk_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        risk_logits.float()[mask], targets.float()[mask]
    )


def longdoc_multitask_loss(
    outputs: Any,
    batch: Mapping[str, torch.Tensor],
    *,
    parent_class_weights: torch.Tensor | None = None,
    child_class_weights: torch.Tensor | None = None,
    loss_weights: Mapping[str, float] | None = None,
    include_mil: bool = True,
    mil_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    factors = {
        "parent13": 1.0,
        "binary": 0.1,
        "fraud_subtype12": 1.0,
        "child39": 0.2,
        "mil": mil_weight,
        "window_evidence": 0.1,
    }
    if loss_weights:
        unknown = set(loss_weights) - set(factors)
        if unknown:
            raise ValueError(f"unknown longdoc loss weights: {sorted(unknown)}")
        factors.update({key: float(value) for key, value in loss_weights.items()})

    parent_weight = batch["parent_weight"].float()
    parent_loss = soft_cross_entropy(
        outputs.joint_logits,
        batch["parent_target"],
        parent_weight,
        parent_class_weights,
    )
    binary_loss = cross_entropy(
        outputs.binary_logits,
        batch["binary_target"],
        parent_weight,
    )
    fraud_mask = batch["subtype_label"] >= 0
    if bool(fraud_mask.any()):
        subtype_loss = cross_entropy(
            outputs.subtype_logits[fraud_mask],
            batch["subtype_label"][fraud_mask],
            parent_weight[fraud_mask],
        )
        child_loss = soft_cross_entropy(
            outputs.official_subcategory_logits[fraud_mask],
            batch["child_target"][fraud_mask],
            batch["child_weight"][fraud_mask],
            child_class_weights,
        )
    else:
        subtype_loss = outputs.subtype_logits.sum() * 0.0
        child_loss = outputs.official_subcategory_logits.sum() * 0.0

    if include_mil and factors["mil"] > 0:
        mil_loss, p_call = mil_window_loss(
            outputs.risk_logits,
            batch["chunk_mask"],
            batch["binary_target"],
        )
    else:
        mil_loss = outputs.risk_logits.sum() * 0.0
        p_call = torch.zeros_like(batch["binary_target"], dtype=torch.float32)

    evidence_loss = trusted_window_evidence_loss(
        outputs.risk_logits,
        batch.get("window_risk_target", torch.zeros_like(outputs.risk_logits)),
        batch.get(
            "window_risk_mask",
            torch.zeros_like(outputs.risk_logits, dtype=torch.bool),
        ),
    )

    total = (
        factors["parent13"] * parent_loss
        + factors["binary"] * binary_loss
        + factors["fraud_subtype12"] * subtype_loss
        + factors["child39"] * child_loss
        + factors["mil"] * mil_loss
        + factors["window_evidence"] * evidence_loss
    )
    return total, {
        "parent13": float(parent_loss.detach().cpu()),
        "binary": float(binary_loss.detach().cpu()),
        "fraud_subtype12": float(subtype_loss.detach().cpu()),
        "child39": float(child_loss.detach().cpu()),
        "mil": float(mil_loss.detach().cpu()),
        "window_evidence": float(evidence_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
        "mil_call_probability_mean": float(p_call.float().mean().cpu()),
    }
