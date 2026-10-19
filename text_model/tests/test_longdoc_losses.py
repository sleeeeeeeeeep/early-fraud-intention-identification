from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from text_model.training.longdoc_losses import mil_window_loss


def test_mil_window_loss_is_autocast_safe() -> None:
    risk_logits = torch.tensor([[0.2, -0.4]], requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, probability = mil_window_loss(
            risk_logits,
            torch.tensor([[True, True]]),
            torch.tensor([1]),
        )
    assert torch.isfinite(loss)
    assert torch.allclose(
        probability,
        1.0 - (1.0 - risk_logits.sigmoid()).prod(dim=-1),
    )
    loss.backward()
    assert torch.isfinite(risk_logits.grad).all()
