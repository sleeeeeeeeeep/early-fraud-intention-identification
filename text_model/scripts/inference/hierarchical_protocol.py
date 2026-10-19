"""Small, testable pieces shared by explicit normal/fraud hierarchy runs."""

from __future__ import annotations

from typing import Any


def h2_binary_mass_consistency_loss(flat_logits: Any, hierarchical_logits: Any) -> Any:
    """Align only the normal/fraud mass of H2's auxiliary flat head.

    The flat head remains an auxiliary representation learner. Inference always
    uses ``hierarchical_logits``; this loss only requires its normal probability
    and aggregate fraud probability to agree with the flat head.
    """
    import torch
    import torch.nn.functional as F

    flat_log_probs = F.log_softmax(flat_logits, dim=-1)
    hierarchical_log_probs = F.log_softmax(hierarchical_logits, dim=-1)
    flat_binary_log_probs = torch.stack(
        [flat_log_probs[:, 0], torch.logsumexp(flat_log_probs[:, 1:], dim=-1)],
        dim=-1,
    )
    hierarchical_binary_log_probs = torch.stack(
        [
            hierarchical_log_probs[:, 0],
            torch.logsumexp(hierarchical_log_probs[:, 1:], dim=-1),
        ],
        dim=-1,
    )
    flat_binary_probs = flat_binary_log_probs.exp()
    hierarchical_binary_probs = hierarchical_binary_log_probs.exp()
    return 0.5 * (
        F.kl_div(flat_binary_log_probs, hierarchical_binary_probs, reduction="batchmean")
        + F.kl_div(
            hierarchical_binary_log_probs,
            flat_binary_probs,
            reduction="batchmean",
        )
    )


def fpr_constrained_checkpoint_decision(
    metrics: dict[str, Any],
    max_normal_fpr: float,
) -> dict[str, Any]:
    """Return a deterministic FPR-first checkpoint ranking decision.

    A passing checkpoint is ranked by (Acc, Macro-F1, fraud recall). When none
    passes, the diagnostic fallback ranks by lower FPR first, then the same
    quality metrics. The caller records this fallback as non-promotable.
    """
    normal_fpr = float(metrics["binary"]["normal_fpr"])
    label13_accuracy = float(metrics["label_13_accuracy"])
    macro_f1 = float(metrics["observed_label_macro_f1"])
    fraud_recall = float(metrics["binary"]["recall"])
    passes = normal_fpr <= float(max_normal_fpr) + 1e-12
    ranking_key = (
        1.0 if passes else 0.0,
        label13_accuracy if passes else -normal_fpr,
        macro_f1 if passes else label13_accuracy,
        fraud_recall if passes else macro_f1,
        0.0 if passes else fraud_recall,
    )
    return {
        "normal_fpr_max": float(max_normal_fpr),
        "normal_fpr": normal_fpr,
        "normal_fpr_pass": passes,
        "label_13_accuracy": label13_accuracy,
        "macro_f1": macro_f1,
        "fraud_recall": fraud_recall,
        "ranking_key": list(ranking_key),
    }
