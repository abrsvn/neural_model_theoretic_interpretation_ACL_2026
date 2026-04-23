"""Evaluation metrics."""

from __future__ import annotations

import torch


def compute_comprehension_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    truth_only: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute the Frank et al. comprehension score."""

    outputs_were_2d = outputs.dim() == 2
    targets_were_2d = targets.dim() == 2

    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    if outputs.dim() == 2:
        outputs = outputs.unsqueeze(1)
    if targets.dim() == 2:
        targets = targets.unsqueeze(1)

    if outputs.size(0) != targets.size(0):
        raise ValueError("Batch sizes of outputs and targets must match")
    if outputs.size(2) != targets.size(2):
        raise ValueError("Vector dimensions of outputs and targets must match")
    if outputs.size(2) == 0:
        raise ValueError("Vector dimensions cannot be zero")

    current_outputs = outputs
    current_targets = targets
    if truth_only and outputs.size(2) == 300:
        current_outputs = outputs[..., :150]
        current_targets = targets[..., :150]

    if current_outputs.size(1) == 1 and current_targets.size(1) > 1:
        current_outputs = current_outputs.expand(-1, current_targets.size(1), -1)
    if current_outputs.size(1) > 1 and current_targets.size(1) == 1:
        current_targets = current_targets.expand(-1, current_outputs.size(1), -1)
    if current_outputs.size(1) != current_targets.size(1):
        raise ValueError("Number of output and target vectors must match after broadcasting")

    dot_products = torch.sum(current_targets * current_outputs, dim=2)
    probs_az_denominator = current_outputs.sum(dim=2)
    probs_az = dot_products / probs_az_denominator

    norm_outputs_sq = torch.sum(current_outputs**2, dim=2)
    norm_targets_sq = torch.sum(current_targets**2, dim=2)
    dot_products_sq = dot_products**2
    are_outputs_nonzero = norm_outputs_sq > epsilon
    are_targets_nonzero = norm_targets_sq > epsilon
    collinearity_mask = torch.isclose(
        dot_products_sq,
        norm_outputs_sq * norm_targets_sq,
        rtol=epsilon,
        atol=epsilon**2,
    )
    identity_mask = collinearity_mask & are_outputs_nonzero & are_targets_nonzero
    probs_az[identity_mask] = 1.0

    probs_a = current_targets.mean(dim=2)
    neg_probs_a = 1.0 - probs_a
    probs_diff = probs_az - probs_a
    pos_mask = probs_az >= probs_a
    neg_mask = ~pos_mask

    scores = torch.zeros_like(probs_diff)
    scores[pos_mask] = probs_diff[pos_mask] / neg_probs_a[pos_mask]
    scores[neg_mask] = probs_diff[neg_mask] / probs_a[neg_mask]

    if outputs_were_2d and targets_were_2d:
        return scores.squeeze(1)
    return scores
