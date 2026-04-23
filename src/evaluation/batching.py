"""Token encoding and final-step selection for evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from data.vocabulary import token_to_index


def encode_tokens(tokens: Sequence[str]) -> torch.Tensor:
    """Encode one token sequence with the frozen vocabulary."""

    return torch.tensor([token_to_index(token) for token in tokens], dtype=torch.long)


def select_final_outputs(
    outputs: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """Select the final timestep from model outputs shaped (batch, output_dim, seq_len)."""

    batch_indices = torch.arange(sequence_lengths.size(0), device=outputs.device)
    final_indices = sequence_lengths - 1
    return outputs[batch_indices, :, final_indices]
