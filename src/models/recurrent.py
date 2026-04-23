"""Recurrent models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


SUPPORTED_INIT_METHODS = frozenset({"uniform", "xavier", "kaiming"})
SUPPORTED_SRN_ACTIVATIONS = frozenset({"sigmoid", "tanh", "relu", "gelu"})


@dataclass(frozen=True)
class RecurrentParams:
    """Recurrent-model parameters for SRN, GRU, and LSTM models."""

    hidden_dim: int
    n_layers: int = 1
    dropout: float = 0.01
    init_method: str = "uniform"
    init_range: float = 0.15
    activation: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, found {self.hidden_dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, found {self.n_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), found {self.dropout}")
        if self.init_method not in SUPPORTED_INIT_METHODS:
            raise ValueError(
                f"init_method must be one of {sorted(SUPPORTED_INIT_METHODS)}, found {self.init_method}"
            )
        if self.activation not in SUPPORTED_SRN_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(SUPPORTED_SRN_ACTIVATIONS)}, found {self.activation}"
            )


class SimpleRNNCell(nn.Module):
    """Single SRN cell with embedding-as-input-projection behavior."""

    def __init__(self, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.recurrent_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

        activation_map = {
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "relu": torch.relu,
            "gelu": torch.nn.functional.gelu,
        }
        self.activation_fn = activation_map[activation]

    def forward(
        self,
        input_t: torch.Tensor,
        hidden_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_prev is None:
            return self.activation_fn(input_t)
        return self.activation_fn(input_t + self.recurrent_layer(hidden_prev))


class SimpleGRUCell(nn.Module):
    """Single GRU cell with explicit reset, update, and candidate projections."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.reset_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.candidate = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

    def forward(
        self,
        input_t: torch.Tensor,
        hidden_prev: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([input_t, hidden_prev], dim=1)
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        update_gate = torch.sigmoid(self.update_gate(combined))

        reset_hidden = reset_gate * hidden_prev
        candidate_input = torch.cat([input_t, reset_hidden], dim=1)
        candidate_hidden = torch.tanh(self.candidate(candidate_input))
        return (1 - update_gate) * hidden_prev + update_gate * candidate_hidden


class SimpleLSTMCell(nn.Module):
    """Single LSTM cell with combined gate projection."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gates = nn.Linear(hidden_dim * 2, hidden_dim * 4, bias=True)

    def forward(
        self,
        input_t: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_prev, cell_prev = states
        combined = torch.cat([input_t, hidden_prev], dim=1)
        gate_output = self.gates(combined)
        input_gate, forget_gate, cell_gate, output_gate = gate_output.chunk(4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        cell_next = forget_gate * cell_prev + input_gate * cell_gate
        hidden_next = output_gate * torch.tanh(cell_next)
        return hidden_next, cell_next


class _BaseRecurrentModel(nn.Module):
    """Shared recurrent implementation for the SRN, GRU, and LSTM models."""

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        params: RecurrentParams,
        cell_type: str,
    ) -> None:
        super().__init__()
        self.hidden_dim = params.hidden_dim
        self.params = params
        self.cell_type = cell_type

        self.embedding = nn.Embedding(vocab_size + 1, self.hidden_dim, padding_idx=vocab_size)
        self.cells = nn.ModuleList(
            [self._create_cell(cell_type, params) for _ in range(params.n_layers)]
        )
        self.dropout = nn.Dropout(params.dropout) if params.dropout > 0 else nn.Identity()
        self.inter_layer_dropout = (
            nn.Dropout(params.dropout)
            if params.n_layers > 1 and params.dropout > 0
            else None
        )
        self.output_layer = nn.Linear(self.hidden_dim, output_dim, bias=True)

        self._init_weights()

    def _create_cell(self, cell_type: str, params: RecurrentParams) -> nn.Module:
        if cell_type == "srn":
            return SimpleRNNCell(params.hidden_dim, params.activation)
        if cell_type == "gru":
            return SimpleGRUCell(params.hidden_dim)
        if cell_type == "lstm":
            return SimpleLSTMCell(params.hidden_dim)
        raise ValueError(f"Unsupported recurrent cell type: {cell_type}")

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -self.params.init_range, self.params.init_range)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].zero_()

        if self.params.init_method == "uniform":
            self._init_uniform_weights()
        elif self.params.init_method == "xavier":
            self._init_xavier_weights()
        elif self.params.init_method == "kaiming":
            self._init_kaiming_weights()

        self._init_lstm_forget_gate_bias()

    def _init_uniform_weights(self) -> None:
        nn.init.uniform_(self.output_layer.weight, -self.params.init_range, self.params.init_range)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

        for cell in self.cells:
            for module in cell.modules():
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, -self.params.init_range, self.params.init_range)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _init_xavier_weights(self) -> None:
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

        for cell in self.cells:
            for module in cell.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _init_kaiming_weights(self) -> None:
        nn.init.kaiming_uniform_(self.output_layer.weight, a=math.sqrt(5))
        if self.output_layer.bias is not None:
            fan_in = self.output_layer.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.output_layer.bias, -bound, bound)

        for cell in self.cells:
            for module in cell.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in = module.in_features
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(module.bias, -bound, bound)

    def _init_lstm_forget_gate_bias(self) -> None:
        if self.cell_type != "lstm":
            return
        for cell in self.cells:
            forget_start = self.hidden_dim
            forget_end = 2 * self.hidden_dim
            cell.gates.bias.data[forget_start:forget_end].fill_(1.0)

    def forward(
        self,
        sequences: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_seq_len = sequences.shape
        device = sequences.device
        sequence_lengths = sequence_lengths.to(device)

        mask = torch.arange(max_seq_len, device=device)[None, :] < sequence_lengths[:, None]
        layer_input = self.dropout(self.embedding(sequences))

        for layer_index, cell in enumerate(self.cells):
            if self.cell_type == "srn":
                prev_hidden = None
            else:
                prev_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
            if self.cell_type == "lstm":
                prev_cell = torch.zeros(batch_size, self.hidden_dim, device=device)

            layer_output = torch.zeros(
                batch_size,
                max_seq_len,
                self.hidden_dim,
                dtype=layer_input.dtype,
                device=device,
            )

            for timestep in range(max_seq_len):
                active_mask = mask[:, timestep]
                input_t = layer_input[:, timestep]

                if self.cell_type == "srn":
                    hidden = cell(input_t, prev_hidden)
                elif self.cell_type == "gru":
                    hidden = cell(input_t, prev_hidden)
                else:
                    hidden, cell_next = cell(input_t, (prev_hidden, prev_cell))

                active_mask_expanded = active_mask.unsqueeze(1).float()
                if self.cell_type != "srn" or prev_hidden is not None:
                    hidden = hidden * active_mask_expanded + prev_hidden * (1 - active_mask_expanded)
                if self.cell_type == "lstm":
                    cell_next = (
                        cell_next * active_mask_expanded
                        + prev_cell * (1 - active_mask_expanded)
                    )
                    prev_cell = cell_next

                prev_hidden = hidden
                layer_output[:, timestep, :] = hidden

            if layer_index < len(self.cells) - 1 and self.inter_layer_dropout is not None:
                layer_output = self.inter_layer_dropout(layer_output)
            layer_input = layer_output

        hidden_states = layer_output
        output_logits = self.output_layer(hidden_states)
        outputs = torch.sigmoid(output_logits.transpose(1, 2))

        outputs = outputs * mask.unsqueeze(1).float()
        hidden_states = hidden_states * mask.unsqueeze(2).float()
        return outputs, hidden_states


class SimpleRN(_BaseRecurrentModel):
    """SRN model."""

    def __init__(self, vocab_size: int, output_dim: int, params: RecurrentParams) -> None:
        super().__init__(vocab_size, output_dim, params, cell_type="srn")


class SimpleGRU(_BaseRecurrentModel):
    """GRU model."""

    def __init__(self, vocab_size: int, output_dim: int, params: RecurrentParams) -> None:
        super().__init__(vocab_size, output_dim, params, cell_type="gru")


class SimpleLSTM(_BaseRecurrentModel):
    """LSTM model."""

    def __init__(self, vocab_size: int, output_dim: int, params: RecurrentParams) -> None:
        super().__init__(vocab_size, output_dim, params, cell_type="lstm")
