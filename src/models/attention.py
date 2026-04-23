"""Attention models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


SUPPORTED_POSITION_ENCODINGS = frozenset({"sinusoidal", "rope"})


@dataclass(frozen=True)
class AttentionParams:
    """Attention-model parameters for ABS and RoPE models."""

    hidden_dim: int = 48
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.01
    dff_factor: int = 4
    init_range: float = 0.15
    max_len: int = 32
    pe_type: str = "rope"
    norm_first: bool = True
    use_bias_qkv: bool = False
    use_bias_out: bool = True
    use_bias_ffn: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, found {self.hidden_dim}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, found {self.n_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, found {self.n_layers}")
        if self.dff_factor <= 0:
            raise ValueError(f"dff_factor must be positive, found {self.dff_factor}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), found {self.dropout}")
        if self.init_range <= 0:
            raise ValueError(f"init_range must be positive, found {self.init_range}")
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, found {self.max_len}")
        if self.pe_type not in SUPPORTED_POSITION_ENCODINGS:
            raise ValueError(
                f"pe_type must be one of {sorted(SUPPORTED_POSITION_ENCODINGS)}, found {self.pe_type}"
            )
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
            )

        head_dim = self.hidden_dim // self.n_heads
        if self.pe_type == "sinusoidal" and self.hidden_dim % 2 != 0:
            raise ValueError(
                f"sinusoidal positional encoding requires even hidden_dim, found {self.hidden_dim}"
            )
        if self.pe_type == "rope" and head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires even head_dim, found {head_dim} "
                f"(hidden_dim={self.hidden_dim}, n_heads={self.n_heads})"
            )


class _FeedForward(nn.Module):
    """Two-layer GELU feed-forward network with dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=use_bias)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear2(x)


class _MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE on Q/K."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        use_bias_qkv: bool,
        use_bias_out: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias_qkv)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias_qkv)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias_qkv)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias_out)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        gain = 1.0 / math.sqrt(2)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=gain)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query = self._reshape_heads(self.q_proj(x), batch_size, seq_len)
        key = self._reshape_heads(self.k_proj(x), batch_size, seq_len)
        value = self._reshape_heads(self.v_proj(x), batch_size, seq_len)

        if rope_freqs is not None:
            freqs_cos, freqs_sin = rope_freqs
            query, key = _apply_rotary_emb(query, key, freqs_cos, freqs_sin)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(context)

    def _reshape_heads(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)


class _AttentionBlock(nn.Module):
    """PreNorm/PostNorm self-attention block with FFN."""

    def __init__(self, params: AttentionParams) -> None:
        super().__init__()
        self.norm_first = params.norm_first
        self.self_attn = _MultiHeadSelfAttention(
            hidden_dim=params.hidden_dim,
            n_heads=params.n_heads,
            dropout=params.dropout,
            use_bias_qkv=params.use_bias_qkv,
            use_bias_out=params.use_bias_out,
        )
        self.feed_forward = _FeedForward(
            input_dim=params.hidden_dim,
            hidden_dim=params.hidden_dim * params.dff_factor,
            dropout=params.dropout,
            use_bias=params.use_bias_ffn,
        )
        self.norm1 = nn.LayerNorm(params.hidden_dim)
        self.norm2 = nn.LayerNorm(params.hidden_dim)
        self.dropout = nn.Dropout(params.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        rope_freqs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout(self.self_attn(self.norm1(x), mask, rope_freqs=rope_freqs))
            x = x + self.dropout(self.feed_forward(self.norm2(x)))
            return x

        x = self.norm1(x + self.dropout(self.self_attn(x, mask, rope_freqs=rope_freqs)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class SimpleAttentionModel(nn.Module):
    """Attention model for ABS and RoPE experiments."""

    def __init__(self, vocab_size: int, output_dim: int, params: AttentionParams) -> None:
        super().__init__()
        self.pe_type = params.pe_type
        self.max_len = params.max_len

        self.embedding = nn.Embedding(vocab_size + 1, params.hidden_dim, padding_idx=vocab_size)
        self.attention_layers = nn.ModuleList(
            [_AttentionBlock(params) for _ in range(params.n_layers)]
        )
        self.final_norm = nn.LayerNorm(params.hidden_dim) if params.norm_first else None
        self.output_layer = nn.Linear(params.hidden_dim, output_dim, bias=True)

        if params.pe_type == "sinusoidal":
            self.register_buffer(
                "sinusoidal_embeddings",
                _build_sinusoidal_embeddings(params.max_len, params.hidden_dim),
                persistent=False,
            )
        else:
            head_dim = params.hidden_dim // params.n_heads
            rope_cos, rope_sin = _precompute_rope_frequencies(
                head_dim=head_dim,
                max_len=params.max_len,
                theta=params.rope_theta,
            )
            self.register_buffer("rope_cos", rope_cos, persistent=False)
            self.register_buffer("rope_sin", rope_sin, persistent=False)

        self._init_weights(params.init_range)

    def _init_weights(self, init_range: float) -> None:
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        with torch.no_grad():
            self.embedding.weight[self.embedding.padding_idx].zero_()
        nn.init.uniform_(self.output_layer.weight, -init_range, init_range)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        sequences: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, seq_len = sequences.shape
        device = sequences.device
        sequence_lengths = sequence_lengths.to(device)

        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_len {self.max_len}"
            )

        hidden_states = self.embedding(sequences)
        rope_freqs = None

        if self.pe_type == "sinusoidal":
            hidden_states = hidden_states + self.sinusoidal_embeddings[:seq_len].to(device).unsqueeze(0)
        else:
            rope_freqs = (
                self.rope_cos[:seq_len].to(device),
                self.rope_sin[:seq_len].to(device),
            )

        mask = torch.arange(seq_len, device=device)[None, :] < sequence_lengths[:, None]
        attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        for attention_layer in self.attention_layers:
            hidden_states = attention_layer(hidden_states, attn_mask, rope_freqs=rope_freqs)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        outputs = torch.sigmoid(self.output_layer(hidden_states)).transpose(1, 2)
        outputs = outputs * mask.unsqueeze(1).float()
        return outputs, hidden_states


def _build_sinusoidal_embeddings(max_len: int, hidden_dim: int) -> torch.Tensor:
    pe = torch.zeros(max_len, hidden_dim)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _precompute_rope_frequencies(
    head_dim: int,
    max_len: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_len, dtype=torch.float32)
    freq_exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inverse_frequencies = 1.0 / (theta ** freq_exponents)
    angles = torch.outer(positions, inverse_frequencies)
    return torch.cos(angles), torch.sin(angles)


def _apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape[-1] % 2 != 0 or key.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires even head dimensions for both query and key")

    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    seq_len_q = query.shape[-2]
    seq_len_k = key.shape[-2]
    cos_q = _reshape_rope_frequencies(freqs_cos[:seq_len_q], query_real)
    sin_q = _reshape_rope_frequencies(freqs_sin[:seq_len_q], query_real)
    cos_k = _reshape_rope_frequencies(freqs_cos[:seq_len_k], key_real)
    sin_k = _reshape_rope_frequencies(freqs_sin[:seq_len_k], key_real)

    rotated_query_real = query_real * cos_q - query_imag * sin_q
    rotated_query_imag = query_real * sin_q + query_imag * cos_q
    rotated_key_real = key_real * cos_k - key_imag * sin_k
    rotated_key_imag = key_real * sin_k + key_imag * cos_k

    rotated_query = torch.stack([rotated_query_real, rotated_query_imag], dim=-1).flatten(-2)
    rotated_key = torch.stack([rotated_key_real, rotated_key_imag], dim=-1).flatten(-2)
    return rotated_query.type_as(query), rotated_key.type_as(key)


def _reshape_rope_frequencies(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    shape = [1] * x.ndim
    shape[-2] = x.shape[-2]
    shape[-1] = x.shape[-1]
    return freqs.view(shape)
