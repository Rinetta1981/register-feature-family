import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from register_feature_family.sequence_encoding import (
    IGNORE_INDEX,
    MAX_SEQUENCE_LENGTH,
)
from register_feature_family.tokenizer import MODEL_VOCABULARY


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the synthetic decoder-only transformer."""

    vocab_size: int = len(MODEL_VOCABULARY)
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_mlp: int = 512

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")

        if self.max_sequence_length < 1:
            raise ValueError(
                "max_sequence_length must be positive"
            )

        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")

        if self.n_heads < 1:
            raise ValueError("n_heads must be positive")

        if self.d_model < 1:
            raise ValueError("d_model must be positive")

        if self.d_mlp < 1:
            raise ValueError("d_mlp must be positive")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads"
            )

    @property
    def head_dimension(self) -> int:
        """Return the dimension of one attention head."""

        return self.d_model // self.n_heads


class CausalSelfAttention(nn.Module):
    """Explicit multi-head causal self-attention."""

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dimension = config.head_dimension
        self.d_model = config.d_model

        self.q_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=True,
        )
        self.k_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=True,
        )
        self.v_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=True,
        )
        self.out_proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=True,
        )

    def _split_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Convert B x T x D to B x H x T x Dh."""

        batch_size, sequence_length, _ = tensor.shape

        return (
            tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dimension,
            )
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Convert B x H x T x Dh to B x T x D."""

        batch_size, _, sequence_length, _ = tensor.shape

        return (
            tensor.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.d_model,
            )
        )

    def forward_with_pattern(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output and attention probabilities."""

        if x.ndim != 3:
            raise ValueError(
                "attention input must have shape "
                "batch x sequence x model"
            )

        batch_size, sequence_length, _ = x.shape

        query_projection = cast(
            torch.Tensor,
            self.q_proj(x),
        )
        key_projection = cast(
            torch.Tensor,
            self.k_proj(x),
        )
        value_projection = cast(
            torch.Tensor,
            self.v_proj(x),
        )

        queries = self._split_heads(
            query_projection
        )
        keys = self._split_heads(
            key_projection
        )
        values = self._split_heads(
            value_projection
        )

        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self.head_dimension
        )

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        minimum_value = torch.finfo(
            scores.dtype
        ).min

        scores = scores.masked_fill(
            causal_mask,
            minimum_value,
        )

        if attention_mask is not None:
            expected_shape = (
                batch_size,
                sequence_length,
            )

            if attention_mask.shape != expected_shape:
                raise ValueError(
                    "attention_mask must have shape "
                    "batch x sequence"
                )

            valid_tokens = attention_mask.to(
                dtype=torch.bool
            )

            key_padding_mask = (
                ~valid_tokens[:, None, None, :]
            )

            scores = scores.masked_fill(
                key_padding_mask,
                minimum_value,
            )

        attention_pattern = torch.softmax(
            scores,
            dim=-1,
        )

        attended_values = torch.matmul(
            attention_pattern,
            values,
        )

        merged = self._merge_heads(
            attended_values
        )

        output = cast(
            torch.Tensor,
            self.out_proj(merged),
        )

        return output, attention_pattern

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the causal self-attention output."""

        output, _ = self.forward_with_pattern(
            x,
            attention_mask=attention_mask,
        )

        return output


class FeedForward(nn.Module):
    """Explicit transformer MLP."""

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.fc_in = nn.Linear(
            config.d_model,
            config.d_mlp,
            bias=True,
        )
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(
            config.d_mlp,
            config.d_model,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the transformer MLP."""

        hidden = cast(
            torch.Tensor,
            self.fc_in(x),
        )

        activated = cast(
            torch.Tensor,
            self.activation(hidden),
        )

        return cast(
            torch.Tensor,
            self.fc_out(activated),
        )


class TransformerBlock(nn.Module):
    """One pre-LayerNorm decoder block."""

    def __init__(
        self,
        config: TransformerConfig,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.d_model
        )
        self.attention = CausalSelfAttention(
            config
        )

        self.ln2 = nn.LayerNorm(
            config.d_model
        )
        self.mlp = FeedForward(
            config
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention and MLP residual updates."""

        attention_input = cast(
            torch.Tensor,
            self.ln1(x),
        )

        attention_output = self.attention(
            attention_input,
            attention_mask=attention_mask,
        )

        x = x + attention_output

        mlp_input = cast(
            torch.Tensor,
            self.ln2(x),
        )
        mlp_output = self.mlp(mlp_input)

        return x + mlp_output


class DecoderOnlyTransformer(nn.Module):
    """Small decoder-only transformer for the synthetic experiment
