import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Create Q, K and V projections
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        # Combine the heads
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x: torch.Tensor):

        is_single_sequence = x.dim() == 2

        if is_single_sequence:
            x = x.unsqueeze(0)

        batch_size, sequence_length, _ = x.shape

        # --------------------------------
        # Create Q, K, V
        # --------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # --------------------------------
        # Split into multiple heads
        # --------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Shape:
        # [batch_size, num_heads, sequence_length, head_dim]

        # --------------------------------
        # Attention scores
        # --------------------------------

        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_dim)

        # --------------------------------
        # Causal mask
        # --------------------------------

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # --------------------------------
        # Softmax
        # --------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # Weighted values
        # --------------------------------

        attention_output = attention_weights @ V

        # --------------------------------
        # Combine heads
        # --------------------------------

        attention_output = attention_output.transpose(
            1,
            2
        )

        attention_output = attention_output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # --------------------------------
        # Final projection
        # --------------------------------

        output = self.output(attention_output)

        if is_single_sequence:
            output = output.squeeze(0)
            attention_weights = attention_weights.squeeze(0)

        return output, attention_weights