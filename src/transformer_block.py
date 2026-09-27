import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_hidden_dim: int
    ):
        super().__init__()

        # Normalize before attention
        self.layer_norm_1 = nn.LayerNorm(
            embedding_dim
        )

        # Multi-head self-attention
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )

        # Normalize before feed-forward network
        self.layer_norm_2 = nn.LayerNorm(
            embedding_dim
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=ff_hidden_dim
        )

    def forward(self, x: torch.Tensor):

        # -----------------------------
        # Attention sub-layer
        # -----------------------------

        normalized_x = self.layer_norm_1(x)

        attention_output, attention_weights = self.attention(
            normalized_x
        )

        # Residual connection
        x = x + attention_output

        # -----------------------------
        # Feed-forward sub-layer
        # -----------------------------

        normalized_x = self.layer_norm_2(x)

        ffn_output = self.feed_forward(
            normalized_x
        )

        # Residual connection
        x = x + ffn_output

        return x, attention_weights