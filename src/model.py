import torch
import torch.nn as nn

from src.embeddings import TokenEmbedding
from src.positional_embedding import PositionalEmbedding
from src.transformer_block import TransformerBlock


class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_hidden_dim: int = 256,
        max_context_length: int = 64
    ):
        super().__init__()

        self.max_context_length = max_context_length

        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            max_context_length=max_context_length,
            embedding_dim=embedding_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            embedding_dim
        )

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        return_debug=False
    ):

        debug_info = {}

        # -----------------------------
        # Token embeddings
        # -----------------------------

        x = self.embedding(token_ids)

        if return_debug:
            debug_info["token_embeddings"] = x

        # -----------------------------
        # Positional embeddings
        # -----------------------------

        x = self.position_embedding(x)

        if return_debug:
            debug_info["positioned_embeddings"] = x

        # -----------------------------
        # Transformer blocks
        # -----------------------------

        attention_weights = []

        for index, block in enumerate(
            self.transformer_blocks
        ):

            x, weights = block(x)

            attention_weights.append(
                weights
            )

            if return_debug:
                debug_info[
                    f"transformer_block_{index + 1}"
                ] = x

        # -----------------------------
        # Final LayerNorm
        # -----------------------------

        x = self.final_layer_norm(x)

        if return_debug:
            debug_info["final_layer_norm"] = x

        # -----------------------------
        # Language model head
        # -----------------------------

        logits = self.lm_head(x)

        if return_debug:
            debug_info["logits"] = logits

            return (
                logits,
                attention_weights,
                debug_info
            )

        return logits, attention_weights