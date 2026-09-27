import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_context_length: int, embedding_dim: int):
        super().__init__()

        self.position_embedding = nn.Embedding(
            num_embeddings=max_context_length,
            embedding_dim=embedding_dim
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:

        if token_embeddings.dim() == 2:
            sequence_length = token_embeddings.size(0)
            positions = torch.arange(
                sequence_length,
                device=token_embeddings.device
            )
            position_embeddings = self.position_embedding(positions)
            return token_embeddings + position_embeddings

        if token_embeddings.dim() == 3:
            sequence_length = token_embeddings.size(1)
            positions = torch.arange(
                sequence_length,
                device=token_embeddings.device
            )
            position_embeddings = self.position_embedding(positions)
            return token_embeddings + position_embeddings.unsqueeze(0)

        raise ValueError(
            "Expected token embeddings with shape "
            "[sequence, embedding] or "
            "[batch, sequence, embedding]."
        )