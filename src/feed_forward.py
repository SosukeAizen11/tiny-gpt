import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int
    ):
        super().__init__()

        self.network = nn.Sequential(

            # Expand representation
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            # Non-linear activation
            nn.GELU(),

            # Project back
            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)