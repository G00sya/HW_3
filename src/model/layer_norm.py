import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Class which:
    1) Makes inputs as a vector with mean = 0, and std = 1
    2) Uses trainable parameters gamma and beta for appropriate modification of inputs: y = gamma * x + beta.
    """

    def __init__(self, features: int, eps=1e-6):
        """
        :param features: Dimensionality of the model’s internal representations (embeddings).
        :param eps: A small number which fixes potential division by zero.
        """
        super().__init__()

        self._gamma = nn.Parameter(torch.ones(features))
        self._beta = nn.Parameter(torch.zeros(features))
        self._eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Calculates y = gamma * x + beta with normalized x.

        :param inputs: The model’s internal representation (embeddings). Shape: (features)
        :return: Result of normalization.
        """
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        inputs_norm = (inputs - mean) / (std + self._eps)
        return self._gamma * inputs_norm + self._beta
