import pytest
import torch
import torch.nn as nn

from src.model.layer_norm import LayerNorm
from src.model.residual_block import ResidualBlock


@pytest.fixture
def init_layer_norm() -> (LayerNorm, torch.Tensor):
    """
    Init LayerNorm and feature_vector.
    """
    d_model = 2  # Dimensionality of the model’s internal representations (embeddings).
    layer_norm = LayerNorm(features=d_model)
    feature_vector = torch.arange(start=1, end=d_model + 1, dtype=torch.float32)
    return layer_norm, feature_vector


@pytest.fixture()
def init_residual_block() -> (ResidualBlock, int, float):
    """
    Init ResidualBlock.
    """
    d_model = 2
    dropout_rate = 0.2
    residual_block = ResidualBlock(size=d_model, dropout_rate=dropout_rate)
    return residual_block, d_model, dropout_rate


@pytest.fixture()
def init_sublayer(init_residual_block) -> nn.Module:
    """
    Init sublayer as linear module.
    """
    residual_block, d_model, dropout_rate = init_residual_block
    linear_model = nn.Linear(in_features=d_model, out_features=d_model)
    return linear_model
