import pytest
import torch

from src.model.layer_norm import LayerNorm


@pytest.fixture
def init_layer_norm() -> (LayerNorm, torch.Tensor):
    """
    Init LayerNorm and feature_vector.
    """
    d_model = 2  # Dimensionality of the model’s internal representations (embeddings).
    layer_norm = LayerNorm(features=d_model)
    feature_vector = torch.arange(start=1, end=d_model + 1, dtype=torch.float32)
    return layer_norm, feature_vector


def test_layer_norm_init() -> None:
    """
    Test init function of LayerNorm.
    """
    d_model = 2  # Dimensionality of the model’s internal representations (embeddings).
    eps = 1e-6
    layer_norm = LayerNorm(features=d_model)

    # Check parameters
    assert torch.allclose(layer_norm._beta, torch.zeros(d_model))  # beta
    assert torch.allclose(layer_norm._gamma, torch.ones(d_model))  # gamma
    assert layer_norm._eps == eps  # epsilon


def test_layer_norm_forward_after_init(init_layer_norm) -> None:
    """
    Test forward right after init.
    """
    layer_norm, feature_vector = init_layer_norm

    true_result_init = torch.tensor([-0.7071, 0.7071])
    layer_norm_result_init = layer_norm.forward(feature_vector)
    assert torch.allclose(true_result_init, layer_norm_result_init, atol=1e-05)


def test_layer_norm_forward(init_layer_norm) -> None:
    """
    Test forward with changed gamma and betta.
    """
    layer_norm, feature_vector = init_layer_norm

    # Change parameters
    layer_norm._gamma.data.add_(1)
    layer_norm._beta.data.add_(1)

    true_result_complex = torch.tensor([-0.4142, 2.4142])
    layer_norm_result_complex = layer_norm.forward(feature_vector)
    assert torch.allclose(true_result_complex, layer_norm_result_complex, atol=1e-05)
