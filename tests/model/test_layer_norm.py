import torch

from src.model.layer_norm import LayerNorm


class TestLayerNorm:
    def test_init(self) -> None:
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

        assert len(layer_norm._beta) == d_model
        assert len(layer_norm._gamma) == d_model

    def test_forward_after_init(self, init_layer_norm) -> None:
        """
        Test forward right after init.
        """
        layer_norm, feature_vector = init_layer_norm

        true_result_init = torch.tensor([-0.7071, 0.7071])
        layer_norm_result_init = layer_norm.forward(feature_vector)
        assert torch.allclose(true_result_init, layer_norm_result_init, atol=1e-05)

    def test_forward_complex(self, init_layer_norm) -> None:
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
