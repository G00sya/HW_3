from typing import Any

import pytest
import torch
from torch import nn

from src.model.encoder_decoder import EncoderDecoder
from src.utils.shared_embedding import SharedEmbedding


class TestEncoderDecoder:
    def test_initialization(self, init_encoder_decoder: tuple[nn.Module, SharedEmbedding, int]):
        """Test model initialization with valid parameters."""
        model = EncoderDecoder(
            target_vocab_size=100,
            shared_embedding=init_encoder_decoder[1],
            d_model=256,
            d_ff=1024,
            blocks_count=4,
            heads_count=8,
            dropout_rate=0.1,
        )

        assert isinstance(model, EncoderDecoder)
        assert isinstance(model.encoder, nn.Module)
        assert isinstance(model.decoder, nn.Module)
        assert isinstance(model.generator, nn.Module)

    @pytest.mark.parametrize(
        "invalid_params, error_type",
        [
            ({"target_vocab_size": -100}, ValueError),
            ({"target_vocab_size": 0.1}, TypeError),
            ({"d_model": 0}, ValueError),
            ({"d_model": "10"}, TypeError),
            ({"d_ff": -1}, ValueError),
            ({"d_ff": 10.01}, TypeError),
            ({"blocks_count": -9}, ValueError),
            ({"blocks_count": "4"}, TypeError),
            ({"heads_count": -10}, ValueError),
            ({"heads_count": "10"}, TypeError),
            ({"dropout_rate": 1.5}, ValueError),
            ({"dropout_rate": "1.5"}, TypeError),
            ({"shared_embedding": nn.Embedding(100, 256)}, TypeError),
        ],
    )
    def test_parameter_validation(
        self,
        init_encoder_decoder: tuple[nn.Module, SharedEmbedding, int],
        invalid_params: dict[str, Any],
        error_type: BaseException,
    ):
        """Test parameter validation raises appropriate errors."""
        default_params = {
            "target_vocab_size": 100,
            "shared_embedding": init_encoder_decoder[1],
            "d_model": 256,
            "d_ff": 1024,
            "blocks_count": 4,
            "heads_count": 8,
            "dropout_rate": 0.1,
        }
        default_params.update(invalid_params)

        with pytest.raises(error_type):
            EncoderDecoder(**default_params)

    def test_forward_pass(
        self,
        init_encoder_decoder: tuple[nn.Module, SharedEmbedding, int],
        sample_inputs_for_encoder_decoder: dict[str, Any],
    ):
        """Test forward pass returns correct shape."""
        model = init_encoder_decoder[0]
        vocab_size = init_encoder_decoder[2]
        batch_size = sample_inputs_for_encoder_decoder["batch_size"]
        tgt_seq_len = sample_inputs_for_encoder_decoder["tgt_seq_len"]

        outputs = model(
            source_inputs=sample_inputs_for_encoder_decoder["source_inputs"],
            target_inputs=sample_inputs_for_encoder_decoder["target_inputs"],
            source_mask=sample_inputs_for_encoder_decoder["source_mask"],
            target_mask=sample_inputs_for_encoder_decoder["target_mask"],
        )

        assert outputs.shape == (batch_size, tgt_seq_len, vocab_size)
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()

    def test_parameter_initialization(self, init_encoder_decoder: tuple[nn.Module, SharedEmbedding, int]):
        """Test Xavier initialization is applied properly."""
        model = init_encoder_decoder[0]

        for name, param in model.named_parameters():
            if param.dim() > 1:
                assert torch.allclose(
                    param.mean(), torch.tensor(0.0, device=param.device), atol=1e-2
                ), f"Parameter {name} not properly initialized."
