import torch
import torch.nn as nn

from src.model.encoder_block import EncoderBlock
from src.model.layer_norm import LayerNorm
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positional_encoding import PositionalEncoding
from src.model.positionwise_feed_forward import PositionwiseFeedForward
from src.utils.shared_embedding import SharedEmbedding


class Encoder(nn.Module):
    """
    Encoder module for a Transformer-based model.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        blocks_count: int,
        heads_count: int,
        dropout_rate: float | int,
        shared_embedding: SharedEmbedding,
    ):
        """
        Initializes the Encoder.

        :param d_model: The dimensionality of the model's embeddings and hidden states.
        :param d_ff: The dimensionality of the feed-forward network's inner layer.
        :param blocks_count: The number of encoder blocks to stack.
        :param heads_count: The number of attention heads in each multi-head attention layer.
        :param dropout_rate: The dropout probability.
        :param shared_embedding: SharedEmbedding object.
        """
        super().__init__()

        int_params = {
            "d_model": d_model,
            "d_ff": d_ff,
            "blocks_count": blocks_count,
            "heads_count": heads_count,
        }
        for name, value in int_params.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value)}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not isinstance(dropout_rate, (float, int)):
            raise TypeError(f"dropout_rate must be a float or an int, but got {type(dropout_rate)}.")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        if not isinstance(shared_embedding, SharedEmbedding):
            raise TypeError(f"shared_embedding must be a SharedEmbedding object, but got {type(dropout_rate)}.")

        # Embedding layer with positional encoding
        self.__emb = nn.Sequential(
            shared_embedding,  # Embed tokens into d_model dimensional vectors
            PositionalEncoding(d_model, dropout_rate),  # Add positional information
        )

        def create_encoder_block():
            return EncoderBlock(
                size=d_model,
                self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
                feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
                dropout_rate=dropout_rate,
            )

        self.__blocks = nn.ModuleList([create_encoder_block() for _ in range(blocks_count)])
        self.__norm = LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        :param inputs: Input tensor of shape (batch_size, sequence_length), representing token indices.
        :param mask: Mask tensor of shape (batch_size, sequence_length, sequence_length) used to mask attention weights.

        :return: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        inputs = self.__emb(inputs)  # Apply embedding and positional encoding

        # Pass through encoder blocks
        for block in self.__blocks:
            inputs = block(inputs, mask)

        return self.__norm(inputs)  # Apply layer normalization
