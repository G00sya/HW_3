import torch
import torch.nn as nn

from src.model.encoder_block import EncoderBlock
from src.model.layer_norm import LayerNorm
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positional_encoding import PositionalEncoding
from src.model.positionwise_feed_forward import PositionwiseFeedForward


class Encoder(nn.Module):
    """
    Encoder module for a Transformer-based model.
    """

    def __init__(
        self, vocab_size: int, d_model: int, d_ff: int, blocks_count: int, heads_count: int, dropout_rate: float
    ):
        """
        Initializes the Encoder.

        :param vocab_size: The size of the vocabulary (number of unique tokens).
        :param d_model: The dimensionality of the model's embeddings and hidden states.
        :param d_ff: The dimensionality of the feed-forward network's inner layer.
        :param blocks_count: The number of encoder blocks to stack.
        :param heads_count: The number of attention heads in each multi-head attention layer.
        :param dropout_rate: The dropout probability.
        """
        super().__init__()

        # Embedding layer with positional encoding
        self._emb = nn.Sequential(
            nn.Embedding(vocab_size, d_model),  # Embed tokens into d_model dimensional vectors
            PositionalEncoding(d_model, dropout_rate),  # Add positional information
        )

        def create_encoder_block():
            return EncoderBlock(
                size=d_model,
                self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
                feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
                dropout_rate=dropout_rate,
            )

        self._blocks = nn.ModuleList([create_encoder_block() for _ in range(blocks_count)])
        self._norm = LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        :param inputs: Input tensor of shape (batch_size, sequence_length), representing token indices.
        :param mask: Mask tensor of shape (batch_size, sequence_length, sequence_length) used to mask attention weights.

        :return: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        # Apply embedding and positional encoding
        inputs = self._emb(inputs)

        # Pass through encoder blocks
        for block in self._blocks:
            inputs = block(inputs, mask)

        return self._norm(inputs)  # Apply layer normalization
