import torch
import torch.nn as nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.generator import Generator
from src.utils.shared_embedding import SharedEmbedding


class EncoderDecoder(nn.Module):
    """
    A transformer-based Encoder-Decoder architecture for sequence-to-sequence tasks.
    Uses shared embeddings between encoder and decoder, and includes a final generator layer.
    """

    def __init__(
        self,
        target_vocab_size: int,
        shared_embedding: SharedEmbedding,
        d_model: int = 256,
        d_ff: int = 1024,
        blocks_count: int = 4,
        heads_count: int = 8,
        dropout_rate: float = 0.1,
    ):
        """
        Initializes the EncoderDecoder.

        :param target_vocab_size: Size of the target vocabulary (output vocabulary size).
        :param shared_embedding: Shared embedding layer for both encoder and decoder input.
        :param d_model: Dimension of model embeddings, defaults to 256.
        :param d_ff: Dimension of feed-forward layer in transformer blocks, defaults to 1024.
        :param blocks_count: Number of transformer blocks in encoder/decoder, defaults to 4.
        :param heads_count: Number of attention heads in multi-head attention, defaults to 8.
        :param dropout_rate: Dropout probability, defaults to 0.1
        """
        super(EncoderDecoder, self).__init__()

        int_params = {
            "target_vocab_size": target_vocab_size,
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

        self.d_model = d_model
        self.encoder = Encoder(d_model, d_ff, blocks_count, heads_count, dropout_rate, shared_embedding)
        self.decoder = Decoder(d_model, d_ff, blocks_count, heads_count, dropout_rate, shared_embedding)
        self.generator = Generator(d_model, target_vocab_size)  # Final projection layer to target vocabulary

        # Xavier initialization for all weight matrices in the model
        for p in self.parameters():
            p = p.float()
            if p.dim() > 1:  # Only initialize matrices (not vectors/scalars)
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        source_inputs: torch.Tensor,
        target_inputs: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder architecture.

        :param source_inputs: Source sequence tensor of shape (batch_size, src_seq_len)
        :param target_inputs: Target sequence tensor of shape (batch_size, tgt_seq_len)
        :param source_mask: Source mask tensor for padding (shape (batch_size, src_seq_len, src_seq_len))
        :param target_mask: Target mask tensor for causal masking and padding
                            (shape (batch_size, tgt_seq_len, tgt_seq_len))
        :return: Generator output tensor with probabilities and shape (batch_size, tgt_seq_len, vocab_size)
        """
        # Returns: (batch_size, src_seq_len, d_model)
        encoder_output = self.encoder(source_inputs, source_mask)
        # Returns: (batch_size, tgt_seq_len, d_model)
        decoder_output = self.decoder(target_inputs, encoder_output, source_mask, target_mask)

        return self.generator(decoder_output)
