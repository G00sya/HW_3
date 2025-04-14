import torch.nn as nn

from src.model.decoder_layer import DecoderLayer
from src.model.layer_norm import LayerNorm
from src.model.multi_headed_attention import MultiHeadedAttention
from src.model.positional_encoding import PositionalEncoding
from src.model.positionwise_feed_forward import PositionwiseFeedForward


class Decoder(nn.Module):
    """
    The Decoder module, responsible for generating the output sequence given the encoder's output.
    It consists of an embedding layer, positional encoding, a stack of DecoderLayers, a layer normalization,
    and a linear output layer.
    """

    def __init__(self, vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate):
        """
        Initializes the Decoder.

        :param vocab_size: The size of the target vocabulary.
        :param d_model: The dimensionality of the model's embeddings and hidden states.
        :param d_ff: The dimensionality of the feed-forward network's inner layer.
        :param blocks_count: The number of decoder blocks (DecoderLayers) to stack.
        :param heads_count: The number of attention heads in each multi-head attention layer.
        :param dropout_rate: The dropout probability.
        """
        super().__init__()

        # Проверка типов для int параметров
        int_params = {
            "vocab_size": vocab_size,
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

        self.__emb = nn.Sequential(nn.Embedding(vocab_size, d_model), PositionalEncoding(d_model, dropout_rate))

        def create_decoder_block():
            """
            Helper function to create a single DecoderLayer.  This is necessary to avoid code duplication
            and make the Decoder's __init__ method more readable.
            """
            return DecoderLayer(
                size=d_model,
                self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
                encoder_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
                feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
                dropout_rate=dropout_rate,
            )

        self.__blocks = nn.ModuleList([create_decoder_block() for _ in range(blocks_count)])
        self.__norm = LayerNorm(d_model)
        self.__out_layer = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        """
        Performs a forward pass through the Decoder.

        :param inputs: The input tensor to the decoder.  Shape: (batch_size, target_seq_len).
        These are the indices of the target vocabulary.
        :param encoder_output: The output tensor from the encoder. Shape: (batch_size, source_seq_len, d_model).
        :param source_mask: The mask for the source sequence, indicating which positions are valid (not padding).
        Shape: (batch_size, 1, source_seq_len).
        :param target_mask: The mask for the target sequence,
        indicating which positions are valid and preventing future positions from being attended to (look-ahead mask).
        Shape: (batch_size, target_seq_len, target_seq_len).
        :return: The output tensor after processing by the decoder.  Shape: (batch_size, target_seq_len, vocab_size).
        These are the logits for each word in the target vocabulary.
        """
        inputs = self.__emb(inputs)
        for block in self.__blocks:
            inputs = block(inputs, encoder_output, source_mask, target_mask)
        return self.__out_layer(self.__norm(inputs))
