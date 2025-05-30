from pathlib import Path

import torch
import torch.nn as nn

from src.data.prepare_data import Data, Tokens
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

    def predict(
        self,
        source_text: str,
        data: Data,
        max_length: int = 100,
        device: torch.device = None,
        verbose: bool = False,  # Add debug prints if needed
    ) -> str:
        """
        Generate a prediction for the given source text using greedy decoding.

        :param source_text: Input text string
        :param data: Data object containing vocabulary and preprocessing
        :param max_length: Maximum length of generated sequence
        :param device: Device to run the computation on
        :param verbose: Print debug info
        :return: Generated sentence as a string
        """
        self.eval()
        device = device or next(self.parameters()).device

        if not source_text.strip():
            return ""

        try:
            # Tokenize with Moses (already handles UNK replacement)
            tokenized = data.word_field.preprocess(source_text)
            if verbose:
                print(f"Tokenized: {tokenized[:10]}...")

            # Numericalize tokens
            vocab = data.word_field.vocab
            numericalized = [vocab.stoi.get(token, vocab.stoi[Tokens.UNK.value]) for token in tokenized]

            # Prepare tensors
            source_inputs = torch.tensor(numericalized, dtype=torch.long, device=device).unsqueeze(0)
            pad_idx = vocab.stoi[Tokens.PAD.value]
            source_mask = (source_inputs != pad_idx).unsqueeze(-2)

            # Initialize generation
            bos_idx = vocab.stoi[Tokens.BOS.value]
            eos_idx = vocab.stoi[Tokens.EOS.value]
            target_inputs = torch.tensor([[bos_idx]], device=device)

            # Autoregressive decoding
            for _ in range(max_length):
                # Create masks
                target_mask = (target_inputs != pad_idx).unsqueeze(-2)
                target_mask = target_mask & self.subsequent_mask(target_inputs.size(-1)).to(device)

                # Forward pass
                with torch.no_grad():
                    logits = self.forward(source_inputs, target_inputs, source_mask, target_mask)
                    next_token = logits[:, -1].argmax(-1)

                    if verbose:
                        topk = logits[:, -1].exp().topk(5)
                        print(
                            "Top predictions:",
                            [(vocab.itos[i], f"{v:.2f}") for v, i in zip(topk.values[0], topk.indices[0])],
                        )

                # Append token
                target_inputs = torch.cat([target_inputs, next_token.unsqueeze(0)], dim=-1)

                # Stop if EOS
                if next_token.item() == eos_idx:
                    break

            # Convert to text
            tokens = target_inputs.squeeze(0).tolist()
            words = [vocab.itos[t] for t in tokens if t not in {bos_idx, eos_idx, pad_idx}]

            # Fallback if all UNK
            if all(w == Tokens.UNK.value for w in words):
                known_words = [t for t in tokenized if t in vocab.stoi]
                fallback = " ".join(known_words[:3]) if known_words else tokenized[0]
                if verbose:
                    print(f"Fallback triggered. Original: {tokenized[:3]}")
                return fallback

            return " ".join(words)

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return source_text.split()[0]  # Ultimate fallback

    @staticmethod
    def subsequent_mask(size: int) -> torch.Tensor:
        """
        Create a mask for subsequent positions to prevent attending to future tokens.

        :param size: Size of the mask (sequence length)
        :return: Mask tensor of shape (1, size, size)
        """
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0

    def save_model(self, name) -> None:
        """Save the model's state dictionary to a file."""
        path = Path(__file__).parent.parent.parent / "model" / name
        torch.save(self.state_dict(), path)
