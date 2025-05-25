from pathlib import Path
from typing import Optional

import torch

from src.data.prepare_data import Data
from src.model.encoder_decoder import EncoderDecoder
from src.utils.device import setup_device
from src.utils.shared_embedding import create_pretrained_embedding


def load_model_and_data(
    model_path: str,
    embedding_path: Optional[str] = None,
    data_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[EncoderDecoder, Data]:
    """Load trained model with all required components"""
    # 1. Validate paths
    embedding_path = embedding_path or str(
        Path(__file__).parent.parent / "embeddings" / "navec_hudlit_v1_12B_500K_300d_100q.tar"
    )
    data_path = data_path or str(Path(__file__).parent.parent / "data" / "news.csv")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not Path(embedding_path).exists():
        raise FileNotFoundError(f"Embedding file not found at {embedding_path}")
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # 2. Load embeddings and data processor
    try:
        shared_embedding, navec, _, _ = create_pretrained_embedding(path=embedding_path)
        data = Data(navec)
        data.init_dataset(data_path)  # Initialize vocabulary
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embeddings/data: {str(e)}")

    # 3. Recreate model architecture
    try:
        vocab_size = len(navec.index_to_key)
        d_model = int(navec.vector_size)
        model = EncoderDecoder(
            target_vocab_size=vocab_size, shared_embedding=shared_embedding, d_model=d_model, heads_count=10
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

    return model, data


def interactive_predict(model: EncoderDecoder, data: Data, device: torch.device):
    """Run interactive prediction session"""
    print("\nEnter text to summarize (or 'quit' to exit):")
    while True:
        try:
            text = input("> ").strip()
            if text.lower() == "quit":
                break
            if not text:
                print("Error: Empty input")
                continue

            prediction = model.predict(source_text=text, data=data, max_length=100, device=device)
            print("\nGenerated summary:", prediction, "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    try:
        DEVICE = setup_device()
        MODEL_PATH = str(Path(__file__).parent.parent / "model" / "model-30_epochs-without_unk_and_punctuation.pt")

        print("Loading model and data...")
        model, data = load_model_and_data(model_path=MODEL_PATH, device=DEVICE)
        print("Model loaded successfully!")

        # After data initialization
        # sample_tokens = ["Пластик", "ученые", "частицы"]  # Words from your input
        # print("\nVocabulary check:")
        # for token in sample_tokens:
        #     idx = data.word_field.vocab.stoi.get(token, -1)
        #     print(f"'{token}': {'Exists' if idx != -1 else 'UNK'} (idx={idx})")

        interactive_predict(model, data, DEVICE)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)
