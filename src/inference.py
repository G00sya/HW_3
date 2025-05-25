import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.data.prepare_data import Data
from src.model.encoder_decoder import EncoderDecoder
from src.utils.shared_embedding import create_pretrained_embedding


def load_model_and_data(
    model_path: str,
    embedding_path: Optional[str] = None,
    data_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[EncoderDecoder, Data]:
    """Loader with validation checks"""
    base_dir = Path(__file__).parent.parent
    embedding_path = embedding_path or str(base_dir / "embeddings" / "navec_hudlit_v1_12B_500K_300d_100q.tar")
    data_path = data_path or str(base_dir / "data" / "raw" / "news.csv")

    for path, name in [(model_path, "model"), (embedding_path, "embedding"), (data_path, "data")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name.capitalize()} file not found at {path}")

    try:
        # Load embeddings and data
        shared_embedding, navec, pad_idx, unk_idx = create_pretrained_embedding(embedding_path)
        data = Data(navec)
        data.init_dataset(data_path)

        # Recreate model
        model = EncoderDecoder(
            target_vocab_size=len(navec.key_to_index),
            shared_embedding=shared_embedding,
            d_model=int(navec.vector_size),
            heads_count=10,
        ).to(device)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        return model, data

    except Exception as e:
        raise RuntimeError(f"Loading failed: {str(e)}")


def batch_predict(
    model: EncoderDecoder,
    data: Data,
    texts: List[str],
    max_length: int = 100,
    device: torch.device = None,
    output_file: Optional[str] = None,
) -> Tuple[List[str], Dict]:
    """
    Run prediction on multiple texts with optional saving.

    :param model: The summarization model.
    :param data: The Data object containing vocabulary and other information.
    :param texts: A list of input texts to summarize.
    :param max_length: The maximum length of the generated summaries.
    :param device: The device to run the model on (CPU or GPU).
    :param output_file: Optional path to a JSON file to save the results.

    :return: Tuple of (predictions, results_dict) where results_dict contains:
        - inputs: original texts
        - predictions: generated summaries
        - tokens: tokenized inputs (for debugging)
    """
    device = device or next(model.parameters()).device
    results = {
        "inputs": texts,
        "predictions": [],
        "tokens": [],
    }

    for i, text in enumerate(texts):
        try:
            # Store tokenized version for debugging
            tokens = data.word_field.preprocess(text)
            results["tokens"].append(tokens)

            pred = model.predict(text, data, max_length, device)
            results["predictions"].append(pred)
        except Exception as e:
            print(f"Failed on text: {text[:50]}... - {str(e)}")
            results["predictions"].append("")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    return results["predictions"], results


def interactive_predict(model: EncoderDecoder, data: Data, device: torch.device):
    """Interactive session with debug options"""
    print("\nEnter text to summarize (or commands):")
    print("- 'debug' to toggle verbose mode")
    print("- 'quit' to exit\n")

    debug = False
    while True:
        try:
            text = input("> ").strip()
            if text.lower() == "quit":
                break
            elif text.lower() == "debug":
                debug = not debug
                print(f"Debug mode {'ON' if debug else 'OFF'}")
                continue

            if not text:
                print("Error: Empty input")
                continue

            prediction = model.predict(source_text=text, data=data, max_length=200, device=device, verbose=debug)
            print(f"\nSummary: {prediction}\n{'=' * 50}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
