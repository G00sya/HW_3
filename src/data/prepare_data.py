from collections import Counter
from enum import Enum

import pandas as pd
import torch
from navec import Navec
from torchtext.data import BucketIterator, Dataset, Example, Field
from torchtext.vocab import Vocab
from tqdm.auto import tqdm

from src.utils.device import setup_device
from src.utils.logger import setup_logger


class Tokens(Enum):
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<pad>"
    UNK = "<unk>"


class Data:
    """
    Class which initializes datasets for train and test data. Can download csv file if it is needed.
    """

    def __init__(self, navec: Navec):
        """
        :param navec: Navec model as pretrained_embedding
        """
        self.__logger = setup_logger("prepare_data")
        self.__device = setup_device()

        # Initialize Field with Navec's special tokens
        self.__navec = navec
        self.word_field = Field(
            tokenize="moses",
            init_token=Tokens.BOS,
            eos_token=Tokens.EOS,
            lower=True,
            pad_token=Tokens.PAD,
            unk_token=Tokens.UNK,
        )
        self.__fields = [("source", self.word_field), ("target", self.word_field)]

    def _get_data_pd(self, csv_path: str) -> pd.DataFrame | None:
        """
        Try to create pd.Dataframe from csv file or return None if csv doesn't exist.

        :param csv_path: Full path to csv.
        :return: Dataframe or None in case of error.
        """
        try:
            data = pd.read_csv(csv_path, delimiter=",")
            return data
        except FileNotFoundError:
            self.__logger.error(f"The file '{csv_path}' was not found.")
            return None

    def _create_datasets(self, data: pd.DataFrame, split_ratio: float) -> (Dataset, Dataset):
        """
        Create train and test datasets with preprocessing from dataframe.

        :param data: Dataframe to convert.
        :param split_ratio: Coefficient for splitting dataset to test and train.
        :return: Train and test dataset.
        """
        examples = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Creating datasets"):
            source_text = self.word_field.preprocess(row.text)
            target_text = self.word_field.preprocess(row.title)
            examples.append(Example.fromlist([source_text, target_text], self.__fields))

        dataset = Dataset(examples, self.__fields)
        train_dataset, test_dataset = dataset.split(split_ratio=split_ratio)

        self.__logger.info(f"Train size = {len(train_dataset)}")
        self.__logger.info(f"Test size = {len(test_dataset)}")
        return train_dataset, test_dataset

    def init_dataset(
        self, csv_path: str, batch_sizes: tuple = (16, 32), split_ratio: float = 0.85
    ) -> tuple[BucketIterator, BucketIterator] | None:
        """
        Initialize train and test BucketIterator from csv file.

        :param csv_path: Full path to csv.
        :param batch_sizes: Batch sizes for iterator (train and test).
        :param split_ratio: Coefficient for splitting dataset to test and train.
        :return: Train and test BucketIterator or None if file is not found.
        """
        data = self._get_data_pd(csv_path)
        if data is None:
            return None
        train_dataset, test_dataset = self._create_datasets(data, split_ratio)

        # Build vocab using Navec's vocabulary
        self._build_vocab_with_navec()

        train_iter, test_iter = BucketIterator.splits(
            datasets=(train_dataset, test_dataset),
            batch_sizes=batch_sizes,
            shuffle=True,
            device=self.__device,
            sort=False,
        )

        return train_iter, test_iter

    def _build_vocab_with_navec(self):
        """
        Manually assign Navec's vocabulary to the Field.
        """
        # Create dummy Vocab
        vocab = Vocab(Counter(), specials_first=False)

        # Build stoi (string-to-index) using navec.vocab.get()
        vocab.stoi = {
            word: self.__navec.vocab.get(word, self.__navec.vocab.unk_id) for word in self.__navec.vocab.words
        }

        # Add punctuation marks
        punctuation = [
            ",",
            ".",
            "!",
            "?",
            ";",
            ":",
            "-",
            "(",
            ")",
            '"',
            "'",
            "«",
            "»",
            "&quot;",
            "&amp;",
            "&lt;",
            "&gt;",
        ]
        for mark in punctuation:
            vocab.stoi[mark] = len(vocab.stoi)

        # Add special tokens explicitly (if not already in navec.vocab.words)
        special_tokens = {
            Tokens.PAD: self.__navec.vocab.pad_id,
            Tokens.UNK: self.__navec.vocab.unk_id,
            Tokens.BOS: getattr(self.__navec.vocab, "bos_id", len(vocab.stoi)),  # Optional
            Tokens.EOS: getattr(self.__navec.vocab, "eos_id", len(vocab.stoi) + 1),  # Optional
        }
        for token, idx in special_tokens.items():
            if token not in vocab.stoi:
                vocab.stoi[token] = idx

        # Create vectors for vocab
        embedding_matrix_shape = tuple(map(int, self.__navec.pq.shape))
        embedding_matrix = torch.zeros(embedding_matrix_shape)
        for word in self.__navec.vocab.words:
            embedding = self.__navec[word]
            word_id = self.__navec.vocab[word]
            embedding_matrix[word_id] = torch.Tensor(embedding)
        vocab.vectors = embedding_matrix

        self.word_field.vocab = vocab
