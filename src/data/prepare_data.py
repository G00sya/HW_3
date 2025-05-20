from collections import Counter
from enum import Enum

import pandas as pd
import torch
from gensim.models import KeyedVectors
from torchtext.data import BucketIterator, Dataset, Example, Field
from torchtext.vocab import Vocab
from tqdm.auto import tqdm

from src.utils.device import setup_device
from src.utils.logger import setup_logger


class Tokens(Enum):
    BOS = "<s>"
    EOS = "</s>"
    UNK = "<unk>"
    PAD = "<pad>"


class Data:
    """
    Class which initializes datasets for train and test data. Can download csv file if it is needed.
    """

    def __init__(self, navec):
        self.__logger = setup_logger("prepare_data")
        self.__device = setup_device()
        self.__navec = self._build_keyed_vectors(navec)

        self.word_field = Field(
            tokenize="moses",
            init_token=Tokens.BOS,
            eos_token=Tokens.EOS,
            pad_token=Tokens.PAD,
            unk_token=Tokens.UNK,
            use_vocab=True,
        )
        self.__fields = [("source", self.word_field), ("target", self.word_field)]

    def _build_keyed_vectors(self, navec) -> KeyedVectors:
        """
        Builds a KeyedVectors object with pre-computed weights from Navec model.
        :return: KeyedVectors model.
        """
        vector_size = navec.pq.dim  # Assuming self.pq.dim is the embedding dimension

        # Get word list and weights
        word_list = navec.vocab.words
        weights = navec.pq.unpack()  # Assuming self.pq.unpack() returns a NumPy array of weights

        # Create a dictionary mapping words to their vectors
        word_vectors = {}
        for word, vector in zip(word_list, weights):
            word_vectors[word] = vector

        # Create KeyedVectors object
        model = KeyedVectors(vector_size)

        # Load the precomputed vectors into the KeyedVectors object.
        model.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))  # load the weights
        return model

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

        vocab, vectors = self._build_vocab_from_gensim()
        self.word_field.vocab = vocab
        self.word_field.embedding_matrix = vectors
        self.__logger.info(f"Vocab size = {len(self.word_field.vocab)}")

        train_iter, test_iter = BucketIterator.splits(
            datasets=(train_dataset, test_dataset),
            batch_sizes=batch_sizes,
            shuffle=True,
            device=self.__device,
            sort=False,
        )

        return train_iter, test_iter

    def _build_vocab_from_gensim(self) -> (Vocab, torch.Tensor):
        """
        Builds a torchtext Vocab object from a Gensim Word2Vec or FastText model.

        :returns: A torchtext Vocab object. Also returns the embedding matrix as a Torch tensor.
        """
        # Get word counts from model's vocabulary
        word_counts = Counter(self.__navec.key_to_index)

        # Create the Vocab object
        specials = [Tokens.BOS.value, Tokens.EOS.value, Tokens.UNK.value, Tokens.PAD.value]
        vocab = Vocab(word_counts, specials=specials)

        # Create an embedding matrix
        embedding_dim = self.__navec.vector_size
        vocab_size = len(vocab)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))

        # Populate the embedding matrix with pre-trained vectors
        for i, word in enumerate(vocab.itos):  # itos for index to string
            if word in self.__navec:
                embedding_matrix[i] = torch.tensor(self.__navec[word])
            else:
                # Handle <BOS>, <EOS> with random initialization
                embedding_matrix[i] = torch.randn(embedding_dim)

        return vocab, embedding_matrix
