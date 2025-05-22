from collections import Counter
from enum import Enum

import pandas as pd
from gensim.models import KeyedVectors
from sacremoses import MosesTokenizer
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

    def __init__(self, embedding_model: KeyedVectors | None = None):
        if embedding_model is not None:
            if not isinstance(embedding_model, KeyedVectors):
                raise TypeError(f"embedding_model must be a KeyedVectors or None, but got {type(embedding_model)}")

        self.__logger = setup_logger("prepare_data")
        self.__device = setup_device()

        self.word_field = Field(
            tokenize=lambda x: self._safe_moses_tokenize(x),
            init_token=Tokens.BOS.value,
            eos_token=Tokens.EOS.value,
            pad_token=Tokens.PAD.value,
            unk_token=Tokens.UNK.value,
            use_vocab=True,
        )
        self.__embedding_model = embedding_model
        self.__mt = MosesTokenizer(lang="ru")

        self.__fields = [("source", self.word_field), ("target", self.word_field)]

    def _safe_moses_tokenize(self, text: str) -> list[str]:
        """
        Tokenize with Moses and convert OOV words to UNK.
        """
        tokens = self.__mt.tokenize(text.lower(), escape=False)
        return [t if t in self.__embedding_model else Tokens.UNK.value for t in tokens]

    def _build_vocab_from_gensim(self):
        """
        Builds a torchtext Vocab object from a Gensim Word2Vec or FastText model.
        :returns: None.
        """
        all_tokens = sorted(self.__embedding_model.key_to_index.items(), key=lambda x: x[1])
        tokens, original_indexes = zip(*all_tokens)
        dummy_counter = Counter({token: 1 for token in tokens})

        self.word_field.vocab = Vocab(dummy_counter, specials=[], specials_first=False)
        self.word_field.vocab.itos = list(tokens)
        self.word_field.vocab.stoi = {token: idx for idx, token in enumerate(tokens)}
        self.word_field.vocab.unk_index = self.__embedding_model.key_to_index[Tokens.UNK.value]

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

        # Build vocabulary
        if self.__embedding_model is None:
            self.word_field.build_vocab(
                train_dataset,
                min_freq=2,
            )
        else:
            self._build_vocab_from_gensim()

        self.__logger.info(f"Vocab size = {len(self.word_field.vocab)}")

        train_iter, test_iter = BucketIterator.splits(
            datasets=(train_dataset, test_dataset),
            batch_sizes=batch_sizes,
            shuffle=True,
            device=self.__device,
            sort=False,
        )

        return train_iter, test_iter
