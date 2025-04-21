from enum import Enum

import pandas as pd
from torchtext.data import BucketIterator, Dataset, Example, Field
from tqdm.auto import tqdm

from src.utils.device import setup_device
from src.utils.logger import setup_logger


class Tokens(Enum):
    BOS = "<s>"
    EOS = "</s>"


class Data:
    """
    Class which initializes datasets for train and test data. Can download csv file if it is needed.
    """

    def __init__(self):
        self.__logger = setup_logger("prepare_data")
        self.__device = setup_device()

        self.__word_field = Field(tokenize="moses", init_token=Tokens.BOS, eos_token=Tokens.EOS, lower=True)
        self.__fields = [("source", self.__word_field), ("target", self.__word_field)]

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
        for _, row in tqdm(data.iterrows(), total=len(data)):
            source_text = self.__word_field.preprocess(row.text)
            target_text = self.__word_field.preprocess(row.title)
            examples.append(Example.fromlist([source_text, target_text], self.__fields))

        dataset = Dataset(examples, self.__fields)
        train_dataset, test_dataset = dataset.split(split_ratio=split_ratio)

        self.__logger.info("Train size =", len(train_dataset))
        self.__logger.info("Test size =", len(test_dataset))
        return train_dataset, test_dataset

    def init_dataset(
        self, csv_path: str, batch_sizes: tuple = (16, 32), split_ratio: float = 0.85, min_frequency: float = 0.7
    ) -> tuple[BucketIterator, BucketIterator] | None:
        """
        Initialize train and test BucketIterator from csv file.

        :param csv_path: Full path to csv.
        :param batch_sizes: Batch sizes for iterator (train and test).
        :param split_ratio: Coefficient for splitting dataset to test and train.
        :param min_frequency: Min frequency for a word to be in text for adding it to source vocabulary.
        :return: Train and test BucketIterator or None if file is not found.
        """
        data = self._get_data_pd(csv_path)
        if data is None:
            return None
        train_dataset, test_dataset = self._create_datasets(data, split_ratio)

        self.__word_field.build_vocab(train_dataset, min_freq=min_frequency)
        self.__logger.info("Vocab size =", len(self.__word_field.vocab))

        train_iter, test_iter = BucketIterator.splits(
            datasets=(train_dataset, test_dataset),
            batch_sizes=batch_sizes,
            shuffle=True,
            device=self.__device,
            sort=False,
        )

        return train_iter, test_iter
