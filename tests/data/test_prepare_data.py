from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from torchtext.data import BucketIterator, Dataset

from src.data.prepare_data import Data


class TestData:
    @pytest.fixture(autouse=True)  # Use it in all tests for having self.data_handler
    def setup(self):
        """
        Create Data object and init logger and device as Mocks.
        """
        with patch("src.data.prepare_data.setup_logger") as mock_logger:
            mock_logger.return_value = MagicMock()
            with patch("src.data.prepare_data.setup_device") as mock_device:
                mock_device.return_value = "cpu"
                self.data_handler = Data()
                yield  # to clear the memory after tests

    @pytest.fixture
    def mock_pd_read_csv(self):
        with patch("pandas.read_csv") as mock_read_csv:  # Make function read_csv as Mock to change return_value
            yield mock_read_csv

    def test_get_data_pd_success(self, mock_pd_read_csv):
        # Mock the read_csv to return a sample DataFrame
        mock_pd_read_csv.return_value = pd.DataFrame(
            {"text": ["example text 1", "example text 2"], "title": ["example title 1", "example title 2"]}
        )

        csv_path = "data/raw/news.csv"
        result = self.data_handler._get_data_pd(csv_path)

        # Assert that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Train and Test

    def test_get_data_pd_file_not_found(self, mock_pd_read_csv):
        # Mock the read_csv to raise a FileNotFoundError
        mock_pd_read_csv.side_effect = FileNotFoundError

        csv_path = "data/raw/news.csv"
        result = self.data_handler._get_data_pd(csv_path)

        # Assert that the result is None
        assert result is None

    def test_create_datasets(self):
        # Create a sample DataFrame
        data = pd.DataFrame(
            {"text": ["example text 1", "example text 2"], "title": ["example title 1", "example title 2"]}
        )

        split_ratio = 0.5
        train_dataset, test_dataset = self.data_handler._create_datasets(data, split_ratio)

        # Assert that the result is a tuple of Dataset objects
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)

    def test_init_dataset(self, temp_csv_file):
        """
        Test the init_dataset function with a temporary CSV file.
        """
        file, data_dict = temp_csv_file
        data = Data()
        batch_sizes = (16, 32)
        train_iter, test_iter = data.init_dataset(file, batch_sizes=batch_sizes, split_ratio=1 / len(data_dict))

        assert train_iter is not None
        assert test_iter is not None
        assert isinstance(train_iter, BucketIterator)
        assert isinstance(test_iter, BucketIterator)

        # Assert vocab size
        assert len(data._Data__word_field.vocab) > 0

        # Check if batch sizes are correct
        for batch in train_iter:
            print(batch)
            assert batch.source.shape[0] <= batch_sizes[0]
            break  # Only need to check one batch.

        for batch in test_iter:
            assert batch.target.shape[1] <= batch_sizes[1]
            break

    def test_init_dataset_file_not_found(self):
        """
        Test the init_dataset function when the CSV file is not found.
        """
        data = Data()
        result = data.init_dataset("nonexistent_file.csv")
        assert result is None
