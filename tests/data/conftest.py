import pandas as pd
import pytest


@pytest.fixture
def temp_csv_file(tmp_path):
    """
    Fixture to create a temporary CSV file for testing.
    """
    csv_path = tmp_path / "test.csv"
    data = {"text": ["this is a test", "another test", "this is another"], "title": ["title1", "title2", "title3"]}
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return str(csv_path), data
