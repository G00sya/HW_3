import pytest
import torch

from src.model.mask import convert_batch


@pytest.fixture
def example_batch_tensors():
    """Creates sample source and target tensors for testing."""
    source_data = torch.tensor([[1, 2, 0, 4], [5, 6, 0, 7]])
    target_data = torch.tensor([[8, 9, 0], [10, 11, 0]])
    return source_data, target_data


def test_convert_batch_shape_no_mock(example_batch_tensors):
    """Tests output shapes without using a MockBatch, using the tensors directly."""
    source_data, target_data = example_batch_tensors

    class MockBatch:  # Inner class for simplicity
        def __init__(self, source, target):
            self.source = source
            self.target = target

    example_batch = MockBatch(source_data, target_data)
    source_inputs, target_inputs, source_mask, target_mask = convert_batch(example_batch, pad_idx=0)

    assert source_inputs.shape == (2, 4)  # (batch_size, source_seq_len)
    assert target_inputs.shape == (2, 3)  # (batch_size, target_seq_len)
    assert source_mask.shape == (2, 1, 4)  # (batch_size, 1, source_seq_len)
    assert target_mask.shape == (2, 3, 3)  # (batch_size, target_seq_len, target_seq_len)


def test_convert_batch_values_no_mock(example_batch_tensors):
    """Tests output values without using MockBatch."""
    source_data, target_data = example_batch_tensors

    class MockBatch:
        def __init__(self, source, target):
            self.source = source
            self.target = target

    example_batch = MockBatch(source_data, target_data)

    source_inputs, target_inputs, source_mask, target_mask = convert_batch(example_batch, pad_idx=0)

    # Check a couple of values in the input tensors (after transpose)
    assert source_inputs[0, 0] == 1
    assert target_inputs[1, 1] == 11

    # Check some values in the source mask - should be false for padding, true otherwise
    assert source_mask[0, 0, 2] is False  # Padding token in source should be masked
    assert source_mask[0, 0, 0] is True  # Non-padding token in source should not be masked
    assert source_mask[1, 0, 2] is False
    assert source_mask[1, 0, 3] is True

    # Basic checks for the target mask. More thorough checks are done in the test for `make_mask`
    assert target_mask[0, 0, 0] is True
    assert target_mask[0, 0, 1] is False  # Subsequent masking
    assert target_mask[1, 2, 2] is True
    assert target_mask[0, 1, 2] is False
