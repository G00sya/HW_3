from src.utils.mask import convert_batch


def test_convert_batch_shape_no_mock(example_batch_tensors):
    """Tests output shapes without using a MockBatch, using the tensors directly."""
    source_data, target_data = example_batch_tensors

    class MockBatch:  # Inner class for simplicity
        def __init__(self, source, target):
            self.source = source
            self.target = target

    example_batch = MockBatch(source_data, target_data)
    source_inputs, target_inputs, source_mask, target_mask = convert_batch(example_batch, pad_idx=0, unk_idx=0)

    assert source_inputs.shape == (4, 2)  # (source_seq_len, batch_size)
    assert target_inputs.shape == (3, 2)  # (target_seq_len, batch_size)
    assert source_mask.shape == (4, 1, 2)  # (batch_size, 1, source_seq_len)
    assert target_mask.shape == (3, 2, 2)  # (batch_size, target_seq_len, target_seq_len)


def test_convert_batch_values_no_mock(example_batch_tensors):
    """Tests output values without using MockBatch."""
    source_data, target_data = example_batch_tensors

    class MockBatch:
        def __init__(self, source, target):
            self.source = source
            self.target = target

    example_batch = MockBatch(source_data, target_data)

    source_inputs, target_inputs, source_mask, target_mask = convert_batch(example_batch, pad_idx=0, unk_idx=0)

    # Check a couple of values in the input tensors (after transpose)
    assert source_inputs[0, 0].item() == 1
    assert target_inputs[1, 1].item() == 11

    # Check some values in the source mask - should be false for padding, true otherwise
    assert source_mask[2, 0, 0].item() is False
    assert source_mask[0, 0, 0].item() is True
    assert source_mask[2, 0, 1].item() is False
    assert source_mask[3, 0, 1].item() is True
