import torch

from src.utils.mask import make_mask, subsequent_mask


def test_subsequent_mask_shape():
    """Tests that subsequent_mask returns a tensor of the correct shape."""
    size = 5
    mask = subsequent_mask(size)
    assert mask.shape == (1, size, size)


def test_subsequent_mask_values():
    """Tests that subsequent_mask creates the correct triangular mask."""
    size = 4
    mask = subsequent_mask(size)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    expected_mask = torch.tensor(
        [[[True, False, False, False], [True, True, False, False], [True, True, True, False], [True, True, True, True]]]
    )

    expected_mask = expected_mask.to(DEVICE)
    assert torch.equal(mask, expected_mask)


def test_subsequent_mask_device():
    """Tests that subsequent_mask creates the mask on the correct device (CPU or CUDA)."""
    size = 3
    mask = subsequent_mask(size)
    if torch.cuda.is_available():
        assert mask.device.type == "cuda"
    else:
        assert mask.device.type == "cpu"


def test_make_mask_shapes():
    """Tests that make_mask returns masks of the correct shapes."""
    batch_size = 2
    source_seq_len = 10
    target_seq_len = 8
    pad_idx = 0
    unk_idx = 1

    source_inputs = torch.randint(1, 100, (batch_size, source_seq_len))
    target_inputs = torch.randint(1, 100, (batch_size, target_seq_len))

    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, unk_idx)

    assert source_mask.shape == (batch_size, 1, source_seq_len)
    assert target_mask.shape == (batch_size, target_seq_len, target_seq_len)


def test_make_mask_padding():
    """Tests that make_mask correctly masks padding tokens."""
    pad_idx = 0
    unk_idx = 0

    source_inputs = torch.tensor([[1, 2, 0, 4, 0]])  # 0 is padding
    target_inputs = torch.tensor([[5, 0, 7, 0]])  # 0 is padding

    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, unk_idx)

    assert source_mask[0, 0, 2].item() is False  # Padding should be False
    assert source_mask[0, 0, 4].item() is False  # Padding should be False
    assert target_mask[0, 0, 1].item() is False  # Padding should be False
    assert target_mask[0, 0, 3].item() is False  # Padding should be False

    assert source_mask[0, 0, 0].item() is True  # Non-padding should be True
    assert source_mask[0, 0, 1].item() is True  # Non-padding should be True
    assert source_mask[0, 0, 3].item() is True  # Non-padding should be True
    assert target_mask[0, 0, 0].item() is True  # Non-padding should be True
    assert target_mask[0, 0, 2].item() is False  # Non-padding should be True


def test_make_mask_subsequent():
    """Tests that make_mask includes the subsequent mask."""
    batch_size = 1
    target_seq_len = 4
    pad_idx = 0
    unk_idx = 1

    source_inputs = torch.randint(1, 100, (batch_size, 5))  # Dummy source inputs
    target_inputs = torch.randint(1, 100, (batch_size, target_seq_len))

    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, unk_idx)

    # Checks that the upper triangle of target_mask (future) is False
    expected_false = torch.zeros(target_seq_len - 1, dtype=torch.bool)  # Tensor of False
    assert (target_mask[0, 0, 1:] == expected_false).all().item() is True

    expected_false = torch.zeros(target_seq_len - 2, dtype=torch.bool)
    assert (target_mask[0, 1, 2:] == expected_false).all().item() is True

    expected_false = torch.zeros(target_seq_len - 3, dtype=torch.bool)
    assert (target_mask[0, 2, 3:] == expected_false).all().item() is True


def test_make_mask_no_padding():
    """Tests that make_mask works correctly when there is no padding."""
    batch_size = 1
    source_seq_len = 5
    target_seq_len = 4
    pad_idx = -1
    unk_idx = 1

    source_inputs = torch.randint(1, 100, (batch_size, source_seq_len))
    target_inputs = torch.randint(1, 100, (batch_size, target_seq_len))

    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, unk_idx)

    expected_mask = subsequent_mask(target_seq_len).type_as(target_mask)
    assert (target_mask[0] == expected_mask).all()
    assert source_mask.all().item() is True


def test_make_mask_different_lengths():
    """Tests that make_mask works correctly with different source and target sequence lengths."""
    batch_size = 2
    source_seq_len = 7
    target_seq_len = 5
    pad_idx = 0
    unk_idx = 1

    source_inputs = torch.randint(1, 100, (batch_size, source_seq_len))
    target_inputs = torch.randint(1, 100, (batch_size, target_seq_len))

    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx, unk_idx)

    assert source_mask.shape == (batch_size, 1, source_seq_len)
    assert target_mask.shape == (batch_size, target_seq_len, target_seq_len)
