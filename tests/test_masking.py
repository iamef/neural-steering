import torch
from neural_steering.utils import create_steering_mask

def test_create_steering_mask():
    # Create a mock encoding
    encoding = type('MockEncoding', (), {})()
    encoding.attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ])
    encoding.input_ids = torch.zeros_like(encoding.attention_mask)

    # Test case 1: Positive indices
    mask = create_steering_mask(encoding, slice(1, 3))
    expected = torch.tensor([
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0]
    ], dtype=torch.bool)
    assert torch.all(mask == expected)

    # Test case 2: Negative indices
    mask = create_steering_mask(encoding, slice(-2, None))
    expected = torch.tensor([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1]
    ], dtype=torch.bool)
    assert torch.all(mask == expected)

    # Test case 3: None as stop index
    mask = create_steering_mask(encoding, slice(2, None))
    expected = torch.tensor([
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1]
    ], dtype=torch.bool)
    assert torch.all(mask == expected)

    # Test case 4: None as start index
    mask = create_steering_mask(encoding, slice(None, 3))
    expected = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0]
    ], dtype=torch.bool)
    assert torch.all(mask == expected)

    # Test case 5: Single index
    mask = create_steering_mask(encoding, slice(2, 3))
    expected = torch.tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype=torch.bool)
    assert torch.all(mask == expected)
