import torch
import pytest
from unittest.mock import Mock
from transformer_lens import HookedTransformer, HookedTransformerConfig
from neural_steering.melbo import MELBO
from neural_steering.steering_vector import AdditiveSteering


@pytest.fixture
def small_transformer():
    config = HookedTransformerConfig(
        n_layers=2,
        n_heads=1,
        n_ctx=8,
        d_model=12,
        d_head=12,
        act_fn="relu",
        d_vocab=12,
        tokenizer_prepends_bos=True,
        device="cpu",
    )

    model = HookedTransformer(config)

    # Create a mock tokenizer
    mock_tokenizer = Mock()

    class MockEncoding:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def to(self, device):
            return self

    mock_tokenizer.batch_encode_plus.return_value = MockEncoding(
        input_ids=torch.randint(0, 12, (3, 8)), attention_mask=torch.ones(3, 8)
    )
    model.tokenizer = mock_tokenizer

    return model


def test_melbo_steering(small_transformer):
    # Define test prompts
    prompts = [
        "Once upon a time",
        "In a galaxy far, far away",
        "It was a dark and stormy night",
    ]

    # Initialize MELBOSteering
    target_norm = 2.5
    melbo = MELBO(
        model=small_transformer,
        src_layer=0,
        tgt_layer=1,
        token_indices=slice(-3, None),
        n_iterations=100,
        r=target_norm,
    )

    # Fit the model
    result = melbo.fit(prompts, n_vectors=2, verbose=False)

    # Check the result
    assert isinstance(result.steering_vectors, list)
    assert len(result.steering_vectors) == 2
    assert all(isinstance(sv, list) for sv in result.steering_vectors)
    assert all(
        isinstance(v, AdditiveSteering) for sv in result.steering_vectors for v in sv
    )

    # Check the losses
    assert isinstance(result.losses, torch.Tensor)
    assert result.losses.shape == (
        len(prompts),
        2,
        100,
    )  # 3 prompts, 2 vectors, 100 iterations

    vectors_torch = torch.stack(
        [torch.vstack([s.value for s in sv]) for sv in result.steering_vectors]
    ).swapaxes(0, 1)
    # Check norms
    vector_norms = torch.norm(vectors_torch, dim=-1).flatten()
    eps = 1e-6
    assert torch.all(vector_norms <= target_norm + eps)

    # Check orthogonality

    inner_prod = (
        torch.einsum("nik,njk->nij", vectors_torch, vectors_torch) / target_norm**2
    )
    assert torch.allclose(inner_prod, torch.eye(2), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
