import torch
from jaxtyping import Bool
from transformers import BatchEncoding


def create_steering_mask(
    encoding: BatchEncoding, indice_slice: slice
) -> Bool[torch.Tensor, "batch sequence"]:
    """
    Creates a mask for steering based on the given encoding and index slice.

    This function generates a boolean mask that can be used to select
    specific token positions in a sequence. It handles both positive and
    negative indexing, as well as None values in the slice.

    Args:
        encoding: The encoding object containing input_ids and attention_mask.
        indice_slice: A slice object specifying the range of indices to include in the mask.

    Returns:
        torch.Tensor: A boolean mask tensor of shape (batch_size, sequence_length)
        where True values indicate positions to apply steering.

    Note:
        This function assumes that the encoding object has 'input_ids' and 'attention_mask'
        attributes, and that these are PyTorch tensors.
    """
    token_counts = encoding.attention_mask.sum(axis=1)
    start = indice_slice.start if indice_slice.start is not None else 0
    end = indice_slice.stop
    if start < 0:
        start = token_counts + start
    else:
        start = torch.full((token_counts.shape[0],), start)

    if end is None:
        end = token_counts
    elif end < 0:
        end = token_counts + end
    else:
        end = torch.full((token_counts.shape[0], 1), end)

    start = start.to(encoding.input_ids.device)
    end = end.to(encoding.input_ids.device)

    indices = (
        torch.arange(encoding.input_ids.shape[-1])
        .repeat(encoding.attention_mask.shape[0], 1)
        .to(encoding.input_ids.device)
    )
    mask = (
        (indices >= start.unsqueeze(1))
        & (indices < end.unsqueeze(1))
        & (indices < token_counts.unsqueeze(1))
    )

    return mask
