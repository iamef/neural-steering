import collections as col
import functools as ft

import torch
from tqdm import tqdm

from .steering_vector import AdditiveSteering
from .utils import create_steering_mask


class ContrastiveSteering:
    def __init__(self, model, token_indices, selected_layers=None):
        self.model = model
        self.selected_layers = selected_layers
        self.token_indices = token_indices

    def _layer_selector(self, hook_name):
        if "resid_post" not in hook_name:
            return False
        if self.selected_layers is None:
            return True
        layer = int(hook_name.split(".")[1])
        return layer in self.selected_layers

    @torch.no_grad()
    def fit(self, positive_prompts, negative_prompts, batch_size=32, verbose=True):
        assert batch_size % 2 == 0, "Batch size must be even"
        assert len(positive_prompts) == len(
            negative_prompts
        ), "Positive and negative prompts must have the same length"

        hb = batch_size // 2
        batch_iterator = [
            positive_prompts[i : i + hb] + negative_prompts[i : i + hb]
            for i in range(0, len(positive_prompts), hb)
        ]

        diff_cache = col.defaultdict(list)

        def compute_diff(activations, hook, mask):
            masked = activations * mask[..., None]
            means = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            difference = means[: batch_size // 2] - means[batch_size // 2 :]
            diff_cache[hook.name].append(difference)

        for batch in tqdm(
            batch_iterator,
            desc="Computing contrastive steering vectors",
            disable=not verbose,
        ):
            # Tokenize prompts
            encoding = self.model.tokenizer.batch_encode_plus(
                batch, return_tensors="pt", padding=True
            )

            encoding = encoding.to(self.model.cfg.device)
            mask = create_steering_mask(encoding, self.token_indices).float()

            hooks = [(self._layer_selector, ft.partial(compute_diff, mask=mask))]
            with self.model.hooks(fwd_hooks=hooks):
                self.model.forward(
                    encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                )

        return [
            AdditiveSteering(
                torch.vstack(v).mean(dim=0),
                src_layer=int(k.split(".")[1]),
                method_name="Contrastive",
            )
            for k, v in diff_cache.items()
        ]
