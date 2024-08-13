from typing import List, NamedTuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .steering_vector import AdditiveSteering, SteeringVector
from .utils import create_steering_mask


class MELBOResult(NamedTuple):
    steering_vectors: Union[List[SteeringVector], List[List[SteeringVector]]]
    losses: torch.Tensor

    def plot_losses(self):
        from matplotlib import pyplot as plt  # ignore

        num_rows = self.losses.shape[0]

        fig, axs = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows), sharex=True)
        if not isinstance(axs, np.ndarray):
            axs = [axs]

        for i in range(num_rows):
            axs[i].plot(self.losses[i].T)
            axs[i].set_title(f"Loss for prompt {i+1}")
            axs[i].set_ylabel("Loss")

        axs[-1].set_xlabel("Iterations")
        plt.tight_layout()
        plt.show()


class MELBO:
    def __init__(
        self,
        model,
        src_layer,
        tgt_layer,
        token_indices=slice(-3, None),
        q=1,
        r=12,
        lr=0.001,
        power=2,
        n_iterations=1000,
        training_mode="separate",
        adjust_gradients=False,
    ):
        self.model = model
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.token_indices = token_indices
        self.r = r
        self.power = power
        self.q = q
        self.lr = lr
        self.n_iterations = n_iterations
        self.training_mode = training_mode
        self.adjust_gradients = adjust_gradients

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def fit(self, examples, n_vectors=1, initial_vectors=None, verbose=True):
        self._freeze_model()  # TODO revert at the end
        encoding = self.model.tokenizer.batch_encode_plus(
            examples, return_tensors="pt", padding=True
        )
        encoding = encoding.to(self.model.cfg.device)
        mask = create_steering_mask(encoding, self.token_indices)

        if initial_vectors is not None:
            vectors = initial_vectors.clone()  # TODO ensure shape
        else:
            vectors = None

        all_losses = []
        steering_vectors = []

        for i in range(n_vectors):
            if vectors is None:
                projection = None
            else:
                normed = torch.nn.functional.normalize(vectors, dim=-1)
                projection = torch.einsum("nmi,nmj->nij", normed, normed)

            steering_vector, losses = self._train_single(
                encoding, mask, projection, verbose=verbose
            )
            all_losses.append(losses)

            if vectors is None:
                vectors = steering_vector.detach()
            else:
                vectors = torch.cat([vectors, steering_vector.detach()], dim=1)

            if self.training_mode == "joint":
                steering_vectors.append(
                    AdditiveSteering(steering_vector, self.src_layer, "MELBO")
                )
            else:
                steering_vectors.append(
                    [
                        AdditiveSteering(v, self.src_layer, "MELBO")
                        for v in steering_vector
                    ]
                )

        return MELBOResult(steering_vectors, torch.stack(all_losses).swapdims(0, 1))

    @torch.enable_grad()
    def _train_single(self, encoding, mask, orthogonal_projection, verbose=True):
        # joint means multiple prompts and a single steering vector
        if self.training_mode == "joint":
            steering_vector = torch.randn(
                1,
                1,
                self.model.cfg.d_model,
                requires_grad=True,
                device=self.model.cfg.device,
            )
        # 1 vector per prompt
        else:
            steering_vector = torch.randn(
                encoding.input_ids.shape[0],
                1,
                self.model.cfg.d_model,
                requires_grad=True,
                device=self.model.cfg.device,
            )
        steering_vector.retain_grad()

        with torch.no_grad():
            if orthogonal_projection is not None:
                steering_vector -= torch.einsum(
                    "nmi,nij->nmj", steering_vector, orthogonal_projection
                )
            if self.r is not None:
                steering_vector *= self.r / steering_vector.norm(dim=-1, keepdim=True)

        optimizer = torch.optim.AdamW(
            [steering_vector],
            lr=self.lr,
        )

        unsteered_activations = self.model.forward(
            encoding.input_ids,
            attention_mask=encoding.attention_mask,
            stop_at_layer=self.src_layer + 1,
        ).detach()

        losses = []

        pbar = tqdm(
            range(self.n_iterations),
            desc="Training iterations",
            disable=not verbose,
        )
        for _ in pbar:
            steered_src = unsteered_activations + steering_vector

            steered_target = self.model.forward(
                steered_src,
                attention_mask=encoding.attention_mask,
                start_at_layer=self.src_layer + 1,
                stop_at_layer=self.tgt_layer + 1,
            )

            loss = self._compute_loss(steered_target, unsteered_activations, mask)

            loss.mean().backward()
            losses.append(loss.detach())

            if self.adjust_gradients:
                self._adjust_gradients(steering_vector, orthogonal_projection)

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                if orthogonal_projection is not None:
                    # Project the steering vector onto the orthogonal complement of the subspace
                    # This ensures that the steering vector remains in the desired subspace
                    steering_vector -= torch.einsum(
                        "nmi,nij->nmj", steering_vector, orthogonal_projection
                    )
                if self.r is not None:
                    # Normalize the steering vector to maintain the hypersphere constraint
                    # This keeps the magnitude of the steering vector constant
                    steering_vector *= self.r / steering_vector.norm(
                        dim=-1, keepdim=True
                    )

            pbar.set_postfix({"loss": loss.mean().item()})

        return steering_vector, torch.stack(losses).cpu().T

    def _compute_loss(self, steered_target, unsteered_activations, mask):
        return (
            -(steered_target[mask] - unsteered_activations[mask])
            .reshape(steered_target.shape[0], -1, self.model.cfg.d_model)
            .norm(dim=-1)
            .pow(self.power)
            .sum(dim=-1)
            .pow(1 / self.q)
        )

    @torch.no_grad()
    def _adjust_gradients(self, steering_vector, orthogonal_projection):
        # If orthogonal projection is provided, project the gradient
        if orthogonal_projection is not None:
            # Remove the component of the gradient that lies in the orthogonal subspace
            # This ensures that the steering vector remains in the desired subspace
            projected_grad = torch.einsum(
                "nmi,nij->nmj", steering_vector.grad, orthogonal_projection
            )
            steering_vector.grad -= projected_grad
        # If a radius constraint is set
        if self.r is not None:
            # Calculate the radial component of the gradient
            radial_component = torch.einsum(
                "nmi,nmj->nm", steering_vector.grad, steering_vector
            )
            # Remove the radial component from the gradient to maintain the hypersphere constraint
            steering_vector.grad -= torch.einsum(
                "nm,nmj->nmj", radial_component, steering_vector
            ) / (self.r**2)


class TargetedMELBO(MELBO):
    def __init__(
        self,
        model,
        src_layer,
        tgt_layer,
        token_indices,
        starting_vector,
        **kwargs,
    ):
        super().__init__(model, src_layer, tgt_layer, token_indices, r=None, **kwargs)
        self.starting_vector = starting_vector

    def _compute_target_activation(self, example):
        assert isinstance(example, str), "Example must be a string"
        encoding = self.model.tokenizer.encode_plus(
            example, return_tensors="pt", padding=True
        )
        encoding = encoding.to(self.model.cfg.device)
        with torch.no_grad(), self.starting_vector.steer(self.model):
            activations = self.model.forward(
                encoding.input_ids,
                attention_mask=encoding.attention_mask,
                stop_at_layer=self.tgt_layer + 1,
            )

        return activations.detach()

    def fit(self, example, n_vectors=1):
        self.target_activation = self._compute_target_activation(example)
        return super().fit([example], n_vectors, self.starting_vector.value[None, None])

    def _compute_loss(self, steered_target, unsteered_activations, mask):
        return (
            steered_target[mask].mean(dim=0) - self.target_activation[mask].mean(dim=0)
        ).norm()
