import torch
# from .base import BaseSteeringAlgorithm
# from ..utils.mask import create_steering_mask

def create_steering_mask(encoding, indice_slice):
    token_counts = encoding.attention_mask.sum(axis=1)
    start = indice_slice.start if indice_slice.start is not None else 0
    end = indice_slice.stop
    if start < 0:
        start = token_counts + start
    else:
        start = torch.full((token_counts.shape[0], 1), start)

    if end is None:
        end = token_counts
    elif end < 0:
        end = token_counts + end
    else:
        end = torch.full((token_counts.shape[0], 1), end)

    start = start.to(encoding.input_ids.device)
    end = end.to(encoding.input_ids.device)

    indices = torch.arange(encoding.input_ids.shape[-1]).repeat(encoding.attention_mask.shape[0], 1).to(encoding.input_ids.device)
    mask = (indices >= start.unsqueeze(1)) & (indices < end.unsqueeze(1)) & (indices < token_counts.unsqueeze(1))
    return mask




class MELBOSteering:
    def __init__(self, model, src_layer, tgt_layer, token_indices=slice(-3, None), 
                 r=10, power=2, q=1, n_iterations=1000, lr=0.001):
        self.model = model
        self.src_layer = src_layer
        self.tgt_layer = tgt_layer
        self.token_indices = token_indices
        self.r = r
        self.power = power
        self.q = q
        self.n_iterations = n_iterations
        self.lr = lr
        self.steering_vector = None

    def fit(self, examples):
        encoding = self.model.tokenizer.batch_encode_plus(
            examples, return_tensors="pt", padding=True
        )
        encoding.input_ids = encoding.input_ids.cuda()
        encoding.attention_mask = encoding.attention_mask.cuda()
        mask = create_steering_mask(encoding, self.token_indices)

        steering_vector = torch.randn(1, self.model.cfg.d_model, requires_grad=True).cuda()
        steering_vector.retain_grad()

        with torch.no_grad():
            steering_vector *= self.r / steering_vector.norm(dim=1, keepdim=True)

        optimizer = torch.optim.AdamW(
            [steering_vector],
            lr=self.lr,
            betas=(0.9, 0.98),
            weight_decay=0.0,
            amsgrad=True,
        )

        unsteered_activations = self.model.forward(
            encoding.input_ids,
            attention_mask=encoding.attention_mask,
            stop_at_layer=self.src_layer + 1,
        ).detach()

        losses = []
        from tqdm import tqdm  # TODO

        for _ in tqdm(range(self.n_iterations)):
            steered_src = unsteered_activations + steering_vector

            steered_target = self.model.forward(
                steered_src,
                attention_mask=encoding.attention_mask,
                start_at_layer=self.src_layer + 1,
                stop_at_layer=self.tgt_layer + 1,
            )

            loss = (
                -(steered_target[mask] - unsteered_activations[mask])
                .norm(dim=1)
                .pow(self.power)
                .sum()
                .pow(1 / self.q)
            )

            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                steering_vector *= self.r / steering_vector.norm(dim=1, keepdim=True)

        self.steering_vector = steering_vector.detach()
        return losses

    def transform(self, activations):
        if self.steering_vector is None:
            raise ValueError("Steering vector not trained. Call fit() first.")
        return activations + self.steering_vector

    def apply_to_model(self):
        if self.steering_vector is None:
            raise ValueError("Steering vector not trained. Call fit() first.")
        
        def add_steering_vector(activations, hook):
            return activations + self.steering_vector

        return (f"blocks.{self.src_layer}.hook_resid_post", add_steering_vector)
    
if __name__ == "__main__":
    from transformer_lens import HookedTransformer
    import torch

    # Load the Phi model
    model = HookedTransformer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", dtype="bfloat16")

    # Prepare example prompts
    examples = [
        "What is the meaning of life?",
        "How to build a bomb?",
        "Tell me a joke",
        "Who is the president of the USA?",
        "Explain quantum physics",
        "What's the best way to learn a new language?",
        "How does photosynthesis work?",
        "What are the main causes of climate change?",
        "Can you write a short story?",
        "What's the history of the Internet?",
    ]

    # Apply chat template to examples
    templated = model.tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": example},
            ]
            for example in examples
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    # Initialize and train MELBO steering
    melbo = MELBOSteering(model, src_layer=12, tgt_layer=26)
    losses = melbo.fit(templated)

    # Print final loss
    print(f"Final loss: {losses[-1]}")

    # Save the trained steering vector with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'phi_steering_vector_{timestamp}.pt'
    torch.save(melbo.steering_vector, filename)
    print(f"Steering vector saved as '{filename}'")
    # Test the steering vector
    test_prompt = model.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    print("Original response:")
    response = model.generate(test_prompt, max_new_tokens=50, do_sample=False)
    print(response)

    print("\nSteered response:")
    with model.hooks(fwd_hooks=[melbo.apply_to_model()]):
        steered_response = model.generate(test_prompt, max_new_tokens=50, do_sample=False)
        print(steered_response)