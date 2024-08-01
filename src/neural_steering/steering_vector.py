import contextlib

import torch


class SteeringVector:
    def __init__(self, value, src_layer, method_name):
        self.value = value
        self.src_layer = src_layer
        self.method_name = method_name
        self._coef = 1

    def _steering_hooks(self):
        raise NotImplementedError("Subclass must implement _steering_hooks")

    @contextlib.contextmanager
    def steer(self, model):
        hooks = self._steering_hooks()
        try:
            with model.hooks(fwd_hooks=hooks):
                yield
        finally:
            pass

    def __and__(self, other):
        return CombinationSteering([self, other])

    def as_method(self, method_type):
        return method_type._from_steering_vector(self)

    def _norm(self):
        return torch.norm(self.value)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"layer={self.src_layer}, "
            f"norm={self._norm():.3f}, "
            f"method={self.method_name}"
            f")"
        )


class AdditiveSteering(SteeringVector):
    def _steering_hooks(self):
        def add_steering_vector(activations, hook):
            return activations + self._coef * self.value.to(activations.dtype)

        return [(f"blocks.{self.src_layer}.hook_resid_post", add_steering_vector)]

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError("Steering vector can only be multiplied by a scalar")
        new_vector = AdditiveSteering(self.value, self.src_layer, self.method_name)
        new_vector._coef = self._coef * other
        return new_vector

    def __rmul__(self, other):
        return self.__mul__(other)

    def combine(self, other):
        if self.src_layer != other.src_layer:
            raise ValueError("Steering vectors must have the same source layer")
        combined_value = self.value + other.value
        avg_norm = (self._norm() + other._norm()) / 2

        combined_norm = torch.norm(combined_value)
        if combined_norm > 0:
            combined_value = combined_value * (avg_norm / combined_norm)

        vector = self.__class__(combined_value, self.src_layer, "Combined")
        vector._coef = self._coef
        return vector

    def _norm(self):
        return torch.norm(self._coef * self.value)

    @classmethod
    def _from_steering_vector(cls, vector):
        return cls(vector.value, vector.src_layer, vector.method_name)


class AblationSteering(SteeringVector):
    def _steering_hooks(self):
        normalized = self.value / torch.norm(self.value)

        def hook_selector(hook_name):
            return any(k in hook_name for k in ["mlp_out", "attn_out"])

        def orthonalize(activations, hook):
            vec = normalized.to(activations.dtype)
            dots = torch.einsum("...k,k->...", activations, vec)
            orts = torch.einsum("...,k->...k", dots, vec)
            activations -= orts
            return activations

        return [(hook_selector, orthonalize)]

    @classmethod
    def _from_steering_vector(cls, vector):
        return cls(vector.value, vector.src_layer, vector.method_name)


class CombinationSteering(SteeringVector):
    def __init__(self, vectors):
        self.vectors = vectors

    def _steering_hooks(self):
        return sum((vector._steering_hooks() for vector in self.vectors), [])

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{'\n'.join('  ' + str(v) for v in self.vectors)}\n)"
