"""Layers — pure NumPy. Each layer knows how to go forward and backward.

The contract every layer follows:

    forward(x)  -> output;  caches whatever it needs for the backward pass
    backward(grad_output) -> grad_input;  and stashes parameter gradients

`grad_output` is dL/d(output) flowing back from the next layer. The layer applies
the chain rule to produce dL/d(input) for the previous layer, and dL/d(params)
for the optimiser. That local "receive upstream gradient, multiply by my local
derivative, pass it on" pattern *is* backpropagation.
"""

from __future__ import annotations

import numpy as np


class Dense:
    """Fully-connected layer: ``y = x @ W + b``.

    Shapes (n = batch size):
        x: (n, in_features)
        W: (in_features, out_features)
        b: (out_features,)
        y: (n, out_features)
    """

    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He initialisation: variance 2/in_features keeps signal/gradient scale
        # stable through ReLU layers (the usual cause of dead/exploding training).
        self.W = rng.normal(0, np.sqrt(2.0 / in_features),
                            size=(in_features, out_features))
        self.b = np.zeros(out_features)

        # Gradient buffers, filled during backward().
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x  # cache input; needed to compute dW in backward()
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # grad_output is dL/dy with shape (n, out_features).
        # Local derivatives via the chain rule:
        #   dL/dW = x^T @ dL/dy          (sum over the batch)
        #   dL/db = sum_n dL/dy
        #   dL/dx = dL/dy @ W^T          (pass back to previous layer)
        x = self._x
        self.dW = x.T @ grad_output
        self.db = grad_output.sum(axis=0)
        return grad_output @ self.W.T

    def params_and_grads(self):
        """Yield (param, grad) pairs so the optimiser can update in place."""
        yield self.W, self.dW
        yield self.b, self.db


class ReLU:
    """Rectified Linear Unit: ``y = max(0, x)``. No learnable parameters."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0           # remember which units were active
        return x * self._mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # d(relu)/dx is 1 where x>0 else 0 -> just gate the upstream gradient.
        return grad_output * self._mask

    def params_and_grads(self):
        return iter(())  # nothing to update
