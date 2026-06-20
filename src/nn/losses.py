"""Softmax + cross-entropy, fused for numerical stability and a clean gradient.

Computing softmax then cross-entropy separately is both numerically fragile
(exponentials overflow) and produces an awkward gradient. Fusing them gives the
single most elegant result in all of deep learning:

    dL/dlogits = (softmax(logits) - y_onehot) / n

i.e. "predicted probability minus the truth". That tidy form is *why* softmax +
cross-entropy is the default classification head.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max subtracted for numerical stability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Combined softmax activation + cross-entropy loss.

    forward(logits, y_onehot) -> scalar mean loss
    backward()                -> dL/dlogits, shape (n, n_classes)
    """

    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y_onehot: np.ndarray) -> float:
        self._probs = softmax(logits)
        self._y = y_onehot
        n = logits.shape[0]
        # Cross-entropy: -mean(sum_k y_k log p_k). Clip to avoid log(0).
        eps = 1e-12
        loss = -np.sum(y_onehot * np.log(self._probs + eps)) / n
        return float(loss)

    def backward(self) -> np.ndarray:
        n = self._probs.shape[0]
        return (self._probs - self._y) / n
