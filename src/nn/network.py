"""The MLP: chains layers, runs forward/backward, and trains with mini-batch SGD.

Forward pass:  push x through every layer in order.
Backward pass: push the loss gradient through every layer in *reverse* order,
               each layer turning dL/d(its output) into dL/d(its input). This is
               backpropagation: one reverse sweep that fills every parameter's
               gradient via the chain rule.
SGD update:    param -= lr * grad, for every (param, grad) the layers expose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .losses import SoftmaxCrossEntropy, softmax


class MLP:
    """A simple feed-forward classifier."""

    def __init__(self, layers: list, seed: int = 0) -> None:
        self.layers = layers
        self.loss_fn = SoftmaxCrossEntropy()
        self._rng = np.random.default_rng(seed)
        self.history_: dict[str, list[float]] = {"loss": [], "val_acc": []}

    # --- core passes --------------------------------------------------- #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits (no softmax — the loss handles that)."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        """Propagate the loss gradient back through every layer in reverse."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _sgd_step(self, lr: float) -> None:
        for layer in self.layers:
            for param, grad in layer.params_and_grads():
                param -= lr * grad  # in-place update

    # --- training ------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y_onehot: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 0.1,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> "MLP":
        n = X.shape[0]
        for epoch in range(epochs):
            # Shuffle each epoch so mini-batches differ -> better SGD.
            perm = self._rng.permutation(n)
            Xs, ys = X[perm], y_onehot[perm]

            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                xb = Xs[start:start + batch_size]
                yb = ys[start:start + batch_size]

                logits = self.forward(xb)
                loss = self.loss_fn.forward(logits, yb)
                grad = self.loss_fn.backward()
                self.backward(grad)
                self._sgd_step(lr)
                epoch_loss += loss * len(xb)

            epoch_loss /= n
            self.history_["loss"].append(epoch_loss)

            msg = f"epoch {epoch + 1:3d}/{epochs}  loss={epoch_loss:.4f}"
            if X_val is not None and y_val is not None:
                acc = self.score(X_val, y_val)
                self.history_["val_acc"].append(acc)
                msg += f"  val_acc={acc:.4f}"
            if verbose:
                print(msg)
        return self

    # --- inference ----------------------------------------------------- #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def score(self, X: np.ndarray, y_labels: np.ndarray) -> float:
        """Accuracy. ``y_labels`` are integer class indices (not one-hot)."""
        return float(np.mean(self.predict(X) == y_labels))

    # --- persistence ----------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        """Save every Dense layer's ``W``/``b`` to ``path`` as an ``.npz`` archive.

        Also stores the layer widths so :meth:`load` can rebuild the same
        ``Dense -> ReLU -> ... -> Dense`` stack without needing the original
        construction code (e.g. ``train.build_mlp``).
        """
        arrays: dict[str, np.ndarray] = {}
        widths = []
        dense_idx = 0
        for layer in self.layers:
            if hasattr(layer, "W"):
                arrays[f"W{dense_idx}"] = layer.W
                arrays[f"b{dense_idx}"] = layer.b
                widths.append(layer.W.shape[0])
                dense_idx += 1
        widths.append(self.layers[-1].b.shape[0])  # trailing output width
        arrays["widths"] = np.array(widths, dtype=np.int64)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        """Rebuild an :class:`MLP` from a file written by :meth:`save`.

        Assumes the ``Dense -> ReLU -> ... -> Dense`` stack shape used
        throughout this project (a ``ReLU`` after every ``Dense`` except the
        last).
        """
        from .layers import Dense, ReLU  # local import: layers doesn't import network

        with np.load(path) as data:
            widths = [int(w) for w in data["widths"]]
            n_dense = len(widths) - 1
            layers: list = []
            for i in range(n_dense):
                dense = Dense(widths[i], widths[i + 1])
                dense.W = data[f"W{i}"]
                dense.b = data[f"b{i}"]
                layers.append(dense)
                if i < n_dense - 1:
                    layers.append(ReLU())
        return cls(layers)
