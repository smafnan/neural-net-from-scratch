"""Dataset loading: MNIST if available, else the smaller sklearn digits set.

MNIST (70k 28x28 images) is the canonical benchmark but needs a ~11 MB download
from OpenML. To keep the project runnable everywhere, we try MNIST first and
transparently fall back to scikit-learn's bundled ``load_digits`` (1,797 8x8
images) which ships with the library. The network code is identical either way.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer labels -> one-hot rows."""
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def load_dataset(prefer_mnist: bool = True):
    """Return ``(X_train, X_test, y_train, y_test, meta)``.

    Features are scaled to [0, 1]. ``meta`` records which dataset and the image
    shape (handy for plotting).
    """
    X = y = None
    name = "digits"
    image_shape = (8, 8)

    if prefer_mnist:
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
            X = mnist.data.astype(np.float32) / 255.0
            y = mnist.target.astype(int)
            name, image_shape = "mnist", (28, 28)
        except Exception as exc:  # network/cache failure -> fall back
            print(f"[data] MNIST unavailable ({exc}); using sklearn digits.")

    if X is None:
        digits = load_digits()
        X = digits.data.astype(np.float32) / 16.0  # digits pixels are 0..16
        y = digits.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    meta = {"name": name, "image_shape": image_shape,
            "n_classes": int(y.max() + 1), "n_features": X.shape[1]}
    return X_train, X_test, y_train, y_test, meta
