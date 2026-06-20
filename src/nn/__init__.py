"""nn - a tiny neural-network library in pure NumPy, with backprop by hand."""

from .layers import Dense, ReLU
from .losses import SoftmaxCrossEntropy, softmax
from .network import MLP
from .data import load_dataset, one_hot

__all__ = [
    "Dense", "ReLU",
    "SoftmaxCrossEntropy", "softmax",
    "MLP",
    "load_dataset", "one_hot",
]
__version__ = "1.0.0"
