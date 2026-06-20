"""Tests for the from-scratch neural network.

The centrepiece is **gradient checking**: we compare the analytical gradients
produced by backprop against numerical gradients estimated with finite
differences. If backprop is correct, they agree to ~1e-6. This is the standard,
decisive test that a hand-written backward pass is right.
"""

from __future__ import annotations

import numpy as np
import pytest

from nn import Dense, ReLU, MLP, SoftmaxCrossEntropy, softmax, one_hot


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(logits)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p > 0)


def test_softmax_is_stable_for_large_logits():
    # Without the max-subtraction trick this overflows to nan.
    p = softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(p))


def test_relu_forward_and_backward():
    relu = ReLU()
    x = np.array([[-1.0, 2.0, -3.0, 4.0]])
    assert np.array_equal(relu.forward(x), [[0.0, 2.0, 0.0, 4.0]])
    grad = relu.backward(np.ones_like(x))
    assert np.array_equal(grad, [[0.0, 1.0, 0.0, 1.0]])


def test_softmax_xent_gradient_formula():
    # The fused gradient must equal (probs - y)/n.
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 3))
    y = one_hot(np.array([0, 1, 2, 1, 0]), 3)
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(logits, y)
    expected = (softmax(logits) - y) / 5
    assert np.allclose(loss_fn.backward(), expected)


# --------------------------------------------------------------------------- #
# Gradient checking (the decisive correctness test)
# --------------------------------------------------------------------------- #

def _loss_on(net: MLP, X, y) -> float:
    return net.loss_fn.forward(net.forward(X), y)


def test_gradient_check_dense_parameters():
    """Numerical vs analytical gradients for every weight/bias in a 2-layer net."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 5))
    y = one_hot(np.array([0, 2, 1, 3]), 4)

    net = MLP([Dense(5, 6, seed=1), ReLU(), Dense(6, 4, seed=2)])

    # Analytical gradients via one backward pass.
    net.loss_fn.forward(net.forward(X), y)
    net.backward(net.loss_fn.backward())

    eps = 1e-5
    for layer in net.layers:
        for param, grad in layer.params_and_grads():
            # Check a handful of random entries per parameter tensor.
            it = np.ndindex(param.shape)
            entries = list(it)
            rng.shuffle(entries)
            for idx in entries[:8]:
                original = param[idx]
                param[idx] = original + eps
                loss_plus = _loss_on(net, X, y)
                param[idx] = original - eps
                loss_minus = _loss_on(net, X, y)
                param[idx] = original  # restore

                numerical = (loss_plus - loss_minus) / (2 * eps)
                analytical = grad[idx]
                assert numerical == pytest.approx(analytical, abs=1e-6, rel=1e-4)


# --------------------------------------------------------------------------- #
# Learning behaviour
# --------------------------------------------------------------------------- #

def test_network_can_overfit_a_tiny_batch():
    """A correct net with enough capacity should drive train loss near zero on a
    tiny dataset. Failure here means learning is broken."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 10))
    labels = rng.integers(0, 3, size=20)
    y = one_hot(labels, 3)
    net = MLP([Dense(10, 32, seed=1), ReLU(), Dense(32, 3, seed=2)])
    net.fit(X, y, epochs=300, batch_size=20, lr=0.2, verbose=False)
    assert net.score(X, labels) == 1.0
    assert net.history_["loss"][-1] < 0.05


def test_predict_shapes():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 4))
    net = MLP([Dense(4, 5, seed=1), ReLU(), Dense(5, 3, seed=2)])
    assert net.predict(X).shape == (7,)
    assert net.predict_proba(X).shape == (7, 3)
