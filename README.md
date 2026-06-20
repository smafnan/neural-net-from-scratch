# Neural Network From Scratch — Backprop in Pure NumPy (MNIST)

> **AI Engineer Roadmap — Project 2.1**
> *Teaches: forward/backward pass, the chain rule made concrete, why initialization and learning rate matter.*
> *Done when: you can hand-trace one backprop step on paper.*

A small feed-forward neural network — **Dense layers, ReLU, softmax
cross-entropy, mini-batch SGD** — with the forward and backward passes written by
hand in NumPy. No PyTorch, no TensorFlow, no autograd. It trains to **97.7% test
accuracy on MNIST**.

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -e ".[dev]"
python train.py            # downloads MNIST, trains, writes reports/
python train.py --no-mnist # use the bundled sklearn digits set (no download)
pytest -q                  # 7 tests, including gradient checking
```

---

## Result

Trained `784 → 128 → 64 → 10` for 20 epochs (lr 0.1, batch 128) on MNIST
(56k train / 14k test):

| Dataset | Architecture | Test accuracy |
| --- | --- | ---: |
| MNIST | 784–128–64–10 | **0.977** |
| sklearn digits (fallback) | 64–128–64–10 | 0.947 |

![Training curves](reports/training_curves.png)

The loss falls smoothly and validation accuracy climbs then plateaus — exactly
the healthy shape you want, and proof the hand-written backward pass is doing the
right thing.

![Confusion matrix](reports/confusion.png)
![Misclassified examples](reports/mistakes.png)

The mistakes are the *human-plausible* ones (sloppy 4s read as 9s, etc.) — a good
sign the model learned real structure rather than noise.

---

## How backprop works here

Every layer implements the same contract:

```python
forward(x)            -> output          # caches what backward needs
backward(grad_output) -> grad_input      # and stores parameter gradients
```

`grad_output` is `dL/d(output)` arriving from the next layer. Each layer
multiplies by its **local derivative** (the chain rule) to produce `dL/d(input)`
for the previous layer. One reverse sweep fills every gradient — that *is*
backpropagation.

**Dense layer** (`y = xW + b`):

```python
dW = x.T @ grad_output      # dL/dW
db = grad_output.sum(0)     # dL/db
dx = grad_output @ W.T      # dL/dx  -> handed to the previous layer
```

**ReLU**: gradient passes through where the input was positive, blocked elsewhere.

**Softmax + cross-entropy (fused)** gives the famous clean gradient:

```
dL/dlogits = (softmax(logits) - y_onehot) / n     #  "prediction minus truth"
```

### Hand-trace one backprop step (the "Done when")

One example, 2-class head, logits `z = [2.0, 0.0]`, true class `0`
(`y = [1, 0]`), batch size `n = 1`:

1. softmax: `p = [e²,e⁰]/(e²+e⁰) = [0.881, 0.119]`
2. loss: `−log(0.881) = 0.127`
3. gradient at logits: `(p − y)/n = [0.881−1, 0.119−0] = [−0.119, 0.119]`
4. into the last Dense layer: `dW = aᵀ·[−0.119, 0.119]` where `a` is that layer's
   input; `db = [−0.119, 0.119]`; pass `[−0.119,0.119]·Wᵀ` further back.
5. SGD: `W -= lr·dW`.

The negative gradient on the true class pushes its logit **up**; the positive
gradient on the wrong class pushes its logit **down** — which is exactly what we
want. `test_gradient_check_dense_parameters` confirms these analytical gradients
match finite-difference numerical gradients to ~1e-6.

### Why initialisation and learning rate matter

- **He initialisation** (`Var = 2/fan_in`) keeps the signal and gradient
  magnitudes stable across ReLU layers. Initialise too large and activations
  saturate / explode; too small and the gradient vanishes and learning stalls.
- **Learning rate** sets the step size down the gradient. Too high overshoots and
  the loss diverges; too low and training crawls. The smooth loss curve above is
  what a well-chosen lr looks like.

---

## Correctness: gradient checking

The decisive test (`tests/test_nn.py`) perturbs each parameter by ±ε and compares
the finite-difference estimate `(L(θ+ε) − L(θ−ε)) / 2ε` against the analytical
gradient from backprop. They agree to **~1e-6**. There is also a test that the
network can **overfit a tiny batch to 100% accuracy** — if it couldn't, learning
would be broken.

## Layout

```
src/nn/
├── layers.py     # Dense (with He init) and ReLU; forward + backward
├── losses.py     # fused softmax cross-entropy + stable softmax
├── network.py    # MLP: forward/backward orchestration + mini-batch SGD
└── data.py       # MNIST via OpenML, with sklearn-digits fallback
train.py          # train + figures + metrics
tests/test_nn.py  # 7 tests incl. gradient checking and overfit sanity
```

## License

MIT. MNIST is a public benchmark dataset.
