"""Train the from-scratch MLP on MNIST (or the digits fallback).

    python train.py [--epochs 20] [--lr 0.1] [--hidden 128 64] [--no-mnist]

Writes training curves, a confusion matrix, and a grid of misclassified images
to reports/, plus a metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.nn import Dense, ReLU, MLP, load_dataset, one_hot


def build_mlp(n_features: int, hidden: list[int], n_classes: int) -> MLP:
    """Stack Dense+ReLU hidden layers, then a final Dense to the classes."""
    layers: list = []
    prev = n_features
    for i, h in enumerate(hidden):
        layers.append(Dense(prev, h, seed=i + 1))
        layers.append(ReLU())
        prev = h
    layers.append(Dense(prev, n_classes, seed=99))
    return MLP(layers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--no-mnist", action="store_true",
                        help="Skip the MNIST download and use sklearn digits.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1) Loading data ...")
    X_train, X_test, y_train, y_test, meta = load_dataset(prefer_mnist=not args.no_mnist)
    print(f"   dataset={meta['name']}  features={meta['n_features']}  "
          f"classes={meta['n_classes']}  train={len(X_train)}  test={len(X_test)}")

    y_train_oh = one_hot(y_train, meta["n_classes"])

    print(f"2) Training MLP {[meta['n_features'], *args.hidden, meta['n_classes']]} ...")
    net = build_mlp(meta["n_features"], args.hidden, meta["n_classes"])
    net.fit(X_train, y_train_oh, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, X_val=X_test, y_val=y_test, verbose=True)

    test_acc = net.score(X_test, y_test)
    print(f"\n3) Final test accuracy: {test_acc:.4f}")

    # --- figures --------------------------------------------------------- #
    _plot_curves(net, args.output_dir / "training_curves.png")
    y_pred = net.predict(X_test)
    _plot_confusion(y_test, y_pred, meta["n_classes"],
                    args.output_dir / "confusion.png")
    _plot_mistakes(X_test, y_test, y_pred, meta["image_shape"],
                   args.output_dir / "mistakes.png")

    (args.output_dir / "metrics.json").write_text(json.dumps({
        "dataset": meta["name"],
        "architecture": [meta["n_features"], *args.hidden, meta["n_classes"]],
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "test_accuracy": float(test_acc),
        "final_train_loss": net.history_["loss"][-1],
    }, indent=2), encoding="utf-8")
    print(f"   Figures + metrics written to {args.output_dir}/")
    return 0


def _plot_curves(net: MLP, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(net.history_["loss"], color="steelblue", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="steelblue")
    if net.history_["val_acc"]:
        ax2 = ax1.twinx()
        ax2.plot(net.history_["val_acc"], color="darkorange", label="val acc")
        ax2.set_ylabel("val accuracy", color="darkorange")
    ax1.set_title("Training loss and validation accuracy")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_confusion(y_true, y_pred, n_classes, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_mistakes(X, y_true, y_pred, image_shape, path: Path) -> None:
    wrong = np.where(y_pred != y_true)[0][:15]
    if len(wrong) == 0:
        return
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(X[idx].reshape(image_shape), cmap="gray")
        ax.set_title(f"t{y_true[idx]} p{y_pred[idx]}", fontsize=8)
        ax.axis("off")
    for ax in np.atleast_1d(axes).ravel()[len(wrong):]:
        ax.axis("off")
    fig.suptitle("Misclassified examples (t=true, p=pred)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
