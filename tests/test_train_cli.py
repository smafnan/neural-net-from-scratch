"""Smoke tests for the ``train.py`` CLI entry point.

Runs the real CLI as a subprocess against the offline sklearn-digits fallback
(``--no-mnist``) with a tiny architecture/epoch count so this stays fast and
never touches the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_train(*extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "train.py"), "--no-mnist",
         "--epochs", "1", "--hidden", "4", *extra_args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )


def test_train_cli_trains_and_saves_model(tmp_path):
    output_dir = tmp_path / "reports"
    result = _run_train("--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    assert (output_dir / "model.npz").exists()
    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "confusion.png").exists()
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["final_train_loss"] is not None


def test_train_cli_load_model_reuses_saved_weights(tmp_path):
    output_dir = tmp_path / "reports"
    trained = _run_train("--output-dir", str(output_dir))
    assert trained.returncode == 0, trained.stderr
    trained_acc = json.loads((output_dir / "metrics.json").read_text())["test_accuracy"]

    loaded = _run_train("--output-dir", str(output_dir), "--load-model")
    assert loaded.returncode == 0, loaded.stderr
    assert "Loading trained weights" in loaded.stdout

    loaded_metrics = json.loads((output_dir / "metrics.json").read_text())
    assert loaded_metrics["test_accuracy"] == trained_acc
    assert loaded_metrics["final_train_loss"] is None  # no training happened
