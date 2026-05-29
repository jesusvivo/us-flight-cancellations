"""Convergence + regularization tests for the from-scratch LR.

No Spark dependency: synthetic 2D dataset, pure numpy. Lets us verify the port
is mathematically correct independently of the rest of the pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.logistic_regression import LogisticRegression


def _make_linearly_separable(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs in 2D with a clear linear margin."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(n // 2, 2))
    neg = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(n // 2, 2))
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    return X, y


def test_fit_converges_on_separable_data():
    X, y = _make_linearly_separable()
    model = LogisticRegression(learning_rate=0.5, n_iterations=300, lambda_value=0.0)
    model.fit(X, y)

    losses = model.loss_history_
    assert len(losses) == 300
    # Loss must strictly drop from the start of training to the end.
    assert losses[-1] < losses[0]
    # And it should be close to zero on cleanly separable data.
    assert losses[-1] < 0.1


def test_predict_recovers_separable_labels():
    X, y = _make_linearly_separable()
    model = LogisticRegression(learning_rate=0.5, n_iterations=300, lambda_value=0.0).fit(X, y)
    accuracy = (model.predict(X) == y).mean()
    assert accuracy > 0.95


def test_strong_regularization_shrinks_weights():
    X, y = _make_linearly_separable()
    weak = LogisticRegression(learning_rate=0.1, n_iterations=200, lambda_value=0.0).fit(X, y)
    strong = LogisticRegression(learning_rate=0.1, n_iterations=200, lambda_value=10.0).fit(X, y)
    assert np.linalg.norm(strong.weights) < np.linalg.norm(weak.weights)


def test_predict_proba_returns_probabilities():
    X, y = _make_linearly_separable(n=20)
    model = LogisticRegression(learning_rate=0.5, n_iterations=100).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (20,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_predict_before_fit_errors():
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(np.zeros((2, 2)))
