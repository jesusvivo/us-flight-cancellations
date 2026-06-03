"""Hand-coded logistic regression with manual gradient descent and L2 regularization.

Deliberately preserved alongside `pyspark.ml.classification.LogisticRegression`
because the *point* of this project's "Statistical Methods for ML" angle was
understanding the math (sigmoid, log loss, gradient descent on weights and
bias) rather than calling a library. The pipeline trains both this and the
MLlib version and the README compares their metrics; near-parity validates the
implementation.

This is a port of the original notebook's class with four corrections:

- `predict_proba` for ROC-curve scoring (the original only had `predict`).
- A `loss_history_` attribute populated during `fit` so convergence is testable
  without re-running training.
- A real L2 contribution to the weight gradient. The original notebook added
  `lambda * ||w||^2` to the *reported* loss but never to `dW`, so `lambda_value`
  was effectively a no-op and the grid search over it was busy-work. Fixed here
  so regularization actually shrinks weights.
- The lambda grid-search bug in the original notebook (it used the loop index
  as `lambda_value` instead of the looked-up value) doesn't apply here because
  this class never executes the search itself; `train.grid_search` is
  responsible for plumbing values in correctly.
"""
from __future__ import annotations

import numpy as np


class LogisticRegression:
    """Binary logistic regression trained by full-batch gradient descent with L2.

    Parameters
    ----------
    learning_rate : float
        Step size for the weight + bias updates each iteration.
    n_iterations : int
        Number of full passes over the training data.
    lambda_value : float
        Coefficient on the L2 weight-norm penalty added to the log-loss.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        lambda_value: float = 1.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_value = lambda_value
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # Numerically guard log(0) to avoid loss spikes.
            eps = 1e-12
            log_loss = -(1.0 / m) * np.sum(
                y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps)
            )
            # Standard textbook L2: (lambda / 2m) * ||w||^2 in the loss,
            # (lambda / m) * w in the gradient. Scaling by 1/m keeps the
            # regularization comparable to the data loss across batch sizes,
            # and stable at the full lambda grid (0.001-100) the original used.
            penalty = (self.lambda_value / (2.0 * m)) * np.sum(np.square(self.weights))
            self.loss_history_.append(float(log_loss + penalty))

            # Bias is not regularized (standard practice).
            dw = (1.0 / m) * (X.T @ (y_hat - y)) + (self.lambda_value / m) * self.weights
            db = (1.0 / m) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("LogisticRegression must be fit before predict_proba.")
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)
