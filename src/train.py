"""Training: from-scratch LR (with grid-search tuning) + PySpark MLlib LR comparison."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
from pyspark.ml.classification import LogisticRegression as MLlibLR
from pyspark.ml.classification import LogisticRegressionModel as MLlibLRModel
from pyspark.sql import DataFrame
from sklearn.metrics import f1_score

from src import config
from src.logistic_regression import LogisticRegression

logger = logging.getLogger(__name__)


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    combinations: Iterable[dict],
) -> tuple[dict, float]:
    """Train + score one custom LR per combination; return the best params + macro F1."""
    best_params: dict | None = None
    best_score = -1.0
    for params in combinations:
        model = LogisticRegression(**params).fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), average="macro")
        logger.info("  grid: %s -> F1=%.4f", params, score)
        if score > best_score:
            best_score = float(score)
            best_params = dict(params)
    assert best_params is not None, "grid_search received no combinations"
    return best_params, best_score


def two_phase_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict, float]:
    """Mirror the original notebook's two-step tuning: first (learning_rate, n_iterations)
    holding lambda fixed at 1.0, then sweep lambda holding the winners fixed."""
    logger.info("Phase 1: tuning learning_rate x n_iterations (lambda=1.0)...")
    phase1 = [
        {"learning_rate": lr, "n_iterations": n, "lambda_value": 1.0}
        for lr in config.LEARNING_RATES
        for n in config.N_ITERATIONS_GRID
    ]
    best_phase1, _ = grid_search(X_train, y_train, X_val, y_val, phase1)

    logger.info("Phase 2: tuning lambda with %s fixed...", best_phase1)
    phase2 = [{**best_phase1, "lambda_value": lam} for lam in config.LAMBDA_GRID]
    return grid_search(X_train, y_train, X_val, y_val, phase2)


def fit_custom_lr(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> LogisticRegression:
    return LogisticRegression(**params).fit(X_train, y_train)


def fit_mllib_lr(train_df: DataFrame) -> MLlibLRModel:
    lr = MLlibLR(
        featuresCol="features",
        labelCol=config.TARGET_COL,
        maxIter=config.MLLIB_MAX_ITER,
    )
    return lr.fit(train_df)
