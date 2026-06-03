"""Metrics + ROC-curve plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Aggregate binary-classification metrics for the README table."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def text_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, zero_division=0)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    output_path: Path,
) -> float:
    """Render and save an ROC curve. Returns the AUC."""
    auc = float(roc_auc_score(y_true, y_score))
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"{title}: AUC = {auc:.3f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return auc
