from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, None)
    y_true = np.clip(y_true, eps, None)
    return np.log(y_pred) + (y_true / y_pred)


def regression_metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    loss = qlike_loss(y_true, y_pred)
    return {
        "qlike": float(np.nanmean(loss)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metric_bundle(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_label: np.ndarray,
) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_label, zero_division=0)),
        "precision": float(precision_score(y_true, y_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_label, zero_division=0)),
    }


def inverse_target_transform(values: np.ndarray, transform: str, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.expm1(values) - epsilon
    if transform == "log":
        return np.exp(values) - epsilon
    return values


def summarise_by_keys(df, group_keys: list[str], metric_cols: list[str]):
    return df.groupby(group_keys, dropna=False)[metric_cols].mean().reset_index()
