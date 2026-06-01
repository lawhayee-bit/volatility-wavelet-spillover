from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west_variance(values: np.ndarray, lag: int) -> float:
    centered = values - np.nanmean(values)
    n = len(centered)
    gamma0 = np.dot(centered, centered) / n
    variance = gamma0
    for k in range(1, lag + 1):
        gamma = np.dot(centered[k:], centered[:-k]) / n
        weight = 1.0 - (k / (lag + 1.0))
        variance += 2.0 * weight * gamma
    return variance


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> dict[str, float]:
    loss_a = np.asarray(loss_a, dtype=float)
    loss_b = np.asarray(loss_b, dtype=float)
    diff = loss_a - loss_b
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 5:
        return {"dm_stat": np.nan, "p_value": np.nan}
    lag = max(horizon - 1, 0)
    long_var = _newey_west_variance(diff, lag)
    dm_stat = np.nanmean(diff) / np.sqrt(long_var / n)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def clark_west(
    y_true: np.ndarray,
    pred_small: np.ndarray,
    pred_large: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pred_small = np.asarray(pred_small, dtype=float)
    pred_large = np.asarray(pred_large, dtype=float)

    adjustment = (y_true - pred_small) ** 2 - ((y_true - pred_large) ** 2 - (pred_small - pred_large) ** 2)
    adjustment = adjustment[np.isfinite(adjustment)]
    n = len(adjustment)
    if n < 5:
        return {"cw_stat": np.nan, "p_value": np.nan}
    mean_adj = np.nanmean(adjustment)
    std_adj = np.nanstd(adjustment, ddof=1) / np.sqrt(n)
    cw_stat = mean_adj / std_adj
    p_value = 1.0 - stats.norm.cdf(cw_stat)
    return {"cw_stat": float(cw_stat), "p_value": float(p_value)}
