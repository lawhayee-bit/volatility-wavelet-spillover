from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt


def close_to_close_variance(close: pd.Series) -> pd.Series:
    ret = np.log(close).diff()
    return ret.pow(2)


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    return (np.log(high / low).pow(2)) / (4.0 * np.log(2.0))


def garman_klass_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    term1 = 0.5 * np.log(high / low).pow(2)
    term2 = (2.0 * np.log(2.0) - 1.0) * np.log(close / open_).pow(2)
    return term1 - term2


def rogers_satchell_variance(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    return np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_hist": hist,
        }
    )


def future_average(series: pd.Series, horizon: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    output = np.full_like(values, fill_value=np.nan, dtype=float)
    for idx in range(len(values) - horizon):
        output[idx] = np.nanmean(values[idx + 1 : idx + 1 + horizon])
    return pd.Series(output, index=series.index)


def _pad_left_to_multiple(values: np.ndarray, multiple: int) -> np.ndarray:
    if multiple <= 1:
        return values
    target_len = ceil(len(values) / multiple) * multiple
    pad_len = target_len - len(values)
    if pad_len <= 0:
        return values
    left_pad = np.repeat(values[0], pad_len)
    return np.concatenate([left_pad, values])


def causal_wavelet_features(
    series: pd.Series,
    basis: str,
    levels: int,
    min_history: int,
    max_history: int,
    energy_windows: list[int],
    prefix: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=series.index)
    values = series.to_numpy(dtype=float)
    multiple = 2**levels

    for idx in range(len(values)):
        history = values[: idx + 1]
        history = history[np.isfinite(history)]
        if len(history) < min_history:
            continue
        if len(history) > max_history:
            history = history[-max_history:]

        padded = _pad_left_to_multiple(history, multiple)
        coeffs = pywt.swt(padded, wavelet=basis, level=levels, norm=True, trim_approx=False)
        approx, details = coeffs[-1][0], [pair[1] for pair in coeffs]

        valid_slice = slice(len(padded) - len(history), len(padded))
        approx_valid = approx[valid_slice]
        result.loc[result.index[idx], f"{prefix}_a{levels}_last"] = approx_valid[-1]

        for level_idx, detail in enumerate(details, start=1):
            detail_valid = detail[valid_slice]
            result.loc[result.index[idx], f"{prefix}_d{level_idx}_last"] = detail_valid[-1]
            for window in energy_windows:
                tail = detail_valid[-window:] if len(detail_valid) >= window else detail_valid
                result.loc[result.index[idx], f"{prefix}_d{level_idx}_mean_{window}"] = float(np.nanmean(tail))
                result.loc[result.index[idx], f"{prefix}_d{level_idx}_std_{window}"] = float(np.nanstd(tail, ddof=0))
                result.loc[result.index[idx], f"{prefix}_d{level_idx}_energy_{window}"] = float(
                    np.nanmean(np.square(tail))
                )

    return result


def _select_target_column(base_volatility: str) -> str:
    mapping = {
        "close_to_close": "cc_var",
        "parkinson": "parkinson_var",
        "garman_klass": "gk_var",
        "rogers_satchell": "rs_var",
    }
    if base_volatility not in mapping:
        raise ValueError(f"Unsupported base volatility target: {base_volatility}")
    return mapping[base_volatility]


def build_feature_panel(raw_panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    features_cfg = config["features"]
    target_cfg = features_cfg["target"]
    base_target_col = _select_target_column(target_cfg["base_volatility"])
    epsilon = float(target_cfg.get("epsilon", 1.0e-8))
    transform = target_cfg.get("transform", "none")
    short_term_windows = list(features_cfg.get("short_term_windows", [2, 3, 5]))

    panels: list[pd.DataFrame] = []
    for _, group in raw_panel.groupby("index_id", sort=False):
        df = group.sort_values("date").copy()

        def has_cols(*cols: str) -> bool:
            return all(col in df.columns for col in cols)

        df["ret_cc_1d"] = np.log(df["close"]).diff()
        df["ret_oc_1d"] = np.log(df["close"] / df["open"])
        df["gap_co_1d"] = np.log(df["open"] / df["close"].shift(1))
        df["abs_ret"] = df["ret_cc_1d"].abs()
        df["abs_ret_oc_1d"] = df["ret_oc_1d"].abs()
        df["abs_gap_co_1d"] = df["gap_co_1d"].abs()
        df["hl_log_range"] = np.log(df["high"] / df["low"])
        df["jump_stress_1d"] = df["abs_gap_co_1d"] + df["abs_ret_oc_1d"]
        df["range_stress_1d"] = df["hl_log_range"] * df["vix_like"]

        df["cc_var"] = close_to_close_variance(df["close"]).clip(lower=0.0)
        df["parkinson_var"] = parkinson_variance(df["high"], df["low"]).clip(lower=0.0)
        df["gk_var"] = garman_klass_variance(df["open"], df["high"], df["low"], df["close"]).clip(lower=0.0)
        df["rs_var"] = rogers_satchell_variance(df["open"], df["high"], df["low"], df["close"]).clip(lower=0.0)

        df["vix_like_chg_1d"] = df["vix_like"].diff()
        df["dff_chg_1d"] = df["dff"].diff()
        df["dgs10_chg_1d"] = df["dgs10"].diff()
        df["t10y3m_chg_1d"] = df["t10y3m"].diff()
        df["nfci_chg_1d"] = df["nfci"].diff()
        df["etf_volchg_1d"] = np.log(df["etf_volume_proxy"].replace(0.0, np.nan)).diff()
        df["abs_vix_chg_1d"] = df["vix_like_chg_1d"].abs()

        if has_cols("dgs2"):
            df["dgs2_chg_1d"] = df["dgs2"].diff()
            df["t10y2y_spread"] = df["dgs10"] - df["dgs2"]
        if has_cols("hy_oas"):
            df["hy_oas_chg_1d"] = df["hy_oas"].diff()
        if has_cols("ig_oas"):
            df["ig_oas_chg_1d"] = df["ig_oas"].diff()
        if has_cols("hy_oas", "ig_oas"):
            df["hy_ig_oas_spread"] = df["hy_oas"] - df["ig_oas"]
        if has_cols("usepu"):
            df["usepu_log"] = np.log1p(df["usepu"].clip(lower=0.0))
            df["usepu_chg_1d"] = df["usepu_log"].diff()
        if has_cols("rvx"):
            df["rvx_chg_1d"] = df["rvx"].diff()
            df["rvx_premium"] = df["rvx"] - df["vix_like"]
        if has_cols("vxv"):
            df["vxv_chg_1d"] = df["vxv"].diff()
            df["vxv_vix_gap"] = df["vxv"] - df["vix_like"]
        if has_cols("hy_oas", "vix_like"):
            df["credit_vol_stress"] = df["hy_oas"] * df["vix_like"]
        if has_cols("usepu_log", "vix_like"):
            df["policy_vol_stress"] = df["usepu_log"] * df["vix_like"]

        tech_cfg = features_cfg["technical"]
        df[f"rsi_{tech_cfg['rsi_period']}"] = compute_rsi(df["close"], tech_cfg["rsi_period"])
        df[f"atr_{tech_cfg['atr_period']}"] = compute_atr(
            df["high"],
            df["low"],
            df["close"],
            tech_cfg["atr_period"],
        )
        macd = compute_macd(df["close"], tech_cfg["macd_fast"], tech_cfg["macd_slow"], tech_cfg["macd_signal"])
        df = pd.concat([df, macd], axis=1)
        atr_col = f"atr_{tech_cfg['atr_period']}"
        df["atr_pct"] = df[atr_col] / df["close"].replace(0.0, np.nan)

        for window in features_cfg["har_windows"]:
            df[f"har_target_{window}"] = df[base_target_col].rolling(window, min_periods=window).mean()
            df[f"har_absret_{window}"] = df["abs_ret"].rolling(window, min_periods=window).mean()
            df[f"har_vix_{window}"] = df["vix_like"].rolling(window, min_periods=window).mean()

        stress_sources = {
            "rs_var": "rs_var",
            "abs_ret": "abs_ret",
            "abs_gap": "abs_gap_co_1d",
            "abs_ret_oc": "abs_ret_oc_1d",
            "abs_vixchg": "abs_vix_chg_1d",
            "jump_stress": "jump_stress_1d",
            "hl_range": "hl_log_range",
            "range_stress": "range_stress_1d",
            "atr_pct": "atr_pct",
        }
        for prefix, source_col in stress_sources.items():
            for window in short_term_windows:
                rolling = df[source_col].rolling(window, min_periods=window)
                df[f"{prefix}_mean_{window}"] = rolling.mean()
                df[f"{prefix}_max_{window}"] = rolling.max()

        if features_cfg["wavelet"]["enabled"]:
            for series_name in features_cfg["wavelet"]["series"]:
                if series_name not in df.columns:
                    continue
                wavelet_df = causal_wavelet_features(
                    series=df[series_name],
                    basis=features_cfg["wavelet"]["basis"],
                    levels=int(features_cfg["wavelet"]["levels"]),
                    min_history=int(features_cfg["wavelet"]["min_history"]),
                    max_history=int(features_cfg["wavelet"]["max_history"]),
                    energy_windows=list(features_cfg["wavelet"]["energy_windows"]),
                    prefix=f"wav_{series_name}",
                )
                df = pd.concat([df, wavelet_df], axis=1)

        df["target_base_var"] = df[base_target_col]
        if transform == "log1p":
            df["target_model"] = np.log1p(df["target_base_var"] + epsilon)
        elif transform == "log":
            df["target_model"] = np.log(df["target_base_var"] + epsilon)
        else:
            df["target_model"] = df["target_base_var"]

        for horizon in config["experiment"]["horizons"]:
            future_var = future_average(df["target_base_var"], horizon)
            df[f"target_base_h{horizon}"] = future_var
            if transform == "log1p":
                df[f"target_model_h{horizon}"] = np.log1p(future_var + epsilon)
            elif transform == "log":
                df[f"target_model_h{horizon}"] = np.log(future_var + epsilon)
            else:
                df[f"target_model_h{horizon}"] = future_var

        panels.append(df)

    panel = pd.concat(panels, ignore_index=True)
    panel = panel.sort_values(["index_id", "date"]).reset_index(drop=True)
    panel = compute_cross_index_spillover_features(panel)
    return panel


def list_feature_columns(panel: pd.DataFrame) -> dict[str, list[str]]:
    metadata_cols = {
        "date",
        "index_id",
        "index_label",
    }
    target_cols = [col for col in panel.columns if col.startswith("target_")]
    wavelet_cols = [col for col in panel.columns if col.startswith("wav_")]
    spillover_cols = [col for col in panel.columns if col.startswith("xspill_")]

    raw_cols = [
        col
        for col in panel.columns
        if col not in metadata_cols
        and col not in target_cols
        and col not in wavelet_cols
        and col not in spillover_cols
        and col not in {"open", "high", "low", "close", "index_volume"}
    ]
    return {
        "raw": raw_cols,
        "wavelet": wavelet_cols,
        "spillover": spillover_cols,
        "all": raw_cols + wavelet_cols,
        "all_spillover": raw_cols + wavelet_cols + spillover_cols,
    }


def compute_cross_index_spillover_features(
    panel: pd.DataFrame,
    spillover_levels: list[int] | None = None,
    spillover_series_prefixes: list[str] | None = None,
) -> pd.DataFrame:
    """Add cross-index wavelet spillover features to the panel.

    For each target index, contemporaneous wavelet detail-component summaries
    (d2 and d3, capturing medium- and long-scale market dynamics) from peer
    indices are appended as ``xspill_{source_idx}_{original_col}`` columns.
    The hypothesis is that multi-scale volatility signals propagate across
    equity indices, and that these spillover components carry incremental
    predictive information for medium-to-long forecast horizons (h=5, h=10).

    Causality guarantee: peer wavelet features at time t are themselves derived
    from causal (non-look-ahead) wavelet transforms, so using them as predictors
    of horizon-h volatility at t+h is leak-free.

    Columns for the self-index are left as NaN and filtered in
    ``_feature_subsets()`` (experiment.py) before model fitting.

    Parameters
    ----------
    panel:
        Full multi-index feature panel produced by ``build_feature_panel``.
    spillover_levels:
        Wavelet detail levels to propagate as spillover features.
        Defaults to [2, 3] (medium and long scale).
    spillover_series_prefixes:
        Wavelet column prefixes of the source series to include.
        Defaults to ``wav_rs_var``, ``wav_abs_ret``, ``wav_vix_like``.
    """
    if spillover_levels is None:
        spillover_levels = [2, 3]
    if spillover_series_prefixes is None:
        spillover_series_prefixes = ["wav_rs_var", "wav_abs_ret", "wav_vix_like"]

    # Select wavelet columns at the target decomposition levels and series
    spillover_cols = [
        col
        for col in panel.columns
        if col.startswith("wav_")
        and any(f"_d{lvl}_" in col for lvl in spillover_levels)
        and any(col.startswith(pref) for pref in spillover_series_prefixes)
    ]
    if not spillover_cols:
        return panel

    indices = sorted(panel["index_id"].unique())
    if len(indices) < 2:
        return panel

    panel = panel.copy()

    # Build per-source date-indexed wavelet frames once
    source_frames: dict[str, pd.DataFrame] = {}
    for src_idx in indices:
        src_data = (
            panel[panel["index_id"] == src_idx][["date"] + spillover_cols]
            .set_index("date")
        )
        source_frames[src_idx] = src_data

    # Inject peer features into every target-index row
    for tgt_idx in indices:
        tgt_mask = panel["index_id"] == tgt_idx
        tgt_dates = panel.loc[tgt_mask, "date"].values
        for src_idx, src_frame in source_frames.items():
            if src_idx == tgt_idx:
                continue
            aligned = src_frame.reindex(tgt_dates)  # NaN for any unmatched dates
            aligned.index = panel.index[tgt_mask]
            for orig_col in spillover_cols:
                new_col = f"xspill_{src_idx}_{orig_col}"
                panel.loc[tgt_mask, new_col] = aligned[orig_col].values

    return panel
