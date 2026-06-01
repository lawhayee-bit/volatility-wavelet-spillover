from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve

from .features import list_feature_columns
from .metrics import classification_metric_bundle, inverse_target_transform, qlike_loss, regression_metric_bundle
from .models import (
    clone_estimator,
    fit_arch_variance_forecast,
    make_classifier,
    make_regressor,
    oof_predictions,
    oof_probabilities,
)
from .stats import clark_west, diebold_mariano
from .utils import ensure_directory


def _sigmoid(x: float | np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _probability_from_ratio(value: float, threshold: float, eps: float = 1.0e-8) -> float:
    score = np.log((value + eps) / (threshold + eps))
    return float(_sigmoid(score))


def _safe_positive(value: float, floor: float = 1.0e-10) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(max(value, floor))


def _should_apply_qlike_calibration(model_name: str, config: dict[str, Any]) -> bool:
    calibration_cfg = config["experiment"].get("qlike_calibration", {})
    if not calibration_cfg.get("enabled", False):
        return False
    model_names = set(calibration_cfg.get("models", []))
    return model_name in model_names


def _get_qlike_calibration_config(config: dict[str, Any], horizon: int) -> dict[str, Any]:
    calibration_cfg = dict(config["experiment"].get("qlike_calibration", {}))
    floor_map = calibration_cfg.get("floor_quantile_by_horizon", {})
    if str(horizon) in floor_map:
        calibration_cfg["floor_quantile"] = floor_map[str(horizon)]
    elif horizon in floor_map:
        calibration_cfg["floor_quantile"] = floor_map[horizon]
    return calibration_cfg


def _fit_qlike_calibration(
    y_true_var: np.ndarray,
    pred_var: np.ndarray | None,
    calibration_cfg: dict[str, Any],
) -> dict[str, float] | None:
    floor_quantile = float(calibration_cfg.get("floor_quantile", 0.01))
    scale_min = float(calibration_cfg.get("scale_clip", [0.25, 4.0])[0])
    scale_max = float(calibration_cfg.get("scale_clip", [0.25, 4.0])[1])
    use_scale = bool(calibration_cfg.get("use_scale", False))

    y_true_var = np.asarray(y_true_var, dtype=float)
    valid_y = np.isfinite(y_true_var)
    if valid_y.sum() < 20:
        return None

    y_valid = np.clip(y_true_var[valid_y], 1.0e-10, None)
    floor = float(max(np.nanquantile(y_valid, floor_quantile), 1.0e-10))
    scale = 1.0

    if use_scale and pred_var is not None:
        pred_var = np.asarray(pred_var, dtype=float)
        valid = valid_y & np.isfinite(pred_var)
        if valid.sum() >= 20:
            y_valid = np.clip(y_true_var[valid], 1.0e-10, None)
            pred_valid = np.clip(pred_var[valid], floor, None)
            scale = float(np.nanmean(y_valid / pred_valid))
            scale = float(np.clip(scale, scale_min, scale_max))
    return {"scale": scale, "floor": floor}


def _apply_qlike_calibration(pred_var: float, calibration: dict[str, float] | None) -> float:
    if calibration is None or not np.isfinite(pred_var):
        return _safe_positive(pred_var)
    calibrated = float(calibration["scale"]) * float(pred_var)
    calibrated = max(calibrated, float(calibration["floor"]))
    return _safe_positive(calibrated)


def _feature_subsets(df: pd.DataFrame) -> dict[str, list[str]]:
    feature_map = list_feature_columns(df)
    har_cols = [col for col in df.columns if col.startswith("har_target_")]
    harx_cols = har_cols + [
        col
        for col in df.columns
        if col.startswith("har_absret_")
        or col.startswith("har_vix_")
        or col in {
            "target_base_var",
            "target_model",
            "vix_like",
            "vix_like_chg_1d",
            "dff",
            "dgs10",
            "t10y3m",
            "nfci",
            "dgs2",
            "hy_oas",
            "ig_oas",
            "usepu",
            "usepu_log",
            "rvx",
            "vxv",
            "t10y2y_spread",
            "hy_ig_oas_spread",
            "rvx_premium",
            "vxv_vix_gap",
            "credit_vol_stress",
            "policy_vol_stress",
            "recession_dummy",
            "ret_cc_1d",
            "abs_ret",
        }
    ]
    warning_wavelet = [col for col in feature_map["wavelet"] if "energy_" in col or col.endswith("_last")]
    warning_raw = [
        col
        for col in feature_map["raw"]
        if col
        not in {
            "target_model",
        }
    ]
    # For cross-index spillover columns, only include those that have actual
    # data for this (single-index) slice of the panel. Self-referential
    # xspill_{this_index}_* columns will be all-NaN and are excluded here.
    active_spillover = [
        col
        for col in feature_map.get("spillover", [])
        if df[col].notna().mean() > 0.05
    ]
    return {
        "raw": sorted(set(feature_map["raw"])),
        "all": sorted(set(feature_map["all"])),
        "all_spillover": sorted(set(feature_map["all"] + active_spillover)),
        "har": sorted(set(har_cols)),
        "harx": sorted(set(harx_cols)),
        "warning_raw": sorted(set(warning_raw)),
        "warning_wavelet": sorted(set(warning_wavelet)),
    }


def _select_decision_threshold(y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    beta_sq = beta**2
    fbeta = (1 + beta_sq) * precision[:-1] * recall[:-1] / (beta_sq * precision[:-1] + recall[:-1] + 1.0e-12)
    best_idx = int(np.nanargmax(fbeta))
    return float(thresholds[best_idx])


def _prepare_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    mask = train[target_col].notna()
    X_train = train.loc[mask, feature_cols].copy()
    y_train = train.loc[mask, target_col].copy()
    X_test = test[feature_cols].copy()
    return X_train, y_train, X_test


def _fit_tabular_regressor_bundle(
    model_name: str,
    train: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target_model_col: str,
    target_var_col: str,
    calibration_cfg: dict[str, Any],
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    spec = make_regressor(model_name, random_state=random_state, n_jobs=n_jobs)
    feature_cols = feature_sets[spec.feature_set]
    X_train, y_train, _ = _prepare_train_test(train, train.iloc[[0]].copy(), feature_cols, target_model_col)
    estimator = clone_estimator(spec.estimator)
    estimator.fit(X_train, y_train)

    calibration = None
    if calibration_cfg.get("enabled", False):
        y_true_var = train.loc[train[target_model_col].notna(), target_var_col].to_numpy(dtype=float)
        calibration = _fit_qlike_calibration(y_true_var, pred_var=None, calibration_cfg=calibration_cfg)

    return {"spec": spec, "feature_cols": feature_cols, "estimator": estimator, "qlike_calibration": calibration}


def _predict_tabular_regressor_bundle(
    bundle: dict[str, Any],
    test: pd.DataFrame,
    transform: str,
    epsilon: float,
) -> float:
    X_test = test[bundle["feature_cols"]].copy()
    pred_model = float(bundle["estimator"].predict(X_test)[0])
    pred_var = inverse_target_transform(np.array([pred_model]), transform, epsilon)[0]
    return _apply_qlike_calibration(pred_var, bundle.get("qlike_calibration"))


def _fit_main_stacking_bundle(
    train: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target_model_col: str,
    target_var_col: str,
    calibration_cfg: dict[str, Any],
    transform: str,
    epsilon: float,
    inner_cv_splits: int,
    random_state: int,
    n_jobs: int,
) -> dict[str, Any]:
    linear_spec = make_regressor("harx", random_state=random_state, n_jobs=n_jobs)
    nonlinear_spec = make_regressor("wavelet_lightgbm", random_state=random_state, n_jobs=n_jobs)

    X_train_linear, y_train, _ = _prepare_train_test(train, train.iloc[[0]].copy(), feature_sets["harx"], target_model_col)
    X_train_nonlinear, _, _ = _prepare_train_test(train, train.iloc[[0]].copy(), feature_sets["all"], target_model_col)

    linear_factory = lambda: clone_estimator(linear_spec.estimator)
    nonlinear_factory = lambda: clone_estimator(nonlinear_spec.estimator)

    oof_linear = oof_predictions(linear_factory, X_train_linear, y_train, n_splits=inner_cv_splits)
    oof_nonlinear = oof_predictions(nonlinear_factory, X_train_nonlinear, y_train, n_splits=inner_cv_splits)
    valid = np.isfinite(oof_linear) & np.isfinite(oof_nonlinear)
    if valid.sum() < max(20, inner_cv_splits):
        return {"test_pred_var": np.nan, "train_oof_var": np.full(len(train), np.nan)}

    stacker = LinearRegression()
    stacker.fit(np.column_stack([oof_linear[valid], oof_nonlinear[valid]]), y_train.iloc[valid])

    linear_model = linear_factory()
    nonlinear_model = nonlinear_factory()
    linear_model.fit(X_train_linear, y_train)
    nonlinear_model.fit(X_train_nonlinear, y_train)

    full_oof_model = np.full(len(train), np.nan, dtype=float)
    valid_train_index = np.flatnonzero(train[target_model_col].notna())
    full_oof_model[valid_train_index[valid]] = stacker.predict(
        np.column_stack([oof_linear[valid], oof_nonlinear[valid]])
    )
    train_oof_var = inverse_target_transform(full_oof_model, transform, epsilon)
    calibration = None
    if calibration_cfg.get("enabled", False):
        y_true_var = train.loc[train[target_model_col].notna(), target_var_col].to_numpy(dtype=float)
        calibration = _fit_qlike_calibration(y_true_var, pred_var=None, calibration_cfg=calibration_cfg)

    return {
        "linear_model": linear_model,
        "nonlinear_model": nonlinear_model,
        "stacker": stacker,
        "linear_feature_cols": feature_sets["harx"],
        "nonlinear_feature_cols": feature_sets["all"],
        "train_oof_var": train_oof_var,
        "qlike_calibration": calibration,
    }


def _predict_main_stacking_bundle(
    bundle: dict[str, Any],
    test: pd.DataFrame,
    transform: str,
    epsilon: float,
) -> float:
    pred_linear = float(bundle["linear_model"].predict(test[bundle["linear_feature_cols"]].copy())[0])
    pred_nonlinear = float(bundle["nonlinear_model"].predict(test[bundle["nonlinear_feature_cols"]].copy())[0])
    pred_model = float(bundle["stacker"].predict(np.array([[pred_linear, pred_nonlinear]]))[0])
    pred_var = inverse_target_transform(np.array([pred_model]), transform, epsilon)[0]
    return _apply_qlike_calibration(pred_var, bundle.get("qlike_calibration"))


def _fit_regression_refit_bundle(
    train: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: dict[str, Any],
    horizon: int,
    regression_models: list[str],
) -> dict[str, Any]:
    target_model_col = f"target_model_h{horizon}"
    target_var_col = f"target_base_h{horizon}"
    transform = config["features"]["target"]["transform"]
    epsilon = float(config["features"]["target"]["epsilon"])
    random_state = int(config["project"]["random_state"])
    n_jobs = int(config["runtime"]["n_jobs"])
    inner_cv_splits = int(config["experiment"]["inner_cv_splits"])

    current_returns = train["ret_cc_1d"]
    qlike_calibration_cfg = _get_qlike_calibration_config(config, horizon)

    bundle: dict[str, Any] = {
        "transform": transform,
        "epsilon": epsilon,
        "models": {},
    }

    for model_name in regression_models:
        try:
            if model_name in {"last_value", "hv_5", "hv_22"}:
                bundle["models"][model_name] = {"type": "naive"}
            elif model_name == "garch":
                pred = fit_arch_variance_forecast(current_returns, mean_model="Zero", vol_model="GARCH", horizon=horizon)
                bundle["models"][model_name] = {"type": "fixed", "pred_var": pred}
            elif model_name == "egarch":
                pred = fit_arch_variance_forecast(current_returns, mean_model="Zero", vol_model="EGARCH", horizon=horizon)
                bundle["models"][model_name] = {"type": "fixed", "pred_var": pred}
            elif model_name == "main_stacking":
                main_bundle = _fit_main_stacking_bundle(
                    train=train,
                    feature_sets=feature_sets,
                    target_model_col=target_model_col,
                    target_var_col=target_var_col,
                    calibration_cfg=qlike_calibration_cfg if _should_apply_qlike_calibration(model_name, config) else {"enabled": False},
                    transform=transform,
                    epsilon=epsilon,
                    inner_cv_splits=inner_cv_splits,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                bundle["models"][model_name] = {"type": "main_stacking", **main_bundle}
            else:
                fitted = _fit_tabular_regressor_bundle(
                    model_name=model_name,
                    train=train,
                    feature_sets=feature_sets,
                    target_model_col=target_model_col,
                    target_var_col=target_var_col,
                    calibration_cfg=qlike_calibration_cfg if _should_apply_qlike_calibration(model_name, config) else {"enabled": False},
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                bundle["models"][model_name] = {"type": "tabular", **fitted}
        except Exception:
            bundle["models"][model_name] = {"type": "failed"}

    return bundle


def _predict_regression_models_from_bundle(
    bundle: dict[str, Any],
    test: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transform = bundle["transform"]
    epsilon = bundle["epsilon"]

    current_target = float(test["target_base_var"].iloc[0])
    current_hv5 = float(test["har_target_5"].iloc[0])
    current_hv22 = float(test["har_target_22"].iloc[0])

    preds: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    for model_name, model_bundle in bundle["models"].items():
        model_type = model_bundle["type"]
        if model_type == "naive":
            if model_name == "last_value":
                pred = current_target
            elif model_name == "hv_5":
                pred = current_hv5
            else:
                pred = current_hv22
        elif model_type == "fixed":
            pred = model_bundle["pred_var"]
        elif model_type == "tabular":
            pred = _predict_tabular_regressor_bundle(model_bundle, test, transform=transform, epsilon=epsilon)
        elif model_type == "main_stacking":
            pred = _predict_main_stacking_bundle(model_bundle, test, transform=transform, epsilon=epsilon)
            cache["main_stacking"] = model_bundle
        else:
            pred = np.nan
        preds.append({"model": model_name, "pred_var": pred})
    return preds, cache


def _build_warning_train_test_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    main_train_oof_var: np.ndarray,
    main_test_pred_var: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = feature_sets["warning_raw"]
    wavelet_cols = feature_sets["warning_wavelet"]
    train_warning = train[base_cols + wavelet_cols].copy()
    test_warning = test[base_cols + wavelet_cols].copy()
    train_warning["main_forecast_var"] = main_train_oof_var
    train_warning["main_forecast_gap"] = main_train_oof_var - train["target_base_var"].to_numpy(dtype=float)
    test_warning["main_forecast_var"] = main_test_pred_var
    test_warning["main_forecast_gap"] = main_test_pred_var - float(test["target_base_var"].iloc[0])
    return train_warning, test_warning


def _predict_classification_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: dict[str, Any],
    horizon: int,
    classification_models: list[str],
    threshold: float,
    regression_pred_lookup: dict[str, float],
    main_train_oof_var: np.ndarray,
) -> list[dict[str, Any]]:
    target_var_col = f"target_base_h{horizon}"
    y_train_var = train[target_var_col].to_numpy(dtype=float)
    y_train_label = (y_train_var > threshold).astype(int)
    if np.unique(y_train_label).size < 2:
        y_train_label = np.where(np.arange(len(y_train_label)) >= len(y_train_label) - 1, 1, 0)

    beta = float(config["experiment"]["warning_beta"])
    random_state = int(config["project"]["random_state"])
    n_jobs = int(config["runtime"]["n_jobs"])

    current_target = float(test["target_base_var"].iloc[0])
    main_test_pred = regression_pred_lookup.get("main_stacking", np.nan)
    target_true = int(float(test[target_var_col].iloc[0]) > threshold)

    preds: list[dict[str, Any]] = []
    for model_name in classification_models:
        try:
            if model_name == "naive_threshold":
                prob = _probability_from_ratio(current_target, threshold)
                label = int(prob >= 0.5)
            elif model_name == "forecast_threshold":
                prob = _probability_from_ratio(main_test_pred, threshold)
                label = int(prob >= 0.5)
            else:
                spec = make_classifier(model_name, random_state=random_state, n_jobs=n_jobs)
                if model_name == "main_warning":
                    X_train, X_test = _build_warning_train_test_features(
                        train=train,
                        test=test,
                        feature_sets=feature_sets,
                        main_train_oof_var=main_train_oof_var,
                        main_test_pred_var=main_test_pred,
                    )
                else:
                    cols = feature_sets[spec.feature_set]
                    X_train = train[cols].copy()
                    X_test = test[cols].copy()

                estimator_factory = lambda: clone_estimator(spec.estimator)
                oof_prob = oof_probabilities(estimator_factory, X_train, pd.Series(y_train_label), n_splits=int(config["experiment"]["inner_cv_splits"]))
                decision_threshold = _select_decision_threshold(y_train_label, oof_prob, beta=beta)

                estimator = estimator_factory()
                estimator.fit(X_train, y_train_label)
                prob = float(estimator.predict_proba(X_test)[:, 1][0])
                label = int(prob >= decision_threshold)
        except Exception:
            prob = np.nan
            label = np.nan

        preds.append(
            {
                "model": model_name,
                "warning_threshold": threshold,
                "target_true": target_true,
                "pred_score": prob,
                "pred_label": label,
            }
        )
    return preds


def _fit_classification_refit_bundle(
    train: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: dict[str, Any],
    horizon: int,
    classification_models: list[str],
    threshold: float,
    main_train_oof_var: np.ndarray,
) -> dict[str, Any]:
    target_var_col = f"target_base_h{horizon}"
    y_train_var = train[target_var_col].to_numpy(dtype=float)
    y_train_label = (y_train_var > threshold).astype(int)
    if np.unique(y_train_label).size < 2:
        y_train_label = np.where(np.arange(len(y_train_label)) >= len(y_train_label) - 1, 1, 0)

    beta = float(config["experiment"]["warning_beta"])
    random_state = int(config["project"]["random_state"])
    n_jobs = int(config["runtime"]["n_jobs"])

    bundle: dict[str, Any] = {
        "threshold": threshold,
        "train_label": y_train_label,
        "models": {},
    }
    for model_name in classification_models:
        try:
            if model_name in {"naive_threshold", "forecast_threshold"}:
                bundle["models"][model_name] = {"type": "rule"}
                continue

            spec = make_classifier(model_name, random_state=random_state, n_jobs=n_jobs)
            if model_name == "main_warning":
                X_train, _ = _build_warning_train_test_features(
                    train=train,
                    test=train.iloc[[0]].copy(),
                    feature_sets=feature_sets,
                    main_train_oof_var=main_train_oof_var,
                    main_test_pred_var=np.nan,
                )
            else:
                cols = feature_sets[spec.feature_set]
                X_train = train[cols].copy()

            estimator_factory = lambda: clone_estimator(spec.estimator)
            oof_prob = oof_probabilities(estimator_factory, X_train, pd.Series(y_train_label), n_splits=int(config["experiment"]["inner_cv_splits"]))
            decision_threshold = _select_decision_threshold(y_train_label, oof_prob, beta=beta)

            estimator = estimator_factory()
            estimator.fit(X_train, y_train_label)
            bundle["models"][model_name] = {
                "type": "classifier",
                "spec": spec,
                "estimator": estimator,
                "decision_threshold": decision_threshold,
            }
        except Exception:
            bundle["models"][model_name] = {"type": "failed"}
    return bundle


def _predict_classification_models_from_bundle(
    bundle: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    classification_models: list[str],
    regression_pred_lookup: dict[str, float],
    main_train_oof_var: np.ndarray,
) -> list[dict[str, Any]]:
    threshold = float(bundle["threshold"])
    current_target = float(test["target_base_var"].iloc[0])
    main_test_pred = regression_pred_lookup.get("main_stacking", np.nan)
    target_true = int(float(test[bundle["target_var_col"]].iloc[0]) > threshold)

    preds: list[dict[str, Any]] = []
    for model_name in classification_models:
        model_bundle = bundle["models"].get(model_name, {"type": "failed"})
        model_type = model_bundle["type"]
        if model_type == "rule":
            if model_name == "naive_threshold":
                prob = _probability_from_ratio(current_target, threshold)
            else:
                prob = _probability_from_ratio(main_test_pred, threshold)
            label = int(prob >= 0.5)
        elif model_type == "classifier":
            if model_name == "main_warning":
                _, X_test = _build_warning_train_test_features(
                    train=train,
                    test=test,
                    feature_sets=feature_sets,
                    main_train_oof_var=main_train_oof_var,
                    main_test_pred_var=main_test_pred,
                )
            else:
                cols = feature_sets[model_bundle["spec"].feature_set]
                X_test = test[cols].copy()
            prob = float(model_bundle["estimator"].predict_proba(X_test)[:, 1][0])
            label = int(prob >= model_bundle["decision_threshold"])
        else:
            prob = np.nan
            label = np.nan
        preds.append(
            {
                "model": model_name,
                "warning_threshold": threshold,
                "target_true": target_true,
                "pred_score": prob,
                "pred_label": label,
            }
        )
    return preds


@dataclass
class ExperimentOutputs:
    regression_predictions: pd.DataFrame
    classification_predictions: pd.DataFrame
    regression_summary: pd.DataFrame
    classification_summary: pd.DataFrame
    dm_table: pd.DataFrame
    cw_table: pd.DataFrame


def run_experiment(
    panel: pd.DataFrame,
    config: dict[str, Any],
    indices: list[str] | None = None,
    horizons: list[int] | None = None,
    regression_models: list[str] | None = None,
    classification_models: list[str] | None = None,
    max_test_steps: int | None = None,
) -> ExperimentOutputs:
    feature_sets_global = _feature_subsets(panel)
    experiment_cfg = config["experiment"]
    runtime_cfg = config["runtime"]
    test_start = pd.Timestamp(experiment_cfg["test_start"])
    test_end = pd.Timestamp(experiment_cfg["test_end"])
    train_window = int(experiment_cfg["train_window_days"])
    refit_interval = int(experiment_cfg.get("refit_interval_days", 1))

    indices = indices or [item["id"] for item in config["data"]["indices"]]
    horizons = horizons or list(experiment_cfg["horizons"])
    regression_models = regression_models or list(experiment_cfg["models"]["regression"])
    classification_models = classification_models or list(experiment_cfg["models"]["classification"])

    regression_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []

    for index_id in indices:
        index_df = panel.loc[panel["index_id"] == index_id].sort_values("date").reset_index(drop=True)
        feature_sets = {key: [col for col in cols if col in index_df.columns] for key, cols in feature_sets_global.items()}

        test_mask = (index_df["date"] >= test_start) & (index_df["date"] <= test_end)
        test_positions = np.flatnonzero(test_mask)
        if max_test_steps is not None:
            test_positions = test_positions[:max_test_steps]
        for horizon in horizons:
            target_var_col = f"target_base_h{horizon}"
            stepped_positions = test_positions[:: int(experiment_cfg["rolling_step"])]
            for batch_start in range(0, len(stepped_positions), refit_interval):
                refit_positions = stepped_positions[batch_start : batch_start + refit_interval]
                anchor_pos = refit_positions[0]
                if anchor_pos < train_window or pd.isna(index_df.loc[anchor_pos, target_var_col]):
                    continue
                train = index_df.iloc[anchor_pos - train_window : anchor_pos].copy().reset_index(drop=True)
                threshold = float(train[target_var_col].quantile(float(experiment_cfg["warning_quantile"])))

                reg_bundle = _fit_regression_refit_bundle(
                    train=train,
                    feature_sets=feature_sets,
                    config=config,
                    horizon=horizon,
                    regression_models=regression_models,
                )
                main_train_oof_var = reg_bundle["models"].get("main_stacking", {}).get("train_oof_var", np.full(len(train), np.nan))
                cls_bundle = _fit_classification_refit_bundle(
                    train=train,
                    feature_sets=feature_sets,
                    config=config,
                    horizon=horizon,
                    classification_models=classification_models,
                    threshold=threshold,
                    main_train_oof_var=main_train_oof_var,
                )
                cls_bundle["target_var_col"] = target_var_col

                for pos in refit_positions:
                    if pos < train_window or pd.isna(index_df.loc[pos, target_var_col]):
                        continue
                    test = index_df.iloc[[pos]].copy().reset_index(drop=True)
                    y_true_var = float(test[target_var_col].iloc[0])

                    reg_preds, cache = _predict_regression_models_from_bundle(
                        bundle=reg_bundle,
                        test=test,
                    )
                    reg_lookup = {row["model"]: row["pred_var"] for row in reg_preds}

                    for row in reg_preds:
                        regression_rows.append(
                            {
                                "date": test["date"].iloc[0],
                                "index_id": index_id,
                                "horizon": horizon,
                                "model": row["model"],
                                "y_true_var": y_true_var,
                                "y_pred_var": row["pred_var"],
                                "warning_threshold": threshold,
                            }
                        )

                    cls_preds = _predict_classification_models_from_bundle(
                        bundle=cls_bundle,
                        train=train,
                        test=test,
                        feature_sets=feature_sets,
                        classification_models=classification_models,
                        regression_pred_lookup=reg_lookup,
                        main_train_oof_var=cache.get("main_stacking", {}).get("train_oof_var", main_train_oof_var),
                    )
                    for row in cls_preds:
                        classification_rows.append(
                            {
                                "date": test["date"].iloc[0],
                                "index_id": index_id,
                                "horizon": horizon,
                                "model": row["model"],
                                "y_true_label": row["target_true"],
                                "y_score": row["pred_score"],
                                "y_pred_label": row["pred_label"],
                                "warning_threshold": row["warning_threshold"],
                            }
                        )

    regression_predictions = pd.DataFrame(regression_rows)
    classification_predictions = pd.DataFrame(classification_rows)

    regression_summary = summarise_regression_predictions(regression_predictions)
    classification_summary = summarise_classification_predictions(classification_predictions)
    dm_table, cw_table = compare_regression_models(regression_predictions, benchmark_model="har")

    return ExperimentOutputs(
        regression_predictions=regression_predictions,
        classification_predictions=classification_predictions,
        regression_summary=regression_summary,
        classification_summary=classification_summary,
        dm_table=dm_table,
        cw_table=cw_table,
    )


def summarise_regression_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (index_id, horizon, model), group in predictions.groupby(["index_id", "horizon", "model"], dropna=False):
        valid = group[["y_true_var", "y_pred_var"]].dropna()
        if valid.empty:
            continue
        metrics = regression_metric_bundle(valid["y_true_var"].to_numpy(), valid["y_pred_var"].to_numpy())
        rows.append({"index_id": index_id, "horizon": horizon, "model": model, **metrics})
    return pd.DataFrame(rows).sort_values(["index_id", "horizon", "qlike", "rmse"])


def summarise_classification_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (index_id, horizon, model), group in predictions.groupby(["index_id", "horizon", "model"], dropna=False):
        valid = group[["y_true_label", "y_score", "y_pred_label"]].dropna()
        if valid.empty or valid["y_true_label"].nunique() < 2:
            continue
        metrics = classification_metric_bundle(
            valid["y_true_label"].to_numpy(dtype=int),
            valid["y_score"].to_numpy(dtype=float),
            valid["y_pred_label"].to_numpy(dtype=int),
        )
        rows.append({"index_id": index_id, "horizon": horizon, "model": model, **metrics})
    return pd.DataFrame(rows).sort_values(["index_id", "horizon", "pr_auc"], ascending=[True, True, False])


def compare_regression_models(predictions: pd.DataFrame, benchmark_model: str = "har") -> tuple[pd.DataFrame, pd.DataFrame]:
    dm_rows: list[dict[str, Any]] = []
    cw_rows: list[dict[str, Any]] = []
    for (index_id, horizon), group in predictions.groupby(["index_id", "horizon"], dropna=False):
        benchmark = group.loc[group["model"] == benchmark_model, ["date", "y_true_var", "y_pred_var"]].rename(
            columns={"y_pred_var": "y_pred_benchmark"}
        )
        if benchmark.empty:
            continue
        for model, model_group in group.groupby("model", dropna=False):
            if model == benchmark_model:
                continue
            merged = benchmark.merge(
                model_group[["date", "y_pred_var"]],
                on="date",
                how="inner",
            ).rename(columns={"y_pred_var": "y_pred_model"})
            merged = merged.dropna()
            if merged.empty:
                continue

            dm = diebold_mariano(
                qlike_loss(merged["y_true_var"], merged["y_pred_model"]),
                qlike_loss(merged["y_true_var"], merged["y_pred_benchmark"]),
                horizon=int(horizon),
            )
            dm_rows.append({"index_id": index_id, "horizon": horizon, "model": model, **dm})

            cw = clark_west(
                merged["y_true_var"].to_numpy(),
                merged["y_pred_benchmark"].to_numpy(),
                merged["y_pred_model"].to_numpy(),
            )
            cw_rows.append({"index_id": index_id, "horizon": horizon, "model": model, **cw})

    return pd.DataFrame(dm_rows), pd.DataFrame(cw_rows)


def save_experiment_outputs(outputs: ExperimentOutputs, output_root: str | Path = "outputs") -> None:
    output_root = Path(output_root)
    tables_dir = ensure_directory(output_root / "tables")
    preds_dir = ensure_directory(output_root / "predictions")

    outputs.regression_predictions.to_csv(preds_dir / "regression_predictions.csv", index=False)
    outputs.classification_predictions.to_csv(preds_dir / "classification_predictions.csv", index=False)
    outputs.regression_summary.to_csv(tables_dir / "regression_summary.csv", index=False)
    outputs.classification_summary.to_csv(tables_dir / "classification_summary.csv", index=False)
    outputs.dm_table.to_csv(tables_dir / "diebold_mariano.csv", index=False)
    outputs.cw_table.to_csv(tables_dir / "clark_west.csv", index=False)
