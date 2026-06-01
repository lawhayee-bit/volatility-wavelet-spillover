from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from plotting import create_figure, save_figure
from .utils import ensure_directory


KEY_REGRESSION_MODELS = ["har", "lightgbm", "wavelet_lightgbm", "main_stacking"]
KEY_WARNING_MODELS = ["naive_threshold", "logistic_raw", "main_warning"]


def plot_main_regression_bars(
    regression_summary: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
    metric: str = "qlike",
    models: Iterable[str] = KEY_REGRESSION_MODELS,
) -> Path:
    output_dir = ensure_directory(output_dir)
    plot_df = regression_summary.loc[regression_summary["model"].isin(models)].copy()
    plot_df = plot_df.sort_values(["index_id", "horizon", "model"])

    fig, ax, style = create_figure("bar")
    palette = style["palette"]["series"]

    groups = sorted(plot_df[["index_id", "horizon"]].drop_duplicates().apply(tuple, axis=1).tolist())
    model_list = list(models)
    x = np.arange(len(groups))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(model_list))

    for idx, model in enumerate(model_list):
        values = []
        for index_id, horizon in groups:
            matched = plot_df.loc[(plot_df["index_id"] == index_id) & (plot_df["horizon"] == horizon) & (plot_df["model"] == model)]
            values.append(np.nan if matched.empty else float(matched[metric].iloc[0]))
        ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=model,
            color=palette[idx % len(palette)],
            edgecolor=style.get("bar_edgecolor", style["axes"]["edgecolor"]),
            linewidth=style.get("bar_linewidth", 0.8),
        )

    ax.set_xlabel("Index / Horizon")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([f"{idx.upper()} / h={h}" for idx, h in groups])
    ax.legend(ncols=2)
    return save_figure(fig, Path(output_dir) / f"main_regression_{metric}.pdf", style)


def plot_forecast_timeseries(
    regression_predictions: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
    index_id: str = "spx",
    horizon: int = 1,
    model: str = "main_stacking",
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
) -> Path:
    output_dir = ensure_directory(output_dir)
    plot_df = regression_predictions.loc[
        (regression_predictions["index_id"] == index_id)
        & (regression_predictions["horizon"] == horizon)
        & (regression_predictions["model"] == model)
    ].copy()
    plot_df = plot_df.loc[
        (plot_df["date"] >= pd.Timestamp(start_date))
        & (plot_df["date"] <= pd.Timestamp(end_date))
    ].sort_values("date")

    fig, ax, style = create_figure("timeseries")
    ax.plot(plot_df["date"], plot_df["y_true_var"], label="Actual", color=style["palette"]["series"][0])
    ax.plot(plot_df["date"], plot_df["y_pred_var"], label="Predicted", color=style["palette"]["series"][1])
    ax.set_xlabel("Date")
    ax.set_ylabel("Variance Proxy")
    ax.legend()
    return save_figure(fig, Path(output_dir) / f"forecast_{index_id}_h{horizon}_{model}.pdf", style)


def plot_warning_pr_curve(
    classification_predictions: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
    index_id: str = "spx",
    horizon: int = 1,
    models: Iterable[str] = KEY_WARNING_MODELS,
) -> Path:
    output_dir = ensure_directory(output_dir)
    fig, ax, style = create_figure("roc_pr")

    palette = style["palette"]["series"]
    for idx, model in enumerate(models):
        plot_df = classification_predictions.loc[
            (classification_predictions["index_id"] == index_id)
            & (classification_predictions["horizon"] == horizon)
            & (classification_predictions["model"] == model)
        ].dropna(subset=["y_true_label", "y_score"])
        if plot_df.empty or plot_df["y_true_label"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(plot_df["y_true_label"], plot_df["y_score"])
        ax.plot(recall, precision, label=model, color=palette[idx % len(palette)])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return save_figure(fig, Path(output_dir) / f"warning_pr_{index_id}_h{horizon}.pdf", style)


def plot_metric_heatmap(
    regression_summary: pd.DataFrame,
    output_dir: str | Path = "outputs/figures",
    metric: str = "qlike",
    models: Iterable[str] = KEY_REGRESSION_MODELS,
) -> Path:
    output_dir = ensure_directory(output_dir)
    plot_df = regression_summary.loc[regression_summary["model"].isin(models)].copy()
    plot_df["index_horizon"] = plot_df["index_id"].str.upper() + "_h" + plot_df["horizon"].astype(str)
    pivot = plot_df.pivot(index="model", columns="index_horizon", values=metric)

    fig, ax, style = create_figure("heatmap")
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="cividis")
    ax.set_xlabel("Index / Horizon")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    fig.colorbar(image, ax=ax, shrink=style.get("cbar_shrink", 0.9))
    return save_figure(fig, Path(output_dir) / f"heatmap_{metric}.pdf", style)
