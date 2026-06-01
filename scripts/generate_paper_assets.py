from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting import apply_plot_style, configure_axes, load_plot_style, save_figure


MAIN_MODELS = ["har", "hv_22", "lightgbm", "wavelet_lightgbm"]
WARNING_MODELS = ["naive_threshold", "logistic_raw"]
MODEL_LABELS = {
    "har": "HAR",
    "hv_22": "HV-22",
    "lightgbm": "LightGBM",
    "wavelet_lightgbm": "Wavelet-LightGBM",
    "main_stacking": "Stacking",
    "naive_threshold": "Naive Threshold",
    "logistic_raw": "Logistic-Raw",
    "main_warning": "Main Warning",
}
INDEX_LABELS = {"spx": "S&P 500", "ndq": "Nasdaq-100", "dji": "DJIA"}
SHORT_INDEX_LABELS = {"spx": "SPX", "ndq": "NDQ", "dji": "DJI"}
PACKAGE_LABELS = {
    "main": "Main RS Target",
    "plusdata": "Expanded Public Risk Data",
    "parkinson": "Parkinson Target",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript figures and tables.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "paper" / "manuscript"),
        help="Base manuscript directory containing figures/ and tables/.",
    )
    return parser.parse_args()


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order].astype(int)
    y_score = y_score[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    positives = np.maximum(tp[-1], 1)
    recall = tp / positives
    precision = np.r_[precision[0], precision]
    recall = np.r_[0.0, recall]
    return precision, recall


def _prepare_assets() -> dict[str, pd.DataFrame]:
    assets = {}
    assets["reg_main"] = _load_csv(ROOT / "outputs" / "full10y_refinedfinal_merged" / "tables" / "regression_summary.csv")
    assets["cls_main"] = _load_csv(ROOT / "outputs" / "full10y_refinedfinal_merged" / "tables" / "classification_summary.csv")
    assets["dm_main"] = _load_csv(ROOT / "outputs" / "full10y_refinedfinal_merged" / "tables" / "diebold_mariano.csv")
    assets["cw_main"] = _load_csv(ROOT / "outputs" / "full10y_refinedfinal_merged" / "tables" / "clark_west.csv")
    assets["pred_reg_main"] = _load_csv(
        ROOT / "outputs" / "full10y_refinedfinal_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    assets["pred_cls_main"] = _load_csv(
        ROOT / "outputs" / "full10y_refinedfinal_merged" / "predictions" / "classification_predictions.csv",
        parse_dates=["date"],
    )
    assets["reg_plus"] = _load_csv(ROOT / "outputs" / "plusdata500_merged" / "tables" / "regression_summary.csv")
    assets["cls_plus"] = _load_csv(ROOT / "outputs" / "plusdata500_merged" / "tables" / "classification_summary.csv")
    assets["pred_reg_plus"] = _load_csv(
        ROOT / "outputs" / "plusdata500_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    assets["pred_cls_plus"] = _load_csv(
        ROOT / "outputs" / "plusdata500_merged" / "predictions" / "classification_predictions.csv",
        parse_dates=["date"],
    )
    assets["reg_park"] = _load_csv(ROOT / "outputs" / "parkinson300_merged" / "tables" / "regression_summary.csv")
    assets["cls_park"] = _load_csv(ROOT / "outputs" / "parkinson300_merged" / "tables" / "classification_summary.csv")
    return assets


def _model_color(style: dict, model: str) -> str:
    colors = _semantic_colors(style)
    mapping = {
        "har": colors["har"],
        "hv_22": colors["hv"],
        "lightgbm": colors["lightgbm"],
        "wavelet_lightgbm": colors["wavelet"],
        "main_stacking": colors["stacking"],
        "naive_threshold": colors["naive_warning"],
        "logistic_raw": colors["logistic_warning"],
        "main_warning": colors["stacking"],
    }
    return mapping.get(model, colors["observed"])


def _semantic_colors(style: dict) -> dict[str, str]:
    palette = style["palette"]["series"]
    return {
        "observed": palette[0],
        "har": palette[5] if len(palette) > 5 else palette[0],
        "hv": palette[4] if len(palette) > 4 else palette[5],
        "lightgbm": palette[1],
        "wavelet": palette[3],
        "stacking": palette[2],
        "logistic_warning": palette[1],
        "naive_warning": palette[5] if len(palette) > 5 else palette[2],
        "stress_fill": palette[3],
    }


def _label_above_axes(ax, text: str, *, x: float = 0.5, y: float = 1.045, ha: str = "center") -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        clip_on=False,
    )


def _save_dual(fig, output_path: Path, style: dict) -> None:
    save_figure(fig, output_path, style)
    png_path = output_path.with_suffix(".png")
    fig.savefig(
        png_path,
        dpi=style["figure"]["save_dpi"],
        bbox_inches=style["figure"]["bbox_inches"],
        pad_inches=style["figure"]["pad_inches"],
        transparent=style["export"]["transparent"],
    )


def _styled_subplots(category: str, nrows: int, ncols: int, **kwargs):
    style = load_plot_style(category)
    apply_plot_style(style)
    width = kwargs.pop("fig_width_in", style.get("fig_width_in", 6.0))
    height = kwargs.pop("fig_height_in", style.get("fig_height_in", 3.0))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        constrained_layout=style["figure"]["constrained_layout"],
        **kwargs,
    )
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        configure_axes(ax, style)
    return fig, axes, style


def render_flowchart(output_path: Path) -> None:
    style = load_plot_style("timeseries")
    apply_plot_style(style)
    fig, ax = plt.subplots(figsize=(6.9, 2.6), constrained_layout=True)
    ax.set_axis_off()
    colors = _semantic_colors(style)
    edge = style["axes"]["edgecolor"]

    boxes = [
        (0.03, 0.32, 0.16, 0.36, "Public Data\nOHLC + FRED"),
        (0.23, 0.32, 0.17, 0.36, "Volatility Target\nRS / Parkinson"),
        (0.45, 0.32, 0.18, 0.36, "Multiscale Features\nHAR + Wavelet"),
        (0.68, 0.32, 0.14, 0.36, "Forecasting\nHybrid ML"),
        (0.86, 0.32, 0.11, 0.36, "Warning\nProbability"),
    ]
    fills = [colors["observed"], colors["har"], colors["stacking"], colors["lightgbm"], colors["wavelet"]]
    for (x, y, w, h, text), color in zip(boxes, fills):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=color,
            alpha=0.13,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes)

    for left, right in zip(boxes[:-1], boxes[1:]):
        x1 = left[0] + left[2]
        x2 = right[0]
        y = left[1] + left[3] / 2
        arrow = FancyArrowPatch(
            (x1 + 0.01, y),
            (x2 - 0.01, y),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=edge,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)

    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_protocol_timeline(output_path: Path) -> None:
    style = load_plot_style("timeseries")
    apply_plot_style(style)
    colors = _semantic_colors(style)

    fig, ax = plt.subplots(figsize=(7.6, 2.55), constrained_layout=True)
    configure_axes(ax, style)

    full_start = pd.Timestamp("2005-02-25")
    design_end = pd.Timestamp("2015-12-31")
    oos_start = pd.Timestamp("2016-01-04")
    full_end = pd.Timestamp("2025-12-31")

    bands = [
        ("Daily market sample", full_start, full_end, colors["observed"], 2.2),
        ("Design and model freeze", full_start, design_end, colors["hv"], 1.2),
        ("Rolling OOS evaluation", oos_start, full_end, colors["wavelet"], 0.2),
    ]
    for label, start, end, color, y in bands:
        ax.barh(
            y=y,
            width=(end - start).days,
            left=start,
            height=0.56,
            color=color,
            alpha=0.88,
            edgecolor=style["axes"]["edgecolor"],
            linewidth=0.9,
        )
        ax.text(start + (end - start) / 2, y, label, ha="center", va="center", color="#FFFFFF", fontsize=9.0)

    ax.axvline(oos_start, color=style["palette"]["neutral_mid"], linewidth=1.0, linestyle="--")
    ax.text(oos_start, 2.78, "OOS starts", ha="center", va="bottom", fontsize=8.7)
    ax.text(
        pd.Timestamp("2021-02-01"),
        -0.48,
        "Rolling window = 1260 trading days    Refit every 5 days    Horizons = 1, 5, 10",
        ha="center",
        va="center",
        fontsize=8.9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=style["figure"]["facecolor"],
            edgecolor=style["axes"]["edgecolor"],
            linewidth=0.8,
        ),
    )

    ax.set_xlim(pd.Timestamp("2004-10-01"), pd.Timestamp("2026-03-01"))
    ax.set_ylim(-0.9, 2.8)
    ax.set_yticks([])
    ax.set_xlabel("Calendar time")
    ax.xaxis.set_major_locator(MonthLocator(bymonth=(1,), interval=2))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(MonthLocator(interval=6))
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_model_architecture(output_path: Path) -> None:
    style = load_plot_style("timeseries")
    apply_plot_style(style)
    colors = _semantic_colors(style)
    edge = style["axes"]["edgecolor"]

    fig, ax = plt.subplots(figsize=(7.6, 3.0), constrained_layout=True)
    ax.set_axis_off()

    blocks = [
        (0.03, 0.62, 0.16, 0.22, "HAR block\n$y_t,y_t^{(5)},y_t^{(22)}$", colors["har"]),
        (0.03, 0.30, 0.16, 0.22, "Raw features\nreturns + macro + VIX", colors["lightgbm"]),
        (0.03, 0.00, 0.16, 0.22, "Wavelet block\n$\\mu^{(j)},\\sigma^{(j)},E^{(j)}$", colors["wavelet"]),
        (0.30, 0.22, 0.18, 0.40, "Feature fusion\n$[x^{HAR},x^{raw},x^{wav}]$", colors["stacking"]),
        (0.57, 0.22, 0.16, 0.40, "Forecast head\nLightGBM for $\\hat y^{(h)}$", colors["lightgbm"]),
        (0.80, 0.52, 0.16, 0.22, "Calibration layer\n$\\tilde y=\\max(\\hat y,q_\\alpha)$", colors["hv"]),
        (0.80, 0.18, 0.16, 0.22, "Warning layer\n$\\Pr(z=1\\mid x^{warn})$", colors["wavelet"]),
    ]

    for x, y, w, h, text, color in blocks:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=color,
            alpha=0.14,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes, fontsize=9.0)

    arrows = [
        ((0.19, 0.73), (0.30, 0.50)),
        ((0.19, 0.41), (0.30, 0.42)),
        ((0.19, 0.11), (0.30, 0.34)),
        ((0.48, 0.42), (0.57, 0.42)),
        ((0.73, 0.52), (0.80, 0.63)),
        ((0.73, 0.32), (0.80, 0.29)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.3,
                color=edge,
                transform=ax.transAxes,
            )
        )

    ax.text(0.885, 0.79, "Calibrated\nvolatility forecast", ha="center", va="center", transform=ax.transAxes, fontsize=8.8)
    ax.text(0.885, 0.05, "Early-warning\nprobability", ha="center", va="center", transform=ax.transAxes, fontsize=8.8)

    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_main_qlike_bars(reg_main: pd.DataFrame, output_path: Path) -> None:
    fig, ax, style = _styled_subplots("bar", 1, 1, fig_width_in=7.8, fig_height_in=3.35)
    df = reg_main.loc[reg_main["model"].isin(MAIN_MODELS)].copy()
    groups = [(i, h) for i in ["spx", "ndq", "dji"] for h in [1, 5, 10]]
    model_order = MAIN_MODELS
    x = np.arange(len(groups))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(model_order))

    best = {
        (r["index_id"], int(r["horizon"])): r["model"]
        for _, r in df.loc[df.groupby(["index_id", "horizon"])["qlike"].idxmin()].iterrows()
    }

    for idx, model in enumerate(model_order):
        vals = []
        for index_id, horizon in groups:
            row = df.loc[(df["index_id"] == index_id) & (df["horizon"] == horizon) & (df["model"] == model)].iloc[0]
            vals.append(float(row["qlike"]))
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            color=_model_color(style, model),
            alpha=0.92 if model != "wavelet_lightgbm" else 1.0,
            edgecolor=style["axes"]["edgecolor"],
            linewidth=style.get("bar_linewidth", 0.8),
            label=MODEL_LABELS[model],
        )
        for bar, group, val in zip(bars, groups, vals):
            if best[group] == model:
                ax.scatter(
                    bar.get_x() + bar.get_width() / 2,
                    val - 0.02,
                    s=28,
                    color=style["palette"]["series"][3],
                    edgecolors=style["axes"]["edgecolor"],
                    linewidths=0.6,
                    zorder=4,
                )

    ax.axhline(0.0, color=style["palette"]["neutral_mid"], linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_ylabel("QLIKE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT_INDEX_LABELS[i]}\n$h={h}$" for i, h in groups])
    ax.tick_params(axis="x", labelsize=9.0)
    ax.margins(x=0.05)
    ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.16))
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_main_heatmap(reg_main: pd.DataFrame, output_path: Path) -> None:
    fig, ax, style = _styled_subplots("heatmap", 1, 1, fig_width_in=7.2, fig_height_in=3.25)
    df = reg_main.loc[reg_main["model"].isin(MAIN_MODELS)].copy()
    colors = _semantic_colors(style)

    winner_color = {
        "har": colors["har"],
        "hv_22": colors["hv"],
        "lightgbm": colors["lightgbm"],
        "wavelet_lightgbm": colors["wavelet"],
    }
    winner_short = {
        "har": "HAR",
        "hv_22": "HV-22",
        "lightgbm": "LGBM",
        "wavelet_lightgbm": "W-LGBM",
    }

    indices = ["spx", "ndq", "dji"]
    horizons = [1, 5, 10]
    ax.set_xlim(0, len(horizons))
    ax.set_ylim(0, len(indices))
    ax.invert_yaxis()

    for r, index_id in enumerate(indices):
        for c, horizon in enumerate(horizons):
            cell = (
                df.loc[(df["index_id"] == index_id) & (df["horizon"] == horizon), ["model", "qlike"]]
                .sort_values("qlike")
                .reset_index(drop=True)
            )
            winner = cell.loc[0, "model"]
            runner_up = cell.loc[1, "model"]
            margin = float(cell.loc[1, "qlike"] - cell.loc[0, "qlike"])
            face = winner_color[winner]
            patch = FancyBboxPatch(
                (c + 0.06, r + 0.08),
                0.88,
                0.84,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                linewidth=1.1,
                edgecolor=style["axes"]["edgecolor"],
                facecolor=face,
                alpha=0.18,
            )
            ax.add_patch(patch)
            ax.text(
                c + 0.50,
                r + 0.34,
                winner_short[winner],
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="bold",
            )
            ax.text(
                c + 0.50,
                r + 0.58,
                rf"$\Delta={margin:.3f}$",
                ha="center",
                va="center",
                fontsize=8.7,
            )
            ax.text(
                c + 0.50,
                r + 0.76,
                f"2nd: {winner_short[runner_up]}",
                ha="center",
                va="center",
                fontsize=7.9,
            )

    ax.set_xticks(np.arange(len(horizons)) + 0.5)
    ax.set_xticklabels([rf"$h={h}$" for h in horizons])
    ax.set_yticks(np.arange(len(indices)) + 0.5)
    ax.set_yticklabels([INDEX_LABELS[i] for i in indices])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Index")
    ax.grid(False)

    legend_handles = [
        Patch(facecolor=winner_color[m], edgecolor=style["axes"]["edgecolor"], alpha=0.22, label=MODEL_LABELS[m])
        for m in MAIN_MODELS
    ]
    ax.legend(handles=legend_handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.18), frameon=False)
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_robustness_lines(assets: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax, style = _styled_subplots("ablation", 1, 1)
    compare = []
    package_map = {
        "main": assets["reg_main"],
        "plusdata": assets["reg_plus"],
        "parkinson": assets["reg_park"],
    }
    for pkg_name, df in package_map.items():
        sub = df.loc[df["model"].isin(["har", "lightgbm", "wavelet_lightgbm"])]
        means = sub.groupby("model", as_index=False)["qlike"].mean()
        means["package"] = pkg_name
        compare.append(means)
    comp = pd.concat(compare, ignore_index=True)
    x = np.arange(3)
    order = ["main", "plusdata", "parkinson"]
    plotted_values = []
    x_offsets = {"har": -0.06, "lightgbm": 0.0, "wavelet_lightgbm": 0.06}
    y_offsets = {"har": 0.018, "lightgbm": 0.036, "wavelet_lightgbm": 0.054}
    for model in ["har", "lightgbm", "wavelet_lightgbm"]:
        vals = [float(comp.loc[(comp["package"] == pkg) & (comp["model"] == model), "qlike"].iloc[0]) for pkg in order]
        plotted_values.extend(vals)
        ax.plot(
            x,
            vals,
            marker=style.get("marker", "o"),
            markersize=style.get("marker_size", 4.5),
            linewidth=style["lines"]["line_width"],
            color=_model_color(style, model),
            label=MODEL_LABELS[model],
        )
        for xi, yi in zip(x, vals):
            ax.text(
                xi + x_offsets[model],
                yi + y_offsets[model],
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Main RS\nTarget",
            "Expanded Public\nRisk Data",
            "Parkinson\nTarget",
        ]
    )
    ax.set_ylabel("Average QLIKE")
    ymin = min(plotted_values)
    ymax = max(plotted_values)
    yrange = max(ymax - ymin, 0.10)
    ax.set_ylim(ymin - 0.10 * yrange, ymax + 0.22 * yrange)
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.18))
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_forecast_panels(preds: pd.DataFrame, output_path: Path) -> None:
    fig, axes, style = _styled_subplots("timeseries", 1, 3, fig_width_in=7.8, fig_height_in=3.0)
    panels = [
        ("spx", 1, "wavelet_lightgbm", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30")),
        ("ndq", 10, "wavelet_lightgbm", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30")),
        ("dji", 10, "wavelet_lightgbm", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30")),
    ]
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    colors = _semantic_colors(style)
    for ax, (index_id, horizon, model, start, end) in zip(axes.ravel(), panels):
        sub = preds.loc[
            (preds["index_id"] == index_id)
            & (preds["horizon"] == horizon)
            & (preds["model"] == model)
            & (preds["date"] >= start)
            & (preds["date"] <= end)
        ].sort_values("date")
        har = preds.loc[
            (preds["index_id"] == index_id)
            & (preds["horizon"] == horizon)
            & (preds["model"] == "har")
            & (preds["date"] >= start)
            & (preds["date"] <= end)
        ].sort_values("date")
        crisis_start = pd.Timestamp("2020-02-15")
        crisis_end = pd.Timestamp("2020-06-15")
        ax.axvspan(crisis_start, crisis_end, color=colors["stress_fill"], alpha=0.08, zorder=0)
        ax.plot(sub["date"], sub["y_true_var"], color=colors["observed"], linewidth=2.0, label="Observed")
        ax.plot(har["date"], har["y_pred_var"], color=colors["har"], linewidth=1.5, linestyle="--", label="HAR")
        ax.plot(sub["date"], sub["y_pred_var"], color=colors["wavelet"], linewidth=1.8, label="Wavelet-LightGBM")
        ax.set_xlabel("")
        ax.set_ylabel("Variance" if index_id == "spx" else "")
        ax.xaxis.set_major_locator(MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        _label_above_axes(ax, f"{INDEX_LABELS[index_id]}, $h={horizon}$")
    axes.ravel()[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncols=3, frameon=False)
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_warning_panel(preds_main: pd.DataFrame, preds_plus: pd.DataFrame, output_path: Path) -> None:
    fig, axes, style = _styled_subplots("roc_pr", 1, 2, fig_width_in=6.6, fig_height_in=2.7)
    settings = [
        ("Main RS Target", preds_main),
        ("Expanded Public Risk Data", preds_plus),
    ]
    colors = _semantic_colors(style)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (title, df) in zip(axes.ravel(), settings):
        for model in WARNING_MODELS:
            sub = df.loc[
                (df["index_id"] == "spx")
                & (df["horizon"] == 1)
                & (df["model"] == model)
            ].dropna(subset=["y_true_label", "y_score"])
            y_true = sub["y_true_label"].to_numpy(dtype=int)
            y_score = sub["y_score"].to_numpy(dtype=float)
            if len(np.unique(y_true)) < 2:
                continue
            precision, recall = _precision_recall_curve(y_true, y_score)
            ax.plot(
                recall,
                precision,
                color=_model_color(style, model),
                linewidth=2.0,
                linestyle="--" if model == "naive_threshold" else "-",
                label=MODEL_LABELS[model],
            )
        baseline = df.loc[(df["index_id"] == "spx") & (df["horizon"] == 1), "y_true_label"].mean()
        ax.axhline(float(baseline), color=colors["har"], linestyle=":", linewidth=1.0, alpha=0.85)
        _label_above_axes(ax, title, x=0.03, ha="left")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision" if title == "Main RS Target" else "")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    axes.ravel()[0].legend(loc="lower left", frameon=False)
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_delta_vs_har(assets: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes, style = _styled_subplots("delta", 1, 3, fig_width_in=8.2, fig_height_in=3.1)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    settings = [
        ("Main RS", assets["reg_main"]),
        ("Expanded Data", assets["reg_plus"]),
        ("Parkinson", assets["reg_park"]),
    ]
    models = ["lightgbm", "wavelet_lightgbm"]
    palette = [_model_color(style, m) for m in models]
    groups = [(i, h) for i in ["spx", "ndq", "dji"] for h in [1, 5, 10]]
    x = np.arange(len(groups))
    width = 0.34
    for ax, (title, df) in zip(axes.ravel(), settings):
        har_map = {
            (r["index_id"], int(r["horizon"])): float(r["qlike"])
            for _, r in df.loc[df["model"] == "har"].iterrows()
        }
        for idx, model in enumerate(models):
            vals = []
            for group in groups:
                q = float(df.loc[(df["index_id"] == group[0]) & (df["horizon"] == group[1]) & (df["model"] == model), "qlike"].iloc[0])
                vals.append(q - har_map[group])
            ax.bar(
                x + (-0.5 + idx) * width,
                vals,
                width=width,
                color=palette[idx],
                edgecolor=style["axes"]["edgecolor"],
                linewidth=0.8,
                label=MODEL_LABELS[model],
            )
        ax.axhline(0.0, color=style["palette"]["neutral_mid"], linewidth=0.9, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i.upper()}\n{h}" for i, h in groups])
        ax.tick_params(axis="x", labelsize=7.0, pad=1.5)
        _label_above_axes(ax, title, x=0.03, ha="left")
        ax.set_ylabel(r"$\Delta$QLIKE vs HAR" if ax is axes.ravel()[0] else "")
    axes.ravel()[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncols=2, frameon=False)
    _save_dual(fig, output_path, style)
    plt.close(fig)


def render_warning_score_distribution(
    preds_main: pd.DataFrame,
    preds_plus: pd.DataFrame,
    reg_main: pd.DataFrame,
    reg_plus: pd.DataFrame,
    output_path: Path,
) -> None:
    style = load_plot_style("heatmap")
    apply_plot_style(style)
    colors = _semantic_colors(style)
    palette = style["palette"]
    cmap = LinearSegmentedColormap.from_list("risk_signal", palette["heatmap_sequential"])

    fig = plt.figure(figsize=(8.2, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.06, wspace=0.10)
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    for ax in axes.ravel():
        configure_axes(ax, style)

    common_start = max(
        preds_main.loc[(preds_main["index_id"] == "spx") & (preds_main["model"] == "logistic_raw"), "date"].min(),
        preds_plus.loc[(preds_plus["index_id"] == "spx") & (preds_plus["model"] == "logistic_raw"), "date"].min(),
        reg_main.loc[(reg_main["index_id"] == "spx") & (reg_main["model"] == "wavelet_lightgbm") & (reg_main["horizon"] == 1), "date"].min(),
        reg_plus.loc[(reg_plus["index_id"] == "spx") & (reg_plus["model"] == "wavelet_lightgbm") & (reg_plus["horizon"] == 1), "date"].min(),
    )
    common_end = min(
        preds_main.loc[(preds_main["index_id"] == "spx") & (preds_main["model"] == "logistic_raw"), "date"].max(),
        preds_plus.loc[(preds_plus["index_id"] == "spx") & (preds_plus["model"] == "logistic_raw"), "date"].max(),
        reg_main.loc[(reg_main["index_id"] == "spx") & (reg_main["model"] == "wavelet_lightgbm") & (reg_main["horizon"] == 1), "date"].max(),
        reg_plus.loc[(reg_plus["index_id"] == "spx") & (reg_plus["model"] == "wavelet_lightgbm") & (reg_plus["horizon"] == 1), "date"].max(),
    )

    def prepare_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        sub = df.loc[
            (df["index_id"] == "spx")
            & (df["model"] == "logistic_raw")
            & (df["date"] >= common_start)
            & (df["date"] <= common_end)
        ].copy()
        score = sub.pivot(index="horizon", columns="date", values="y_score").sort_index()
        label = sub.pivot(index="horizon", columns="date", values="y_true_label").sort_index()
        return score, label

    def prepare_vol(df: pd.DataFrame) -> pd.DataFrame:
        sub = df.loc[
            (df["index_id"] == "spx")
            & (df["model"] == "wavelet_lightgbm")
            & (df["horizon"] == 1)
            & (df["date"] >= common_start)
            & (df["date"] <= common_end)
        ].copy()
        return sub.sort_values("date")

    settings = [
        ("Main RS target", prepare_vol(reg_main), *prepare_scores(preds_main)),
        ("Expanded public risk data", prepare_vol(reg_plus), *prepare_scores(preds_plus)),
    ]

    im = None
    for col, (title, vol_df, score, label) in enumerate(settings):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        ax_top.plot(vol_df["date"], vol_df["y_true_var"], color=colors["observed"], linewidth=1.9, label="Observed variance")
        ax_top.plot(vol_df["date"], vol_df["y_pred_var"], color=colors["wavelet"], linewidth=1.6, alpha=0.95, label="Wavelet-LightGBM")
        ax_top.plot(vol_df["date"], vol_df["warning_threshold"], color=colors["har"], linewidth=1.25, linestyle="--", label="Dynamic risk threshold")
        risk_mask = (vol_df["y_true_var"] > vol_df["warning_threshold"]).to_numpy(dtype=bool)
        ax_top.fill_between(
            vol_df["date"],
            vol_df["y_true_var"],
            vol_df["warning_threshold"],
            where=risk_mask,
            color=colors["lightgbm"],
            alpha=0.12,
        )
        ax_top.set_ylabel("Variance" if col == 0 else "")
        ax_top.set_xticklabels([])
        _label_above_axes(ax_top, title, x=0.02, y=1.005, ha="left")
        ax_top.set_xlim(common_start, common_end)

        vals = score.to_numpy(dtype=float)
        im = ax_bottom.imshow(vals, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax_bottom.set_yticks(np.arange(len(score.index)))
        ax_bottom.set_yticklabels([f"$h={int(h)}$" for h in score.index] if col == 0 else [""] * len(score.index))
        cols = list(score.columns)
        tick_positions = np.linspace(0, len(cols) - 1, 5, dtype=int).tolist()
        ax_bottom.set_xticks(tick_positions)
        ax_bottom.set_xticklabels([pd.Timestamp(cols[i]).strftime("%Y-%m") for i in tick_positions], rotation=30, ha="right")
        event_y, event_x = np.where(label.to_numpy(dtype=float) > 0.5)
        ax_bottom.scatter(
            event_x,
            event_y,
            marker="s",
            s=10,
            facecolors="none",
            edgecolors=palette["neutral_dark"],
            linewidths=0.55,
        )
        ax_bottom.set_ylabel("Horizon" if col == 0 else "")
        ax_bottom.set_xlabel("Date")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncols=3, frameon=False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.96, location="right")
    cbar.set_label("Warning score")
    _save_dual(fig, output_path, style)
    plt.close(fig)


def _export_table(df: pd.DataFrame, csv_path: Path, tex_path: Path, caption: str, label: str, float_format: dict[str, str] | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    fmt_map = float_format or {}

    def format_cell(col: str, value) -> str:
        if pd.isna(value):
            return ""
        if col in fmt_map and isinstance(value, (float, np.floating)):
            return format(float(value), fmt_map[col])
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        text = str(value)
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "_": r"\_",
            "#": r"\#",
        }
        for src, tgt in replacements.items():
            text = text.replace(src, tgt)
        return text

    header = " & ".join(df.columns) + r" \\"
    body_lines = []
    for _, row in df.iterrows():
        cells = [format_cell(col, row[col]) for col in df.columns]
        body_lines.append(" & ".join(cells) + r" \\")

    latex = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3.5pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{" + ("l" * len(df.columns)) + r"}",
            r"\toprule",
            header,
            r"\midrule",
            *body_lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text(latex, encoding="utf-8")


def _latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    out = str(text)
    for src, tgt in replacements.items():
        out = out.replace(src, tgt)
    return out


def _apply_rank_macros(values: list[float], *, higher_is_better: bool) -> list[str | None]:
    order = np.argsort(np.asarray(values, dtype=float))
    if higher_is_better:
        order = order[::-1]
    macros = [None] * len(values)
    for pos, idx in enumerate(order[:3]):
        macros[idx] = ["best", "second", "third"][pos]
    return macros


def _styled_num(value: float, *, fmt: str = ".4f", macro: str | None = None) -> str:
    text = format(float(value), fmt)
    if macro is None:
        return text
    return rf"\{macro}{{{text}}}"


def _write_main_regression_table(reg_main: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    models = MAIN_MODELS
    reg = reg_main.loc[reg_main["model"].isin(models)].copy()
    reg.to_csv(csv_path, index=False)
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Main regression QLIKE across indices and forecast horizons under the Rogers--Satchell target. Best, second-best, and third-best values within each index--horizon cell are marked by \best{bold shading}, \second{underline}, and \third{light shading}, respectively.}",
        r"\label{tab:main_regression}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Index & Model & $h=1$ & $h=5$ & $h=10$ \\",
        r"\midrule",
    ]
    for idx_pos, index_id in enumerate(["spx", "ndq", "dji"]):
        block = reg.loc[reg["index_id"] == index_id].copy()
        rank_map: dict[tuple[int, str], str | None] = {}
        for horizon in [1, 5, 10]:
            vals = [float(block.loc[(block["horizon"] == horizon) & (block["model"] == m), "qlike"].iloc[0]) for m in models]
            macros = _apply_rank_macros(vals, higher_is_better=False)
            for model, macro in zip(models, macros):
                rank_map[(horizon, model)] = macro
        for row_pos, model in enumerate(models):
            index_label = INDEX_LABELS[index_id] if row_pos == 0 else ""
            row = [ _latex_escape(index_label), MODEL_LABELS[model] ]
            for horizon in [1, 5, 10]:
                value = float(block.loc[(block["horizon"] == horizon) & (block["model"] == model), "qlike"].iloc[0])
                row.append(_styled_num(value, fmt=".4f", macro=rank_map[(horizon, model)]))
            lines.append(" & ".join(row) + r" \\")
        if idx_pos < 2:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_warning_table(cls_main: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    models = WARNING_MODELS
    cls = cls_main.loc[cls_main["model"].isin(models)].copy()
    cls.to_csv(csv_path, index=False)
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Main warning results under the Rogers--Satchell target. PR-AUC is maximized and Brier score is minimized within each index--horizon cell.}",
        r"\label{tab:warning_results}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5.2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Index & Model & PR$(1)$ & Brier$(1)$ & PR$(5)$ & Brier$(5)$ & PR$(10)$ & Brier$(10)$ \\",
        r"\midrule",
    ]
    for idx_pos, index_id in enumerate(["spx", "ndq", "dji"]):
        block = cls.loc[cls["index_id"] == index_id].copy()
        pr_rank: dict[tuple[int, str], str | None] = {}
        br_rank: dict[tuple[int, str], str | None] = {}
        for horizon in [1, 5, 10]:
            pr_vals = [float(block.loc[(block["horizon"] == horizon) & (block["model"] == m), "pr_auc"].iloc[0]) for m in models]
            br_vals = [float(block.loc[(block["horizon"] == horizon) & (block["model"] == m), "brier"].iloc[0]) for m in models]
            for model, macro in zip(models, _apply_rank_macros(pr_vals, higher_is_better=True)):
                pr_rank[(horizon, model)] = macro
            for model, macro in zip(models, _apply_rank_macros(br_vals, higher_is_better=False)):
                br_rank[(horizon, model)] = macro
        for row_pos, model in enumerate(models):
            index_label = INDEX_LABELS[index_id] if row_pos == 0 else ""
            row = [_latex_escape(index_label), MODEL_LABELS[model]]
            for horizon in [1, 5, 10]:
                pr = float(block.loc[(block["horizon"] == horizon) & (block["model"] == model), "pr_auc"].iloc[0])
                br = float(block.loc[(block["horizon"] == horizon) & (block["model"] == model), "brier"].iloc[0])
                row.append(_styled_num(pr, fmt=".4f", macro=pr_rank[(horizon, model)]))
                row.append(_styled_num(br, fmt=".4f", macro=br_rank[(horizon, model)]))
            lines.append(" & ".join(row) + r" \\")
        if idx_pos < 2:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def _write_robustness_table(assets: dict[str, pd.DataFrame], csv_path: Path, tex_path: Path) -> None:
    rows = []
    settings = [("Main RS", assets["reg_main"]), ("Expanded Data", assets["reg_plus"]), ("Parkinson", assets["reg_park"])]
    models = ["har", "lightgbm", "wavelet_lightgbm"]
    for setting, df in settings:
        row = {"Setting": setting}
        for model in models:
            row[MODEL_LABELS[model]] = float(df.loc[df["model"] == model, "qlike"].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(csv_path, index=False)
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Average QLIKE across the main and robustness settings. Ranking is applied within each row.}",
        r"\label{tab:robustness_summary}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Setting & HAR & LightGBM & Wavelet-LightGBM \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        vals = [row["HAR"], row["LightGBM"], row["Wavelet-LightGBM"]]
        macros = _apply_rank_macros([float(v) for v in vals], higher_is_better=False)
        styled = [_styled_num(v, fmt=".4f", macro=m) for v, m in zip(vals, macros)]
        lines.append(_latex_escape(row["Setting"]) + " & " + " & ".join(styled) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def export_tables(assets: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_main = assets["reg_main"].copy()
    _write_main_regression_table(
        reg_main,
        output_dir / "table_main_regression.csv",
        output_dir / "table_main_regression.tex",
    )

    cls_main = assets["cls_main"].copy()
    _write_warning_table(
        cls_main,
        output_dir / "table_warning.csv",
        output_dir / "table_warning.tex",
    )

    _write_robustness_table(
        assets,
        output_dir / "table_robustness.csv",
        output_dir / "table_robustness.tex",
    )

    dm = assets["dm_main"].copy()
    cw = assets["cw_main"].copy()
    dm = dm.loc[dm["model"].isin(["lightgbm", "wavelet_lightgbm"])].rename(columns={"p_value": "DM p-value", "dm_stat": "DM stat"})
    cw = cw.loc[cw["model"].isin(["lightgbm", "wavelet_lightgbm"])].rename(columns={"p_value": "CW p-value", "cw_stat": "CW stat"})
    stat = dm.merge(cw, on=["index_id", "horizon", "model"], how="outer")
    stat["Index"] = stat["index_id"].map(INDEX_LABELS)
    stat["Model"] = stat["model"].map(MODEL_LABELS)
    stat = stat[["Index", "horizon", "Model", "DM stat", "DM p-value", "CW stat", "CW p-value"]].rename(columns={"horizon": "Horizon"})
    stat = stat.sort_values(["Index", "Horizon", "Model"])
    _export_table(
        stat,
        output_dir / "table_significance.csv",
        output_dir / "table_significance.tex",
        "Forecast comparison tests against HAR in the main specification.",
        "tab:significance_tests",
        {"DM stat": ".3f", "DM p-value": ".4f", "CW stat": ".3f", "CW p-value": ".4f"},
    )


def main() -> None:
    args = parse_args()
    manuscript_dir = Path(args.output_dir)
    fig_dir = manuscript_dir / "figures"
    tab_dir = manuscript_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    assets = _prepare_assets()
    render_flowchart(fig_dir / "fig01_method_flow.pdf")
    render_protocol_timeline(fig_dir / "fig09_protocol_timeline.pdf")
    render_model_architecture(fig_dir / "fig10_model_architecture.pdf")
    render_main_qlike_bars(assets["reg_main"], fig_dir / "fig02_main_qlike_bars.pdf")
    render_main_heatmap(assets["reg_main"], fig_dir / "fig03_main_qlike_heatmap.pdf")
    render_robustness_lines(assets, fig_dir / "fig04_robustness_comparison.pdf")
    render_forecast_panels(assets["pred_reg_main"], fig_dir / "fig05_forecast_panels.pdf")
    render_warning_panel(assets["pred_cls_main"], assets["pred_cls_plus"], fig_dir / "fig06_warning_pr_panels.pdf")
    render_delta_vs_har(assets, fig_dir / "fig07_delta_vs_har.pdf")
    render_warning_score_distribution(
        assets["pred_cls_main"],
        assets["pred_cls_plus"],
        assets["pred_reg_main"],
        assets["pred_reg_plus"],
        fig_dir / "fig08_warning_score_distribution.pdf",
    )
    export_tables(assets, tab_dir)
    print(f"Generated manuscript assets in {manuscript_dir}")


if __name__ == "__main__":
    main()
