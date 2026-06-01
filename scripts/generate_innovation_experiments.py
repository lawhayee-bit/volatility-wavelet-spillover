from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import dates as mdates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting import apply_plot_style, configure_axes, load_plot_style, save_figure


MANUSCRIPT_DIR = ROOT / "paper" / "manuscript"
FIG_DIR = MANUSCRIPT_DIR / "figures"
TABLE_DIR = MANUSCRIPT_DIR / "tables"

INDEX_LABELS = {"spx": "S&P 500", "ndq": "Nasdaq-100", "dji": "DJIA"}
MODEL_LABELS = {"naive_threshold": "Naive-Threshold", "logistic_raw": "Logistic-Raw"}


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.clip(np.asarray(y_true, dtype=float), 1e-12, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-12, None)
    return np.log(y_pred) + y_true / y_pred


def _save_dual(fig, output_path: Path, style: dict) -> None:
    save_figure(fig, output_path, style)
    fig.savefig(
        output_path.with_suffix(".png"),
        dpi=style["figure"]["save_dpi"],
        bbox_inches=style["figure"]["bbox_inches"],
        pad_inches=style["figure"]["pad_inches"],
        transparent=style["export"]["transparent"],
    )


def _best_text(value: float, values: pd.Series, higher_is_better: bool) -> str:
    if higher_is_better:
        is_best = np.isclose(value, values.max())
    else:
        is_best = np.isclose(value, values.min())
    body = f"{value:.3f}"
    return f"\\best{{{body}}}" if is_best else body


def build_quantile_conditioned_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(
        ROOT / "outputs" / "full10y_refinedfinal_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    reg = reg[reg["model"].isin(["har", "wavelet_lightgbm"])].copy()
    base = reg[["date", "index_id", "horizon", "y_true_var"]].drop_duplicates().copy()
    base["decile"] = base.groupby(["index_id", "horizon"])["y_true_var"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
    )
    reg = reg.merge(
        base[["date", "index_id", "horizon", "decile"]],
        on=["date", "index_id", "horizon"],
        how="left",
    )
    reg["qlike_obs"] = qlike(reg["y_true_var"], reg["y_pred_var"])
    wide = (
        reg.pivot_table(
            index=["date", "index_id", "horizon", "decile"],
            columns="model",
            values="qlike_obs",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    wide["delta_wavelet_vs_har"] = wide["wavelet_lightgbm"] - wide["har"]
    quantile_summary = (
        wide.groupby(["horizon", "decile"])["delta_wavelet_vs_har"]
        .mean()
        .reset_index()
        .sort_values(["decile", "horizon"])
    )

    spill = pd.read_csv(
        ROOT / "outputs" / "spillover_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    spill = spill[
        spill["model"].isin(["wavelet_lightgbm", "spillover_lightgbm"])
        & spill["index_id"].isin(["dji", "spx"])
    ].copy()
    sbase = spill[["date", "index_id", "horizon", "y_true_var"]].drop_duplicates().copy()
    sbase["decile"] = sbase.groupby(["index_id", "horizon"])["y_true_var"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
    )
    spill = spill.merge(
        sbase[["date", "index_id", "horizon", "decile"]],
        on=["date", "index_id", "horizon"],
        how="left",
    )
    spill["qlike_obs"] = qlike(spill["y_true_var"], spill["y_pred_var"])
    swide = (
        spill.pivot_table(
            index=["date", "index_id", "horizon", "decile"],
            columns="model",
            values="qlike_obs",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    swide["delta_spill_vs_wavelet"] = swide["spillover_lightgbm"] - swide["wavelet_lightgbm"]
    spill_summary = (
        swide.groupby(["horizon", "decile"])["delta_spill_vs_wavelet"]
        .mean()
        .reset_index()
        .sort_values(["decile", "horizon"])
    )

    quantile_summary.to_csv(TABLE_DIR / "table_quantile_conditioned_delta.csv", index=False)
    spill_summary.to_csv(TABLE_DIR / "table_spillover_quantile_delta.csv", index=False)
    return quantile_summary, spill_summary


def render_quantile_conditioned_figure(
    quantile_summary: pd.DataFrame,
    spill_summary: pd.DataFrame,
) -> None:
    style = load_plot_style("heatmap")
    apply_plot_style(style)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.3, 3.3),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    axes = np.atleast_1d(axes)
    for ax in axes:
        configure_axes(ax, style)

    palette = style["palette"]["heatmap_diverging"]
    cmap = LinearSegmentedColormap.from_list("paper_diverging", palette, N=256)

    left = quantile_summary.pivot(index="decile", columns="horizon", values="delta_wavelet_vs_har")
    right = spill_summary.pivot(index="decile", columns="horizon", values="delta_spill_vs_wavelet")
    vmax = max(abs(left.to_numpy()).max(), abs(right.to_numpy()).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for ax, data, panel_label in [
        (axes[0], left, "(a)"),
        (axes[1], right, "(b)"),
    ]:
        im = ax.imshow(data.to_numpy(), aspect="auto", origin="lower", cmap=cmap, norm=norm)
        ax.set_xticks(range(data.shape[1]), [f"$h={int(c)}$" for c in data.columns])
        ax.set_yticks(range(data.shape[0]), [str(int(v)) for v in data.index])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Realized-volatility decile")
        ax.text(0.01, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom")
        for i, decile in enumerate(data.index):
            for j, horizon in enumerate(data.columns):
                value = float(data.loc[decile, horizon])
                txt_color = "#FFFFFF" if abs(value) > 0.45 * vmax else style["text"]["color"]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=txt_color, fontsize=8.1)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label(r"$\Delta$QLIKE (negative = advanced model better)")

    output_path = FIG_DIR / "fig13_quantile_conditioned_delta.pdf"
    _save_dual(fig, output_path, style)
    plt.close(fig)


def build_warning_event_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    cls = pd.read_csv(
        ROOT / "outputs" / "full10y_refinedfinal_merged" / "predictions" / "classification_predictions.csv",
        parse_dates=["date"],
    )
    cls = cls[cls["model"].isin(["naive_threshold", "logistic_raw"])].copy()

    def event_metrics(df: pd.DataFrame, lead_window: int = 5) -> dict[str, float]:
        df = df.sort_values("date").copy()
        y_true = df["y_true_label"].fillna(0).astype(int).to_numpy()
        y_pred = df["y_pred_label"].fillna(0).astype(int).to_numpy()
        starts: list[int] = []
        ends: list[int] = []
        in_event = False
        for i, value in enumerate(y_true):
            if value == 1 and not in_event:
                starts.append(i)
                in_event = True
            if value == 0 and in_event:
                ends.append(i - 1)
                in_event = False
        if in_event:
            ends.append(len(y_true) - 1)

        hits = 0
        leads: list[int] = []
        pred_pos = np.where(y_pred == 1)[0]
        false_alarm_days = int(((y_pred == 1) & (y_true == 0)).sum())

        for start, _ in zip(starts, ends):
            left = max(0, start - lead_window)
            candidates = pred_pos[(pred_pos >= left) & (pred_pos <= start)]
            if len(candidates):
                hits += 1
                leads.append(int(start - candidates[0]))

        events = len(starts)
        return {
            "events": events,
            "hit_rate_5d": hits / max(events, 1),
            "median_lead": float(np.median(leads)) if leads else np.nan,
            "mean_lead": float(np.mean(leads)) if leads else np.nan,
            "false_alarm_days": false_alarm_days,
            "false_alarm_per_event": false_alarm_days / max(events, 1),
        }

    rows = []
    for (index_id, horizon, model), group in cls.groupby(["index_id", "horizon", "model"]):
        metrics = event_metrics(group, lead_window=5)
        rows.append([index_id, horizon, model, *metrics.values()])
    cell_level = pd.DataFrame(
        rows,
        columns=[
            "index_id",
            "horizon",
            "model",
            "events",
            "hit_rate_5d",
            "median_lead",
            "mean_lead",
            "false_alarm_days",
            "false_alarm_per_event",
        ],
    )

    summary = (
        cell_level.groupby(["horizon", "model"])
        .agg(
            avg_hit_rate=("hit_rate_5d", "mean"),
            avg_median_lead=("median_lead", "mean"),
            avg_false_alarm_per_event=("false_alarm_per_event", "mean"),
            total_events=("events", "sum"),
        )
        .reset_index()
    )
    cell_level.to_csv(TABLE_DIR / "table_warning_event_frontier_cells.csv", index=False)
    summary.to_csv(TABLE_DIR / "table_warning_event_timing.csv", index=False)
    return cell_level, summary


def render_warning_event_frontier(cell_level: pd.DataFrame) -> None:
    style = load_plot_style("ablation")
    apply_plot_style(style)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    colors = {
        "naive_threshold": style["palette"]["series"][5],
        "logistic_raw": style["palette"]["series"][1],
    }
    markers = {"naive_threshold": "o", "logistic_raw": "s"}
    for ax in axes:
        configure_axes(ax, style)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 60)
        ax.set_ylim(0.35, 0.90)
        ax.set_xlabel("False-alarm days per event")
    axes[0].set_ylabel("5-day event hit rate")

    text_offsets = {
        (1, "spx"): (1.04, 0.010),
        (1, "ndq"): (1.04, 0.022),
        (1, "dji"): (1.04, -0.005),
        (5, "spx"): (1.05, 0.014),
        (5, "ndq"): (1.05, 0.002),
        (5, "dji"): (1.05, -0.012),
        (10, "spx"): (1.05, -0.002),
        (10, "ndq"): (1.05, 0.018),
        (10, "dji"): (1.05, 0.003),
    }

    for ax, horizon in zip(axes, [1, 5, 10]):
        sub = cell_level[cell_level["horizon"] == horizon].copy()
        ax.text(0.5, 1.03, f"$h={horizon}$", transform=ax.transAxes, ha="center", va="bottom")
        for index_id in ["spx", "ndq", "dji"]:
            pair = sub[sub["index_id"] == index_id].set_index("model")
            if pair.empty:
                continue
            x_vals = pair["false_alarm_per_event"]
            y_vals = pair["hit_rate_5d"]
            ax.plot(
                x_vals.reindex(["naive_threshold", "logistic_raw"]),
                y_vals.reindex(["naive_threshold", "logistic_raw"]),
                color="#B0B0B0",
                linewidth=0.9,
                zorder=1,
            )
            for model in ["naive_threshold", "logistic_raw"]:
                if model not in pair.index:
                    continue
                row = pair.loc[model]
                ax.scatter(
                    row["false_alarm_per_event"],
                    row["hit_rate_5d"],
                    s=30 + 18 * float(row["mean_lead"]),
                    color=colors[model],
                    marker=markers[model],
                    edgecolor=style["axes"]["edgecolor"],
                    linewidth=0.55,
                    alpha=0.95,
                    zorder=3,
                )
                if model == "logistic_raw":
                    x_scale, y_shift = text_offsets[(horizon, index_id)]
                    ax.text(
                        row["false_alarm_per_event"] * x_scale,
                        row["hit_rate_5d"] + y_shift,
                        index_id.upper(),
                        fontsize=8.0,
                        ha="left",
                        va="bottom",
                    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["naive_threshold"],
                   markeredgecolor=style["axes"]["edgecolor"], markersize=6, label="Naive Threshold"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["logistic_raw"],
                   markeredgecolor=style["axes"]["edgecolor"], markersize=6, label="Logistic-Raw"),
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.33), ncol=2)

    output_path = FIG_DIR / "fig14_warning_event_frontier.pdf"
    _save_dual(fig, output_path, style)
    plt.close(fig)


def write_warning_event_table(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Event-based warning timing summary using a five-day pre-event detection window. Hit rate and median lead are maximized; false-alarm days per event are minimized within each horizon.}",
        "\\label{tab:warning_event_timing}",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4.6pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Horizon & Model & Hit rate & Median lead & FA / event & Events \\\\",
        "\\midrule",
    ]
    for horizon in [1, 5, 10]:
        sub = summary[summary["horizon"] == horizon].copy()
        for i, (_, row) in enumerate(sub.iterrows()):
            hit = _best_text(row["avg_hit_rate"], sub["avg_hit_rate"], higher_is_better=True)
            lead = _best_text(row["avg_median_lead"], sub["avg_median_lead"], higher_is_better=True)
            fa = _best_text(row["avg_false_alarm_per_event"], sub["avg_false_alarm_per_event"], higher_is_better=False)
            h_label = f"$h={horizon}$" if i == 0 else ""
            lines.append(
                f"{h_label} & {MODEL_LABELS[row['model']]} & {hit} & {lead} & {fa} & {int(row['total_events'])} \\\\"
            )
        if horizon != 10:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "table_warning_event_timing.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_stress_spans(
    feature_path: Path,
    sample_start: str = "2016-01-01",
    quantile: float = 0.9,
    min_obs: int = 15,
    merge_gap_days: int = 40,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    feat = pd.read_csv(feature_path, usecols=["date", "vix_like"])
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat[feat["date"] >= sample_start]
    pooled = feat.groupby("date", as_index=False)["vix_like"].mean().sort_values("date")
    threshold = pooled["vix_like"].quantile(quantile)
    stress = pooled[pooled["vix_like"] >= threshold].copy()
    if stress.empty:
        return []

    group_id = (stress["date"].diff().dt.days.fillna(1) > 4).cumsum()
    spans: list[list[pd.Timestamp]] = []
    for _, group in stress.groupby(group_id):
        if len(group) >= min_obs:
            spans.append([group["date"].min(), group["date"].max()])

    merged: list[list[pd.Timestamp]] = []
    for start, end in spans:
        if not merged or (start - merged[-1][1]).days > merge_gap_days:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def build_timevarying_gain_assets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reg = pd.read_csv(
        ROOT / "outputs" / "full10y_refinedfinal_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    reg = reg[reg["model"].isin(["har", "wavelet_lightgbm"])].copy()
    reg["qlike_obs"] = qlike(reg["y_true_var"], reg["y_pred_var"])
    reg_wide = (
        reg.pivot_table(
            index=["date", "index_id", "horizon"],
            columns="model",
            values="qlike_obs",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    reg_wide["delta_advanced_minus_baseline"] = reg_wide["wavelet_lightgbm"] - reg_wide["har"]
    reg_roll = (
        reg_wide.groupby(["date", "horizon"])["delta_advanced_minus_baseline"]
        .mean()
        .reset_index()
        .sort_values(["horizon", "date"])
    )
    reg_roll["rolling_delta"] = reg_roll.groupby("horizon")["delta_advanced_minus_baseline"].transform(
        lambda s: s.rolling(126, min_periods=42).mean()
    )
    reg_roll["comparison"] = "Wavelet-LGB vs HAR"

    spill = pd.read_csv(
        ROOT / "outputs" / "spillover_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    spill = spill[
        spill["model"].isin(["wavelet_lightgbm", "spillover_lightgbm"])
        & spill["index_id"].isin(["dji", "spx"])
    ].copy()
    spill["qlike_obs"] = qlike(spill["y_true_var"], spill["y_pred_var"])
    spill_wide = (
        spill.pivot_table(
            index=["date", "index_id", "horizon"],
            columns="model",
            values="qlike_obs",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    spill_wide["delta_advanced_minus_baseline"] = (
        spill_wide["spillover_lightgbm"] - spill_wide["wavelet_lightgbm"]
    )
    spill_roll = (
        spill_wide.groupby(["date", "horizon"])["delta_advanced_minus_baseline"]
        .mean()
        .reset_index()
        .sort_values(["horizon", "date"])
    )
    spill_roll["rolling_delta"] = spill_roll.groupby("horizon")["delta_advanced_minus_baseline"].transform(
        lambda s: s.rolling(126, min_periods=42).mean()
    )
    spill_roll["comparison"] = "Spillover-LGB vs Wavelet-LGB"

    summary_frames = []
    for frame, comparison in [
        (reg_roll, "Wavelet-LGB vs HAR"),
        (spill_roll, "Spillover-LGB vs Wavelet-LGB"),
    ]:
        summary = (
            frame.dropna(subset=["rolling_delta"])
            .groupby("horizon")
            .agg(
                rolling_negative_share=("rolling_delta", lambda s: float((s < 0).mean())),
                rolling_mean_delta=("rolling_delta", "mean"),
                rolling_min_delta=("rolling_delta", "min"),
                rolling_max_delta=("rolling_delta", "max"),
            )
            .reset_index()
        )
        summary["comparison"] = comparison
        summary_frames.append(summary)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    reg_roll.to_csv(TABLE_DIR / "table_timevarying_gain_wavelet.csv", index=False)
    spill_roll.to_csv(TABLE_DIR / "table_timevarying_gain_spillover.csv", index=False)
    summary_df.to_csv(TABLE_DIR / "table_timevarying_gain_summary.csv", index=False)
    return reg_roll, spill_roll, summary_df


def render_timevarying_gain_figure(
    reg_roll: pd.DataFrame,
    spill_roll: pd.DataFrame,
    stress_spans: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    style = load_plot_style("timeseries")
    apply_plot_style(style)
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.15), constrained_layout=True)
    axes = np.atleast_1d(axes)
    horizon_colors = {
        1: style["palette"]["series"][0],
        5: style["palette"]["series"][3],
        10: style["palette"]["series"][2],
    }

    for ax, data, panel_label in [
        (axes[0], reg_roll, "(a)"),
        (axes[1], spill_roll, "(b)"),
    ]:
        configure_axes(ax, style)
        for start, end in stress_spans:
            ax.axvspan(
                start,
                end,
                facecolor=style["palette"]["series"][3],
                alpha=0.07,
                linewidth=0.0,
                edgecolor="none",
                zorder=0,
            )
        ax.axhline(0.0, color=style["palette"]["neutral_mid"], linewidth=0.95, linestyle="--", zorder=1)
        for horizon in [1, 5, 10]:
            sub = data[data["horizon"] == horizon].dropna(subset=["rolling_delta"])
            ax.plot(
                sub["date"],
                sub["rolling_delta"],
                color=horizon_colors[horizon],
                linewidth=2.0 if horizon == 10 else 1.8,
                label=f"$h={horizon}$",
                zorder=2,
            )
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(data["date"].min(), data["date"].max())
        ax.set_xlabel("Date")
        ax.set_ylabel(r"126-day rolling $\Delta$QLIKE")
        ax.text(0.01, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom")

    handles = [
        plt.Line2D([0], [0], color=horizon_colors[h], linewidth=2.0 if h == 10 else 1.8, label=f"$h={h}$")
        for h in [1, 5, 10]
    ]
    axes[1].legend(handles=handles, loc="upper right", ncol=1)

    output_path = FIG_DIR / "fig15_timevarying_gain_paths.pdf"
    _save_dual(fig, output_path, style)
    plt.close(fig)


def build_spillover_regime_assets() -> pd.DataFrame:
    spill = pd.read_csv(
        ROOT / "outputs" / "spillover_merged" / "predictions" / "regression_predictions.csv",
        parse_dates=["date"],
    )
    spill = spill[
        spill["model"].isin(["wavelet_lightgbm", "spillover_lightgbm"])
        & spill["index_id"].isin(["dji", "spx"])
    ].copy()
    feat = pd.read_csv(
        ROOT / "data" / "processed" / "panel_features_spillover.csv.gz",
        usecols=["date", "index_id", "vix_like", "recession_dummy"],
    )
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.drop_duplicates(["date", "index_id"])
    spill = spill.merge(feat, on=["date", "index_id"], how="left")

    rows = []
    for horizon, group in spill.groupby("horizon"):
        unique = group[["date", "index_id", "vix_like", "recession_dummy"]].drop_duplicates()
        vix80 = unique["vix_like"].quantile(0.8)
        vix20 = unique["vix_like"].quantile(0.2)
        regimes = {
            "Top 20\\% VIX": unique["vix_like"] >= vix80,
            "Bottom 20\\% VIX": unique["vix_like"] <= vix20,
            "Recession": unique["recession_dummy"] >= 0.5,
            "Expansion": unique["recession_dummy"] < 0.5,
            "COVID 2020--2021": (unique["date"] >= "2020-01-01") & (unique["date"] <= "2021-12-31"),
            "Post-2022": unique["date"] >= "2022-01-01",
        }
        for regime, mask in regimes.items():
            use = unique.loc[mask, ["date", "index_id"]]
            merged = group.merge(use, on=["date", "index_id"])
            metrics = {}
            count = len(merged) // 2 if len(merged) else 0
            for model, gm in merged.groupby("model"):
                metrics[model] = qlike(gm["y_true_var"], gm["y_pred_var"]).mean()
            rows.append(
                [
                    regime,
                    horizon,
                    count,
                    metrics.get("wavelet_lightgbm"),
                    metrics.get("spillover_lightgbm"),
                    metrics.get("spillover_lightgbm") - metrics.get("wavelet_lightgbm"),
                ]
            )
    result = pd.DataFrame(
        rows,
        columns=["regime", "horizon", "n_obs", "wavelet_q", "spill_q", "delta_spill_minus_wavelet"],
    )
    result.to_csv(TABLE_DIR / "table_spillover_regime.csv", index=False)
    return result


def write_spillover_regime_table(result: pd.DataFrame) -> None:
    pivot = result.pivot(index="regime", columns="horizon", values="delta_spill_minus_wavelet")
    lines = [
        "\\begin{table}[!t]",
        "\\caption{Broad-market regime-conditioned spillover gains. Entries report $\\Delta$QLIKE = Spillover-LGB minus Wavelet-LGB for DJIA and S\\&P 500 pooled observations. Negative values indicate that cross-index spillover features improve accuracy.}",
        "\\label{tab:spillover_regime}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5.0pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Regime & $h=1$ & $h=5$ & $h=10$ \\\\",
        "\\midrule",
    ]
    for regime, row in pivot.iterrows():
        row_text = []
        for horizon in [1, 5, 10]:
            value = float(row[horizon])
            col = pivot[horizon]
            row_text.append(_best_text(value, col, higher_is_better=False))
        lines.append(f"{regime} & {row_text[0]} & {row_text[1]} & {row_text[2]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABLE_DIR / "table_spillover_regime.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    quantile_summary, spill_summary = build_quantile_conditioned_assets()
    render_quantile_conditioned_figure(quantile_summary, spill_summary)

    cell_level, warning_summary = build_warning_event_assets()
    render_warning_event_frontier(cell_level)
    write_warning_event_table(warning_summary)

    spill_regime = build_spillover_regime_assets()
    write_spillover_regime_table(spill_regime)

    reg_roll, spill_roll, _ = build_timevarying_gain_assets()
    stress_spans = _extract_stress_spans(ROOT / "data" / "processed" / "panel_features.csv.gz")
    render_timevarying_gain_figure(reg_roll, spill_roll, stress_spans)

    print("Generated innovation experiment assets.")


if __name__ == "__main__":
    main()
