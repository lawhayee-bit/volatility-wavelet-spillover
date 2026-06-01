"""Generate SHAP feature-importance analysis for wavelet_lightgbm.

Usage
-----
python scripts/generate_shap_analysis.py [--index spx] [--horizon 1]
                                          [--output-dir outputs/figures]
                                          [--config config/experiment.yaml]
                                          [--top-n 20]

The script trains one wavelet_lightgbm model on the full available
training window ending just before the test period, then computes
SHAP values for the first 500 test observations.  It produces:

  outputs/figures/shap_beeswarm_{index}_h{horizon}.pdf
  outputs/figures/shap_bar_{index}_h{horizon}.pdf
  outputs/tables/shap_importance_{index}_h{horizon}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.volatility_lab.config import load_yaml_config
from src.volatility_lab.data import read_dataframe
from src.volatility_lab.experiment import _feature_subsets, _prepare_train_test
from src.volatility_lab.models import make_regressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nice_name(col: str) -> str:
    """Return a human-readable short label for a feature column."""
    mapping = {
        "har_target_1": "HAR-RV(1d)",
        "har_target_5": "HAR-RV(5d)",
        "har_target_22": "HAR-RV(22d)",
        "vix_like": "VIX proxy",
        "dff": "Fed funds rate",
        "dgs10": "10Y yield",
        "t10y3m": "Term spread",
        "nfci": "NFCI",
        "rs_var": "RS variance",
        "abs_ret": "Abs return",
        "rsi_14": "RSI(14)",
        "atr_14": "ATR(14)",
        "macd_hist": "MACD hist",
        "recession_dummy": "Recession",
    }
    if col in mapping:
        return mapping[col]
    # wav_<series>_<level>_energy_<window>
    if col.startswith("wav_"):
        parts = col.split("_")
        # e.g. wav_rs_var_d1_energy_5 → DWT rs_var d1 eng5
        try:
            series = "_".join(parts[1:-3])
            level = parts[-3]
            window = parts[-1]
            return f"DWT {series} {level} e{window}"
        except (IndexError, ValueError):
            return col
    return col.replace("_", " ")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis for wavelet_lightgbm.")
    parser.add_argument("--index", default="spx", choices=["spx", "ndq", "dji"])
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 5, 10])
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--top-n", type=int, default=12, help="Top N features to display.")
    parser.add_argument("--max-shap-samples", type=int, default=500,
                        help="Number of test rows to compute SHAP values over.")
    args = parser.parse_args()

    config = load_yaml_config(ROOT / args.config)
    panel = read_dataframe(ROOT / config["data"]["processed_panel_path"])

    index_df = (
        panel.loc[panel["index_id"] == args.index]
        .sort_values("date")
        .reset_index(drop=True)
    )
    feature_sets = _feature_subsets(panel)
    feature_sets = {
        k: [c for c in v if c in index_df.columns]
        for k, v in feature_sets.items()
    }

    horizon = args.horizon
    target_model_col = f"target_model_h{horizon}"
    target_var_col = f"target_base_h{horizon}"
    transform = config["features"]["target"]["transform"]
    epsilon = float(config["features"]["target"]["epsilon"])

    test_start = pd.Timestamp(config["experiment"]["test_start"])
    train_window = int(config["experiment"]["train_window_days"])
    random_state = int(config["project"]["random_state"])
    n_jobs = int(config["runtime"]["n_jobs"])

    # Find the last training anchor just before test_start
    test_mask = index_df["date"] >= test_start
    test_positions = np.flatnonzero(test_mask)
    if len(test_positions) == 0:
        raise RuntimeError("No test positions found. Check test_start in config.")

    # Train on the window ending at the first test position
    anchor_pos = int(test_positions[0])
    if anchor_pos < train_window:
        raise RuntimeError(
            f"Insufficient history: anchor_pos={anchor_pos} < train_window={train_window}."
        )

    train = index_df.iloc[anchor_pos - train_window : anchor_pos].copy().reset_index(drop=True)

    # Fit wavelet_lightgbm
    spec = make_regressor("wavelet_lightgbm", random_state=random_state, n_jobs=n_jobs)
    feature_cols = feature_sets[spec.feature_set]

    X_train, y_train, _ = _prepare_train_test(
        train, train.iloc[[0]].copy(), feature_cols, target_model_col
    )
    from sklearn.base import clone as _clone
    from sklearn.impute import SimpleImputer
    import sklearn.pipeline as _pl

    estimator = _clone(spec.estimator)
    estimator.fit(X_train, y_train)

    # Gather test data for SHAP
    n_shap = min(args.max_shap_samples, len(test_positions))
    shap_positions = test_positions[:n_shap]
    X_shap_rows = []
    for pos in shap_positions:
        row = index_df.iloc[[pos]]
        X_shap_rows.append(row[feature_cols].copy())
    X_shap = pd.concat(X_shap_rows, ignore_index=True)

    # Compute SHAP values on the LightGBM step (after preprocessing steps)
    # Transform through imputer/scaler pipeline steps, then call TreeExplainer
    pipe_steps = list(estimator.named_steps.items())
    X_transformed = X_shap.copy()
    final_step_name = pipe_steps[-1][0]
    for step_name, step_obj in pipe_steps[:-1]:
        X_transformed = pd.DataFrame(
            step_obj.transform(X_transformed),
            columns=feature_cols,
        )

    lgbm_model = estimator.named_steps[final_step_name]
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_transformed)  # shape: (n_samples, n_features)

    # Build importance DataFrame
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["label"] = importance_df["feature"].map(_nice_name)

    # Save importance table
    tables_dir = ROOT / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"shap_importance_{args.index}_h{horizon}.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved SHAP importance table → {csv_path}")

    # ----------- Plots -----------
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n = args.top_n
    top_idx = importance_df.head(top_n).index.tolist()
    top_features = importance_df.head(top_n)["feature"].tolist()
    top_labels = importance_df.head(top_n)["label"].tolist()

    # ----------- Paper palette (from config/plot_style.toml) -----------
    # Wavelet features: series[0] = #1F3A5F (navy)
    # Non-wavelet features: neutral_light = #D9D9D9
    WAV_COLOR = "#1F3A5F"
    NON_WAV_COLOR = "#D9D9D9"

    import matplotlib.font_manager as _fm
    # Use Times New Roman (or fallback) matching paper style
    _preferred = ["TeX Gyre Termes", "DejaVu Serif", "serif"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": _preferred,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7.5,
        "axes.edgecolor": "#1F1F1F",
        "axes.linewidth": 1.0,
        "text.color": "#1F1F1F",
    })

    # Figure sizing: bar chart uses ~0.30 in per row + fixed margins
    bar_h = top_n * 0.30 + 1.3   # e.g. top_n=12 → ~4.9 in
    bee_h = top_n * 0.30 + 1.3

    # 1. Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(6.4, bar_h))
    vals = importance_df.head(top_n)["mean_abs_shap"].values
    colors = [WAV_COLOR if "wav_" in f else NON_WAV_COLOR for f in top_features]
    ax.barh(range(top_n), vals[::-1], color=colors[::-1], height=0.65,
            edgecolor="#1F1F1F", linewidth=0.6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=7.5)
    ax.set_xlabel("Mean |SHAP value|", fontsize=8)
    ax.tick_params(axis="x", direction="out", length=4, width=0.9, color="#1F1F1F")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=WAV_COLOR, edgecolor="#1F1F1F", linewidth=0.6, label="Wavelet"),
        Patch(facecolor=NON_WAV_COLOR, edgecolor="#1F1F1F", linewidth=0.6, label="Non-wavelet"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="lower right", frameon=False)
    fig.tight_layout(pad=0.4)
    bar_path = output_dir / f"shap_bar_{args.index}_h{horizon}.pdf"
    fig.savefig(bar_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved SHAP bar chart → {bar_path}")

    # 2. Beeswarm / summary plot
    shap_top = shap_values[:, [feature_cols.index(f) for f in top_features]]
    X_top = X_transformed[top_features].copy()

    # Paper diverging palette: navy → cream → auburn (matches heatmap_diverging in plot_style.toml)
    import matplotlib.colors as _mc
    paper_shap_cmap = _mc.LinearSegmentedColormap.from_list(
        "paper_shap", ["#1F3A5F", "#F6F2EA", "#A54E2A"]
    )

    fig2 = plt.figure(figsize=(6.9, bee_h))
    shap.summary_plot(
        shap_top,
        X_top,
        feature_names=top_labels,
        plot_type="dot",
        max_display=top_n,
        show=False,
        plot_size=None,
        color=paper_shap_cmap,
        color_bar_label="Feature value",
    )
    ax_cur = plt.gca()
    ax_cur.spines["top"].set_visible(False)
    ax_cur.spines["right"].set_visible(False)
    ax_cur.tick_params(labelsize=7.5)
    # Recolor dots: SHAP uses its own colormap (red-blue), which is acceptable
    # just ensure axis labels use paper font
    bee_path = output_dir / f"shap_beeswarm_{args.index}_h{horizon}.pdf"
    plt.savefig(bee_path, bbox_inches="tight", dpi=600)
    plt.close("all")
    print(f"Saved SHAP beeswarm plot → {bee_path}")

    # Print top 10 to console
    print(f"\nTop-10 features by mean |SHAP| ({args.index.upper()}, h={horizon}d):")
    print(importance_df.head(10)[["label", "mean_abs_shap"]].to_string(index=False))


if __name__ == "__main__":
    main()
