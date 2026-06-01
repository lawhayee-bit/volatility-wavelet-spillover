from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting import apply_plot_style, configure_axes, load_plot_style, save_figure


INDEX_LABELS = {"spx": "S&P 500", "ndq": "Nasdaq-100", "dji": "DJIA"}
HORIZONS = [1, 5, 10]
SCALE_ORDER = [("d1", "Short"), ("d2", "Medium"), ("d3", "Long"), ("a3", "Approx.")]
SOURCE_ORDER = [("rs_var", "Target variance"), ("abs_ret", "Absolute return"), ("vix_like", "Implied volatility")]
BLOCK_ORDER = ["Persistence", "Raw market-risk", "Macro-financial", "Wavelet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interpretability figures for the manuscript.")
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "processed" / "panel_features_plusdata.csv.gz"),
        help="Processed feature panel used for interpretability figures.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "paper" / "manuscript" / "figures"),
        help="Directory for figure outputs.",
    )
    parser.add_argument(
        "--table-dir",
        default=str(ROOT / "paper" / "manuscript" / "tables"),
        help="Directory for intermediate CSV outputs.",
    )
    return parser.parse_args()


def _semantic_colors(style: dict) -> dict[str, str]:
    palette = style["palette"]["series"]
    return {
        "har": palette[5] if len(palette) > 5 else palette[0],
        "lightgbm": palette[1],
        "wavelet": palette[3],
        "macro": palette[2],
        "raw": palette[0],
    }


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


def _load_panel(panel_path: Path) -> pd.DataFrame:
    return pd.read_csv(panel_path, parse_dates=["date"])


def _wavelet_group_value(df: pd.DataFrame, target_col: str, source: str, scale: str) -> float:
    cols = [c for c in df.columns if c.startswith(f"wav_{source}_{scale}")]
    y = df[target_col]
    corrs: list[float] = []
    for col in cols:
        x = df[col]
        mask = x.notna() & y.notna()
        if mask.sum() < 40:
            continue
        rho = spearmanr(x[mask], y[mask]).statistic
        if rho is not None and np.isfinite(rho):
            corrs.append(abs(float(rho)))
    return float(np.median(corrs)) if corrs else np.nan


def render_multiscale_signal_heatmap(panel: pd.DataFrame, fig_path: Path, csv_path: Path) -> pd.DataFrame:
    style = load_plot_style("heatmap")
    apply_plot_style(style)

    oos = panel.loc[panel["date"] >= pd.Timestamp("2016-01-04")].copy()
    rows = []
    for index_id in ["spx", "ndq", "dji"]:
        sub = oos.loc[oos["index_id"] == index_id].copy()
        for horizon in HORIZONS:
            target_col = f"target_model_h{horizon}"
            for source, source_label in SOURCE_ORDER:
                for scale, scale_label in SCALE_ORDER:
                    rows.append(
                        {
                            "index_id": index_id,
                            "index_label": INDEX_LABELS[index_id],
                            "horizon": horizon,
                            "source": source_label,
                            "scale": scale_label,
                            "value": _wavelet_group_value(sub, target_col, source, scale),
                        }
                    )

    data = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 3, figsize=(7.8, 6.4), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        configure_axes(ax, style)

    cmap = LinearSegmentedColormap.from_list("multiscale_signal", style["palette"]["heatmap_sequential"])
    vmax = float(np.nanmax(data["value"]))
    vmin = float(np.nanmin(data["value"]))
    image = None
    for row_idx, index_id in enumerate(["spx", "ndq", "dji"]):
        for col_idx, horizon in enumerate(HORIZONS):
            ax = axes[row_idx, col_idx]
            block = data.loc[(data["index_id"] == index_id) & (data["horizon"] == horizon)].copy()
            mat = (
                block.pivot(index="source", columns="scale", values="value")
                .reindex(index=[label for _, label in SOURCE_ORDER], columns=[label for _, label in SCALE_ORDER])
                .to_numpy(dtype=float)
            )
            image = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(SCALE_ORDER)))
            ax.set_xticklabels([label for _, label in SCALE_ORDER], rotation=0)
            ax.set_yticks(np.arange(len(SOURCE_ORDER)))
            ax.set_yticklabels([label for _, label in SOURCE_ORDER] if col_idx == 0 else [""] * len(SOURCE_ORDER))
            ax.set_xlabel("" if row_idx < 2 else "Wavelet scale")
            ax.set_ylabel("Signal source" if col_idx == 0 else "")
            ax.text(
                0.02,
                1.03,
                f"{INDEX_LABELS[index_id]}, $h={horizon}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
            )
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.3)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, location="right")
    cbar.set_label(r"Median $|\rho_s|$ with future volatility")
    _save_dual(fig, fig_path, style)
    plt.close(fig)
    return data


def _feature_blocks(columns: list[str]) -> dict[str, list[str]]:
    target_cols = {c for c in columns if c.startswith("target_")}
    excluded = {"date", "open", "high", "low", "close", "index_volume", "index_id", "index_label"} | target_cols
    wavelet = sorted(c for c in columns if c.startswith("wav_"))
    persistence = sorted(set([c for c in columns if c.startswith("har_")] + [c for c in ["target_base_var"] if c in columns]))

    raw: list[str] = []
    for c in columns:
        if c in excluded or c in wavelet or c in persistence:
            continue
        if c.startswith(("ret_", "gap_", "abs_ret", "abs_gap", "abs_vix", "jump_", "hl_", "range_", "rsi_", "atr_", "macd_", "etf_")) or c.endswith("_var"):
            raw.append(c)

    macro: list[str] = []
    for c in columns:
        if c in excluded or c in wavelet or c in persistence or c in raw:
            continue
        macro.append(c)

    return {
        "Persistence": sorted(set(persistence)),
        "Raw market-risk": sorted(set(raw)),
        "Macro-financial": sorted(set(macro)),
        "Wavelet": sorted(set(wavelet)),
    }


def _block_embedding(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X_train = train_df[cols].astype(float).copy()
    X_test = test_df[cols].astype(float).copy()

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    mu = X_train.mean()
    sd = X_train.std().replace(0, 1.0)
    Z_train = ((X_train - mu) / sd).to_numpy(dtype=float)
    Z_test = ((X_test - mu) / sd).to_numpy(dtype=float)

    if Z_train.shape[1] == 1:
        return Z_train[:, 0], Z_test[:, 0]

    _, _, vt = np.linalg.svd(Z_train, full_matrices=False)
    vec = vt[0]
    return Z_train @ vec, Z_test @ vec


def _ridge_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    Z_train = (X_train - mu) / sd
    Z_test = (X_test - mu) / sd
    y_bar = float(np.mean(y_train))
    y_center = y_train - y_bar
    beta = np.linalg.solve(Z_train.T @ Z_train + alpha * np.eye(Z_train.shape[1]), Z_train.T @ y_center)
    return y_bar + Z_test @ beta


def _r2_oos(y_true: np.ndarray, y_pred: np.ndarray, benchmark_mean: float) -> float:
    denom = np.mean((y_true - benchmark_mean) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - float(np.mean((y_true - y_pred) ** 2) / denom)


def render_block_contribution_heatmap(panel: pd.DataFrame, fig_path: Path, csv_path: Path) -> pd.DataFrame:
    style = load_plot_style("heatmap")
    apply_plot_style(style)

    blocks = _feature_blocks(list(panel.columns))
    train = panel.loc[panel["date"] <= pd.Timestamp("2015-12-31")].copy()
    test = panel.loc[panel["date"] >= pd.Timestamp("2016-01-04")].copy()

    rows = []
    for index_id in ["spx", "ndq", "dji"]:
        train_idx = train.loc[train["index_id"] == index_id].copy()
        test_idx = test.loc[test["index_id"] == index_id].copy()
        for horizon in HORIZONS:
            y_col = f"target_model_h{horizon}"
            tr = train_idx.loc[train_idx[y_col].notna()].copy()
            te = test_idx.loc[test_idx[y_col].notna()].copy()

            names: list[str] = []
            emb_train: list[np.ndarray] = []
            emb_test: list[np.ndarray] = []
            for name in BLOCK_ORDER:
                cols = [c for c in blocks[name] if c in tr.columns]
                score_train, score_test = _block_embedding(tr, te, cols)
                names.append(name)
                emb_train.append(score_train)
                emb_test.append(score_test)

            X_train = np.column_stack(emb_train)
            X_test = np.column_stack(emb_test)
            y_train = tr[y_col].to_numpy(dtype=float)
            y_test = te[y_col].to_numpy(dtype=float)
            y_bar = float(np.mean(y_train))

            full_pred = _ridge_predict(X_train, y_train, X_test, alpha=1.0)
            full_r2 = _r2_oos(y_test, full_pred, y_bar)

            for j, name in enumerate(names):
                keep = [k for k in range(len(names)) if k != j]
                pred = _ridge_predict(X_train[:, keep], y_train, X_test[:, keep], alpha=1.0)
                reduced_r2 = _r2_oos(y_test, pred, y_bar)
                rows.append(
                    {
                        "index_id": index_id,
                        "index_label": INDEX_LABELS[index_id],
                        "horizon": horizon,
                        "block": name,
                        "full_r2_oos": full_r2,
                        "delta_r2_oos": float(full_r2 - reduced_r2),
                    }
                )

    data = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.45), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr:
        configure_axes(ax, style)

    cmap = LinearSegmentedColormap.from_list("block_contrib", style["palette"]["heatmap_diverging"])
    vmax = float(np.nanmax(np.abs(data["delta_r2_oos"])))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = None
    for ax, index_id in zip(axes_arr, ["spx", "ndq", "dji"]):
        block = data.loc[data["index_id"] == index_id].copy()
        mat = (
            block.pivot(index="block", columns="horizon", values="delta_r2_oos")
            .reindex(index=BLOCK_ORDER, columns=HORIZONS)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(HORIZONS)))
        ax.set_xticklabels([f"$h={h}$" for h in HORIZONS])
        ax.set_yticks(np.arange(len(BLOCK_ORDER)))
        ax.set_yticklabels(BLOCK_ORDER if index_id == "spx" else [""] * len(BLOCK_ORDER))
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel(r"Feature block" if index_id == "spx" else "")
        ax.text(0.02, 1.03, INDEX_LABELS[index_id], transform=ax.transAxes, ha="left", va="bottom")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8.2)

    cbar = fig.colorbar(image, ax=axes_arr.tolist(), shrink=0.95, location="right")
    cbar.set_label(r"Leave-one-block-out $\Delta R^2_{OOS}$")
    _save_dual(fig, fig_path, style)
    plt.close(fig)
    return data


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_path)
    fig_dir = Path(args.output_dir)
    table_dir = Path(args.table_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(panel_path)
    render_multiscale_signal_heatmap(
        panel,
        fig_dir / "fig11_multiscale_signal_heatmap.pdf",
        table_dir / "fig11_multiscale_signal_heatmap.csv",
    )
    render_block_contribution_heatmap(
        panel,
        fig_dir / "fig12_block_contribution_heatmap.pdf",
        table_dir / "fig12_block_contribution_heatmap.csv",
    )
    print(f"Generated interpretability figures in {fig_dir}")


if __name__ == "__main__":
    main()
