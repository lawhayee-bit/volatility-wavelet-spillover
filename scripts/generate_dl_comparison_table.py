"""Merge DL baseline predictions with existing results and generate
a DL comparison summary table in LaTeX format.

Usage
-----
python scripts/generate_dl_comparison_table.py \
    [--existing outputs/full10y_refinedfinal_merged/predictions/regression_predictions.csv] \
    [--dl-tags dl_full dl_lstm_full] \
    [--output-tex paper/manuscript/tables/table_dl_comparison.tex] \
    [--output-csv outputs/tables/dl_comparison.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# QLIKE helper
# ---------------------------------------------------------------------------

def qlike_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quasi-likelihood loss per Patton (2011)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred > 0) & (y_true > 0)
    if mask.sum() < 10:
        return np.nan
    ratio = y_true[mask] / y_pred[mask]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

BEST_FMT = r"\best{{{}}}"
SECOND_FMT = r"\second{{{}}}"


def _fmt(val: float, rank: int | None) -> str:
    s = f"{val:.4f}"
    if rank == 1:
        return BEST_FMT.format(s)
    if rank == 2:
        return SECOND_FMT.format(s)
    return s


MODEL_LABELS = {
    "har": "HAR",
    "lightgbm": "LightGBM",
    "wavelet_lightgbm": r"\textit{wavelet\_lightgbm}",
    "mlp": "MLP",
    "gru": "GRU",
    "bilstm": "BiLSTM",
    "lstm": "LSTM",
    "hv_22": "HV-22",
    "last_value": "Last-value",
}

INDEX_LABELS = {
    "spx": r"S\&P 500",
    "ndq": "Nasdaq-100",
    "dji": "DJIA",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DL comparison table.")
    parser.add_argument(
        "--existing",
        default="outputs/full10y_refinedfinal_merged/predictions/regression_predictions.csv",
    )
    parser.add_argument(
        "--dl-tags",
        nargs="+",
        default=["dl_full", "dl_lstm_full"],
        help="Output tags of DL experiments (subdirs of outputs/).",
    )
    parser.add_argument(
        "--output-tex",
        default="paper/manuscript/tables/table_dl_comparison.tex",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/tables/dl_comparison.csv",
    )
    args = parser.parse_args()

    # 1. Load existing predictions
    existing_path = ROOT / args.existing
    parts = [pd.read_csv(existing_path)]

    # 2. Load DL experiment predictions
    for tag in args.dl_tags:
        p = ROOT / "outputs" / tag / "predictions" / "regression_predictions.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
            print(f"Loaded: {p}")
        else:
            print(f"NOT FOUND (skipping): {p}")

    df = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["date", "index_id", "horizon", "model"]
    )
    print(f"Combined rows: {len(df)}")
    print(f"Models: {df['model'].unique().tolist()}")

    # 3. Compute QLIKE per (model, index_id, horizon)
    records = []
    for (model, idx, h), grp in df.groupby(["model", "index_id", "horizon"]):
        q = qlike_safe(grp["y_true_var"].values, grp["y_pred_var"].values)
        records.append({"model": model, "index_id": idx, "horizon": int(h), "qlike": q})

    summary = pd.DataFrame(records)

    # 4. Save CSV
    csv_path = ROOT / args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # 5. Pivot and build LaTeX table
    pivot = summary.pivot_table(index=["index_id", "model"], columns="horizon", values="qlike")
    pivot.columns = [int(c) for c in pivot.columns]

    # Define display models order
    model_order = ["har", "hv_22", "lightgbm", "wavelet_lightgbm", "mlp", "gru", "bilstm", "lstm"]
    model_order = [m for m in model_order if m in df["model"].unique()]
    index_order = ["spx", "ndq", "dji"]

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{QLIKE comparison of benchmark models and the HAR-LSTM deep-learning")
    lines.append(r"baseline under the same 1260-day rolling walk-forward protocol.")
    lines.append(r"The LSTM is trained on the three HAR features (1-, 5-, and 22-day realized")
    lines.append(r"variance) with 16 hidden units and chronological early stopping.")
    lines.append(r"Best value per index--horizon cell is \best{bold-shaded}; second-best is \second{underlined}.}")
    lines.append(r"\label{tab:dl_comparison}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Index & Model & $h=1$ & $h=5$ & $h=10$ \\")
    lines.append(r"\midrule")

    for idx_id in index_order:
        idx_label = INDEX_LABELS.get(idx_id, idx_id)
        first_row = True
        # Collect values for ranking
        values_by_h = {h: {} for h in [1, 5, 10]}
        for model in model_order:
            try:
                for h in [1, 5, 10]:
                    v = float(pivot.loc[(idx_id, model), h])
                    values_by_h[h][model] = v
            except (KeyError, TypeError):
                pass

        # Compute ranks (lower QLIKE = better)
        ranks_by_h: dict[int, dict[str, int | None]] = {h: {} for h in [1, 5, 10]}
        for h in [1, 5, 10]:
            sorted_models = sorted(values_by_h[h], key=lambda m: values_by_h[h][m])
            for r, m in enumerate(sorted_models[:2]):
                ranks_by_h[h][m] = r + 1

        # Add separator before DL block
        dl_models = {"mlp", "gru", "bilstm", "lstm"}
        first_dl_in_order = next((m for m in model_order if m in dl_models), None)

        for model in model_order:
            if model not in values_by_h[1] and model not in values_by_h[5]:
                continue
            label = MODEL_LABELS.get(model, model)
            prefix = idx_label if first_row else " "
            first_row = False

            # Add midrule before first DL model
            if model == first_dl_in_order:
                lines.append(r"\cmidrule(lr){2-5}")

            cells = []
            for h in [1, 5, 10]:
                v = values_by_h[h].get(model, None)
                if v is None or not np.isfinite(v):
                    cells.append("---")
                else:
                    rank = ranks_by_h[h].get(model, None)
                    cells.append(_fmt(v, rank))

            lines.append(f"{prefix} & {label} & {' & '.join(cells)} \\\\")

        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)

    tex_path = ROOT / args.output_tex
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex)
    print(f"Saved LaTeX table: {tex_path}")

    # Print summary to console
    print("\nQlike summary (averaged across indices):")
    avg = summary.groupby(["model", "horizon"])["qlike"].mean().unstack()
    for model in model_order:
        if model in avg.index:
            row = avg.loc[model]
            print(f"  {model:25s}: h1={row.get(1, np.nan):.4f}, h5={row.get(5, np.nan):.4f}, h10={row.get(10, np.nan):.4f}")


if __name__ == "__main__":
    main()
