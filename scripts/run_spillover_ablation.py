"""
Cross-Index Wavelet Spillover Ablation Analysis
================================================
Computes DM and CW statistical tests to isolate the incremental predictive
contribution of cross-index wavelet spillover features, following the ablation
ladder defined in config/experiment_spillover.yaml:

  Step 1: har                 (pure persistence baseline)
  Step 2: wavelet_lightgbm    (+ within-index multiscale)
  Step 3: spillover_lightgbm  (+ cross-index spillover, the novel contribution)

Scientific claim being tested:
  H0: E[QLIKE(spillover_lightgbm) - QLIKE(wavelet_lightgbm)] = 0
  H1: spillover_lightgbm has strictly smaller expected QLIKE loss
      (i.e. cross-index spillover carries incremental predictive content)

The DM test (Newey-West HAC) and CW nested test are reported for every
(index, horizon) cell. Win rates across cells serve as the publishable
summary statistic.

Usage
-----
  python scripts/run_spillover_ablation.py \
      --merged-tag spillover_merged \
      --output-dir outputs/spillover_merged/tables
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (SRC, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from volatility_lab.stats import clark_west, diebold_mariano


# ─── QLIKE loss on variance predictions ────────────────────────────────────

def qlike_from_preds(df: pd.DataFrame, eps: float = 1e-8) -> pd.Series:
    y = df["y_true_var"].clip(lower=eps)
    yhat = df["y_pred_var"].clip(lower=eps)
    return np.log(yhat) + y / yhat


# ─── Pivot predictions into (date, model) loss frames ─────────────────────

def loss_pivot(preds: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by (index_id, horizon, date) with one
    column per model containing per-step QLIKE losses."""
    records = []
    for (idx_id, horizon, model), grp in preds.groupby(
        ["index_id", "horizon", "model"], sort=False
    ):
        loss = qlike_from_preds(grp.sort_values("date"))
        for date, l in zip(grp.sort_values("date")["date"], loss):
            records.append(
                {"index_id": idx_id, "horizon": horizon, "date": date, model: l}
            )
    # Merge into wide format
    wide = (
        pd.DataFrame(records)
        .groupby(["index_id", "horizon", "date"])
        .first()
        .reset_index()
    )
    return wide


# ─── Pairwise DM test ──────────────────────────────────────────────────────

def pairwise_dm(
    loss_a: pd.Series, loss_b: pd.Series, horizon: int
) -> dict[str, float]:
    """DM: H0 E[la - lb] = 0.  Positive stat => a is worse than b."""
    return diebold_mariano(loss_a.to_numpy(), loss_b.to_numpy(), horizon=horizon)


# ─── Pairwise CW test ─────────────────────────────────────────────────────

def pairwise_cw(
    preds: pd.DataFrame, model_small: str, model_large: str
) -> dict[str, float]:
    """Clark-West: H0 E[adj] <= 0.  model_large is 'larger' (more features)."""
    y = preds["y_true_var"].to_numpy(float)
    p_small = preds[model_small].to_numpy(float)
    p_large = preds[model_large].to_numpy(float)
    return clark_west(y, p_small, p_large)


# ─── Main comparison pairs ─────────────────────────────────────────────────

ABLATION_PAIRS = [
    # (model_a, model_b, description, test_type)
    # DM: tests if wavelet_lightgbm strictly outperforms har (within-index step)
    ("har",              "wavelet_lightgbm",   "Step1: HAR vs Wavelet-LGB",       "DM"),
    # DM: tests if spillover_lightgbm strictly outperforms wavelet_lightgbm
    ("wavelet_lightgbm", "spillover_lightgbm", "Step2: Wavelet-LGB vs Spillover",  "DM"),
    # DM: full gap, spillover vs baseline har
    ("har",              "spillover_lightgbm", "Full: HAR vs Spillover",           "DM"),
    # CW (nested): spillover contains wavelet features, so CW is appropriate
    ("wavelet_lightgbm", "spillover_lightgbm", "Step2 CW (nested)",                "CW"),
]


# ─── Main ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spillover ablation statistical tests.")
    p.add_argument("--merged-tag",  default="spillover_merged",
                   help="Tag under outputs/ containing the merged predictions.")
    p.add_argument("--output-dir",  default=None,
                   help="Write CSV results here (defaults to merged-tag/tables).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merged_root = ROOT / "outputs" / args.merged_tag
    preds_path  = merged_root / "predictions" / "regression_predictions.csv"

    if not preds_path.exists():
        print(f"[ERROR] Predictions not found: {preds_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else merged_root / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(preds_path, parse_dates=["date"])
    print(f"Loaded {len(preds):,} prediction rows.")
    print(f"Models present: {sorted(preds['model'].unique())}")
    print(f"Indices: {sorted(preds['index_id'].unique())}")
    print(f"Horizons: {sorted(preds['horizon'].unique())}\n")

    # ── Build QLIKE loss series per (index, horizon, model, date) ──────────
    preds["qlike_loss"] = preds.groupby(
        ["index_id", "horizon", "model"], group_keys=False
    ).apply(lambda g: qlike_from_preds(g))

    results: list[dict] = []

    for idx_id, h_grp in preds.groupby(["index_id"]):
        for horizon, cell in h_grp.groupby("horizon"):
            cell = cell.sort_values("date")
            model_losses: dict[str, pd.Series] = {}
            model_preds:  dict[str, pd.Series] = {}
            for model, m_grp in cell.groupby("model"):
                m_grp_sorted = m_grp.sort_values("date")
                model_losses[model] = m_grp_sorted["qlike_loss"].reset_index(drop=True)
                model_preds[model]  = m_grp_sorted["y_pred_var"].reset_index(drop=True)
            y_true = (
                cell[cell["model"] == list(model_losses.keys())[0]]
                .sort_values("date")["y_true_var"]
                .reset_index(drop=True)
            )

            for model_a, model_b, desc, test_type in ABLATION_PAIRS:
                if model_a not in model_losses or model_b not in model_losses:
                    continue

                la = model_losses[model_a]
                lb = model_losses[model_b]

                # Align on common non-NaN indices
                valid = la.notna() & lb.notna()

                if test_type == "DM":
                    stat = diebold_mariano(
                        la[valid].to_numpy(),
                        lb[valid].to_numpy(),
                        horizon=int(horizon),
                    )
                    results.append({
                        "index_id":  idx_id if isinstance(idx_id, str) else idx_id[0],
                        "horizon":   int(horizon),
                        "comparison": desc,
                        "model_a":   model_a,
                        "model_b":   model_b,
                        "n_obs":     int(valid.sum()),
                        "mean_loss_a": float(la[valid].mean()),
                        "mean_loss_b": float(lb[valid].mean()),
                        "dm_stat":   stat["dm_stat"],
                        "p_value":   stat["p_value"],
                        "sig_05":    stat["p_value"] < 0.05 if np.isfinite(stat["p_value"]) else False,
                        "sig_10":    stat["p_value"] < 0.10 if np.isfinite(stat["p_value"]) else False,
                        "b_wins":    float(lb[valid].mean()) < float(la[valid].mean()),
                        "test_type": "DM",
                    })
                else:  # CW
                    pa = model_preds[model_a][valid].to_numpy()
                    pb = model_preds[model_b][valid].to_numpy()
                    yt = y_true[valid].to_numpy()
                    stat = clark_west(yt, pa, pb)
                    results.append({
                        "index_id":  idx_id if isinstance(idx_id, str) else idx_id[0],
                        "horizon":   int(horizon),
                        "comparison": desc,
                        "model_a":   model_a,
                        "model_b":   model_b,
                        "n_obs":     int(valid.sum()),
                        "mean_loss_a": float(la[valid].mean()),
                        "mean_loss_b": float(lb[valid].mean()),
                        "cw_stat":   stat["cw_stat"],
                        "p_value":   stat["p_value"],
                        "sig_05":    stat["p_value"] < 0.05 if np.isfinite(stat["p_value"]) else False,
                        "sig_10":    stat["p_value"] < 0.10 if np.isfinite(stat["p_value"]) else False,
                        "b_wins":    float(lb[valid].mean()) < float(la[valid].mean()),
                        "test_type": "CW",
                    })

    df = pd.DataFrame(results)
    out_path = output_dir / "spillover_ablation_tests.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults written to: {out_path}")

    # ── Publication-ready summary ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CROSS-INDEX SPILLOVER CONTRIBUTION — STATISTICAL EVIDENCE")
    print("=" * 72)

    for test_label in ["Step2: Wavelet-LGB vs Spillover", "Step2 CW (nested)"]:
        sub = df[df["comparison"] == test_label]
        if sub.empty:
            continue
        n      = len(sub)
        b_wins = sub["b_wins"].sum()
        sig05  = sub["sig_05"].sum()
        sig10  = sub["sig_10"].sum()
        mean_gain = (sub["mean_loss_a"] - sub["mean_loss_b"]).mean()

        print(f"\n  {test_label}")
        print(f"    Cells where spillover model wins (lower QLIKE): {int(b_wins)}/{n}  ({100*b_wins/n:.0f}%)")
        print(f"    Significant at p<0.05: {int(sig05)}/{n}  ({100*sig05/n:.0f}%)")
        print(f"    Significant at p<0.10: {int(sig10)}/{n}  ({100*sig10/n:.0f}%)")
        print(f"    Mean QLIKE gain over wavelet_lightgbm: {mean_gain:.5f}")

    print("\n  Full ablation by (index, horizon):")
    pivot = df[df["comparison"].isin([
        "Step1: HAR vs Wavelet-LGB",
        "Step2: Wavelet-LGB vs Spillover",
        "Full: HAR vs Spillover",
    ])][["index_id", "horizon", "comparison", "mean_loss_a", "mean_loss_b", "p_value", "sig_05"]].copy()
    pivot["gain_b"] = pivot["mean_loss_a"] - pivot["mean_loss_b"]
    print(pivot.to_string(index=False))

    # ── Horizon-stratified win rates (key table for paper) ─────────────────
    print("\n  Horizon-stratified win rates (spillover vs wavelet_lightgbm, DM):")
    step2 = df[(df["comparison"] == "Step2: Wavelet-LGB vs Spillover") & (df["test_type"] == "DM")]
    for h, sg in step2.groupby("horizon"):
        wrate = sg["b_wins"].mean()
        srate = sg["sig_05"].mean()
        print(f"    h={h}: win rate {wrate:.0%}, sig@5% rate {srate:.0%}")


if __name__ == "__main__":
    main()
