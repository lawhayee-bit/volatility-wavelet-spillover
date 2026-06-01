from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from volatility_lab.plots import (
    plot_forecast_timeseries,
    plot_main_regression_bars,
    plot_metric_heatmap,
    plot_warning_pr_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render plots from merged batch outputs.")
    parser.add_argument("--input-tag", default="formal_merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT / "outputs" / args.input_tag
    reg_summary = pd.read_csv(root / "tables" / "regression_summary.csv")
    reg_preds = pd.read_csv(root / "predictions" / "regression_predictions.csv", parse_dates=["date"])
    cls_preds = pd.read_csv(root / "predictions" / "classification_predictions.csv", parse_dates=["date"])

    plot_main_regression_bars(reg_summary, output_dir=root / "figures")
    plot_forecast_timeseries(reg_preds, output_dir=root / "figures")
    plot_warning_pr_curve(cls_preds, output_dir=root / "figures")
    plot_metric_heatmap(reg_summary, output_dir=root / "figures")
    print(f"Rendered merged plots into {root / 'figures'}")


if __name__ == "__main__":
    main()
