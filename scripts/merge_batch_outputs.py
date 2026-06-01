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

from volatility_lab.experiment import (
    compare_regression_models,
    summarise_classification_predictions,
    summarise_regression_predictions,
)
from volatility_lab.utils import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge batch experiment outputs.")
    parser.add_argument("--batch-root", default="outputs")
    parser.add_argument("--pattern", default="formal_*")
    parser.add_argument("--output-tag", default="formal_merged")
    return parser.parse_args()


def concat_from_batches(batch_root: Path, pattern: str, relative_path: str, exclude_dir: Path | None = None) -> pd.DataFrame:
    frames = []
    for batch_dir in sorted(batch_root.glob(pattern)):
        if exclude_dir is not None and batch_dir.resolve() == exclude_dir.resolve():
            continue
        file_path = batch_dir / relative_path
        if file_path.exists():
            frames.append(pd.read_csv(file_path, parse_dates=["date"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def main() -> None:
    args = parse_args()
    batch_root = ROOT / args.batch_root
    output_root = ensure_directory(batch_root / args.output_tag)
    preds_root = ensure_directory(output_root / "predictions")
    tables_root = ensure_directory(output_root / "tables")

    reg_preds = concat_from_batches(
        batch_root,
        args.pattern,
        "predictions/regression_predictions.csv",
        exclude_dir=output_root,
    )
    cls_preds = concat_from_batches(
        batch_root,
        args.pattern,
        "predictions/classification_predictions.csv",
        exclude_dir=output_root,
    )

    reg_preds.to_csv(preds_root / "regression_predictions.csv", index=False)
    cls_preds.to_csv(preds_root / "classification_predictions.csv", index=False)

    reg_summary = summarise_regression_predictions(reg_preds) if not reg_preds.empty else pd.DataFrame()
    cls_summary = summarise_classification_predictions(cls_preds) if not cls_preds.empty else pd.DataFrame()
    dm_table, cw_table = compare_regression_models(reg_preds, benchmark_model="har") if not reg_preds.empty else (pd.DataFrame(), pd.DataFrame())

    reg_summary.to_csv(tables_root / "regression_summary.csv", index=False)
    cls_summary.to_csv(tables_root / "classification_summary.csv", index=False)
    dm_table.to_csv(tables_root / "diebold_mariano.csv", index=False)
    cw_table.to_csv(tables_root / "clark_west.csv", index=False)

    print(f"Merged outputs written to {output_root}")


if __name__ == "__main__":
    main()
