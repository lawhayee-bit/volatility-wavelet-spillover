from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from volatility_lab.config import load_yaml_config
from volatility_lab.data import build_raw_panel, download_all_sources, read_dataframe, save_dataframe
from volatility_lab.experiment import run_experiment, save_experiment_outputs
from volatility_lab.features import build_feature_panel
from volatility_lab.plots import (
    plot_forecast_timeseries,
    plot_main_regression_bars,
    plot_metric_heatmap,
    plot_warning_pr_curve,
)
from volatility_lab.utils import ensure_directory

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values:",
)
warnings.filterwarnings(
    "ignore",
    category=pd.errors.PerformanceWarning,
)


def apply_smoke_overrides(config: dict) -> dict:
    smoke = config["runtime"]["smoke_defaults"]
    config = json.loads(json.dumps(config))
    config["data"]["start_date"] = smoke["start_date"]
    config["experiment"]["test_end"] = smoke["test_end"]
    allowed = set(smoke["indices"])
    config["data"]["indices"] = [item for item in config["data"]["indices"] if item["id"] in allowed]
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Applied Sciences volatility experiment pipeline.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run a smaller smoke configuration.")
    parser.add_argument("--reuse-panel", action="store_true", help="Reuse an existing processed feature panel.")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading raw data snapshots.")
    parser.add_argument("--output-tag", default=None, help="Write outputs to outputs/<tag>/ instead of outputs/.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override runtime.n_jobs for model fitting.")
    parser.add_argument("--indices", nargs="+", default=None, help="Optional subset of index IDs.")
    parser.add_argument("--horizons", nargs="+", type=int, default=None, help="Optional subset of forecast horizons.")
    parser.add_argument("--reg-models", nargs="+", default=None, help="Optional subset of regression models.")
    parser.add_argument("--cls-models", nargs="+", default=None, help="Optional subset of classification models.")
    parser.add_argument("--max-test-steps", type=int, default=None, help="Optional limit on the number of test points.")
    parser.add_argument(
        "--stage",
        choices=["download", "build", "run", "plot", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(ROOT / args.config)
    if args.smoke:
        config = apply_smoke_overrides(config)
    if args.n_jobs is not None:
        config["runtime"]["n_jobs"] = args.n_jobs

    processed_path = ROOT / config["data"]["processed_panel_path"]
    ensure_directory(processed_path.parent)
    output_root = ROOT / "outputs"
    if args.output_tag:
        output_root = output_root / args.output_tag

    panel = None
    outputs = None

    if args.stage in {"download", "all", "build", "run", "plot"} and not args.skip_download:
        print("Downloading public data snapshots...")
        download_all_sources(config, raw_root=ROOT / "data" / "raw")

    if args.stage in {"build", "all", "run", "plot"}:
        if args.reuse_panel and processed_path.exists():
            print(f"Loading existing feature panel from {processed_path} ...")
            panel = read_dataframe(processed_path)
        else:
            print("Building merged raw panel and engineered features...")
            raw_panel = build_raw_panel(config, raw_root=ROOT / "data" / "raw")
            panel = build_feature_panel(raw_panel, config)
            save_dataframe(panel, processed_path)
            print(f"Feature panel saved to {processed_path}")

    if args.stage in {"run", "all", "plot"}:
        if panel is None:
            raise RuntimeError("Feature panel is not available. Run build stage first.")
        reg_models = args.reg_models
        cls_models = args.cls_models
        indices = args.indices
        horizons = args.horizons
        max_test_steps = args.max_test_steps
        if args.smoke:
            smoke = config["runtime"]["smoke_defaults"]
            reg_models = reg_models or smoke["regression_models"]
            cls_models = cls_models or smoke["classification_models"]
            indices = indices or smoke["indices"]
            horizons = horizons or smoke["horizons"]
            max_test_steps = max_test_steps or smoke["max_test_steps"]

        outputs = run_experiment(
            panel=panel,
            config=config,
            indices=indices,
            horizons=horizons,
            regression_models=reg_models,
            classification_models=cls_models,
            max_test_steps=max_test_steps,
        )
        save_experiment_outputs(outputs, output_root=output_root)
        print(f"Experiment predictions and summary tables saved to {output_root}.")

    if args.stage in {"plot", "all"}:
        if outputs is None:
            raise RuntimeError("Experiment outputs are not available. Run the experiment stage first.")
        print("Rendering standardised figures...")
        plot_main_regression_bars(outputs.regression_summary, output_dir=output_root / "figures")
        plot_forecast_timeseries(outputs.regression_predictions, output_dir=output_root / "figures")
        plot_warning_pr_curve(outputs.classification_predictions, output_dir=output_root / "figures")
        plot_metric_heatmap(outputs.regression_summary, output_dir=output_root / "figures")
        print("Figure export completed.")


if __name__ == "__main__":
    main()
