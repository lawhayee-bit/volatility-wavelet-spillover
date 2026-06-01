from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORE_REG_MODELS = [
    "last_value",
    "hv_5",
    "hv_22",
    "har",
    "harx",
    "lightgbm",
    "wavelet_lightgbm",
    "main_stacking",
]

CORE_CLS_MODELS = [
    "naive_threshold",
    "forecast_threshold",
    "logistic_raw",
    "main_warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal experiment batches with controlled concurrency.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--n-jobs-per-task", type=int, default=1)
    parser.add_argument("--indices", nargs="+", default=["spx", "ndq", "dji"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--tag-prefix", default="formal")
    parser.add_argument("--merged-output-tag", default="formal_merged")
    parser.add_argument("--reg-models", nargs="+", default=None)
    parser.add_argument("--cls-models", nargs="+", default=None)
    parser.add_argument("--max-test-steps", type=int, default=None)
    parser.add_argument("--plot-after-merge", action="store_true")
    return parser.parse_args()


def run_batch(
    config_path: str,
    index_id: str,
    horizon: int,
    n_jobs_per_task: int,
    max_test_steps: int | None,
    tag_prefix: str,
    reg_models: list[str],
    cls_models: list[str],
) -> tuple[str, int]:
    tag = f"{tag_prefix}_{index_id}_h{horizon}"
    log_dir = ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-u",
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--config",
        config_path,
        "--stage",
        "run",
        "--reuse-panel",
        "--skip-download",
        "--output-tag",
        tag,
        "--indices",
        index_id,
        "--horizons",
        str(horizon),
        "--n-jobs",
        str(n_jobs_per_task),
        "--reg-models",
        *reg_models,
        "--cls-models",
        *cls_models,
    ]
    if max_test_steps is not None:
        cmd.extend(["--max-test-steps", str(max_test_steps)])

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    return tag, process.returncode


def main() -> None:
    args = parse_args()
    jobs = [(index_id, horizon) for index_id in args.indices for horizon in args.horizons]
    reg_models = args.reg_models or CORE_REG_MODELS
    cls_models = args.cls_models or CORE_CLS_MODELS

    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_map = {
            executor.submit(
                run_batch,
                args.config,
                index_id,
                horizon,
                args.n_jobs_per_task,
                args.max_test_steps,
                args.tag_prefix,
                reg_models,
                cls_models,
            ): (index_id, horizon)
            for index_id, horizon in jobs
        }
        for future in as_completed(future_map):
            index_id, horizon = future_map[future]
            tag, code = future.result()
            print(f"[batch] {tag} finished with code {code}")
            if code != 0:
                failures.append((tag, code))

    if failures:
        for tag, code in failures:
            print(f"[failure] {tag} -> {code}")
        raise SystemExit(1)

    merge_cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "merge_batch_outputs.py"),
        "--batch-root",
        "outputs",
        "--pattern",
        f"{args.tag_prefix}_*_h*",
        "--output-tag",
        args.merged_output_tag,
    ]
    subprocess.run(merge_cmd, cwd=ROOT, check=True)

    if args.plot_after_merge:
        plot_cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            str(ROOT / "scripts" / "render_merged_plots.py"),
            "--input-tag",
            args.merged_output_tag,
        ]
        subprocess.run(plot_cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
