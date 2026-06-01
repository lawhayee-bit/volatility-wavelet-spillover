# Reviewer Reproducibility Package

This package documents the project assets needed to reproduce the manuscript
tables, figures, and compiled PDF from the repository snapshot used for
submission.

## Scope

The package is designed to support editorial and reviewer verification of:

- public-data acquisition routes,
- causal feature construction,
- walk-forward experiment launchers,
- merged prediction outputs used by the manuscript,
- figure and table generation scripts,
- manuscript compilation.

## Core Files

- `requirements.txt`
  Python dependency specification.
- `config/experiment.yaml`
  Main experiment configuration.
- `config/experiment_plusdata.yaml`
  Expanded public risk-data configuration.
- `config/experiment_plusdata_parkinson.yaml`
  Alternative-target configuration.
- `config/experiment_spillover.yaml`
  Spillover ablation configuration.
- `config/plot_style.toml`
  Unified manuscript plotting style.
- `scripts/run_pipeline.py`
  Main end-to-end pipeline entry point.
- `scripts/run_formal_batches.py`
  Batch launcher used for formal experiment runs.
- `scripts/run_spillover_ablation.py`
  Spillover nested-comparison script.
- `scripts/generate_innovation_experiments.py`
  Conditional, state-dependent, and event-based extension generator.
- `scripts/compile_manuscript.sh`
  Manuscript compilation script.
- `outputs/full10y_refinedfinal_merged/`
  Main merged prediction package.
- `outputs/plusdata500_merged/`
  Expanded-data merged prediction package.
- `outputs/parkinson300_merged/`
  Parkinson-target merged prediction package.
- `outputs/spillover_merged/`
  Spillover ablation merged prediction package.
- `paper/manuscript/`
  Manuscript source, tables, figures, and references.

## Minimal Reproduction Path

1. Install dependencies from `requirements.txt`.
2. Regenerate the innovation extensions:

```bash
python scripts/generate_innovation_experiments.py
```

3. Compile the manuscript:

```bash
./scripts/compile_manuscript.sh
```

This reproduces the paper assets from the merged experiment outputs already
stored in the repository snapshot.

## Public Raw Data Sources

- Stooq: https://stooq.com
- FRED: https://fred.stlouisfed.org

The repository snapshot intentionally keeps the public raw-data sources and the
processed experiment outputs separate so reviewers can distinguish data access,
feature construction, model evaluation, and manuscript rendering.
