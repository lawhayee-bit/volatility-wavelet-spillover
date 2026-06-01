# Causal Multiscale Wavelet Spillover Learning for Stock Index Volatility Forecasting

> **Paper accepted** — *Risks* (MDPI), 2026.
>
> **Title:** Public-Data Causal Multiscale Wavelet Spillover Learning for Stock Index Volatility Forecasting and Risk Early Warning

This repository is the official implementation for the above paper. It provides a fully reproducible experiment pipeline for:

- stock index volatility forecasting (S&P 500, Nasdaq-100, DJIA)
- risk early warning system
- causal undecimated wavelet multiscale feature extraction
- cross-index spillover modelling
- rolling / walk-forward evaluation with statistical tests (Diebold–Mariano, Clark–West)

## Quick Start

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download raw data (Stooq OHLC + FRED macro series):

```bash
python -u scripts/run_pipeline.py --stage download
```

Run a short smoke experiment to verify the pipeline end-to-end:

```bash
python -u scripts/run_pipeline.py --smoke --stage all
```

To reproduce the full paper results, run the formal batch experiments:

```bash
python -u scripts/run_formal_batches.py          # 3 indices × 3 horizons
python -u scripts/merge_batch_outputs.py         # merge + summary tables
python -u scripts/render_merged_plots.py         # figures
python -u scripts/generate_paper_assets.py       # paper-ready tables & figures
```

## Repository Structure

```
config/         experiment YAML configurations (main / plusdata / spillover)
data/           raw data downloaded by the pipeline (not tracked in git)
docs/           extended methodology notes and experiment synthesis
outputs/        experiment results, figures, tables (not tracked in git)
paper/          manuscript source (LaTeX), compiled PDF, and cover letter
plotting/       unified plot style loader
scripts/        pipeline entry-points and paper-asset generation scripts
src/            volatility_lab Python package (data, features, models, stats)
tests/          unit tests
```

## Main Outputs

- `outputs/predictions/regression_predictions.csv`
- `outputs/predictions/classification_predictions.csv`
- `outputs/tables/regression_summary.csv`
- `outputs/tables/classification_summary.csv`
- `outputs/tables/diebold_mariano.csv`
- `outputs/tables/clark_west.csv`
- `outputs/figures/*.pdf`

## Current Scope

Implemented:

- public data download from `Stooq` and `FRED`
- OHLC-based volatility target construction (Rogers–Satchell, Parkinson)
- 18 regression models: HAR / HAR-X / GARCH / EGARCH / SVR / Random Forest / LightGBM / XGBoost / wavelet-LightGBM / MLP / LSTM / GRU / BiLSTM / stacking
- 6 classification models for risk early warning
- causal undecimated DWT multiscale features (Sym4, 3 levels)
- cross-index wavelet spillover features
- unified plotting style controlled by `config/plot_style.toml`
- Diebold–Mariano and Clark–West statistical tests

## Citation

> *To be updated upon official publication.*

## License

MIT
