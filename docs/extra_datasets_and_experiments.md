# Extra Datasets And Experiments

## Verified Public Data Additions

These datasets were verified as publicly downloadable on 2026-03-19.

| Block | Field | Series ID / Symbol | Source | Frequency | Verified URL |
|---|---|---|---|---|---|
| Credit risk | `hy_oas` | `BAMLH0A0HYM2` | FRED | Daily | https://fred.stlouisfed.org/series/BAMLH0A0HYM2 |
| Credit risk | `ig_oas` | `BAMLC0A0CM` | FRED | Daily | https://fred.stlouisfed.org/series/BAMLC0A0CM |
| Policy uncertainty | `usepu` | `USEPUINDXD` | FRED | Daily | https://fred.stlouisfed.org/series/USEPUINDXD |
| Small-cap risk | `rvx` | `RVXCLS` | FRED | Daily | https://fred.stlouisfed.org/series/RVXCLS |
| Implied-vol term structure | `vxv` | `VXVCLS` | FRED | Daily | https://fred.stlouisfed.org/series/VXVCLS |
| Yield curve | `dgs2` | `DGS2` | FRED | Daily | https://fred.stlouisfed.org/series/DGS2 |
| Interbank stress | `tedrate` | `TEDRATE` | FRED | Daily | https://fred.stlouisfed.org/series/TEDRATE |
| ETF proxy | `iwm.us` | `iwm.us` | Stooq | Daily | https://stooq.com/q/d/l/?s=iwm.us&i=d |

## Added Derived Features

The `experiment_plusdata.yaml` pipeline now engineers:

- `t10y2y_spread`
- `hy_ig_oas_spread`
- `usepu_log`
- `usepu_chg_1d`
- `rvx_premium`
- `vxv_vix_gap`
- `credit_vol_stress`
- `policy_vol_stress`

It also allows optional wavelet decomposition of:

- `hy_ig_oas_spread`
- `vxv_vix_gap`

## Added Experiment Configurations

| Config | Purpose |
|---|---|
| `config/experiment_plusdata.yaml` | Expanded public-data block with credit, policy uncertainty, and extra implied-volatility proxies |
| `config/experiment_plusdata_parkinson.yaml` | Robustness configuration using Parkinson volatility target |

## Recommended New Experiments

### 1. Data Block Expansion

Goal:
- Test whether expanded public macro-risk proxies improve `QLIKE`, `PR-AUC`, and `Brier`.

Recommended run:

```bash
.venv/bin/python scripts/run_formal_batches.py \
  --config config/experiment_plusdata.yaml \
  --concurrency 3 \
  --n-jobs-per-task 1 \
  --indices spx ndq dji \
  --horizons 1 5 10 \
  --tag-prefix plusdata500 \
  --merged-output-tag plusdata500_merged \
  --max-test-steps 500 \
  --plot-after-merge
```

### 2. Target Robustness

Goal:
- Check whether conclusions depend on OHLC volatility proxy choice.

Recommended run:

```bash
.venv/bin/python scripts/run_formal_batches.py \
  --config config/experiment_plusdata_parkinson.yaml \
  --concurrency 3 \
  --n-jobs-per-task 1 \
  --indices spx ndq dji \
  --horizons 1 5 10 \
  --tag-prefix parkinson300 \
  --merged-output-tag parkinson300_merged \
  --max-test-steps 300 \
  --plot-after-merge
```

### 3. Feature Block Ablation

Goal:
- Separate the gain from:
  - core baseline features
  - extra macro-risk block
  - wavelet on extra risk block

Suggested blocks:

- `baseline`
- `baseline + credit/policy block`
- `baseline + credit/policy block + wavelet risk block`

### 4. Market Regime Comparison

Goal:
- Report results for:
  - `2016-2019`
  - `2020-2021`
  - `2022-2025`

This is especially useful for showing whether expanded stress proxies are more helpful in turbulent periods.

## Practical Notes

- `TEDRATE` is verified but marked as discontinued on FRED, so it is better treated as an optional robustness feature than a main-paper feature.
- Stooq did not return a clean Russell 2000 index history in current tests, but `iwm.us` is available and can be used as an ETF-based robustness proxy.
- For the main paper, the cleanest extension path is still `FRED-only` extra macro-risk data because it preserves reproducibility.
