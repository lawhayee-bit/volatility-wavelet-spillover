# Experiment Synthesis (2026-03-20)

## Result Packages

| Package | Role | Status |
|---|---|---|
| `outputs/full10y_refinedfinal_merged` | Main full-sample result package | Completed |
| `outputs/plusdata500_merged` | Expanded-data comparison on 500 OOS steps | Completed |
| `outputs/parkinson300_merged` | Target robustness package using Parkinson volatility | Completed |

## Recommended Usage In Paper

### Main Text

Use:

- `full10y_refinedfinal_merged`

Reason:

- Full-sample rolling evaluation
- Cleanest and most stable main result package
- Best aligned with current article narrative

### Extended Result Section / Main Robustness

Use:

- `plusdata500_merged`

Reason:

- Same core model pool as `formal_merged`
- Expanded public macro-risk data clearly improves `lightgbm`, `wavelet_lightgbm`, and `main_stacking`
- Strong evidence that additional public risk proxies add incremental predictive value

### Appendix Robustness

Use:

- `parkinson300_merged`

Reason:

- Validates that conclusions are not entirely tied to Rogers-Satchell target construction
- Under Parkinson target, nonlinear models become more competitive or dominant in several cells

## Key Findings

### Main Full Sample

- `HAR` remains the strongest overall QLIKE benchmark
- `wavelet_lightgbm` wins in part of medium/long horizons
- `logistic_raw` is the strongest warning model by average `PR-AUC`

### Expanded Public Data

Compared with `formal_merged`, `plusdata500_merged` shows:

- large QLIKE improvements for `lightgbm`
- large QLIKE improvements for `wavelet_lightgbm`
- large QLIKE improvements for `main_stacking`
- no change for naive historical baselines, as expected

This supports the argument that extra public risk data are genuinely useful for nonlinear models.

### Parkinson Robustness

- `wavelet_lightgbm` becomes the best model by average QLIKE
- `lightgbm` is a close second
- `HAR` remains competitive but no longer dominates on average

This supports the robustness claim that model ranking depends partly on volatility proxy definition.

## Practical Writing Strategy

Do not claim:

- universal dominance over `HAR`

Do claim:

- nonlinear and multiscale models gain clear strength after adding richer public risk proxies
- medium/long horizons are more favorable to the proposed approach
- findings are robust to an alternative OHLC volatility proxy
