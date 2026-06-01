# Manuscript Map

## 1. Main Text Result Package

- Primary result package:
  - `outputs/full10y_refinedfinal_merged`
- Main tables:
  - `tables/regression_summary.csv`
  - `tables/classification_summary.csv`
  - `tables/diebold_mariano.csv`
  - `tables/clark_west.csv`
- Main figures:
  - `figures/main_regression_qlike.pdf`
  - `figures/heatmap_qlike.pdf`
  - `figures/forecast_spx_h1_main_stacking.pdf`
  - `figures/warning_pr_spx_h1.pdf`

## 2. Robustness / Extension Result Packages

- Expanded public risk-data experiment:
  - `outputs/plusdata500_merged`
- Alternative Parkinson target:
  - `outputs/parkinson300_merged`

## 3. Recommended Main-Text Tables

- Table 1. Data sources and variable blocks
- Table 2. Baseline models and proposed models
- Table 3. Main regression results under the RS target
- Table 4. Main warning results
- Table 5. Statistical comparison against HAR

## 4. Recommended Main-Text Figures

- Figure 1. Overall methodology flowchart
- Figure 2. Main QLIKE comparison across indices and horizons
- Figure 3. QLIKE heatmap for main regression results
- Figure 4. Example forecast trace for SPX, \(h=1\)
- Figure 5. Precision--recall curve for the warning task
- Figure 6. Relative QLIKE change against HAR across settings
- Figure 7. Warning-score distribution by realized risk state

## 5. Recommended Appendix Tables

- Table A1. Full variable dictionary
- Table A2. Expanded public risk-data experiment
- Table A3. Parkinson-target robustness results
- Table A4. Additional significance-test details
- Table A5. Implementation and hyperparameter summary

## 6. Section-to-Result Mapping

- Introduction:
  - research motivation
  - literature gap
  - contributions
- Data:
  - public data sources
  - sample period
  - target construction
- Methodology:
  - wavelet features
  - hybrid models
  - warning-label definition
- Experimental Design:
  - rolling-window protocol
  - horizons
  - metrics
  - significance tests
- Results:
  - main RS-target results
  - warning results
- Discussion:
  - why HAR remains strong
  - where wavelet/hybrid methods help
  - limits and implications
- Conclusions:
  - concise empirical takeaways
  - future extensions
