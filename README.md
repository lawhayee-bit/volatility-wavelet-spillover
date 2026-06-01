# Causal Multiscale Wavelet Spillover Learning for Stock Index Volatility Forecasting

<p align="center">
  <a href="https://www.mdpi.com/journal/risks">
    <img src="https://img.shields.io/badge/Journal-Risks%20(MDPI)-blue?style=flat-square&logo=open-access&logoColor=white" alt="Journal: Risks (MDPI)">
  </a>
  <img src="https://img.shields.io/badge/Status-Accepted-brightgreen?style=flat-square" alt="Status: Accepted">
  <a href="https://github.com/lawhayee-bit/volatility-wavelet-spillover/blob/main/paper/manuscript/main.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square&logo=adobe-acrobat-reader&logoColor=white" alt="Paper PDF">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License: MIT">
  <img src="https://img.shields.io/badge/Reproducible-Yes-success?style=flat-square&logo=github" alt="Reproducible">
</p>

<p align="center">
  <b>Official implementation of:</b><br>
  <i>Public-Data Causal Multiscale Wavelet Spillover Learning for Stock Index Volatility Forecasting and Risk Early Warning</i><br>
  <b>Risks</b> (MDPI), 2026 — <i>Accepted</i>
</p>

---

## 👥 Authors

| Name | Affiliation | Contact |
|------|-------------|---------|
| **Hengyan Liu** | Sino-European School of Technology, Shanghai University, Shanghai 200444, China | hengyan@shu.edu.cn |
| **Yisu Shen** | Sino-European School of Technology, Shanghai University, Shanghai 200444, China | yisu@shu.edu.cn |
| **Aiping Jiang** ✉️ *(Corresponding)* | SHU-UTS SILC Business School, Shanghai University, Jiading District, Shanghai 201800, China | ajiang@shu.edu.cn |

---

This repository provides a fully reproducible experiment pipeline for:

| | |
|---|---|
| 📈 | Stock index volatility forecasting (S&P 500, Nasdaq-100, DJIA) |
| 🚨 | Risk early warning system |
| 🌊 | Causal undecimated wavelet multiscale feature extraction |
| 🔗 | Cross-index spillover modelling |
| 📊 | Rolling / walk-forward evaluation with Diebold–Mariano & Clark–West tests |

## 🚀 Quick Start

**1. Install dependencies**

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**2. Download raw data** (Stooq OHLC + FRED macro series)

```bash
python -u scripts/run_pipeline.py --stage download
```

**3. Smoke test** — verify the pipeline end-to-end in minutes

```bash
python -u scripts/run_pipeline.py --smoke --stage all
```

**4. Full reproduction** — replicate all paper results

```bash
python -u scripts/run_formal_batches.py          # 3 indices × 3 horizons
python -u scripts/merge_batch_outputs.py         # merge + summary tables
python -u scripts/render_merged_plots.py         # figures
python -u scripts/generate_paper_assets.py       # paper-ready tables & figures
```

## 📁 Repository Structure

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

## 📦 Main Outputs

| File | Description |
|------|-------------|
| `outputs/predictions/regression_predictions.csv` | All model roll-forward predictions |
| `outputs/predictions/classification_predictions.csv` | Risk warning predictions |
| `outputs/tables/regression_summary.csv` | QLIKE / RMSE / MAE / R² by model |
| `outputs/tables/diebold_mariano.csv` | DM test vs HAR benchmark |
| `outputs/tables/clark_west.csv` | Clark–West test for nested models |
| `outputs/figures/*.pdf` | Publication-ready figures |

## 🔬 Methods at a Glance

### Method Overview

<p align="center">
  <img src="paper/manuscript/figures/Method.png" alt="CMWSL framework architecture" width="85%">
  <br>
  <em>Figure 1 — Overall architecture of the proposed CMWSL stock-index volatility forecasting and risk-warning framework.</em>
</p>

### Main Results — Risk Warning Score Distribution

<p align="center">
  <img src="paper/manuscript/figures/fig08_warning_score_distribution.png" alt="Warning score distribution" width="85%">
  <br>
  <em>Figure 8 — Risk early-warning score distributions across high- and low-volatility regimes. The causal wavelet feature pipeline achieves clear separation between the two regimes, enabling reliable threshold-based alerting.</em>
</p>

<details>
<summary><b>18 Regression Models</b></summary>

| Category | Models |
|----------|--------|
| Linear baselines | Last-value, HV-5, HV-22, HAR, HAR-X |
| Volatility models | GARCH, EGARCH |
| ML baselines | SVR, Random Forest, LightGBM, XGBoost |
| **Main models** | **Wavelet-LightGBM, Stacking (HARX + Wavelet-LGB)** |
| Deep learning | MLP, LSTM, GRU, BiLSTM |

</details>

<details>
<summary><b>Wavelet Feature Engineering</b></summary>

- Causal undecimated DWT (Sym4 basis, 3 levels)
- Applied to: Rogers–Satchell variance, |returns|, VIX-like series
- Per-scale features: coefficients, rolling mean / std, energy
- Cross-index spillover features (d2/d3 inter-index transmission)

</details>

<details>
<summary><b>Risk Early Warning</b></summary>

- 6 classifiers: Logistic, Random Forest, LightGBM, main warning, naive/forecast threshold
- Rolling quantile labels for high-volatility events
- F-β = 2.0 threshold selection (recall-weighted)
- Evaluation: PR-AUC, Brier score, ROC-AUC

</details>

## 📊 Data Sources

| Source | Series | Description |
|--------|--------|-------------|
| [Stooq](https://stooq.com) | ^SPX, ^NDQ, ^DJI | Daily OHLC, 2005–2025 |
| [FRED](https://fred.stlouisfed.org) | DFF, DGS10, T10Y3M, NFCI, VIXCLS, … | Macro & volatility indicators |

> All data is **publicly available** and downloaded automatically by the pipeline.

## � Links

| Resource | URL |
|----------|-----|
| � Paper (PDF) | [paper/manuscript/main.pdf](https://github.com/lawhayee-bit/volatility-wavelet-spillover/blob/main/paper/manuscript/main.pdf) |
| �📰 Journal — *Risks* (MDPI) | https://www.mdpi.com/journal/risks |
| 📊 FRED (macro data) | https://fred.stlouisfed.org |
| 📈 Stooq (market data) | https://stooq.com |
| 🐍 PyWavelets | https://pywavelets.readthedocs.io |
| 💡 LightGBM | https://lightgbm.readthedocs.io |
| 🤗 ARCH library | https://arch.readthedocs.io |

## 📄 Citation

If you use this code or build on this work, please cite:

**Plain text (MDPI style):**

> Liu, H.; Shen, Y.; Jiang, A. Public-Data Causal Multiscale Wavelet Spillover Learning for Stock Index Volatility Forecasting and Risk Early Warning. *Risks* **2026**, *XX*, XXXX. https://doi.org/10.3390/risksXXXXXXXX

**BibTeX:**

```bibtex
@article{liu2026cmwsl,
  author   = {Liu, Hengyan and Shen, Yisu and Jiang, Aiping},
  title    = {Public-Data Causal Multiscale Wavelet Spillover Learning for
              Stock Index Volatility Forecasting and Risk Early Warning},
  journal  = {Risks},
  year     = {2026},
  volume   = {XX},
  number   = {XX},
  pages    = {XXXX},
  doi      = {10.3390/risksXXXXXXXX},
  url      = {https://doi.org/10.3390/risksXXXXXXXX}
}
```

> Volume, issue, article number, and DOI will be updated upon official online publication.

## 🙏 Acknowledgements

We sincerely thank all collaborators and advisors who contributed to this work.

Special thanks to **[Haiyi Li (Gatsby0916)](https://github.com/Gatsby0916)** for his invaluable guidance, expert advice, and consistent support throughout the entire research and development process of this project.

We also gratefully acknowledge our supervisors and research partners for their insightful feedback, constructive discussions, and encouragement at every stage of this work.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
