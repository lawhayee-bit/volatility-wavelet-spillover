22 April 2026

Editorial Office  
*Risks*  
MDPI

---

Dear Editors,

We submit the manuscript **"Public-Data Causal Multiscale Wavelet Spillover Learning for Stock Index Volatility Forecasting and Risk Early Warning"** for consideration as an Article in *Risks*.

**Research problem and motivation.** Accurate volatility forecasting and timely risk early warning are foundational requirements for financial risk management. Value-at-Risk calculations, portfolio risk limits, derivative hedging ratios, and stress-test scenario calibrations all depend on forward-looking volatility signals that remain reliable when markets depart from average conditions---yet existing methods either rely on strong linear persistence alone or on complex machine learning systems that lack transparency and rigorous out-of-sample evaluation protocols. Our study addresses this gap by building a reproducible, public-data framework that integrates volatility forecasting and high-volatility early warning within a single, causally strict pipeline.

**Contributions relevant to *Risks*.** The manuscript makes four contributions directly aligned with the scope of *Risks* in financial risk management and statistical modelling:

1. **Integrated market risk monitoring framework.** We propose a Causal Multiscale Wavelet Spillover Learning (CMWSL) framework that links volatility forecasting and high-volatility early warning within a single causal, public-data pipeline---eliminating the information gap that typically separates risk quantification from operational risk alerts.
2. **Frequency-specific cross-market risk transmission.** We formalize and test cross-index wavelet spillover as a nested incremental-information problem, providing traceable evidence that medium- and long-scale wavelet channels from peer indices materially contribute to portfolio volatility risk exposure at medium and longer horizons.
3. **Risk-management-relevant evaluation.** Beyond average forecast rankings, we report tail-conditioned, market-state-conditioned, rolling-window, and stress-event diagnostics that reveal when multiscale representation delivers actionable improvements---directly informing when a risk manager should activate supplementary models.
4. **Auditable, reproducible risk model design.** The entire pipeline is built on publicly available daily market and macro-financial data (Stooq; FRED), with strict walk-forward chronology, deterministic scripts, and reviewer-ready experiment artifacts consistent with model risk management transparency requirements.

**Selected findings.** The empirical evidence covers the S\&P 500, Nasdaq-100, and Dow Jones Industrial Average over a 2513-step out-of-sample period from 2016 to 2025. HAR remains the dominant benchmark at short horizons, confirming that any credible market risk model must compete against persistence---not circumvent it. The CMWSL extension delivers statistically significant improvements at $h = 5$ and $h = 10$ in richer information environments, with Clark--West tests detecting significant spillover gains in five of nine index--horizon cells ($\text{CW} = 4.83$, $p < 0.001$ for S\&P 500 at $h = 10$). Critically, tail-conditioned diagnostics show that all gains concentrate in upper-volatility regimes and synchronized stress episodes---the conditions in which risk management decisions are most consequential and Value-at-Risk models most likely to underestimate true exposure. The integrated logistic early-warning classifier delivers stable precision--recall performance across settings and provides a directly interpretable, threshold-calibrated alert mechanism suited to operational risk monitoring.

**Fit with *Risks*.** *Risks* is the natural venue for this work. The paper addresses core topics in the journal's scope---financial risk management, market risk, and statistical modelling---with a study that is finance-specific in its question, data, interpretation, and practical implications. Recent publications in *Risks* (e.g., LSTM--Attention models for investment risk forecasting; ML-powered risk scoring) confirm that rigorous, reproducible ML-based financial risk research is valued by the journal's readership.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have read and approved the submitted version. We declare no conflicts of interest; the research received no external funding.

We appreciate your consideration and look forward to your response.

Sincerely,

**Aiping Jiang** *(Corresponding Author)*  
SHU-UTS SILC Business School, Shanghai University  
Jiading District, Shanghai 201800, People's Republic of China  
ajiang@shu.edu.cn

on behalf of Hengyan Liu and Yisu Shen  
Sino-European School of Technology, Shanghai University
