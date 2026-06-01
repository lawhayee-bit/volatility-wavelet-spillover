#!/usr/bin/env python3
"""
Generate MDPI Graphical Abstract  –  v2 (complete redesign).

Output: paper/manuscript/figures/graphical_abstract.png  (~850 × 500 px)
        paper/manuscript/figures/graphical_abstract.pdf
Layout (left → right):
  [A] Public-Data Inputs   [B] Wavelet Decomposition   [C] ML Pipeline   [D] Key Results
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── colour palette (paper standard) ─────────────────────────────────────────
C_NAVY    = "#1F3A5F"
C_RUST    = "#A54E2A"
C_GREEN   = "#4C6A5D"
C_GOLD    = "#B08B3E"
C_DARK    = "#1F1F1F"
C_CREAM   = "#F6F2EA"
C_WHITE   = "#FFFFFF"
C_LGREY   = "#EBEBEB"
C_LNAVY   = "#D1DCE8"
C_LRUST   = "#F0D8CC"
C_LGOLD   = "#EDE2C8"
C_LGREEN  = "#D4E0DB"
C_SLATE   = "#5A6772"
C_MIDNAV  = "#2E567A"

FONT = "Liberation Serif"

# ── figure ───────────────────────────────────────────────────────────────────
FW, FH = 8.0, 4.2
DPI     = 200          # → ~1250 × 655 px after tight crop; satisfies MDPI min 1100 × 560 (w × h)

fig, ax = plt.subplots(figsize=(FW, FH), facecolor=C_WHITE)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.2, alpha=1.0, zorder=2, pad=0.06):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(p)

def txt(x, y, s, fs=7.5, c=C_DARK, ha="center", va="center",
        bold=False, italic=False, zorder=6):
    kw = dict(fontsize=fs, color=c, ha=ha, va=va,
              fontfamily=FONT, zorder=zorder)
    if bold:   kw["fontweight"] = "bold"
    if italic: kw["fontstyle"]  = "italic"
    ax.text(x, y, s, **kw)

def arr(x1, y1, x2, y2, ec=C_DARK, lw=1.4, hw=0.10, hl=0.11, zorder=5):
    ax.annotate("",
                xy=(x2, y2), xytext=(x1, y1),
                xycoords="data", textcoords="data", zorder=zorder,
                arrowprops=dict(
                    arrowstyle=f"-> ,head_width={hw},head_length={hl}",
                    color=ec, lw=lw,
                ))

def spark(cx, cy, w, h, color, seed=0, lw=0.9):
    """Draw a small random time series."""
    rng = np.random.default_rng(seed)
    n = 35
    y = np.cumsum(rng.normal(0, 0.5, n))
    y = (y - y.min()) / (y.max() - y.min() + 1e-9) * h + cy
    x = np.linspace(cx - w / 2, cx + w / 2, n)
    ax.plot(x, y, color=color, lw=lw, zorder=5, solid_capstyle="round")

def wavelet_band(bx, by, bw, bh, label, highlighted=False, freq_cycles=3):
    """Draw one wavelet frequency band with interior wave and optional badge."""
    fc = C_LRUST  if highlighted else C_LNAVY
    ec = C_RUST   if highlighted else C_MIDNAV
    lw = 1.6      if highlighted else 0.9
    clr = C_RUST  if highlighted else C_SLATE
    rbox(bx, by, bw, bh, fc=fc, ec=ec, lw=lw, pad=0.03)
    # Cross-index badge on the right for highlighted bands
    badge_w = 0.50
    if highlighted:
        bx_b = bx + bw - badge_w - 0.04
        rbox(bx_b, by + bh/2 - 0.13, badge_w, 0.26,
             fc=C_RUST, ec=C_RUST, lw=0, pad=0.02)
        txt(bx_b + badge_w/2, by + bh/2 + 0.04, "cross-index",
            fs=5.5, c=C_WHITE, bold=True)
        txt(bx_b + badge_w/2, by + bh/2 - 0.09, "spillover",
            fs=5.0, c=C_WHITE)
    # label (centred in the non-badge part)
    x_lbl = bx + (bw - (badge_w + 0.04) if highlighted else bw) / 2
    txt(x_lbl, by + bh / 2, label,
        fs=6.2, c=ec, bold=highlighted)
    # decorative interior sine (stop before badge area)
    wave_x1 = bx + bw - (badge_w + 0.18 if highlighted else 0.04)
    xw = np.linspace(bx + 0.04, wave_x1, 90)
    amp = bh * 0.28
    yw  = amp * np.sin(
        2 * np.pi * freq_cycles * (xw - bx) / bw
    ) + by + bh / 2
    ax.plot(xw, yw, color=clr, lw=0.75,
            alpha=0.55 if highlighted else 0.35, zorder=5)

# ════════════════════════════════════════════════════════════════════════════
#  TITLE STRIP
# ════════════════════════════════════════════════════════════════════════════
rbox(0.06, 3.90, FW - 0.12, 0.24, fc=C_NAVY, ec=C_NAVY, lw=0, pad=0.04)
txt(FW / 2, 4.02,
    "Public-Data Causal Multiscale Wavelet Spillover Learning"
    " for Stock Index Volatility Forecasting and Risk Early Warning",
    fs=8.2, c=C_WHITE, bold=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION BOUNDARIES
# ════════════════════════════════════════════════════════════════════════════
# x regions (inner border):
# [A] Data      0.06 – 1.80
# [B] Wavelet   2.00 – 3.95
# [C] Model     4.15 – 5.70
# [D] Output    5.90 – 7.94
# y main area:  0.10 – 3.85

# ════════════════════════════════════════════════════════════════════════════
#  SECTION A — PUBLIC DATA INPUTS
# ════════════════════════════════════════════════════════════════════════════
AX, AY, AW, AH = 0.06, 0.10, 1.74, 3.75
rbox(AX, AY, AW, AH, fc=C_CREAM, ec=C_NAVY, lw=1.4)
txt(AX + AW / 2, AY + AH - 0.18, "Public-Data Inputs",
    fs=8, c=C_NAVY, bold=True)

# Three index boxes — S&P 500 (primary) on top, DJIA at bottom
indices = [("DJIA", 99), ("Nasdaq-100", 42), ("S&P 500", 0)]
for k, (label, seed) in enumerate(indices):
    yb = AY + 0.20 + k * 1.05
    rbox(AX + 0.10, yb, AW - 0.20, 0.85,
         fc=C_WHITE, ec=C_SLATE, lw=0.9, pad=0.04)
    txt(AX + AW / 2, yb + 0.68, label,
        fs=7.5, c=C_DARK, bold=True)
    spark(AX + AW / 2, yb + 0.12, AW - 0.38, 0.44,
          color=C_NAVY, seed=seed)

# OHLC + FRED strip at bottom of section A
rbox(AX + 0.10, AY + 0.12, AW - 0.20, 0.22,
     fc=C_LGOLD, ec=C_GOLD, lw=0.9, pad=0.03)
txt(AX + AW / 2, AY + 0.23, "OHLC price data  +  FRED macro / VIX",
    fs=6.2, c=C_DARK)

# ════════════════════════════════════════════════════════════════════════════
#  ARROW  A → B
# ════════════════════════════════════════════════════════════════════════════
arr(AX + AW + 0.05, AY + AH / 2,
    2.00 - 0.05, AY + AH / 2,
    ec=C_NAVY, lw=1.6, hw=0.13)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION B — MULTISCALE WAVELET DECOMPOSITION
# ════════════════════════════════════════════════════════════════════════════
BX, BY, BW, BH = 2.00, 0.10, 1.95, 3.75
rbox(BX, BY, BW, BH, fc=C_LNAVY, ec=C_NAVY, lw=1.4)
txt(BX + BW / 2, BY + BH - 0.18,
    "Multiscale Wavelet Decomposition",
    fs=8, c=C_NAVY, bold=True)

# Input signal at top of wavelet box
rbox(BX + 0.10, BY + BH - 0.60, BW - 0.20, 0.32,
     fc=C_WHITE, ec=C_MIDNAV, lw=1.0, pad=0.03)
txt(BX + BW / 2, BY + BH - 0.44,
    "Return / Volatility signal  \u2192  SWT (stationary wavelet)",
    fs=6.2, c=C_DARK, italic=True)

# Four wavelet bands stacked bottom-up  (paper uses 3 decomposition levels: d1,d2,d3 + approx)
band_labels = ["d1  (~2 day)", "d2  (~4-8 day)",
               "d3  (~8-16 day)", "Approx.  (>16 day)"]
highlights   = [False, True, True, False]
freq_list    = [4, 3, 2, 0]

bw = BW - 0.20
bh = 0.55
b_gap = 0.10
b_y0 = BY + 0.18

for i, (lab, hl, fc_) in enumerate(zip(band_labels, highlights, freq_list)):
    by = b_y0 + i * (bh + b_gap)
    wavelet_band(BX + 0.10, by, bw, bh, lab,
                 highlighted=hl, freq_cycles=max(1, 4 - i))

# Clean legend note at the bottom of the wavelet section
txt(BX + BW / 2, BY + 0.14,
    "d2, d3  \u2192  cross-index spillover features",
    fs=5.8, c=C_RUST, bold=True)

# ════════════════════════════════════════════════════════════════════════════
#  ARROW  B → C
# ════════════════════════════════════════════════════════════════════════════
arr(BX + BW + 0.05, AY + AH / 2,
    4.15 - 0.05, AY + AH / 2,
    ec=C_NAVY, lw=1.6, hw=0.13)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION C — HYBRID ML
# ════════════════════════════════════════════════════════════════════════════
CX, CY, CW, CH = 4.15, 0.10, 1.55, 3.75
rbox(CX, CY, CW, CH, fc=C_LRUST, ec=C_RUST, lw=1.4)
txt(CX + CW / 2, CY + CH - 0.18,
    "Hybrid Machine Learning",
    fs=8, c=C_RUST, bold=True)

# Feature fusion box
rbox(CX + 0.10, CY + CH - 0.72, CW - 0.20, 0.45,
     fc=C_WHITE, ec=C_RUST, lw=1.1, pad=0.04)
txt(CX + CW / 2, CY + CH - 0.54,
    "Feature fusion",
    fs=7, c=C_RUST, bold=True)
txt(CX + CW / 2, CY + CH - 0.67,
    "HAR · Wavelet · Macro · VIX",
    fs=5.8, c=C_DARK, italic=True)

# Arrow from fusion down
arr(CX + CW / 2, CY + CH - 0.72,
    CX + CW / 2, CY + CH - 1.05,
    ec=C_RUST, lw=1.2, hw=0.09)

# Walk-forward protocol box
rbox(CX + 0.10, CY + CH - 1.48, CW - 0.20, 0.36,
     fc=C_LGOLD, ec=C_GOLD, lw=1.0, pad=0.03)
txt(CX + CW / 2, CY + CH - 1.30,
    "Walk-forward protocol",
    fs=6.5, c=C_DARK, bold=True)
txt(CX + CW / 2, CY + CH - 1.44,
    "2016 – 2025  (2 513 steps)",
    fs=6.0, c=C_DARK)

# Regression model
arr(CX + CW / 2, CY + CH - 1.48,
    CX + CW / 2, CY + CH - 1.78,
    ec=C_RUST, lw=1.2, hw=0.09)

rbox(CX + 0.10, CY + CH - 2.28, CW - 0.20, 0.42,
     fc=C_WHITE, ec=C_NAVY, lw=1.1, pad=0.04)
txt(CX + CW / 2, CY + CH - 2.08,
    "LightGBM  (regression)",
    fs=6.8, c=C_NAVY, bold=True)
txt(CX + CW / 2, CY + CH - 2.22,
    r"$\hat{y}^{(h)}$ : volatile forecast",
    fs=6.0, c=C_DARK, italic=True)

# Classification model
arr(CX + CW / 2, CY + CH - 2.28,
    CX + CW / 2, CY + CH - 2.55,
    ec=C_RUST, lw=1.2, hw=0.09)

rbox(CX + 0.10, CY + CH - 3.05, CW - 0.20, 0.42,
     fc=C_WHITE, ec=C_GREEN, lw=1.1, pad=0.04)
txt(CX + CW / 2, CY + CH - 2.85,
    "Logistic (classification)",
    fs=6.8, c=C_GREEN, bold=True)
txt(CX + CW / 2, CY + CH - 2.99,
    r"Pr$(z=1\mid x^{\,\rm warn})$",
    fs=6.0, c=C_DARK, italic=True)

# Ablation / evaluation note
rbox(CX + 0.10, CY + 0.12, CW - 0.20, 0.30,
     fc=C_LGREEN, ec=C_GREEN, lw=0.9, pad=0.03)
txt(CX + CW / 2, CY + 0.27,
    "3-step controlled ablation",
    fs=6.5, c=C_GREEN, bold=True)
txt(CX + CW / 2, CY + 0.15,
    "Clark–West  ·  Diebold–Mariano",
    fs=5.8, c=C_DARK)

# ════════════════════════════════════════════════════════════════════════════
#  ARROW  C → D
# ════════════════════════════════════════════════════════════════════════════
arr(CX + CW + 0.05, AY + AH / 2,
    5.90 - 0.05, AY + AH / 2,
    ec=C_NAVY, lw=1.6, hw=0.13)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION D — OUTPUTS & KEY FINDINGS
# ════════════════════════════════════════════════════════════════════════════
DX, DY, DW, DH = 5.90, 0.10, 2.04, 3.75
rbox(DX, DY, DW, DH, fc=C_LGOLD, ec=C_GOLD, lw=1.4)
txt(DX + DW / 2, DY + DH - 0.18,
    "Results & Outputs",
    fs=8, c=C_GOLD, bold=True)

# ── Forecast output panel ──
rbox(DX + 0.10, DY + DH - 1.72, DW - 0.20, 1.42,
     fc=C_WHITE, ec=C_NAVY, lw=1.1, pad=0.04)
txt(DX + DW / 2, DY + DH - 0.40,
    "Volatility Forecast  (QLIKE)",
    fs=7, c=C_NAVY, bold=True)

# Mock forecast vs actual mini-chart
rng = np.random.default_rng(7)
n   = 45
t   = np.linspace(0, 1, n)
actual   = 0.5 + 0.3 * np.sin(3 * np.pi * t) + 0.12 * np.cumsum(rng.normal(0, 0.05, n))
actual   = np.clip(actual, 0.1, 1.0)
forecast = actual + rng.normal(0, 0.06, n)
forecast = np.clip(forecast, 0.05, 1.0)

px_x = np.linspace(DX + 0.18, DX + DW - 0.18, n)
px_y0 = DY + DH - 1.58
px_h  = 0.82

def scale_y(v, y0, h):
    vmin, vmax = actual.min(), actual.max()
    return y0 + (v - vmin) / (vmax - vmin + 1e-9) * h

ax.plot(px_x, scale_y(actual,   px_y0, px_h), color=C_NAVY, lw=1.4,
        label="Actual",   zorder=5)
ax.plot(px_x, scale_y(forecast, px_y0, px_h), color=C_RUST, lw=1.2,
        ls="--", label="Wavelet-LightGBM", zorder=5)
txt(DX + 0.22, px_y0 + px_h + 0.04, "Actual",
    fs=5.5, c=C_NAVY, ha="left")
txt(DX + 0.22, px_y0 + px_h - 0.06, "Wavelet-LightGBM",
    fs=5.5, c=C_RUST, ha="left")

# Key CW result callout inside the forecast box
rbox(DX + 0.14, px_y0 - 0.01, DW - 0.28, 0.24,
     fc=C_LRUST, ec=C_RUST, lw=1.1, pad=0.03)
txt(DX + DW / 2, px_y0 + 0.08,
    "CW = 4.83, p < 0.001  (SPX, h = 10)",
    fs=6.2, c=C_RUST, bold=True)
txt(DX + DW / 2, px_y0 - 0.002,
    "Spillover gain: 5 / 9 index\u2013horizon cells",
    fs=5.8, c=C_DARK)

# ── Early-warning panel ──
rbox(DX + 0.10, DY + 0.44, DW - 0.20, 1.30,
     fc=C_WHITE, ec=C_GREEN, lw=1.1, pad=0.04)
txt(DX + DW / 2, DY + 0.44 + 1.16,
    "Risk Early Warning",
    fs=7, c=C_GREEN, bold=True)
txt(DX + DW / 2, DY + 0.44 + 0.99,
    r"Pr$(z = 1 \mid x^{\,\rm warn})$",
    fs=6.5, c=C_DARK, italic=True)

# Mock probability bar chart  (just decorative)
bar_heights = [0.45, 0.70, 0.55, 0.80, 0.38, 0.60, 0.88, 0.42, 0.65, 0.50]
n_bars  = len(bar_heights)
bar_w   = (DW - 0.40) / (n_bars * 1.35)
bar_x0  = DX + 0.20

for j, bh_j in enumerate(bar_heights):
    bx_j = bar_x0 + j * (bar_w * 1.35)
    by_j = DY + 0.56
    bh_jj = bh_j * 0.62
    fc = C_RUST if bh_j > 0.65 else C_LGREEN
    ec = C_RUST if bh_j > 0.65 else C_GREEN
    rbox(bx_j, by_j, bar_w, bh_jj, fc=fc, ec=ec, lw=0.7, pad=0.01)

# threshold line
thr_y = DY + 0.56 + 0.65 * 0.62
ax.axhline(thr_y, xmin=(DX + 0.18) / FW, xmax=(DX + DW - 0.10) / FW,
           color=C_SLATE, lw=0.9, ls="--", zorder=5)
txt(DX + DW - 0.15, thr_y + 0.04, "τ", fs=6, c=C_SLATE, ha="right")

# PR-AUC note
rbox(DX + 0.14, DY + 0.16, DW - 0.28, 0.24,
     fc=C_LGREEN, ec=C_GREEN, lw=1.0, pad=0.03)
txt(DX + DW / 2, DY + 0.28,
    "Logistic: most stable PR-AUC", fs=6.5, c=C_GREEN, bold=True)
txt(DX + DW / 2, DY + 0.17,
    "across all indices & horizons", fs=6.0, c=C_DARK)

# ════════════════════════════════════════════════════════════════════════════
#  BOTTOM FOOTNOTE BANNER
# ════════════════════════════════════════════════════════════════════════════
rbox(0.06, 0.02, FW - 0.12, 0.10, fc=C_LGREY, ec=C_LGREY, lw=0, pad=0.01)
txt(FW / 2, 0.07,
    "Three US equity indices (S&P 500 · Nasdaq-100 · DJIA) · "
    "2016–2025 · HAR + SWT + LightGBM + Logistic · "
    "walk-forward evaluation",
    fs=5.5, c=C_SLATE)

# ════════════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════════════
out_dir = os.path.join(
    os.path.dirname(__file__), "..", "paper", "manuscript", "figures"
)
os.makedirs(out_dir, exist_ok=True)

for ext in ("png", "pdf"):
    path = os.path.join(out_dir, f"graphical_abstract.{ext}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved: {path}")

plt.close(fig)
print("Done.")
