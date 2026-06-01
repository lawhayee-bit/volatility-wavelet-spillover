"""
Generate publication-quality figures for the cross-index spillover ablation analysis.
Uses the project-standard plotting style (config/plot_style.toml + plotting/style.py).
Outputs:
  paper/manuscript/figures/fig_spillover_qlike_bars.pdf
  paper/manuscript/figures/fig_spillover_cw_heatmap.pdf
  paper/manuscript/figures/fig_spillover_gain_ladder.pdf
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# ── ensure project root is on the path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plotting.style import load_plot_style, apply_plot_style

# ── apply unified paper style ─────────────────────────────────────────────────
_style = load_plot_style("bar")
apply_plot_style(_style)

# pull palette constants from the TOML
_pal    = _style["palette"]
SERIES  = _pal["series"]          # ["#1F3A5F","#A54E2A","#4C6A5D","#B08B3E","#7B4B3A","#5A6772"]
NDARK   = _pal["neutral_dark"]    # "#1F1F1F"
NMID    = _pal["neutral_mid"]     # "#666666"
NLIGHT  = _pal["neutral_light"]   # "#D9D9D9"
HDIV    = _pal["heatmap_diverging"]  # ["#1F3A5F","#F6F2EA","#A54E2A"]
HSEQ    = _pal["heatmap_sequential"] # ["#F6F2EA",...,"#1F3A5F"]

# ── data paths ─────────────────────────────────────────────────────────────────
MERGED   = os.path.join(ROOT, "outputs", "spillover_merged")
FIG_DIR  = os.path.join(ROOT, "paper", "manuscript", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

reg_path  = os.path.join(MERGED, "tables", "regression_summary.csv")
test_path = os.path.join(MERGED, "tables", "spillover_ablation_tests.csv")

if not os.path.exists(reg_path):
    sys.exit(f"[ERROR] {reg_path} not found. Run merge first.")

reg   = pd.read_csv(reg_path)
tests = pd.read_csv(test_path)

# ── model / index meta ─────────────────────────────────────────────────────────
# Assign palette colours matching the paper convention:
#   HAR          → series[0]  navy    #1F3A5F
#   Wavelet-LGB  → series[1]  terracotta  #A54E2A
#   Spillover-LGB→ series[2]  dark teal   #4C6A5D
IDX_LABELS   = {"dji": "DJIA", "ndq": "Nasdaq-100", "spx": "S&P 500"}
# Semantic colour mapping — identical to generate_paper_assets.py _semantic_colors()
#   har          → palette[5]  #5A6772  steel blue-grey
#   wavelet_lgbm → palette[3]  #B08B3E  amber/gold
#   spillover    → palette[2]  #4C6A5D  dark teal
MODEL_COLORS = {
    "har":                SERIES[5],   # #5A6772  steel blue-grey
    "wavelet_lightgbm":   SERIES[3],   # #B08B3E  amber/gold
    "spillover_lightgbm": SERIES[2],   # #4C6A5D  dark teal
}
MODEL_LABELS = {
    "har":                "HAR",
    "wavelet_lightgbm":   "Wavelet-LGB",
    "spillover_lightgbm": "Spillover-LGB",
}
HORIZONS = [1, 5, 10]
INDICES  = ["dji", "ndq", "spx"]

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: grouped QLIKE bar chart  (fig_spillover_qlike_bars.pdf)
# ══════════════════════════════════════════════════════════════════════════════
def make_qlike_bars():
    models_plot = ["har", "wavelet_lightgbm", "spillover_lightgbm"]
    n_models = len(models_plot)
    width    = 0.22
    gap      = 0.14          # gap between index groups

    x_positions  = []
    x_labels     = []
    cell_centers = []
    pos = 0.0
    for idx in INDICES:
        for h in HORIZONS:
            cell_centers.append(pos + (n_models - 1) * width / 2)
            x_positions.append(pos)
            x_labels.append(f"$h={h}$")
            pos += n_models * width + 0.08
        pos += gap

    # index background bands  — use lightest heatmap_sequential tone
    band_alphas = [0.18, 0.0, 0.18]   # alternating tint of HSEQ[0]
    idx_ranges = []
    pos2 = 0.0
    for k, idx in enumerate(INDICES):
        grp_start = pos2
        for h in HORIZONS:
            pos2 += n_models * width + 0.08
        idx_ranges.append((grp_start - 0.05, pos2 - 0.08 + 0.05))
        pos2 += gap

    fig, ax = plt.subplots(figsize=(_style["fig_width_in"] * 1.55,
                                    _style["fig_height_in"]))

    for k, (lo, hi) in enumerate(idx_ranges):
        ax.axvspan(lo, hi, color=HSEQ[0], alpha=band_alphas[k], zorder=0)

    offset = np.arange(n_models) * width

    for cell_i, (idx, h) in enumerate([(i, h) for i in INDICES for h in HORIZONS]):
        base_x = x_positions[cell_i]
        for m_i, model in enumerate(models_plot):
            row = reg[(reg["index_id"] == idx) & (reg["horizon"] == h) & (reg["model"] == model)]
            if row.empty:
                continue
            qlike = row["qlike"].values[0]
            xpos  = base_x + offset[m_i]
            ax.bar(xpos, qlike, width=width * 0.92,
                   color=MODEL_COLORS[model], alpha=0.90,
                   edgecolor=NDARK, linewidth=0.5, zorder=3)

    # CW significance markers (spillover vs wavelet)
    s2cw = tests[tests["comparison"].str.contains("Step2 CW")].copy()
    for cell_i, (idx, h) in enumerate([(i, hv) for i in INDICES for hv in HORIZONS]):
        base_x = x_positions[cell_i]
        cw_row = s2cw[(s2cw["index_id"] == idx) & (s2cw["horizon"] == h)]
        if cw_row.empty:
            continue
        p     = cw_row["p_value"].values[0]
        b_wins = cw_row["b_wins"].values[0]
        sig   = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if sig:
            row = reg[(reg["index_id"] == idx) & (reg["horizon"] == h) & (reg["model"] == "spillover_lightgbm")]
            if not row.empty:
                qlike = row["qlike"].values[0]
                xpos  = base_x + offset[2]
                ax.text(xpos + width / 2, qlike + 0.012, sig,
                        ha="center", va="bottom", fontsize=9,
                        color=SERIES[2], fontweight="bold")

    ax.set_xticks(cell_centers)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("QLIKE (lower = better)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ── index section titles in axes-fraction coords ─────────────────────────
    # This avoids data-coord/legend-coord collisions at the axes top edge.
    xlim  = ax.get_xlim()
    xspan = xlim[1] - xlim[0]
    for idx, (lo, hi) in zip(INDICES, idx_ranges):
        cx_frac = ((lo + hi) / 2 - xlim[0]) / xspan
        ax.text(cx_frac, 1.02, IDX_LABELS[idx],
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=NDARK, transform=ax.transAxes)

    # ── legend: horizontal, centred, above the index titles ─────────────────
    patches = [mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m],
                               edgecolor=NDARK, linewidth=0.5)
               for m in models_plot]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, 1.14), ncol=3,
              frameon=False, fontsize=8.5)

    # ── note: placed in figure-fraction coords — never overlaps tick labels ──
    fig.text(0.02, 0.01,
             r"Note: $^*$/$^{**}$/$^{***}$ = CW $p<0.05/0.01/0.001$"
             r" (one-tailed nested test, Spillover-LGB vs. Wavelet-LGB)",
             fontsize=9, color=NMID, style="italic",
             ha="left", va="bottom")

    # leave room at top for legend + index titles, at bottom for note
    fig.tight_layout(rect=[0, 0.08, 1, 0.80])
    out = os.path.join(FIG_DIR, "fig_spillover_qlike_bars.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: CW p-value heatmap  (fig_spillover_cw_heatmap.pdf)
# ══════════════════════════════════════════════════════════════════════════════
def make_cw_heatmap():
    s1_dm = tests[tests["comparison"].str.contains("Step1")]
    s2_cw = tests[tests["comparison"].str.contains("Step2 CW")]

    # build diverging colormap from paper palette
    # negative score → SERIES[0] (navy, b loses), zero → HSEQ[0] (cream), positive → SERIES[2] (teal, b wins)
    cmap_div = LinearSegmentedColormap.from_list(
        "paper_div",
        [(0.0, SERIES[0]),    # navy  – b loses
         (0.5, HSEQ[0]),      # cream – neutral
         (1.0, SERIES[2])],   # teal  – b wins
    )

    _hm_style = load_plot_style("heatmap")
    w = _hm_style["fig_width_in"] * 1.55
    h = _hm_style["fig_height_in"] * 0.78

    fig, axes = plt.subplots(1, 2, figsize=(w, h))

    for ax_i, (subset, step_title) in enumerate([
        (s1_dm, "Step 1 \u2014 HAR vs. Wavelet-LGB\n(DM two-tailed; \u2191 = wavelet wins)"),
        (s2_cw, "Step 2 \u2014 Wavelet-LGB vs. Spillover-LGB\n(CW one-tailed nested; \u2191 = spillover wins)"),
    ]):
        ax = axes[ax_i]
        mat_p  = np.ones((3, 3))
        mat_bw = np.zeros((3, 3), dtype=bool)
        mat_st = np.zeros((3, 3))

        for i_i, idx in enumerate(INDICES):
            for h_i, h in enumerate(HORIZONS):
                row = subset[(subset["index_id"] == idx) & (subset["horizon"] == h)]
                if not row.empty:
                    mat_p[i_i, h_i]  = row["p_value"].values[0]
                    mat_bw[i_i, h_i] = bool(row["b_wins"].values[0])
                    col = "cw_stat" if "cw_stat" in row.columns and pd.notna(row["cw_stat"].values[0]) else "dm_stat"
                    mat_st[i_i, h_i] = row[col].values[0]

        # score: +1 = b wins at p=0, -1 = b loses at p=0
        score = np.where(mat_bw, 1 - mat_p, -(1 - mat_p))
        img = ax.imshow(score, cmap=cmap_div, vmin=-1, vmax=1, aspect="auto")

        # grid lines
        for x in np.arange(-0.5, 3, 1):
            ax.axvline(x, color="#FFFFFF", linewidth=0.8)
        for y in np.arange(-0.5, 3, 1):
            ax.axhline(y, color="#FFFFFF", linewidth=0.8)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"$h={h}$" for h in HORIZONS])
        ax.set_yticklabels([IDX_LABELS[i] for i in INDICES])
        ax.set_title(step_title, fontsize=9.5, pad=6)

        # cell annotations
        for i_i in range(3):
            for h_i in range(3):
                p   = mat_p[i_i, h_i]
                bw  = mat_bw[i_i, h_i]
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else "ns")))
                direction = "↑" if bw else "↓"
                txt = f"{direction} {sig}\n$p={p:.3f}$"
                intensity = abs(score[i_i, h_i])
                color = "#FFFFFF" if intensity > 0.55 else NDARK
                ax.text(h_i, i_i, txt, ha="center", va="center",
                        fontsize=8.5, color=color,
                        linespacing=1.4)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_spillover_cw_heatmap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: QLIKE gain ladder  (fig_spillover_gain_ladder.pdf)
# ══════════════════════════════════════════════════════════════════════════════
def make_gain_ladder():
    s1 = tests[tests["comparison"].str.contains("Step1")].copy()
    s2_dm = tests[tests["comparison"].str.contains("Step2: Wavelet")].copy()
    s2_cw = tests[tests["comparison"].str.contains("Step2 CW")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)

    for ax_i, idx in enumerate(INDICES):
        ax = axes[ax_i]
        gains1, gains2, pvals1, pvals2_cw, bwins1, bwins2 = [], [], [], [], [], []
        h_labels = []
        for h in HORIZONS:
            r1 = s1[(s1["index_id"] == idx) & (s1["horizon"] == h)]
            r2 = s2_dm[(s2_dm["index_id"] == idx) & (s2_dm["horizon"] == h)]
            r2c = s2_cw[(s2_cw["index_id"] == idx) & (s2_cw["horizon"] == h)]
            if r1.empty or r2.empty:
                continue
            g1 = r1["mean_loss_b"].values[0] - r1["mean_loss_a"].values[0]  # wavelet - HAR
            g2 = r2["mean_loss_b"].values[0] - r2["mean_loss_a"].values[0]  # spillover - wavelet
            gains1.append(g1); gains2.append(g2)
            pvals1.append(r1["p_value"].values[0])
            pvals2_cw.append(r2c["p_value"].values[0] if not r2c.empty else 1.0)
            bwins1.append(bool(r1["b_wins"].values[0]))
            bwins2.append(bool(r2["b_wins"].values[0]))
            h_labels.append(f"$h={h}$")

        x    = np.arange(len(h_labels))
        w    = 0.35

        # colors aligned to paper semantic palette:
        # Step-1 significant win (wavelet > HAR) → SERIES[3] amber (wavelet colour)
        # Step-1 marginal (p<0.10)               → NMID grey
        # Not significant                         → NLIGHT light grey
        # Step-2 significant win (CW, spill>wav)  → SERIES[2] teal (spillover colour)
        def bar_color(p, bw, cw=False):
            if bw and p < 0.05:
                return (SERIES[2] if cw else SERIES[3])
            if bw and p < 0.10:
                return NMID
            return NLIGHT

        c1 = [bar_color(p, bw)          for p, bw in zip(pvals1, bwins1)]
        c2 = [bar_color(p, bw, cw=True) for p, bw in zip(pvals2_cw, bwins2)]

        ax.bar(x - w/2, gains1, width=w * 0.9, color=c1,
               edgecolor=NDARK, linewidth=0.4, label="Step 1: Wavelet vs HAR (DM)")
        ax.bar(x + w/2, gains2, width=w * 0.9, color=c2,
               edgecolor=NDARK, linewidth=0.4, label="Step 2: Spillover vs Wavelet (CW)")

        ax.axhline(0, color=NMID, linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(h_labels)
        ax.set_title(IDX_LABELS[idx], fontsize=9)
        if ax_i == 0:
            ax.set_ylabel(r"QLIKE gain ($b - a$); positive = $b$ improves")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

        # significance star annotations
        def sig_str(p):
            return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else "")))

        for xi, (g1, g2, p1, p2, bw1, bw2) in enumerate(
                zip(gains1, gains2, pvals1, pvals2_cw, bwins1, bwins2)):
            s1s = sig_str(p1)
            s2s = sig_str(p2)
            ann_color1 = SERIES[3] if bw1 else NMID
            ann_color2 = SERIES[2] if bw2 else NMID
            if s1s:
                yoff = g1 + 0.005 if g1 > 0 else g1 - 0.008
                ax.text(xi - w/2, yoff, s1s, ha="center",
                        va="bottom" if g1 > 0 else "top",
                        fontsize=9, color=ann_color1)
            if s2s:
                yoff = g2 + 0.005 if g2 > 0 else g2 - 0.008
                ax.text(xi + w/2, yoff, s2s, ha="center",
                        va="bottom" if g2 > 0 else "top",
                        fontsize=9, color=ann_color2)

    # unified legend — placed below all panels to avoid data overlap
    legend_patches = [
        mpatches.Patch(facecolor=SERIES[3], edgecolor=NDARK, linewidth=0.5,
                       label="Step 1 wins (DM $p<0.05$)"),
        mpatches.Patch(facecolor=NMID,      edgecolor=NDARK, linewidth=0.5,
                       label="Step 1 marginal ($p<0.10$)"),
        mpatches.Patch(facecolor=NLIGHT,    edgecolor=NDARK, linewidth=0.5,
                       label="Not significant"),
        mpatches.Patch(facecolor=SERIES[2], edgecolor=NDARK, linewidth=0.5,
                       label="Step 2 wins (CW $p<0.05$)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=4,
               frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = os.path.join(FIG_DIR, "fig_spillover_gain_ladder.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_qlike_bars()
    make_cw_heatmap()
    make_gain_ladder()
    print("\nAll spillover figures generated successfully.")
