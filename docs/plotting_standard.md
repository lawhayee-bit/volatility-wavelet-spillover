# Plotting Standard

## Purpose

This repository uses a single fixed plotting input file:

- `config/plot_style.toml`

All plotting scripts must:

1. load the global settings from `plot_style.toml`
2. apply exactly one category override
3. only change narrowly scoped local parameters when the figure class requires it

This keeps every figure consistent in font, density, line quality, palette, and export quality.

## Non-negotiable rules

1. Use `Times New Roman` as the primary font.
2. Match the figure font size to the manuscript body font size.
3. Do not place titles inside figures.
4. Use thicker axis lines than the matplotlib default.
5. Keep major ticks longer than minor ticks.
6. Keep minor ticks visible and denser than major ticks.
7. Use one shared palette for the whole paper.
8. Export vector figures as `PDF` and raster figures at `600 dpi`.
9. Do not let each script redefine the global palette, font family, or export resolution.

## What scripts may override

Scripts may adjust:

- figure width and height
- line width within the category range
- marker choice
- confidence band alpha
- bar spacing
- heatmap line width and colorbar shrink

Scripts may not override:

- font family
- body font size
- title visibility
- global palette
- export dpi
- axis line width baseline

## Suggested figure goals

1. Pipeline and method diagrams:
   no title, clean labels, strong outlines, minimal color usage
2. Time-series forecast plots:
   highlight real vs predicted, keep crisis windows readable
3. Main result bars:
   compare horizons and models using the same palette order
4. Warning-performance plots:
   PR curves, calibration curves, and threshold behavior
5. Interpretability plots:
   scale-wise importance summaries and ablation effects
6. Efficiency plots:
   runtime and feature-cost comparisons with clean log-scale axes when needed

## Recommended loader pattern

Use one helper such as:

```python
from plotting import create_figure, save_figure

fig, ax, style = create_figure("timeseries")
# draw the figure here
save_figure(fig, "outputs/figures/main_result.pdf", style)
```

The helper should:

1. read `global.*`
2. merge one `overrides.<category>` block
3. push the merged settings into plotting defaults

Current helper module:

- `plotting/style.py`

## Category names

Supported categories in the current file:

- `timeseries`
- `bar`
- `heatmap`
- `roc_pr`
- `ablation`
- `efficiency`

## Maintenance rule

If the manuscript body font size changes, update only:

- `global.font.body_font_size_pt`

All scripts should inherit the new value automatically.
