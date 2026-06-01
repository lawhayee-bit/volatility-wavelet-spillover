from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.figure
    from matplotlib.axes import Axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STYLE_PATH = PROJECT_ROOT / "config" / "plot_style.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_plot_style(
    category: str,
    style_path: str | Path = DEFAULT_STYLE_PATH,
) -> dict[str, Any]:
    style_path = Path(style_path)
    with style_path.open("rb") as f:
        raw = tomllib.load(f)

    global_style = raw["global"]
    category_overrides = raw.get("overrides", {})
    if category not in category_overrides:
        valid = ", ".join(sorted(category_overrides))
        raise ValueError(f"Unknown plot category '{category}'. Valid categories: {valid}")

    style = _deep_merge(global_style, category_overrides[category])
    style["meta"] = raw.get("meta", {})
    style["category"] = category
    return style


def build_matplotlib_rcparams(style: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    font_size = style["font"]["body_font_size_pt"]
    label_size = font_size * style["font"]["label_size_scale"]
    tick_size = font_size * style["font"]["tick_size_scale"]
    legend_size = font_size * style["font"]["legend_size_scale"]

    rcparams = {
        "font.family": "serif",
        "font.serif": [
            style["font"]["family"],
            *style["font"].get("fallback_families", []),
        ],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": label_size,
        "axes.facecolor": style["figure"]["facecolor"],
        "axes.edgecolor": style["axes"]["edgecolor"],
        "axes.linewidth": style["axes"]["linewidth"],
        "axes.labelcolor": style["axes"]["labelcolor"],
        "axes.labelpad": style["axes"]["labelpad"],
        "axes.titlepad": style["axes"]["titlepad"],
        "axes.grid": style["axes"]["grid_major"],
        "figure.facecolor": style["figure"]["facecolor"],
        "figure.dpi": style["figure"]["dpi"],
        "savefig.dpi": style["figure"]["save_dpi"],
        "savefig.facecolor": style["figure"]["facecolor"],
        "savefig.bbox": style["figure"]["bbox_inches"],
        "savefig.pad_inches": style["figure"]["pad_inches"],
        "legend.frameon": style["legend"]["frameon"],
        "legend.handlelength": style["legend"]["handlelength"],
        "legend.borderpad": style["legend"]["borderpad"],
        "legend.columnspacing": style["legend"]["columnspacing"],
        "legend.labelspacing": style["legend"]["labelspacing"],
        "legend.fontsize": legend_size,
        "xtick.direction": style["ticks"]["direction"],
        "ytick.direction": style["ticks"]["direction"],
        "xtick.color": style["ticks"]["color"],
        "ytick.color": style["ticks"]["color"],
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.major.width": style["ticks"]["major_width"],
        "ytick.major.width": style["ticks"]["major_width"],
        "xtick.minor.width": style["ticks"]["minor_width"],
        "ytick.minor.width": style["ticks"]["minor_width"],
        "xtick.major.size": style["ticks"]["major_size"],
        "ytick.major.size": style["ticks"]["major_size"],
        "xtick.minor.size": style["ticks"]["minor_size"],
        "ytick.minor.size": style["ticks"]["minor_size"],
        "xtick.major.pad": style["ticks"]["major_pad"],
        "ytick.major.pad": style["ticks"]["major_pad"],
        "xtick.minor.visible": style["ticks"]["minor_visible"],
        "ytick.minor.visible": style["ticks"]["minor_visible"],
        "lines.linewidth": style["lines"]["line_width"],
        "lines.markersize": style["lines"]["marker_size"],
        "lines.solid_capstyle": style["lines"]["solid_capstyle"],
        "lines.solid_joinstyle": style["lines"]["solid_joinstyle"],
        "lines.antialiased": style["lines"]["antialiased"],
        "text.color": style["text"]["color"],
        "axes.prop_cycle": plt.cycler(color=style["palette"]["series"]),
    }
    return rcparams


def apply_plot_style(style: dict[str, Any]) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(build_matplotlib_rcparams(style))


def configure_axes(ax: Axes, style: dict[str, Any]) -> Axes:
    from matplotlib.ticker import AutoMinorLocator

    ax.spines["top"].set_visible(style["axes"]["show_top_spine"])
    ax.spines["right"].set_visible(style["axes"]["show_right_spine"])
    ax.spines["bottom"].set_linewidth(style["axes"]["linewidth"])
    ax.spines["left"].set_linewidth(style["axes"]["linewidth"])
    ax.spines["top"].set_linewidth(style["axes"]["linewidth"])
    ax.spines["right"].set_linewidth(style["axes"]["linewidth"])

    if style["ticks"]["minor_visible"]:
        ax.xaxis.set_minor_locator(
            AutoMinorLocator(style["ticks"]["minor_locator_divisions_x"])
        )
        ax.yaxis.set_minor_locator(
            AutoMinorLocator(style["ticks"]["minor_locator_divisions_y"])
        )

    ax.tick_params(
        axis="both",
        which="major",
        length=style["ticks"]["major_size"],
        width=style["ticks"]["major_width"],
        direction=style["ticks"]["direction"],
        colors=style["ticks"]["color"],
        pad=style["ticks"]["major_pad"],
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=style["ticks"]["minor_size"],
        width=style["ticks"]["minor_width"],
        direction=style["ticks"]["direction"],
        colors=style["ticks"]["color"],
    )

    if not style["font"]["title_enabled"]:
        ax.set_title("")
    return ax


def create_figure(category: str, style_path: str | Path = DEFAULT_STYLE_PATH, **kwargs: Any):
    import matplotlib.pyplot as plt

    style = load_plot_style(category, style_path=style_path)
    apply_plot_style(style)

    width = kwargs.pop("fig_width_in", style.get("fig_width_in", 6.0))
    height = kwargs.pop("fig_height_in", style.get("fig_height_in", 3.0))
    fig, ax = plt.subplots(
        figsize=(width, height),
        constrained_layout=style["figure"]["constrained_layout"],
        **kwargs,
    )
    configure_axes(ax, style)
    return fig, ax, style


def save_figure(
    fig: matplotlib.figure.Figure,
    output_path: str | Path,
    style: dict[str, Any],
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=style["figure"]["save_dpi"],
        bbox_inches=style["figure"]["bbox_inches"],
        pad_inches=style["figure"]["pad_inches"],
        transparent=style["export"]["transparent"],
        metadata={
            "Creator": style["export"]["metadata_creator"],
            "Subject": style["export"]["metadata_description"],
        },
    )
    return output_path
