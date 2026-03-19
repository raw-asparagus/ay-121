"""Lab 03 plotting presets and shared plotting helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10

LW_NONE = 0.0
LW_GRID = 0.4
LW_FINE = 0.6
LW_GUIDE = 0.8
LW_LIGHT = 0.9
LW_STANDARD = 1.0
LW_MEDIUM = 1.1
LW_STRONG = 1.3
LW_FIT = 1.5
LW_MODEL = 1.6
LW_EMPHASIS = 1.8
LW_CALLOUT = 2.2
LW_LEVEL = 2.6
LW_HAIRLINE = 0.5

SCATTER_S_FINE = 10
SCATTER_S_STANDARD = 20
SCATTER_S_EMPHASIS = 30
SCATTER_S_CALLOUT = 60

MARKER_MS_FINE = 1.6
MARKER_MS_SMALL = 5
MARKER_MS_STANDARD = 8
MARKER_MS_MEDIUM = 10.5
MARKER_MS_LARGE = 16

PRIMARY_COLOR = "C0"
SECONDARY_COLOR = "C1"
TERTIARY_COLOR = "C2"
QUATERNARY_COLOR = "C3"
QUINARY_COLOR = "C4"
SENARY_COLOR = "C5"
SEPTENARY_COLOR = "C6"
NEUTRAL_COLOR = "C7"
LIGHT_NEUTRAL_COLOR = "C8"
NONARY_COLOR = "C9"
COMPONENT_COLORS = (
    QUINARY_COLOR,
    SENARY_COLOR,
    SEPTENARY_COLOR,
    LIGHT_NEUTRAL_COLOR,
    NONARY_COLOR,
)

ALPHA_SPAN_LIGHT = 0.07
ALPHA_SHADE_LIGHT = 0.12
ALPHA_SHADE_STANDARD = 0.15
ALPHA_FILL = 0.20
ALPHA_BAR = 0.80
ERRORBAR_CAPSIZE_SMALL = 3
BAR_HEIGHT_STANDARD = 0.8
WATERFALL_HA_MIN_PER_IN = 12.0
WATERFALL_PANEL_HEIGHT_IN = 2.4

mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.grid": True,
        "axes.titlesize": EMPHASIS_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "grid.linewidth": LW_GRID,
        "grid.alpha": 0.5,
        "legend.fontsize": LEGEND_SIZE,
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _tight_layout(fig: Figure) -> None:
    fig.tight_layout()


def _single_panel(figsize: tuple[float, float]):
    return plt.subplots(figsize=figsize)


def _textwidth_figsize(height_out_of_8: float) -> tuple[float, float]:
    return (TEXTWIDTH_IN, height_out_of_8 / 8 * TEXTWIDTH_IN)


def _columnwidth_figsize(height_out_of_3_5: float) -> tuple[float, float]:
    return (COLUMNWIDTH_IN, height_out_of_3_5 / 3.5 * COLUMNWIDTH_IN)


def _stacked_panels(
    nrows: int,
    figsize: tuple[float, float],
    height_ratios: list[int] | tuple[int, ...],
    hspace: float,
):
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": hspace},
    )
    if hspace == 0.0:
        fig.subplots_adjust(hspace=0.0)
    return fig, axes


def _zero_line(ax) -> None:
    ax.axhline(0.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--")


def _unity_line(ax, label: str) -> None:
    ax.axhline(1.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--", alpha=0.7, label=label)


def _reference_vline(ax, x: float, label: str, color: str, lw: float, ls: str, alpha: float) -> None:
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, label=label)
