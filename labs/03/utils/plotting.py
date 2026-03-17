"""Lab 03 plotting presets — visual constants and rcParams only."""
import matplotlib as mpl

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10

LW_NONE      = 0.0
LW_GRID      = 0.4
LW_FINE      = 0.6
LW_GUIDE     = 0.8
LW_LIGHT     = 0.9
LW_STANDARD  = 1.0
LW_MEDIUM    = 1.1
LW_STRONG    = 1.3
LW_FIT       = 1.5
LW_MODEL     = 1.6
LW_EMPHASIS  = 1.8
LW_CALLOUT   = 2.2
LW_LEVEL     = 2.6

SCATTER_S_FINE      = 10
SCATTER_S_STANDARD  = 20
SCATTER_S_EMPHASIS  = 30
SCATTER_S_CALLOUT   = 60

MARKER_MS_FINE     = 1.6
MARKER_MS_STANDARD = 8
MARKER_MS_MEDIUM   = 10.5
MARKER_MS_LARGE    = 16

PRIMARY_COLOR      = "C0"
SECONDARY_COLOR    = "C1"
TERTIARY_COLOR     = "C2"
QUATERNARY_COLOR   = "C3"
QUINARY_COLOR      = "C4"
SENARY_COLOR       = "C5"
SEPTENARY_COLOR    = "C6"
NEUTRAL_COLOR      = "C7"
LIGHT_NEUTRAL_COLOR = "C8"
NONARY_COLOR       = "C9"
COMPONENT_COLORS   = (QUINARY_COLOR, SENARY_COLOR, SEPTENARY_COLOR, LIGHT_NEUTRAL_COLOR, NONARY_COLOR)

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
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)
