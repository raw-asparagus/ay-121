"""Lab 4 - reusable plotting functions for 21 cm HI survey data.

All functions use the style constants from ``ugradiolab.plotting`` and return
axes for interactive use.  Notebooks import individual functions::

    from plotters import plot_hi_spectrum, plot_survey_mollweide
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import astropy.coordinates as ac
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ugradiolab.plotting import (
    TEXTWIDTH_IN,
    COLUMNWIDTH_IN,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    EMPHASIS_SIZE,
    LW_NONE,
    LW_FINE,
    LW_LIGHT,
    LW_STANDARD,
    MS_MICRO,
    MS_FINE,
    SS_FINE,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    GRID_STYLE,
    GUIDE_STYLE,
    ERRORBAR_STYLE,
    FIT_STYLE,
    SCATTER_STYLE,
    FILL_STYLE,
    textwidth_figure,
    columnwidth_figure,
    landscapewidth_figure,
    subpanels,
    zero_line,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HI_REST_MHZ = 1420.405
C_KMS = 299792.458

MOLL_CENTER_L = 120.0
LEO_LAT_DEG = 37.9183
MIN_ALT_DEG = 17.0
MAX_ALT_DEG = 83.0
AZ_MIN_DEG = 7.0
AZ_MAX_DEG = 348.0

_LAB04_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _LAB04_DIR / "report" / "figures"


def savefig(fig: plt.Figure, name: str) -> None:
    """Save *fig* as a PDF in ``report/figures/``."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)
    print(f"  {name}")


# ---------------------------------------------------------------------------
# Single HI spectrum
# ---------------------------------------------------------------------------

def plot_hi_spectrum(
    v_kms: np.ndarray,
    R: np.ndarray,
    *,
    title: str | None,
    ylabel: str,
    xlabel: str,
    color: str,
    ax: plt.Axes | None,
) -> plt.Axes:
    """Plot a single frequency-switched HI profile."""
    if ax is None:
        _, ax = columnwidth_figure(4)
    ax.plot(v_kms, R, lw=LW_FINE, color=color, alpha=ALPHA_STANDARD, zorder=2)
    zero_line(ax)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TICK_SIZE)
    return ax


# ---------------------------------------------------------------------------
# Mollweide accessibility overlay
# ---------------------------------------------------------------------------

def add_never_observable_overlay(
    ax: plt.Axes,
    *,
    center_l: float = MOLL_CENTER_L,
    latitude_deg: float = LEO_LAT_DEG,
    min_alt_deg: float = MIN_ALT_DEG,
    max_alt_deg: float = MAX_ALT_DEG,
    az_min_deg: float = AZ_MIN_DEG,
    az_max_deg: float = AZ_MAX_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the never-observable galactic region on a Mollweide axis."""
    l_dense = np.arange(-179.5, 180.5, 1.0)
    b_dense = np.arange(-89.5, 90.5, 1.0)
    L_dense, B_dense = np.meshgrid(l_dense, b_dense)

    L_true = (L_dense + center_l) % 360
    gc_grid = ac.SkyCoord(
        l=L_true.ravel() * u.deg,
        b=B_dense.ravel() * u.deg,
        frame="galactic",
    )
    dec_rad = np.deg2rad(gc_grid.transform_to(ac.ICRS()).dec.deg).reshape(L_dense.shape)
    lat_rad = np.deg2rad(latitude_deg)

    ha_steps = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    alt_lo_r = np.deg2rad(min_alt_deg)
    alt_hi_r = np.deg2rad(max_alt_deg)
    az_lo_r = np.deg2rad(az_min_deg)
    az_hi_r = np.deg2rad(az_max_deg)

    observable = np.zeros(L_dense.shape, dtype=bool)
    for ha in ha_steps:
        sin_alt = (
            np.sin(lat_rad) * np.sin(dec_rad)
            + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha)
        )
        alt = np.arcsin(np.clip(sin_alt, -1, 1))
        cos_az_num = np.sin(dec_rad) - np.sin(lat_rad) * sin_alt
        cos_az_den = np.cos(lat_rad) * np.cos(alt)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_az = np.clip(cos_az_num / cos_az_den, -1, 1)
        az = np.arccos(cos_az)
        az = np.where(np.sin(ha) > 0, 2 * np.pi - az, az)
        observable |= (
            (alt >= alt_lo_r)
            & (alt <= alt_hi_r)
            & (az >= az_lo_r)
            & (az <= az_hi_r)
        )

    inacc_mask = ~observable

    with mpl.rc_context({"hatch.color": "red", "hatch.linewidth": 0.5}):
        ax.contourf(
            np.deg2rad(L_dense),
            np.deg2rad(B_dense),
            inacc_mask.astype(float),
            levels=[0.5, 1.5],
            colors=["red"],
            alpha=0.08,
            hatches=["///"],
            zorder=0,
        )
        ax.contour(
            np.deg2rad(L_dense),
            np.deg2rad(B_dense),
            inacc_mask.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=0.6,
            alpha=0.4,
            zorder=1,
        )

    return L_dense, B_dense, inacc_mask


# ---------------------------------------------------------------------------
# Spectrum grid (all pointings for one DR)
# ---------------------------------------------------------------------------

def plot_spectra_grid(
    v_kms: np.ndarray,
    spectra: dict,
    *,
    ncols: int,
    title: str,
    color: str,
) -> plt.Figure:
    """Plot all spectra for one DR on a single page.

    Parameters
    ----------
    v_kms : array
        Velocity axis (shared).
    spectra : dict
        Mapping ``(l, b) -> R_overlap`` array.
    ncols : int
        Number of columns.
    title : str
        Figure suptitle.

    Returns
    -------
    fig : Figure
    """
    keys = sorted(spectra.keys(), key=lambda k: (k[1], k[0]))
    n = len(keys)

    if n == 0:
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.0))
        ax.set_axis_off()
        if title:
            fig.suptitle(title, fontsize=EMPHASIS_SIZE)
        return fig

    ROW_HEIGHT_IN = 0.9
    nrows_total = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows_total, ncols,
        figsize=(TEXTWIDTH_IN, ROW_HEIGHT_IN * nrows_total),
        gridspec_kw={"hspace": 0.35, "wspace": 0.3},
    )
    axes = np.atleast_2d(axes)

    for i_global, (gl, gb) in enumerate(keys):
        ri, ci = divmod(i_global, ncols)
        ax = axes[ri, ci]
        ax.plot(v_kms, spectra[(gl, gb)], lw=LW_FINE, color=color,
                alpha=ALPHA_STANDARD, zorder=2)
        ax.axhline(0, **GUIDE_STYLE, zorder=1)
        ax.set_title(
            rf"$\ell$={gl} $b$={gb:+d}",
            fontsize=TICK_SIZE - 3, pad=2,
        )
        ax.tick_params(labelsize=TICK_SIZE - 4)
        if ri < nrows_total - 1:
            ax.set_xticklabels([])
        if ci == 0:
            ax.set_ylabel(r"$R$", fontsize=TICK_SIZE - 3)

    # Hide unused axes on the last row.
    n_on_last_row = n % ncols or ncols
    for ci in range(n_on_last_row, ncols):
        axes[-1, ci].set_visible(False)

    # Bottom labels.
    for ci in range(n_on_last_row):
        axes[-1, ci].set_xlabel(r"$v$ [km\,s$^{-1}$]", fontsize=TICK_SIZE - 3)

    if title:
        fig.suptitle(title, fontsize=EMPHASIS_SIZE)
    fig.subplots_adjust(top=0.92)
    return fig


# ---------------------------------------------------------------------------
# Flat heatmap (per-DR)
# ---------------------------------------------------------------------------

def plot_heatmap(
    gl: np.ndarray,
    gb: np.ndarray,
    vals: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a flat (l, b) heatmap from scattered data."""
    gl_int = np.round(gl).astype(int)
    gb_int = np.round(gb).astype(int)
    l_unique = np.arange(gl_int.min(), gl_int.max() + 1)
    b_unique = np.arange(gb_int.min(), gb_int.max() + 1)
    grid = np.full((len(b_unique), len(l_unique)), np.nan)
    for i in range(len(vals)):
        li = np.searchsorted(l_unique, gl_int[i])
        bi = np.searchsorted(b_unique, gb_int[i])
        if 0 <= li < len(l_unique) and 0 <= bi < len(b_unique):
            grid[bi, li] = vals[i]

    fig, ax = textwidth_figure(5)
    im = ax.imshow(
        grid, origin="lower", aspect="equal",
        extent=[l_unique[0] - 0.5, l_unique[-1] + 0.5,
                b_unique[0] - 0.5, b_unique[-1] + 0.5],
        cmap=cmap,
    )
    ax.set_xlabel(r"Galactic longitude $\ell$ [deg]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Galactic latitude $b$ [deg]", fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=EMPHASIS_SIZE)
    plt.colorbar(im, ax=ax, label=cbar_label)
    return fig, ax


# ---------------------------------------------------------------------------
# Mollweide all-sky map
# ---------------------------------------------------------------------------

def plot_survey_mollweide(
    gl: np.ndarray,
    gb: np.ndarray,
    vals: np.ndarray,
    *,
    center_l: float,
    title: str,
    cbar_label: str,
    cmap: str,
    marker_size: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot survey data on a Mollweide projection in galactic coords.

    Uses landscape width (A4 usable height) for maximum sky coverage.
    Includes overlay showing never-observable regions.
    """
    gl_shifted = gl - center_l
    gl_shifted = np.where(gl_shifted > 180, gl_shifted - 360, gl_shifted)
    gl_shifted = np.where(gl_shifted < -180, gl_shifted + 360, gl_shifted)

    fig, _ax = landscapewidth_figure(5)
    _ax.remove()
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, **{k: v for k, v in GRID_STYLE.items() if k != "color"},
            color=NEUTRAL_COLOR)

    # Galactic plane
    l_line = np.linspace(-np.pi, np.pi, 500)
    ax.plot(l_line, np.zeros_like(l_line), lw=LW_FINE,
            color="k", alpha=ALPHA_FAINT, zorder=1)

    # Never-observable overlay
    add_never_observable_overlay(ax, center_l=center_l, latitude_deg=LEO_LAT_DEG,
                                 min_alt_deg=MIN_ALT_DEG, max_alt_deg=MAX_ALT_DEG,
                                 az_min_deg=AZ_MIN_DEG, az_max_deg=AZ_MAX_DEG)

    sc = ax.scatter(
        np.deg2rad(gl_shifted), np.deg2rad(gb),
        c=vals, cmap=cmap, s=marker_size,
        edgecolors="none", zorder=5,
    )
    plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6)

    # Tick labels
    tick_locs = np.arange(-150, 180, 30)
    tick_labels = [rf"{int((t + center_l) % 360)}$^\circ$" for t in tick_locs]
    ax.set_xticks(np.deg2rad(tick_locs))
    ax.set_xticklabels(tick_labels, fontsize=TICK_SIZE - 3)

    if title:
        ax.set_title(title, fontsize=EMPHASIS_SIZE)
    return fig, ax


# ---------------------------------------------------------------------------
# Alt/Az Mollweide
# ---------------------------------------------------------------------------

def plot_altaz_mollweide(
    az: np.ndarray,
    alt: np.ndarray,
    bad: np.ndarray,
    *,
    min_alt: float,
    az_min: float,
    az_max: float,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot telescope pointings on an alt/az Mollweide.

    Uses textwidth for inline display.
    """
    az_moll = np.where(az > 180, az - 360, az)
    good = ~bad

    fig, _ax = textwidth_figure(8)
    _ax.remove()
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, **{k: v for k, v in GRID_STYLE.items() if k != "color"},
            color=NEUTRAL_COLOR)

    # Horizon line
    az_line = np.linspace(-np.pi, np.pi, 500)
    ax.plot(az_line, np.full_like(az_line, np.deg2rad(min_alt)),
            lw=LW_LIGHT, ls="--", color="C3", alpha=ALPHA_LIGHT,
            label=rf"min alt = {min_alt}$^\circ$", zorder=1)

    # Az exclusion
    for az_lim in [az_min, az_max]:
        az_l = az_lim if az_lim <= 180 else az_lim - 360
        ax.axvline(np.deg2rad(az_l), lw=LW_LIGHT, ls="--",
                   color="C1", alpha=ALPHA_LIGHT, zorder=1)

    ax.scatter(np.deg2rad(az_moll[good]), np.deg2rad(alt[good]),
               **SCATTER_STYLE, color="C0", zorder=5, label="OK")
    if bad.any():
        ax.scatter(np.deg2rad(az_moll[bad]), np.deg2rad(alt[bad]),
                   s=SS_FINE * 2, color="C3", marker="x",
                   zorder=6, label="out of limits")

    for azd, name in [(0, "N"), (90, "E"), (180, "S"), (-90, "W")]:
        ax.annotate(name, (np.deg2rad(azd), np.deg2rad(-2)),
                    fontsize=TICK_SIZE, ha="center", va="top",
                    fontweight="bold", color=NEUTRAL_COLOR)

    ax.set_title(title, fontsize=EMPHASIS_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="lower left")
    return fig, ax


# ---------------------------------------------------------------------------
# Scan pattern (galactic flat)
# ---------------------------------------------------------------------------

def plot_scan_pattern(
    sim: list[dict],
    *,
    hpbw_deg: float,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot planned scan pointings with beam circles.

    Uses textwidth for inline display.
    """
    from matplotlib.patches import Circle

    fig, ax = textwidth_figure(6)

    for s in sim:
        color = "C3" if s["bad"] else "C0"
        beam = Circle(
            (s["l"], s["b"]), hpbw_deg / 2,
            fill=False, edgecolor=color,
            alpha=ALPHA_EXTRA_LIGHT, lw=LW_FINE,
            zorder=1,
        )
        ax.add_patch(beam)
        ax.plot(s["l"], s["b"], "o", color=color, ms=MS_MICRO, zorder=2)

    ax.set_xlabel(r"$\ell$ [deg]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$b$ [deg]", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=EMPHASIS_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, **GRID_STYLE)
    return fig, ax


# ---------------------------------------------------------------------------
# Timeline (alt + az vs cell index)
# ---------------------------------------------------------------------------

def plot_timeline(
    sim: list[dict],
    *,
    min_alt: float,
    az_min: float,
    az_max: float,
    title: str,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot alt and az vs cell index.

    Uses textwidth for inline display.
    """
    idxs = [s["idx"] for s in sim]
    alts = [s["alt"] for s in sim]
    azs = [s["az"] for s in sim]

    fig, _ax = textwidth_figure(6)
    _ax.remove()
    ax1, ax2 = subpanels(fig, 2, height_ratios=(1, 1), hspace=0.1)

    ax1.plot(idxs, alts, "o-", ms=MS_MICRO, lw=LW_FINE, color="C0", zorder=2)
    ax1.axhline(min_alt, lw=LW_LIGHT, ls="--", color="C3", alpha=ALPHA_LIGHT, zorder=1)
    ax1.set_ylabel(r"Alt [deg]", fontsize=LABEL_SIZE)
    ax1.grid(True, **GRID_STYLE)

    ax2.plot(idxs, azs, "o-", ms=MS_MICRO, lw=LW_FINE, color="C1", zorder=2)
    ax2.axhline(az_min, lw=LW_LIGHT, ls="--", color="C1", alpha=ALPHA_LIGHT, zorder=1)
    ax2.axhline(az_max, lw=LW_LIGHT, ls="--", color="C1", alpha=ALPHA_LIGHT, zorder=1)
    ax2.set_ylabel(r"Az [deg]", fontsize=LABEL_SIZE)
    ax2.set_xlabel("Cell index", fontsize=LABEL_SIZE)
    ax2.grid(True, **GRID_STYLE)

    fig.suptitle(title, fontsize=EMPHASIS_SIZE)
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Calibrated HI profile (antenna temperature)
# ---------------------------------------------------------------------------

def plot_calibrated_profiles(
    profiles: list[dict],
    *,
    v_lim: tuple[float, float] | None,
    f_lim: tuple[float, float] | None,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot calibrated T_A profiles for multiple chips.

    Uses textwidth for inline display.

    Parameters
    ----------
    profiles : list of dict
        Each dict has keys ``v_kms``, ``T_A``, ``label``, ``color``.
    v_lim : tuple or None
        Velocity axis limits.
    f_lim : tuple or None
        Frequency axis limits for top axis.
    title : str
        Figure title.
    """
    fig, ax = textwidth_figure(5)

    for p in profiles:
        ax.plot(p["v_kms"], p["T_A"], lw=LW_FINE,
                color=p.get("color", "C0"),
                alpha=ALPHA_STANDARD,
                label=p.get("label", ""),
                zorder=2)

    zero_line(ax)
    ax.set_xlabel(r"Velocity [km\,s$^{-1}$]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Antenna temperature [K]", fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=EMPHASIS_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)

    if v_lim:
        ax.set_xlim(v_lim)

    if f_lim:
        ax_freq = ax.twiny()
        ax_freq.set_xlim(f_lim)
        ax_freq.set_xlabel(r"Frequency [MHz]", fontsize=LABEL_SIZE)
        ax_freq.ticklabel_format(axis="x", useOffset=False)

    return fig, ax
