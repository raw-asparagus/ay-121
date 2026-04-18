"""Lab 4 – reusable plotting functions for 21 cm HI survey data.

All functions use the style constants from ``plotting.py`` and return
axes for interactive use.  Notebooks import individual functions::

    from plotters import plot_hi_spectrum, plot_survey_mollweide
"""

from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from plotting import (
    TEXTWIDTH_IN,
    COLUMNWIDTH_IN,
    A4_USABLE_HEIGHT_IN,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    EMPHASIS_SIZE,
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
    FIT_STYLE,
    SCATTER_STYLE,
    FILL_STYLE,
    textwidth_figure,
    columnwidth_figure,
    landscapewidth_figure,
    zero_line,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HI_REST_MHZ = 1420.405
C_KMS = 299792.458


# ---------------------------------------------------------------------------
# Single HI spectrum
# ---------------------------------------------------------------------------

def plot_hi_spectrum(
    v_kms: np.ndarray,
    R: np.ndarray,
    *,
    title: str | None = None,
    ylabel: str = r"$R = (I_1 - I_2) / I_2$",
    xlabel: str = r"Velocity [km\,s$^{-1}$]",
    color: str = "C0",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a single frequency-switched HI profile."""
    if ax is None:
        _, ax = columnwidth_figure(4)
    ax.plot(v_kms, R, lw=LW_FINE, color=color, alpha=ALPHA_STANDARD)
    zero_line(ax)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TICK_SIZE)
    return ax


# ---------------------------------------------------------------------------
# Spectrum grid (all pointings for one DR)
# ---------------------------------------------------------------------------

def plot_spectra_grid(
    v_kms: np.ndarray,
    spectra: dict,
    *,
    ncols: int = 5,
    title: str = "",
    color: str = "C0",
) -> plt.Figure:
    """Plot all spectra in a compact grid.

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
    from plotting import A4_USABLE_HEIGHT_IN as PAGE_H

    keys = sorted(spectra.keys(), key=lambda k: (k[1], k[0]))
    n = len(keys)

    # Scale: fit as many rows as possible per A4 page height
    ROW_HEIGHT_IN = 0.9
    rows_per_page = max(1, int(PAGE_H / ROW_HEIGHT_IN))
    nrows_total = math.ceil(n / ncols)

    # Build list of figures (one per page)
    figs = []
    for page_start_row in range(0, nrows_total, rows_per_page):
        page_end_row = min(page_start_row + rows_per_page, nrows_total)
        page_nrows = page_end_row - page_start_row
        page_start_idx = page_start_row * ncols
        page_end_idx = min(page_end_row * ncols, n)

        fig, axes = plt.subplots(
            page_nrows, ncols,
            figsize=(TEXTWIDTH_IN, ROW_HEIGHT_IN * page_nrows),
            gridspec_kw={"hspace": 0.35, "wspace": 0.3},
        )
        axes = np.atleast_2d(axes)

        for i_in_page, i_global in enumerate(range(page_start_idx, page_end_idx)):
            gl, gb = keys[i_global]
            ri, ci = divmod(i_in_page, ncols)
            ax = axes[ri, ci]
            ax.plot(v_kms, spectra[(gl, gb)], lw=LW_FINE * 0.7, color=color,
                    alpha=ALPHA_STANDARD)
            ax.axhline(0, **{**GUIDE_STYLE, "lw": LW_FINE * 0.5})
            ax.set_title(
                rf"$\ell$={gl} $b$={gb:+d}",
                fontsize=TICK_SIZE - 3, pad=2,
            )
            ax.tick_params(labelsize=TICK_SIZE - 4)
            if ri < page_nrows - 1:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(r"$R$", fontsize=TICK_SIZE - 3)

        # Hide unused axes on last page
        n_on_page = page_end_idx - page_start_idx
        for i_in_page in range(n_on_page, page_nrows * ncols):
            ri, ci = divmod(i_in_page, ncols)
            axes[ri, ci].set_visible(False)

        # Bottom labels
        for ci in range(min(ncols, n_on_page)):
            axes[-1, ci].set_xlabel(r"$v$ [km\,s$^{-1}$]", fontsize=TICK_SIZE - 3)

        page_label = title
        if nrows_total > rows_per_page:
            page_num = page_start_row // rows_per_page + 1
            total_pages = math.ceil(nrows_total / rows_per_page)
            page_label = f"{title} ({page_num}/{total_pages})"
        if page_label:
            fig.suptitle(page_label, fontsize=EMPHASIS_SIZE)
        fig.tight_layout()
        figs.append(fig)

    return figs[0] if len(figs) == 1 else figs


# ---------------------------------------------------------------------------
# Flat heatmap (per-DR)
# ---------------------------------------------------------------------------

def plot_heatmap(
    gl: np.ndarray,
    gb: np.ndarray,
    vals: np.ndarray,
    *,
    title: str = "",
    cbar_label: str = r"$W = \Sigma R \cdot \Delta v$ [km\,s$^{-1}$]",
    cmap: str = "inferno",
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
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Mollweide all-sky map
# ---------------------------------------------------------------------------

def plot_survey_mollweide(
    gl: np.ndarray,
    gb: np.ndarray,
    vals: np.ndarray,
    *,
    center_l: float = 120.0,
    title: str = "",
    cbar_label: str = r"$W = \Sigma R \cdot \Delta v$ [km\,s$^{-1}$]",
    cmap: str = "inferno",
    marker_size: float = SS_FINE,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot survey data on a Mollweide projection in galactic coords.

    Uses landscape width (A4 usable height) for maximum sky coverage.
    """
    gl_shifted = gl - center_l
    gl_shifted = np.where(gl_shifted > 180, gl_shifted - 360, gl_shifted)
    gl_shifted = np.where(gl_shifted < -180, gl_shifted + 360, gl_shifted)

    # landscapewidth_figure(5) → width = A4_USABLE_HEIGHT_IN, height = 5/10 * width
    fig, _ = landscapewidth_figure(5)
    plt.close(fig)
    fig = plt.figure(figsize=(A4_USABLE_HEIGHT_IN, A4_USABLE_HEIGHT_IN / 10 * 5))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, **{k: v for k, v in GRID_STYLE.items() if k != "color"},
            color=NEUTRAL_COLOR)

    # Galactic plane
    l_line = np.linspace(-np.pi, np.pi, 500)
    ax.plot(l_line, np.zeros_like(l_line), lw=LW_FINE,
            color="k", alpha=ALPHA_FAINT)

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
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Alt/Az Mollweide
# ---------------------------------------------------------------------------

def plot_altaz_mollweide(
    az: np.ndarray,
    alt: np.ndarray,
    bad: np.ndarray,
    *,
    min_alt: float = 17.0,
    az_min: float = 7.0,
    az_max: float = 348.0,
    title: str = r"Alt/Az at observation time",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot telescope pointings on an alt/az Mollweide.

    Uses textwidth for inline display.
    """
    az_moll = np.where(az > 180, az - 360, az)
    good = ~bad

    fig, _ = textwidth_figure(8)
    plt.close(fig)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / 16 * 8))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, **{k: v for k, v in GRID_STYLE.items() if k != "color"},
            color=NEUTRAL_COLOR)

    # Horizon line
    az_line = np.linspace(-np.pi, np.pi, 500)
    ax.plot(az_line, np.full_like(az_line, np.deg2rad(min_alt)),
            lw=LW_LIGHT, ls="--", color="C3", alpha=ALPHA_LIGHT,
            label=rf"min alt = {min_alt}$^\circ$")

    # Az exclusion
    for az_lim in [az_min, az_max]:
        az_l = az_lim if az_lim <= 180 else az_lim - 360
        ax.axvline(np.deg2rad(az_l), lw=LW_LIGHT, ls="--",
                   color="C1", alpha=ALPHA_LIGHT)

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
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Scan pattern (galactic flat)
# ---------------------------------------------------------------------------

def plot_scan_pattern(
    sim: list[dict],
    *,
    hpbw_deg: float = 3.4,
    title: str = "Scan pattern (galactic)",
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
        )
        ax.add_patch(beam)
        ax.plot(s["l"], s["b"], "o", color=color, ms=MS_MICRO)

    ax.set_xlabel(r"$\ell$ [deg]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$b$ [deg]", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=EMPHASIS_SIZE)
    ax.set_aspect("equal")
    ax.grid(True, **GRID_STYLE)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Timeline (alt + az vs cell index)
# ---------------------------------------------------------------------------

def plot_timeline(
    sim: list[dict],
    *,
    min_alt: float = 17.0,
    az_min: float = 7.0,
    az_max: float = 348.0,
    title: str = "Telescope pointing vs scan progress",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot alt and az vs cell index.

    Uses textwidth for inline display.
    """
    from plotting import subpanels

    idxs = [s["idx"] for s in sim]
    alts = [s["alt"] for s in sim]
    azs = [s["az"] for s in sim]

    fig, _ = textwidth_figure(6)
    plt.close(fig)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN / 16 * 6))
    ax1, ax2 = subpanels(fig, 2, height_ratios=(1, 1), hspace=0.1, sharex=True)

    ax1.plot(idxs, alts, "o-", ms=MS_MICRO, lw=LW_FINE, color="C0")
    ax1.axhline(min_alt, lw=LW_LIGHT, ls="--", color="C3", alpha=ALPHA_LIGHT)
    ax1.set_ylabel(r"Alt [deg]", fontsize=LABEL_SIZE)
    ax1.grid(True, **GRID_STYLE)

    ax2.plot(idxs, azs, "o-", ms=MS_MICRO, lw=LW_FINE, color="C1")
    ax2.axhline(az_min, lw=LW_LIGHT, ls="--", color="C1", alpha=ALPHA_LIGHT)
    ax2.axhline(az_max, lw=LW_LIGHT, ls="--", color="C1", alpha=ALPHA_LIGHT)
    ax2.set_ylabel(r"Az [deg]", fontsize=LABEL_SIZE)
    ax2.set_xlabel("Cell index", fontsize=LABEL_SIZE)
    ax2.grid(True, **GRID_STYLE)

    fig.suptitle(title, fontsize=EMPHASIS_SIZE)
    fig.tight_layout()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Calibrated HI profile (antenna temperature)
# ---------------------------------------------------------------------------

def plot_calibrated_profiles(
    profiles: list[dict],
    *,
    v_lim: tuple[float, float] | None = None,
    f_lim: tuple[float, float] | None = None,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot calibrated T_A profiles for multiple chips.

    Uses textwidth for inline display.

    Parameters
    ----------
    profiles : list of dict
        Each dict has keys ``v_kms``, ``T_A``, ``label``, ``color``.
    v_lim : tuple, optional
        Velocity axis limits.
    f_lim : tuple, optional
        Frequency axis limits for top axis.
    """
    fig, ax = textwidth_figure(5)

    for p in profiles:
        ax.plot(p["v_kms"], p["T_A"], lw=LW_FINE,
                color=p.get("color", "C0"),
                alpha=ALPHA_STANDARD,
                label=p.get("label", ""))

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

    fig.tight_layout()
    return fig, ax
