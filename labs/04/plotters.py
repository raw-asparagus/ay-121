"""Lab 4 - reusable plotting functions for 21 cm HI survey data.

All functions use the style constants from ``ugradiolab.plotting`` and return
axes for interactive use.  Notebooks import individual functions::

    from plotters import plot_hi_spectrum, plot_survey_mollweide
"""

from __future__ import annotations

import math
from pathlib import Path

import astropy.coordinates as ac
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ugradiolab.plotting import (
    TEXTWIDTH_IN,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    EMPHASIS_SIZE,
    LW_FINE,
    LW_LIGHT,
    MS_MICRO,
    SS_FINE,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    GRID_STYLE,
    GUIDE_STYLE,
    SCATTER_STYLE,
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
# Beam-outline polygons for Mollweide overlays
# ---------------------------------------------------------------------------

def beam_outline_polys(
    center_lon_deg: np.ndarray,
    center_lat_deg: np.ndarray,
    hpbw_deg: float,
    *,
    n_vertices: int = 25,
    lat_clamp_deg: float = 78.0,
    seam_lon_deg: float = 180.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build small-circle beam outlines on a cylindrical (lon, lat) grid.

    Returns ``(polys, src_idx)`` where each entry of ``polys`` is an
    ``(n, 2)`` array of (lon, lat) vertices in radians, suitable for
    ``PolyCollection``, and ``src_idx[i]`` is the index of the input
    pointing that produced ``polys[i]``.  A pointing whose outline
    straddles the +/-``seam_lon_deg`` meridian yields two polygons (one
    on each side of the seam) with the same ``src_idx``; callers can use
    ``src_idx`` to fan out per-pointing colours.

    The 1/cos(lat) longitude correction is clamped at ``lat_clamp_deg``
    so beams pointed near the pole/zenith do not blow up across the
    projection.
    """
    cl = np.atleast_1d(np.asarray(center_lon_deg, dtype=float))
    cb = np.atleast_1d(np.asarray(center_lat_deg, dtype=float))
    phi = np.linspace(0.0, 2.0 * np.pi, n_vertices)
    r = hpbw_deg / 2.0
    polys: list[np.ndarray] = []
    src_idx: list[int] = []
    for i, (lon0, lat0) in enumerate(zip(cl, cb)):
        clamped_lat = min(abs(float(lat0)), lat_clamp_deg)
        cos_lat = math.cos(math.radians(clamped_lat))
        dlon = r * np.sin(phi) / cos_lat
        dlat = r * np.cos(phi)
        lon_unwrapped = float(lon0) + dlon
        lat_pts = float(lat0) + dlat
        lon_wrapped = ((lon_unwrapped + seam_lon_deg) % (2 * seam_lon_deg)) - seam_lon_deg
        if lon_wrapped.max() - lon_wrapped.min() > seam_lon_deg:
            for mask in (lon_wrapped < 0, lon_wrapped >= 0):
                if mask.any():
                    polys.append(np.column_stack([
                        np.deg2rad(lon_wrapped[mask]),
                        np.deg2rad(lat_pts[mask]),
                    ]))
                    src_idx.append(i)
        else:
            polys.append(np.column_stack([
                np.deg2rad(lon_wrapped),
                np.deg2rad(lat_pts),
            ]))
            src_idx.append(i)
    return polys, np.asarray(src_idx, dtype=int)


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
# Candidate survey footprints
# ---------------------------------------------------------------------------

# (name, l_min, l_max, b_min, b_max, color)
SURVEY_FOOTPRINTS = [
    ("Gal. plane (narrow)",     -10, 250,   -4,   4, "C0"),
    ("Gal. plane (wide)",         0, 360,  -20,  20, "C0"),
    ("Great circle l=220/40",   218, 222,  -90,  90, "C1"),
    ("Great circle l=40/220",    38,  42,  -90,  90, "C1"),
    ("Great circle l=130/310",  128, 132,  -90,  90, "C2"),
    ("Great circle l=310/130",  308, 312,  -90,  90, "C2"),
    ("NCP",                     105, 160,   15,  50, "C3"),
    ("Orion-Eridanus",          160, 220,  -70, -10, "C4"),
    ("North Polar Spur",        210, 380,    0,  90, "C5"),
    ("HVC",                      60, 180,   20,  60, "C6"),
    ("Magellanic Stream",        60, 110,  -90, -30, "C7"),
]


def add_survey_footprints(
    ax: plt.Axes,
    *,
    center_l: float = MOLL_CENTER_L,
) -> None:
    """Draw candidate survey region rectangles on a Mollweide axis."""
    seam = (center_l + 180) % 360

    def _normalize_l(l):
        return l % 360

    def _crosses_seam(l_min, l_max):
        span = l_max - l_min
        if span >= 360:
            return True
        ln = _normalize_l(l_min)
        lx = _normalize_l(l_max)
        if ln <= lx:
            return ln < seam < lx
        return seam > ln or seam < lx

    def _wrap_l(l):
        l = l - center_l
        if l > 180:
            l -= 360
        if l < -180:
            l += 360
        return l

    def _make_rect(l_min, l_max, b_min, b_max):
        n = 50
        wl_min, wl_max = _wrap_l(l_min), _wrap_l(l_max)
        if wl_min > wl_max:
            return None
        l_bot = np.linspace(wl_min, wl_max, n)
        l_top = np.linspace(wl_max, wl_min, n)
        b_left = np.linspace(b_min, b_max, n)
        b_right = np.linspace(b_max, b_min, n)
        ls = np.concatenate([l_bot, np.full(n, wl_max), l_top, np.full(n, wl_min)])
        bs = np.concatenate([np.full(n, b_min), b_left, np.full(n, b_max), b_right])
        return np.column_stack([np.deg2rad(ls), np.deg2rad(bs)])

    # Split footprints that cross the Mollweide seam
    split = []
    for name, l_min, l_max, b_min, b_max, color in SURVEY_FOOTPRINTS:
        if l_max - l_min >= 360:
            split.append((name, seam + 0.5, seam + 180, b_min, b_max, color))
            split.append((name, seam + 180, seam + 359.5, b_min, b_max, color))
        elif _crosses_seam(l_min, l_max):
            split.append((name, l_min, seam - 0.5, b_min, b_max, color))
            split.append((name, seam + 0.5, l_max, b_min, b_max, color))
        else:
            split.append((name, l_min, l_max, b_min, b_max, color))

    plotted_labels: set[str] = set()
    for name, l_min, l_max, b_min, b_max, color in split:
        verts = _make_rect(l_min, l_max, b_min, b_max)
        if verts is None:
            continue
        label = name if name not in plotted_labels else None
        poly = plt.Polygon(
            verts, alpha=ALPHA_EXTRA_LIGHT,
            facecolor=color, edgecolor=color,
            lw=LW_FINE, linestyle="--", label=label,
        )
        ax.add_patch(poly)
        if label:
            plotted_labels.add(name)


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


def plot_survey_mollweide_gridded(
    gl: np.ndarray,
    gb: np.ndarray,
    vals: np.ndarray,
    *,
    center_l: float,
    title: str,
    cbar_label: str,
    cmap: str,
    hpbw_deg: float = 3.4,
    pixel_deg: float = 0.5,
    cutoff_hpbw: float = 2.0,
    min_weight: float = 0.1,
) -> tuple[plt.Figure, plt.Axes]:
    """Beam-weighted Mollweide map of survey values.

    Each pixel is a Gaussian-beam-weighted average of pointings within
    ``cutoff_hpbw * hpbw_deg`` of its center. Pixels with total weight below
    ``min_weight`` are masked. Produces a pcolormesh fill instead of a scatter.
    """
    sigma_rad = np.deg2rad(hpbw_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    cutoff_rad = np.deg2rad(cutoff_hpbw * hpbw_deg)

    gl_shifted = gl - center_l
    gl_shifted = np.where(gl_shifted > 180, gl_shifted - 360, gl_shifted)
    gl_shifted = np.where(gl_shifted < -180, gl_shifted + 360, gl_shifted)

    half = pixel_deg / 2.0
    l_centers = np.deg2rad(np.arange(-180.0 + half, 180.0, pixel_deg))
    b_centers = np.deg2rad(np.arange(-90.0 + half, 90.0, pixel_deg))
    l_edges = np.deg2rad(np.arange(-180.0, 180.0 + pixel_deg, pixel_deg))
    b_edges = np.deg2rad(np.arange(-90.0, 90.0 + pixel_deg, pixel_deg))
    LL, BB = np.meshgrid(l_centers, b_centers)

    def _uv(l, b):
        return np.stack([np.cos(b) * np.cos(l),
                         np.cos(b) * np.sin(l),
                         np.sin(b)], axis=-1)

    P = _uv(np.deg2rad(gl_shifted), np.deg2rad(gb))         # (N, 3)
    G = _uv(LL, BB).reshape(-1, 3)                          # (M, 3)
    cosang = np.clip(G @ P.T, -1.0, 1.0)                    # (M, N)
    theta = np.arccos(cosang)
    w = np.exp(-0.5 * (theta / sigma_rad) ** 2)
    w = np.where(theta > cutoff_rad, 0.0, w)

    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    w = w * finite[None, :]
    vals_safe = np.where(finite, vals, 0.0)

    num = w @ vals_safe
    den = w.sum(axis=1)
    img = np.where(den > min_weight, num / den, np.nan).reshape(LL.shape)

    fig, _ax = landscapewidth_figure(5)
    _ax.remove()
    ax = fig.add_subplot(111, projection="mollweide")
    ax.grid(True, **{k: v for k, v in GRID_STYLE.items() if k != "color"},
            color=NEUTRAL_COLOR)

    LE, BE = np.meshgrid(l_edges, b_edges)
    pcm = ax.pcolormesh(LE, BE, img, cmap=cmap, shading="flat", zorder=2)

    l_line = np.linspace(-np.pi, np.pi, 500)
    ax.plot(l_line, np.zeros_like(l_line), lw=LW_FINE,
            color="k", alpha=ALPHA_FAINT, zorder=3)

    add_never_observable_overlay(ax, center_l=center_l, latitude_deg=LEO_LAT_DEG,
                                 min_alt_deg=MIN_ALT_DEG, max_alt_deg=MAX_ALT_DEG,
                                 az_min_deg=AZ_MIN_DEG, az_max_deg=AZ_MAX_DEG)

    plt.colorbar(pcm, ax=ax, label=cbar_label, shrink=0.6)

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


# ---------------------------------------------------------------------------
# Per-session T_sys diagnostics
# ---------------------------------------------------------------------------

def plot_tsys_histograms_per_session(
    tsys_per_session: dict,
    *,
    T_cal: float,
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Stacked per-session histograms of cell T_sys.

    ``tsys_per_session`` maps a session label to an iterable of T_sys
    values (one per cell).  Each panel shows the session histogram,
    the session median (solid), and the global median across all
    sessions (dashed).
    """
    sessions = [s for s, v in tsys_per_session.items() if len(v)]
    if not sessions:
        fig, ax = textwidth_figure(2)
        ax.set_axis_off()
        return fig, [ax]

    all_tsys = np.concatenate([np.asarray(tsys_per_session[s], dtype=float)
                               for s in sessions])
    lo = np.floor(np.percentile(all_tsys, 1))
    hi = np.ceil(np.percentile(all_tsys, 99))
    bins = np.linspace(lo, hi, 41)
    global_med = float(np.median(all_tsys))

    n = len(sessions)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 1.4 * n + 0.5))
    fig.set_layout_engine("tight")
    axes = fig.subplots(n, 1, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        vals = np.asarray(tsys_per_session[sess], dtype=float)
        ax.hist(vals, bins=bins, color="C0",
                edgecolor="black", linewidth=LW_FINE)
        med = float(np.median(vals))
        ax.axvline(med, color="C3", lw=LW_LIGHT,
                   label=rf"median = {med:.0f} K")
        ax.axvline(global_med, color=NEUTRAL_COLOR, ls="--", lw=LW_FINE,
                   label=rf"global = {global_med:.0f} K")
        ax.set_ylabel(sess.replace("session_", "s"),
                      rotation=0, ha="right", va="center",
                      fontsize=TICK_SIZE - 1)
        ax.text(0.99, 0.85, rf"$N={len(vals)}$",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=TICK_SIZE - 2)
        ax.legend(loc="upper left", fontsize=LEGEND_SIZE - 2,
                  framealpha=0.85)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 2)
        ax.grid(True, **GRID_STYLE)

    axes[-1].set_xlabel(r"$T_\mathrm{sys}$ [K]", fontsize=LABEL_SIZE)
    suptitle = title or rf"Per-session $T_\mathrm{{sys}}$ ($T_\mathrm{{cal}} = {T_cal:g}$ K, pol 1)"
    fig.suptitle(suptitle, fontsize=EMPHASIS_SIZE, y=1.0)
    return fig, list(axes)


def plot_tsys_vs_local_time_per_session(
    points_per_session: dict,
    *,
    T_cal: float,
    tz,
    cmap: str = "plasma",
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Per-session scatter of cell T_sys vs local time, coloured by Sun separation.

    ``points_per_session`` maps a session label to a 3-tuple
    ``(times_local, tsys_vals, sun_sep_deg)`` -- all equal-length
    sequences.  ``times_local`` must be tz-aware datetimes (in ``tz``).
    A single shared colorbar spans all sessions.
    """
    import matplotlib.dates as mdates

    sessions = [s for s, (t, *_rest) in points_per_session.items() if len(t)]
    if not sessions:
        fig, ax = textwidth_figure(2)
        ax.set_axis_off()
        return fig, [ax]

    all_tsys = np.concatenate([np.asarray(points_per_session[s][1], dtype=float)
                               for s in sessions])
    all_seps = np.concatenate([np.asarray(points_per_session[s][2], dtype=float)
                               for s in sessions])
    global_med = float(np.median(all_tsys))
    norm = mpl.colors.Normalize(vmin=float(np.min(all_seps)),
                                 vmax=float(np.max(all_seps)))

    n = len(sessions)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.0 * n + 0.5))
    fig.set_layout_engine("tight")
    axes = fig.subplots(n, 1, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        times_local, tsys_vals, seps = points_per_session[sess]
        order = np.argsort(times_local)
        t_ord = [times_local[i] for i in order]
        tsys_ord = np.asarray(tsys_vals, dtype=float)[order]
        sep_ord = np.asarray(seps, dtype=float)[order]
        ax.scatter(t_ord, tsys_ord, c=sep_ord, cmap=cmap, norm=norm,
                   **{**SCATTER_STYLE, "s": SS_FINE * 1.5})
        med = float(np.median(tsys_ord))
        ax.axhline(med, color="C3", lw=LW_LIGHT,
                   label=rf"median = {med:.0f} K")
        ax.axhline(global_med, color=NEUTRAL_COLOR, ls="--", lw=LW_FINE,
                   label=rf"global = {global_med:.0f} K")
        ax.set_ylabel(sess.replace("session_", "s") + "\n" + r"$T_\mathrm{sys}$ [K]",
                      fontsize=TICK_SIZE - 1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
        ax.tick_params(axis="x", rotation=20, labelsize=TICK_SIZE - 2)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 2)
        ax.legend(loc="upper left", fontsize=LEGEND_SIZE - 2,
                  framealpha=0.85)
        ax.grid(True, **GRID_STYLE)

    axes[-1].set_xlabel("Leuschner local time (Berkeley, CA / Pacific Time)",
                        fontsize=LABEL_SIZE)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                         location="bottom", pad=0.08, shrink=0.6,
                         aspect=40)
    cbar.set_label("Angular distance from Sun [deg]", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 2)
    suptitle = title or (
        rf"Per-session $T_\mathrm{{sys}}$ vs Leuschner local time "
        rf"($T_\mathrm{{cal}} = {T_cal:g}$ K, pol 1)"
    )
    fig.suptitle(suptitle, fontsize=EMPHASIS_SIZE, y=1.0)
    return fig, list(axes)


def plot_sun_sep_vs_local_time_per_session(
    points_per_session: dict,
    *,
    tz,
    cmap: str = "plasma",
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Per-session scatter of Sun angular distance vs local time, coloured by T_sys.

    Sister to :func:`plot_tsys_vs_local_time_per_session` -- swaps the
    y-axis and colour mapping so structure that depends on Sun proximity
    (rather than T_sys) is visible at a glance.
    """
    import matplotlib.dates as mdates

    sessions = [s for s, (t, *_rest) in points_per_session.items() if len(t)]
    if not sessions:
        fig, ax = textwidth_figure(2)
        ax.set_axis_off()
        return fig, [ax]

    all_tsys = np.concatenate([np.asarray(points_per_session[s][1], dtype=float)
                               for s in sessions])
    norm = mpl.colors.Normalize(vmin=float(np.min(all_tsys)),
                                 vmax=float(np.max(all_tsys)))

    n = len(sessions)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.0 * n + 0.5))
    fig.set_layout_engine("tight")
    axes = fig.subplots(n, 1, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        times_local, tsys_vals, seps = points_per_session[sess]
        order = np.argsort(times_local)
        t_ord = [times_local[i] for i in order]
        tsys_ord = np.asarray(tsys_vals, dtype=float)[order]
        sep_ord = np.asarray(seps, dtype=float)[order]
        ax.scatter(t_ord, sep_ord, c=tsys_ord, cmap=cmap, norm=norm,
                   **{**SCATTER_STYLE, "s": SS_FINE * 1.5})
        ax.set_ylabel(sess.replace("session_", "s") + "\n" + r"Sun sep. [deg]",
                      fontsize=TICK_SIZE - 1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
        ax.tick_params(axis="x", rotation=20, labelsize=TICK_SIZE - 2)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 2)
        ax.grid(True, **GRID_STYLE)

    axes[-1].set_xlabel("Leuschner local time (Berkeley, CA / Pacific Time)",
                        fontsize=LABEL_SIZE)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                         location="bottom", pad=0.08, shrink=0.6,
                         aspect=40)
    cbar.set_label(r"$T_\mathrm{sys}$ [K]", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 2)
    suptitle = title or (
        r"Per-session Sun angular distance vs Leuschner local time "
        r"(coloured by $T_\mathrm{sys}$, pol 1)"
    )
    fig.suptitle(suptitle, fontsize=EMPHASIS_SIZE, y=1.0)
    return fig, list(axes)
