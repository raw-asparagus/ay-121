"""Lab 4 - reusable plotting functions for 21 cm HI survey data.

All functions use the style constants from ``ugradiolab.plotting`` and return
axes for interactive use.  Notebooks import individual functions::

    from plotters import plot_hi_spectrum, plot_survey_mollweide
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import astropy.coordinates as ac
import astropy.units as u
from astropy.time import Time
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
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
    LW_STANDARD,
    COLUMNWIDTH_IN,
    SS_MICRO,
    SS_FINE,
    SS_STANDARD,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    ALPHA_FULL,
    NEUTRAL_COLOR,
    GRID_STYLE,
    GUIDE_STYLE,
    SCATTER_STYLE,
    subpanels,
    textwidth_figure,
    landscapewidth_figure,
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


def az_to_lon(az_deg: np.ndarray) -> np.ndarray:
    """Wrap azimuth into [-180, 180] for Mollweide plotting.

    Azimuths in [0, 180] map to themselves; (180, 360) map to negative
    longitudes so the cable-wrap wedge near az=0/360 sits on the central
    meridian.
    """
    az = az_deg % 360.0
    return np.where(az <= 180.0, az, az - 360.0)


def savefig(fig: plt.Figure, name: str) -> None:
    """Save *fig* as a PDF in ``report/figures/``."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)
    print(f"  {name}")


# ---------------------------------------------------------------------------
# Single HI spectrum
# ---------------------------------------------------------------------------

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

def never_observable_mask(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    *,
    latitude_deg: float = LEO_LAT_DEG,
    min_alt_deg: float = MIN_ALT_DEG,
    max_alt_deg: float = MAX_ALT_DEG,
    az_min_deg: float = AZ_MIN_DEG,
    az_max_deg: float = AZ_MAX_DEG,
    ha_step_deg: float = 1.0,
) -> np.ndarray:
    """Boolean mask: True where (l, b) is never observable from Leuschner.

    ``l_deg`` and ``b_deg`` are broadcastable galactic-coordinate arrays
    (any shape). The hour-angle sweep at 1 deg resolution is enough for
    sub-degree mask boundaries.
    """
    l_arr = np.asarray(l_deg, dtype=float)
    b_arr = np.broadcast_to(np.asarray(b_deg, dtype=float), l_arr.shape)
    gc = ac.SkyCoord(l=l_arr.ravel() * u.deg, b=b_arr.ravel() * u.deg,
                     frame="galactic")
    dec_rad = np.deg2rad(gc.transform_to(ac.ICRS()).dec.deg).reshape(l_arr.shape)
    lat_rad = np.deg2rad(latitude_deg)

    ha_steps = np.deg2rad(np.arange(0.0, 360.0, ha_step_deg))
    alt_lo_r = np.deg2rad(min_alt_deg)
    alt_hi_r = np.deg2rad(max_alt_deg)
    az_lo_r = np.deg2rad(az_min_deg)
    az_hi_r = np.deg2rad(az_max_deg)

    observable = np.zeros(l_arr.shape, dtype=bool)
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
    return ~observable


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
    inacc_mask = never_observable_mask(
        L_true, B_dense,
        latitude_deg=latitude_deg,
        min_alt_deg=min_alt_deg, max_alt_deg=max_alt_deg,
        az_min_deg=az_min_deg, az_max_deg=az_max_deg,
    )

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
# Topocentric (alt/az) Mollweide accessibility overlay
# ---------------------------------------------------------------------------

def add_topo_inaccessible_overlay(
    ax: plt.Axes,
    *,
    min_alt_deg: float = MIN_ALT_DEG,
    max_alt_deg: float = MAX_ALT_DEG,
    az_min_deg: float = AZ_MIN_DEG,
    az_max_deg: float = AZ_MAX_DEG,
) -> None:
    """Draw the dish's inaccessible alt/az region on a Mollweide axis.

    Convention used: the axis is plotted in topocentric coordinates with
    az=0 (=360) mapped to lon=0 (projection center).  Azimuth wraps via
    ``az if az <= 180 else az - 360`` so the 348..360..7 dead wedge sits
    on the central meridian.  Altitude maps directly to latitude.

    Three exclusions are hatched in the same style as
    :func:`add_never_observable_overlay`:

    * ``alt < min_alt_deg`` (horizon band)
    * ``alt > max_alt_deg`` (zenith cap)
    * ``az`` outside ``[az_min_deg, az_max_deg]`` (cable-wrap wedge near 0)
    """
    az_grid = np.arange(-179.5, 180.5, 1.0)
    alt_grid = np.arange(-89.5, 90.5, 1.0)
    AZ, ALT = np.meshgrid(az_grid, alt_grid)

    az_true = AZ % 360.0
    az_bad = (az_true < az_min_deg) | (az_true > az_max_deg)
    alt_bad = (ALT < min_alt_deg) | (ALT > max_alt_deg)
    mask = az_bad | alt_bad

    with mpl.rc_context({"hatch.color": "red", "hatch.linewidth": 0.5}):
        ax.contourf(
            np.deg2rad(AZ),
            np.deg2rad(ALT),
            mask.astype(float),
            levels=[0.5, 1.5],
            colors=["red"],
            alpha=0.08,
            hatches=["///"],
            zorder=0,
        )
        ax.contour(
            np.deg2rad(AZ),
            np.deg2rad(ALT),
            mask.astype(float),
            levels=[0.5],
            colors=["red"],
            linewidths=0.6,
            alpha=0.4,
            zorder=1,
        )


# ---------------------------------------------------------------------------
# Per-session coverage (topocentric + galactic flat-sky)
# ---------------------------------------------------------------------------

def plot_session_coverage(
    session: dict,
    even_grid,
    *,
    hpbw_deg: float = 3.4,
) -> plt.Figure:
    """Two-panel coverage view of one archived session.

    Left: topocentric Mollweide with the dish's inaccessible regions
    hatched, science pointings coloured by hours into the session, recal
    detours as red crosses, and a seam-aware dashed track in observation
    order.  Right: galactic flat-sky with the full survey grid as a faint
    backdrop, per-cell beam ellipses (cos(b)-stretched in longitude), the
    same time-coloured scatter, the same recal markers, and the same
    track.  Axis limits expand to include off-grid recal pointings.

    ``session`` is a dict as returned by :func:`utils.io.rank_sessions_by_contiguity`
    (keys ``all_cells, t_start, name, n_cells, n_recals, duration_h, l_span``).
    ``even_grid`` is the iterable of ``(*, *, l, b, *)`` records produced by
    :func:`utils.mapping.build_galplane_grid`.
    """
    from matplotlib.collections import LineCollection, PolyCollection

    from utils.mapping import cell_radec, fast_altaz

    all_l = np.array([c['l'] for c in session['all_cells']])
    all_b = np.array([c['b'] for c in session['all_cells']])
    all_t = np.array([c['t'] for c in session['all_cells']])
    is_recal = np.array([c['is_recal'] for c in session['all_cells']])
    all_h = (all_t - session['t_start']) / 3600.0
    sci = ~is_recal

    alts = np.empty(all_l.shape); azs = np.empty(all_l.shape)
    for i, (l, b, t) in enumerate(zip(all_l, all_b, all_t)):
        ra, dec = cell_radec(l, b)
        alts[i], azs[i] = fast_altaz(ra, dec, t)
    lons = az_to_lon(azs)

    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=all_h[sci].min(), vmax=all_h[sci].max())

    fig, _ax = textwidth_figure(6)
    _ax.remove()
    # Mollweide left, flat-sky middle, slim colorbar column right.
    # Constrained-layout would collapse the Mollweide projection, so we
    # tune the spacing by hand.
    gs = fig.add_gridspec(1, 3, width_ratios=(0.95, 0.95, 0.04),
                          wspace=0.22, left=0.04, right=0.97,
                          top=0.93, bottom=0.10)
    ax_topo = fig.add_subplot(gs[0, 0], projection='mollweide')
    ax_gal = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    recal_kw = dict(marker='x', color='C3', s=SS_FINE * 1.5,
                    linewidths=LW_LIGHT, zorder=6, label='recal drift')

    # Topocentric Mollweide
    ax_topo.grid(True, **{k: v for k, v in GRID_STYLE.items() if k != 'color'},
                 color=NEUTRAL_COLOR)
    add_topo_inaccessible_overlay(ax_topo)

    # --- Altitude-distribution highlight ---------------------------------
    # The planner concentrates pointings well above the horizon to suppress
    # ground pickup and air-mass excess. We shade the band between the 25th
    # and 75th percentiles of the science altitudes and overplot the median
    # so the bias toward high altitudes is plain at a glance.
    sci_alt = alts[sci]
    alt_p25, alt_med, alt_p75 = np.percentile(sci_alt, [25, 50, 75])
    alt_thresh = 60.0
    frac_high = float(np.mean(sci_alt >= alt_thresh))
    az_line = np.linspace(-np.pi, np.pi, 400)
    ax_topo.fill_between(
        az_line,
        np.full_like(az_line, np.deg2rad(alt_p25)),
        np.full_like(az_line, np.deg2rad(alt_p75)),
        color='C2', alpha=0.12, zorder=1,
    )
    ax_topo.plot(az_line, np.full_like(az_line, np.deg2rad(alt_med)),
                 color='C2', lw=LW_LIGHT, ls='--', alpha=ALPHA_STANDARD,
                 zorder=2,
                 label=rf'median alt = ${alt_med:.0f}^\circ$')

    ax_topo.scatter(
        np.deg2rad(lons[sci]), np.deg2rad(alts[sci]),
        c=all_h[sci], cmap=cmap, norm=norm,
        s=SS_MICRO, edgecolors='none', zorder=5,
    )
    if is_recal.any():
        ax_topo.scatter(np.deg2rad(lons[is_recal]),
                        np.deg2rad(alts[is_recal]), **recal_kw)

    pts = np.column_stack([np.deg2rad(lons), np.deg2rad(alts)])
    segs = [[p0, p1] for p0, p1 in zip(pts[:-1], pts[1:])
            if abs(p1[0] - p0[0]) <= math.pi]
    ax_topo.add_collection(LineCollection(
        segs, colors='k', linestyles='--',
        linewidths=LW_FINE, alpha=ALPHA_FAINT, zorder=3,
    ))

    az_ticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax_topo.set_xticks(np.deg2rad(az_ticks))
    ax_topo.set_xticklabels(
        [rf'{int(t % 360)}$^\circ$' for t in az_ticks],
        fontsize=TICK_SIZE - 3,
    )

    # Galactic flat-sky
    grid_l = np.array([c[2] for c in even_grid])
    grid_b = np.array([c[3] for c in even_grid])
    ax_gal.scatter(grid_l, grid_b, c='lightgrey', s=SS_FINE * 0.5,
                   edgecolors='none', alpha=ALPHA_FAINT, zorder=1,
                   label='full grid')

    phi = np.linspace(0.0, 2.0 * np.pi, 33)
    r = hpbw_deg / 2.0
    polys_gal = []
    for lc, bc in zip(all_l[sci], all_b[sci]):
        cos_b = math.cos(math.radians(bc))
        polys_gal.append(np.column_stack(
            [lc + (r / cos_b) * np.sin(phi), bc + r * np.cos(phi)]
        ))
    ax_gal.add_collection(PolyCollection(
        polys_gal, facecolors='none', edgecolors=cmap(norm(all_h[sci])),
        linewidths=LW_LIGHT, alpha=ALPHA_FAINT, zorder=3,
    ))
    ax_gal.scatter(all_l[sci], all_b[sci], c=all_h[sci], cmap=cmap, norm=norm,
                   s=SS_FINE, edgecolors='none', zorder=4)
    if is_recal.any():
        ax_gal.scatter(all_l[is_recal], all_b[is_recal], **recal_kw)
    ax_gal.plot(all_l, all_b, color='k', ls='--', lw=LW_FINE,
                alpha=ALPHA_FAINT, zorder=2)

    sci_l, sci_b = all_l[sci], all_b[sci]
    l_pad = hpbw_deg / max(math.cos(math.radians(np.max(np.abs(sci_b)))), 0.1)
    l_hi = max(sci_l.max(), all_l.max()) + l_pad
    l_lo = min(sci_l.min(), all_l.min()) - l_pad
    b_hi = max(sci_b.max(), all_b.max()) + hpbw_deg
    b_lo = min(sci_b.min(), all_b.min()) - hpbw_deg
    ax_gal.set_xlim(l_hi, l_lo)
    ax_gal.set_ylim(b_lo, b_hi)
    ax_gal.set_aspect('equal')
    ax_gal.set_xlabel(r'$\ell$ [deg]', fontsize=LABEL_SIZE)
    ax_gal.set_ylabel(r'$b$ [deg]', fontsize=LABEL_SIZE)
    ax_gal.legend(fontsize=LEGEND_SIZE - 1, loc='lower left')

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation='vertical',
                 label='hours into session')

    return fig


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
    """Beam-weighted Mollweide map of survey values (FFT-based).

    Samples are deposited (nearest-pixel) onto an intermediate sinusoidal
    grid u, v = (l - center_l) * cos(b), b, in which the dish beam is well
    approximated as a stationary 2-D Gaussian.  Two grids -- weighted-value
    and weight -- are convolved via scipy.signal.fftconvolve with a Gaussian
    kernel of FWHM = hpbw_deg (truncated at cutoff_hpbw * hpbw_deg), then
    bilinearly sampled back onto the (l, b) output grid for the Mollweide
    pcolormesh.  Pixels with total weight below ``min_weight`` are masked.

    The sinusoidal-projection approximation incurs an angular-distance
    error of order (kernel / R_curvature) ** 2 with R = 1 rad ~= 57 deg,
    i.e. <0.1% for HPBW = 3.4 deg -- negligible compared to per-pointing
    noise.  Cost is O(N + Npix log Npix) vs. O(N * Npix) for the exact
    great-circle matrix formulation it replaces, with a constant memory
    footprint set by the grid resolution rather than N.
    """
    from scipy.signal import fftconvolve
    from scipy.ndimage import map_coordinates

    sigma_deg = hpbw_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    cutoff_deg = cutoff_hpbw * hpbw_deg

    gl_shifted = (np.asarray(gl) - center_l + 180.0) % 360.0 - 180.0
    gb = np.asarray(gb)

    half = pixel_deg / 2.0
    u_centers = np.arange(-180.0 + half, 180.0, pixel_deg)
    v_centers = np.arange(-90.0 + half, 90.0, pixel_deg)
    nu, nv = len(u_centers), len(v_centers)

    # Deposit samples on (u, v).  u = l_shifted * cos(b) puts the kernel
    # in a locally-Euclidean frame -- sample-to-pixel separation in (u, v)
    # tracks great-circle distance to (kernel / R) ** 2.
    u_samp = gl_shifted * np.cos(np.deg2rad(gb))
    v_samp = gb

    vals = np.asarray(vals, dtype=float)
    finite = np.isfinite(vals)
    vals_safe = np.where(finite, vals, 0.0)
    w_samp = finite.astype(float)

    iu = np.clip(np.round((u_samp - u_centers[0]) / pixel_deg).astype(int),
                 0, nu - 1)
    iv = np.clip(np.round((v_samp - v_centers[0]) / pixel_deg).astype(int),
                 0, nv - 1)

    num_grid = np.zeros((nv, nu))
    den_grid = np.zeros((nv, nu))
    np.add.at(num_grid, (iv, iu), vals_safe * w_samp)
    np.add.at(den_grid, (iv, iu), w_samp)

    # Gaussian beam kernel, truncated at cutoff_deg.
    sigma_pix = sigma_deg / pixel_deg
    half_kernel = int(np.ceil(cutoff_deg / pixel_deg))
    k_axis = np.arange(-half_kernel, half_kernel + 1)
    KX, KY = np.meshgrid(k_axis, k_axis)
    r2_pix2 = KX ** 2 + KY ** 2
    kernel = np.exp(-0.5 * r2_pix2 / sigma_pix ** 2)
    kernel[r2_pix2 > (cutoff_deg / pixel_deg) ** 2] = 0.0

    num_conv = fftconvolve(num_grid, kernel, mode='same')
    den_conv = fftconvolve(den_grid, kernel, mode='same')

    # Output (l, b) grid; sample the convolved field back at each pixel
    # center via bilinear interpolation in (u, v).
    l_centers = np.arange(-180.0 + half, 180.0, pixel_deg)
    b_centers = np.arange(-90.0 + half, 90.0, pixel_deg)
    l_edges = np.deg2rad(np.arange(-180.0, 180.0 + pixel_deg, pixel_deg))
    b_edges = np.deg2rad(np.arange(-90.0, 90.0 + pixel_deg, pixel_deg))
    LL, BB = np.meshgrid(l_centers, b_centers)

    U_out = LL * np.cos(np.deg2rad(BB))
    iu_o = (U_out - u_centers[0]) / pixel_deg
    iv_o = (BB - v_centers[0]) / pixel_deg
    num_out = map_coordinates(num_conv, [iv_o, iu_o], order=1,
                              mode='constant', cval=0.0)
    den_out = map_coordinates(den_conv, [iv_o, iu_o], order=1,
                              mode='constant', cval=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        img = np.where(den_out > min_weight, num_out / den_out, np.nan)

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
# l-v strip
# ---------------------------------------------------------------------------

def plot_lv_strip(
    l_fine: np.ndarray,
    v_axis: np.ndarray,
    lv_image: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "inferno",
    v_lim: tuple[float, float] = (-150.0, 130.0),
    b_overlay_deg: float = 0.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a beam-weighted l-v image with never-observable overlay.

    ``lv_image`` has shape ``(nv, M)`` matching ``len(v_axis)`` and
    ``len(l_fine)``. NaN pixels are rendered transparent. The horizon
    mask is computed at ``b_overlay_deg`` so the overlay matches the
    strip's effective latitude.
    """
    dl = float(l_fine[1] - l_fine[0])
    v_step = float(v_axis[1] - v_axis[0])
    v_edges = np.concatenate([
        [v_axis[0] - 0.5 * v_step],
        0.5 * (v_axis[:-1] + v_axis[1:]),
        [v_axis[-1] + 0.5 * v_step],
    ])
    l_edges = np.concatenate([
        [l_fine[0] - 0.5 * dl],
        0.5 * (l_fine[:-1] + l_fine[1:]),
        [l_fine[-1] + 0.5 * dl],
    ])

    finite_vals = lv_image[np.isfinite(lv_image)]
    vmin = max(0.0, np.nanpercentile(finite_vals, 1)) if finite_vals.size else 0.0
    vmax = np.nanpercentile(finite_vals, 99) if finite_vals.size else 1.0

    fig, ax = plt.subplots(figsize=(11, 6))
    pcm = ax.pcolormesh(l_edges, v_edges, lv_image,
                        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")

    # Never-observable wedge at this latitude (1-D slice).
    inacc = never_observable_mask(l_fine.astype(float) % 360.0,
                                  np.full_like(l_fine, b_overlay_deg, dtype=float))
    runs = []
    k = 0
    while k < len(l_fine):
        if inacc[k]:
            k0 = k
            while k < len(l_fine) and inacc[k]:
                k += 1
            runs.append((l_edges[k0], l_edges[k]))
        else:
            k += 1
    first_patch = True
    for l0, l1 in runs:
        label = "Never observable" if first_patch else None
        ax.fill_between([l0, l1], v_lim[0], v_lim[1],
                        color="red", alpha=0.08, zorder=3)
        with mpl.rc_context({"hatch.color": "red", "hatch.linewidth": 0.5}):
            ax.fill_between([l0, l1], v_lim[0], v_lim[1],
                            facecolor="none", edgecolor="red", linewidth=0,
                            hatch="///", alpha=0.4, zorder=4, label=label)
        first_patch = False

    ax.set_xlabel(r"Galactic longitude $\ell$ [deg]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$v_{\rm LSR}$ [km s$^{-1}$]", fontsize=LABEL_SIZE)
    ax.set_ylim(v_lim)
    ax.invert_xaxis()
    if title:
        ax.set_title(title, fontsize=EMPHASIS_SIZE)
    if runs:
        ax.legend(loc="lower left", fontsize=LEGEND_SIZE, framealpha=0.9)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Per-session T_sys diagnostics
# ---------------------------------------------------------------------------

def plot_tsys_histogram(
    tsys_values,
    *,
    title: str | None = None,
):
    """Single-panel histogram of the calibrated per-cell T_sys distribution.

    Parameters
    ----------
    tsys_values : array-like
        Flat iterable of per-(session, cell) T_sys values in K.
    """
    vals = np.asarray(list(tsys_values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        fig, ax = plt.subplots(figsize=(COLUMNWIDTH_IN, COLUMNWIDTH_IN * 0.55))
        ax.set_axis_off()
        return fig, ax

    lo = np.floor(np.percentile(vals, 1))
    hi = np.ceil(np.percentile(vals, 99))
    bins = np.linspace(lo, hi, 41)
    median = float(np.median(vals))
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH_IN, COLUMNWIDTH_IN * 0.55))
    ax.hist(vals, bins=bins, color='C0', alpha=0.7, edgecolor='C0', lw=LW_FINE)
    ax.axvline(median, color='C3', lw=LW_LIGHT, ls='-',
               label=f'median = {median:.1f} K')
    ax.axvline(q25, color='0.4', lw=LW_FINE, ls='--',
               label=f'IQR = [{q25:.1f}, {q75:.1f}] K')
    ax.axvline(q75, color='0.4', lw=LW_FINE, ls='--')
    ax.set_xlabel(r'$T_{\rm sys,\,pol\,1}$ [K]')
    ax.set_ylabel('Cell count')
    ax.legend(loc='upper right', fontsize='small')
    if title is not None:
        ax.set_title(title, fontsize='small')
    plt.tight_layout()
    return fig, ax


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
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Per-session scatter of cell T_sys vs local time.

    ``points_per_session`` maps a session label to a 2-tuple
    ``(times_local, tsys_vals)`` -- both equal-length sequences.
    ``times_local`` must be tz-aware datetimes (in ``tz``).
    """
    import matplotlib.dates as mdates

    sessions = [s for s, (t, *_rest) in points_per_session.items() if len(t)]
    if not sessions:
        fig, ax = textwidth_figure(2)
        ax.set_axis_off()
        return fig, [ax]

    all_tsys = np.concatenate([np.asarray(points_per_session[s][1], dtype=float)
                               for s in sessions])
    global_med = float(np.median(all_tsys))

    n = len(sessions)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.0 * n + 0.5))
    fig.set_layout_engine("tight")
    axes = fig.subplots(n, 1, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        times_local, tsys_vals = points_per_session[sess][:2]
        order = np.argsort(times_local)
        t_ord = [times_local[i] for i in order]
        tsys_ord = np.asarray(tsys_vals, dtype=float)[order]
        ax.scatter(t_ord, tsys_ord,
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
    suptitle = title or (
        rf"Per-session $T_\mathrm{{sys}}$ vs Leuschner local time "
        rf"($T_\mathrm{{cal}} = {T_cal:g}$ K, pol 1)"
    )
    fig.suptitle(suptitle, fontsize=EMPHASIS_SIZE, y=1.0)
    return fig, list(axes)


def plot_repeat_cell_drift(
    sessions_ordered: list,
    tsys_by_session_cell: dict,
    gain_by_session_cell: dict,
    *,
    T_cal: float,
    title: str | None = None,
    bins: int = 30,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Histogram of per-cell max - min for repeat cells (>=2 sessions).

    Top: spread of T_sys [K] across sessions for the same pointing.
    Bottom: spread of the diode step ``gain * T_cal`` [K-equivalent] across
    sessions for the same pointing. A narrow distribution near zero means
    the instrument (receiver + diode) is stable cell-by-cell; a wide or
    offset distribution implies drift between sessions.
    """
    cell_to_sessions = {}
    for (sess, gl, gb), _ in tsys_by_session_cell.items():
        cell_to_sessions.setdefault((gl, gb), []).append(sess)
    repeat_cells = [k for k, ss in cell_to_sessions.items() if len(set(ss)) >= 2]

    t_spreads, g_spreads = [], []
    for (gl, gb) in repeat_cells:
        ts, gs = [], []
        for sess in sessions_ordered:
            t = tsys_by_session_cell.get((sess, gl, gb))
            g = gain_by_session_cell.get((sess, gl, gb))
            if t is not None and np.isfinite(t):
                ts.append(float(t))
            if g is not None and np.isfinite(g):
                gs.append(float(g) * T_cal)
        if len(ts) >= 2:
            t_spreads.append(max(ts) - min(ts))
        if len(gs) >= 2:
            g_spreads.append(max(gs) - min(gs))

    t_spreads = np.asarray(t_spreads, dtype=float)
    g_spreads = np.asarray(g_spreads, dtype=float)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.5))
    fig.set_layout_engine("tight")
    ax_t, ax_g = fig.subplots(2, 1)

    for ax, data, xlabel in (
        (ax_t, t_spreads, r"$T_\mathrm{sys}^{\max} - T_\mathrm{sys}^{\min}$ [K]"),
        (ax_g, g_spreads,
         rf"$(g \cdot T_\mathrm{{cal}})^{{\max}} - (g \cdot T_\mathrm{{cal}})^{{\min}}$ "
         rf"[K, $T_\mathrm{{cal}}={T_cal:g}$ K]"),
    ):
        if data.size:
            ax.hist(data, bins=bins, color="C0",
                    edgecolor=NEUTRAL_COLOR, alpha=ALPHA_STANDARD)
            med = float(np.median(data))
            p90 = float(np.percentile(data, 90))
            ax.axvline(med, color="C3", lw=LW_LIGHT,
                       label=rf"median = {med:.1f}")
            ax.axvline(p90, color=NEUTRAL_COLOR, ls="--", lw=LW_FINE,
                       label=rf"p90 = {p90:.1f}")
            ax.legend(loc="upper right", fontsize=LEGEND_SIZE - 1,
                      framealpha=0.85)
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        ax.set_ylabel("cells", fontsize=LABEL_SIZE)
        ax.grid(True, **GRID_STYLE)

    suptitle = title or (
        rf"Per-cell cross-session spread (max $-$ min), "
        rf"{len(repeat_cells)} cells observed in $\geq 2$ sessions"
    )
    fig.suptitle(suptitle, fontsize=EMPHASIS_SIZE, y=1.0)
    return fig, [ax_t, ax_g]


def plot_tsys_vs_alt_per_session(
    points_per_session: dict,
    *,
    T_cal: float,
    title: str | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Per-session scatter of cell T_sys vs mean altitude.

    ``points_per_session`` maps a session label to a 2-tuple
    ``(alt_deg, tsys_vals)`` of equal-length sequences.
    """
    sessions = [s for s, (a, *_rest) in points_per_session.items() if len(a)]
    if not sessions:
        fig, ax = textwidth_figure(2)
        ax.set_axis_off()
        return fig, [ax]

    all_tsys = np.concatenate([np.asarray(points_per_session[s][1], dtype=float)
                               for s in sessions])
    global_med = float(np.median(all_tsys))

    n = len(sessions)
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.0 * n + 0.5))
    fig.set_layout_engine("tight")
    axes = fig.subplots(n, 1, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, sess in zip(axes, sessions):
        alt_vals, tsys_vals = points_per_session[sess]
        alt_arr = np.asarray(alt_vals, dtype=float)
        tsys_arr = np.asarray(tsys_vals, dtype=float)
        ax.scatter(alt_arr, tsys_arr, color="C0",
                   **{**SCATTER_STYLE, "s": SS_FINE * 1.5})
        med = float(np.median(tsys_arr))
        ax.axhline(med, color="C3", lw=LW_LIGHT,
                   label=rf"median = {med:.0f} K")
        ax.axhline(global_med, color=NEUTRAL_COLOR, ls="--", lw=LW_FINE,
                   label=rf"global = {global_med:.0f} K")
        ax.set_ylabel(sess.replace("session_", "s") + "\n" + r"$T_\mathrm{sys}$ [K]",
                      fontsize=TICK_SIZE - 1)
        ax.tick_params(axis="x", labelsize=TICK_SIZE - 2)
        ax.tick_params(axis="y", labelsize=TICK_SIZE - 2)
        ax.legend(loc="upper right", fontsize=LEGEND_SIZE - 2,
                  framealpha=0.85)
        ax.grid(True, **GRID_STYLE)

    axes[-1].set_xlabel("Mean altitude [deg]", fontsize=LABEL_SIZE)
    suptitle = title or (
        rf"Per-session $T_\mathrm{{sys}}$ vs altitude "
        rf"($T_\mathrm{{cal}} = {T_cal:g}$ K, pol 1)"
    )
    fig.suptitle(suptitle, fontsize=EMPHASIS_SIZE, y=1.0)
    return fig, list(axes)


def spectra_per_session_pdf(
    out_path,
    v_axis,
    viable_pairs_per_cell: dict,
    excluded_cells: set,
    sessions: list,
    *,
    spectrum_key: str = 'R_lsr',
    ncols: int = 5,
    color: str = 'C0',
    title_suffix: str = 'post pair filter + QA, LSR frame',
) -> int:
    """Write a per-session spectra grid PDF.

    For each session, average the per-pair spectra at each non-excluded cell,
    then lay them out via :func:`plot_spectra_grid`.  Returns the number of
    PDF pages written.
    """
    from collections import defaultdict
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import matplotlib.pyplot as plt

    viable_per_session_cell: dict = defaultdict(list)
    for (gl, gb), pairs in viable_pairs_per_cell.items():
        if (gl, gb) in excluded_cells:
            continue
        for p in pairs:
            viable_per_session_cell[(p['session'], gl, gb)].append(p[spectrum_key])

    def _safe_nanmean(rs):
        stack = np.array(rs)
        col_has_data = np.any(np.isfinite(stack), axis=0)
        mean = np.full(stack.shape[1], np.nan)
        if col_has_data.any():
            mean[col_has_data] = np.nanmean(stack[:, col_has_data], axis=0)
        return mean

    n_pages = 0
    with PdfPages(out_path) as pdf:
        for dr in sessions:
            dr_spectra = {
                (l, b): _safe_nanmean(rs)
                for (d, l, b), rs in viable_per_session_cell.items()
                if d == dr
            }
            if not dr_spectra:
                continue
            result = plot_spectra_grid(
                v_axis, dr_spectra,
                ncols=ncols,
                color=color,
                title=f'{dr} -- {len(dr_spectra)} pointings ({title_suffix})',
            )
            figs = result if isinstance(result, list) else [result]
            for f in figs:
                pdf.savefig(f)
                plt.close(f)
                n_pages += 1
    return n_pages



# ---------------------------------------------------------------------------
# Tcal(t) calibration figures (main_scan_calibration.ipynb)
# ---------------------------------------------------------------------------

LEUSCHNER_LOC = ac.EarthLocation(lat=37.9183 * u.deg, lon=-122.1067 * u.deg,
                                 height=304 * u.m)
PDT_TZ = dt.timezone(dt.timedelta(hours=-7), name="PDT")
SIDEREAL_RATE = 1.00273790935
SIDEREAL_DAY_S = 86400.0 / SIDEREAL_RATE
SOLAR_DAY_S = 86400.0
LST_TICK_HOURS = (0, 6, 12, 18)
PDT_TICK_HOURS = (0, 6, 12, 18)

SESSION_SPAN_ALPHA = 0.08
SESSION_BOUNDARY_ALPHA = 0.45
SESSION_BOUNDARY_COLOR = "0.35"


def _lst_hour(unix_t):
    return float(Time(unix_t, format="unix", location=LEUSCHNER_LOC)
                 .sidereal_time("apparent").hour) % 24.0


def _ticks_at_lst(t_lo, t_hi, hours):
    lst0 = _lst_hour(t_lo)
    out = []
    for tgt in hours:
        dlst = (tgt - lst0) % 24.0
        t = t_lo + (dlst / SIDEREAL_RATE) * 3600.0
        while t <= t_hi:
            out.append(t)
            t += SIDEREAL_DAY_S
    return sorted(out)


def _ticks_at_pdt(t_lo, t_hi, hours):
    out = []
    for tgt in hours:
        day0 = dt.datetime.fromtimestamp(t_lo, PDT_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0)
        t = (day0 + dt.timedelta(hours=tgt)).timestamp()
        while t < t_lo:
            t += SOLAR_DAY_S
        while t <= t_hi:
            out.append(t)
            t += SOLAR_DAY_S
    return sorted(out)


def time_axes_lst_pdt(axes):
    """Annotate a shared-x stack with LST ticks (bottom) and PDT ticks (top).

    The Unix-epoch axis the panels were plotted against is preserved; only
    tick locators/labels are added.  Returns the twinned top axis.
    """
    bottom = axes[-1]
    t_lo, t_hi = bottom.get_xlim()
    lst_ticks = _ticks_at_lst(t_lo, t_hi, LST_TICK_HOURS)
    bottom.set_xticks(lst_ticks)
    bottom.set_xticklabels([f"{int(round(_lst_hour(t))) % 24:d}h"
                            for t in lst_ticks], rotation=30, ha="right")
    bottom.set_xlabel("LST")
    top = axes[0].twiny()
    top.set_xlim(axes[0].get_xlim())
    pdt_ticks = _ticks_at_pdt(t_lo, t_hi, PDT_TICK_HOURS)
    top.set_xticks(pdt_ticks)
    top.set_xticklabels(
        [dt.datetime.fromtimestamp(t, PDT_TZ).strftime("%H:%M")
         for t in pdt_ticks], rotation=30, ha="left")
    top.set_xlabel("PDT")
    return top


def shade_sessions(ax, session_spans):
    """Shade observation sessions on a Unix-epoch axis.

    ``session_spans`` maps a session label to ``(t_start, t_end)`` in
    Unix-epoch seconds.  Sessions are filled with NEUTRAL_COLOR and their
    boundaries drawn as faint vertical lines.
    """
    boundaries = set()
    for t0, t1 in session_spans.values():
        ax.axvspan(t0, t1, color=NEUTRAL_COLOR,
                   alpha=SESSION_SPAN_ALPHA, zorder=0)
        boundaries.add(t0)
        boundaries.add(t1)
    for t in boundaries:
        ax.axvline(t, color=SESSION_BOUNDARY_COLOR,
                   alpha=SESSION_BOUNDARY_ALPHA,
                   lw=LW_FINE, zorder=1)


def plot_ebhis_vs_leuschner_R(
    ebhis_spectra,
    pointing_R_avg,
    pointing_R_N,
    ebhis_peak_v, ebhis_peak_tb,
    R_peak_v, R_peak_ref,
    v_lsr_axis,
    pointing_labels,
    *,
    v_lsr_window,
    n_visits_by_pointing,
):
    """Two-panel sanity check: EBHIS T_B (top) and Leuschner R(v) (bottom).

    Peak markers are overlaid on both panels.  ``ebhis_spectra`` maps
    pointing -> (v_eb, T_eb); ``pointing_R_avg`` maps pointing -> R(c);
    ``pointing_R_N`` maps pointing -> (n_lo1, n_lo2).  ``v_lsr_window`` is
    the ``(lo, hi)`` shading range for the Leuschner window on EBHIS.
    """
    v_lo, v_hi = v_lsr_window
    fig, axes = plt.subplots(2, 1,
                             figsize=(TEXTWIDTH_IN * 0.6, TEXTWIDTH_IN * 0.8),
                             constrained_layout=True)

    ax = axes[0]
    for name in ebhis_spectra:
        v, T = ebhis_spectra[name]
        ax.axvspan(v_lo, v_hi, color="C2", alpha=ALPHA_FAINT, zorder=0,
                   label=r"Leuschner $v_{\rm LSR}$ window")
        ax.plot(v, T, color="C0", lw=LW_LIGHT, zorder=2)
        ax.fill_between(v, 0, T, where=((v >= v_lo) & (v <= v_hi)),
                        color="C0", alpha=ALPHA_LIGHT, zorder=1)
        ax.axhline(0, color="0.5", lw=LW_FINE, alpha=ALPHA_LIGHT)
        for vp, Tp in zip(ebhis_peak_v[name], ebhis_peak_tb[name]):
            ax.axvline(vp, color="C3", ls=":", lw=LW_FINE,
                       alpha=ALPHA_STANDARD, zorder=3)
            ax.scatter([vp], [Tp], color="C3", s=SS_STANDARD,
                       edgecolor="k", linewidth=LW_FINE, zorder=4)
        ax.set_xlabel(r"$v_{\rm LSR}$ (km/s)")
        ax.set_ylabel(r"EBHIS $T_B$ (K)")
        ax.set_xlim(-400, 400)

    ax = axes[1]
    for name in pointing_R_avg:
        ax.axvspan(v_lo, v_hi, color="C2", alpha=ALPHA_FAINT, zorder=0,
                   label=r"INT\_MASK ($v_{\rm LSR}$ window)")
        ax.axhline(0, color="0.5", lw=LW_FINE, alpha=ALPHA_LIGHT, zorder=1)
        R_c = pointing_R_avg[name]
        n_lo1, n_lo2 = pointing_R_N[name]
        ax.plot(v_lsr_axis, R_c, color="C0", lw=LW_LIGHT,
                alpha=ALPHA_STANDARD, zorder=2,
                label=rf"$R(c)$ (Stokes I, N={n_lo1}+{n_lo2})")
        for vp, Rp in zip(R_peak_v[name], R_peak_ref[name]):
            ax.axvline(vp, color="C3", ls=":", lw=LW_FINE,
                       alpha=ALPHA_STANDARD, zorder=3)
            ax.scatter([vp], [Rp], color="C3", s=SS_STANDARD,
                       edgecolor="k", linewidth=LW_FINE, zorder=4)
        ax.set_xlabel(r"$v_{\rm LSR}$ (km/s, LO1 mapping)")
        ax.set_ylabel(r"$R(c) = (P_{\rm LO1}-P_{\rm LO2})/P_{\rm LO2}$")
        ax.set_xlim(-400, 400)

    return fig, axes


def plot_ebhis_vs_leuschner_R_per_pol(
    ebhis_spectra,
    pointing_R_avg,
    pointing_R_N,
    ebhis_peak_v, ebhis_peak_tb,
    R_peak_v, R_peak_ref,
    v_lsr_axis,
    pointing_labels,
    *,
    v_lsr_window,
    n_visits_by_pointing,
    pols=(0, 1),
    width_in=None,
    row_height=None,
):
    """Stacked sanity check: EBHIS (top) + Leuschner R for each pol in ``pols``.

    Pass ``pols=(1,)`` for a column-width compact pol-1-only variant.
    """
    v_lo, v_hi = v_lsr_window
    nrows = 1 + len(pols)
    if width_in is None:
        width_in = TEXTWIDTH_IN * 0.6 if len(pols) == 2 else COLUMNWIDTH_IN
    if row_height is None:
        row_height = (TEXTWIDTH_IN * 1.2 / 3) if len(pols) == 2 \
            else (COLUMNWIDTH_IN * 0.45)
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(width_in, row_height * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    for name in ebhis_spectra:
        v, T = ebhis_spectra[name]
        ax.axvspan(v_lo, v_hi, color="C2", alpha=ALPHA_FAINT, zorder=0,
                   label=r"Leuschner $v_{\rm LSR}$ window")
        ax.plot(v, T, color="C0", lw=LW_LIGHT, zorder=2)
        ax.fill_between(v, 0, T, where=((v >= v_lo) & (v <= v_hi)),
                        color="C0", alpha=ALPHA_LIGHT, zorder=1)
        ax.axhline(0, color="0.5", lw=LW_FINE, alpha=ALPHA_LIGHT)
        for vp, Tp in zip(ebhis_peak_v[name], ebhis_peak_tb[name]):
            ax.axvline(vp, color="C3", ls=":", lw=LW_FINE,
                       alpha=ALPHA_STANDARD, zorder=3)
            ax.scatter([vp], [Tp], color="C3", s=SS_STANDARD,
                       edgecolor="k", linewidth=LW_FINE, zorder=4)
        ax.set_ylabel(r"EBHIS $T_B$ (K)")
        ax.set_xlim(-400, 400)

    for row, pi in enumerate(pols, start=1):
        ax = axes[row]
        ax.axvspan(v_lo, v_hi, color="C2", alpha=ALPHA_FAINT, zorder=0,
                   label=r"INT\_MASK ($v_{\rm LSR}$ window)")
        ax.axhline(0, color="0.5", lw=LW_FINE, alpha=ALPHA_LIGHT, zorder=1)
        for name in {k[0] for k in pointing_R_avg if k[1] == pi}:
            R_c = pointing_R_avg[(name, pi)]
            n_lo1, n_lo2 = pointing_R_N[(name, pi)]
            ax.plot(v_lsr_axis, R_c, color="C0", lw=LW_LIGHT,
                    alpha=ALPHA_STANDARD, zorder=2,
                    label=rf"$R_{{\rm pol\,{pi}}}(c)$ (N={n_lo1}+{n_lo2})")
            for vp, Rp in zip(R_peak_v[(name, pi)], R_peak_ref[(name, pi)]):
                ax.axvline(vp, color="C3", ls=":", lw=LW_FINE,
                           alpha=ALPHA_STANDARD, zorder=3)
                ax.scatter([vp], [Rp], color="C3", s=SS_STANDARD,
                           edgecolor="k", linewidth=LW_FINE, zorder=4)
        ax.set_ylabel(rf"$R_{{\rm pol\,{pi}}}(c)$")
        ax.set_xlim(-400, 400)
    axes[-1].set_xlabel(r"$v_{\rm LSR}$ (km/s)")
    return fig, axes


def plot_tcal_vs_time(
    tcal_df,
    pointing_labels,
    *,
    pol0_plot_max=10.0,
    alt_range=(17.0, 83.0),
    pols=(0, 1),
    width_in=None,
    height_ratio=10 / 16,
    title=None,
):
    """Tcal(t) scatter with LST/PDT time axes.  ``pols`` selects which
    polarisations to render as panels; pass ``pols=(1,)`` for a single-panel
    pol-1-only column-width variant.
    """
    pointings = sorted(tcal_df["target_id"].unique())
    markers = dict(zip(pointings, ["o", "^"]))
    alt_min, alt_max = alt_range
    cmap = plt.cm.viridis_r

    if width_in is None:
        width_in = TEXTWIDTH_IN if len(pols) == 2 else COLUMNWIDTH_IN
    fig = plt.figure(figsize=(width_in, width_in * height_ratio),
                     constrained_layout=True)
    if len(pols) == 1:
        axes = [fig.add_subplot(111)]
    else:
        axes = subpanels(fig, len(pols), sharex=True)

    def _plot_keep(df, pi):
        if pi == 0:
            return df[f"Tcal_pol{pi}"].fillna(np.inf) <= pol0_plot_max
        return np.ones(len(df), dtype=bool)

    sc = None
    for ax, pi in zip(axes, pols):
        for tid in pointings:
            sub = tcal_df[tcal_df["target_id"] == tid].dropna(
                subset=[f"Tcal_pol{pi}"])
            plot_sub = sub[_plot_keep(sub, pi)]
            sc = ax.scatter(plot_sub["t_mid"], plot_sub[f"Tcal_pol{pi}"],
                            c=plot_sub["alt"], cmap=cmap,
                            vmin=alt_min, vmax=alt_max,
                            marker=markers[tid], s=SS_STANDARD,
                            edgecolor="k", linewidth=LW_FINE, zorder=3)
        median = tcal_df[f"Tcal_pol{pi}"].median()
        ax.axhline(median, color=f"C{pi}", ls="--", lw=LW_FINE,
                   alpha=ALPHA_LIGHT,
                   label=rf"median = {median:.2f} K")
        ax.set_ylabel(rf"$T_{{\rm cal}}$ pol {pi} (K)")
        ax.legend(loc="upper right", frameon=True, fontsize="small")

    marker_handles = [
        Line2D([], [], marker=markers[tid], color="none",
               markerfacecolor="lightgray", markeredgecolor="k",
               markersize=8, label=pointing_labels[tid])
        for tid in pointings
    ]
    extra = []
    if 0 in pols:
        extra.append(
            Line2D([], [], color="C0", ls="--", lw=LW_FINE,
                   label=rf"pol 0 median = "
                         rf"{tcal_df['Tcal_pol0'].median():.2f} K")
        )
    axes[0].legend(
        handles=marker_handles + extra,
        loc="upper right", frameon=True, fontsize="small",
        title="pointing",
    )

    time_axes_lst_pdt(axes)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, location="right",
                            fraction=0.04, pad=0.02, shrink=0.9)
        cbar.set_label("altitude (deg)")
    return fig, axes


def plot_tcal_24h_fold(
    fold,
    n_harmonics,
    fourier_design_fn,
    pointing_label,
    *,
    pol0_plot_max=10.0,
    pols=(0, 1),
    width_in=None,
    height_ratio=10 / 16,
):
    """24 h PDT fold + Fourier-fit overlay.  ``pols`` selects panels; pass
    ``pols=(1,)`` for a single-panel pol-1-only column-width variant.
    """
    h_smooth = np.linspace(0.0, 24.0, 481)
    if width_in is None:
        width_in = TEXTWIDTH_IN if len(pols) == 2 else COLUMNWIDTH_IN
    fig = plt.figure(figsize=(width_in, width_in * height_ratio),
                     constrained_layout=True)
    if len(pols) == 1:
        axes = [fig.add_subplot(111)]
    else:
        axes = subpanels(fig, len(pols), sharex=True)

    for ax, pi in zip(axes, pols):
        d = fold.get(pi)
        if d is None:
            continue
        mask = (d["y"] <= pol0_plot_max) if pi == 0 \
            else np.ones_like(d["y"], dtype=bool)
        ax.scatter(d["h"][mask], d["y"][mask],
                   s=20, color=f"C{pi}", edgecolor="k",
                   linewidth=LW_FINE, alpha=ALPHA_STANDARD, zorder=2)
        y_fit = fourier_design_fn(h_smooth, d["K"]) @ d["coef"]
        ax.plot(h_smooth, y_fit, color="C3", lw=LW_LIGHT, zorder=3,
                label=rf"$K={d['K']}$ fit, "
                      rf"RMS = {d['rms']:.2f} K (dof = {d['dof']})")
        ax.axhline(d["coef"][0], color="0.3", ls="--", lw=LW_FINE,
                   alpha=ALPHA_LIGHT, zorder=1,
                   label=rf"$a_0={d['coef'][0]:.2f}$ K")
        ax.set_ylabel(rf"$T_{{\rm cal}}$ pol {pi} (K)")
        ax.legend(loc="upper right", fontsize="small", frameon=True)

    axes[-1].set_xlabel("hour of day (PDT)")
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    return fig, axes


# ---------------------------------------------------------------------------
# Report-bound helpers: rotation curve, masses, spiral overlay,
# top-down kinematic-distance projection.  These mirror the analysis blocks
# in galactic_plane_project.ipynb so the report figures match the notebook
# exactly, and read from the persisted artifacts/galactic_plane_products.pkl.
# ---------------------------------------------------------------------------

G_KPC_KMS2 = 4.30091e-6      # kpc (km/s)^2 / M_sun
M_H_G      = 1.6735575e-24   # g
M_SUN_G    = 1.98892e33      # g
KPC_CM     = 3.0857e21       # cm


def _V_brand_blitz(R, R_sun_kpc, V_sun_kms):
    a1, a2, a3 = 1.00767, 0.0394, 0.00712
    return V_sun_kms * (a1 * (R / R_sun_kpc) ** a2 + a3)


def plot_rotation_curve_burton(
    R_Q1, V_Q1,
    *,
    R_sun_kpc=8.5,
    V_sun_kms=220.0,
    L_TP_MIN_DEG=20.0,
    L_TP_MAX_DEG=65.0,
    FRAC_OF_PEAK=0.25,
    sigma_v_t_kms=4.0,
):
    """Inner-Galaxy rotation curve V(R_t) figure, matching the notebook."""
    R_lit = np.linspace(0.1, R_sun_kpc, 200)
    R_Q1 = np.asarray(R_Q1, dtype=float)
    V_Q1 = np.asarray(V_Q1, dtype=float)
    order = np.argsort(R_Q1)
    R_Q1, V_Q1 = R_Q1[order], V_Q1[order]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(COLUMNWIDTH_IN, COLUMNWIDTH_IN * 0.65),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
    )
    ax, ax_res = axes
    ax.plot(R_lit, _V_brand_blitz(R_lit, R_sun_kpc, V_sun_kms),
            color=NEUTRAL_COLOR, lw=LW_LIGHT, ls='--', alpha=ALPHA_FAINT,
            label=r'Brand \& Blitz 1993')
    sigma_V = np.full_like(V_Q1, float(sigma_v_t_kms))
    ax.fill_between(R_Q1, V_Q1 - sigma_V, V_Q1 + sigma_V,
                    color='C0', alpha=0.20, lw=0, zorder=2,
                    label=r'$\pm 1\sigma$ band')
    ax.plot(R_Q1, V_Q1, color='C0', lw=LW_LIGHT, zorder=3,
            label=r'measured, $0^\circ < \ell < 90^\circ$')
    ax.axhline(V_sun_kms, color='0.6', lw=LW_FINE, ls=':')
    ax.axvline(R_sun_kpc, color='0.6', lw=LW_FINE, ls=':')
    ax.set_ylabel(r'$V(R)$ [km/s]')
    ax.set_ylim(0, 320)
    ax.legend(loc='lower right', fontsize='small')

    V_ref = _V_brand_blitz(R_Q1, R_sun_kpc, V_sun_kms)
    residual = V_Q1 - V_ref
    ax_res.fill_between(R_Q1, residual - sigma_V, residual + sigma_V,
                        color='C0', alpha=0.20, lw=0, zorder=2)
    ax_res.plot(R_Q1, residual, color='C0', lw=LW_LIGHT, zorder=3)
    ax_res.axhline(0, color=NEUTRAL_COLOR, lw=LW_FINE, ls='--',
                   alpha=ALPHA_FAINT)
    ax_res.axvline(R_sun_kpc, color='0.6', lw=LW_FINE, ls=':')
    ax_res.set_xlabel(r'Galactocentric radius $R$ [kpc]')
    ax_res.set_ylabel(r'$\Delta V$ [km/s]')
    ax_res.set_xlim(0, R_sun_kpc)

    plt.tight_layout()
    return fig, axes


def plot_mass_grav(
    R_Q1, V_Q1,
    *,
    R_sun_kpc=8.5,
    V_sun_kms=220.0,
    sigma_v_t_kms=4.0,
):
    """Enclosed gravitational mass profile (tangent-point inversion)."""
    R_Q1 = np.asarray(R_Q1, dtype=float)
    V_Q1 = np.asarray(V_Q1, dtype=float)
    order = np.argsort(R_Q1)
    R_Q1, V_Q1 = R_Q1[order], V_Q1[order]
    M_grav_Q1 = V_Q1 ** 2 * R_Q1 / G_KPC_KMS2
    sigma_M = 2.0 * M_grav_Q1 * (float(sigma_v_t_kms) / V_Q1)
    R_lit = np.linspace(0.1, R_sun_kpc, 200)
    M_grav_lit = _V_brand_blitz(R_lit, R_sun_kpc, V_sun_kms) ** 2 * R_lit / G_KPC_KMS2
    M_sun_anchor = V_sun_kms ** 2 * R_sun_kpc / G_KPC_KMS2

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH_IN, COLUMNWIDTH_IN * 0.7))
    ax.plot(R_lit, M_grav_lit / 1e10,
            color=NEUTRAL_COLOR, lw=LW_LIGHT, ls='--', alpha=ALPHA_FAINT,
            label=r'Brand \& Blitz 1993')
    ax.fill_between(R_Q1,
                    (M_grav_Q1 - sigma_M) / 1e10,
                    (M_grav_Q1 + sigma_M) / 1e10,
                    color='C0', alpha=0.20, lw=0, zorder=2,
                    label=r'$\pm 1\sigma$ band')
    ax.plot(R_Q1, M_grav_Q1 / 1e10, color='C0', lw=LW_LIGHT, zorder=3,
            label=r'measured, $0^\circ < \ell < 90^\circ$')
    ax.axhline(M_sun_anchor / 1e10, color='0.6', lw=LW_FINE, ls=':')
    ax.axvline(R_sun_kpc, color='0.6', lw=LW_FINE, ls=':')
    ax.set_xlabel(r'Galactocentric radius $R$ [kpc]')
    ax.set_ylabel(r'$M_{\rm grav}(R)$ [$10^{10}\,M_\odot$]')
    ax.set_xlim(0, R_sun_kpc)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    return fig, ax


def plot_mass_gas(
    R_bin_centres, M_gas_near_bin, M_gas_far_bin,
    *,
    R_sun_kpc=8.5,
):
    """Cumulative HI gas mass with near and far branches of the KDA."""
    R = np.asarray(R_bin_centres, dtype=float)
    M_near = np.cumsum(np.asarray(M_gas_near_bin, dtype=float))
    M_far  = np.cumsum(np.asarray(M_gas_far_bin,  dtype=float))

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH_IN, COLUMNWIDTH_IN * 0.7))
    pos_n = M_near > 0
    pos_f = M_far > 0
    ax.semilogy(R[pos_n], M_near[pos_n] / 1e9, color='C0', lw=LW_LIGHT,
                label=r'$M_{\rm gas}(<R)$, near branch')
    ax.semilogy(R[pos_f], M_far[pos_f] / 1e9, color='C3', lw=LW_LIGHT,
                ls='-.', label=r'$M_{\rm gas}(<R)$, far branch')
    ax.set_xlabel(r'Galactocentric radius $R$ [kpc]')
    ax.set_ylabel(r'Enclosed gas mass [$10^{9}\,M_\odot$]')
    ax.set_xlim(0, R_sun_kpc)
    ax.axvline(R_sun_kpc, color='0.6', lw=LW_FINE, ls=':')
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Internal: full-disk V(R) interpolator + sightline inversion.
# Both reused by the spiral overlay and the top-down kinematic projection.
# ---------------------------------------------------------------------------

def _full_disk_V(R_Q1, V_Q1, R_sun_kpc, V_sun_kms):
    """Build a callable V(R) that is the measured inner curve up to R_sun
    and flat outside, plus the inner-edge clip radius."""
    R_Q1 = np.asarray(R_Q1, dtype=float)
    V_Q1 = np.asarray(V_Q1, dtype=float)
    order = np.argsort(R_Q1)
    R_Q1, V_Q1 = R_Q1[order], V_Q1[order]
    R_in = np.append(R_Q1, R_sun_kpc)
    V_in = np.append(V_Q1, V_sun_kms)
    R_invert_min = float(R_Q1.min())

    def V_full(R):
        R = np.atleast_1d(np.asarray(R, dtype=float))
        Rc = np.clip(R, R_invert_min, R_sun_kpc)
        V_in_inner = np.interp(Rc, R_in, V_in)
        return np.where(R >= R_sun_kpc, V_sun_kms, V_in_inner)

    return V_full, R_invert_min


def _find_ridge_peaks(l_fine, v_lsr, lv_image,
                      *, V_BAND_MAX_KMS=130.0,
                      L_SAMPLE_STEP_DEG=2.0,
                      PEAK_MIN_SEP_KMS=5.0,
                      PEAK_HEIGHT_FRAC=0.10,
                      PEAK_MIN_PROMINENCE_K=0.5):
    """Find every isolated peak in T_B(v_LSR) at each sampled longitude.

    Returns (l, v, T) arrays of equal length.
    """
    from scipy.signal import find_peaks
    dv = float(v_lsr[1] - v_lsr[0])
    peak_min_sep_chan = max(int(round(PEAK_MIN_SEP_KMS / abs(dv))), 1)
    l_grid = np.arange(np.floor(l_fine.min()),
                       np.ceil(l_fine.max()) + 1e-9,
                       L_SAMPLE_STEP_DEG)
    out_l, out_v, out_T = [], [], []
    for l_deg in l_grid:
        j = int(np.argmin(np.abs(l_fine - l_deg)))
        spec = lv_image[:, j].astype(float)
        edge_ok = np.abs(v_lsr) < V_BAND_MAX_KMS
        spec_use = np.where(edge_ok & np.isfinite(spec), spec, -np.inf)
        sig_max = float(np.nanmax(spec_use))
        if not np.isfinite(sig_max) or sig_max <= 0:
            continue
        pk_idx, _ = find_peaks(spec_use,
                               height=PEAK_HEIGHT_FRAC * sig_max,
                               prominence=PEAK_MIN_PROMINENCE_K,
                               distance=peak_min_sep_chan)
        for i in pk_idx:
            out_l.append(float(l_deg))
            out_v.append(float(v_lsr[i]))
            out_T.append(float(spec_use[i]))
    return np.array(out_l), np.array(out_v), np.array(out_T)


def _lv_to_Rphi(l_deg, v_obs, V_full, R_invert_min,
                R_sun_kpc, V_sun_kms,
                R_TRACE_MAX_KPC=18.0, SIN_L_FLOOR=0.02):
    """Invert (l, v_LSR) -> (R, phi); near branch inside R_sun, far branch outside."""
    l_rad = np.deg2rad(l_deg)
    sin_l, cos_l = np.sin(l_rad), np.cos(l_rad)
    if abs(sin_l) < SIN_L_FLOOR:
        return np.nan, np.nan
    R_fine = np.linspace(R_invert_min, R_TRACE_MAX_KPC, 2000)
    v_at_R = (V_full(R_fine) / R_fine - V_sun_kms / R_sun_kpc) \
        * R_sun_kpc * sin_l
    order = np.argsort(v_at_R)
    v_sorted, R_sorted = v_at_R[order], R_fine[order]
    if v_obs < v_sorted[0] or v_obs > v_sorted[-1]:
        return np.nan, np.nan
    R = float(np.interp(v_obs, v_sorted, R_sorted))
    disc = R ** 2 - R_sun_kpc ** 2 * sin_l ** 2
    if disc < 0:
        return np.nan, np.nan
    sqrt_disc = float(np.sqrt(disc))
    if R >= R_sun_kpc:
        d = R_sun_kpc * cos_l + sqrt_disc
    else:
        d = R_sun_kpc * cos_l - sqrt_disc
        if d <= 0:
            d = R_sun_kpc * cos_l + sqrt_disc
    if d <= 0:
        return np.nan, np.nan
    x_gc = R_sun_kpc - d * cos_l
    y_gc = d * sin_l
    return R, float(np.arctan2(y_gc, x_gc))


def _kmeansK_lnRphi(R, phi, K=3, n_iter=300, seed=0):
    """K-means in standardised (ln R, phi); returns labels 0..K-1 sorted by
    increasing mean R (label 0 = smallest mean R).
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.log(R), phi])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    # Initialise by partitioning ln R into K equal-population quantile bins.
    qs = np.quantile(Xn[:, 0], np.linspace(0, 1, K + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    labels = np.clip(np.digitize(Xn[:, 0], qs) - 1, 0, K - 1)
    for _ in range(n_iter):
        centres = np.array([
            Xn[labels == k].mean(axis=0) if (labels == k).any()
            else Xn[rng.integers(0, len(Xn))]
            for k in range(K)
        ])
        dists = ((Xn[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    # Relabel so cluster 0 has smallest mean R, ..., K-1 largest.
    mean_R = np.array([R[labels == k].mean() if (labels == k).any() else np.inf
                       for k in range(K)])
    order = np.argsort(mean_R)
    remap = np.empty(K, dtype=int)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    return remap[labels]


def _kmeans2_lnRphi(R, phi, **kw):
    """Back-compat alias: 2-means."""
    return _kmeansK_lnRphi(R, phi, K=2, **kw)


def _kmeans2_lnRphi_legacy(R, phi, n_iter=200):
    """Original 2-means used in early notebooks; kept for reference."""
    X = np.column_stack([np.log(R), phi])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    labels = (Xn[:, 0] >= np.median(Xn[:, 0])).astype(int)
    for _ in range(n_iter):
        c0 = Xn[labels == 0].mean(axis=0) if (labels == 0).any() else np.zeros(2)
        c1 = Xn[labels == 1].mean(axis=0) if (labels == 1).any() else np.zeros(2)
        d0 = ((Xn - c0) ** 2).sum(axis=1)
        d1 = ((Xn - c1) ** 2).sum(axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    mean_R0 = R[labels == 0].mean() if (labels == 0).any() else np.inf
    mean_R1 = R[labels == 1].mean() if (labels == 1).any() else 0.0
    if mean_R0 > mean_R1:
        labels = 1 - labels
    return labels


def _fit_log_spiral(R, phi):
    if R.size < 2:
        return None
    slope, intercept = np.polyfit(phi, np.log(R), 1)
    return dict(kappa=float(slope), R0=float(np.exp(intercept)),
                pitch_deg=float(np.rad2deg(np.arctan(slope))))


def _arm_to_lv(fit, V_full, R_sun_kpc, V_sun_kms, l_fine, v_lsr,
               lv_data_mask, R_TRACE_MAX_KPC=18.0,
               L_JUMP_BREAK_DEG=30.0):
    phi = np.linspace(-5 * np.pi, 5 * np.pi, 8000)
    R = fit["R0"] * np.exp(fit["kappa"] * phi)
    ok = (R > 0.2) & (R < R_TRACE_MAX_KPC)
    R, phi = R[ok], phi[ok]
    if R.size == 0:
        return np.array([]), np.array([])
    x_gc = R * np.cos(phi); y_gc = R * np.sin(phi)
    x_h = R_sun_kpc - x_gc; y_h = y_gc
    l_rad = np.arctan2(y_h, x_h)
    l_deg = (np.rad2deg(l_rad) + 180.0) % 360.0 - 180.0
    v_arm = (V_full(R) / R - V_sun_kms / R_sun_kpc) * R_sun_kpc * np.sin(l_rad)
    L_MIN = float(l_fine.min()); L_MAX = float(l_fine.max())
    V_MIN = float(v_lsr.min());  V_MAX = float(v_lsr.max())
    V_LSR_ASC = v_lsr[0] < v_lsr[-1]
    l_idx = np.searchsorted(l_fine, l_deg).clip(0, lv_data_mask.shape[1] - 1)
    if V_LSR_ASC:
        v_idx = np.searchsorted(v_lsr, v_arm).clip(0, lv_data_mask.shape[0] - 1)
    else:
        v_idx = np.searchsorted(-v_lsr, -v_arm).clip(0, lv_data_mask.shape[0] - 1)
    in_box = ((l_deg >= L_MIN) & (l_deg <= L_MAX)
              & (v_arm >= V_MIN) & (v_arm <= V_MAX)
              & lv_data_mask[v_idx, l_idx])
    l_out = np.where(in_box, l_deg, np.nan)
    v_out = np.where(in_box, v_arm, np.nan)
    dl = np.zeros_like(l_out); dl[1:] = np.abs(np.diff(l_out))
    big = dl > L_JUMP_BREAK_DEG
    l_out[big] = np.nan; v_out[big] = np.nan
    return l_out, v_out


# High-contrast colour cycle used for K-means clusters on inferno backgrounds.
ARM_COLORS  = ["#00ffff", "#39ff14", "#ff44ff", "#ffff33"]   # cyan, green, magenta, yellow
ARM_MARKERS = ["o", "s", "D", "^"]
ARM_LABELS  = ["A", "B", "C", "D", "E"]


def plot_spiral_lv_overlay(
    l_fine, v_lsr, lv_image,
    R_Q1, V_Q1,
    *,
    R_sun_kpc=8.5,
    V_sun_kms=220.0,
    K=4,
):
    """K-means K-arm log-spiral fit overlaid on T_B(l, v_LSR)."""
    V_full, R_invert_min = _full_disk_V(R_Q1, V_Q1, R_sun_kpc, V_sun_kms)
    pt_l_raw, pt_v_raw, _ = _find_ridge_peaks(l_fine, v_lsr, lv_image)
    pt_l, pt_v, pt_R, pt_phi = [], [], [], []
    for li, vi in zip(pt_l_raw, pt_v_raw):
        R, phi = _lv_to_Rphi(li, vi, V_full, R_invert_min,
                             R_sun_kpc, V_sun_kms)
        if np.isfinite(R) and np.isfinite(phi):
            pt_l.append(li); pt_v.append(vi)
            pt_R.append(R); pt_phi.append(phi)
    pt_l   = np.array(pt_l);   pt_v   = np.array(pt_v)
    pt_R   = np.array(pt_R);   pt_phi = np.array(pt_phi)

    labels = _kmeansK_lnRphi(pt_R, pt_phi, K=K)
    masks = [(labels == k) for k in range(K)]
    fits  = [_fit_log_spiral(pt_R[m], pt_phi[m]) for m in masks]

    lv_data_mask = np.isfinite(lv_image) & (lv_image != 0)
    arms_lv = [
        _arm_to_lv(fit, V_full, R_sun_kpc, V_sun_kms, l_fine, v_lsr,
                   lv_data_mask)
        for fit in fits
    ]

    fig, ax = plot_lv_strip(
        l_fine, v_lsr, lv_image,
        title=None,
        cbar_label=r"$T_B$ [K]",
    )
    ax.axhline(0.0, color="w", lw=LW_FINE, alpha=ALPHA_FAINT, zorder=4)
    for k, (mask, fit, (l_arm, v_arm)) in enumerate(zip(masks, fits, arms_lv)):
        color = ARM_COLORS[k % len(ARM_COLORS)]
        marker = ARM_MARKERS[k % len(ARM_MARKERS)]
        label = ARM_LABELS[k]
        ax.scatter(pt_l[mask], pt_v[mask], s=SS_FINE, color=color,
                   edgecolor="k", linewidth=LW_FINE, marker=marker, zorder=5,
                   label=f"cluster {label} (N = {mask.sum()})")
        if fit is None:
            continue
        ax.plot(l_arm, v_arm, lw=LW_STANDARD, color=color,
                alpha=ALPHA_FULL, zorder=4,
                label=rf"arm {label} (pitch $= {fit['pitch_deg']:+.1f}^\circ$)")
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    return fig, ax, fits


def plot_kinematic_distance(
    l_fine, v_lsr, lv_image,
    R_Q1, V_Q1,
    fits=None,
    *,
    R_sun_kpc=8.5,
    V_sun_kms=220.0,
    XY_MAX_KPC=16.0,
    XY_STEP_KPC=0.2,
    L_PROJ_STEP_DEG=1.0,
    GAUSSIAN_SIGMA=2.0,
    V_BAND_MAX_KMS=130.0,
):
    """Top-down kinematic-distance projection of T_B(l, v) (near branch)."""
    from scipy.ndimage import gaussian_filter

    V_full, R_invert_min = _full_disk_V(R_Q1, V_Q1, R_sun_kpc, V_sun_kms)
    R_TRACE_MAX_KPC = 18.0
    SIN_L_FLOOR = 0.02

    n_xy = int(round(2 * XY_MAX_KPC / XY_STEP_KPC))
    xy_edges = np.linspace(-XY_MAX_KPC, XY_MAX_KPC, n_xy + 1)
    H_T = np.zeros((n_xy, n_xy))
    H_N = np.zeros((n_xy, n_xy))

    lv_data_mask = np.isfinite(lv_image) & (lv_image != 0)
    if lv_data_mask.any():
        l_data_cols = lv_data_mask.any(axis=0)
        L_MIN_SURVEY = float(l_fine[l_data_cols].min())
        L_MAX_SURVEY = float(l_fine[l_data_cols].max())
    else:
        L_MIN_SURVEY = float(l_fine.min()); L_MAX_SURVEY = float(l_fine.max())

    l_proj_grid = np.arange(L_MIN_SURVEY, L_MAX_SURVEY + 1e-9, L_PROJ_STEP_DEG)
    R_inv_grid  = np.linspace(R_invert_min, R_TRACE_MAX_KPC, 2000)

    for l_deg in l_proj_grid:
        l_rad = np.deg2rad(l_deg)
        sin_l, cos_l = np.sin(l_rad), np.cos(l_rad)
        if abs(sin_l) < SIN_L_FLOOR:
            continue
        j = int(np.argmin(np.abs(l_fine - l_deg)))
        spec = lv_image[:, j].astype(float)
        valid = (np.abs(v_lsr) < V_BAND_MAX_KMS) & np.isfinite(spec) & (spec > 0)
        if not valid.any():
            continue
        v_use = v_lsr[valid]
        T_use = spec[valid]
        v_at_R = (V_full(R_inv_grid) / R_inv_grid - V_sun_kms / R_sun_kpc) \
            * R_sun_kpc * sin_l
        order = np.argsort(v_at_R)
        R_sol = np.interp(v_use, v_at_R[order], R_inv_grid[order],
                          left=np.nan, right=np.nan)
        keep = np.isfinite(R_sol)
        R_sol = R_sol[keep]; T_use = T_use[keep]
        disc = R_sol ** 2 - R_sun_kpc ** 2 * sin_l ** 2
        keep = disc >= 0
        R_sol = R_sol[keep]; T_use = T_use[keep]; disc = disc[keep]
        sqrt_disc = np.sqrt(disc)
        d_near = R_sun_kpc * cos_l - sqrt_disc
        d_far  = R_sun_kpc * cos_l + sqrt_disc
        d = np.where(R_sol >= R_sun_kpc, d_far, d_near)
        pos = d > 0
        R_sol = R_sol[pos]; d = d[pos]; T_use = T_use[pos]
        x_gc = R_sun_kpc - d * cos_l
        y_gc = d * sin_l
        ix = np.digitize(x_gc, xy_edges) - 1
        iy = np.digitize(y_gc, xy_edges) - 1
        in_bins = (ix >= 0) & (ix < n_xy) & (iy >= 0) & (iy < n_xy)
        np.add.at(H_T, (iy[in_bins], ix[in_bins]), T_use[in_bins])
        np.add.at(H_N, (iy[in_bins], ix[in_bins]), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        H_mean = np.where(H_N > 0, H_T / H_N, 0.0)
    H_smooth = gaussian_filter(H_mean, sigma=GAUSSIAN_SIGMA)
    H_display = np.where(H_smooth > 0.01, H_smooth, np.nan)

    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85))
    im = ax.imshow(H_display, origin="lower",
                   extent=[-XY_MAX_KPC, XY_MAX_KPC, -XY_MAX_KPC, XY_MAX_KPC],
                   cmap="inferno", aspect="equal")
    plt.colorbar(im, ax=ax, label=r"smoothed $\langle T_B\rangle$ [K]",
                 fraction=0.035, pad=0.02, shrink=0.55)

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_sun_kpc * np.cos(theta), R_sun_kpc * np.sin(theta),
            color="w", lw=LW_FINE, ls=":", alpha=ALPHA_FAINT, zorder=4)
    ax.plot(R_sun_kpc, 0, marker="*", markersize=12, color="white",
            markeredgecolor="k", markeredgewidth=0.7, zorder=6, label="Sun")
    ax.plot(0, 0, marker="+", markersize=12, color="white", mew=2, zorder=6,
            label="GC")

    # Project the never-observable galactic-longitude band into a wedge of
    # unreachable Galactocentric positions, anchored at the Sun. A ray from
    # the Sun at galactic longitude l points along (-cos l, sin l) in (x_gc,
    # y_gc), so its angle from +x_gc is (180 - l) deg.
    l_wedge = np.arange(0.0, 360.0, 0.5)
    inacc_b0 = never_observable_mask(l_wedge, np.zeros_like(l_wedge))
    if inacc_b0.any():
        padded = np.r_[False, inacc_b0, False]
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0] - 1
        R_wedge = 2.5 * XY_MAX_KPC
        wedge_label_done = False
        with mpl.rc_context({"hatch.color": "red", "hatch.linewidth": 0.5}):
            for i0, i1 in zip(starts, ends):
                l_a = float(l_wedge[i0]); l_b = float(l_wedge[i1])
                theta1 = 180.0 - l_b
                theta2 = 180.0 - l_a
                label = None if wedge_label_done else "never observed"
                wedge_label_done = True
                ax.add_patch(mpatches.Wedge(
                    (R_sun_kpc, 0.0), R_wedge, theta1, theta2,
                    facecolor="none", edgecolor="red", linewidth=0.5,
                    hatch="///", alpha=0.4, zorder=4, label=label))

    phi_plot = np.linspace(-3 * np.pi, 3 * np.pi, 4000)
    for k, fit in enumerate(fits or []):
        if fit is None:
            continue
        color = ARM_COLORS[k % len(ARM_COLORS)]
        label = ARM_LABELS[k]
        R = fit["R0"] * np.exp(fit["kappa"] * phi_plot)
        ok = (R > 0.5) & (R < R_TRACE_MAX_KPC)
        Rk, pk = R[ok], phi_plot[ok]
        ax.plot(Rk * np.cos(pk), Rk * np.sin(pk),
                color=color, lw=LW_STANDARD, alpha=ALPHA_FULL, zorder=5,
                label=rf"arm {label} (pitch $= {fit['pitch_deg']:+.1f}^\circ$)")

    ax.set_xlim(-XY_MAX_KPC, XY_MAX_KPC)
    ax.set_ylim(-XY_MAX_KPC, XY_MAX_KPC)
    ax.set_xlabel(r"$x_{\rm gc}$ [kpc]")
    ax.set_ylabel(r"$y_{\rm gc}$ [kpc]")
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    return fig, ax

