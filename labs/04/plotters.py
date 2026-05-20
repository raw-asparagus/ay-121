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
    SS_FINE,
    ALPHA_FAINT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    GRID_STYLE,
    GUIDE_STYLE,
    SCATTER_STYLE,
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
        ax.legend(loc="upper right", fontsize=LEGEND_SIZE, framealpha=0.9)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    plt.tight_layout()
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
