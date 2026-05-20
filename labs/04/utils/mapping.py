"""Velocity-integrated mapping and gridding utilities."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ugradiolab.astronomy import LEO_LAT_DEG, LEO_LON_DEG


# ---------------------------------------------------------------------------
# Galactic-plane survey grid
# ---------------------------------------------------------------------------

def _build_l_row(
    b_deg: float,
    *,
    l_center: float,
    l_min: float,
    l_max: float,
    physical_spacing_deg: float,
) -> list[float]:
    """Non-integer longitude grid at latitude b, anchored at l_center.

    Walks outward from l_center in steps of physical_spacing_deg / cos(b),
    keeping only values inside [l_min, l_max].  When l_center sits outside
    the survey window the filter trims the row.
    """
    cos_b = math.cos(math.radians(b_deg))
    if cos_b <= 0:
        return [l_center] if l_min <= l_center <= l_max else []
    dl = physical_spacing_deg / cos_b
    l_vals = [round(l_center, 2)]
    l = l_center + dl
    while l <= l_max:
        l_vals.append(round(l, 2))
        l += dl
    l = l_center - dl
    while l >= l_min:
        l_vals.append(round(l, 2))
        l -= dl
    return sorted(v for v in l_vals if l_min <= v <= l_max)


def build_galplane_grid(
    *,
    b_min: int,
    b_max: int,
    b_step: int,
    l_min: float,
    l_max: float,
    l_center: float,
    physical_spacing_deg: float,
    phase: str = 'even',
    verbose: bool = False,
) -> list[tuple[int, int, float, int]]:
    """Column-major brick grid: contiguous l columns swept through b, zig-zagged.

    Returns ``(col_idx, row_idx, l, b)`` tuples in scan order.  Even phase
    uses b at multiples of ``b_step`` from ``b_min``; odd phase shifts by
    one b step and offsets l by half a step (hexagonal interleave).
    """
    if phase == 'even':
        b_vals = list(range(b_min, b_max + 1, b_step))
    else:
        b_vals = list(range(b_min + 1, b_max, b_step))

    all_cells: list[tuple[float, int]] = []
    for b in b_vals:
        if phase == 'odd':
            half_step = physical_spacing_deg / (2 * math.cos(math.radians(b)))
            row_center = l_center + half_step
        else:
            row_center = l_center
        l_vals = _build_l_row(
            b,
            l_center=row_center,
            l_min=l_min,
            l_max=l_max,
            physical_spacing_deg=physical_spacing_deg,
        )
        if not l_vals:
            if verbose:
                print(f'  b={b:+3d}: 0 cells (filtered out of range)')
            continue
        if verbose:
            dl = physical_spacing_deg / math.cos(math.radians(b))
            print(f'  b={b:+3d}: Delta_l={dl:.2f} deg, {len(l_vals)} cells, '
                  f'l=[{l_vals[0]:.1f}, {l_vals[-1]:.1f}]')
        for l in l_vals:
            all_cells.append((l, b))

    if not all_cells:
        return []

    all_cells.sort(key=lambda c: c[0])
    col_tol = physical_spacing_deg / 2
    columns: list[list[tuple[float, int]]] = [[all_cells[0]]]
    for cell in all_cells[1:]:
        if cell[0] - columns[-1][0][0] <= col_tol:
            columns[-1].append(cell)
        else:
            columns.append([cell])

    cells: list[tuple[int, int, float, int]] = []
    for col_idx, col in enumerate(columns):
        col_sorted = sorted(col, key=lambda c: c[1])
        if col_idx % 2 == 1:
            col_sorted = list(reversed(col_sorted))
        for row_idx, (l, b) in enumerate(col_sorted):
            cells.append((col_idx, row_idx, l, b))

    if verbose:
        print(f'  Column-major: {len(columns)} columns, {len(cells)} cells total')
    return cells


# ---------------------------------------------------------------------------
# Closed-form ICRS -> Leuschner alt/az (shared between planner + notebooks)
# ---------------------------------------------------------------------------

_LAT_R = math.radians(LEO_LAT_DEG)
_SIN_LAT = math.sin(_LAT_R)
_COS_LAT = math.cos(_LAT_R)


@lru_cache(maxsize=None)
def cell_radec(l_deg: float, b_deg: float) -> tuple[float, float]:
    """Cached ICRS (ra, dec) in degrees for galactic (l, b)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    gc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame='galactic')
    return float(gc.icrs.ra.deg), float(gc.icrs.dec.deg)


def gmst_hours(unix_t: float) -> float:
    """Approximate GMST in hours from unix time (accurate to ~1 s)."""
    jd = unix_t / 86400.0 + 2440587.5
    d = jd - 2451545.0
    return (18.697374558 + 24.06570982441908 * d) % 24.0


def fast_altaz(ra_deg: float, dec_deg: float, unix_t: float) -> tuple[float, float]:
    """Closed-form alt/az for Leuschner; matches astropy under a degree."""
    lst_deg = (gmst_hours(unix_t) * 15.0 + LEO_LON_DEG) % 360.0
    ha_deg = ((lst_deg - ra_deg + 180.0) % 360.0) - 180.0
    ha = math.radians(ha_deg)
    dec = math.radians(dec_deg)
    cos_dec = math.cos(dec)
    sin_dec = math.sin(dec)
    sin_alt = _SIN_LAT * sin_dec + _COS_LAT * cos_dec * math.cos(ha)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.degrees(math.asin(sin_alt))
    cos_alt_sq = 1.0 - sin_alt * sin_alt
    if cos_alt_sq <= 1e-18:
        return alt, 0.0
    cos_alt = math.sqrt(cos_alt_sq)
    sin_az = -cos_dec * math.sin(ha) / cos_alt
    cos_az = (sin_dec - _SIN_LAT * sin_alt) / (_COS_LAT * cos_alt)
    az = (math.degrees(math.atan2(sin_az, cos_az)) + 360.0) % 360.0
    return alt, az


def assemble_W_R_arrays(
    cell_combined: dict,
    dv_kms: float,
    *,
    spectrum_key: str,
) -> dict:
    """Velocity-integrate one spectrum key per cell into flat arrays.

    ``W_R = nansum(R) * dv_kms`` per cell.  Operates on a single spectrum
    key (one pol or a caller-prepared summed key) -- call twice for
    per-pol output.

    Returns
    -------
    dict
        ``{'gl', 'gb', 'W_R', 'valid', 'n_sci'}`` -- the first three are
        1-D arrays.
    """
    keys = list(cell_combined)
    gl = np.fromiter((k[0] for k in keys), dtype=float, count=len(keys))
    gb = np.fromiter((k[1] for k in keys), dtype=float, count=len(keys))
    W_R = np.fromiter(
        (np.nansum(cell_combined[k][spectrum_key]) * dv_kms for k in keys),
        dtype=float, count=len(keys),
    )
    return {
        'gl': gl, 'gb': gb, 'W_R': W_R,
        'valid': np.isfinite(W_R),
        'n_sci': int(np.isfinite(W_R).sum()),
    }


def compute_lv_strip(
    cell_combined: dict,
    excluded: set,
    *,
    b_max_deg: float,
    dl_fine_deg: float,
    hpbw_deg: float,
    cutoff_hpbw: float = 2.0,
    keep_near_hpbw: float = 1.0,
    min_weight: float = 0.1,
    spectrum_key: str,
) -> dict:
    """Beam-weighted (l, v) resampling of the b ~ 0 strip.

    For each cell in ``cell_combined`` with ``|b| <= b_max_deg`` and not in
    ``excluded``, accumulate its spectrum onto a fine longitude grid with
    Gaussian beam weights (sigma = HPBW / 2.355) and a hard cutoff at
    ``cutoff_hpbw * hpbw_deg``. Pixels with no cell within
    ``keep_near_hpbw * hpbw_deg`` are masked, which blanks unobserved
    longitudes without explicit segment bookkeeping.

    Returns
    -------
    dict
        ``{'l_fine', 'lv_image', 'sigma_deg', 'n_cells', 'n_populated'}``.
        ``lv_image`` is shape ``(nv, M)`` with NaNs where weight is below
        ``min_weight`` or no cell is within ``keep_near_hpbw * hpbw_deg``.
    """
    keys = [k for k in cell_combined
            if abs(k[1]) <= b_max_deg and k not in excluded]
    if not keys:
        return {'l_fine': np.array([]), 'lv_image': np.zeros((0, 0)),
                'sigma_deg': hpbw_deg / (2.0 * np.sqrt(2.0 * np.log(2.0))),
                'n_cells': 0, 'n_populated': 0}

    gl = np.array([k[0] for k in keys])
    gb = np.array([k[1] for k in keys])
    spec = np.array([cell_combined[k][spectrum_key] for k in keys])  # (N, nv)
    # Wrap into (-90, 270] so the survey footprint (l = -10 .. 250) sits
    # contiguously inside the standard galactic-plane plot range.
    gl_w = ((gl + 90.0) % 360.0) - 90.0

    l_lo = -90.0
    l_hi = 270.0
    l_fine = np.arange(l_lo, l_hi + dl_fine_deg, dl_fine_deg)

    sigma_deg = hpbw_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_rad = np.deg2rad(sigma_deg)
    cutoff_rad = np.deg2rad(cutoff_hpbw * hpbw_deg)
    near_rad = np.deg2rad(keep_near_hpbw * hpbw_deg)

    G_l = np.deg2rad(l_fine)[:, None]
    P_l = np.deg2rad(gl_w)[None, :]
    P_b = np.deg2rad(gb)[None, :]
    cosang = np.clip(np.cos(P_b) * np.cos(G_l - P_l), -1.0, 1.0)
    theta = np.arccos(cosang)
    w = np.exp(-0.5 * (theta / sigma_rad) ** 2)
    w[theta > cutoff_rad] = 0.0

    finite = np.isfinite(spec)
    spec_safe = np.where(finite, spec, 0.0)
    w_S = w @ finite
    with np.errstate(invalid='ignore', divide='ignore'):
        lv_image = np.where(w_S > min_weight, (w @ spec_safe) / w_S, np.nan).T

    no_near = theta.min(axis=1) > near_rad
    lv_image[:, no_near] = np.nan

    return {
        'l_fine': l_fine,
        'lv_image': lv_image,
        'sigma_deg': sigma_deg,
        'n_cells': len(keys),
        'n_populated': np.isfinite(lv_image).any(axis=0).sum(),
    }


