"""Velocity-integrated mapping and gridding utilities."""

from __future__ import annotations

import numpy as np


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


