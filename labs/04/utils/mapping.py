"""Velocity-integrated mapping and gridding utilities."""

from __future__ import annotations

import numpy as np


def compute_cell_W(
    results_dict: dict,
    dv_kms: float,
    *,
    spectrum_key: str = 'R',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocity-integrated spectrum for each cell.

    NaN channels (DC mask, RFI) are filled by linear interpolation from
    neighboring valid channels before integration.

    Parameters
    ----------
    results_dict : dict
        Mapping of ``(l, b)`` or ``(dr, l, b)`` to result dicts
        containing a spectrum at ``spectrum_key``.
    dv_kms : float
        Channel width in km/s.
    spectrum_key : str
        Key holding the spectrum to integrate (default ``'R'``).

    Returns
    -------
    gl, gb, vals : 1-D float arrays
        Galactic coordinates and integrated values.
    """
    gl, gb, vals = [], [], []
    for key, cr in results_dict.items():
        if len(key) == 2:
            l, b = key
        else:
            _, l, b = key
        gl.append(l)
        gb.append(b)
        R_ov = cr[spectrum_key].copy()
        valid = np.isfinite(R_ov)
        if valid.sum() >= 2:
            channels = np.arange(len(R_ov))
            R_ov[~valid] = np.interp(
                channels[~valid], channels[valid], R_ov[valid],
            )
            W = np.sum(R_ov) * dv_kms
        elif valid.sum() == 1:
            W = R_ov[valid][0] * dv_kms * len(R_ov)
        else:
            W = np.nan
        vals.append(W)
    return np.array(gl), np.array(gb), np.array(vals)


def assemble_W_R_arrays(
    cell_combined: dict,
    recal_cells: dict,
    qa_flagged_set: set,
    dv_kms: float,
) -> dict:
    """Stack science + recal cells into flat arrays for Mollweide plotting.

    Science cells in ``qa_flagged_set`` are excluded; recal cells (already
    aggregated across visits and sessions) are appended after the science
    cells.  ``W_R = sum(R) * dv_kms`` (NaNs propagate) for science cells;
    recal cells carry a pre-computed ``W_R``.

    Returns
    -------
    dict
        ``{'gl', 'gb', 'W_R', 'valid', 'n_sci', 'n_recal'}`` -- the first
        three are 1-D arrays with science cells followed by recal cells.
    """
    clean_keys = [k for k in cell_combined if k not in qa_flagged_set]
    gl_sci = np.array([k[0] for k in clean_keys])
    gb_sci = np.array([k[1] for k in clean_keys])
    R_stack = np.array([cell_combined[k]['R'] for k in clean_keys])
    W_R_sci = np.nansum(R_stack, axis=1) * dv_kms

    gl_rec = np.array([c['gl'] for c in recal_cells.values()])
    gb_rec = np.array([c['gb'] for c in recal_cells.values()])
    W_R_rec = np.array([c['W_R'] for c in recal_cells.values()])

    gl = np.concatenate([gl_sci, gl_rec])
    gb = np.concatenate([gb_sci, gb_rec])
    W_R = np.concatenate([W_R_sci, W_R_rec])
    return {
        'gl': gl, 'gb': gb, 'W_R': W_R,
        'valid': np.isfinite(W_R),
        'n_sci': np.isfinite(W_R_sci).sum(),
        'n_recal': np.isfinite(W_R_rec).sum(),
    }


def compute_lv_strip(
    cell_combined: dict,
    excluded: set,
    *,
    b_max_deg: float,
    dl_fine_deg: float,
    hpbw_deg: float,
    cutoff_hpbw: float = 1.5,
    keep_near_hpbw: float = 1.0,
    min_weight: float = 0.1,
    spectrum_key: str = 'R',
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


def build_heatmap(
    gl: np.ndarray,
    gb: np.ndarray,
    vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid scattered (l, b, W) onto an integer-degree map.

    Parameters
    ----------
    gl, gb : 1-D float arrays
        Galactic longitude and latitude.
    vals : 1-D float array
        Values to grid.

    Returns
    -------
    map_val : 2-D float array
        Gridded values, shape ``(n_b, n_l)``.
    l_unique, b_unique : 1-D int arrays
        Coordinate axes.
    """
    gl_int = np.round(gl).astype(int)
    gb_int = np.round(gb).astype(int)
    l_unique = np.arange(gl_int.min(), gl_int.max() + 1)
    b_unique = np.arange(gb_int.min(), gb_int.max() + 1)
    map_val = np.full((len(b_unique), len(l_unique)), np.nan)
    # Cells in this survey are unique (gl, gb) pairs by construction, so
    # fancy-index assignment is unambiguous (no duplicate-overwrite races).
    li = np.searchsorted(l_unique, gl_int)
    bi = np.searchsorted(b_unique, gb_int)
    in_bounds = (li >= 0) & (li < len(l_unique)) & (bi >= 0) & (bi < len(b_unique))
    map_val[bi[in_bounds], li[in_bounds]] = vals[in_bounds]
    return map_val, l_unique, b_unique
