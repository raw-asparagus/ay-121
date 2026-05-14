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
    return np.array(gl, dtype=float), np.array(gb, dtype=float), np.array(vals)


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
    map_val[bi[in_bounds], li[in_bounds]] = np.asarray(vals)[in_bounds]
    return map_val, l_unique, b_unique
