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
