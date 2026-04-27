"""Per-cell gain and T_sys calibration from noise-diode cal dumps.

For each (session, gl, gb), use the cal-on/cal-off pair to estimate the
single-pol gain g = median[(P_on - P_off) / T_cal], then derive
T_sys = mean(P_off) / g and convert R -> T_B = R * T_sys.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def group_dumps_by_cell(records: list[dict]) -> tuple[dict, dict]:
    """Split records into cal-on and science groups keyed by (session, gl, gb)."""
    cal_dumps: dict[tuple, list[dict]] = defaultdict(list)
    obs_dumps_by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r['session'], r['gl'], r['gb'])
        if r['noise_on']:
            cal_dumps[key].append(r)
        else:
            obs_dumps_by_cell[key].append(r)
    return cal_dumps, obs_dumps_by_cell


def compute_cell_gains(
    cal_dumps: dict,
    obs_dumps_by_cell: dict,
    dc_mask: np.ndarray,
    t_cal: float,
    *,
    spectrum_key: str = 'corr11',
    gain_key: str = 'gain_11',
) -> dict:
    """Per-(session, cell) gain from the noise-diode pair.

    Returns
    -------
    dict
        ``{(session, gl, gb): {gain_key: float}}``.  Cells without both
        cal and science dumps are omitted.
    """
    cell_gains: dict[tuple, dict] = {}
    for key, cals in cal_dumps.items():
        obs = obs_dumps_by_cell.get(key, [])
        if not cals or not obs:
            continue
        P_on = np.nanmean([c[spectrum_key] for c in cals], axis=0)
        P_off = np.nanmean([o[spectrum_key] for o in obs], axis=0)
        diff = P_on - P_off
        diff[diff <= 0] = np.nan
        cell_gains[key] = {gain_key: float(np.nanmedian(diff[dc_mask] / t_cal))}
    return cell_gains


def apply_tsys_calibration(
    cell_results: dict,
    cell_gains: dict,
    obs_dumps_by_cell: dict,
    dc_mask: np.ndarray,
    *,
    spectrum_key: str = 'corr11',
    gain_key: str = 'gain_11',
) -> dict:
    """Convert per-(session, cell) R spectra to T_B using each cell's T_sys.

    For each cell, T_sys = mean(P_off) / gain, where P_off is the median over
    *dc_mask*-selected channels averaged across that cell's science dumps.

    Returns
    -------
    dict
        ``{(session, gl, gb): {'T_B_overlap', 'R_overlap', 'T_sys',
        gain_key, 'n_pairs'}}``.
    """
    cell_results_TB: dict[tuple, dict] = {}
    for key, res in cell_results.items():
        g = cell_gains.get(key)
        obs = obs_dumps_by_cell.get(key, [])
        if g is None or not obs:
            continue
        mean_P = float(np.mean([np.nanmedian(o[spectrum_key][dc_mask]) for o in obs]))
        T_sys = mean_P / g[gain_key]
        cell_results_TB[key] = {
            'T_B_overlap': res['R_overlap'] * T_sys,
            'R_overlap': res['R_overlap'],
            'T_sys': T_sys,
            gain_key: g[gain_key],
            'n_pairs': res['n_pairs'],
        }
    return cell_results_TB
