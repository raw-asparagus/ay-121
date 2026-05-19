"""Per-(session, cell) calibration scalars.

Band-averaged noise-on / noise-off powers per pol, keyed by
``(session, gl, gb)``, used downstream by ``main_scan_qa.ipynb`` to
apply ``Tcal_pol(t)`` from ``artifacts/tcal_drift_state.pkl``:

    T_sys_pol = Tcal_pol(t) * P_off_pol / (P_on_pol - P_off_pol)
    T_B_pol   = R_pol * T_sys_pol
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_cell_scalars(
    records: list[dict],
    *,
    int_mask: np.ndarray,
    lo_freqs: tuple[float, float],
) -> dict:
    """Band-average corr00 / corr11 power per pol for each (session, cell, LO).

    For each science dump that survived upstream outlier filtering, the
    pol-0 and pol-1 powers are averaged over ``int_mask`` (the LO overlap
    with DC excluded), then aggregated per (session, gl, gb) into a tiny
    scalar table.

    Parameters
    ----------
    records : list of dict
        Dump records with ``corr00``, ``corr11``, ``session``, ``gl``,
        ``gb``, ``lo_mhz``, ``noise_on``, ``time``.
    int_mask : np.ndarray
        Boolean integration mask over the 1024-channel grid (overlap with
        DC bin excluded).
    lo_freqs : (float, float)
        The two LO frequencies that define the freq-switched pair.

    Returns
    -------
    dict
        ``(session, gl, gb) -> entry``.  Each entry has ``t_median`` and,
        for each ``lo_mhz`` in ``lo_freqs`` and each pol ``pol0``/``pol1``,
        the keys ``P_on_{pol}_{lo:g}``, ``P_off_{pol}_{lo:g}`` plus
        ``n_on_{lo:g}``, ``n_off_{lo:g}``.  Missing on/off groups yield
        NaN powers (and zero counts) so consumers can detect partial
        coverage.
    """
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        p0 = np.nanmean(r['corr00'][int_mask])
        p1 = np.nanmean(r['corr11'][int_mask])
        bucket[(r['session'], r['gl'], r['gb'])][(r['lo_mhz'], r['noise_on'])].append(
            (r['time'], p0, p1)
        )

    cell_scalars: dict = {}
    for scell, by_lo_noise in bucket.items():
        times = [t for buckets in by_lo_noise.values() for t, _, _ in buckets]
        entry = {'t_median': np.median(times)}
        for lo_mhz in lo_freqs:
            on  = by_lo_noise.get((lo_mhz, True),  [])
            off = by_lo_noise.get((lo_mhz, False), [])
            for pol_idx, pol_label in ((1, 'pol0'), (2, 'pol1')):
                P_on  = np.mean([row[pol_idx] for row in on])  if on  else np.nan
                P_off = np.mean([row[pol_idx] for row in off]) if off else np.nan
                entry[f'P_on_{pol_label}_{lo_mhz:g}']  = P_on
                entry[f'P_off_{pol_label}_{lo_mhz:g}'] = P_off
            entry[f'n_on_{lo_mhz:g}']  = len(on)
            entry[f'n_off_{lo_mhz:g}'] = len(off)
        cell_scalars[scell] = entry
    return cell_scalars


def count_fully_calibratable(
    cell_scalars: dict,
    *,
    lo_freqs: tuple[float, float],
) -> int:
    """Count cells with finite P_on/P_off for both pols at both LOs."""
    n = 0
    for e in cell_scalars.values():
        ok = all(
            np.isfinite(e[f'P_on_{pol}_{lo:g}']) and np.isfinite(e[f'P_off_{pol}_{lo:g}'])
            for lo in lo_freqs
            for pol in ('pol0', 'pol1')
        )
        if ok:
            n += 1
    return n
