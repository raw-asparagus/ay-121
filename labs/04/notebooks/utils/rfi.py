"""RFI flagging and outlier dump filtering."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def flag_rfi_channels(spectrum: np.ndarray, window: int = 15,
                      sigma_thresh: float = 5.0) -> int:
    """Flag RFI channels in a single spectrum using rolling median + MAD.

    Outlier channels (both positive spikes and negative dropouts) are
    replaced with NaN **in-place**.

    Parameters
    ----------
    spectrum : 1-D float array
        The spectrum to flag (modified in-place).
    window : int
        Rolling median window width.
    sigma_thresh : float
        Number of MAD-based sigma for the flag threshold.

    Returns
    -------
    int
        Number of channels flagged.
    """
    local_med = pd.Series(spectrum).rolling(
        window, center=True, min_periods=1,
    ).median().to_numpy()
    resid = spectrum - local_med
    mad = np.nanmedian(np.abs(resid))
    sigma = mad / 0.6745
    bad = np.abs(resid) > sigma_thresh * sigma
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        spectrum[bad] = np.nan
    return n_bad


def flag_outlier_dumps(records: list[dict],
                       dev_thresh: float = 0.10,
                       frac_thresh: float = 0.20) -> list[dict]:
    """Flag and remove dumps whose spectral shape deviates from group median.

    Groups dumps by (DR, target, LO) and compares each dump's Stokes I
    to the group median.  Dumps with too many deviant channels are removed
    from *records* (in-place) and returned separately.

    Parameters
    ----------
    records : list of dict
        Dump records; each must have 'dr', 'target', 'lo_mhz', 'row',
        'noise_on', and 'stokes_I' keys.
    dev_thresh : float
        Per-channel deviation threshold (fraction of median ratio).
    frac_thresh : float
        Fraction of channels that must deviate to flag a dump.

    Returns
    -------
    list of dict
        The removed outlier records.
    """
    from collections import defaultdict

    cell_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get('row', -1) < 0 or r['noise_on']:
            continue
        cell_groups[(r['dr'], r['target'], r['lo_mhz'])].append(r)

    outlier_records: list[dict] = []
    for group in cell_groups.values():
        if len(group) < 3:
            continue

        spectra = np.array([r['stokes_I'] for r in group])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med_spec = np.nanmedian(spectra, axis=0)

        if np.all(np.isnan(med_spec)) or np.nanmax(np.abs(med_spec)) == 0:
            continue

        for r, spec in zip(group, spectra):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                ratio = spec / med_spec
            valid = np.isfinite(ratio)
            if valid.sum() == 0:
                frac_bad = 1.0
            else:
                frac_bad = (
                    np.sum(np.abs(ratio[valid] - 1.0) > dev_thresh) / valid.sum()
                )
            if frac_bad > frac_thresh:
                records.remove(r)
                outlier_records.append(r)

    return outlier_records
