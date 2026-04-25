"""RFI flagging and outlier dump filtering."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import argrelextrema

from .cache import _cache_stable


def _dump_coordinate_key(record: dict) -> tuple[int, int] | None:
    """Return the galactic (l, b) coordinate key for a dump record."""
    gl = record.get('gl')
    gb = record.get('gb')
    if gl is not None and gb is not None and np.isfinite(gl) and np.isfinite(gb):
        return (int(round(float(gl))), int(round(float(gb))))
    return None


@_cache_stable(module='utils.rfi')
def _cheb_pseudocontinuum(spectrum: np.ndarray, window: int = 15,
                          degree: int = 3, sample_frac: float = 0.7,
                          extrema_order: int = 2,
                          seed: int = 42) -> np.ndarray:
    """Sliding-window Chebyshev pseudo-continuum with reflect padding.

    For each channel, a degree-N Chebyshev polynomial is fit to the
    surrounding window after excluding local maxima and minima (which
    may be RFI or dropouts) and drawing a random subsample of the
    remaining points. The spectrum is reflect-padded at both edges so
    every window position sees a full-width window.

    Parameters
    ----------
    spectrum : 1-D float array
        Input spectrum.
    window : int
        Sliding window width in channels.
    degree : int
        Chebyshev polynomial degree.
    sample_frac : float
        Fraction of non-extrema points to subsample for the fit.
    extrema_order : int
        ``order`` parameter for `scipy.signal.argrelextrema`; a point
        must dominate this many neighbors on each side to be excluded.
    seed : int
        RNG seed for reproducible subsampling.

    Returns
    -------
    np.ndarray
        Pseudo-continuum estimate, same length as *spectrum*.
    """
    n = len(spectrum)
    half_w = window // 2
    rng = np.random.default_rng(seed)
    padded = np.pad(spectrum, half_w, mode='reflect')

    continuum = np.full(n, np.nan)
    for i in range(n):
        chunk = padded[i : i + window]
        x_local = np.arange(window)
        finite = np.isfinite(chunk)
        if finite.sum() < degree + 2:
            continue
        x_f = x_local[finite]
        y_f = chunk[finite]

        max_idx = set(argrelextrema(y_f, np.greater, order=extrema_order)[0])
        min_idx = set(argrelextrema(y_f, np.less, order=extrema_order)[0])
        keep = np.array([j not in max_idx and j not in min_idx
                         for j in range(len(y_f))])
        if keep.sum() < degree + 1:
            x_fit, y_fit = x_f, y_f
        else:
            x_fit, y_fit = x_f[keep], y_f[keep]
            n_sample = max(degree + 1, int(len(x_fit) * sample_frac))
            if n_sample < len(x_fit):
                idx = np.sort(rng.choice(len(x_fit), n_sample, replace=False))
                x_fit, y_fit = x_fit[idx], y_fit[idx]

        coeffs = np.polynomial.chebyshev.chebfit(x_fit, y_fit, degree)
        continuum[i] = np.polynomial.chebyshev.chebval(half_w, coeffs)
    return continuum


def flag_rfi_channels(spectrum: np.ndarray, window: int = 15,
                      sigma_thresh: float = 10.0, degree: int = 3,
                      sample_frac: float = 0.7,
                      extrema_order: int = 2) -> int:
    """Flag RFI channels using a Chebyshev pseudo-continuum + MAD clip.

    A sliding-window Chebyshev polynomial is fit to the spectrum after
    excluding local extrema, producing a robust pseudo-continuum.
    Channels whose residual exceeds *sigma_thresh* MAD-based sigma are
    replaced with NaN **in-place**.

    Parameters
    ----------
    spectrum : 1-D float array
        The spectrum to flag (modified in-place).
    window : int
        Sliding window width in channels.
    sigma_thresh : float
        Number of MAD-based sigma for the flag threshold.
    degree : int
        Chebyshev polynomial degree for the pseudo-continuum fit.
    sample_frac : float
        Fraction of non-extrema points to subsample per window.
    extrema_order : int
        ``order`` parameter for local extrema detection.

    Returns
    -------
    int
        Number of channels flagged.
    """
    continuum = _cheb_pseudocontinuum(spectrum, window, degree, sample_frac,
                                     extrema_order)
    resid = spectrum - continuum
    mad = np.nanmedian(np.abs(resid))
    sigma = mad / 0.6745
    finite = np.isfinite(spectrum)
    bad = (np.abs(resid) > sigma_thresh * sigma) & finite
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        spectrum[bad] = np.nan
    return n_bad


def flag_outlier_dumps(records: list[dict],
                       dev_thresh: float = 0.10,
                       frac_thresh: float = 0.20) -> list[dict]:
    """Flag and remove dumps whose spectral shape deviates from group median.

    Groups dumps by (session, galactic coordinate, LO) and compares each
    dump's Stokes I to the group median.  Dumps with too many deviant
    channels are removed from *records* (in-place) and returned separately.

    Parameters
    ----------
    records : list of dict
        Dump records; each must have 'session', 'lo_mhz', 'noise_on',
        and 'stokes_I' keys, plus 'gl'/'gb'.
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
        if r['noise_on']:
            continue
        coord_key = _dump_coordinate_key(r)
        if coord_key is None:
            continue
        cell_groups[(r['session'], coord_key, r['lo_mhz'])].append(r)

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
