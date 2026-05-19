"""RFI flagging and outlier dump filtering."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import argrelextrema

from .cache import _cache_stable


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


def flag_rfi_channels(spectrum: np.ndarray, *, window: int,
                      sigma_thresh: float, degree: int,
                      sample_frac: float,
                      extrema_order: int) -> int:
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
    n_bad = np.sum(bad)
    if n_bad > 0:
        spectrum[bad] = np.nan
    return n_bad


def preprocess_dumps(
    records: list[dict],
    *,
    rfi_window: int,
    rfi_sigma: float,
    rfi_degree: int,
    rfi_sample_frac: float,
    rfi_extrema_order: int,
) -> int:
    """In-place fftshift + per-pol RFI flag.

    For each record: ``np.fft.fftshift`` ``corr00`` and ``corr11`` so
    baseband indices run lowest-to-highest, then flag RFI on each pol
    independently with :func:`flag_rfi_channels`.

    The two pols are NOT summed here -- they are propagated separately
    through the pipeline so that downstream temperature calibration can
    apply pol-specific ``T_cal`` (pol 0 = 58 K, pol 1 = 79 K).  The sum
    into a Stokes-I science product is deferred to the plotting stage,
    after calibration.

    Returns the total number of channels flagged across both pols.
    """
    total = 0
    for r in records:
        r['corr00'] = np.fft.fftshift(r['corr00'])
        r['corr11'] = np.fft.fftshift(r['corr11'])
        for spec_key in ('corr00', 'corr11'):
            total += flag_rfi_channels(
                r[spec_key],
                window=rfi_window,
                sigma_thresh=rfi_sigma,
                degree=rfi_degree,
                sample_frac=rfi_sample_frac,
                extrema_order=rfi_extrema_order,
            )
    return total


def flag_outlier_dumps(records: list[dict],
                       *,
                       dev_thresh: float,
                       frac_thresh: float,
                       min_group_size: int,
                       pols: tuple[str, ...] = ('corr00', 'corr11')) -> list[dict]:
    """Flag and remove dumps whose spectral shape deviates from group median.

    Groups dumps by (session, galactic coordinate, LO) and compares each
    dump's spectrum to the group median **independently per pol** (pol 0
    = ``corr00``, pol 1 = ``corr11``).  A dump is flagged if any selected
    pol's deviant-channel fraction exceeds ``frac_thresh``; flagged dumps
    are removed from *records* (in-place) and returned separately.

    Per-pol shape checking is consistent with the per-pol propagation of
    the science spectra (each pol carries its own bandpass and calibration
    later) -- a dump with a glitch in only one pol is still bad.

    Parameters
    ----------
    records : list of dict
        Dump records; each must have 'session', 'lo_mhz', 'noise_on',
        and 'corr00'/'corr11' keys, plus 'gl'/'gb'.
    dev_thresh : float
        Per-channel deviation threshold (fraction of median ratio).
    frac_thresh : float
        Fraction of channels that must deviate to flag a dump (applied
        independently to each pol).
    pols : tuple of str
        Which pols to flag on: ``('corr00',)`` for pol 0 only,
        ``('corr11',)`` for pol 1 only, or ``('corr00', 'corr11')``
        (default) for both.  A dump is flagged if any listed pol with a
        usable group-median reference exceeds ``frac_thresh``.

    Returns
    -------
    list of dict
        The removed outlier records.
    """
    from collections import defaultdict

    valid_pols = ('corr00', 'corr11')
    if not pols or any(p not in valid_pols for p in pols):
        raise ValueError(
            f'pols must be a non-empty subset of {valid_pols}, got {pols!r}'
        )

    cell_groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r['noise_on']:
            continue
        cell_groups[(r['session'], r['gl'], r['gb'], r['lo_mhz'])].append(r)

    outlier_records: list[dict] = []
    for group in cell_groups.values():
        if len(group) < min_group_size:
            continue

        pol_stacks = {p: np.array([r[p] for r in group]) for p in pols}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pol_meds = {k: np.nanmedian(v, axis=0) for k, v in pol_stacks.items()}

        # Drop pols whose median is unusable (all-NaN or all-zero); a dump
        # is flagged only by a pol that has a viable reference.
        usable_pols = [
            k for k, med in pol_meds.items()
            if not np.all(np.isnan(med)) and np.nanmax(np.abs(med)) > 0
        ]
        if not usable_pols:
            continue

        for i, r in enumerate(group):
            flagged = False
            for pol_key in usable_pols:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    ratio = pol_stacks[pol_key][i] / pol_meds[pol_key]
                valid = np.isfinite(ratio)
                if valid.sum() == 0:
                    frac_bad = 1.0
                else:
                    frac_bad = (
                        np.sum(np.abs(ratio[valid] - 1.0) > dev_thresh)
                        / valid.sum()
                    )
                if frac_bad > frac_thresh:
                    flagged = True
                    break
            if flagged:
                records.remove(r)
                outlier_records.append(r)

    return outlier_records
