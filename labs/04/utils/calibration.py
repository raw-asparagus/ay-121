"""Per-cell gain and T_sys calibration from noise-diode cal dumps.

For each (session, gl, gb), use the cal-on/cal-off pair to estimate the
single-pol gain spectrum g(nu) = (P_on - P_off) / T_cal, then derive
T_sys(nu) = mean(P_off) / g(nu) and convert R(nu) -> T_B(nu) = R(nu) * T_sys(nu).

Both g(nu) and T_sys(nu) are full-band 1-D arrays indexed by baseband channel.
Channel-by-channel division is shot-noisy on a single dump pair, so g(nu) is
median-smoothed over GAIN_SMOOTH_CHANNELS before being passed to T_sys; the
diode response is intrinsically smooth in frequency, so this removes the
noise without biasing the band shape.

A scalar band-median T_sys is also stored for backward-compatible reporting
(QA flagging, summary tables, pickle handoff to legacy diagnostics).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import median_filter


# Channel width (samples) of the running-median smoothing applied to g(nu)
# before it is inverted to compute T_sys. The diode response is smooth on
# scales much wider than this, so 15 channels removes shot noise while
# preserving any real bandpass curvature.
GAIN_SMOOTH_CHANNELS = 15

# Hard floor on g(nu); channels with smoothed gain below this are masked
# (NaN propagates into T_sys -> T_B). Catches dead-band channels where
# P_on ~ P_off (e.g. saturated RFI in the cal dump).
GAIN_FLOOR_FRAC = 0.05  # fraction of the band-median gain

# T_sys(nu) = P_off / g is biased upward at HI-line channels because P_off
# includes the line itself. Galactic-plane sightlines have line emission
# spanning >100 km/s, wider than any reasonable running-median window. The
# robust fix is a low-order polynomial fit to T_sys_raw with iterative
# sigma-clipping: positive outliers (line emission) get masked out and the
# fit converges to the smooth receiver-bandpass T_sys.
#
# Degree 4 captures real bandpass curvature without overfitting line
# residuals. Three sigma-clip iterations at 2.5*sigma converges quickly.
TSYS_FIT_DEGREE = 2
TSYS_CLIP_ITERS = 5
TSYS_CLIP_SIGMA = 2.0


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


def _nan_safe_median_filter(spec: np.ndarray, size: int) -> np.ndarray:
    """Running-median filter that interpolates NaNs internally then restores them."""
    finite = np.isfinite(spec)
    if finite.sum() < size:
        return spec.copy()
    filled = spec.copy()
    filled[~finite] = np.interp(
        np.flatnonzero(~finite),
        np.flatnonzero(finite),
        spec[finite],
    )
    smoothed = median_filter(filled, size=size, mode='nearest')
    smoothed[~finite] = np.nan
    return smoothed


def smooth_gain_spectrum(gain_raw: np.ndarray, dc_mask: np.ndarray) -> np.ndarray:
    """Median-smooth g(nu) with NaN-safe handling and a dead-band floor.

    DC-bin and pre-existing NaN channels are preserved as NaN. The floor is
    set relative to the band-median of the smoothed gain.
    """
    g = gain_raw.copy()
    g[~dc_mask] = np.nan
    g_smooth = _nan_safe_median_filter(g, GAIN_SMOOTH_CHANNELS)
    g_med = np.nanmedian(g_smooth)
    if np.isfinite(g_med) and g_med > 0:
        g_smooth[g_smooth < GAIN_FLOOR_FRAC * g_med] = np.nan
    return g_smooth


def smooth_tsys_spectrum(tsys_nu: np.ndarray) -> np.ndarray:
    """Robust polynomial fit to T_sys(nu) with line-region sigma-clipping.

    P_off includes the HI line, biasing T_sys = P_off / g upward at line
    channels. The receiver-bandpass component of T_sys is smooth, so we fit a
    degree-TSYS_FIT_DEGREE Chebyshev polynomial with iterative one-sided
    clipping: channels with positive residuals beyond TSYS_CLIP_SIGMA are
    masked as line emission and the fit re-runs.

    Returns a smooth T_sys(nu) on the same channel grid; NaNs in the input
    are preserved in the output.
    """
    tsys_nu = np.asarray(tsys_nu, dtype=float)
    n = tsys_nu.size
    finite = np.isfinite(tsys_nu)
    if finite.sum() < TSYS_FIT_DEGREE + 2:
        return tsys_nu.copy()

    x = np.linspace(-1.0, 1.0, n)
    keep = finite.copy()

    for _ in range(TSYS_CLIP_ITERS + 1):
        if keep.sum() < TSYS_FIT_DEGREE + 2:
            break
        coeffs = np.polynomial.chebyshev.chebfit(
            x[keep], tsys_nu[keep], TSYS_FIT_DEGREE,
        )
        model = np.polynomial.chebyshev.chebval(x, coeffs)
        resid = tsys_nu - model
        # MAD-based sigma over surviving channels; one-sided clip on positive
        # residuals (line emission inflates T_sys upward only).
        resid_keep = resid[keep]
        mad = float(np.median(np.abs(resid_keep - np.median(resid_keep))))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(resid_keep))
        if not np.isfinite(sigma) or sigma <= 0:
            break
        new_keep = finite & (resid <= TSYS_CLIP_SIGMA * sigma)
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep

    smoothed = np.polynomial.chebyshev.chebval(x, coeffs)
    smoothed[~finite] = np.nan
    return smoothed


def compute_cell_gains(
    cal_dumps: dict,
    obs_dumps_by_cell: dict,
    dc_mask: np.ndarray,
    t_cal: float,
    *,
    spectrum_key: str,
    gain_key: str,
) -> dict:
    """Per-(session, cell) channel-dependent gain from the noise-diode pair.

    Returns
    -------
    dict
        ``{(session, gl, gb): {gain_key: g_nu, gain_key + '_scalar': g_med}}``
        where ``g_nu`` is a smoothed 1-D array (NFFT,) in raw-power-per-K and
        ``g_med`` is the band-median scalar (legacy-compatible reporting).
        Cells without both cal and science dumps are omitted.
    """
    cell_gains: dict[tuple, dict] = {}
    for key, cals in cal_dumps.items():
        obs = obs_dumps_by_cell.get(key, [])
        if not cals or not obs:
            continue
        P_on = np.nanmean([c[spectrum_key] for c in cals], axis=0)
        P_off = np.nanmean([o[spectrum_key] for o in obs], axis=0)
        diff = P_on - P_off
        diff_safe = diff.astype(float).copy()
        diff_safe[diff_safe <= 0] = np.nan
        gain_raw = diff_safe / t_cal
        gain_nu = smooth_gain_spectrum(gain_raw, dc_mask)
        cell_gains[key] = {
            gain_key: gain_nu,
            gain_key + '_scalar': float(np.nanmedian(gain_nu)),
        }
    return cell_gains


def apply_tsys_calibration(
    cell_results: dict,
    cell_gains: dict,
    obs_dumps_by_cell: dict,
    dc_mask: np.ndarray,
    *,
    spectrum_key: str,
    gain_key: str,
    overlap_mask: np.ndarray | None = None,
) -> dict:
    """Convert per-(session, cell) R spectra to T_B using channel-dependent T_sys.

    For each cell, ``T_sys(nu) = mean(P_off(nu)) / g(nu)``, where ``P_off`` is
    the dump-mean noise-off spectrum and ``g(nu)`` is the smoothed gain. The
    scalar band-median ``T_sys`` is also stored for legacy reporting.

    Parameters
    ----------
    overlap_mask : 1-D bool array or None
        Channel mask selecting the post-edge-trim overlap region from the
        full NFFT-channel band. When provided, ``T_sys_overlap`` and
        ``T_B_overlap`` are sliced to it (matching ``R_overlap``). When None,
        ``R_overlap`` is assumed to already live on the full NFFT grid and is
        multiplied directly.

    Returns
    -------
    dict
        ``{(session, gl, gb): {'T_B_overlap', 'R_overlap', 'T_sys_overlap',
        'T_sys', 'T_sys_nu', gain_key, gain_key + '_scalar', 'n_pairs'}}``.
        ``T_sys_nu`` is the full-band (NFFT,) array; ``T_sys_overlap`` is the
        overlap-trimmed slice; ``T_sys`` is the scalar band-median over the
        overlap region (or full band if overlap_mask is None).
    """
    cell_results_TB: dict[tuple, dict] = {}
    for key, res in cell_results.items():
        g = cell_gains.get(key)
        obs = obs_dumps_by_cell.get(key, [])
        if g is None or not obs:
            continue
        P_off_nu = np.nanmean([o[spectrum_key] for o in obs], axis=0)
        P_off_nu = P_off_nu.astype(float).copy()
        P_off_nu[~dc_mask] = np.nan

        gain_nu = g[gain_key]
        T_sys_raw = P_off_nu / gain_nu  # NaN where gain or P_off is NaN
        T_sys_nu = smooth_tsys_spectrum(T_sys_raw)

        if overlap_mask is not None:
            T_sys_overlap = T_sys_nu[overlap_mask]
            T_B_overlap = res['R_overlap'] * T_sys_overlap
        else:
            T_sys_overlap = T_sys_nu
            T_B_overlap = res['R_overlap'] * T_sys_nu

        T_sys_scalar = float(np.nanmedian(T_sys_overlap))

        cell_results_TB[key] = {
            'T_B_overlap': T_B_overlap,
            'R_overlap': res['R_overlap'],
            'T_sys_nu': T_sys_nu,
            'T_sys_overlap': T_sys_overlap,
            'T_sys': T_sys_scalar,
            gain_key: gain_nu,
            gain_key + '_scalar': g[gain_key + '_scalar'],
            'n_pairs': res['n_pairs'],
        }
    return cell_results_TB
