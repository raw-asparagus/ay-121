"""Frequency-switched spectrum computation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


HI_REST_MHZ_DEFAULT = 1420.405751768
C_KMS_DEFAULT = 299792.458


def build_overlap_grid(
    f1_mhz: float,
    f2_mhz: float,
    sample_rate_hz: float,
    nfft: int,
    *,
    edge_trim_mhz: float,
    hi_rest_mhz: float = HI_REST_MHZ_DEFAULT,
    c_kms: float = C_KMS_DEFAULT,
) -> dict:
    """Build the baseband, sky, overlap, and topocentric-velocity grids.

    Frequency-switched spectra at LO pair (f1, f2) share a sky-frequency
    band where both LO settings have valid samples (after trimming the
    SDR edges). The mask of channels in that band -- ``overlap_mask`` --
    plus the topocentric LSR velocity grid over the overlap region --
    ``v_overlap`` -- are produced once and reused by every cell.

    Returns
    -------
    dict
        ``f_bb_mhz``, ``overlap_mask``, ``f_overlap``, ``v_overlap``,
        ``dv_kms``, ``f_overlap_lo``, ``f_overlap_hi``.
    """
    f_bb_mhz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate_hz)) / 1e6
    f_sky1 = f1_mhz + f_bb_mhz
    f_sky2 = f2_mhz + f_bb_mhz
    f_overlap_lo = max(f_sky1[0], f_sky2[0]) + edge_trim_mhz
    f_overlap_hi = min(f_sky1[-1], f_sky2[-1]) - edge_trim_mhz
    overlap_mask = (f_sky1 >= f_overlap_lo) & (f_sky1 <= f_overlap_hi)
    f_overlap = f_sky1[overlap_mask]
    v_overlap = c_kms * (1 - f_overlap / hi_rest_mhz)
    dv_kms = np.abs(np.median(np.diff(v_overlap)))
    return {
        'f_bb_mhz': f_bb_mhz,
        'overlap_mask': overlap_mask,
        'f_overlap': f_overlap,
        'v_overlap': v_overlap,
        'dv_kms': dv_kms,
        'f_overlap_lo': f_overlap_lo,
        'f_overlap_hi': f_overlap_hi,
    }


def _interp_to_lsr_inc(
    spec_topo: np.ndarray,
    v_topo_inc: np.ndarray,
    v_lsr_inc: np.ndarray,
) -> np.ndarray:
    """Interp a topocentric overlap spectrum (decreasing v) onto the LSR grid."""
    return np.interp(
        v_lsr_inc, v_topo_inc[::-1], spec_topo[::-1],
        left=np.nan, right=np.nan,
    )[::-1]


def build_lsr_pairs(
    records: list[dict],
    *,
    lo1: float,
    lo2: float,
    overlap_mask: np.ndarray,
    v_overlap: np.ndarray,
    vlsr_correction_fn,
) -> tuple[dict, dict, np.ndarray, float]:
    """Group science dumps by cell, pair LO1/LO2 dumps, and LSR-interpolate R.

    Parameters
    ----------
    records : list of dict
        Science records with 'session', 'gl', 'gb', 'noise_on', 'lo_mhz',
        'corr00', 'corr11', 'time', 'ra', 'dec' keys.  ``corr00`` and
        ``corr11`` are propagated as separate pols all the way through
        the ratio computation; downstream calibration applies pol-specific
        ``T_cal``.
    lo1, lo2 : float
        Local-oscillator frequencies (MHz).
    overlap_mask : 1-D bool array
        Channels inside the f1/f2 sky-frequency overlap.
    v_overlap : 1-D float array
        Topocentric LSR velocities for the overlap channels.
    vlsr_correction_fn : callable(ra_deg, dec_deg, unix_time) -> float
        Returns ``v_corr`` such that ``v_LSR = v_topo + v_corr``.

    Returns
    -------
    cell_pairs : dict
        ``(gl, gb) -> [{'session', 'pair_idx', 'R_lsr_pol0', 'R_lsr_pol1'}, ...]``.
    obs_dumps_by_cell : dict
        ``(session, gl, gb) -> list of obs records``.
    v_lsr_overlap : 1-D float array
        Mean-LSR-corrected velocity grid (decreasing, like v_overlap).
    mean_vcorr : float
        Mean v_corr applied (km/s).
    """
    from collections import defaultdict

    obs_dumps_by_cell: dict = defaultdict(list)
    for r in records:
        if r['noise_on']:
            continue
        obs_dumps_by_cell[(r['session'], r['gl'], r['gb'])].append(r)

    session_cell_vcorr = {}
    for key, group in obs_dumps_by_cell.items():
        r0 = group[0]
        mean_t = np.mean([r['time'] for r in group])
        session_cell_vcorr[key] = vlsr_correction_fn(r0['ra'], r0['dec'], mean_t)

    vcorrs = list(session_cell_vcorr.values())
    mean_vcorr = np.mean(vcorrs) if vcorrs else 0.0
    v_lsr_overlap = v_overlap + mean_vcorr
    v_lsr_inc = v_lsr_overlap[::-1]

    cell_pairs: dict = defaultdict(list)
    for (dr, gl, gb), dumps in obs_dumps_by_cell.items():
        d1 = [r for r in dumps if r['lo_mhz'] == lo1]
        d2 = [r for r in dumps if r['lo_mhz'] == lo2]
        n_p = min(len(d1), len(d2))
        if n_p == 0:
            continue

        v_shifted = v_overlap + session_cell_vcorr.get((dr, gl, gb), 0.0)

        R_ov_per_pol: dict[str, np.ndarray] = {}
        for pol_key in ('corr00', 'corr11'):
            I1 = np.array([r[pol_key] for r in d1[:n_p]])
            I2 = np.array([r[pol_key] for r in d2[:n_p]])
            with np.errstate(divide='ignore', invalid='ignore'):
                R_pairs = (I1 - I2) / I2
            R_ov_per_pol[pol_key] = R_pairs[:, overlap_mask]

        for i in range(n_p):
            cell_pairs[(gl, gb)].append({
                'session': dr,
                'pair_idx': i,
                'R_lsr_pol0': _interp_to_lsr_inc(
                    R_ov_per_pol['corr00'][i], v_shifted, v_lsr_inc),
                'R_lsr_pol1': _interp_to_lsr_inc(
                    R_ov_per_pol['corr11'][i], v_shifted, v_lsr_inc),
            })

    return dict(cell_pairs), dict(obs_dumps_by_cell), v_lsr_overlap, mean_vcorr
