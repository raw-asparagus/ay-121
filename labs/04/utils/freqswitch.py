"""Frequency-switched spectrum computation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def compute_R_for_dumps(
    dump_list: list[dict],
    lo1: float,
    lo2: float,
    overlap_mask: np.ndarray,
    v_overlap: np.ndarray,
    *,
    lsr_correct: bool = False,
    v_lsr_grid: np.ndarray | None = None,
) -> dict | None:
    """Compute mean frequency-switched R from science dump records.

    Pairs LO1 and LO2 dumps within each session.  When *lsr_correct* is
    True, each session's R spectrum is interpolated onto a common LSR
    velocity grid before averaging across sessions.

    Parameters
    ----------
    dump_list : list of dict
        Science records with 'session', 'lo_mhz', 'stokes_I',
        'v_corr_lsr', 'ra', 'dec' keys.
    lo1, lo2 : float
        The two LO frequencies in MHz.
    overlap_mask : 1-D bool array
        Channel mask selecting the overlap region from the full 1024-ch band.
    v_overlap : 1-D float array
        Topocentric velocity grid for the overlap channels.
    lsr_correct : bool
        If True, shift each session to *v_lsr_grid* before averaging.
    v_lsr_grid : 1-D float array or None
        Common LSR velocity grid (required when *lsr_correct* is True).

    Returns
    -------
    dict or None
        ``{'R_overlap': ..., 'n_pairs': ..., 'ra': ..., 'dec': ...}``
        (LSR mode) or ``{'R_mean': ..., 'R_overlap': ..., ...}``
        (topocentric mode).  Returns None if no valid pairs.
    """
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in dump_list:
        by_session[r['session']].append(r)

    R_all: list[np.ndarray] = []
    session_labels: list[str] = []
    pairs_per_session: list[int] = []
    total_pairs = 0

    for sess_label, sess_dumps in by_session.items():
        d1 = [r for r in sess_dumps if r['lo_mhz'] == lo1]
        d2 = [r for r in sess_dumps if r['lo_mhz'] == lo2]
        n_p = min(len(d1), len(d2))
        if n_p == 0:
            continue

        I1 = np.array([r['stokes_I'] for r in d1[:n_p]])
        I2 = np.array([r['stokes_I'] for r in d2[:n_p]])
        R_pairs = (I1 - I2) / I2  # (n_p, 1024)

        if lsr_correct:
            R_sess = np.nanmean(R_pairs, axis=0)
            v_corr = np.mean([r['v_corr_lsr'] for r in d1[:n_p] + d2[:n_p]])
            v_sess_lsr = v_overlap + v_corr
            R_ov = R_sess[overlap_mask]
            R_interp = np.interp(
                v_lsr_grid[::-1], v_sess_lsr[::-1], R_ov[::-1],
                left=np.nan, right=np.nan,
            )[::-1]
            R_all.append(R_interp)
            session_labels.append(sess_label)
            pairs_per_session.append(n_p)
        else:
            R_all.append(R_pairs)

        total_pairs += n_p

    if total_pairs == 0:
        return None

    mean_ra = np.mean([r['ra'] for r in dump_list])
    mean_dec = np.mean([r['dec'] for r in dump_list])

    if lsr_correct:
        R_mean = np.nanmean(R_all, axis=0)
        return {
            'R_overlap': R_mean,
            'n_pairs': total_pairs, 'ra': mean_ra, 'dec': mean_dec,
            'n_sessions': len(R_all),
            'R_per_session': R_all,
            'session_labels': session_labels,
            'pairs_per_session': pairs_per_session,
        }
    R_cat = np.concatenate(R_all, axis=0)
    R_mean = np.nanmean(R_cat, axis=0)
    return {
        'R_mean': R_mean, 'R_overlap': R_mean[overlap_mask],
        'n_pairs': total_pairs, 'ra': mean_ra, 'dec': mean_dec,
    }


# --- Lab 4 main-pipeline helpers -----------------------------------------

# These were inlined in the analysis notebook prior to the strip of
# temperature calibration. They live here so the notebook stays a thin
# driver and the band-overlap / LSR-pair construction is reusable.

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
        'stokes_I', 'time', 'ra', 'dec' keys.
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
        ``(gl, gb) -> [{'session', 'pair_idx', 'R_lsr'}, ...]``.
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
        I1 = np.array([r['stokes_I'] for r in d1[:n_p]])
        I2 = np.array([r['stokes_I'] for r in d2[:n_p]])
        with np.errstate(divide='ignore', invalid='ignore'):
            R_pairs = (I1 - I2) / I2
        R_pairs_ov = R_pairs[:, overlap_mask]

        v_shifted = v_overlap + session_cell_vcorr.get((dr, gl, gb), 0.0)
        for i in range(n_p):
            R_lsr = _interp_to_lsr_inc(R_pairs_ov[i], v_shifted, v_lsr_inc)
            cell_pairs[(gl, gb)].append({
                'session': dr,
                'pair_idx': i,
                'R_lsr': R_lsr,
            })

    return dict(cell_pairs), dict(obs_dumps_by_cell), v_lsr_overlap, mean_vcorr


# --- Recal-pointing reduction ------------------------------------------

def build_recal_visits(
    records: list[dict],
    *,
    lo1: float,
    lo2: float,
    overlap_mask: np.ndarray,
    v_overlap: np.ndarray,
    v_lsr_overlap: np.ndarray,
    vlsr_correction_fn,
    visit_gap_sec: float = 300.0,
) -> dict:
    """Cluster recal dumps into visits and reduce each visit to LSR R per pol.

    The recal targets sit at fixed (RA, Dec) and are visited periodically
    (every ``recal_every_n_cells`` survey cells in the observation
    schedule).  Within one session the dumps from a single visit are
    contiguous in time; visits are separated by long gaps spent on
    science cells.

    Parameters
    ----------
    records
        Recal dumps from :func:`utils.io.load_recal_dumps` (preprocessed:
        fftshift + RFI applied to ``corr00`` and ``corr11``).
    lo1, lo2
        LO frequencies (MHz).  Each visit must pair LO1 and LO2 noise-off
        dumps to compute R.
    overlap_mask, v_overlap
        Topocentric overlap channel selector and velocity grid, from
        :func:`build_overlap_grid`.
    v_lsr_overlap
        LSR-corrected velocity grid produced by :func:`build_lsr_pairs`
        for the science cells.  Recal visits are interpolated onto the
        same grid so the spectra are directly comparable to science.
    vlsr_correction_fn
        ``v_corr = f(ra_deg, dec_deg, unix_time)`` such that
        ``v_LSR = v_topo + v_corr``.
    visit_gap_sec
        Time gap (seconds) above which consecutive dumps are split into
        separate visits.  Default 300 s comfortably exceeds the per-visit
        cadence (~5.25 s/dump * 12 dumps ~= 1 min) and stays well below
        the inter-visit interval.

    Returns
    -------
    dict
        Mapping ``(session, target_id) -> list of visit dicts`` with keys
        ``t_mid``, ``alt_mean``, ``n_pairs``, ``gl``, ``gb``,
        ``R_lsr`` (Stokes-I R = (I_LO1 - I_LO2)/I_LO2 with
        I = corr00 + corr11; this is the canonical recal spectrum),
        plus ``R_lsr_pol0`` and ``R_lsr_pol1`` for diagnostic comparison.
    """
    from collections import defaultdict

    v_lsr_inc = v_lsr_overlap[::-1]

    by_st: dict = defaultdict(list)
    for r in records:
        by_st[(r['session'], r['target_id'])].append(r)

    visits_by_st: dict = {}
    for key, dumps in by_st.items():
        dumps.sort(key=lambda d: d['time'])
        if not dumps:
            continue

        clusters: list[list[dict]] = [[dumps[0]]]
        for d in dumps[1:]:
            if d['time'] - clusters[-1][-1]['time'] > visit_gap_sec:
                clusters.append([d])
            else:
                clusters[-1].append(d)

        visits: list[dict] = []
        for cluster in clusters:
            obs = [d for d in cluster if not d['noise_on']]
            d1 = [d for d in obs if d['lo_mhz'] == lo1]
            d2 = [d for d in obs if d['lo_mhz'] == lo2]
            n_p = min(len(d1), len(d2))
            if n_p == 0:
                continue

            t_mid = np.mean([d['time'] for d in cluster])
            alt_mean = np.mean([d['alt'] for d in cluster])
            ra0, dec0 = obs[0]['ra'], obs[0]['dec']
            v_corr = vlsr_correction_fn(ra0, dec0, t_mid)
            v_visit_topo = v_overlap + v_corr

            visit = {
                't_mid': t_mid,
                'alt_mean': alt_mean,
                'n_pairs': n_p,
                'gl': obs[0]['gl'],
                'gb': obs[0]['gb'],
            }

            for pol_key, out_key in (('corr00', 'R_lsr_pol0'),
                                     ('corr11', 'R_lsr_pol1'),
                                     ('stokes_I', 'R_lsr')):
                I1 = np.array([d[pol_key] for d in d1[:n_p]])
                I2 = np.array([d[pol_key] for d in d2[:n_p]])
                with np.errstate(divide='ignore', invalid='ignore'):
                    R_pairs = (I1 - I2) / I2
                R_pairs_ov = R_pairs[:, overlap_mask]
                R_topo_mean = np.nanmean(R_pairs_ov, axis=0)
                visit[out_key] = _interp_to_lsr_inc(
                    R_topo_mean, v_visit_topo, v_lsr_inc,
                )

            visits.append(visit)

        if visits:
            visits_by_st[key] = visits

    return visits_by_st


def aggregate_recal_visits(
    recal_visits: dict,
    dv_kms: float,
) -> dict:
    """Pool visits across all sessions per ``target_id`` into one cell entry.

    Output mirrors ``cell_combined``: each target becomes a single
    (gl, gb) cell whose ``R`` is the visit-mean Stokes-I spectrum
    (matching the science pipeline), with per-pol means retained
    alongside for diagnostic comparison.

    Returns
    -------
    dict
        ``target_id -> {'gl', 'gb', 'R', 'R_pol0', 'R_pol1', 'n_visits',
        'n_sessions', 'W_R'}``.  ``W_R`` is the velocity-integrated
        Stokes-I ratio in km/s, directly comparable to the science cells'
        ``W_R``.
    """
    from collections import defaultdict

    by_tgt: dict = defaultdict(list)
    sessions_by_tgt: dict = defaultdict(set)
    for (sess, tgt), visits in recal_visits.items():
        for v in visits:
            by_tgt[tgt].append(v)
            sessions_by_tgt[tgt].add(sess)

    cells: dict = {}
    for tgt, visits in by_tgt.items():
        if not visits:
            continue
        R_I = np.nanmean([v['R_lsr'] for v in visits], axis=0)
        R_pol0 = np.nanmean([v['R_lsr_pol0'] for v in visits], axis=0)
        R_pol1 = np.nanmean([v['R_lsr_pol1'] for v in visits], axis=0)
        cells[tgt] = {
            'gl': visits[0]['gl'],
            'gb': visits[0]['gb'],
            'R': R_I,
            'R_pol0': R_pol0,
            'R_pol1': R_pol1,
            'n_visits': len(visits),
            'n_sessions': len(sessions_by_tgt[tgt]),
            'W_R': np.nansum(R_I) * dv_kms,
        }
    return cells
