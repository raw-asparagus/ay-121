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
            if np.ndim(R_sess) == 0:
                continue
            v_corr = np.mean([r['v_corr_lsr'] for r in d1[:n_p] + d2[:n_p]])
            v_sess_lsr = v_overlap + v_corr
            R_ov = R_sess[overlap_mask]
            R_interp = np.interp(
                v_lsr_grid[::-1], v_sess_lsr[::-1], R_ov[::-1],
                left=np.nan, right=np.nan,
            )[::-1]
            R_all.append(R_interp)
            session_labels.append(str(sess_label))
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
        result = {
            'R_overlap': R_mean,
            'n_pairs': total_pairs, 'ra': mean_ra, 'dec': mean_dec,
            'n_sessions': len(R_all),
            'R_per_session': R_all,
            'session_labels': session_labels,
            'pairs_per_session': pairs_per_session,
        }
        return result
    else:
        R_cat = np.concatenate(R_all, axis=0)
        R_mean = np.nanmean(R_cat, axis=0)
        if np.ndim(R_mean) == 0:
            return None
        return {
            'R_mean': R_mean, 'R_overlap': R_mean[overlap_mask],
            'n_pairs': total_pairs, 'ra': mean_ra, 'dec': mean_dec,
        }
