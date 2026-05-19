"""Neighbor-based quality assurance for HI survey cells.

Computes per-cell metrics (W, peak_v, SNR, noise) and flags outliers
using beam-weighted local plane fits against neighboring cells.

## peak_v false positives on bimodal spectra

Argmax-style peak velocities are unstable on cells whose spectra contain
two near-equal peaks (e.g. local arm + Perseus arm toward the outer
Galaxy, or local + tangent-velocity components in the inner Galaxy).
Tiny noise fluctuations decide which peak wins the global argmax, so the
reported peak_v can flip by tens of km/s between sessions even though
the underlying spectra agree. Concrete false positives observed in
`02_scan_load.ipynb`:

| Cell        | Primary peak     | Secondary peak    | Note                              |
|-------------|------------------|-------------------|-----------------------------------|
| (118, 0)    | -50 km/s, R=0.40 | 0   km/s, R=0.36  | Perseus + local, ratio ~0.95      |
| (120, 0)    | -52 km/s, R=0.40 | 0   km/s, R=0.37  | Perseus + local, ratio ~0.95      |
| (43,  -1)   | +58 km/s, R=0.46 | +35 km/s, R=0.43  | Inner-galaxy multi-component      |
| (234, 0)    | +45 km/s, R=0.58 | +22 km/s, R=0.38  | Outer arm + local, both sessions  |

In every case the primary peak fails the neighbor plane fit but the
secondary peak is consistent with neighbors. To absorb this, when a
cell's primary `peak_v` fails the neighbor test, the QA also checks the
second-most-prominent peak; if that peak is consistent with neighbors,
the flag is cleared and the cell is marked `bimodal=True`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


# -- Weighted statistics -------------------------------------------------

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of *values* with positive *weights*."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = 0.5 * cdf[-1]
    return values[np.searchsorted(cdf, cutoff)]


def weighted_mad(
    values: np.ndarray,
    weights: np.ndarray,
    center: float | None = None,
) -> float:
    """Weighted median absolute deviation."""
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.nan
    if center is None:
        center = weighted_median(values, weights)
    return weighted_median(np.abs(values - center), weights)


# -- Beam geometry --------------------------------------------------------

def beam_overlap_weight(sep_deg: float, hpbw_deg: float) -> float:
    """Normalized overlap of two identical Gaussian beams."""
    return np.exp(-2.0 * np.log(2.0) * (sep_deg / hpbw_deg) ** 2)


def local_tangent_offsets(
    gl0: float, gb0: float, gl1: float, gb1: float,
) -> tuple[float, float, float]:
    """Return (x, y, sep) in the local tangent plane at (gl0, gb0)."""
    dl = ((gl1 - gl0 + 180.0) % 360.0) - 180.0
    db = gb1 - gb0
    x = dl * np.cos(np.deg2rad(0.5 * (gb0 + gb1)))
    y = db
    sep = np.hypot(x, y)
    return x, y, sep


# -- Local plane fit ------------------------------------------------------

def fit_weighted_local_plane(
    target_gl: float,
    target_gb: float,
    neighbors: list[dict],
    value_key: str,
    *,
    sigma_floor: float,
    hpbw_deg: float,
) -> tuple:
    """Fit a weighted linear plane to neighbor values.

    Returns
    -------
    predicted_center, coeffs, resid_sigma, residuals, weights, resid_center
        or all-NaN 6-tuple when the fit is underdetermined.
    """
    rows = []
    values = []
    weights = []

    for n in neighbors:
        value = n.get(value_key, np.nan)
        if not np.isfinite(value):
            continue
        x, y, sep = local_tangent_offsets(target_gl, target_gb, n['gl'], n['gb'])
        weight = beam_overlap_weight(sep, hpbw_deg)
        if not np.isfinite(weight) or weight <= 0:
            continue
        rows.append([1.0, x, y])
        values.append(value)
        weights.append(weight)

    if len(values) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    design = np.asarray(rows)
    values = np.asarray(values)
    weights = np.asarray(weights)
    weighted_design = design * np.sqrt(weights)[:, None]
    if np.linalg.matrix_rank(weighted_design) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    coeffs, *_ = np.linalg.lstsq(
        weighted_design, values * np.sqrt(weights), rcond=None,
    )
    predicted_center = coeffs[0]
    residuals = values - design @ coeffs
    resid_center = weighted_median(residuals, weights)
    resid_mad = weighted_mad(residuals, weights, resid_center)
    resid_sigma = max(resid_mad * 1.4826, sigma_floor)
    return predicted_center, coeffs, resid_sigma, residuals, weights, resid_center


# -- Per-cell metrics -----------------------------------------------------

def compute_cell_metrics(
    cell_results_combined: dict,
    v_lsr_overlap: np.ndarray,
    dv_kms: float,
    *,
    min_valid_ch: int,
    noise_v_max_kms: float,
    signal_v_lo_kms: float,
    signal_v_hi_kms: float,
    smooth_kernel: int,
    peak_min_sep_kms: float,
    peak_prom_nsigma: float,
    min_noise_ch: int,
    spectrum_key: str,
) -> dict:
    """Compute W, peak_v, SNR, and noise for each cell.

    Operates on a single spectrum key (one pol or a caller-prepared
    summed key); call twice for per-pol metrics.

    Parameters
    ----------
    cell_results_combined : dict
        ``(gl, gb) -> {spectrum_key: array, 'n_pairs': int, ...}``.
    v_lsr_overlap : 1-D array
        LSR velocity grid matching the spectrum.
    dv_kms : float
        Channel width in km/s.
    spectrum_key : str
        Key holding the spectrum (e.g. ``'R_pol0'``).

    Returns
    -------
    dict
        ``(gl, gb) -> {metrics dict}``
    """
    cell_metrics: dict[tuple, dict] = {}

    for (gl, gb), cr in cell_results_combined.items():
        R = cr[spectrum_key]
        finite = np.isfinite(R)
        n_valid_ch = np.count_nonzero(finite)
        if n_valid_ch < min_valid_ch:
            continue

        # Integrated intensity W
        R_work = R.copy()
        if n_valid_ch >= 2:
            chans = np.arange(R_work.size)
            R_work[~finite] = np.interp(
                chans[~finite], chans[finite], R_work[finite],
            )
            W = np.sum(R_work) * dv_kms
        elif n_valid_ch == 1:
            W = R_work[finite][0] * dv_kms * R_work.size
        else:
            W = np.nan

        # Noise RMS from off-signal channels
        noise_mask = v_lsr_overlap < noise_v_max_kms
        noise_vals = R[noise_mask]
        noise_vals = noise_vals[np.isfinite(noise_vals)]
        noise_rms = (
            np.std(noise_vals) if noise_vals.size > min_noise_ch else np.nan
        )

        # Peak detection in signal window
        signal_mask = (v_lsr_overlap >= signal_v_lo_kms) & (v_lsr_overlap <= signal_v_hi_kms)
        v_sig = v_lsr_overlap[signal_mask]
        R_sig = R[signal_mask]
        sig_valid = np.isfinite(R_sig)

        peak_R = np.nan
        peak_v = np.nan
        peak_prom = np.nan
        peak_count = 0
        peak_v_2nd = np.nan
        peak_R_2nd = np.nan
        peak_prom_2nd = np.nan

        if sig_valid.sum() >= 3:
            sig_idx = np.arange(R_sig.size)
            R_fill = R_sig.copy()
            R_fill[~sig_valid] = np.interp(
                sig_idx[~sig_valid], sig_idx[sig_valid], R_sig[sig_valid],
            )

            kernel = np.ones(smooth_kernel) / smooth_kernel
            R_smooth = np.convolve(R_fill, kernel, mode='same')

            # find_peaks requires a Python int for `distance`.
            distance = max(1, int(round(peak_min_sep_kms / dv_kms)))
            prominence = (
                peak_prom_nsigma * noise_rms
                if np.isfinite(noise_rms) and noise_rms > 0
                else 0.0
            )
            peaks, props = find_peaks(
                R_smooth, distance=distance, prominence=prominence,
            )
            peak_count = len(peaks)

            if peak_count > 0:
                order = np.argsort(props['prominences'])[::-1]
                best = order[0]
                peak_idx = peaks[best]
                peak_R = R_fill[peak_idx]
                peak_v = v_sig[peak_idx]
                peak_prom = props['prominences'][best]
                if peak_count > 1:
                    second = order[1]
                    peak_idx_2 = peaks[second]
                    peak_R_2nd = R_fill[peak_idx_2]
                    peak_v_2nd = v_sig[peak_idx_2]
                    peak_prom_2nd = props['prominences'][second]
            else:
                peak_idx = np.nanargmax(R_fill)
                peak_R = R_fill[peak_idx]
                peak_v = v_sig[peak_idx]

        snr = (
            peak_R / noise_rms
            if np.isfinite(noise_rms) and noise_rms > 0 and np.isfinite(peak_R)
            else np.nan
        )

        cell_metrics[(gl, gb)] = {
            'gl': gl, 'gb': gb,
            'W': W, 'peak_R': peak_R, 'peak_v': peak_v,
            'peak_prom': peak_prom, 'peak_count': peak_count,
            'peak_v_2nd': peak_v_2nd, 'peak_R_2nd': peak_R_2nd,
            'peak_prom_2nd': peak_prom_2nd,
            'snr': snr, 'noise_rms': noise_rms,
            'n_pairs': cr['n_pairs'], 'n_valid_ch': n_valid_ch,
        }

    return cell_metrics


# -- Neighbor QA ----------------------------------------------------------


def neighbor_qa(
    cell_metrics: dict,
    *,
    dv_kms: float,
    hpbw_deg: float,
    neighbor_max_sep_deg: float,
    min_neighbors: int,
    w_z_thresh: float,
    w_frac_thresh: float,
    w_scale_floor: float,
    peak_v_z_thresh: float,
    peak_v_abs_thresh: float,
    peak_v_min_sigma: float,
    peak_v_scale_floor: float,
    bimodal_min_ratio: float,
) -> list[dict]:
    """Run neighbor-based QA on cell metrics.

    Compares each cell's W and peak_v to a beam-weighted local plane
    fit of its neighbors.  Flags cells that deviate beyond thresholds.

    Parameters
    ----------
    cell_metrics : dict
        ``(gl, gb) -> {metrics dict}`` from :func:`compute_cell_metrics`.
    dv_kms : float
        Channel width in km/s (used for W noise floor).

    Returns
    -------
    list of dict
        One dict per cell with all metrics plus neighbor QA fields
        (``W_flag``, ``peak_v_flag``, ``W_frac_resid``, etc.).
    """
    cell_keys = list(cell_metrics.keys())
    cell_coords = np.array(cell_keys)
    neighbor_cells: list[dict] = []

    # Precompute the full N x N separation matrix once. The per-row math is
    # identical to the original per-cell inline computation (same operand
    # order, same broadcasts), so neighbor selection is bit-for-bit
    # equivalent to the previous loop body.
    gl_col = cell_coords[:, 0]
    gb_col = cell_coords[:, 1]
    dl_mat = ((gl_col[None, :] - gl_col[:, None] + 180.0) % 360.0) - 180.0
    db_mat = gb_col[None, :] - gb_col[:, None]
    x_mat = dl_mat * np.cos(np.deg2rad(0.5 * (gb_col[None, :] + gb_col[:, None])))
    sep_mat = np.hypot(x_mat, db_mat)

    for i, (gl, gb) in enumerate(cell_keys):
        cell = cell_metrics.get((gl, gb))
        if cell is None:
            continue
        cell = dict(cell)  # copy so we don't mutate the input

        sep = sep_mat[i]
        neighbor_indices = np.where(
            (sep > 0) & (sep <= neighbor_max_sep_deg),
        )[0]

        neighbors = []
        for j in neighbor_indices:
            ngl, ngb = cell_keys[j]
            nc = cell_metrics.get((ngl, ngb))
            if nc is not None:
                neighbors.append({
                    'gl': ngl, 'gb': ngb,
                    'W': nc['W'], 'peak_v': nc['peak_v'],
                })

        cell['neighbor_count'] = len(neighbors)

        if len(neighbors) < min_neighbors:
            for k in ('W_local_pred', 'W_resid', 'W_frac_resid', 'W_sigma',
                       'W_z', 'W_grad_l', 'W_grad_b',
                       'peak_v_local_pred', 'peak_v_resid',
                       'peak_v_frac_resid', 'peak_v_abs_resid',
                       'peak_v_sigma', 'peak_v_z',
                       'peak_v_grad_l', 'peak_v_grad_b'):
                cell[k] = np.nan
            cell['W_flag'] = False
            cell['peak_v_flag'] = False
            neighbor_cells.append(cell)
            continue

        # W plane fit
        W_sigma_floor = (
            max(1e-6, cell['noise_rms'] * dv_kms * np.sqrt(cell['n_valid_ch']))
            if np.isfinite(cell['noise_rms'])
            else 1e-6
        )
        W_pred, W_coeffs, W_sigma, *_ = fit_weighted_local_plane(
            gl, gb, neighbors, 'W',
            sigma_floor=W_sigma_floor, hpbw_deg=hpbw_deg,
        )
        pv_pred, pv_coeffs, pv_sigma, *_ = fit_weighted_local_plane(
            gl, gb, neighbors, 'peak_v',
            sigma_floor=peak_v_min_sigma, hpbw_deg=hpbw_deg,
        )

        # W residuals
        cell['W_local_pred'] = W_pred
        cell['W_resid'] = (
            cell['W'] - W_pred
            if np.isfinite(cell['W']) and np.isfinite(W_pred)
            else np.nan
        )
        W_scale = max(abs(W_pred) if np.isfinite(W_pred) else 0, w_scale_floor)
        cell['W_frac_resid'] = (
            cell['W_resid'] / W_scale if np.isfinite(cell['W_resid']) else np.nan
        )
        cell['W_sigma'] = W_sigma
        cell['W_z'] = (
            cell['W_resid'] / W_sigma
            if np.isfinite(cell['W_resid']) and np.isfinite(W_sigma) and W_sigma > 0
            else np.nan
        )
        cell['W_grad_l'] = (
            W_coeffs[1] if np.ndim(W_coeffs) > 0 and len(W_coeffs) > 1
            else np.nan
        )
        cell['W_grad_b'] = (
            W_coeffs[2] if np.ndim(W_coeffs) > 0 and len(W_coeffs) > 2
            else np.nan
        )

        # peak_v residuals
        cell['peak_v_local_pred'] = pv_pred
        cell['peak_v_resid'] = (
            cell['peak_v'] - pv_pred
            if np.isfinite(cell['peak_v']) and np.isfinite(pv_pred)
            else np.nan
        )
        pv_scale = max(
            abs(pv_pred) if np.isfinite(pv_pred) else 0, peak_v_scale_floor,
        )
        cell['peak_v_frac_resid'] = (
            cell['peak_v_resid'] / pv_scale
            if np.isfinite(cell['peak_v_resid'])
            else np.nan
        )
        cell['peak_v_abs_resid'] = (
            abs(cell['peak_v_resid'])
            if np.isfinite(cell['peak_v_resid'])
            else np.nan
        )
        cell['peak_v_sigma'] = pv_sigma
        cell['peak_v_z'] = (
            cell['peak_v_resid'] / pv_sigma
            if np.isfinite(cell['peak_v_resid'])
            and np.isfinite(pv_sigma) and pv_sigma > 0
            else np.nan
        )
        cell['peak_v_grad_l'] = (
            pv_coeffs[1] if np.ndim(pv_coeffs) > 0 and len(pv_coeffs) > 1
            else np.nan
        )
        cell['peak_v_grad_b'] = (
            pv_coeffs[2] if np.ndim(pv_coeffs) > 0 and len(pv_coeffs) > 2
            else np.nan
        )

        # Flags
        cell['W_flag'] = (
            np.isfinite(cell['W_frac_resid'])
            and abs(cell['W_frac_resid']) > w_frac_thresh
            and np.isfinite(cell['W_z'])
            and abs(cell['W_z']) > w_z_thresh
        )
        primary_fails = (
            np.isfinite(cell['peak_v_abs_resid'])
            and cell['peak_v_abs_resid'] > peak_v_abs_thresh
            and np.isfinite(cell['peak_v_z'])
            and abs(cell['peak_v_z']) > peak_v_z_thresh
        )

        # Bimodal fallback: if the primary peak fails the neighbor plane
        # test but the second-most-prominent peak (1) is comparable in
        # amplitude to the primary and (2) is consistent with neighbors,
        # the cell is bimodal and the argmax simply flipped to the wrong
        # lobe. The amplitude floor (`bimodal_min_ratio`, default 0.68)
        # rejects cases where a weak secondary happens to align with
        # neighbors -- those are genuine outliers, not bimodal spectra.
        cell['bimodal'] = False
        cell['peak_v_2nd_resid'] = np.nan
        cell['peak_v_2nd_z'] = np.nan
        cell['peak_R_ratio'] = np.nan
        if primary_fails and np.isfinite(cell.get('peak_v_2nd', np.nan)) \
                and np.isfinite(pv_pred):
            R_primary = cell.get('peak_R', np.nan)
            R_secondary = cell.get('peak_R_2nd', np.nan)
            if np.isfinite(R_primary) and R_primary > 0 \
                    and np.isfinite(R_secondary):
                cell['peak_R_ratio'] = R_secondary / R_primary
            resid_2 = cell['peak_v_2nd'] - pv_pred
            z_2 = (resid_2 / pv_sigma
                   if np.isfinite(pv_sigma) and pv_sigma > 0 else np.nan)
            cell['peak_v_2nd_resid'] = resid_2
            cell['peak_v_2nd_z'] = z_2
            ratio_ok = (
                np.isfinite(cell['peak_R_ratio'])
                and cell['peak_R_ratio'] >= bimodal_min_ratio
            )
            secondary_passes = (
                abs(resid_2) <= peak_v_abs_thresh
                or (np.isfinite(z_2) and abs(z_2) <= peak_v_z_thresh)
            )
            if ratio_ok and secondary_passes:
                cell['bimodal'] = True
                primary_fails = False

        cell['peak_v_flag'] = primary_fails

        neighbor_cells.append(cell)

    return neighbor_cells


def flag_outlier_pairs(
    cell_pairs: dict,
    *,
    n_sigma: float,
    frac_thresh: float,
    min_pairs: int,
    spectrum_key: str,
) -> tuple[dict, list[dict]]:
    """Cross-session per-pair outlier filter using population MAD.

    For each cell, compute a robust population reference (median + MAD)
    over all per-pair spectra from all sessions for the given
    ``spectrum_key``, then flag any pair whose channels deviate from the
    reference too often.  Catches bad sessions that the within-session
    dump filter missed (drifting bandpass, transient line-region RFI,
    bad calibration).

    Pols are propagated separately, so call this once per pol (e.g.
    ``spectrum_key='R_lsr_pol0'`` then ``'R_lsr_pol1'``) -- each call
    consumes the output of the previous so a pair flagged on either pol
    is dropped.

    Parameters
    ----------
    cell_pairs : dict
        ``{(gl, gb): [{'session', 'pair_idx', spectrum_key: array}, ...]}``.
        Each pair's spectrum must be on a common (e.g. LSR) velocity grid so
        cross-session comparison is meaningful.
    n_sigma : float
        Per-channel MAD threshold; channels with ``|z| > n_sigma`` are
        counted as deviant.
    frac_thresh : float
        A pair is flagged if its deviant-channel fraction exceeds this.
    min_pairs : int
        Cells with fewer than this many pairs skip filtering and pass
        through (population statistics are not robust below the threshold).
    spectrum_key : str
        Per-pol key on each pair dict to test (e.g. ``'R_lsr_pol0'``).

    Returns
    -------
    viable_pairs_per_cell : dict
        ``{(gl, gb): [pair_dicts]}`` after rejection.
    flagged : list of dict
        Records ``{'gl', 'gb', 'session', 'pair_idx', 'deviant_frac',
        'spectrum_key'}`` for each rejected pair.
    """
    viable_pairs_per_cell: dict[tuple, list[dict]] = {}
    flagged: list[dict] = []

    for (gl, gb), pairs in cell_pairs.items():
        if len(pairs) < min_pairs:
            viable_pairs_per_cell[(gl, gb)] = list(pairs)
            continue

        R_stack = np.array([p[spectrum_key] for p in pairs])
        median = np.nanmedian(R_stack, axis=0)
        mad = np.nanmedian(np.abs(R_stack - median), axis=0) * 1.4826
        mad_safe = np.where(mad > 0, mad, np.nan)
        z = (R_stack - median) / mad_safe
        deviant_frac = np.nanmean(np.abs(z) > n_sigma, axis=1)

        viable: list[dict] = []
        for p, frac in zip(pairs, deviant_frac):
            if frac > frac_thresh:
                flagged.append({
                    'gl': gl,
                    'gb': gb,
                    'session': p['session'],
                    'pair_idx': p['pair_idx'],
                    'deviant_frac': frac,
                    'spectrum_key': spectrum_key,
                })
            else:
                viable.append(p)
        viable_pairs_per_cell[(gl, gb)] = viable

    return viable_pairs_per_cell, flagged


def combine_viable_pairs(
    viable_pairs_per_cell: dict,
    *,
    min_pairs: int,
    spectrum_key: str,
    out_key: str,
    cell_combined: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Average per-pair spectra into one spectrum per (gl, gb) cell.

    Operates on a single pol per call.  Pass ``cell_combined`` from a
    previous call to merge another pol's mean into the same dict.

    Parameters
    ----------
    viable_pairs_per_cell : dict
        ``(gl, gb) -> list of pair dicts`` (output of ``flag_outlier_pairs``).
    min_pairs : int
        Cells with fewer surviving pairs are still combined but flagged for
        reobservation.
    spectrum_key : str
        Key on each pair dict holding the spectrum to average (e.g.
        ``'R_lsr_pol0'``).
    out_key : str
        Key under which the averaged spectrum is stored on each cell entry
        (e.g. ``'R_pol0'``).
    cell_combined : dict or None
        Existing dict to update in place.  If ``None``, a new dict is
        created.  When merging multiple pols, reuse the dict from the first
        call so each entry accumulates per-pol keys.

    Returns
    -------
    cell_combined : dict
        ``(gl, gb) -> {**out_keys, 'n_pairs'}`` -- nanmean over surviving
        pairs for the requested pol.
    cells_insufficient_pairs : list of dict
        Records describing cells with fewer than ``min_pairs`` viable pairs
        (suitable for the reobserve list).
    """
    if cell_combined is None:
        cell_combined = {}
    cells_insufficient: list[dict] = []
    for (gl, gb), pairs in viable_pairs_per_cell.items():
        n_viable = len(pairs)
        if n_viable == 0:
            cells_insufficient.append({'l': gl, 'b': gb, 'n_viable': 0})
            continue
        entry = cell_combined.setdefault((gl, gb), {})
        entry['n_pairs'] = n_viable
        entry[out_key] = np.nanmean([p[spectrum_key] for p in pairs], axis=0)
        if n_viable < min_pairs:
            cells_insufficient.append({'l': gl, 'b': gb,
                                       'n_viable': n_viable})
    return cell_combined, cells_insufficient


def collect_reobserve(
    neighbor_cells: list[dict],
    cells_insufficient_pairs: list[dict],
) -> list[dict]:
    """Union neighbor-QA flags and pair-count failures into a reobserve list.

    Each returned record is ``{'l', 'b', 'reason'}`` with reasons joined by
    ``'+'`` when a cell trips multiple checks. Sorted by (b, l).
    """
    reobs_map: dict = {}
    for c in neighbor_cells:
        if not (c['W_flag'] or c['peak_v_flag']):
            continue
        flags = []
        if c['W_flag']:
            flags.append('W')
        if c['peak_v_flag']:
            flags.append('peak_v')
        key = (c['gl'], c['gb'])
        reobs_map[key] = {'l': c['gl'], 'b': c['gb'],
                          'reason': '+'.join(flags)}
    for c in cells_insufficient_pairs:
        key = (c['l'], c['b'])
        tag = f"insufficient_pairs(n={c['n_viable']})"
        if key in reobs_map:
            reobs_map[key]['reason'] += '+' + tag
        else:
            reobs_map[key] = {'l': c['l'], 'b': c['b'], 'reason': tag}
    return sorted(reobs_map.values(), key=lambda r: (r['b'], r['l']))


def detect_edge_clipped_cells(
    cell_combined: dict,
    *,
    v_lsr_overlap: np.ndarray,
    edge_kms: float,
    sigma_thresh: float,
    excluded=(),
    spectrum_key: str = 'R',
) -> list[dict]:
    """Flag cells whose line emission reaches either end of v_lsr_overlap.

    For each cell, estimate a robust per-channel noise (MAD/0.6745) from
    the interior of the spectrum (channels not within ``edge_kms`` of
    either v_LSR end), then test the maximum |R - median| in each edge
    region against ``sigma_thresh`` * noise.  A flagged cell is one whose
    line wing extends to the edge of the current frequency-switched
    overlap band -- the current LO pair is not wide enough to contain it.

    The ``edge`` field on each returned record is ``'high'`` when the
    hot edge is at the most-positive v_LSR end (line wing extends beyond
    the positive-velocity limit -> need a lower LO pair to cover more
    positive velocities) and ``'low'`` when at the most-negative end
    (need a higher LO pair).

    Parameters
    ----------
    cell_combined : dict
        ``(gl, gb) -> entry`` with ``entry[spectrum_key]`` a 1-D array on
        the ``v_lsr_overlap`` grid.
    v_lsr_overlap : np.ndarray
        LSR velocity grid, sorted descending (``v[0]`` is most positive).
    edge_kms : float
        Width of each edge region in km/s.
    sigma_thresh : float
        Sigma threshold (in MAD-sigma units) for flagging an edge.
    excluded : iterable of dict
        Cells to skip, as ``cells_insufficient_pairs`` records with
        ``'l'``, ``'b'`` keys.
    spectrum_key : str
        Entry key for the spectrum to test (default ``'R'`` -- the
        summed-pol Stokes-I proxy set by Section 5).

    Returns
    -------
    list of dict
        ``{'gl', 'gb', 'edge', 'edge_max_sigma', 'edge_max_v_kms'}`` per
        flagged cell.  Unsorted; the caller is expected to add
        ``suggested_pair`` and sort.
    """
    excluded_keys = {(e['l'], e['b']) for e in excluded}

    v = v_lsr_overlap
    if v.size < 4:
        return []
    # v is descending; v[0] is most positive ('high'), v[-1] most negative.
    high_mask = v >= (v[0] - edge_kms)
    low_mask  = v <= (v[-1] + edge_kms)
    interior_mask = ~(high_mask | low_mask)

    def _edge_max(R: np.ndarray, mask: np.ndarray, median: float, sigma: float):
        edge = R[mask]
        finite = np.isfinite(edge)
        if not finite.any():
            return 0.0, float('nan')
        dev = np.abs(edge[finite] - median) / sigma
        i = np.argmax(dev)
        v_edge = v[mask][finite][i]
        return float(dev[i]), float(v_edge)

    flagged: list[dict] = []
    for cell, entry in cell_combined.items():
        if cell in excluded_keys or spectrum_key not in entry:
            continue
        R = entry[spectrum_key]
        if R.shape != v.shape:
            continue
        interior = R[interior_mask]
        interior = interior[np.isfinite(interior)]
        if interior.size < 20:
            continue
        median = np.median(interior)
        mad = np.median(np.abs(interior - median))
        sigma = mad / 0.6745
        if not np.isfinite(sigma) or sigma <= 0:
            continue
        high_sig, high_v = _edge_max(R, high_mask, median, sigma)
        low_sig,  low_v  = _edge_max(R, low_mask,  median, sigma)
        if high_sig <= sigma_thresh and low_sig <= sigma_thresh:
            continue
        if high_sig >= low_sig:
            edge, sig, v_at = 'high', high_sig, high_v
        else:
            edge, sig, v_at = 'low', low_sig, low_v
        flagged.append({
            'gl': cell[0],
            'gb': cell[1],
            'edge': edge,
            'edge_max_sigma': sig,
            'edge_max_v_kms': v_at,
        })
    return flagged


def write_edge_recheck_manifest(
    cell_combined: dict,
    *,
    v_lsr_overlap: np.ndarray,
    edge_kms: float,
    sigma_thresh: float,
    excluded,
    current_pair_mhz: tuple[float, float],
    path,
    spectrum_key: str = 'R',
    source_notebook: str = 'main_scan_load.ipynb',
) -> dict:
    """Detect edge-clipped cells and write a recheck manifest as JSON.

    Builds the lower / higher LO pair from ``current_pair_mhz`` by
    shifting by ``Delta_LO = |F2 - F1|``.  Hot edges at the most-positive
    v_LSR end need the lower pair (more positive v_LSR coverage); hot
    edges at the most-negative end need the higher pair.  Manifest cells
    are sorted by ``|b|`` ascending, then by ``gl``.

    The manifest is consumed by ``scripts/main/edges.py``.

    Returns the manifest dict (also written to ``path``).
    """
    f1, f2 = current_pair_mhz
    delta_lo = abs(f2 - f1)
    lower_pair = (round(f1 - delta_lo, 4), f1)
    higher_pair = (f2, round(f2 + delta_lo, 4))

    flagged = detect_edge_clipped_cells(
        cell_combined,
        v_lsr_overlap=v_lsr_overlap,
        edge_kms=edge_kms,
        sigma_thresh=sigma_thresh,
        excluded=excluded,
        spectrum_key=spectrum_key,
    )

    manifest_cells = []
    for f in flagged:
        if f['edge'] == 'high':
            pair_label, pair = 'lower', lower_pair
        else:
            pair_label, pair = 'higher', higher_pair
        manifest_cells.append({
            **f,
            'abs_b': abs(f['gb']),
            'suggested_pair': pair_label,
            'suggested_lo_mhz': list(pair),
        })
    manifest_cells.sort(key=lambda c: (c['abs_b'], c['gl']))

    manifest = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_notebook': source_notebook,
        'current_lo_pair_mhz': [f1, f2],
        'delta_lo_mhz': delta_lo,
        'lower_pair_mhz': list(lower_pair),
        'higher_pair_mhz': list(higher_pair),
        'detection': {'edge_kms': edge_kms, 'sigma_thresh': sigma_thresh},
        'cells': manifest_cells,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
