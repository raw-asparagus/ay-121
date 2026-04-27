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

import numpy as np
from scipy.signal import find_peaks


# ── Weighted statistics ─────────────────────────────────────────────────

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of *values* with positive *weights*."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
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
    return float(values[np.searchsorted(cdf, cutoff)])


def weighted_mad(
    values: np.ndarray,
    weights: np.ndarray,
    center: float | None = None,
) -> float:
    """Weighted median absolute deviation."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return np.nan
    if center is None:
        center = weighted_median(values, weights)
    return weighted_median(np.abs(values - center), weights)


# ── Beam geometry ────────────────────────────────────────────────────────

def beam_overlap_weight(sep_deg: float, hpbw_deg: float = 3.4) -> float:
    """Normalized overlap of two identical Gaussian beams."""
    return float(np.exp(-2.0 * np.log(2.0) * (sep_deg / hpbw_deg) ** 2))


def local_tangent_offsets(
    gl0: float, gb0: float, gl1: float, gb1: float,
) -> tuple[float, float, float]:
    """Return (x, y, sep) in the local tangent plane at (gl0, gb0)."""
    dl = ((gl1 - gl0 + 180.0) % 360.0) - 180.0
    db = gb1 - gb0
    x = dl * np.cos(np.deg2rad(0.5 * (gb0 + gb1)))
    y = db
    sep = float(np.hypot(x, y))
    return float(x), float(y), sep


# ── Local plane fit ──────────────────────────────────────────────────────

def fit_weighted_local_plane(
    target_gl: float,
    target_gb: float,
    neighbors: list[dict],
    value_key: str,
    sigma_floor: float = 0.0,
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
        weight = beam_overlap_weight(sep)
        if not np.isfinite(weight) or weight <= 0:
            continue
        rows.append([1.0, x, y])
        values.append(value)
        weights.append(weight)

    if len(values) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    design = np.asarray(rows, dtype=float)
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
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


# ── Per-cell metrics ─────────────────────────────────────────────────────

def compute_cell_metrics(
    cell_results_combined: dict,
    v_lsr_overlap: np.ndarray,
    dv_kms: float,
) -> dict:
    """Compute W, peak_v, SNR, and noise for each cell.

    Parameters
    ----------
    cell_results_combined : dict
        ``(gl, gb) -> {'R_overlap': array, 'n_pairs': int, ...}``
    v_lsr_overlap : 1-D array
        LSR velocity grid matching ``R_overlap``.
    dv_kms : float
        Channel width in km/s.

    Returns
    -------
    dict
        ``(gl, gb) -> {metrics dict}``
    """
    cell_metrics: dict[tuple, dict] = {}

    for (gl, gb), cr in cell_results_combined.items():
        R = np.asarray(cr['R_overlap'], dtype=float)
        if R.ndim != 1 or R.size != len(v_lsr_overlap):
            continue

        finite = np.isfinite(R)
        n_valid_ch = int(np.count_nonzero(finite))
        if n_valid_ch < 8:
            continue

        # Integrated intensity W
        R_work = R.copy()
        if n_valid_ch >= 2:
            chans = np.arange(R_work.size)
            R_work[~finite] = np.interp(
                chans[~finite], chans[finite], R_work[finite],
            )
            W = float(np.sum(R_work) * dv_kms)
        elif n_valid_ch == 1:
            W = float(R_work[finite][0] * dv_kms * R_work.size)
        else:
            W = np.nan

        # Noise RMS from off-signal channels
        noise_mask = v_lsr_overlap < -100.0
        noise_vals = R[noise_mask]
        noise_vals = noise_vals[np.isfinite(noise_vals)]
        noise_rms = float(np.std(noise_vals)) if noise_vals.size > 5 else np.nan

        # Peak detection in signal window
        signal_mask = (v_lsr_overlap >= -80.0) & (v_lsr_overlap <= 60.0)
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

            kernel = np.ones(5) / 5.0
            R_smooth = np.convolve(R_fill, kernel, mode='same')

            distance = max(1, int(round(4.0 / dv_kms)))
            prominence = (
                2.5 * noise_rms
                if np.isfinite(noise_rms) and noise_rms > 0
                else 0.0
            )
            peaks, props = find_peaks(
                R_smooth, distance=distance, prominence=prominence,
            )
            peak_count = len(peaks)

            if peak_count > 0:
                order = np.argsort(props['prominences'])[::-1]
                best = int(order[0])
                peak_idx = int(peaks[best])
                peak_R = float(R_fill[peak_idx])
                peak_v = float(v_sig[peak_idx])
                peak_prom = float(props['prominences'][best])
                if peak_count > 1:
                    second = int(order[1])
                    peak_idx_2 = int(peaks[second])
                    peak_R_2nd = float(R_fill[peak_idx_2])
                    peak_v_2nd = float(v_sig[peak_idx_2])
                    peak_prom_2nd = float(props['prominences'][second])
            else:
                peak_idx = int(np.nanargmax(R_fill))
                peak_R = float(R_fill[peak_idx])
                peak_v = float(v_sig[peak_idx])

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


# ── Neighbor QA ──────────────────────────────────────────────────────────

# Default thresholds
HPBW_DEG = 3.4
NEIGHBOR_MAX_SEP_DEG = 4.5
MIN_NEIGHBORS = 5
W_Z_THRESH = 3.5
W_FRAC_THRESH = 0.30
W_SCALE_FLOOR = 5.0
PEAK_V_Z_THRESH = 4.0
PEAK_V_ABS_THRESH = 15.0
PEAK_V_MIN_SIGMA = 3.0
PEAK_V_SCALE_FLOOR = 20.0


def neighbor_qa(
    cell_metrics: dict,
    *,
    dv_kms: float,
    hpbw_deg: float = HPBW_DEG,
    neighbor_max_sep_deg: float = NEIGHBOR_MAX_SEP_DEG,
    min_neighbors: int = MIN_NEIGHBORS,
    w_z_thresh: float = W_Z_THRESH,
    w_frac_thresh: float = W_FRAC_THRESH,
    w_scale_floor: float = W_SCALE_FLOOR,
    peak_v_z_thresh: float = PEAK_V_Z_THRESH,
    peak_v_abs_thresh: float = PEAK_V_ABS_THRESH,
    peak_v_min_sigma: float = PEAK_V_MIN_SIGMA,
    peak_v_scale_floor: float = PEAK_V_SCALE_FLOOR,
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
    cell_coords = np.array(cell_keys, dtype=float)
    neighbor_cells: list[dict] = []

    for gl, gb in cell_keys:
        cell = cell_metrics.get((gl, gb))
        if cell is None:
            continue
        cell = dict(cell)  # copy so we don't mutate the input

        dl = ((cell_coords[:, 0] - gl + 180.0) % 360.0) - 180.0
        db = cell_coords[:, 1] - gb
        x = dl * np.cos(np.deg2rad(0.5 * (cell_coords[:, 1] + gb)))
        sep = np.hypot(x, db)
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
            gl, gb, neighbors, 'W', sigma_floor=W_sigma_floor,
        )
        pv_pred, pv_coeffs, pv_sigma, *_ = fit_weighted_local_plane(
            gl, gb, neighbors, 'peak_v', sigma_floor=peak_v_min_sigma,
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
        cell['W_flag'] = bool(
            np.isfinite(cell['W_frac_resid'])
            and abs(cell['W_frac_resid']) > w_frac_thresh
            and np.isfinite(cell['W_z'])
            and abs(cell['W_z']) > w_z_thresh
        )
        primary_fails = bool(
            np.isfinite(cell['peak_v_abs_resid'])
            and cell['peak_v_abs_resid'] > peak_v_abs_thresh
            and np.isfinite(cell['peak_v_z'])
            and abs(cell['peak_v_z']) > peak_v_z_thresh
        )

        # Bimodal fallback: if the primary peak fails the neighbor plane
        # test but the second-most-prominent peak passes, the cell is
        # bimodal and the argmax simply flipped to the wrong lobe.
        cell['bimodal'] = False
        cell['peak_v_2nd_resid'] = np.nan
        cell['peak_v_2nd_z'] = np.nan
        if primary_fails and np.isfinite(cell.get('peak_v_2nd', np.nan)) \
                and np.isfinite(pv_pred):
            resid_2 = cell['peak_v_2nd'] - pv_pred
            z_2 = (resid_2 / pv_sigma
                   if np.isfinite(pv_sigma) and pv_sigma > 0 else np.nan)
            cell['peak_v_2nd_resid'] = resid_2
            cell['peak_v_2nd_z'] = z_2
            secondary_passes = (
                abs(resid_2) <= peak_v_abs_thresh
                or (np.isfinite(z_2) and abs(z_2) <= peak_v_z_thresh)
            )
            if secondary_passes:
                cell['bimodal'] = True
                primary_fails = False

        cell['peak_v_flag'] = primary_fails

        neighbor_cells.append(cell)

    return neighbor_cells


def flag_outlier_pairs(
    cell_pairs: dict,
    *,
    n_sigma: float = 5.0,
    frac_thresh: float = 0.20,
    min_pairs: int = 3,
    spectrum_key: str = 'R_lsr',
) -> tuple[dict, list[dict]]:
    """Cross-session per-pair outlier filter using population MAD.

    For each cell, compute a robust population reference (median + MAD)
    over all per-pair spectra from all sessions, then flag any pair whose
    channels deviate from the reference too often.  Catches bad sessions
    that the within-session dump filter missed (drifting bandpass,
    transient line-region RFI, bad calibration).

    Parameters
    ----------
    cell_pairs : dict
        ``{(gl, gb): [{'session', 'pair_idx', spectrum_key: 1-D array}, ...]}``.
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
        Key under which each pair stores its spectrum.

    Returns
    -------
    viable_pairs_per_cell : dict
        ``{(gl, gb): [pair_dicts]}`` after rejection.
    flagged : list of dict
        Records ``{'gl', 'gb', 'session', 'pair_idx', 'deviant_frac'}``
        for each rejected pair.
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
                    'deviant_frac': float(frac),
                })
            else:
                viable.append(p)
        viable_pairs_per_cell[(gl, gb)] = viable

    return viable_pairs_per_cell, flagged
