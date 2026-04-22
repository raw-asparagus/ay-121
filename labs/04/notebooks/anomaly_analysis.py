#!/usr/bin/env python3
"""Comprehensive anomaly analysis of HI 21cm survey cell_results_combined.

Replicates the data loading pipeline from 02_scan_load.ipynb, then performs
per-cell quality metrics and anomaly detection.
"""

import sys
import os
import re
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

# Setup paths
sys.path.insert(0, '.')
sys.path.insert(0, '..')

os.chdir('/home/ikaros/projects/ay-121/labs/04/notebooks')

from utils import (flag_rfi_channels, flag_outlier_dumps, vlsr_correction,
                   compute_R_for_dumps, compute_cell_W, build_heatmap)

import astropy.coordinates as ac
import astropy.units as u_ast

# -- Constants ----------------------------------------------------------
SAMPLE_RATE_HZ = 2.56e6
NFFT = 1024
HI_REST_MHZ = 1420.405
C_KMS = 299792.458
RFI_WINDOW = 15
RFI_SIGMA = 5.0
SHAPE_DEV_THRESH = 0.10
SHAPE_FRAC_THRESH = 0.20
EDGE_TRIM_MHZ = 0.256

# -- 1. Load all scan dumps --------------------------------------------
print("=" * 70)
print("PHASE 1: Loading data")
print("=" * 70)

STREAMING_DIR = Path('../../../data/lab04/streaming')
DATA_RELEASES = ['DR1', 'DR2a', 'DR2b', 'DR3a', 'DR3b', 'DR4', 'DR5a', 'DR5b']

scan_dirs = []
for dr in DATA_RELEASES:
    dr_dir = STREAMING_DIR / dr
    if not dr_dir.exists():
        continue
    found = sorted(dr_dir.glob('scan_r*_c*'))
    print(f'  {dr}: {len(found)} cells')
    scan_dirs.extend(found)
print(f'  Total: {len(scan_dirs)} cells')

records = []
for d in scan_dirs:
    dr_label = d.parent.name
    for p in sorted(d.glob('*.npz')):
        with np.load(p, allow_pickle=True) as f:
            records.append({
                'path': p,
                'dr': dr_label,
                'target': str(f['target_name']),
                'corr00': f['corr00'].astype(float),
                'corr11': f['corr11'].astype(float),
                'lo_mhz': float(f['lo_freq_mhz']),
                'noise_on': bool(f['noise_on']),
                'time': float(f['time']),
                'alt': float(f['alt_deg']),
                'az': float(f['az_deg']),
                'ra': float(f['ra_deg']),
                'dec': float(f['dec_deg']),
            })

N = len(records)
print(f'  {N} total dumps loaded')

for r in records:
    m = re.match(r'scan_r(\d+)_c(\d+)', r['target'])
    if m:
        r['row'] = int(m.group(1))
        r['col'] = int(m.group(2))
    else:
        r['row'] = -1
        r['col'] = -1

lo_unique = sorted(set(r['lo_mhz'] for r in records if not r['noise_on']))

# -- 2. fftshift + Stokes I + RFI -------------------------------------
print("\nPHASE 2: Preprocessing (fftshift, Stokes I, medfilt RFI flagging)")
f_bb_mhz = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0/SAMPLE_RATE_HZ)) / 1e6

for r in records:
    r['corr00'] = np.fft.fftshift(r['corr00'])
    r['corr11'] = np.fft.fftshift(r['corr11'])
    r['stokes_I'] = r['corr00'] + r['corr11']

n_flagged = sum(flag_rfi_channels(r['stokes_I'], RFI_WINDOW, RFI_SIGMA) for r in records)
print(f'  RFI flagged: {n_flagged} samples across {N} dumps')

# -- 2a. Outlier dump filter -------------------------------------------
print("  Outlier dump filter...")
outlier_records = flag_outlier_dumps(records, SHAPE_DEV_THRESH, SHAPE_FRAC_THRESH)
print(f'  Removed {len(outlier_records)} outlier dumps, {len(records)} remaining')

# -- 3. Frequency-switched profiles -----------------------------------
print("\nPHASE 3: Frequency switching + LSR correction")
lo1, lo2 = lo_unique[0], lo_unique[1]
f_sky = lo1 + f_bb_mhz
f_sky_2 = lo2 + f_bb_mhz
f_overlap_lo = max(f_sky[0], f_sky_2[0]) + EDGE_TRIM_MHZ
f_overlap_hi = min(f_sky[-1], f_sky_2[-1]) - EDGE_TRIM_MHZ
overlap_mask = (f_sky >= f_overlap_lo) & (f_sky <= f_overlap_hi)
f_overlap = f_sky[overlap_mask]
v_overlap = C_KMS * (1 - f_overlap / HI_REST_MHZ)

print(f'  LO pair: ({lo1}, {lo2}) MHz')
print(f'  Overlap channels: {overlap_mask.sum()}')

# Assign galactic coords
for r in records:
    if r['row'] < 0:
        r['gl'], r['gb'] = None, None
        continue
    c = ac.SkyCoord(ra=r['ra'] * u_ast.deg, dec=r['dec'] * u_ast.deg, frame='icrs')
    r['gl'] = round(c.galactic.l.deg)
    r['gb'] = round(c.galactic.b.deg)

# LSR corrections
cell_dr_groups = defaultdict(list)
for r in records:
    if r.get('gl') is None or r['noise_on']:
        r['v_corr_lsr'] = 0.0
        continue
    cell_dr_groups[(r['dr'], r['gl'], r['gb'])].append(r)

for key, group in cell_dr_groups.items():
    r0 = group[0]
    mean_t = np.mean([r['time'] for r in group])
    v_corr = vlsr_correction(r0['ra'], r0['dec'], mean_t)
    for r in group:
        r['v_corr_lsr'] = v_corr

sci_vcorr = [r['v_corr_lsr'] for r in records
             if r.get('gl') is not None and not r['noise_on']]
mean_vcorr = np.mean(sci_vcorr)
v_lsr_overlap = v_overlap + mean_vcorr
print(f'  Velocity (LSR): [{v_lsr_overlap[-1]:.0f}, {v_lsr_overlap[0]:.0f}] km/s')

# Build combined results
all_pointings = sorted(set((r['gl'], r['gb']) for r in records if r['gl'] is not None))

cell_results_combined = {}
for gl, gb in all_pointings:
    sci_dumps = [r for r in records
                 if r['gl'] == gl and r['gb'] == gb and not r['noise_on']]
    result = compute_R_for_dumps(sci_dumps, lo1, lo2, overlap_mask, v_overlap,
                                 lsr_correct=True, v_lsr_grid=v_lsr_overlap)
    if result is not None:
        cell_results_combined[(gl, gb)] = result

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

print(f'  Combined cells: {len(cell_results_combined)}')

# -- Velocity grid info ------------------------------------------------
dv_kms = np.abs(np.median(np.diff(v_overlap)))
v_grid = v_lsr_overlap  # LSR velocity grid for spectra

# ======================================================================
# PHASE 4: ANOMALY ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 4: Anomaly Analysis")
print("=" * 70)

# -- 4a. Per-cell metrics ---------------------------------------------
print("\nComputing per-cell metrics...")

# Define signal-free region for noise estimation: v < -100 km/s
NOISE_V_MAX = -100.0  # km/s -- anything below this is signal-free
noise_mask = v_grid < NOISE_V_MAX
n_noise_ch = noise_mask.sum()
print(f'  Noise estimation region: v < {NOISE_V_MAX} km/s ({n_noise_ch} channels)')

# Signal region for peak detection
SIGNAL_V_MIN = -80.0
SIGNAL_V_MAX = 60.0
signal_mask = (v_grid >= SIGNAL_V_MIN) & (v_grid <= SIGNAL_V_MAX)
n_signal_ch = signal_mask.sum()
print(f'  Signal region: {SIGNAL_V_MIN} < v < {SIGNAL_V_MAX} km/s ({n_signal_ch} channels)')

cell_metrics = {}

for (l, b), cr in cell_results_combined.items():
    R = cr['R_overlap']
    n_pairs = cr['n_pairs']
    n_ch = len(R)

    # NaN statistics
    nan_mask = np.isnan(R)
    n_nan = nan_mask.sum()
    nan_frac = n_nan / n_ch

    # Noise RMS in signal-free region
    R_noise = R[noise_mask]
    R_noise_valid = R_noise[np.isfinite(R_noise)]
    if len(R_noise_valid) > 5:
        noise_rms = np.std(R_noise_valid)
        noise_mean = np.mean(R_noise_valid)
    else:
        noise_rms = np.nan
        noise_mean = np.nan

    # Peak in signal region
    R_signal = R[signal_mask].copy()
    v_signal = v_grid[signal_mask]
    R_signal_valid = np.where(np.isfinite(R_signal), R_signal, -np.inf)
    if np.any(np.isfinite(R_signal)):
        peak_idx = np.nanargmax(R_signal)
        peak_R = R_signal[peak_idx]
        peak_v = v_signal[peak_idx]
    else:
        peak_R = np.nan
        peak_v = np.nan

    # SNR
    snr = peak_R / noise_rms if noise_rms > 0 and np.isfinite(peak_R) else np.nan

    # Integrated intensity W (with NaN interpolation, same as compute_cell_W)
    R_copy = R.copy()
    valid = np.isfinite(R_copy)
    if valid.sum() >= 2:
        channels = np.arange(n_ch)
        R_copy[~valid] = np.interp(channels[~valid], channels[valid], R_copy[valid])
        W = np.sum(R_copy) * dv_kms
    elif valid.sum() == 1:
        W = R_copy[valid][0] * dv_kms * n_ch
    else:
        W = np.nan

    # Baseline ripple: RMS of residual after removing a low-order polynomial
    # in the noise region
    if len(R_noise_valid) > 10:
        v_noise_valid = v_grid[noise_mask][np.isfinite(R_noise)]
        try:
            poly_coeffs = np.polyfit(v_noise_valid, R_noise_valid, 2)
            poly_val = np.polyval(poly_coeffs, v_noise_valid)
            baseline_residual_rms = np.std(R_noise_valid - poly_val)
        except Exception:
            baseline_residual_rms = noise_rms
    else:
        baseline_residual_rms = np.nan

    cell_metrics[(l, b)] = {
        'peak_R': peak_R,
        'peak_v': peak_v,
        'W': W,
        'snr': snr,
        'n_nan': n_nan,
        'nan_frac': nan_frac,
        'noise_rms': noise_rms,
        'noise_mean': noise_mean,
        'baseline_residual_rms': baseline_residual_rms,
        'n_pairs': n_pairs,
        'n_ch': n_ch,
    }

print(f'  Computed metrics for {len(cell_metrics)} cells')

# -- 4b. Summary statistics -------------------------------------------
print("\n--- SUMMARY STATISTICS ---")
all_W = np.array([m['W'] for m in cell_metrics.values()])
all_snr = np.array([m['snr'] for m in cell_metrics.values()])
all_peak_v = np.array([m['peak_v'] for m in cell_metrics.values()])
all_peak_R = np.array([m['peak_R'] for m in cell_metrics.values()])
all_noise_rms = np.array([m['noise_rms'] for m in cell_metrics.values()])
all_nan_frac = np.array([m['nan_frac'] for m in cell_metrics.values()])
all_n_pairs = np.array([m['n_pairs'] for m in cell_metrics.values()])

def pstats(name, arr):
    v = arr[np.isfinite(arr)]
    if len(v) == 0:
        print(f'  {name}: no valid values')
        return
    print(f'  {name}: median={np.median(v):.4f}, mean={np.mean(v):.4f}, '
          f'std={np.std(v):.4f}, min={np.min(v):.4f}, max={np.max(v):.4f}')

pstats('W (integrated intensity)', all_W)
pstats('SNR', all_snr)
pstats('Peak velocity [km/s]', all_peak_v)
pstats('Peak R', all_peak_R)
pstats('Noise RMS', all_noise_rms)
pstats('NaN fraction', all_nan_frac)
pstats('n_pairs', all_n_pairs)

# -- 4c. Anomaly detection --------------------------------------------
print("\n" + "=" * 70)
print("ANOMALY DETECTION")
print("=" * 70)

anomalies = defaultdict(list)  # (l, b) -> list of anomaly descriptions

# --- (1) Negative W values ---
print("\n--- CHECK 1: Negative W (should not happen for real emission) ---")
neg_W_count = 0
for (l, b), m in cell_metrics.items():
    if m['W'] < 0:
        anomalies[(l, b)].append(f"Negative W = {m['W']:.2f} km/s")
        neg_W_count += 1
print(f'  Found {neg_W_count} cells with W < 0')

# --- (2) Excessive NaN fraction ---
print("\n--- CHECK 2: Excessive NaN fraction ---")
NAN_THRESH = 0.15  # >15% NaN channels is suspicious
high_nan_count = 0
for (l, b), m in cell_metrics.items():
    if m['nan_frac'] > NAN_THRESH:
        anomalies[(l, b)].append(f"High NaN fraction = {m['nan_frac']:.1%} ({m['n_nan']}/{m['n_ch']})")
        high_nan_count += 1
print(f'  Found {high_nan_count} cells with NaN fraction > {NAN_THRESH:.0%}')

# --- (3) W outliers relative to neighbors at similar b ---
print("\n--- CHECK 3: W outliers relative to galactic latitude band ---")
# Group cells by b, compute median W per band, flag > 3 sigma outliers
b_groups = defaultdict(list)
for (l, b), m in cell_metrics.items():
    if np.isfinite(m['W']):
        b_groups[b].append((l, m['W']))

W_outlier_count = 0
for b_val, entries in b_groups.items():
    if len(entries) < 5:
        continue
    W_arr = np.array([w for _, w in entries])
    med = np.median(W_arr)
    mad = np.median(np.abs(W_arr - med))
    sigma = mad * 1.4826  # MAD -> sigma
    if sigma < 1e-6:
        continue
    for l_val, w_val in entries:
        z_score = (w_val - med) / sigma
        if abs(z_score) > 3.5:
            direction = "high" if z_score > 0 else "low"
            anomalies[(l_val, b_val)].append(
                f"W outlier ({direction}): W={w_val:.2f}, band median={med:.2f}, "
                f"z={z_score:.1f}sigma"
            )
            W_outlier_count += 1
print(f'  Found {W_outlier_count} cells with anomalous W (|z| > 3.5 in latitude band)')

# --- (4) Anomalous peak velocity relative to neighbors ---
print("\n--- CHECK 4: Peak velocity outliers ---")
# For cells along the galactic plane (|b| <= 4), v_peak should vary smoothly with l
# Group by b, fit a smooth trend, flag outliers
v_outlier_count = 0
for b_val, entries_raw in b_groups.items():
    lb_entries = [(l, cell_metrics[(l, b_val)]['peak_v']) for l, _ in entries_raw
                  if np.isfinite(cell_metrics[(l, b_val)]['peak_v'])]
    if len(lb_entries) < 5:
        continue
    ls = np.array([l for l, v in lb_entries])
    vs = np.array([v for l, v in lb_entries])

    # Use rolling median (window ~5 neighbors) to get local expected velocity
    sort_idx = np.argsort(ls)
    ls_sorted = ls[sort_idx]
    vs_sorted = vs[sort_idx]

    NEIGHBOR_HALF = 3
    for i in range(len(ls_sorted)):
        lo = max(0, i - NEIGHBOR_HALF)
        hi = min(len(ls_sorted), i + NEIGHBOR_HALF + 1)
        neighbors = np.concatenate([vs_sorted[lo:i], vs_sorted[i+1:hi]])
        if len(neighbors) < 3:
            continue
        local_med = np.median(neighbors)
        local_mad = np.median(np.abs(neighbors - local_med))
        local_sigma = max(local_mad * 1.4826, 3.0)  # min 3 km/s expected scatter
        deviation = abs(vs_sorted[i] - local_med)
        if deviation > 4 * local_sigma:
            anomalies[(ls_sorted[i], b_val)].append(
                f"Peak velocity outlier: v_peak={vs_sorted[i]:.1f} km/s, "
                f"local median={local_med:.1f} km/s, dev={deviation:.1f} km/s "
                f"({deviation/local_sigma:.1f}sigma)"
            )
            v_outlier_count += 1
print(f'  Found {v_outlier_count} cells with anomalous peak velocity')

# --- (5) Poor SNR relative to n_pairs ---
print("\n--- CHECK 5: Anomalously low SNR given n_pairs ---")
# Expect SNR ~ sqrt(n_pairs) * SNR_per_pair
# Group by latitude band since expected signal varies with b
snr_outlier_count = 0
for b_val in set(b for _, b in cell_metrics.keys()):
    band_cells = [(l, cell_metrics[(l, b_val)])
                  for l in range(0, 360)
                  if (l, b_val) in cell_metrics and np.isfinite(cell_metrics[(l, b_val)]['snr'])]
    if len(band_cells) < 5:
        continue
    # Normalize SNR by sqrt(n_pairs) for fair comparison
    snr_norm = np.array([m['snr'] / np.sqrt(m['n_pairs']) for _, m in band_cells])
    med_snr_norm = np.median(snr_norm)
    mad_snr_norm = np.median(np.abs(snr_norm - med_snr_norm))
    sigma_snr = mad_snr_norm * 1.4826
    if sigma_snr < 0.1:
        continue
    for i, (l_val, m) in enumerate(band_cells):
        z = (snr_norm[i] - med_snr_norm) / sigma_snr
        if z < -3.5:  # only flag low SNR
            anomalies[(l_val, b_val)].append(
                f"Low SNR: SNR={m['snr']:.1f} with {m['n_pairs']} pairs "
                f"(norm={snr_norm[i]:.2f}, band median={med_snr_norm:.2f}, z={z:.1f}sigma)"
            )
            snr_outlier_count += 1
print(f'  Found {snr_outlier_count} cells with anomalously low SNR')

# --- (6) Baseline ripple / artifacts ---
print("\n--- CHECK 6: Baseline ripple ---")
# Compare baseline_residual_rms to noise_rms; if residual is still high
# after removing a polynomial, there may be ripple
ripple_count = 0
all_resid = np.array([m['baseline_residual_rms'] for m in cell_metrics.values()])
all_resid_valid = all_resid[np.isfinite(all_resid)]
resid_med = np.median(all_resid_valid)
resid_mad = np.median(np.abs(all_resid_valid - resid_med))
resid_sigma = resid_mad * 1.4826

for (l, b), m in cell_metrics.items():
    if np.isfinite(m['baseline_residual_rms']):
        z = (m['baseline_residual_rms'] - resid_med) / resid_sigma if resid_sigma > 0 else 0
        if z > 4.0:
            anomalies[(l, b)].append(
                f"Baseline ripple: residual RMS={m['baseline_residual_rms']:.5f} "
                f"(median={resid_med:.5f}, z={z:.1f}sigma)"
            )
            ripple_count += 1
print(f'  Found {ripple_count} cells with baseline ripple')

# --- (7) Very low n_pairs (incomplete cells) ---
print("\n--- CHECK 7: Incomplete cells (low n_pairs) ---")
low_pairs_count = 0
for (l, b), m in cell_metrics.items():
    if m['n_pairs'] < 4:
        anomalies[(l, b)].append(
            f"Low pair count: n_pairs={m['n_pairs']} (target=4)"
        )
        low_pairs_count += 1
print(f'  Found {low_pairs_count} cells with n_pairs < 4')

# --- (8) Unusually high noise RMS ---
print("\n--- CHECK 8: High noise RMS ---")
noise_rms_all = np.array([m['noise_rms'] for m in cell_metrics.values()])
noise_rms_valid = noise_rms_all[np.isfinite(noise_rms_all)]
nrms_med = np.median(noise_rms_valid)
nrms_mad = np.median(np.abs(noise_rms_valid - nrms_med))
nrms_sigma = nrms_mad * 1.4826
high_noise_count = 0
for (l, b), m in cell_metrics.items():
    if np.isfinite(m['noise_rms']):
        z = (m['noise_rms'] - nrms_med) / nrms_sigma if nrms_sigma > 0 else 0
        if z > 4.0:
            anomalies[(l, b)].append(
                f"High noise RMS: {m['noise_rms']:.5f} "
                f"(median={nrms_med:.5f}, z={z:.1f}sigma)"
            )
            high_noise_count += 1
print(f'  Found {high_noise_count} cells with unusually high noise RMS')

# --- (9) Non-zero baseline mean (DC offset in noise region) ---
print("\n--- CHECK 9: Baseline DC offset ---")
noise_means = np.array([m['noise_mean'] for m in cell_metrics.values()])
nm_valid = noise_means[np.isfinite(noise_means)]
nm_med = np.median(nm_valid)
nm_mad = np.median(np.abs(nm_valid - nm_med))
nm_sigma = nm_mad * 1.4826
dc_offset_count = 0
for (l, b), m in cell_metrics.items():
    if np.isfinite(m['noise_mean']):
        z = (m['noise_mean'] - nm_med) / nm_sigma if nm_sigma > 0 else 0
        if abs(z) > 4.0:
            anomalies[(l, b)].append(
                f"Baseline DC offset: mean={m['noise_mean']:.5f} "
                f"(global median={nm_med:.5f}, z={z:.1f}sigma)"
            )
            dc_offset_count += 1
print(f'  Found {dc_offset_count} cells with baseline DC offset')

# ======================================================================
# DETAILED ANOMALY REPORT
# ======================================================================
print("\n" + "=" * 70)
print("DETAILED ANOMALY REPORT")
print("=" * 70)

n_anomalous = len(anomalies)
n_total = len(cell_metrics)
print(f'\nTotal cells analyzed: {n_total}')
print(f'Cells with at least one anomaly: {n_anomalous} ({n_anomalous/n_total:.1%})')

# Sort by number of anomalies (most problematic first)
sorted_anomalies = sorted(anomalies.items(), key=lambda x: -len(x[1]))

for (l, b), issues in sorted_anomalies:
    m = cell_metrics[(l, b)]
    print(f'\n  ({l:3d}, {b:+3d})  n_pairs={m["n_pairs"]}  SNR={m["snr"]:.1f}  '
          f'W={m["W"]:.2f}  peak_v={m["peak_v"]:.1f} km/s  '
          f'NaN={m["nan_frac"]:.1%}  noise_rms={m["noise_rms"]:.5f}')
    for issue in issues:
        print(f'    -> {issue}')

# -- Anomaly category summary -----------------------------------------
print("\n" + "=" * 70)
print("ANOMALY CATEGORY SUMMARY")
print("=" * 70)
print(f'  Negative W:                {neg_W_count}')
print(f'  Excessive NaN:             {high_nan_count}')
print(f'  W outlier (lat. band):     {W_outlier_count}')
print(f'  Peak velocity outlier:     {v_outlier_count}')
print(f'  Low SNR (given n_pairs):   {snr_outlier_count}')
print(f'  Baseline ripple:           {ripple_count}')
print(f'  Incomplete (n_pairs < 4):  {low_pairs_count}')
print(f'  High noise RMS:            {high_noise_count}')
print(f'  Baseline DC offset:        {dc_offset_count}')
print(f'  ---')
print(f'  Total anomalous cells:     {n_anomalous} / {n_total}')

# -- Top 10 worst cells -----------------------------------------------
print("\n" + "=" * 70)
print("TOP 10 WORST CELLS (by number of anomaly flags)")
print("=" * 70)
for i, ((l, b), issues) in enumerate(sorted_anomalies[:10]):
    m = cell_metrics[(l, b)]
    print(f'\n  #{i+1}: (l={l}, b={b:+d})  [{len(issues)} flags]')
    print(f'       n_pairs={m["n_pairs"]}, SNR={m["snr"]:.1f}, W={m["W"]:.2f}, '
          f'peak_v={m["peak_v"]:.1f} km/s')
    for issue in issues:
        print(f'       - {issue}')

# -- Cells that are likely OK but worth double-checking ----------------
print("\n" + "=" * 70)
print("SPATIAL PATTERNS")
print("=" * 70)

# Check if anomalies cluster in specific DRs
print("\n--- Anomalous cells by galactic latitude ---")
b_anomaly_count = defaultdict(int)
b_total_count = defaultdict(int)
for (l, b) in cell_metrics:
    b_total_count[b] += 1
for (l, b) in anomalies:
    b_anomaly_count[b] += 1

for b_val in sorted(b_total_count.keys()):
    n_anom = b_anomaly_count.get(b_val, 0)
    n_tot = b_total_count[b_val]
    bar = '#' * n_anom
    print(f'  b={b_val:+3d}: {n_anom:3d}/{n_tot:3d} anomalous  {bar}')

# Check longitude distribution
print("\n--- Anomalous cells by galactic longitude (10-deg bins) ---")
l_bins = range(60, 300, 10)
for l_lo in l_bins:
    l_hi = l_lo + 10
    n_tot = sum(1 for (l, b) in cell_metrics if l_lo <= l < l_hi)
    n_anom = sum(1 for (l, b) in anomalies if l_lo <= l < l_hi)
    if n_tot > 0:
        bar = '#' * n_anom
        print(f'  l=[{l_lo:3d},{l_hi:3d}): {n_anom:3d}/{n_tot:3d} anomalous  {bar}')

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
