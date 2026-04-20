#!/usr/bin/env python3
"""Compare two RFI flagging pipelines on the HI survey data.

Pipeline A (current):
    Raw -> fftshift -> DC bin mask -> sky-freq mask -> Stokes I
    -> rolling median RFI (pandas) -> outlier dump removal -> edge trim

Pipeline B (proposed):
    Raw -> fftshift -> Stokes I -> medfilt RFI (scipy) -> outlier dump removal
    -> edge trim

Metrics compared:
    - NaN fraction per cell
    - Noise RMS in signal-free region (v < -100 km/s)
    - Peak SNR in signal region
    - Integrated intensity W
    - Number of outlier dumps removed
    - Runtime
"""

import sys
import os
import re
import copy
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import medfilt

sys.path.insert(0, '.')
sys.path.insert(0, '..')

os.chdir('/home/ikaros/projects/ay-121/labs/04/notebooks')

from utils import (flag_rfi_channels, flag_outlier_dumps, vlsr_correction,
                   compute_R_for_dumps, compute_cell_W, build_heatmap)
import astropy.coordinates as ac
import astropy.units as u_ast

# ── Constants ─────────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 2.56e6
NFFT = 1024
HI_REST_MHZ = 1420.405
C_KMS = 299792.458
DC_BIN = NFFT // 2
RFI_WINDOW = 15
RFI_SIGMA = 5.0
SHAPE_DEV_THRESH = 0.10
SHAPE_FRAC_THRESH = 0.20
EDGE_TRIM_MHZ = 0.256

NOISE_V_MAX = -100.0   # km/s boundary for noise region
SIGNAL_V_MIN = -80.0
SIGNAL_V_MAX = 60.0

# ── 1. Load raw data (shared) ────────────────────────────────────────
print("=" * 70)
print("Loading data")
print("=" * 70)

STREAMING_DIR = Path('../../../data/lab04/streaming')
DATA_RELEASES = ['DR1', 'DR2a', 'DR2b', 'DR3a', 'DR3b', 'DR4a', 'DR4b',
                 'DR5a', 'DR5b']

scan_dirs = []
for dr in DATA_RELEASES:
    dr_dir = STREAMING_DIR / dr
    if not dr_dir.exists():
        continue
    found = sorted(dr_dir.glob('scan_r*_c*'))
    print(f'  {dr}: {len(found)} cells')
    scan_dirs.extend(found)
print(f'  Total: {len(scan_dirs)} cell dirs')

raw_records = []
for d in scan_dirs:
    dr_label = d.parent.name
    for p in sorted(d.glob('*.npz')):
        with np.load(p, allow_pickle=True) as f:
            raw_records.append({
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

N = len(raw_records)
print(f'  {N} total dumps loaded')

for r in raw_records:
    m = re.match(r'scan_r(\d+)_c(\d+)', r['target'])
    if m:
        r['row'] = int(m.group(1))
        r['col'] = int(m.group(2))
    else:
        r['row'] = -1
        r['col'] = -1

f_bb_mhz = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1.0 / SAMPLE_RATE_HZ)) / 1e6


# ── Helper: medfilt-based RFI flagger ────────────────────────────────
def flag_rfi_medfilt(spectrum, window=15, sigma_thresh=5.0):
    """Flag RFI using scipy.signal.medfilt (no NaN handling)."""
    kernel = window if window % 2 == 1 else window + 1
    local_med = medfilt(spectrum, kernel_size=kernel)
    resid = spectrum - local_med
    mad = np.median(np.abs(resid))
    sigma = mad / 0.6745
    bad = np.abs(resid) > sigma_thresh * sigma
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        spectrum[bad] = np.nan
    return n_bad


# ── Helper: compute cell metrics ─────────────────────────────────────
def compute_cell_metrics(cell_results, v_grid):
    noise_mask = v_grid < NOISE_V_MAX
    signal_mask = (v_grid >= SIGNAL_V_MIN) & (v_grid <= SIGNAL_V_MAX)
    dv_kms = np.abs(np.median(np.diff(v_grid)))

    metrics = {}
    for (gl, gb), cr in cell_results.items():
        R = cr['R_overlap']
        n_ch = len(R)
        nan_mask = np.isnan(R)
        nan_frac = nan_mask.sum() / n_ch

        R_noise = R[noise_mask]
        R_noise_valid = R_noise[np.isfinite(R_noise)]
        if len(R_noise_valid) > 5:
            noise_rms = np.std(R_noise_valid)
        else:
            noise_rms = np.nan

        R_signal = R[signal_mask].copy()
        v_signal = v_grid[signal_mask]
        if np.any(np.isfinite(R_signal)):
            peak_idx = np.nanargmax(R_signal)
            peak_R = R_signal[peak_idx]
            peak_v = v_signal[peak_idx]
        else:
            peak_R = np.nan
            peak_v = np.nan

        snr = peak_R / noise_rms if noise_rms > 0 and np.isfinite(peak_R) else np.nan

        R_copy = R.copy()
        valid = np.isfinite(R_copy)
        if valid.sum() >= 2:
            channels = np.arange(n_ch)
            R_copy[~valid] = np.interp(channels[~valid], channels[valid], R_copy[valid])
            W = np.sum(R_copy) * dv_kms
        else:
            W = np.nan

        metrics[(gl, gb)] = {
            'snr': snr, 'noise_rms': noise_rms, 'nan_frac': nan_frac,
            'peak_R': peak_R, 'peak_v': peak_v, 'W': W,
            'n_pairs': cr['n_pairs'],
        }
    return metrics


# ── Helper: run full pipeline from fftshifted records ────────────────
def run_pipeline(records, label):
    lo_unique = sorted(set(r['lo_mhz'] for r in records if not r['noise_on']))
    lo1, lo2 = lo_unique[0], lo_unique[1]
    f_sky = lo1 + f_bb_mhz
    f_sky_2 = lo2 + f_bb_mhz
    f_overlap_lo = max(f_sky[0], f_sky_2[0]) + EDGE_TRIM_MHZ
    f_overlap_hi = min(f_sky[-1], f_sky_2[-1]) - EDGE_TRIM_MHZ
    overlap_mask = (f_sky >= f_overlap_lo) & (f_sky <= f_overlap_hi)
    v_overlap = C_KMS * (1 - f_sky[overlap_mask] / HI_REST_MHZ)

    # Galactic coords
    for r in records:
        if r['row'] < 0:
            r['gl'], r['gb'] = None, None
            continue
        c = ac.SkyCoord(ra=r['ra'] * u_ast.deg, dec=r['dec'] * u_ast.deg,
                        frame='icrs')
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

    # Build combined results
    all_pointings = sorted(set((r['gl'], r['gb']) for r in records
                               if r['gl'] is not None))
    cell_results = {}
    for gl, gb in all_pointings:
        sci_dumps = [r for r in records
                     if r['gl'] == gl and r['gb'] == gb and not r['noise_on']]
        result = compute_R_for_dumps(sci_dumps, lo1, lo2, overlap_mask,
                                     v_overlap, lsr_correct=True,
                                     v_lsr_grid=v_lsr_overlap)
        if result is not None:
            cell_results[(gl, gb)] = result

    metrics = compute_cell_metrics(cell_results, v_lsr_overlap)
    return metrics, v_lsr_overlap


# ══════════════════════════════════════════════════════════════════════
# PIPELINE A: Current (DC mask + sky-freq mask + pandas rolling median)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PIPELINE A: Current (DC mask + sky-freq mask + pandas rolling median)")
print("=" * 70)

records_a = copy.deepcopy(raw_records)

t0 = time.perf_counter()

# fftshift + DC mask + sky-freq mask
for r in records_a:
    r['corr00'] = np.fft.fftshift(r['corr00'])
    r['corr11'] = np.fft.fftshift(r['corr11'])
    r['corr00'][DC_BIN] = np.nan
    r['corr11'][DC_BIN] = np.nan
    if r['lo_mhz'] == 1420.0:
        for ch in [DC_BIN - 2, DC_BIN - 1]:
            r['corr00'][ch] = np.nan
            r['corr11'][ch] = np.nan
    r['stokes_I'] = r['corr00'] + r['corr11']

n_rfi_a = sum(flag_rfi_channels(r['stokes_I'], RFI_WINDOW, RFI_SIGMA)
              for r in records_a)
outliers_a = flag_outlier_dumps(records_a, SHAPE_DEV_THRESH, SHAPE_FRAC_THRESH)
t_preprocess_a = time.perf_counter() - t0

print(f'  RFI channels flagged: {n_rfi_a}')
print(f'  Outlier dumps removed: {len(outliers_a)}')
print(f'  Preprocessing time: {t_preprocess_a:.2f} s')

t0 = time.perf_counter()
metrics_a, v_grid_a = run_pipeline(records_a, 'A')
t_pipeline_a = time.perf_counter() - t0
print(f'  Pipeline time: {t_pipeline_a:.2f} s')
print(f'  Cells produced: {len(metrics_a)}')

# ══════════════════════════════════════════════════════════════════════
# PIPELINE B: Proposed (medfilt on raw per-pol, no explicit DC/sky mask)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PIPELINE B: Proposed (scipy.signal.medfilt, no DC/sky mask)")
print("=" * 70)

records_b = copy.deepcopy(raw_records)

t0 = time.perf_counter()

# fftshift only, then Stokes I, then medfilt RFI
for r in records_b:
    r['corr00'] = np.fft.fftshift(r['corr00'])
    r['corr11'] = np.fft.fftshift(r['corr11'])
    r['stokes_I'] = r['corr00'] + r['corr11']

n_rfi_b = sum(flag_rfi_medfilt(r['stokes_I'], RFI_WINDOW, RFI_SIGMA)
              for r in records_b)
outliers_b = flag_outlier_dumps(records_b, SHAPE_DEV_THRESH, SHAPE_FRAC_THRESH)
t_preprocess_b = time.perf_counter() - t0

print(f'  RFI channels flagged: {n_rfi_b}')
print(f'  Outlier dumps removed: {len(outliers_b)}')
print(f'  Preprocessing time: {t_preprocess_b:.2f} s')

t0 = time.perf_counter()
metrics_b, v_grid_b = run_pipeline(records_b, 'B')
t_pipeline_b = time.perf_counter() - t0
print(f'  Pipeline time: {t_pipeline_b:.2f} s')
print(f'  Cells produced: {len(metrics_b)}')

# ══════════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("HEAD-TO-HEAD COMPARISON")
print("=" * 70)

# Cells in common
common_keys = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
print(f'\nCells in common: {len(common_keys)}')
only_a = set(metrics_a.keys()) - set(metrics_b.keys())
only_b = set(metrics_b.keys()) - set(metrics_a.keys())
if only_a:
    print(f'  Only in A: {len(only_a)} — e.g. {sorted(only_a)[:5]}')
if only_b:
    print(f'  Only in B: {len(only_b)} — e.g. {sorted(only_b)[:5]}')

# Aggregate stats
def agg(metric_name, metrics_dict, keys):
    vals = np.array([metrics_dict[k][metric_name] for k in keys])
    v = vals[np.isfinite(vals)]
    return np.median(v), np.mean(v), np.std(v), np.min(v), np.max(v), len(v)

header = f'{"Metric":<22} {"Pipeline":>8}  {"median":>10} {"mean":>10} {"std":>10} {"min":>10} {"max":>10} {"N":>5}'
print(f'\n{header}')
print('-' * len(header))

for metric in ['snr', 'noise_rms', 'nan_frac', 'peak_R', 'W']:
    med_a, mean_a, std_a, min_a, max_a, n_a = agg(metric, metrics_a, common_keys)
    med_b, mean_b, std_b, min_b, max_b, n_b = agg(metric, metrics_b, common_keys)
    print(f'{metric:<22} {"A":>8}  {med_a:>10.5f} {mean_a:>10.5f} {std_a:>10.5f} {min_a:>10.5f} {max_a:>10.5f} {n_a:>5d}')
    print(f'{"":<22} {"B":>8}  {med_b:>10.5f} {mean_b:>10.5f} {std_b:>10.5f} {min_b:>10.5f} {max_b:>10.5f} {n_b:>5d}')

# Per-cell deltas
print(f'\n{"Per-cell deltas (B - A)":}')
print('-' * 60)
for metric in ['snr', 'noise_rms', 'nan_frac', 'W']:
    deltas = []
    for k in common_keys:
        va = metrics_a[k][metric]
        vb = metrics_b[k][metric]
        if np.isfinite(va) and np.isfinite(vb):
            deltas.append(vb - va)
    deltas = np.array(deltas)
    if len(deltas) > 0:
        print(f'  {metric:<18} median={np.median(deltas):+.6f}  '
              f'mean={np.mean(deltas):+.6f}  '
              f'std={np.std(deltas):.6f}  '
              f'|max|={np.max(np.abs(deltas)):.6f}  N={len(deltas)}')

# Cells where the two pipelines disagree most
print(f'\n--- Top 10 cells with largest |SNR difference| ---')
snr_diffs = []
for k in common_keys:
    sa = metrics_a[k]['snr']
    sb = metrics_b[k]['snr']
    if np.isfinite(sa) and np.isfinite(sb):
        snr_diffs.append((k, sa, sb, sb - sa))

snr_diffs.sort(key=lambda x: -abs(x[3]))
for (l, b), sa, sb, diff in snr_diffs[:10]:
    nf_a = metrics_a[(l, b)]['nan_frac']
    nf_b = metrics_b[(l, b)]['nan_frac']
    print(f'  ({l:3d}, {b:+3d})  SNR_A={sa:7.1f}  SNR_B={sb:7.1f}  '
          f'delta={diff:+7.1f}  NaN_A={nf_a:.1%}  NaN_B={nf_b:.1%}')

# DC bin / sky-freq check: does pipeline B miss the known artifacts?
print(f'\n--- DC bin / sky-freq artifact check ---')
print(f'  Pipeline A explicitly masks DC bin (ch {DC_BIN}) + sky-freq channels')
print(f'  Pipeline B relies on medfilt to catch them as outliers')

# Check a sample of records_b to see if DC bin was flagged by medfilt
records_b_check = copy.deepcopy(raw_records[:50])
dc_caught = 0
dc_missed = 0
for r in records_b_check:
    stokes = np.fft.fftshift(r['corr00']) + np.fft.fftshift(r['corr11'])
    original_dc = stokes[DC_BIN]
    flag_rfi_medfilt(stokes, RFI_WINDOW, RFI_SIGMA)
    if np.isnan(stokes[DC_BIN]):
        dc_caught += 1
    else:
        dc_missed += 1

print(f'  DC bin flagged by medfilt: {dc_caught}/{dc_caught+dc_missed} dumps')
print(f'  DC bin missed by medfilt:  {dc_missed}/{dc_caught+dc_missed} dumps')

# Timing summary
print(f'\n--- Timing ---')
print(f'  Pipeline A preprocessing: {t_preprocess_a:.2f} s')
print(f'  Pipeline B preprocessing: {t_preprocess_b:.2f} s')
print(f'  Pipeline A total:         {t_preprocess_a + t_pipeline_a:.2f} s')
print(f'  Pipeline B total:         {t_preprocess_b + t_pipeline_b:.2f} s')

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
