"""Illustrate how a single recal visit collapses to (mean, SEM) scalars.

Top panel: per-dump spectra (noise-off only, single pol, single LO) for one
visit, with INT_MASK shaded.

Bottom panel: per-dump band-averaged scalars vs dump index within the visit,
with visit mean line and +/-SEM shaded band.

Writes labs/04/artifacts/visit_band_average_illustration.png.
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

LAB04 = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LAB04.parent.parent
sys.path.insert(0, str(LAB04))
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    load_recal_dumps,
    preprocess_dumps,
    flag_outlier_dumps,
    build_overlap_grid,
)
from ugradiolab.plotting import (
    TEXTWIDTH_IN, subpanels,
    LW_FINE, LW_LIGHT, ALPHA_FAINT, ALPHA_LIGHT, ALPHA_STANDARD,
)

# --- Constants (match main_scan_calibration.ipynb) ---
SAMPLE_RATE_HZ = 3.2e6
NFFT = 1024
F1_MHZ = 1419.86
F2_MHZ = 1421.14
EDGE_TRIM_MHZ = 0.256

RFI_WINDOW = 15
RFI_SIGMA = 10.0
RFI_CHEB_DEGREE = 3
RFI_SAMPLE_FRAC = 0.7
RFI_EXTREMA_ORDER = 2

SHAPE_DEV_THRESH = 0.05
SHAPE_FRAC_THRESH = 0.05
SHAPE_MIN_GROUP_SIZE = 3

VISIT_GAP_SEC = 300.0

DATA_DIRS = [LAB04 / 'data' / 'main', LAB04 / 'data' / 'nps']
OUT_PATH = LAB04 / 'artifacts' / 'visit_band_average_illustration.png'


def main():
    # --- Sections 1-3 ---
    records = load_recal_dumps(DATA_DIRS)
    preprocess_dumps(
        records,
        rfi_window=RFI_WINDOW, rfi_sigma=RFI_SIGMA,
        rfi_degree=RFI_CHEB_DEGREE,
        rfi_sample_frac=RFI_SAMPLE_FRAC,
        rfi_extrema_order=RFI_EXTREMA_ORDER,
    )
    flag_outlier_dumps(
        records,
        dev_thresh=SHAPE_DEV_THRESH,
        frac_thresh=SHAPE_FRAC_THRESH,
        min_group_size=SHAPE_MIN_GROUP_SIZE,
    )
    grid = build_overlap_grid(F1_MHZ, F2_MHZ, SAMPLE_RATE_HZ, NFFT,
                              edge_trim_mhz=EDGE_TRIM_MHZ)
    overlap_mask = grid['overlap_mask']
    INT_MASK = overlap_mask.copy()
    INT_MASK[NFFT // 2] = False

    # --- Cluster into visits, pick one ---
    groups = defaultdict(list)
    for r in records:
        groups[(r['session'], r['target_id'])].append(r)

    # Pick the visit with the most noise-off LO1 pol-0 dumps for a clean plot.
    best_run, best_meta = None, None
    for (session, target_id), grp in groups.items():
        grp = sorted(grp, key=lambda r: r['time'])
        runs = [[grp[0]]]
        for r in grp[1:]:
            if r['time'] - runs[-1][-1]['time'] > VISIT_GAP_SEC:
                runs.append([r])
            else:
                runs[-1].append(r)
        for visit_idx, run in enumerate(runs):
            off_lo1 = [d for d in run
                       if (not d['noise_on']) and d['lo_mhz'] == F1_MHZ]
            if best_run is None or len(off_lo1) > len(best_run):
                best_run = off_lo1
                best_meta = dict(session=session, target_id=target_id,
                                 visit_idx=visit_idx, n_total=len(run))
    if best_run is None or len(best_run) < 2:
        raise SystemExit('No visit with >=2 noise-off LO1 dumps found.')

    print(f'Chose visit: {best_meta}, n_off_LO1={len(best_run)}')

    # --- Build per-dump band-averaged scalars (pol 0 = corr00) ---
    freq_mhz = F1_MHZ + (np.arange(NFFT) - NFFT // 2) * (SAMPLE_RATE_HZ / NFFT) / 1e6
    spectra = np.array([d['corr00'] for d in best_run])  # (N_dumps, NFFT)
    times = np.array([d['time'] for d in best_run])
    t0 = times.min()
    dt = times - t0  # seconds since visit start

    # Per-dump band-average over INT_MASK (NaN-aware).
    scalars = np.nanmean(spectra[:, INT_MASK], axis=1)
    n = len(scalars)
    mu = float(np.nanmean(scalars))
    sigma = float(np.nanstd(scalars, ddof=1))
    sem = sigma / np.sqrt(n)
    frac_sem = sem / mu if mu else float('nan')

    print(f'mean={mu:.4g}, std={sigma:.4g}, SEM={sem:.4g} '
          f'(frac={frac_sem*100:.3f}%, N={n})')

    # --- Plot ---
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.75),
                     constrained_layout=True)
    axes = subpanels(fig, 2, sharex=False)

    # Top: spectra with INT_MASK shaded
    ax = axes[0]
    f_lo = freq_mhz[INT_MASK].min()
    f_hi = freq_mhz[INT_MASK].max()
    ax.axvspan(f_lo, f_hi, color='C2', alpha=ALPHA_FAINT, zorder=0,
               label=rf'INT\_MASK ({INT_MASK.sum()} ch)')
    for i, s in enumerate(spectra):
        ax.plot(freq_mhz, s, color='0.4', lw=LW_FINE,
                alpha=ALPHA_LIGHT, zorder=2)
    median_spec = np.nanmedian(spectra, axis=0)
    ax.plot(freq_mhz, median_spec, color='C0', lw=LW_LIGHT,
            alpha=ALPHA_STANDARD, zorder=3, label='dump-median spectrum')
    ax.axhline(mu, color='C3', ls='--', lw=LW_LIGHT,
               alpha=ALPHA_STANDARD, zorder=4,
               label=rf'visit mean $\mu$={mu:.3g}')
    ax.set_xlabel('sky frequency (MHz)')
    ax.set_ylabel(r'$P_{\rm off}$ (arb.)')
    ax.set_title(
        rf'Visit: session={best_meta["session"].replace("_", r"\_")}, '
        rf'target={best_meta["target_id"].replace("_", r"\_")}, '
        rf'visit\_idx={best_meta["visit_idx"]}, '
        rf'pol 0, LO1, N={n} noise-off dumps'
    )
    ax.legend(loc='upper right', fontsize='small', frameon=True)

    # Bottom: per-dump band-averaged scalars with mean +/- SEM band
    ax = axes[1]
    ax.axhspan(mu - sem, mu + sem, color='C3', alpha=ALPHA_FAINT,
               zorder=1, label=rf'$\mu \pm$ SEM (SEM={sem:.3g})')
    ax.axhline(mu, color='C3', ls='--', lw=LW_LIGHT,
               alpha=ALPHA_STANDARD, zorder=3,
               label=rf'$\mu$={mu:.4g}')
    ax.axhspan(mu - sigma, mu + sigma, color='0.5', alpha=ALPHA_FAINT/2,
               zorder=0, label=rf'$\mu \pm \sigma$ ($\sigma$={sigma:.3g})')
    ax.scatter(dt, scalars, color='C0', s=30, edgecolor='k',
               linewidth=LW_FINE, zorder=4, label='per-dump band-avg')
    ax.set_xlabel('time within visit (s)')
    ax.set_ylabel(r'$\langle P_{\rm off} \rangle_{\rm INT\_MASK}$ (arb.)')
    ax.set_title(
        rf'Per-dump scalars $\rightarrow$ visit $(\mu,$ SEM$)$: '
        rf'frac. SEM = {frac_sem*100:.3f}\%'
    )
    ax.legend(loc='best', fontsize='small', frameon=True, ncol=2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
