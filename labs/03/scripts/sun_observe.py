#!/usr/bin/env python3
"""Lab 3 Phase 2 — Sun science observation with delay compensation.

Requires a calibrated baseline from Phase 1 (sun_calibration.py).
Fill in BASELINE_EW_M, BASELINE_NS_M, and DELAY_MAX_NS below before running.

The geometric delay τ_g = (B_ew/c) cos(dec) sin(ha) is computed at each
capture and applied via the coaxial delay line to compensate for the
path-length difference between the two antennas.

RF chain (NCH X-band interferometer):
  two X-band horn antennas → coaxial delay line → coaxial combiner
    → SNAP digital correlator

Usage:
    python sun_observe.py

Output:
    data/lab03/sun_observe/sun_corr_<timestamp>.npz  (one file per capture)
"""

import sys

import ugradio.interf as interf
import ugradio.interf_delay as interf_delay
from snap_spec.snap import UGRadioSnap

from ugradiolab import SunExperiment, compute_sun_pointing

# ---------------------------------------------------------------------------
# Configuration — fill in after Phase 1 baseline fit
# ---------------------------------------------------------------------------

OUTDIR       = 'data/lab03/sun_observe'
MIN_ALT_DEG  = 15.0   # elevation floor; abort below this

# Fitted from Phase 1 fringe data.  Set both before running.
BASELINE_EW_M = None   # e.g. 19.86  — east-west baseline in metres
BASELINE_NS_M = 0.0    # e.g.  0.12  — north-south baseline in metres (often ~0)

# Hardware delay limit (ugradio.interf_delay.MAX_DELAY, calibrated 2019-03-21).
DELAY_MAX_NS  = 64.8

# Capture parameters
DUMP_SEC   = 0.625   # SNAP accumulation period: ACC_LEN×SPEC_PER_ACC×1024/f_s
N_CAPTURES = 1920    # total dumps (~20 min at ~0.625s each)

# ---------------------------------------------------------------------------


def check_config():
    """Abort with a clear message if baseline has not been filled in."""
    missing = []
    if BASELINE_EW_M is None:
        missing.append('BASELINE_EW_M')
    if missing:
        print('ERROR: the following must be set before running Phase 2:')
        for name in missing:
            print(f'  {name} = None   ← edit this in sun_observe.py')
        print()
        print('Run sun_calibration.py (Phase 1) first and fit the baseline')
        print('from the fringe data.')
        sys.exit(1)


def check_sun_visible(min_alt_deg):
    """Print current Sun position and return (alt, az).  Exits if below horizon."""
    alt, az, ra, dec, jd = compute_sun_pointing()
    print(f'  Sun position:  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Equatorial  :  RA = {ra:.4f}°,  Dec = {dec:.4f}°')
    print(f'  Julian date :  {jd:.5f}')
    print()
    if alt < min_alt_deg:
        print(f'  ERROR: Sun is only {alt:.1f}° above the horizon '
              f'(minimum: {min_alt_deg}°).')
        print('  Wait until the Sun rises higher and re-run.')
        sys.exit(1)
    return alt, az


def main():
    check_config()

    print('Lab 3 Phase 2 — Sun science observation')
    print('=' * 50)
    print()
    print('Checking Sun position ...')
    print()
    check_sun_visible(MIN_ALT_DEG)

    print(f'  Baseline    : B_ew = {BASELINE_EW_M:.4f} m,  B_ns = {BASELINE_NS_M:.4f} m')
    print(f'  Delay clip  : ±{DELAY_MAX_NS} ns')
    print(f'  Dumps       : {N_CAPTURES}  (~{DUMP_SEC:.3f}s each)')
    print(f'  Total time  : ~{N_CAPTURES * DUMP_SEC / 60:.0f} min')
    print(f'  Output      : {OUTDIR}/')
    print()
    input('  Connect hardware, then press Enter to begin: ')
    print()

    # --- Hardware setup ---
    interferometer = interf.Interferometer()
    delay_line     = interf_delay.DelayClient()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    snap.initialize(mode='corr', sample_rate=500)
    snap.input.use_adc()

    # --- Capture loop (one file per SNAP dump, ~0.625 s each) ---
    # A single SunExperiment is reused; _prev_cnt is carried between calls so
    # read_data() blocks until the next fresh accumulation (no duplicate dumps).
    exp = SunExperiment(
        interferometer = interferometer,
        snap           = snap,
        delay_line     = delay_line,
        outdir         = OUTDIR,
        prefix         = 'sun-000000',
        baseline_ew_m  = BASELINE_EW_M,
        baseline_ns_m  = BASELINE_NS_M,
        delay_max_ns   = DELAY_MAX_NS,
    )

    paths = []

    for i in range(N_CAPTURES):
        exp.prefix = f'sun-{i:06d}'
        print(f'[{i + 1:6d}/{N_CAPTURES}] ', end='', flush=True)
        path = exp.run()
        paths.append(path)
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  τ_g≈{_last_tau(path):.2f}ns  → {path}')

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/')


def _last_tau(path):
    """Read delay_ns from the most recently saved .npz for display."""
    try:
        import numpy as np
        return float(np.load(path)['delay_ns'])
    except Exception:
        return float('nan')


if __name__ == '__main__':
    main()
