#!/usr/bin/env python3
"""Lab 3 Phase 1 — Sun fringe calibration.

Tracks the Sun over an extended period without any delay-line compensation.
The raw fringe pattern as a function of hour angle is used to fit the
interferometer baseline (BASELINE_EW_M, BASELINE_NS_M).  The fitted values
go into sun_observe.py for Phase 2.

At X-band (~10 GHz) with a ~20 m baseline the fringe period is ~28 seconds,
so captures are made back-to-back — no sleep between them.  30–60 minutes of
data (~65–130 fringe cycles) is sufficient for a reliable baseline fit.

RF chain (NCH X-band interferometer):
  two X-band horn antennas → coaxial combiner → SNAP digital correlator

Usage:
    python sun_calibration.py

Output:
    data/lab03/sun_calibration/sun_cal_corr_<timestamp>.npz  (one file per capture)

Fit baseline from the output in a notebook by modelling fringe phase:
    φ(t) = 2π f₀ τ_g(t) = 2π f₀ (B_ew/c) cos(dec) sin(ha(t))
"""

import sys
import time

import ugradio.interf as interf

# NOTE: confirm snap_spec import path before running
# import snap_spec

from ugradiolab import SunExperiment, compute_sun_pointing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR       = 'data/lab03/sun_calibration'
MIN_ALT_DEG  = 15.0   # elevation floor; abort below this

DURATION_SEC = 10.0   # integration time per SNAP capture
N_CAPTURES   = 180    # ~30 min of back-to-back 10s captures

# ---------------------------------------------------------------------------


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
    print('Lab 3 Phase 1 — Sun fringe calibration')
    print('=' * 50)
    print()
    print('Checking Sun position ...')
    print()
    check_sun_visible(MIN_ALT_DEG)

    total_min = N_CAPTURES * DURATION_SEC / 60
    print(f'  Captures    : {N_CAPTURES} × {DURATION_SEC}s  (back-to-back, no sleep)')
    print(f'  Total time  : ~{total_min:.0f} min')
    print(f'  Output      : {OUTDIR}/')
    print()
    print('  No delay-line compensation — recording raw fringes for baseline fit.')
    print()
    input('  Connect hardware, then press Enter to begin: ')
    print()

    # --- Hardware setup ---
    interferometer = interf.Interferometer()
    # snap = snap_spec.Snap(...)   # TODO: confirm snap_spec API

    # --- Capture loop (back-to-back) ---
    paths = []
    t0    = time.time()

    for i in range(N_CAPTURES):
        exp = SunExperiment(
            duration_sec = DURATION_SEC,
            outdir       = OUTDIR,
            prefix       = f'sun-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied
        )

        print(f'[{i + 1:3d}/{N_CAPTURES}] ', end='', flush=True)
        path = exp.run(interferometer, snap=None)  # replace None with snap object
        paths.append(path)
        elapsed = time.time() - t0
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  t={elapsed:.0f}s  → {path}')

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {time.time() - t0:.0f}s')
    print()
    print('Next steps:')
    print('  1. Load the .npz files in a notebook.')
    print('  2. Extract fringe phase vs. time and fit:')
    print('       φ(t) = 2π f₀ (B_ew / c) cos(dec) sin(ha(t))')
    print('  3. Copy the fitted BASELINE_EW_M (and BASELINE_NS_M if non-zero)')
    print('     into labs/03/scripts/sun_observe.py for Phase 2.')


if __name__ == '__main__':
    main()
