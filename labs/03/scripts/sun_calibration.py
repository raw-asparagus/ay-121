#!/usr/bin/env python3
"""Lab 3 Phase 1 — Sun fringe calibration.

Tracks the Sun over a 15-minute window without delay-line compensation.
The raw fringe pattern as a function of hour angle is used to fit the
interferometer baseline (BASELINE_EW_M, BASELINE_NS_M).  The fitted values
go into sun_observe.py for Phase 2.

Baseline estimate: 10–20 m.  At X-band (~10 GHz) this gives a fringe period
of roughly 21–41 seconds (depending on the Sun's declination), so 10 s
captures resolve 2–4 points per fringe cycle.  Over 15 minutes (~900 s) you
accumulate 22–43 fringe cycles — sufficient for a reliable baseline fit.

Pointing offset note
--------------------
The two antennas may not be perfectly co-pointed; an offset of up to ~1° is
expected.  This has two effects:

  1. Reduced fringe amplitude — a 1° separation across a ~20° primary beam
     costs only ~5 % in visibility, so fringes remain clearly detectable.

  2. Constant phase offset — the geometric delay τ_g is computed from the
     commanded pointing direction, not the true per-antenna pointing.  Any
     residual pointing error between the two antennas introduces a slowly
     varying (and partly constant) additive phase term φ_off that cannot be
     removed by the geometric delay alone.

Consequence for fitting: include a free constant phase offset φ₀ in the
fringe model (see "Next steps" below).

RF chain (NCH X-band interferometer):
  two X-band horn antennas → coaxial combiner → SNAP digital correlator

Usage:
    python sun_calibration.py

Output:
    data/lab03/sun_calibration/sun_cal_corr_<timestamp>.npz  (one file per capture)

Fit baseline from the output in a notebook by modelling fringe phase:
    φ(t) = 2π f₀ (B_ew / c) cos(dec) sin(ha(t)) + φ₀
where φ₀ absorbs pointing-offset and instrumental phase.
"""

import sys
import time

import ugradio.interf as interf
from snap_spec.snap import UGRadioSnap

from ugradiolab import SunExperiment, compute_sun_pointing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR      = 'data/lab03/sun_calibration'
MIN_ALT_DEG = 15.0   # elevation floor; abort below this

EST_DUMP_SEC = 0.625
N_CAPTURES = 180
EST_OBS_WINDOW_SEC = N_CAPTURES * EST_DUMP_SEC

# ---------------------------------------------------------------------------


def check_sun_visible(min_alt_deg):
    """Print current Sun position and return (alt, az, dec).  Exits if below horizon."""
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
    return alt, az, dec


def print_observation_plan() -> None:
    """Print a diagnostic summary of the planned observation."""

    print('  Observation plan')
    print(f'    Duration       : {N_CAPTURES} × {EST_DUMP_SEC:.3f}s = {EST_OBS_WINDOW_SEC / 60:.0f} min')
    print()
    print('  No delay-line compensation — recording raw fringes for baseline fit.')
    print()


def main():
    print('Lab 3 Phase 1 — Sun fringe calibration')
    print('=' * 50)
    print()
    print('Checking Sun position ...')
    print()
    check_sun_visible(MIN_ALT_DEG)

    print_observation_plan()

    input('  Connect hardware, then press Enter to begin: ')
    print()

    # --- Hardware setup ---
    interferometer = interf.Interferometer()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    snap.initialize(mode='corr', sample_rate=500)
    snap.input.use_adc()

    # --- Capture loop (one file per SNAP dump, ~0.625 s each) ---
    # A single SunExperiment is reused; _prev_cnt is carried between calls so
    # read_data() blocks until the next fresh accumulation (no duplicate dumps).
    exp = SunExperiment(
        interferometer = interferometer,
        snap           = snap,
        outdir         = OUTDIR,
        prefix         = 'sun-cal-000',
        # baseline_ew_m=None (default) → no delay applied in Phase 1
    )

    paths = []
    t0    = time.time()

    for i in range(N_CAPTURES):
        exp.prefix = f'sun-cal-{i:03d}'
        print(f'[{i + 1:3d}/{N_CAPTURES}] ', end='', flush=True)
        path = exp.run()
        paths.append(path)
        elapsed = time.time() - t0
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  t={elapsed:.0f}s  → {path}')

    total_elapsed = time.time() - t0
    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {total_elapsed:.0f}s')


if __name__ == '__main__':
    main()
