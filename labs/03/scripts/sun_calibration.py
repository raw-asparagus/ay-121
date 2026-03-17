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

DURATION_SEC = 10.0  # integration time per SNAP capture
OBS_WINDOW_SEC = 8 * 60  # 8-minute observation window
N_CAPTURES = round(OBS_WINDOW_SEC / DURATION_SEC)  # = 90

# Centre of the observable RF band (Hz).
# LO chain: LO1=8750 MHz, LO2=1540 MHz, f_s=500 MHz
#   Band: (LO1+LO2-f_s) to (LO1+LO2-f_s/2) = 9790–10040 MHz  →  centre ≈ 9915 MHz
OBS_FREQ_HZ = 9.915e9

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


def print_observation_plan(dec_deg: float) -> None:
    """Print a diagnostic summary of the planned observation."""
    print('  Observation plan')
    print(f'    Duration       : {N_CAPTURES} × {DURATION_SEC:.0f}s = {OBS_WINDOW_SEC / 60:.0f} min')
    print()
    print('  No delay-line compensation — recording raw fringes for baseline fit.')
    print()


def main():
    print('Lab 3 Phase 1 — Sun fringe calibration')
    print('=' * 80)
    print()
    print('Checking Sun position ...')
    print()
    alt, az, dec = check_sun_visible(MIN_ALT_DEG)

    print_observation_plan(dec)

    input('  Connect hardware, then press Enter to begin: ')
    print()

    # --- Hardware setup ---
    interferometer = interf.Interferometer()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    snap.initialize(mode='corr', sample_rate=500, force=True)
    snap.input.use_adc()

    # --- Capture loop (back-to-back, no sleep) ---
    paths  = []
    t0     = time.time()

    for i in range(N_CAPTURES):
        exp = SunExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = DURATION_SEC,
            outdir         = OUTDIR,
            prefix         = f'sun-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied in Phase 1
        )

        print(f'[{i + 1:3d}/{N_CAPTURES}] ', end='', flush=True)
        try:
            path = exp.run()
        except RuntimeError as exc:
            print(f'SKIP  ({exc})')
            continue
        paths.append(path)
        elapsed = time.time() - t0
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  t={elapsed:.0f}s  → {path}')

    total_elapsed = time.time() - t0
    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {total_elapsed:.0f}s')


if __name__ == '__main__':
    main()
