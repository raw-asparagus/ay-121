#!/usr/bin/env python3
"""Lab 3 Phase 1 — Sun fringe calibration.

Tracks the Sun over a 15-minute window without delay-line compensation.
The raw fringe pattern as a function of hour angle is used to fit the
interferometer baseline (BASELINE_EW_M, BASELINE_NS_M).  The fitted values
go into sun_observe.py for Phase 2.

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
"""

import sys
import time

from ugradiolab import SunExperiment, compute_sun_pointing

from utils import lst_deg, optimal_duration, reinit_snap, setup_hardware

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR      = 'data/lab03/sun_calibration'
MIN_ALT_DEG = 6.25   # elevation floor; hardware limit is 6°

OBS_WINDOW_SEC   = 8 * 60   # total observation window (s)
BASELINE_EST_M   = 15.0     # baseline estimate used only for duration optimisation (m)
TARGET_PHASE_DEG = 60.0     # desired fringe phase advance per capture (deg)

# Centre of the observable RF band (Hz).
# LO chain: LO1=8750 MHz, LO2=1540 MHz, f_s=500 MHz
#   SNAP: 2048-pt real FFT → 1024 channels, Δf=244 kHz/ch
#   Band: 9790–10040 MHz  →  centre ≈ 9915 MHz  (ch 512 of 1024)
OBS_FREQ_HZ = 9.915e9

# ---------------------------------------------------------------------------


def check_sun_visible(min_alt_deg):
    """Print current Sun position and return (alt, az, ra, dec, jd).  Exits if below horizon."""
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
    return alt, az, ra, dec, jd


def main():
    print('Lab 3 Phase 1 — Sun fringe calibration')
    print('=' * 80)
    print()
    print('Checking Sun position ...')
    print()
    _, _, sun_ra, sun_dec, jd = check_sun_visible(MIN_ALT_DEG)

    ha_deg = (lst_deg(jd) - sun_ra) % 360.0
    if ha_deg > 180.0:
        ha_deg -= 360.0

    print('  Observation plan')
    print(f'    Source         : Sun  RA={sun_ra:.4f}°  Dec={sun_dec:.4f}°')
    print(f'    Window         : {OBS_WINDOW_SEC / 60:.0f} min  (variable N captures)')
    print()
    print('  No delay-line compensation — recording raw fringes for baseline fit.')
    print()

    # --- Hardware setup ---
    interferometer, snap = setup_hardware()

    # --- Time-bounded capture loop with per-capture duration ---
    paths = []
    t0    = time.time()
    t_end = t0 + OBS_WINDOW_SEC
    i     = 0

    while time.time() < t_end:
        # Recompute Sun's current RA/Dec and HA (Sun moves ~1°/day)
        alt_now, _, sun_ra_now, sun_dec_now, jd_now = compute_sun_pointing()
        if alt_now < MIN_ALT_DEG:
            print(f'  Sun below {MIN_ALT_DEG}° ({alt_now:.1f}°) — stopping.')
            break
        ha_now = (lst_deg(jd_now) - sun_ra_now) % 360.0
        if ha_now > 180.0:
            ha_now -= 360.0
        duration_sec = optimal_duration(ha_now, sun_dec_now, BASELINE_EST_M, TARGET_PHASE_DEG,
                                        OBS_FREQ_HZ)

        exp = SunExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = duration_sec,
            outdir         = OUTDIR,
            prefix         = f'sun-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied in Phase 1
        )

        print(f'[{i + 1:3d}] ', end='', flush=True)
        try:
            path = exp.run()
        except RuntimeError as exc:
            print(f'SKIP  ({exc})')
            reinit_snap(snap)
            i += 1
            continue
        paths.append(path)
        elapsed = time.time() - t0
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  dur={duration_sec:.1f}s  t={elapsed:.0f}s  → {path}')
        i += 1

    total_elapsed = time.time() - t0
    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {total_elapsed:.0f}s')


if __name__ == '__main__':
    main()
