#!/usr/bin/env python3
"""Lab 3 — Orion A (M42) fringe calibration.

Tracks Orion A without delay-line compensation to fit the interferometer
baseline (BASELINE_EW_M) from the raw fringe pattern.  This is identical in
procedure to sun_calibration.py but uses a fixed catalog position.

Source: Orion A / M42 (Orion Nebula HII region core)
  RA  =  83.8221°  (J2000 = 5h 35m 17.3s)
  Dec =  -5.3911°  (J2000 = -5° 23' 28")
  S(10 GHz) ≈ 340 Jy

Visibility from Berkeley (lat 37.9°N):
  Culmination altitude ≈ 46.7°  (culminates due south, Az ≈ 180°)
  Above 15° for roughly 6 hours each night (rises ~E, sets ~W)

Usage:
    python orion_calibration.py

Output:
    data/lab03/orion_calibration/orion-cal-NNN_corr_<timestamp>.npz
"""

import sys
import time

from ugradiolab import RadecExperiment, compute_radec_pointing

from utils import lst_deg, optimal_duration, reinit_snap, setup_hardware

# ---------------------------------------------------------------------------
# Source catalog position (J2000)
# ---------------------------------------------------------------------------

ORION_RA_DEG  =  83.8221   # 5h 35m 17.3s
ORION_DEC_DEG =  -5.3911   # -5° 23' 28"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR       = 'data/lab03/orion_calibration'
MIN_ALT_DEG  = 6.25   # elevation floor; hardware limit is 6°

OBS_WINDOW_SEC   = 15 * 60  # total observation window (s)
BASELINE_EST_M   = 15.0     # baseline estimate used only for duration optimisation (m)
TARGET_PHASE_DEG = 60.0     # desired fringe phase advance per capture (deg)

# ---------------------------------------------------------------------------


def check_orion_visible(min_alt_deg):
    """Print current Orion position and return (alt, az, jd).  Exits if below limit."""
    alt, az, jd = compute_radec_pointing(ORION_RA_DEG, ORION_DEC_DEG)
    print(f'  Orion A position:  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Julian date     :  {jd:.5f}')
    print()
    if alt < min_alt_deg:
        print(f'  ERROR: Orion A is only {alt:.1f}° above the horizon '
              f'(minimum: {min_alt_deg}°).')
        print('  Orion A transits ~47° above the horizon from Berkeley.')
        print('  It is visible (>15°) for roughly 6 h each night.')
        sys.exit(1)
    return alt, az, jd


def main():
    print('Lab 3 — Orion A fringe calibration (no delay compensation)')
    print('=' * 70)
    print()
    print('Checking Orion A position ...')
    print()
    _, _, jd = check_orion_visible(MIN_ALT_DEG)

    ha_deg = (lst_deg(jd) - ORION_RA_DEG) % 360.0
    if ha_deg > 180.0:
        ha_deg -= 360.0

    print('  Observation plan')
    print(f'    Source         : Orion A  RA={ORION_RA_DEG:.4f}°  Dec={ORION_DEC_DEG:.4f}°')
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
        jd_now = time.time() / 86400.0 + 2440587.5
        ha_now = (lst_deg(jd_now) - ORION_RA_DEG) % 360.0
        if ha_now > 180.0:
            ha_now -= 360.0
        duration_sec = optimal_duration(ha_now, ORION_DEC_DEG, BASELINE_EST_M, TARGET_PHASE_DEG)

        alt_now, _, _ = compute_radec_pointing(ORION_RA_DEG, ORION_DEC_DEG)
        if alt_now < MIN_ALT_DEG:
            print(f'  Orion A below {MIN_ALT_DEG}° ({alt_now:.1f}°) — stopping.')
            break

        exp = RadecExperiment(
            interferometer = interferometer,
            snap           = snap,
            ra_deg         = ORION_RA_DEG,
            dec_deg        = ORION_DEC_DEG,
            duration_sec   = duration_sec,
            outdir         = OUTDIR,
            prefix         = f'orion-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied
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
        print(f'dur={duration_sec:.1f}s  HA={ha_now:+.2f}°  Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  t={elapsed:.0f}s  → {path}')
        i += 1

    total_elapsed = time.time() - t0
    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {total_elapsed:.0f}s')


if __name__ == '__main__':
    main()
