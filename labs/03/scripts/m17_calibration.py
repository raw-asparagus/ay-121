#!/usr/bin/env python3
"""Lab 3 — M17 / Omega Nebula (Sgr B) fringe calibration.

Tracks M17 without delay-line compensation to fit the interferometer baseline
(BASELINE_EW_M) from the raw fringe pattern.

Source: M17 / Omega Nebula / Swan Nebula (HII region in Sagittarius)
  RA  = 275.1083°  (J2000 = 18h 20m 26s)
  Dec = -16.1767°  (J2000 = -16° 10' 36")
  S(10 GHz) ≈ 500 Jy

Visibility from Berkeley (lat 37.9°N):
  Culmination altitude ≈ 35.9°  (culminates due south, Az ≈ 180°)
  Above 15° for roughly 7.4 hours each day

Usage:
    python m17_calibration.py

Output:
    data/lab03/m17_calibration/m17-cal-NNN_corr_<timestamp>.npz
"""

import sys
import time

from ugradiolab import RadecExperiment, compute_radec_pointing

from utils import lst_deg, optimal_duration, reinit_snap, setup_hardware

# ---------------------------------------------------------------------------
# Source catalog position (J2000)
# ---------------------------------------------------------------------------

M17_RA_DEG  = 275.1083   # 18h 20m 26s
M17_DEC_DEG = -16.1767   # -16° 10' 36"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR       = 'data/lab03/m17_calibration'
MIN_ALT_DEG  = 6.25   # elevation floor; hardware limit is 6°

OBS_WINDOW_SEC   = 6 * 60 * 60   # total observation window (s)
BASELINE_EST_M   = 12.5          # baseline estimate used only for duration optimisation (m)
TARGET_PHASE_DEG = 90.0          # desired fringe phase advance per capture (deg)

# ---------------------------------------------------------------------------


def check_m17_visible(min_alt_deg):
    """Print current M17 position and return (alt, az, jd).  Exits if below limit."""
    alt, az, jd = compute_radec_pointing(M17_RA_DEG, M17_DEC_DEG)
    print(f'  M17 position :  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Julian date  :  {jd:.5f}')
    print()
    if alt < min_alt_deg:
        print(f'  ERROR: M17 is only {alt:.1f}° above the horizon '
              f'(minimum: {min_alt_deg}°).')
        print('  M17 transits ~36° above the horizon from Berkeley.')
        print('  It is visible (>15°) for roughly 7.4 h each day.')
        sys.exit(1)
    return alt, az, jd


def main():
    print('Lab 3 — M17 / Omega Nebula fringe calibration (no delay compensation)')
    print('=' * 80)
    print()
    print('Checking M17 position ...')
    print()
    _, _, jd = check_m17_visible(MIN_ALT_DEG)

    ha_deg = (lst_deg(jd) - M17_RA_DEG) % 360.0
    if ha_deg > 180.0:
        ha_deg -= 360.0

    print('  Observation plan')
    print(f'    Source         : M17 / Omega Nebula  RA={M17_RA_DEG:.4f}°  Dec={M17_DEC_DEG:.4f}°')
    print(f'    Window         : {OBS_WINDOW_SEC / 60:.0f} min  (variable N captures)')
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
        ha_now = (lst_deg(jd_now) - M17_RA_DEG) % 360.0
        if ha_now > 180.0:
            ha_now -= 360.0
        duration_sec = optimal_duration(ha_now, M17_DEC_DEG, BASELINE_EST_M, TARGET_PHASE_DEG)

        alt_now, _, _ = compute_radec_pointing(M17_RA_DEG, M17_DEC_DEG)
        if alt_now < MIN_ALT_DEG:
            print(f'  M17 below {MIN_ALT_DEG}° ({alt_now:.1f}°) — stopping.')
            break

        exp = RadecExperiment(
            interferometer = interferometer,
            snap           = snap,
            ra_deg         = M17_RA_DEG,
            dec_deg        = M17_DEC_DEG,
            duration_sec   = duration_sec,
            outdir         = OUTDIR,
            prefix         = f'm17-cal-{i:03d}',
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
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  dur={duration_sec:.1f}s  t={elapsed:.0f}s  → {path}')
        i += 1

    total_elapsed = time.time() - t0
    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {total_elapsed:.0f}s')


if __name__ == '__main__':
    main()
