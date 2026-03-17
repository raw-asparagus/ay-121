#!/usr/bin/env python3
"""Lab 3 — Orion A (M42) fringe calibration.

Tracks Orion A without delay-line compensation to fit the interferometer
baseline (BASELINE_EW_M) from the raw fringe pattern.  This is identical in
procedure to sun_calibration.py but uses a fixed catalog position.

Source: Orion A / M42 (Orion Nebula HII region core)
  RA  =  83.8221°  (J2000 = 5h 35m 17.3s)
  Dec =  -5.3911°  (J2000 = -5° 23' 28")
  S(10 GHz) ≈ 400 Jy  (free-free thermal emission; nearly flat spectrum)

Visibility from Berkeley (lat 37.9°N):
  Transit altitude ≈ 46.7°  (transits due south, Az ≈ 180°)
  Above 15° for roughly 6 hours each night (rises ~E, sets ~W)

Fringe period (B_ew ~ 15 m, Dec = -5.4°):
  T_fringe = λ / (B_ew · cos(dec) · ω_Earth) ≈ 28 s
  10 s captures give ~3 points per fringe cycle.

Usage:
    python orion_calibration.py

Output:
    data/lab03/orion_calibration/orion-cal-NNN_corr_<timestamp>.npz

Fit baseline from output using the same notebook procedure as Sun:
    φ(t) = 2π f₀ (B_ew / c) cos(dec) sin(ha(t)) + φ₀
"""

import sys
import time

import ugradio.interf as interf
from snap_spec.snap import UGRadioSnap

from ugradiolab import RadecExperiment, compute_radec_pointing

# ---------------------------------------------------------------------------
# Source catalog position (J2000)
# ---------------------------------------------------------------------------

ORION_RA_DEG  =  83.8221   # 5h 35m 17.3s
ORION_DEC_DEG =  -5.3911   # -5° 23' 28"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR       = 'data/lab03/orion_calibration'
MIN_ALT_DEG  = 15.0   # elevation floor; abort below this

DURATION_SEC   = 10.0       # integration time per SNAP capture
OBS_WINDOW_SEC = 15 * 60    # 15-minute observation window
N_CAPTURES     = round(OBS_WINDOW_SEC / DURATION_SEC)  # = 90

# ---------------------------------------------------------------------------


def check_orion_visible(min_alt_deg):
    """Print current Orion position and return (alt, az).  Exits if below limit."""
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
    return alt, az


def main():
    print('Lab 3 — Orion A fringe calibration (no delay compensation)')
    print('=' * 70)
    print()
    print('Checking Orion A position ...')
    print()
    check_orion_visible(MIN_ALT_DEG)

    print('  Observation plan')
    print(f'    Duration : {N_CAPTURES} × {DURATION_SEC:.0f}s = {OBS_WINDOW_SEC / 60:.0f} min')
    print(f'    Source   : Orion A  RA={ORION_RA_DEG:.4f}°  Dec={ORION_DEC_DEG:.4f}°')
    print()
    print('  No delay-line compensation — recording raw fringes for baseline fit.')
    print()

    input('  Connect hardware, then press Enter to begin: ')
    print()

    # --- Hardware setup ---
    interferometer = interf.Interferometer()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    snap.initialize(mode='corr', sample_rate=500, force=True)
    snap.input.use_adc()

    # --- Capture loop ---
    paths = []
    t0    = time.time()

    for i in range(N_CAPTURES):
        exp = RadecExperiment(
            interferometer = interferometer,
            snap           = snap,
            ra_deg         = ORION_RA_DEG,
            dec_deg        = ORION_DEC_DEG,
            duration_sec   = DURATION_SEC,
            outdir         = OUTDIR,
            prefix         = f'orion-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied
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
