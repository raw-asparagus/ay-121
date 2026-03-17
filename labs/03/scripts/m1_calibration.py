#!/usr/bin/env python3
"""Lab 3 — M1 / Crab Nebula (Tau A) fringe calibration.

Tracks M1 without delay-line compensation to fit the interferometer baseline
(BASELINE_EW_M) from the raw fringe pattern.

Source: M1 / Crab Nebula / Tau A (supernova remnant)
  RA  =  83.6331°  (J2000 = 5h 34m 31.9s)
  Dec = +22.0145°  (J2000 = +22° 00' 52")
  S(10 GHz) ≈ 500 Jy  (power-law spectrum S ∝ ν^{-0.30}; relatively flat at X-band)

Visibility from Berkeley (lat 37.9°N):
  Transit altitude ≈ 74.1°  (transits due south, Az ≈ 180°)
  Above 15° for roughly 10 hours each day

Fringe period (B_ew ~ 15 m, Dec = +22°):
  T_fringe = λ / (B_ew · cos(dec) · ω_Earth) ≈ 30 s
  10 s captures give ~3 points per fringe cycle.

Usage:
    python m1_calibration.py

Output:
    data/lab03/m1_calibration/m1-cal-NNN_corr_<timestamp>.npz

Fit baseline from output using the standard fringe model:
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

M1_RA_DEG  =  83.6331   # 5h 34m 31.9s
M1_DEC_DEG = +22.0145   # +22° 00' 52"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR       = 'data/lab03/m1_calibration'
MIN_ALT_DEG  = 15.0   # elevation floor; abort below this

DURATION_SEC   = 10.0       # integration time per SNAP capture
OBS_WINDOW_SEC = 15 * 60    # 15-minute observation window
N_CAPTURES     = round(OBS_WINDOW_SEC / DURATION_SEC)  # = 90

# ---------------------------------------------------------------------------


def check_m1_visible(min_alt_deg):
    """Print current M1 position and return (alt, az).  Exits if below limit."""
    alt, az, jd = compute_radec_pointing(M1_RA_DEG, M1_DEC_DEG)
    print(f'  M1 position :  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Julian date :  {jd:.5f}')
    print()
    if alt < min_alt_deg:
        print(f'  ERROR: M1 is only {alt:.1f}° above the horizon '
              f'(minimum: {min_alt_deg}°).')
        print('  M1 transits ~74° above the horizon from Berkeley.')
        print('  It is visible (>15°) for roughly 10 h each day.')
        sys.exit(1)
    return alt, az


def main():
    print('Lab 3 — M1 / Crab Nebula fringe calibration (no delay compensation)')
    print('=' * 70)
    print()
    print('Checking M1 position ...')
    print()
    check_m1_visible(MIN_ALT_DEG)

    print('  Observation plan')
    print(f'    Duration : {N_CAPTURES} × {DURATION_SEC:.0f}s = {OBS_WINDOW_SEC / 60:.0f} min')
    print(f'    Source   : M1 / Crab Nebula  RA={M1_RA_DEG:.4f}°  Dec={M1_DEC_DEG:.4f}°')
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
            ra_deg         = M1_RA_DEG,
            dec_deg        = M1_DEC_DEG,
            duration_sec   = DURATION_SEC,
            outdir         = OUTDIR,
            prefix         = f'm1-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied
        )

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
