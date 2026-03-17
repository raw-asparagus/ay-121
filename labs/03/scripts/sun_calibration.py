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

import math
import sys
import time

import ugradio.interf as interf
import ugradio.nch as nch
from snap_spec.snap import UGRadioSnap

from ugradiolab import SunExperiment, compute_sun_pointing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR      = 'data/lab03/sun_calibration'
MIN_ALT_DEG = 15.0   # elevation floor; abort below this

OBS_WINDOW_SEC   = 8 * 60   # total observation window (s)
BASELINE_EST_M   = 15.0     # baseline estimate used only for duration optimisation (m)
TARGET_PHASE_DEG = 60.0     # desired fringe phase advance per capture (deg)

# Centre of the observable RF band (Hz).
# LO chain: LO1=8750 MHz, LO2=1540 MHz, f_s=500 MHz
#   SNAP: 2048-pt real FFT → 1024 channels, Δf=244 kHz/ch
#   Band: 9790–10040 MHz  →  centre ≈ 9915 MHz  (ch 512 of 1024)
OBS_FREQ_HZ = 9.915e9

# ---------------------------------------------------------------------------


def _lst_deg(jd):
    T = (jd - 2451545.0) / 36525.0
    g = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
         + T ** 2 * 0.000387933 - T ** 3 / 38710000.0)
    return (g + nch.lon) % 360.0


def _optimal_duration(ha_deg, dec_deg):
    omega_e = 2 * math.pi / 86164.0
    rate = (OBS_FREQ_HZ * BASELINE_EST_M / 299792458.0
            * math.cos(math.radians(dec_deg))
            * omega_e
            * abs(math.cos(math.radians(ha_deg))))
    if rate < 1e-12:
        return 60.0
    tau = (TARGET_PHASE_DEG / 360.0) / rate
    return max(5.0, min(60.0, tau))


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

    ha_deg = (_lst_deg(jd) - sun_ra) % 360.0
    if ha_deg > 180.0:
        ha_deg -= 360.0

    duration_sec = _optimal_duration(ha_deg, sun_dec)
    n_captures   = round(OBS_WINDOW_SEC / duration_sec)
    t_fringe_est = duration_sec * 360.0 / TARGET_PHASE_DEG

    print('  Observation plan')
    print(f'    HA at start    : {ha_deg:.1f}°')
    print(f'    Baseline est.  : {BASELINE_EST_M:.0f} m  (duration optimisation only)')
    print(f'    T_fringe est.  : {t_fringe_est:.0f} s')
    print(f'    Duration/cap   : {duration_sec:.1f} s  ({TARGET_PHASE_DEG:.0f}° phase/cap)')
    print(f'    N captures     : {n_captures} × {duration_sec:.1f} s = {OBS_WINDOW_SEC / 60:.0f} min')
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

    # --- Capture loop (back-to-back, no sleep) ---
    paths  = []
    t0     = time.time()

    for i in range(n_captures):
        exp = SunExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = duration_sec,
            outdir         = OUTDIR,
            prefix         = f'sun-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied in Phase 1
        )

        print(f'[{i + 1:3d}/{n_captures}] ', end='', flush=True)
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
