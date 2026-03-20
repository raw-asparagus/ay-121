#!/usr/bin/env python3
"""Lab 3 — Multi-target interferometer calibration.

Priority (highest first):
  1. Sun  — observed when alt ≥ SUN_MIN_ALT_DEG  (6.25°)
  2. Moon — observed when alt ≥ MOON_MIN_ALT_DEG (6.25°)
  3. M17  — fallback target when neither Sun nor Moon is up
  4. M1   — lowest priority fallback (Crab Nebula)

Usage:
    python multi_calibration.py

Output:
    data/lab03/sun_calibration/sun-NNN_corr_<timestamp>.npz
    data/lab03/moon_calibration/moon-NNN_corr_<timestamp>.npz
    data/lab03/m17_calibration/m17-NNN_corr_<timestamp>.npz
    data/lab03/m1_calibration/m1-NNN_corr_<timestamp>.npz
"""

import threading
import time

import numpy as np

from ugradiolab.astronomy import (
    compute_moon_pointing,
    compute_radec_pointing,
    compute_sun_pointing,
)
from ugradiolab.capture import (
    MoonExperiment,
    PipelinedCapture,
    RadecExperiment,
    SunExperiment,
)

from utils import lst_deg, optimal_duration, setup_hardware

# ---------------------------------------------------------------------------
# Source catalog position (J2000)
# ---------------------------------------------------------------------------

M17_RA_DEG  = 275.1083   # 18h 20m 26s
M17_DEC_DEG = -16.1767   # -16° 10' 36"

M1_RA_DEG   =  83.6331   # 05h 34m 31.9s  (Crab Nebula)
M1_DEC_DEG  = +22.0145   # +22° 00' 52"

NCH_LON_DEG = -122.2573  # NCH site longitude (degrees east)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUN_MIN_ALT_DEG  = 6.5
MOON_MIN_ALT_DEG = 6.5
M17_MIN_ALT_DEG  = 6.5
M1_MIN_ALT_DEG   = 6.5

SUN_OUTDIR  = 'data/lab03/sun_calibration'
MOON_OUTDIR = 'data/lab03/moon_calibration'
M17_OUTDIR  = 'data/lab03/m17_calibration'
M1_OUTDIR   = 'data/lab03/m1_calibration'

BASELINE_EST_M   = 12.5   # baseline estimate for fringe-rate duration calculation (m)
TARGET_PHASE_DEG = 15.0   # desired fringe phase advance per capture (deg)

# ---------------------------------------------------------------------------


def _ha_deg(ra_deg):
    """Current hour angle [degrees] for a source at ra_deg."""
    jd  = time.time() / 86400.0 + 2440587.5
    return _wrap_signed_deg(lst_deg(jd) - ra_deg)


def _wrap_signed_deg(value):
    """Wrap an angle to (-180, 180] degrees."""
    wrapped = value % 360.0
    if wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def _sun_ha_deg():
    _, _, sun_ra, _, jd = compute_sun_pointing()
    return _wrap_signed_deg(lst_deg(jd) - sun_ra)


def select_target():
    """Return (name, alt, az, duration) for the highest-priority visible target, or None.

    Computes each source's pointing only as needed (short-circuits on first visible
    target) and folds in the capture duration so callers need not re-query ephemeris.
    """
    sun_alt, sun_az, sun_ra, sun_dec, jd = compute_sun_pointing()
    if sun_alt >= SUN_MIN_ALT_DEG:
        ha = _wrap_signed_deg(lst_deg(jd) - sun_ra)
        return 'sun', sun_alt, sun_az, optimal_duration(ha, sun_dec, BASELINE_EST_M, TARGET_PHASE_DEG)

    moon_alt, moon_az, *_ = compute_moon_pointing()
    if moon_alt >= MOON_MIN_ALT_DEG:
        return 'moon', moon_alt, moon_az, 10.0

    m17_alt, m17_az, m17_jd = compute_radec_pointing(M17_RA_DEG, M17_DEC_DEG)
    if m17_alt >= M17_MIN_ALT_DEG:
        ha = _wrap_signed_deg(lst_deg(m17_jd) - M17_RA_DEG)
        return 'm17', m17_alt, m17_az, optimal_duration(ha, M17_DEC_DEG, BASELINE_EST_M, TARGET_PHASE_DEG)

    m1_alt, m1_az, m1_jd = compute_radec_pointing(M1_RA_DEG, M1_DEC_DEG)
    if m1_alt >= M1_MIN_ALT_DEG:
        ha = _wrap_signed_deg(lst_deg(m1_jd) - M1_RA_DEG)
        return 'm1', m1_alt, m1_az, optimal_duration(ha, M1_DEC_DEG, BASELINE_EST_M, TARGET_PHASE_DEG)

    return None


def make_experiment(target, interferometer, snap, capture_index, duration):
    """Build the appropriate Experiment object for *target* with pre-computed *duration*."""
    if target == 'sun':
        return SunExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = duration,
            outdir         = SUN_OUTDIR,
            prefix         = f'sun-{capture_index:03d}',
        )
    if target == 'moon':
        return MoonExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = duration,
            outdir         = MOON_OUTDIR,
            prefix         = f'moon-{capture_index:03d}',
        )
    if target == 'm17':
        return RadecExperiment(
            interferometer = interferometer,
            snap           = snap,
            ra_deg         = M17_RA_DEG,
            dec_deg        = M17_DEC_DEG,
            duration_sec   = duration,
            outdir         = M17_OUTDIR,
            prefix         = f'm17-{capture_index:03d}',
        )
    # M1
    return RadecExperiment(
        interferometer = interferometer,
        snap           = snap,
        ra_deg         = M1_RA_DEG,
        dec_deg        = M1_DEC_DEG,
        duration_sec   = duration,
        outdir         = M1_OUTDIR,
        prefix         = f'm1-{capture_index:03d}',
    )


def main():
    print('Lab 3 — Multi-target calibration  (Sun > Moon > M17 > M1)')
    print('=' * 80)
    print()
    print('Initialising hardware ...')
    interferometer, snap = setup_hardware()
    print('Hardware ready.')
    print()
    print('Press Ctrl-C to stop.\n')

    counters = {'sun': 0, 'moon': 0, 'm17': 0, 'm1': 0}
    n_saved  = {'sun': 0, 'moon': 0, 'm17': 0, 'm1': 0}
    save_lock = threading.Lock()

    def make_fn():
        while True:
            result = select_target()
            if result is not None:
                break
            time.sleep(30)
        target, _alt, _az, duration = result
        idx = counters[target]
        counters[target] += 1
        return make_experiment(target, interferometer, snap, idx, duration)

    def on_save(path, exp):
        target = exp.prefix.split('-')[0]   # 'sun-001' → 'sun'
        if target == 'sun':
            ha = _sun_ha_deg()
        elif target == 'm17':
            ha = _ha_deg(M17_RA_DEG)
        elif target == 'm1':
            ha = _ha_deg(M1_RA_DEG)
        else:
            ha = float('nan')
        ha_str = f'{ha:+.2f}°' if np.isfinite(ha) else 'n/a'
        with save_lock:
            n_saved[target] += 1
            idx = n_saved[target]
            print(
                f'  [{target} {idx:3d}]  dur={exp.duration_sec:.1f}s  HA={ha_str}'
                f'  Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  → {path}'
            )

    pipeline = PipelinedCapture(interferometer, snap, verify_every_n=5)
    try:
        pipeline.run(make_fn, on_save=on_save)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.flush()


if __name__ == '__main__':
    main()
