#!/usr/bin/env python3
"""Lab 3 — Multi-target interferometer calibration.

Priority (highest first):
  1. Sun  — observed when alt ≥ SUN_MIN_ALT_DEG  (6.25°)
  2. Moon — observed when alt ≥ MOON_MIN_ALT_DEG (6.25°)
  3. M17  — fallback target when neither Sun nor Moon is up

Usage:
    python multi_calibration.py

Output:
    data/lab03/sun_calibration/sun-NNN_corr_<timestamp>.npz
    data/lab03/moon_calibration/moon-NNN_corr_<timestamp>.npz
    data/lab03/m17_calibration/m17-NNN_corr_<timestamp>.npz
"""

import time

import numpy as np

from ugradiolab import (
    MoonExperiment,
    RadecExperiment,
    SunExperiment,
    compute_moon_pointing,
    compute_radec_pointing,
    compute_sun_pointing,
)

from utils import lst_deg, optimal_duration, reinit_snap, setup_hardware

# ---------------------------------------------------------------------------
# Source catalog position (J2000)
# ---------------------------------------------------------------------------

M17_RA_DEG  = 275.1083   # 18h 20m 26s
M17_DEC_DEG = -16.1767   # -16° 10' 36"

NCH_LON_DEG = -122.2573  # NCH site longitude (degrees east)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUN_MIN_ALT_DEG  = 6.25
MOON_MIN_ALT_DEG = 6.25
M17_MIN_ALT_DEG  = 6.25   # just above hardware floor of 6°

SUN_OUTDIR  = 'data/lab03/sun_calibration'
MOON_OUTDIR = 'data/lab03/moon_calibration'
M17_OUTDIR  = 'data/lab03/m17_calibration'

BASELINE_EST_M   = 12.5   # baseline estimate for fringe-rate duration calculation (m)
TARGET_PHASE_DEG = 30.0   # desired fringe phase advance per capture (deg)

SLEW_TOL_DEG     = 0.3    # skip point() if antennas are already within this of target;
                           # at 0.3° max amplitude loss is ~0.04%, phase loss is 0%

# ---------------------------------------------------------------------------


def _ha_deg(ra_deg):
    """Current hour angle [degrees] for a source at ra_deg."""
    jd  = time.time() / 86400.0 + 2440587.5
    ha  = (lst_deg(jd) - ra_deg) % 360.0
    if ha > 180.0:
        ha -= 360.0
    return ha


def _sun_ha_deg():
    _, _, sun_ra, _, jd = compute_sun_pointing()
    ha = (lst_deg(jd) - sun_ra) % 360.0
    if ha > 180.0:
        ha -= 360.0
    return ha


def select_target():
    """Return (name, alt, az) for the highest-priority visible target, or None."""
    sun_alt,  sun_az,  *_ = compute_sun_pointing()
    moon_alt, moon_az, *_ = compute_moon_pointing()
    m17_alt,  m17_az,  *_ = compute_radec_pointing(M17_RA_DEG, M17_DEC_DEG)

    if sun_alt >= SUN_MIN_ALT_DEG:
        return 'sun', sun_alt, sun_az
    if moon_alt >= MOON_MIN_ALT_DEG:
        return 'moon', moon_alt, moon_az
    if m17_alt >= M17_MIN_ALT_DEG:
        return 'm17', m17_alt, m17_az
    return None


def make_experiment(target, interferometer, snap, capture_index):
    """Build the appropriate Experiment object for *target*."""
    if target == 'sun':
        return SunExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = _sun_duration(),
            outdir         = SUN_OUTDIR,
            prefix         = f'sun-{capture_index:03d}',
            slew_tol_deg   = SLEW_TOL_DEG,
        )
    if target == 'moon':
        return MoonExperiment(
            interferometer = interferometer,
            snap           = snap,
            duration_sec   = _moon_duration(),
            outdir         = MOON_OUTDIR,
            prefix         = f'moon-{capture_index:03d}',
            slew_tol_deg   = SLEW_TOL_DEG,
        )
    # M17
    jd_now   = time.time() / 86400.0 + 2440587.5
    ha_now   = (lst_deg(jd_now) - M17_RA_DEG) % 360.0
    if ha_now > 180.0:
        ha_now -= 360.0
    duration = optimal_duration(ha_now, M17_DEC_DEG, BASELINE_EST_M, TARGET_PHASE_DEG)
    return RadecExperiment(
        interferometer = interferometer,
        snap           = snap,
        ra_deg         = M17_RA_DEG,
        dec_deg        = M17_DEC_DEG,
        duration_sec   = duration,
        outdir         = M17_OUTDIR,
        prefix         = f'm17-{capture_index:03d}',
        slew_tol_deg   = SLEW_TOL_DEG,
    )


def _sun_duration():
    """Fringe-rate-based capture duration for the Sun."""
    _, _, sun_ra, sun_dec, jd = compute_sun_pointing()
    ha = (lst_deg(jd) - sun_ra) % 360.0
    if ha > 180.0:
        ha -= 360.0
    return optimal_duration(ha, sun_dec, BASELINE_EST_M, TARGET_PHASE_DEG)


def _moon_duration():
    """Fixed 10 s captures for the Moon (moves too fast for fringe-rate formula)."""
    return 10.0


def main():
    print('Lab 3 — Multi-target calibration  (Sun > Moon > M17)')
    print('=' * 80)
    print()
    print('Initialising hardware ...')
    interferometer, snap = setup_hardware()
    print('Hardware ready.')
    print()
    print('Press Ctrl-C to stop.\n')

    counters  = {'sun': 0, 'moon': 0, 'm17': 0}
    n_saved   = {'sun': 0, 'moon': 0, 'm17': 0}
    prev_target = None

    while True:
        result = select_target()

        target, alt, az = result
        idx = counters[target]

        if target != prev_target:
            print(f'\n[Target → {target.upper()}]  Alt={alt:.1f}°  Az={az:.1f}°')
            prev_target = target

        exp = make_experiment(target, interferometer, snap, idx)
        dur = exp.duration_sec

        if target == 'sun':
            ha = _sun_ha_deg()
        elif target == 'm17':
            ha = _ha_deg(M17_RA_DEG)
        else:
            ha = float('nan')   # Moon RA changes too fast; skip

        ha_str = f'{ha:+.2f}°' if np.isfinite(ha) else 'n/a'
        print(f'  [{target} {idx + 1:3d}]  dur={dur:.1f}s  HA={ha_str}  ', end='', flush=True)
        try:
            path = exp.run()
            n_saved[target] += 1
            print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  → {path}')
        except RuntimeError as exc:
            print(f'SKIP  ({exc})')
            reinit_snap(snap)

        counters[target] += 1


if __name__ == '__main__':
    main()
