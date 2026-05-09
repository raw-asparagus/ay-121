#!/usr/bin/env python3
"""Lab 3 - Streaming interferometer capture with auto-correlation calibration.

Runs a brief auto-correlation calibration scan (spec mode) at the start of the
observation, then switches to cross-correlation mode (corr) and streams every
SNAP accumulator dump to disk.

Priority (highest first):
  1. Sun  - observed when alt >= 6.5 deg
  2. Moon - observed when alt >= 6.5 deg
  3. M17  - fallback
  4. M1   - lowest priority fallback (Crab Nebula)

Usage:
    python stream_calibration.py

Output:
    labs/03/data/streaming/calibration/autocorr_<timestamp>.npz
    labs/03/data/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
"""

import os
import time

import numpy as np

from ugradiolab.astronomy import (
    compute_moon_pointing,
    compute_radec_pointing,
    compute_sun_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_snap_reader

# ---------------------------------------------------------------------------
# Source catalog (J2000)
# ---------------------------------------------------------------------------

M17_RA_DEG  = 275.1083   # 18h 20m 26s
M17_DEC_DEG = -16.1767   # -16d 10' 36"

M1_RA_DEG   =  83.6331   # 05h 34m 31.9s  (Crab Nebula)
M1_DEC_DEG  = +22.0145   # +22d 00' 52"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Interferometer hardware: alt 6-174, az 88-300. Conservative guard: +2 deg margin.
MIN_ALT_DEG  =  8.0
MAX_ALT_DEG  = 172.0
AZ_MIN_DEG   = 90.0
AZ_MAX_DEG   = 298.0

OUTDIR    = 'labs/03/data/streaming'
CAL_DUMPS = 20   # number of auto-correlation dumps to collect


# ---------------------------------------------------------------------------

def setup_hardware(snap_retries=5):
    """Initialise interferometer and SNAP correlator.  Returns (interferometer, snap)."""
    import ugradio.interf as interf
    from snap_spec.snap import UGRadioSnap

    interferometer = interf.Interferometer()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    for attempt in range(1, snap_retries + 1):
        try:
            snap.initialize(mode='corr', sample_rate=500, force=True)
            snap.input.use_adc()
            if attempt > 1:
                print(f'  SNAP initialized on attempt {attempt}.')
            return interferometer, snap
        except AssertionError as exc:
            print(f'  SNAP init attempt {attempt}/{snap_retries} failed ({exc}), retrying...')
    raise RuntimeError(f'SNAP initialization failed after {snap_retries} attempts.')


def collect_autocorrelation(snap, n_dumps=CAL_DUMPS):
    """Switch to spec mode, collect n_dumps of auto-correlation, switch back.

    Returns the path to the saved calibration .npz file.
    """
    # Switch to spec mode (auto-correlations).
    snap.mode = 'spec'
    snap.corr_0.set_input(snap.stream_1, snap.stream_1)  # auto0: (0, 0)
    snap.corr_1.set_input(snap.stream_2, snap.stream_2)  # auto1: (1, 1)

    # Discard first dump (stale accumulation from corr mode).
    snap.read_data(prev_cnt=None)

    auto0_list = []
    auto1_list = []
    times = []
    prev_cnt = None
    for i in range(n_dumps):
        d = snap.read_data(prev_cnt=prev_cnt)
        auto0_list.append(d['auto0'])
        auto1_list.append(d['auto1'])
        times.append(d['time'])
        prev_cnt = d['acc_cnt']

    # Switch back to corr mode (cross-correlation).
    snap.mode = 'corr'
    snap.corr_0.set_input(snap.stream_1, snap.stream_2)  # cross: (0, 1)

    # Save.
    cal_dir = os.path.join(OUTDIR, 'calibration')
    os.makedirs(cal_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(cal_dir, f'autocorr_{ts}.npz')
    np.savez(
        path,
        auto0=np.array(auto0_list),
        auto1=np.array(auto1_list),
        times=np.array(times),
        n_dumps=n_dumps,
    )
    return path


def _in_bounds(alt, az):
    """Check if pointing is within conservative interferometer limits."""
    return (MIN_ALT_DEG <= alt <= MAX_ALT_DEG and AZ_MIN_DEG <= az <= AZ_MAX_DEG)


def target_selector():
    """Return (name, alt, az, ra, dec) for the highest-priority visible target."""
    sun_alt, sun_az, sun_ra, sun_dec, _ = compute_sun_pointing()
    if _in_bounds(sun_alt, sun_az):
        return 'sun', sun_alt, sun_az, sun_ra, sun_dec

    # moon_alt, moon_az, moon_ra, moon_dec, _ = compute_moon_pointing()
    # if _in_bounds(moon_alt, moon_az):
    #     return 'moon', moon_alt, moon_az, moon_ra, moon_dec
    #
    # m17_alt, m17_az, _ = compute_radec_pointing(M17_RA_DEG, M17_DEC_DEG)
    # if _in_bounds(m17_alt, m17_az):
    #     return 'm17', m17_alt, m17_az, M17_RA_DEG, M17_DEC_DEG
    #
    # m1_alt, m1_az, _ = compute_radec_pointing(M1_RA_DEG, M1_DEC_DEG)
    # if _in_bounds(m1_alt, m1_az):
    #     return 'm1', m1_alt, m1_az, M1_RA_DEG, M1_DEC_DEG

    return None


def on_save(path, dump):
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}  '
        f'acc={dump["acc_cnt"]}  -> {path}'
    )


def main():
    print('Lab 3 - Streaming capture  (Sun > Moon > M17 > M1)')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    interferometer, snap = setup_hardware()
    print('Hardware ready.\n')

    # --- Wait for target and point ---
    print('Waiting for target to rise ...')
    while True:
        result = target_selector()
        if result is not None:
            break
        time.sleep(10)
    name, alt, az, ra, dec = result
    print(f'  Slewing to {name} (alt={alt:.1f}, az={az:.1f}) ...')
    interferometer.point(alt, az, wait=True)
    print('  Pointing complete.\n')

    # --- Auto-correlation calibration scan (on-target) ---
    print(f'Collecting auto-correlation calibration ({CAL_DUMPS} dumps) ...')
    cal_path = collect_autocorrelation(snap)
    print(f'  Calibration saved -> {cal_path}\n')

    # --- Cross-correlation streaming ---
    read_fn = make_snap_reader(snap)
    capture = StreamingCapture(
        telescope=interferometer,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=OUTDIR,
        n_writers=2,
        repoint_interval_sec=30.0,
        on_save=on_save,
    )
    capture.run()


if __name__ == '__main__':
    main()
