#!/usr/bin/env python3
"""Lab 3 — Streaming interferometer capture.

Saves every single SNAP accumulator dump to its own .npz file using a
producer-consumer architecture (see ``ugradiolab.capture.streaming``).

Priority (highest first):
  1. Sun  — observed when alt >= 6.5°
  2. Moon — observed when alt >= 6.5°
  3. M17  — fallback
  4. M1   — lowest priority fallback (Crab Nebula)

Usage:
    python stream_calibration.py

Output:
    data/lab03/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
"""

from ugradiolab.astronomy import (
    compute_moon_pointing,
    compute_radec_pointing,
    compute_sun_pointing,
)
from ugradiolab.capture import StreamingCapture

# ---------------------------------------------------------------------------
# Source catalog (J2000)
# ---------------------------------------------------------------------------

M17_RA_DEG  = 275.1083   # 18h 20m 26s
M17_DEC_DEG = -16.1767   # -16° 10' 36"

M1_RA_DEG   =  83.6331   # 05h 34m 31.9s  (Crab Nebula)
M1_DEC_DEG  = +22.0145   # +22° 00' 52"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUN_MIN_ALT_DEG  = 6.5
MOON_MIN_ALT_DEG = 6.5
M17_MIN_ALT_DEG  = 6.5
M1_MIN_ALT_DEG   = 6.5

OUTDIR = 'data/lab03/streaming'


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


def target_selector():
    """Return (name, alt, az, ra, dec) for the highest-priority visible target."""
    sun_alt, sun_az, sun_ra, sun_dec, _ = compute_sun_pointing()
    if sun_alt >= SUN_MIN_ALT_DEG:
        return 'sun', sun_alt, sun_az, sun_ra, sun_dec

    moon_alt, moon_az, moon_ra, moon_dec, _ = compute_moon_pointing()
    if moon_alt >= MOON_MIN_ALT_DEG:
        return 'moon', moon_alt, moon_az, moon_ra, moon_dec

    m17_alt, m17_az, _ = compute_radec_pointing(M17_RA_DEG, M17_DEC_DEG)
    if m17_alt >= M17_MIN_ALT_DEG:
        return 'm17', m17_alt, m17_az, M17_RA_DEG, M17_DEC_DEG

    m1_alt, m1_az, _ = compute_radec_pointing(M1_RA_DEG, M1_DEC_DEG)
    if m1_alt >= M1_MIN_ALT_DEG:
        return 'm1', m1_alt, m1_az, M1_RA_DEG, M1_DEC_DEG

    return None


def on_save(path, dump):
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}  '
        f'acc={dump["acc_cnt"]}  → {path}'
    )


def main():
    print('Lab 3 — Streaming capture  (Sun > Moon > M17 > M1)')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    interferometer, snap = setup_hardware()
    print('Hardware ready.\n')

    capture = StreamingCapture(
        interferometer=interferometer,
        snap=snap,
        target_selector=target_selector,
        outdir=OUTDIR,
        n_writers=2,
        repoint_interval_sec=30.0,
        on_save=on_save,
    )
    capture.run()


if __name__ == '__main__':
    main()
