#!/usr/bin/env python3
"""Lab 2 hot calibration — human-body blackbody load.

Horn stays horizontal (alt=0, az=0) throughout.  A person covers the
dish aperture as a ~310 K blackbody hot load for the Y-factor method.

Sequence (12 steps):
  1.  HOT-BASE-PRE    baseline (dish open, gen OFF)
  --- Confirmation: person gets into position ---
  --- 3-minute countdown ---
  2.  HOT-LOAD-01     person covering aperture
  ...
  11. HOT-LOAD-10     person covering aperture
  --- Confirmation: person steps away ---
  12. HOT-BASE-POST   baseline (dish open, gen OFF)

Usage:
    python lab_2_hot_cal.py
"""

import time

from ugradio.sdr import SDR
from ugradiolab.experiment import ObsExperiment

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_hot_cal'
TIMER_SEC = 180  # 3 minutes
N_HOT = 10

SDR_DEFAULTS = dict(
    direct=False,
    center_freq=1420e6,
    sample_rate=2.56e6,
    gain=0.0,
)

HORIZONTAL = dict(alt_deg=0.0, az_deg=0.0)


def build_plan():
    """Build the pre-baseline, hot-load, and post-baseline experiment lists."""
    common = dict(
        outdir=OUTDIR,
        nsamples=32768,
        nblocks=10,
        **SDR_DEFAULTS,
    )

    pre = ObsExperiment(prefix='HOT-BASE-PRE', **HORIZONTAL, **common)
    hot_loads = [
        ObsExperiment(prefix=f'HOT-LOAD-{i + 1:02d}', **HORIZONTAL, **common)
        for i in range(N_HOT)
    ]
    post = ObsExperiment(prefix='HOT-BASE-POST', **HORIZONTAL, **common)

    return pre, hot_loads, post


def _countdown(seconds):
    """Print a live countdown, then clear the line."""
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(
            f'\r  Get into position — starting in {mins}:{secs:02d}...',
            end='',
            flush=True,
        )
        time.sleep(1)
    print('\r  Starting hot load sequence now.                        ')


def main():
    pre, hot_loads, post = build_plan()
    total = 1 + len(hot_loads) + 1

    print(f'Lab 2 hot calibration: {total} steps')
    print(f'Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)

    try:
        # Step 1: pre-baseline (dish open, no load)
        print(f'[1/{total}] {pre.prefix}')
        path = pre.run(sdr)
        print(f'  -> {path}')
        print()

        # Operator confirmation before person covers aperture
        input('  Press Enter when the person is ready to cover the aperture: ')
        _countdown(TIMER_SEC)

        # Steps 2–11: hot load (person covering aperture)
        for i, exp in enumerate(hot_loads):
            step = i + 2
            print(f'[{step}/{total}] {exp.prefix}')
            path = exp.run(sdr)
            print(f'  -> {path}')

        print()

        # Operator confirmation before person steps away
        input('  Press Enter when the person has stepped away from the aperture: ')
        print()

        # Step 12: post-baseline (dish open, no load)
        print(f'[{total}/{total}] {post.prefix}')
        path = post.run(sdr)
        print(f'  -> {path}')

    finally:
        sdr.close()

    print()
    print(f'Done. {total} files saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
