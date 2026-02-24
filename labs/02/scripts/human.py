#!/usr/bin/env python3
"""Lab 2 high precision human noise observation.

  LO:   1420 – 1421 MHz in 1 MHz steps  →  HI line offset per step:
    1420 MHz  →  +0.406 MHz
    1421 MHz  →  −0.594 MHz

Usage:
    python human.py
"""

import time

from ugradio.sdr import SDR
import ugradio.timing as timing

from ugradiolab.experiment import ObsExperiment
from ugradiolab.queue import QueueRunner
from ugradiolab.utils import get_unix_time

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab02/human'

ALT = 0.0  # deg
AZI = 0.0  # deg

FREQ_1 = 1420.0e6
FREQ_2 = 1421.0e6

ITERATIONS = 8

COMMON = dict(
    outdir=OUTDIR,
    nsamples=8192,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
    alt_deg=ALT,
    az_deg=AZI
)

SETTLE_SEC = 120


# ---------------------------------------------------------------------------

def build_plan():
    """Build several copies of (FREQ_1, FREQ_2) frequency-switched experiment list."""
    experiments = []

    for i in range(ITERATIONS):
        experiments.append(ObsExperiment(prefix=f'HUMAN-{FREQ_1 / 1e6:.0f}-{i}', center_freq=FREQ_1, **COMMON))
        experiments.append(ObsExperiment(prefix=f'HUMAN-{FREQ_2 / 1e6:.0f}-{i}', center_freq=FREQ_2, **COMMON))

    return experiments


def main():
    print(f'Lab 2 human noise calibration, pointed horizontally ...')
    print()

    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)

    print(f'  Local alt/az    :  Alt = {ALT:.2f}°,  Az = {AZI:.2f}°')
    print(f'  Julian date     :  {jd:.5f}')
    print()

    print(f'  >>> Point the horn to:  Alt = {ALT:.2f}°,  Az = {AZI:.2f}° <<<')
    print()
    input('  Press Enter once the horn is pointed: ')
    print()

    print(f'  Waiting {SETTLE_SEC}s...', end='', flush=True)
    for remaining in range(SETTLE_SEC, 0, -1):
        print(f'\r  {remaining:3d}s remaining...   ', end='', flush=True)
        time.sleep(1)
    print()

    experiments = build_plan()
    total = len(experiments)

    print(f'Starting {total} captures...')
    print(f'  LO:   {FREQ_1 / 1e6:.0f} & {FREQ_2 / 1e6:.0f} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=FREQ_1, sample_rate=2.56e6, gain=0.0)

    try:
        runner = QueueRunner(experiments=experiments, sdr=sdr, confirm=False)
        t0 = time.time()
        paths = runner.run()
        elapsed = time.time() - t0
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
