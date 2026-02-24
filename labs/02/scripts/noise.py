#!/usr/bin/env python3
"""Lab 2 galactic-plane varying precision observations @ 1420 MHz at varying
precisions to characterize noise.

  LO:   1420 MHz

Usage:
    python noise.py
"""

import sys
import time

from ugradio.sdr import SDR

from ugradiolab.experiment import ObsExperiment
from ugradiolab.queue import QueueRunner
from ugradiolab.utils import compute_pointing

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab02/noise'

GAL_L = 120.0  # degrees
GAL_B = 0.0    # degrees

LO_FREQ = 1420.0e6

NBLOCKS_STEPS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

MIN_ALT_DEG = 10.0  # elevation floor; warn below this

COMMON = dict(
    outdir=OUTDIR,
    nsamples=32768,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
)


# ---------------------------------------------------------------------------

def build_plan(alt_deg, az_deg):
    """Build one ObsExperiment per nblocks step."""
    pointing = dict(alt_deg=alt_deg, az_deg=az_deg)
    experiments = [
        ObsExperiment(prefix=f'NOISE-nblocks={n}', center_freq=LO_FREQ, nblocks=n, **pointing, **COMMON)
        for n in NBLOCKS_STEPS
    ]

    return experiments


def main():
    print(f'Lab 2 galactic observation — computing pointing for (l={GAL_L}°, b={GAL_B}°) ...')
    print()

    alt, az, ra, dec, jd = compute_pointing(GAL_L, GAL_B)

    print(f'  Galactic        :  l = {GAL_L:.1f}°,  b = {GAL_B:.1f}°')
    print(f'  Equatorial J2000:  RA = {ra:.4f}°,  Dec = {dec:.4f}°')
    print(f'  Local alt/az    :  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Julian date     :  {jd:.5f}')
    print()

    if alt < MIN_ALT_DEG:
        print(f'  WARNING: target is only {alt:.1f}° above the horizon '
              f'(minimum recommended: {MIN_ALT_DEG}°).')
        print('  Consider waiting until the target rises or choose a different LST.')
        print()
        cont = input('  Continue anyway? [y/N] ').strip().lower()
        if cont != 'y':
            print('Aborted.')
            sys.exit(0)
        print()

    print(f'  >>> Point the horn to:  Alt = {alt:.2f}°,  Az = {az:.2f}° <<<')
    print()
    input('  Press Enter once the horn is pointed and you are ready to begin: ')
    print()

    experiments = build_plan(alt, az)
    total = len(experiments)

    print(f'Starting {total} captures...')
    print(f'  LO:   {LO_FREQ / 1e6:.0f} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=LO_FREQ,
              sample_rate=COMMON['sample_rate'], gain=COMMON['gain'])

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
