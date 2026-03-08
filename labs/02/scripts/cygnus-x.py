#!/usr/bin/env python3
"""Lab 2 high precision Cygnus X frequency-swept observation.

  LO:   1420 – 1421 MHz in 1 MHz steps  →  HI line offset per step:
    1420 MHz  →  +0.406 MHz
    1421 MHz  →  −0.594 MHz

Usage:
    python standard.py
"""

import sys
import time

from ugradio.sdr import SDR

from ugradiolab.run import ObsExperiment
from ugradiolab.run import QueueRunner
from ugradiolab.pointing import compute_pointing

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab02/cygnus-x'

TARGET_NAME = 'Cygnus X'
TARGET_SIMBAD_QUERY = 'NAME Cyg X'
# One-time SIMBAD resolution performed with astroquery.simbad on 2026-03-05.
TARGET_RA_DEG = 307.5199890137
TARGET_DEC_DEG = 40.8600006104
# Converted from (RA, Dec) -> Galactic using astropy.coordinates.SkyCoord.
TARGET_GAL_L_DEG = 79.5043879159
TARGET_GAL_B_DEG = 1.0005912555

FREQ_1 = 1420.0e6
FREQ_2 = 1421.0e6

MIN_ALT_DEG = 15.0  # elevation floor; warn below this

ITERATIONS = 16

COMMON = dict(
    outdir=OUTDIR,
    nsamples=32768,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
)


# ---------------------------------------------------------------------------

def build_plan(alt_deg, az_deg):
    """Build several copies of (FREQ_1, FREQ_2) frequency-switched experiment list."""
    pointing = dict(alt_deg=alt_deg, az_deg=az_deg)
    experiments = []

    for i in range(ITERATIONS):
        experiments.append(ObsExperiment(prefix=f'CYGX-{FREQ_1 / 1e6:.0f}-{i}', center_freq=FREQ_1,
                                         **pointing, **COMMON))
        experiments.append(ObsExperiment(prefix=f'CYGX-{FREQ_2 / 1e6:.0f}-{i}', center_freq=FREQ_2,
                                         **pointing, **COMMON))

    return experiments


def main():
    print('Lab 2 Cygnus X observation — computing pointing from SIMBAD-resolved target ...')
    print()

    alt, az, ra, dec, jd = compute_pointing(TARGET_GAL_L_DEG, TARGET_GAL_B_DEG)

    print(f'  Target          :  {TARGET_NAME} (SIMBAD query: {TARGET_SIMBAD_QUERY})')
    print(f'  Equatorial J2000:  RA = {TARGET_RA_DEG:.4f}°,  Dec = {TARGET_DEC_DEG:.4f}°')
    print(f'  Galactic        :  l = {TARGET_GAL_L_DEG:.4f}°,  b = {TARGET_GAL_B_DEG:.4f}°')
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
    print(f'  LO:   {FREQ_1 / 1e6:.0f} & {FREQ_2 / 1e6:.0f} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=FREQ_1,
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
