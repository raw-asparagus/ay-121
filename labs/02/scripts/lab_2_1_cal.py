#!/usr/bin/env python3
"""Lab 2 galactic-plane observation @ 1420 MHz with signal generator tuned
at the 21-cm line.

Usage:
    python lab_2_1_cal.py
"""

import sys
import time

import ugradio.nch as nch
from ugradio.sdr import SDR

from ugradiolab import SignalGenerator
from ugradiolab.experiment import CalExperiment, ObsExperiment
from ugradiolab.queue import QueueRunner
from ugradiolab.utils import compute_pointing

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_1_cal'

GAL_L = 120.0   # degrees
GAL_B = 0.0     # degrees

LO_FREQ = 1420.0e6

MIN_ALT_DEG = 10.0     # elevation floor; warn below this

SIGGEN_FREQ_MHZ = 1421.0
SIGGEN_AMP_STEPS_DBM = [-80.0, -60.0, -40.0]

COMMON = dict(
    outdir=OUTDIR,
    nsamples=16384,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
)


# ---------------------------------------------------------------------------

def build_plan(alt_deg, az_deg):
    """Build one CalExperiment per amplitude step."""
    pointing = dict(alt_deg=alt_deg, az_deg=az_deg)
    return [
        CalExperiment(
            prefix=f'CAL-{amp:.0f}dBm',
            siggen_freq_mhz=SIGGEN_FREQ_MHZ,
            siggen_amp_dbm=amp,
            **pointing,
            **COMMON,
        )
        for amp in SIGGEN_AMP_STEPS_DBM
    ]


def main():
    print('Lab 2 galactic observation — computing pointing for (l=120°, b=0°) ...')
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

    print(f'Starting {total} captures (amp steps: {SIGGEN_AMP_STEPS_DBM} dBm)...')
    print(f'  CAL:  {SIGGEN_FREQ_MHZ} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=LO_FREQ, sample_rate=2.56e6, gain=0.0)
    synth = SignalGenerator()

    archive = f'{OUTDIR}/lab_2_1_cal_{time.strftime("%Y%m%d_%H%M%S")}.tar.gz'
    try:
        runner = QueueRunner(experiments=experiments, sdr=sdr, synth=synth, confirm=False)
        t0 = time.time()
        paths = runner.run(archive=archive)
        elapsed = time.time() - t0
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
