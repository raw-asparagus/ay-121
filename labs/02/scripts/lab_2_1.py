#!/usr/bin/env python3
"""Lab 2 galactic-plane two frequency-switched measurements with calibration.

Computes the current alt/az for galactic (l=120°, b=0°) at NCH using
NTP time, prints pointing instructions, then runs a calibration +
frequency-swept observation sequence after operator confirmation.

Each cycle consists of one CalExperiment followed by two ObsExperiments:

  CAL:  siggen at 1420.405751768 MHz, −80 dBm
  LO:   1420 MHz  →  HI line at +0.406 MHz
        1421 MHz  →  HI line at −0.594 MHz

Output files are saved to OUTDIR and archived to a timestamped .tar.gz.

Usage:
    python lab_2_1.py
"""

import sys
import time

from ugradio.sdr import SDR

from ugradiolab import SignalGenerator
from ugradiolab.experiment import CalExperiment, ObsExperiment
from ugradiolab.queue import QueueRunner
from ugradiolab.utils import compute_pointing

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_1'

GAL_L = 120.0   # degrees
GAL_B = 0.0     # degrees

LO_FREQ_1 = 1420.0e6
LO_FREQ_2 = 1421.0e6

MIN_ALT_DEG = 10.0     # elevation floor; warn below this

SIGGEN_FREQ_MHZ = 1420.405751768
SIGGEN_AMP_DBM = -80.0

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
    """Build [CAL, GAL-1420, GAL-1421] experiment list."""
    pointing = dict(alt_deg=alt_deg, az_deg=az_deg)
    return [
        CalExperiment(
            prefix='SKY-SWITCH-FREQ-CAL',
            siggen_freq_mhz=SIGGEN_FREQ_MHZ,
            siggen_amp_dbm=SIGGEN_AMP_DBM,
            **pointing,
            **COMMON,
        ),
        ObsExperiment(prefix='GAL-1420', center_freq=LO_FREQ_1, **pointing, **COMMON),
        ObsExperiment(prefix='GAL-1421', center_freq=LO_FREQ_2, **pointing, **COMMON),
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

    print(f'Starting {total} captures (CAL + 2 LO steps)...')
    print(f'  CAL:  {SIGGEN_FREQ_MHZ} MHz,  {SIGGEN_AMP_DBM} dBm')
    print(f'  LO:   {LO_FREQ_1 / 1e6:.0f} MHz,  {LO_FREQ_2 / 1e6:.0f} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=LO_FREQ_1, sample_rate=2.56e6, gain=0.0)
    synth = SignalGenerator()

    try:
        runner = QueueRunner(experiments=experiments, sdr=sdr, synth=synth, confirm=False)
        t0 = time.time()
        paths = runner.run()
        elapsed = time.time() - t0
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
