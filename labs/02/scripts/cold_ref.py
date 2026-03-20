#!/usr/bin/env python3
"""Lab 2 cold-sky reference observation at l=165°, b=36°.

RF chain used for this sky observation (same chain as standard/cygnus-x/human):
  big horn
    -[SMA to SMA]-> Mini Circuits ZKL-33ULN-S+
    -[SMA to SMA]-> Reactel 1420-15S11 Bandpass Filter
    -[SMA to SMA]-> Mini Circuits ZKL-33ULN-S+
    -[SMA to SMA]-> Mini Circuits ZKL-33ULN-S+
    -[SMA to N]-> cable of unknown length (~35 m prior from power-loss calibration)
    -[N to SMA]-> K&L Microwave Inc 5B120-1380/160-0 Bandpass Filter
    -[SMA to SMA]-> Wideband RF Amplifier (~20 dB, 10-2000 MHz)
    -[SMA to SMA]-> Narrowband RF Amplifier (~20 dB, 1-2 GHz)
    -[SMA to SMA]-> SDR

  LO:   1420 – 1421 MHz in 1 MHz steps  →  HI line offset per step:
    1420 MHz  →  +0.406 MHz
    1421 MHz  →  −0.594 MHz

Usage:
    python cold_ref.py
"""

import time

from ugradio.sdr import SDR
import ugradio.timing as timing

from ugradiolab.capture import ObsExperiment, SequentialRunner
from ugradiolab.io import get_unix_time

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab02/cold_ref'

ALT_DEG = 50.0   # degrees — high enough to avoid ground spillover
AZ_DEG  = 50.0   # degrees

GAL_L   = 165.0  # degrees  (for reference / metadata)
GAL_B   =  36.0  # degrees

FREQ_1  = 1420.0e6
FREQ_2  = 1421.0e6

ITERATIONS = 16

COMMON = dict(
    outdir=OUTDIR,
    nsamples=8192,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
    alt_deg=ALT_DEG,
    az_deg=AZ_DEG,
)


# ---------------------------------------------------------------------------

def build_plan(sdr):
    experiments = []
    for i in range(ITERATIONS):
        experiments.append(ObsExperiment(sdr=sdr, prefix=f'COLD-{FREQ_1 / 1e6:.0f}-{i}', center_freq=FREQ_1,
                                         **COMMON))
        experiments.append(ObsExperiment(sdr=sdr, prefix=f'COLD-{FREQ_2 / 1e6:.0f}-{i}', center_freq=FREQ_2,
                                         **COMMON))

    return experiments


def main():
    print('Lab 2 cold-sky reference observation')
    print()

    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)

    print(f'  Galactic        :  l = {GAL_L:.1f}°,  b = {GAL_B:.1f}°')
    print(f'  Local alt/az    :  Alt = {ALT_DEG:.1f}°,  Az = {AZ_DEG:.1f}°')
    print(f'  Julian date     :  {jd:.5f}')
    print()

    print(f'  >>> Point the horn to:  Alt = {ALT_DEG:.1f}°,  Az = {AZ_DEG:.1f}° <<<')
    print()
    input('  Press Enter once the horn is pointed and you are ready to begin: ')
    print()

    sdr = SDR(direct=False, center_freq=FREQ_1,
              sample_rate=COMMON['sample_rate'], gain=COMMON['gain'])

    experiments = build_plan(sdr)
    total = len(experiments)

    print(f'Starting {total} captures...')
    print(f'  LO:   {FREQ_1 / 1e6:.0f} & {FREQ_2 / 1e6:.0f} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    try:
        runner = SequentialRunner(experiments=experiments, confirm=False)
        t0 = time.time()
        paths = runner.run()
        elapsed = time.time() - t0
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
