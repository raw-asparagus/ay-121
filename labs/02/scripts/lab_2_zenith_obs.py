#!/usr/bin/env python3
"""Lab 2 frequency-switched zenith observation — §3.2.2.

Horn points straight up (alt=90°) throughout.  Alternates between two LO
frequencies so the HI line falls in opposite halves of the baseband
spectrum, enabling the frequency-switched line measurement (s_on / s_off).

  LO-A: center_freq = 1420.0 MHz  →  HI line at +0.406 MHz (upper half)
  LO-B: center_freq = 1421.0 MHz  →  HI line at −0.594 MHz (lower half)

The resulting file pairs are passed to analysis as:
  s_on  = ZEN-A upper-half channels  (or ZEN-B lower-half channels)
  s_off = ZEN-B upper-half channels  (or ZEN-A lower-half channels)

Sequence (2 × N_PAIRS captures, fully automated):
  ZEN-A, ZEN-B, ZEN-A, ZEN-B, ...

Usage:
    python lab_2_zenith_obs.py
"""

from ugradio.sdr import SDR
from ugradiolab.experiment import ObsExperiment
from ugradiolab.queue import QueueRunner

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_zenith_obs'
N_PAIRS = 200

LO_A_FREQ = 1420.0e6  # HI line lands at +0.406 MHz in baseband
LO_B_FREQ = 1421.0e6  # HI line lands at −0.594 MHz in baseband

COMMON = dict(
    outdir=OUTDIR,
    nsamples=32768,
    nblocks=10,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
    alt_deg=90.0,
    az_deg=0.0,
)


def build_plan():
    """Build interleaved [A, B, A, B, ...] experiment list."""
    experiments = []
    for _ in range(N_PAIRS):
        experiments.append(ObsExperiment(prefix='ZEN-A', center_freq=LO_A_FREQ, **COMMON))
        experiments.append(ObsExperiment(prefix='ZEN-B', center_freq=LO_B_FREQ, **COMMON))
    return experiments


def main():
    experiments = build_plan()
    total = len(experiments)

    print(f'Lab 2 zenith frequency-switched observation: {total} captures ({N_PAIRS} pairs)')
    print(f'  LO-A: {LO_A_FREQ / 1e6:.1f} MHz  →  HI line at +0.406 MHz')
    print(f'  LO-B: {LO_B_FREQ / 1e6:.1f} MHz  →  HI line at -0.594 MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=LO_A_FREQ, sample_rate=2.56e6, gain=0.0)

    try:
        runner = QueueRunner(experiments=experiments, sdr=sdr, confirm=False)
        paths = runner.run()
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
