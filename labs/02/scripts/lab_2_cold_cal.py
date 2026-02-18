#!/usr/bin/env python3
"""Lab 2 cold calibration — SIGGEN power sweep at horizontal pointing.

Horn stays horizontal (alt=0, az=0) throughout.  Alternates between
SIGGEN OFF (baseline) and SIGGEN ON at 1420.7 MHz, sweeping from
-90 to -30 dBm in 4 dBm steps.  Interleaved baselines track drift.

Tone frequency: 1420.7 MHz
  Offset from center (1420.0 MHz) : +0.700 MHz
  Fraction of Nyquist (±1.28 MHz) :  54.7%  — well within the flat passband
  Distance from HI line (1420.406 MHz) : 0.294 MHz

Sequence (34 steps):
  1.  COLD-BASE-PRE    baseline (gen OFF)
  2.  COLD-TONE--90    1420.7 MHz, -90 dBm
  3.  COLD-BASE--90    baseline
  4.  COLD-TONE--86    1420.7 MHz, -86 dBm
  5.  COLD-BASE--86    baseline
      ...
  32. COLD-TONE--30    1420.7 MHz, -30 dBm
  33. COLD-BASE--30    baseline
  34. COLD-BASE-POST   baseline (gen OFF)

Usage:
    python lab_2_cold_cal.py
"""

from ugradio.sdr import SDR
from ugradiolab.drivers.signal_generator import SignalGenerator
from ugradiolab.experiment import ObsExperiment, CalExperiment
from ugradiolab.queue import QueueRunner

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_cold_cal'

SDR_DEFAULTS = dict(
    direct=False,
    center_freq=1420e6,
    sample_rate=2.56e6,
    gain=0.0,
)

HORIZONTAL = dict(alt_deg=0.0, az_deg=0.0)

# 1420.7 MHz: +0.700 MHz from center, 54.7% of Nyquist — flat passband region
TONE_FREQ = 1420.7  # MHz


def build_plan():
    """Build the 34-step cold calibration experiment list."""
    common = dict(outdir=OUTDIR, nsamples=32768, nblocks=10, **SDR_DEFAULTS)

    experiments = [
        ObsExperiment(prefix='COLD-BASE-PRE', **HORIZONTAL, **common),
    ]

    for dbm in range(-90, -29, 4):  # -90 to -30 in 4 dBm steps (16 levels)
        experiments.append(
            CalExperiment(
                prefix=f'COLD-TONE-{dbm}',
                siggen_freq_mhz=TONE_FREQ,
                siggen_amp_dbm=dbm,
                **HORIZONTAL,
                **common,
            )
        )
        experiments.append(
            ObsExperiment(prefix=f'COLD-BASE-{dbm}', **HORIZONTAL, **common)
        )

    experiments.append(
        ObsExperiment(prefix='COLD-BASE-POST', **HORIZONTAL, **common),
    )

    return experiments


def main():
    experiments = build_plan()

    print(f'Lab 2 cold calibration: {len(experiments)} steps')
    print(f'  Tone: {TONE_FREQ} MHz  (+0.700 MHz from center, 54.7% of Nyquist)')
    print(f'  Sweep: -90 to -30 dBm in 4 dBm steps')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)
    synth = SignalGenerator()

    try:
        runner = QueueRunner(
            experiments=experiments,
            sdr=sdr,
            synth=synth,
            confirm=False,
        )
        paths = runner.run()
    finally:
        synth.rf_off()
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
