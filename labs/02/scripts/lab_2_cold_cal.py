#!/usr/bin/env python3
"""Lab 2 cold calibration — SIGGEN power sweep at horizontal pointing.

Horn stays horizontal (alt=0, az=0) throughout.  Alternates between
SIGGEN OFF (baseline) and SIGGEN ON at 1421.2058 MHz, sweeping from
-50 to -30 dBm in 4 dB steps.  Interleaved baselines track drift.

Sequence (44 steps):
  1.  COLD-BASE-PRE    baseline (gen OFF)
  2.  COLD-TONE--50    1421.2058 MHz, -50 dBm
  3.  COLD-BASE--50    baseline
  4.  COLD-TONE--49    1421.2058 MHz, -49 dBm
  5.  COLD-BASE--49    baseline
      ...
  42. COLD-TONE--30    1421.2058 MHz, -30 dBm
  43. COLD-BASE--30    baseline
  44. COLD-BASE-POST   baseline (gen OFF)

Usage:
    python lab_2_cold_cal.py
"""

from ugradio.sdr import SDR
from ugradiolab.drivers.SignalGenerator import SignalGenerator
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

COLD = dict(alt_deg=0.0, az_deg=0.0)
TONE_FREQ = 1421.2058  # MHz


def build_plan():
    """Build the 44-step cold calibration experiment list."""

    common = dict(outdir=OUTDIR, **SDR_DEFAULTS)

    experiments = [
        ObsExperiment(prefix='COLD-BASE-PRE', **COLD, **common),
    ]

    for dbm in range(-50, -29):  # -50 to -30 inclusive
        experiments.append(
            CalExperiment(
                prefix=f'COLD-TONE-{dbm}',
                siggen_freq_mhz=TONE_FREQ,
                siggen_amp_dbm=dbm,
                **COLD,
                **common,
            ))
        experiments.append(
            ObsExperiment(prefix=f'COLD-BASE-{dbm}', **COLD, **common))

    experiments.append(
        ObsExperiment(prefix='COLD-BASE-POST', **COLD, **common),
    )

    return experiments


def main():
    experiments = build_plan()

    print(f'Lab 2 cold calibration: {len(experiments)} steps')
    print(f'Output: {OUTDIR}/')
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
