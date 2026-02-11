#!/usr/bin/env python3
"""Lab 2 calibration script — 13-step observation plan.

Sequence:
  1.  Z-BASE-1      Zenith, generator OFF
  2.  Z-BASE-2      Zenith, generator OFF (repeat)
  3.  H-BASE-1      Horizontal, generator OFF
  4.  H-BASE-2      Horizontal, generator OFF (repeat)
  5.  Z-TONE-PWR1   Zenith, 1421.2058 MHz, -41 dBm (~35 counts)
  6.  Z-TONE-PWR2   Zenith, 1421.2058 MHz, -38 dBm (~50 counts)
  7.  Z-TONE-PWR3   Zenith, 1421.2058 MHz, -35 dBm (~71 counts, default)
  8.  Z-TONE-UP100  Zenith, 1421.3058 MHz, -35 dBm
  9.  Z-TONE-DN100  Zenith, 1421.1058 MHz, -35 dBm
 10.  Z-TONE-LOWER  Zenith, 1419.6058 MHz, -35 dBm
 11.  H-TONE        Horizontal, 1421.2058 MHz, -35 dBm
 12.  H-POST        Horizontal, generator OFF
 13.  Z-POST        Zenith, generator OFF

Usage:
    python lab_2_calibration.py [--outdir DATA_DIR] [--nsamples N] [--nblocks N]
"""

import argparse
import sys
import os

# Ensure ugradiolab is importable when running from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ugradio.sdr import SDR
from ugradiolab.drivers.siggen import connect as connect_siggen
from ugradiolab.experiment import ObsExperiment, CalExperiment, run_queue

# ---------------------------------------------------------------------------
# SDR defaults (I/Q mode for 21-cm work)
# ---------------------------------------------------------------------------
SDR_DEFAULTS = dict(
    direct=False,
    center_freq=1420e6,
    sample_rate=2.56e6,
    gain=0.0,
)

# Pointing shorthands
ZENITH = dict(alt_deg=90.0, az_deg=0.0)
HORIZONTAL = dict(alt_deg=0.0, az_deg=0.0)


def build_plan(outdir, nsamples, nblocks):
    """Build the 13-step calibration experiment list."""

    common = dict(nsamples=nsamples, nblocks=nblocks, outdir=outdir,
                  **SDR_DEFAULTS)

    experiments = [
        # --- Baselines (generator OFF) ---
        ObsExperiment(prefix='Z-BASE-1',  **ZENITH,     **common),
        ObsExperiment(prefix='Z-BASE-2',  **ZENITH,     **common),
        ObsExperiment(prefix='H-BASE-1',  **HORIZONTAL, **common),
        ObsExperiment(prefix='H-BASE-2',  **HORIZONTAL, **common),

        # --- Power sweep at zenith ---
        CalExperiment(prefix='Z-TONE-PWR1', siggen_freq_mhz=1421.2058,
                      siggen_amp_dbm=-41, **ZENITH, **common),
        CalExperiment(prefix='Z-TONE-PWR2', siggen_freq_mhz=1421.2058,
                      siggen_amp_dbm=-38, **ZENITH, **common),
        CalExperiment(prefix='Z-TONE-PWR3', siggen_freq_mhz=1421.2058,
                      siggen_amp_dbm=-35, **ZENITH, **common),

        # --- Frequency offsets at zenith ---
        CalExperiment(prefix='Z-TONE-UP100', siggen_freq_mhz=1421.3058,
                      siggen_amp_dbm=-35, **ZENITH, **common),
        CalExperiment(prefix='Z-TONE-DN100', siggen_freq_mhz=1421.1058,
                      siggen_amp_dbm=-35, **ZENITH, **common),
        CalExperiment(prefix='Z-TONE-LOWER', siggen_freq_mhz=1419.6058,
                      siggen_amp_dbm=-35, **ZENITH, **common),

        # --- Horizontal with tone ---
        CalExperiment(prefix='H-TONE', siggen_freq_mhz=1421.2058,
                      siggen_amp_dbm=-35, **HORIZONTAL, **common),

        # --- Post-cal baselines (generator OFF) ---
        ObsExperiment(prefix='H-POST', **HORIZONTAL, **common),
        ObsExperiment(prefix='Z-POST', **ZENITH,     **common),
    ]
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Lab 2 calibration sequence')
    parser.add_argument('--outdir', default='data/lab2_cal',
                        help='Output directory for .npz files')
    parser.add_argument('--nsamples', type=int, default=2048,
                        help='Samples per block')
    parser.add_argument('--nblocks', type=int, default=10,
                        help='Number of blocks per experiment')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Run without confirmation prompts')
    args = parser.parse_args()

    experiments = build_plan(args.outdir, args.nsamples, args.nblocks)

    print(f'Lab 2 calibration: {len(experiments)} steps')
    print(f'Output: {args.outdir}/')
    print()

    # Initialise hardware
    sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)
    synth = connect_siggen()

    try:
        paths = run_queue(experiments, sdr=sdr, synth=synth,
                          confirm=not args.no_confirm)
    finally:
        # Always turn off RF output when done
        synth.rf_off()
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {args.outdir}/')


if __name__ == '__main__':
    main()