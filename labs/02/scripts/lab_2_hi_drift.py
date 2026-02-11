#!/usr/bin/env python3
"""Lab 2 HI drift scan — 21-cm neutral hydrogen observation at zenith.

Horn points at zenith (alt=90°, az=0°).  As Earth rotates the beam
traces a constant-declination arc at δ ≈ +37.87° (Berkeley latitude).
Interleaves sky captures with calibration tones every 2 minutes to
track gain drift.

Sequence (for 1.5 hr = 90 min → 45 cycles):
  1.  HI-SKY-PRE       sky obs (baseline reference)
  2.  HI-CAL-001       cal tone at −35 dBm
  3.  HI-SKY-001       sky obs
      (sleep until next 2-min mark)
  4.  HI-CAL-002       cal tone
  5.  HI-SKY-002       sky obs
      ...
  N-1. HI-CAL-045      cal tone
  N.   HI-SKY-POST     sky obs (final baseline)

Usage:
    python lab_2_hi_drift.py [--outdir DIR] [--duration HOURS] [--cadence SEC]
                             [--nsamples N] [--nblocks N]
                             [--cal-freq-mhz F] [--cal-dbm D] [--no-confirm]
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ugradio.sdr import SDR
from ugradiolab.drivers.siggen import connect as connect_siggen
from ugradiolab.experiment import (
    ObsExperiment, CalExperiment, _format_experiment,
)

# ---------------------------------------------------------------------------
SDR_DEFAULTS = dict(
    direct=False,
    center_freq=1420e6,
    sample_rate=2.56e6,
    gain=0.0,
)

ZENITH = dict(alt_deg=90.0, az_deg=0.0)
CAL_FREQ_MHZ = 1421.2058
CAL_DBM = -35.0


def main():
    parser = argparse.ArgumentParser(
        description='Lab 2 HI drift scan — 21-cm observation at zenith')
    parser.add_argument('--outdir', default='data/lab2_hi_drift',
                        help='Output directory for .npz files')
    parser.add_argument('--duration', type=float, default=1.5,
                        help='Total observation duration in hours (default: 1.5)')
    parser.add_argument('--cadence', type=float, default=120.0,
                        help='Seconds between cycle starts (default: 120)')
    parser.add_argument('--nsamples', type=int, default=2048,
                        help='Samples per block (default: 2048)')
    parser.add_argument('--nblocks', type=int, default=100,
                        help='Number of blocks per capture (default: 100)')
    parser.add_argument('--cal-freq-mhz', type=float, default=CAL_FREQ_MHZ,
                        help=f'Cal tone frequency in MHz (default: {CAL_FREQ_MHZ})')
    parser.add_argument('--cal-dbm', type=float, default=CAL_DBM,
                        help=f'Cal tone power in dBm (default: {CAL_DBM})')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Run without confirmation prompts')
    args = parser.parse_args()

    n_cycles = int(args.duration * 3600 / args.cadence)
    # 1 pre-baseline + n_cycles*(cal+sky) + 1 post-baseline
    total_steps = 1 + 2 * n_cycles + 1

    common = dict(nsamples=args.nsamples, nblocks=args.nblocks,
                  outdir=args.outdir, **SDR_DEFAULTS)

    print(f'Lab 2 HI drift scan: {n_cycles} cycles, {total_steps} steps')
    print(f'Duration: {args.duration} hr | Cadence: {args.cadence} s')
    print(f'Cal tone: {args.cal_freq_mhz} MHz, {args.cal_dbm} dBm')
    print(f'Output: {args.outdir}/')
    print()

    sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)
    synth = connect_siggen()

    paths = []
    step = 0

    def run_one(exp):
        nonlocal step
        step += 1
        print(_format_experiment(exp, step, total_steps))
        if not args.no_confirm:
            resp = input('  [Enter]=run  s=skip  q=quit: ').strip().lower()
            if resp == 'q':
                return None  # signal abort
            if resp == 's':
                print('  skipped.')
                return 'skipped'
        path = exp.run(sdr, synth=synth)
        paths.append(path)
        print(f'  -> {path}')
        return path

    try:
        # --- Pre-baseline ---
        result = run_one(ObsExperiment(prefix='HI-SKY-PRE', **ZENITH, **common))
        if result is None:
            print('Aborted.')
            return

        # --- Main loop ---
        for i in range(1, n_cycles + 1):
            cycle_start = time.time()
            tag = f'{i:03d}'

            # Cal tone
            result = run_one(CalExperiment(
                prefix=f'HI-CAL-{tag}',
                siggen_freq_mhz=args.cal_freq_mhz,
                siggen_amp_dbm=args.cal_dbm,
                **ZENITH, **common))
            if result is None:
                print('Aborted.')
                return

            # Sky observation
            result = run_one(ObsExperiment(
                prefix=f'HI-SKY-{tag}', **ZENITH, **common))
            if result is None:
                print('Aborted.')
                return

            # Sleep until the next cadence boundary
            if i < n_cycles:
                elapsed = time.time() - cycle_start
                wait = args.cadence - elapsed
                if wait > 0:
                    print(f'  sleeping {wait:.1f}s until next cycle...')
                    time.sleep(wait)

        # --- Post-baseline ---
        run_one(ObsExperiment(prefix='HI-SKY-POST', **ZENITH, **common))

    finally:
        synth.rf_off()
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {args.outdir}/')


if __name__ == '__main__':
    main()