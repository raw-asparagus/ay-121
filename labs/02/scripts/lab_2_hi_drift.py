#!/usr/bin/env python3
"""Lab 2 HI drift scan — 21-cm neutral hydrogen observation at zenith.

Horn points at zenith (alt=90°, az=0°).  As Earth rotates the beam
traces a constant-declination arc at δ ≈ +37.87° (Berkeley latitude).

By default only sky captures are taken.  Pass --cal to interleave
calibration tones (requires a signal generator).

Usage:
    python lab_2_hi_drift.py [--outdir DIR] [--duration HOURS] [--cadence SEC]
                             [--nsamples N] [--nblocks N] [--cal]
                             [--cal-freq-mhz F] [--cal-dbm D] [--no-confirm]
"""

import argparse

from ugradio.sdr import SDR
from ugradiolab.experiment import ObsExperiment, CalExperiment
from ugradiolab.queue import QueueRunner

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


def build_plan(outdir, nsamples, nblocks, n_cycles, cal=False,
               cal_freq_mhz=CAL_FREQ_MHZ, cal_dbm=CAL_DBM):
    """Build the HI drift-scan experiment list."""

    common = dict(nsamples=nsamples, nblocks=nblocks, outdir=outdir,
                  **SDR_DEFAULTS)

    experiments = [ObsExperiment(prefix='HI-SKY-PRE', **ZENITH, **common)]

    for i in range(1, n_cycles + 1):
        tag = f'{i:03d}'
        if cal:
            experiments.append(CalExperiment(
                prefix=f'HI-CAL-{tag}', siggen_freq_mhz=cal_freq_mhz,
                siggen_amp_dbm=cal_dbm, **ZENITH, **common))
        experiments.append(ObsExperiment(
            prefix=f'HI-SKY-{tag}', **ZENITH, **common))

    experiments.append(ObsExperiment(prefix='HI-SKY-POST', **ZENITH, **common))
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description='Lab 2 HI drift scan — 21-cm observation at zenith')
    parser.add_argument('--outdir', default='data/lab2_hi_drift',
                        help='Output directory for .npz files')
    parser.add_argument('--duration', type=float, default=1.5,
                        help='Total observation duration in hours (default: 1.5)')
    parser.add_argument('--cadence', type=float, default=120.0,
                        help='Seconds between cycle starts (default: 120)')
    parser.add_argument('--nsamples', type=int, default=16384,
                        help='Samples per block (default: 16384)')
    parser.add_argument('--nblocks', type=int, default=100,
                        help='Number of blocks per capture (default: 100)')
    parser.add_argument('--cal', action='store_true',
                        help='Enable interleaved cal tones (requires signal generator)')
    parser.add_argument('--cal-freq-mhz', type=float, default=CAL_FREQ_MHZ,
                        help=f'Cal tone frequency in MHz (default: {CAL_FREQ_MHZ})')
    parser.add_argument('--cal-dbm', type=float, default=CAL_DBM,
                        help=f'Cal tone power in dBm (default: {CAL_DBM})')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Run without confirmation prompts')
    args = parser.parse_args()

    n_cycles = int(args.duration * 3600 / args.cadence)

    experiments = build_plan(args.outdir, args.nsamples, args.nblocks,
                             n_cycles, cal=args.cal,
                             cal_freq_mhz=args.cal_freq_mhz,
                             cal_dbm=args.cal_dbm)

    print(f'Lab 2 HI drift scan: {n_cycles} cycles, {len(experiments)} steps')
    print(f'Duration: {args.duration} hr | Cadence: {args.cadence} s')
    if args.cal:
        print(f'Cal tone: {args.cal_freq_mhz} MHz, {args.cal_dbm} dBm')
    else:
        print('Cal tones: disabled (sky only)')
    print(f'Output: {args.outdir}/')
    print()

    sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)
    synth = None
    if args.cal:
        from ugradiolab.drivers.SignalGenerator import connect as connect_siggen
        synth = connect_siggen()

    try:
        runner = QueueRunner(
            experiments=experiments,
            sdr=sdr,
            synth=synth,
            confirm=not args.no_confirm,
            cadence_sec=args.cadence,
        )
        paths = runner.run()
    finally:
        if synth is not None:
            synth.rf_off()
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {args.outdir}/')


if __name__ == '__main__':
    main()
