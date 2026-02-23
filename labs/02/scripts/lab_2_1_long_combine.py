#!/usr/bin/env python3
"""Combine n files of per-LO-frequency captures from lab_2_1_long into one
Record per LO frequency (n * 2048 blocks each).

Input layout  (produced by lab_2_1_long.py)
----------------------------------------------
  <indir>/GAL-1420-{0..n}_obs_*.npz  →  <outdir>/GAL-1420_combined.npz
  <indir>/GAL-1421-{0..n}_obs_*.npz  →  <outdir>/GAL-1421_combined.npz

Each input file has nblocks=2048, nsamples=32768.
4 × 2048 = 8192 blocks in the combined output.

Usage
-----
  python lab_2_1_long_combine.py [indir [outdir]]

Defaults:
  indir  = data/lab_2_1_long
  outdir = data/lab_2_1_long_combined
"""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

from ugradiolab.data import Record

INDIR_DEFAULT  = 'data/lab_2_1_long'
OUTDIR_DEFAULT = 'data/lab_2_1_long_combined'

LO_LABELS              = ('GAL-1421',)
N_FILES_PER_LO         = 4
EXPECTED_NBLOCKS_PER_FILE  = 2048
EXPECTED_COMBINED_NBLOCKS  = N_FILES_PER_LO * EXPECTED_NBLOCKS_PER_FILE  # 8192


def _find_files(indir, lo_label):
    """Return paths matching <lo_label>-<index>_obs_*.npz, sorted by index."""
    raw = glob.glob(os.path.join(indir, f'{lo_label}-*_obs_*.npz'))
    return sorted(raw, key=lambda p: _parse_index(p, lo_label))


def _parse_index(path, lo_label):
    """Extract the integer run index from a filename like GAL-1420-3_obs_*.npz."""
    name = Path(path).name
    m = re.match(rf'{re.escape(lo_label)}-(\d+)_obs_', name)
    if m is None:
        raise ValueError(f'Cannot parse run index from filename: {name!r}')
    return int(m.group(1))


def _combine(records):
    """Stack Records along the blocks axis; metadata taken from the first."""
    data = np.concatenate([r.data for r in records], axis=0)
    r0 = records[0]
    return Record(
        data=data,
        sample_rate=r0.sample_rate,
        center_freq=r0.center_freq,
        gain=r0.gain,
        direct=r0.direct,
        unix_time=r0.unix_time,
        jd=r0.jd,
        lst=r0.lst,
        alt=r0.alt,
        az=r0.az,
        observer_lat=r0.observer_lat,
        observer_lon=r0.observer_lon,
        observer_alt=r0.observer_alt,
        nblocks=data.shape[0],
        nsamples=data.shape[1],
        siggen_freq=r0.siggen_freq,
        siggen_amp=r0.siggen_amp,
        siggen_rf_on=r0.siggen_rf_on,
    )


def main():
    indir  = sys.argv[1] if len(sys.argv) > 1 else INDIR_DEFAULT
    outdir = sys.argv[2] if len(sys.argv) > 2 else OUTDIR_DEFAULT

    os.makedirs(outdir, exist_ok=True)
    print(f'Input  : {indir}')
    print(f'Output : {outdir}')
    print()

    for lo_label in LO_LABELS:
        paths = _find_files(indir, lo_label)
        if not paths:
            print(f'[{lo_label}] No files found — skipping.')
            continue

        if len(paths) != N_FILES_PER_LO:
            print(
                f'[{lo_label}] WARNING: expected {N_FILES_PER_LO} files, '
                f'found {len(paths)} — proceeding anyway.'
            )

        print(f'[{lo_label}] Loading {len(paths)} files...')
        records = []
        for p in paths:
            r = Record.load(p)
            print(f'  {Path(p).name}  nblocks={r.nblocks}')
            if r.nblocks != EXPECTED_NBLOCKS_PER_FILE:
                print(
                    f'    WARNING: expected nblocks={EXPECTED_NBLOCKS_PER_FILE}, '
                    f'got {r.nblocks}'
                )
            records.append(r)

        combined = _combine(records)
        print(f'  Combined: nblocks={combined.nblocks}, nsamples={combined.nsamples}')

        if combined.nblocks != EXPECTED_COMBINED_NBLOCKS:
            print(
                f'  WARNING: expected {EXPECTED_COMBINED_NBLOCKS} combined blocks, '
                f'got {combined.nblocks}'
            )

        outpath = os.path.join(outdir, f'{lo_label}_combined.npz')
        combined.save(outpath)
        print(f'  -> {outpath}')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
