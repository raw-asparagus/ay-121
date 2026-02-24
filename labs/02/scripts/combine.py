#!/usr/bin/env python3
"""Combine n files of per-LO-frequency captures into one"""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

from ugradiolab.data import Record

INDIR_DEFAULT  = 'data/lab02/human'
OUTDIR_DEFAULT = 'data/lab02/human_combined'

LO_FREQS = ('1420', '1421')


def _find_files(indir, lo_freq):
    """Return paths matching *-<lo_freq>-<index>_obs_*.npz, sorted by index."""
    raw = glob.glob(os.path.join(indir, f'*-{lo_freq}-*_obs_*.npz'))
    return sorted(raw, key=lambda p: _parse_index(p, lo_freq))


def _parse_index(path, lo_freq):
    """Extract run index from script-style names ending with -<lo_freq>-<i>_obs_."""
    name = Path(path).name
    m = re.match(rf'.*-{re.escape(lo_freq)}-(\d+)_obs_', name)
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

    for lo_freq in LO_FREQS:
        out_label = f'GAL-{lo_freq}'
        paths = _find_files(indir, lo_freq)
        if not paths:
            print(f'[{out_label}] No files found — skipping.')
            continue

        print(f'[{out_label}] Loading {len(paths)} files...')
        records = []
        for p in paths:
            r = Record.load(p)
            print(f'  {Path(p).name}  nblocks={r.nblocks}')
            records.append(r)

        combined = _combine(records)
        print(f'  Combined: nblocks={combined.nblocks}, nsamples={combined.nsamples}')

        outpath = os.path.join(outdir, f'{out_label}_combined.npz')
        combined.save(outpath)
        print(f'  -> {outpath}')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
