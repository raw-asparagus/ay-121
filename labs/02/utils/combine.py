#!/usr/bin/env python3
"""Combine n files of per-LO-frequency captures into one"""

import re
import sys
from pathlib import Path

import numpy as np

from ugradiolab import Record

INDIR_DEFAULT = Path("data/lab02/standard")
OUTDIR_DEFAULT = Path("data/lab02/standard_combined")

LO_FREQS = ("1420", "1421")


def find_lo_files(indir: str | Path, lo_freq: str) -> list[Path]:
    """Return paths matching *-<lo_freq>-<index>_obs_*.npz, sorted by index."""
    indir = Path(indir)
    raw = indir.glob(f"*-{lo_freq}-*_obs_*.npz")
    return sorted(raw, key=lambda path: _parse_index(path, lo_freq))


def _parse_index(path: str | Path, lo_freq: str) -> int:
    """Extract run index from script-style names ending with -<lo_freq>-<i>_obs_."""
    name = Path(path).name
    m = re.match(rf".*-{re.escape(lo_freq)}-(\d+)_obs_", name)
    if m is None:
        raise ValueError(f"Cannot parse run index from filename: {name!r}")
    return int(m.group(1))


def combine_records(records: list[Record]) -> Record:
    """Stack Records along the blocks axis; metadata taken from the first."""
    if not records:
        raise ValueError("Need at least one Record to combine.")
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
        obs_lat=r0.obs_lat,
        obs_lon=r0.obs_lon,
        obs_alt=r0.obs_alt,
        nblocks=data.shape[0],
        nsamples=data.shape[1],
        siggen_freq=r0.siggen_freq,
        siggen_amp=r0.siggen_amp,
        siggen_rf_on=r0.siggen_rf_on,
    )


def combine_capture_dir(
    indir: str | Path,
    outdir: str | Path,
    lo_freqs: tuple[str, ...] = LO_FREQS,
) -> list[Path]:
    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Input  : {indir}")
    print(f"Output : {outdir}")
    print()

    outputs: list[Path] = []
    for lo_freq in lo_freqs:
        out_label = f"GAL-{lo_freq}"
        paths = find_lo_files(indir, lo_freq)
        if not paths:
            print(f"[{out_label}] No files found — skipping.")
            continue

        print(f"[{out_label}] Loading {len(paths)} files...")
        records = []
        for p in paths:
            r = Record.load(p)
            print(f"  {p.name}  nblocks={r.nblocks}")
            records.append(r)

        combined = combine_records(records)
        print(f"  Combined: nblocks={combined.nblocks}, nsamples={combined.nsamples}")

        outpath = outdir / f"{out_label}_combined.npz"
        combined.save(outpath)
        outputs.append(outpath)
        print(f"  -> {outpath}")
        print()

    print("Done.")
    return outputs


def main():
    indir = Path(sys.argv[1]) if len(sys.argv) > 1 else INDIR_DEFAULT
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTDIR_DEFAULT
    combine_capture_dir(indir, outdir)


if __name__ == "__main__":
    main()
