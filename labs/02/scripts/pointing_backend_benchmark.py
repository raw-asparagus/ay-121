#!/usr/bin/env python3
"""Benchmark matrix-based pointing against astropy using existing lab-02 captures.

This script reads observation metadata from existing NPZ captures and compares
the explicit rotation-matrix backend (`ugradiolab.pointing`) to astropy.
It writes per-file residuals to CSV for direct insertion into report tables.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ugradiolab.pointing import compare_pointing_backends


DEFAULT_DATA_ROOT = Path("data/lab02")
DEFAULT_OUTPUT_CSV = Path("labs/02/report/pointing_matrix_benchmark.csv")

# Targets are defined in Galactic coordinates per dataset.
DATASET_TARGETS_GAL = {
    "standard": (120.0, 0.0),
    "cygnus-x": (79.5043879159, 1.0005912555),
    "SGP": (0.0, -90.0),
}


def _dataset_files(data_root: Path, dataset: str, max_files: int | None) -> list[Path]:
    files = sorted((data_root / dataset).glob("*1420*_obs_*.npz"))
    if max_files is not None:
        return files[: max(0, int(max_files))]
    return files


def _load_metadata(path: Path) -> dict[str, float]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "jd": float(data["jd"]),
            "lst_rad": float(data["lst"]),
            "alt_deg": float(data["alt"]),
            "az_deg": float(data["az"]),
            "obs_lat_deg": float(data["obs_lat"]),
            "obs_lon_deg": float(data["obs_lon"]),
            "obs_alt_m": float(data["obs_alt"]),
            "unix_time": float(data["unix_time"]),
        }


def _summarize(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    d_alt = np.array([abs(float(r["d_alt_deg"])) for r in rows], float)
    d_az = np.array([abs(float(r["d_az_deg"])) for r in rows], float)
    d_ra = np.array([abs(float(r["d_ra_deg"])) for r in rows], float)
    d_dec = np.array([abs(float(r["d_dec_deg"])) for r in rows], float)

    print("Global residual summary (matrix - astropy):")
    print(f"  N files             : {len(rows)}")
    print(f"  max |dRA|   [deg]   : {np.max(d_ra):.6f}")
    print(f"  max |dDec|  [deg]   : {np.max(d_dec):.6f}")
    print(f"  max |dAlt|  [deg]   : {np.max(d_alt):.6f}")
    print(f"  max |dAz|   [deg]   : {np.max(d_az):.6f}")
    print(f"  rms dAlt    [deg]   : {np.sqrt(np.mean(d_alt**2)):.6f}")
    print(f"  rms dAz     [deg]   : {np.sqrt(np.mean(d_az**2)):.6f}")
    print()

    for dataset in sorted({str(r["dataset"]) for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        d_alt_sub = np.array([abs(float(r["d_alt_deg"])) for r in subset], float)
        d_az_sub = np.array([abs(float(r["d_az_deg"])) for r in subset], float)
        print(f"[{dataset}] N={len(subset)}  max|dAlt|={np.max(d_alt_sub):.6f}°  max|dAz|={np.max(d_az_sub):.6f}°")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--max-files-per-dataset",
        type=int,
        default=None,
        help="Optional cap for faster quick-check runs.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset, (gal_l_deg, gal_b_deg) in DATASET_TARGETS_GAL.items():
        files = _dataset_files(args.data_root, dataset, args.max_files_per_dataset)
        if not files:
            print(f"[warn] no LO1420 observation files found for dataset={dataset!r}")
            continue

        for path in files:
            meta = _load_metadata(path)
            result = compare_pointing_backends(
                gal_l_deg=gal_l_deg,
                gal_b_deg=gal_b_deg,
                jd=meta["jd"],
                lst_rad=meta["lst_rad"],
                lat_deg=meta["obs_lat_deg"],
                lon_deg=meta["obs_lon_deg"],
                obs_alt_m=meta["obs_alt_m"],
                recorded_alt_deg=meta["alt_deg"],
                recorded_az_deg=meta["az_deg"],
            )

            row: dict[str, object] = {
                "dataset": dataset,
                "filename": path.name,
                "gal_l_deg": gal_l_deg,
                "gal_b_deg": gal_b_deg,
                "jd": meta["jd"],
                "lst_rad": meta["lst_rad"],
                "unix_time": meta["unix_time"],
                "alt_recorded_deg": meta["alt_deg"],
                "az_recorded_deg": meta["az_deg"],
            }
            row.update(asdict(result))
            rows.append(row)

    if not rows:
        print("No benchmark rows generated.")
        return 1

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {args.output_csv}")
    _summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
