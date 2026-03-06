#!/usr/bin/env python3
"""Collect one unknown-length cable set with ObsExperiment captures.

Workflow:
1. Start a setup countdown (default 5 minutes) to configure cable path.
2. Capture LO=1420 MHz and LO=1421 MHz with ObsExperiment.
3. Print SDR metrics and append one row to a CSV manifest.

This script intentionally does not connect to or control a signal generator.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ugradio.sdr import SDR

from ugradiolab import Record, Spectrum
from ugradiolab.run import ObsExperiment

LO_1420_HZ = 1420.0e6
LO_1421_HZ = 1421.0e6
LO_FREQS_HZ = (LO_1420_HZ, LO_1421_HZ)


MANIFEST_FIELDS = [
    "set_id",
    "session_start_iso",
    "set_start_iso",
    "set_end_iso",
    "cable_length_m",
    "power_meter_dbm",
    "siggen_freq_mhz",
    "siggen_amp_dbm",
    "lo1420_path",
    "lo1421_path",
    "lo1420_total_power",
    "lo1421_total_power",
    "total_power_ratio_1420_over_1421",
    "lo1420_i_min",
    "lo1420_i_max",
    "lo1420_i_median",
    "lo1420_i_rms",
    "lo1420_i_clip_frac",
    "lo1420_q_min",
    "lo1420_q_max",
    "lo1420_q_median",
    "lo1420_q_rms",
    "lo1420_q_clip_frac",
    "lo1421_i_min",
    "lo1421_i_max",
    "lo1421_i_median",
    "lo1421_i_rms",
    "lo1421_i_clip_frac",
    "lo1421_q_min",
    "lo1421_q_max",
    "lo1421_q_median",
    "lo1421_q_rms",
    "lo1421_q_clip_frac",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one unknown-length cable set at LO 1420/1421 using "
            "ObsExperiment captures and no signal-generator control."
        )
    )
    parser.add_argument(
        "--outdir",
        default="data/lab02/unknown_length/raw",
        help="Output directory for SDR Record files.",
    )
    parser.add_argument(
        "--manifest",
        default="data/lab02/unknown_length/manifest.csv",
        help="CSV manifest path (appended if it already exists).",
    )
    parser.add_argument("--sample-rate", type=float, default=2.56e6)
    parser.add_argument("--nsamples", type=int, default=8192)
    parser.add_argument("--nblocks", type=int, default=2048)
    parser.add_argument("--gain", type=float, default=0.0)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument(
        "--setup-seconds",
        type=int,
        default=300,
        help="Setup countdown duration before capture (default: 300s).",
    )
    parser.add_argument(
        "--start-set-id",
        type=int,
        default=None,
        help="Optional explicit set_id (overrides manifest-derived id).",
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def next_set_id_from_manifest(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        return 1

    max_set_id = 0
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get("set_id", "").strip()
            if not raw:
                continue
            try:
                max_set_id = max(max_set_id, int(raw))
            except ValueError:
                continue
    return max_set_id + 1


def _channel_stats(channel: np.ndarray) -> dict[str, float]:
    flat = np.asarray(channel, dtype=float).ravel()
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "clip_frac": float(np.mean(np.abs(flat) >= 127.0)),
    }


def compute_capture_metrics(path: str | Path) -> dict[str, float]:
    record = Record.load(path)
    i_stats = _channel_stats(record.data.real)
    q_stats = _channel_stats(record.data.imag)
    total_power = float(Spectrum.from_data(path).total_power)
    return {
        "total_power": total_power,
        "i_min": i_stats["min"],
        "i_max": i_stats["max"],
        "i_median": i_stats["median"],
        "i_rms": i_stats["rms"],
        "i_clip_frac": i_stats["clip_frac"],
        "q_min": q_stats["min"],
        "q_max": q_stats["max"],
        "q_median": q_stats["median"],
        "q_rms": q_stats["rms"],
        "q_clip_frac": q_stats["clip_frac"],
    }


def print_capture_metrics(lo_mhz: int, metrics: dict[str, float]) -> None:
    print(f"  LO={lo_mhz} MHz metrics:")
    print(
        "    total_power={total_power:.6g}  "
        "I[min,max,median,rms,clip]={i_min:.1f},{i_max:.1f},{i_median:.1f},{i_rms:.3f},{i_clip_frac:.4f}  "
        "Q[min,max,median,rms,clip]={q_min:.1f},{q_max:.1f},{q_median:.1f},{q_rms:.3f},{q_clip_frac:.4f}".format(
            **metrics
        )
    )


def run_capture_for_lo(
    *,
    set_id: int,
    lo_hz: float,
    outdir: str | Path,
    sample_rate: float,
    nsamples: int,
    nblocks: int,
    gain: float,
    direct: bool,
    sdr,
) -> str:
    lo_mhz = int(round(lo_hz / 1e6))
    prefix = f"UNKNOWN-set{set_id:04d}-LO{lo_mhz}"

    exp = ObsExperiment(
        nsamples=nsamples,
        nblocks=nblocks,
        sample_rate=sample_rate,
        center_freq=lo_hz,
        gain=gain,
        direct=direct,
        outdir=str(outdir),
        prefix=prefix,
        alt_deg=0.0,
        az_deg=0.0,
    )
    return exp.run(sdr)


def append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _ratio(num: float, den: float) -> float:
    if den == 0.0:
        return math.nan
    return float(num / den)


def _prompt_begin_or_quit() -> bool:
    raw = input(
        "Press Enter to start the 5-minute setup timer (or q to quit): "
    ).strip().lower()
    return raw != "q"


def run_setup_countdown(seconds: int) -> None:
    seconds = max(0, int(seconds))
    if seconds == 0:
        return

    print(f"Starting setup timer: {seconds} seconds")
    for remaining in range(seconds, 0, -1):
        print(f"\r  {remaining:3d}s remaining...   ", end="", flush=True)
        time.sleep(1)
    print()


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    session_start_iso = iso_now()
    set_id = (
        args.start_set_id
        if args.start_set_id is not None
        else next_set_id_from_manifest(manifest_path)
    )

    print("Unknown-length cable capture session")
    print(
        f"  SDR profile: sample_rate={args.sample_rate/1e6:.3f} MHz, "
        f"nsamples={args.nsamples}, nblocks={args.nblocks}, gain={args.gain}, "
        f"direct={args.direct}"
    )
    print(f"  Setup timer: {args.setup_seconds}s")
    print(f"  Output directory: {outdir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Set ID: {set_id:04d}")
    print()

    if not _prompt_begin_or_quit():
        print("Session aborted before capture.")
        print("Session complete")
        print("  completed sets: 0")
        print(f"  manifest: {manifest_path}")
        return 0

    run_setup_countdown(args.setup_seconds)

    completed = 0
    sdr = None
    try:
        sdr = SDR(
            direct=args.direct,
            center_freq=LO_1420_HZ,
            sample_rate=args.sample_rate,
            gain=args.gain,
        )

        set_start_iso = iso_now()
        path_1420 = run_capture_for_lo(
            set_id=set_id,
            lo_hz=LO_1420_HZ,
            outdir=outdir,
            sample_rate=args.sample_rate,
            nsamples=args.nsamples,
            nblocks=args.nblocks,
            gain=args.gain,
            direct=args.direct,
            sdr=sdr,
        )
        metrics_1420 = compute_capture_metrics(path_1420)
        print_capture_metrics(1420, metrics_1420)

        path_1421 = run_capture_for_lo(
            set_id=set_id,
            lo_hz=LO_1421_HZ,
            outdir=outdir,
            sample_rate=args.sample_rate,
            nsamples=args.nsamples,
            nblocks=args.nblocks,
            gain=args.gain,
            direct=args.direct,
            sdr=sdr,
        )
        metrics_1421 = compute_capture_metrics(path_1421)
        print_capture_metrics(1421, metrics_1421)

        set_end_iso = iso_now()
        nan = math.nan
        row = {
            "set_id": set_id,
            "session_start_iso": session_start_iso,
            "set_start_iso": set_start_iso,
            "set_end_iso": set_end_iso,
            "cable_length_m": nan,
            "power_meter_dbm": nan,
            "siggen_freq_mhz": nan,
            "siggen_amp_dbm": nan,
            "lo1420_path": path_1420,
            "lo1421_path": path_1421,
            "lo1420_total_power": metrics_1420["total_power"],
            "lo1421_total_power": metrics_1421["total_power"],
            "total_power_ratio_1420_over_1421": _ratio(
                metrics_1420["total_power"],
                metrics_1421["total_power"],
            ),
            "lo1420_i_min": metrics_1420["i_min"],
            "lo1420_i_max": metrics_1420["i_max"],
            "lo1420_i_median": metrics_1420["i_median"],
            "lo1420_i_rms": metrics_1420["i_rms"],
            "lo1420_i_clip_frac": metrics_1420["i_clip_frac"],
            "lo1420_q_min": metrics_1420["q_min"],
            "lo1420_q_max": metrics_1420["q_max"],
            "lo1420_q_median": metrics_1420["q_median"],
            "lo1420_q_rms": metrics_1420["q_rms"],
            "lo1420_q_clip_frac": metrics_1420["q_clip_frac"],
            "lo1421_i_min": metrics_1421["i_min"],
            "lo1421_i_max": metrics_1421["i_max"],
            "lo1421_i_median": metrics_1421["i_median"],
            "lo1421_i_rms": metrics_1421["i_rms"],
            "lo1421_i_clip_frac": metrics_1421["i_clip_frac"],
            "lo1421_q_min": metrics_1421["q_min"],
            "lo1421_q_max": metrics_1421["q_max"],
            "lo1421_q_median": metrics_1421["q_median"],
            "lo1421_q_rms": metrics_1421["q_rms"],
            "lo1421_q_clip_frac": metrics_1421["q_clip_frac"],
        }
        append_manifest_row(manifest_path, row)

        completed = 1
        print(f"  recorded set {set_id:04d}")
    finally:
        if sdr is not None:
            sdr.close()

    print()
    print("Session complete")
    print(f"  completed sets: {completed}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
