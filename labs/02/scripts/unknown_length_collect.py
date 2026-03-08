#!/usr/bin/env python3
"""Collect one unknown-length cable set with ObsExperiment captures.

Hardware setup
--------------

Signal path (common trunk):
  Signal generator (+20 dBm)
    → [N-male to BNC] reference cable (RG58, 50 Ω, variable length)
    → [BNC to SMA] ZFSC-2-372-S+ 2-way splitter (port S)

Splitter output (1) — reference arm:
    → [SMA] Power meter  (read manually before each SDR capture)

Splitter output (2) — SDR arm:
    → [SMA] 6-ft RG58 cable (50 Ω)
    → [SMA] 3 dB attenuator
    → [SMA] SDR

Workflow:
1. Capture all LO frequencies with ObsExperiment.
2. Print SDR metrics and append one row to a CSV manifest.

This script intentionally does not connect to or control a signal generator.
"""

from __future__ import annotations

from pathlib import Path

from ugradio.sdr import SDR

from ugradiolab.run import ObsExperiment

from utils.tools import (
    LO_FREQS_HZ,
    next_id_from_manifest,
    compute_capture_metrics, print_capture_metrics,
    append_manifest_row, build_manifest_row,
)

# ---------------------------------------------------------------------------
# Session configuration

OUTDIR        = "data/lab02/unknown_length/raw"
MANIFEST_PATH = "data/lab02/unknown_length/manifest.csv"
START_SET_ID: int | None = None

COMMON_CAPTURE = dict(
    nsamples=8192,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
    alt_deg=0.0,
    az_deg=0.0,
)


# ---------------------------------------------------------------------------
# Capture

def _capture(*, set_id, lo_hz, sdr):
    lo_mhz = int(round(lo_hz / 1e6))
    exp = ObsExperiment(
        **COMMON_CAPTURE,
        center_freq=lo_hz,
        outdir=OUTDIR, prefix=f"UNKNOWN-set{set_id:04d}-LO{lo_mhz}",
    )
    return exp.run(sdr)


# ---------------------------------------------------------------------------

def main() -> int:
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(MANIFEST_PATH)

    set_id = START_SET_ID if START_SET_ID is not None else next_id_from_manifest(manifest_path, "set_id")

    print("Unknown-length cable capture session")
    print(
        "  SDR profile: "
        f"sample_rate={COMMON_CAPTURE['sample_rate']/1e6:.3f} MHz, "
        f"nsamples={COMMON_CAPTURE['nsamples']}, "
        f"nblocks={COMMON_CAPTURE['nblocks']}, "
        f"gain={COMMON_CAPTURE['gain']}, "
        f"direct={COMMON_CAPTURE['direct']}"
    )
    print(f"  Output directory: {outdir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Set ID: {set_id:04d}")
    print()

    sdr = None
    try:
        sdr = SDR(
            direct=COMMON_CAPTURE["direct"],
            center_freq=LO_FREQS_HZ[0],
            sample_rate=COMMON_CAPTURE["sample_rate"],
            gain=COMMON_CAPTURE["gain"],
        )

        paths = {}
        metrics = {}
        for lo_hz in LO_FREQS_HZ:
            lo_mhz = int(round(lo_hz / 1e6))
            path = _capture(set_id=set_id, lo_hz=lo_hz, sdr=sdr)
            m = compute_capture_metrics(path)
            print_capture_metrics(lo_mhz, m)
            paths[lo_mhz] = path
            metrics[lo_mhz] = m

        row = build_manifest_row(
            set_id=set_id,
            paths=paths, metrics=metrics,
        )
        append_manifest_row(manifest_path, row)
        print(f"  recorded set {set_id:04d}")
    finally:
        if sdr is not None:
            sdr.close()

    print()
    print("Session complete")
    print(f"  completed sets: 1")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
