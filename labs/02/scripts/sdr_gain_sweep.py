#!/usr/bin/env python3
"""Collect SDR gain-sweep calibration data.

Hardware setup
--------------

Signal path (common trunk):
  Signal generator
    → [BNC to SMA] ZFSC-2-372-S+ 2-way splitter (port S)

Splitter output (1) — reference arm:
    → [SMA] Power meter  (read manually before each SDR capture point)

Splitter output (2) — SDR arm:
    → [SMA] 6-ft RG58 cable (50 Ω)
    → [SMA] 20 dB attenuator
    → [SMA] SDR

Notes:
- The splitter is connected directly to the signal generator.
- The SDR-arm attenuator is 20 dB (replacing the older 3 dB attenuator).

Workflow
--------

This script sweeps over:
- LO frequency (MHz)
- requested SDR gain (dB)
- signal-generator amplitude (dBm)

Per sweep point:
1. Set synth frequency and amplitude.
2. Optionally read the manual power meter (RF on, then RF off).
3. Capture one calibrated SDR file.
4. Compute capture metrics and append one row to the manifest CSV.
"""

import math
from pathlib import Path

from ugradio.sdr import SDR

from ugradiolab import SignalGenerator
from ugradiolab.run import CalExperiment
from utils.tools import (
    append_csv_row,
    compute_capture_metrics,
    count_csv_rows,
    next_id_from_manifest,
    print_capture_metrics,
)

# ---------------------------------------------------------------------------
# Session configuration
OUTDIR = "data/lab02/sdr_gain_sweep/raw"
MANIFEST_PATH = "data/lab02/sdr_gain_sweep/manifest.csv"

SIGGEN_FREQ_MHZ = 1420.405751768
LO_LIST_MHZ = (1420.0, 1421.0)
SIGGEN_AMP_LIST_DBM = tuple(18.0 - 2.5 * i for i in range(21))
SDR_GAIN_LIST_DB = (0.0,)
REPEATS = 1
MANUAL_METER = True
START_POINT_ID: int | None = None
DRY_RUN = False

COMMON_CAPTURE = dict(
    nsamples=8192,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    alt_deg=0.0,
    az_deg=0.0,
)

MANIFEST_FIELDS = [
    "point_id",
    "repeat_idx",
    "lo_mhz",
    "center_freq_hz",
    "sdr_gain_db",
    "siggen_freq_mhz",
    "siggen_amp_dbm",
    "manual_meter_dbm",
    "capture_path",
    "total_power",
    "total_power_db",
    "i_min",
    "i_max",
    "i_median",
    "i_rms",
    "i_clip_frac",
    "q_min",
    "q_max",
    "q_median",
    "q_rms",
    "q_clip_frac",
]


def sanitize_tag(value: float) -> str:
    text = f"{value:+.3f}"
    return text.replace("+", "p").replace("-", "m").replace(".", "p")


def run_capture_point(
    *,
    point_id: int,
    lo_mhz: float,
    sdr_gain_db: float,
    siggen_amp_dbm: float,
    siggen_freq_mhz: float,
    outdir: Path,
    sdr,
    synth,
) -> str:
    lo_hz = lo_mhz * 1e6
    prefix = (
        f"GAINSWEEP-p{point_id:04d}"
        f"-LO{int(round(lo_mhz))}"
        f"-G{sanitize_tag(sdr_gain_db)}"
        f"-A{sanitize_tag(siggen_amp_dbm)}"
    )

    exp = CalExperiment(
        sdr=sdr, synth=synth,
        **COMMON_CAPTURE,
        center_freq=lo_hz,
        gain=sdr_gain_db,
        outdir=str(outdir),
        prefix=prefix,
        siggen_freq_mhz=siggen_freq_mhz,
        siggen_amp_dbm=siggen_amp_dbm,
    )
    return exp.run()


def prompt_float_allow_blank(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return math.nan
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, please try again.")


def main() -> int:
    lo_list_mhz = [float(value) for value in LO_LIST_MHZ]
    siggen_amp_list_dbm = [float(value) for value in SIGGEN_AMP_LIST_DBM]
    sdr_gain_list_db = [float(value) for value in SDR_GAIN_LIST_DB]

    if not lo_list_mhz:
        raise ValueError("LO_LIST_MHZ must contain at least one value.")
    if not siggen_amp_list_dbm:
        raise ValueError("SIGGEN_AMP_LIST_DBM must contain at least one value.")
    if not sdr_gain_list_db:
        raise ValueError("SDR_GAIN_LIST_DB must contain at least one value.")
    if REPEATS < 1:
        raise ValueError("REPEATS must be >= 1.")

    outdir = Path(OUTDIR)
    manifest_path = Path(MANIFEST_PATH)
    point_id = (
        START_POINT_ID
        if START_POINT_ID is not None
        else next_id_from_manifest(manifest_path, "point_id")
    )

    points = [
        (repeat_idx, lo_mhz, sdr_gain_db, siggen_amp_dbm)
        for repeat_idx in range(1, REPEATS + 1)
        for lo_mhz in lo_list_mhz
        for sdr_gain_db in sdr_gain_list_db
        for siggen_amp_dbm in siggen_amp_list_dbm
    ]

    print("SDR gain-sweep session")
    print(f"  siggen freq [MHz] : {SIGGEN_FREQ_MHZ:.9f}")
    print(f"  LO list [MHz]     : {', '.join(f'{x:g}' for x in lo_list_mhz)}")
    print(f"  SDR gains [dB]    : {', '.join(f'{x:g}' for x in sdr_gain_list_db)}")
    print(f"  siggen amps [dBm] : {', '.join(f'{x:g}' for x in siggen_amp_list_dbm)}")
    print(f"  repeats           : {REPEATS}")
    print(
        f"  SDR profile       : sample_rate={COMMON_CAPTURE['sample_rate']/1e6:.3f} MHz, "
        f"nsamples={COMMON_CAPTURE['nsamples']}, nblocks={COMMON_CAPTURE['nblocks']}, "
        f"direct={COMMON_CAPTURE['direct']}"
    )
    print(f"  manual meter      : {MANUAL_METER}")
    print(f"  points            : {len(points)}")
    print(f"  outdir            : {outdir}")
    print(f"  manifest          : {manifest_path}")

    if DRY_RUN:
        print("\nDry run enabled; no hardware capture performed.")
        for idx, point in enumerate(points, start=1):
            repeat_idx, lo_mhz, sdr_gain_db, siggen_amp_dbm = point
            print(
                f"  [{idx:03d}] rep={repeat_idx} LO={lo_mhz:g} MHz "
                f"gain={sdr_gain_db:g} dB amp={siggen_amp_dbm:g} dBm"
            )
        return 0

    outdir.mkdir(parents=True, exist_ok=True)

    sdr = None
    synth = None
    completed = 0
    try:
        sdr = SDR(
            direct=COMMON_CAPTURE["direct"],
            center_freq=lo_list_mhz[0] * 1e6,
            sample_rate=COMMON_CAPTURE["sample_rate"],
            gain=sdr_gain_list_db[0],
        )
        synth = SignalGenerator()
        synth.set_freq_mhz(SIGGEN_FREQ_MHZ)

        for repeat_idx, lo_mhz, sdr_gain_db, siggen_amp_dbm in points:
            print(
                f"\nPoint {point_id:04d} / rep={repeat_idx} "
                f"LO={lo_mhz:g} MHz gain={sdr_gain_db:g} dB amp={siggen_amp_dbm:g} dBm"
            )

            manual_meter_dbm = math.nan
            if MANUAL_METER:
                synth.set_freq_mhz(SIGGEN_FREQ_MHZ)
                synth.set_ampl_dbm(siggen_amp_dbm)
                synth.rf_on()
                try:
                    manual_meter_dbm = prompt_float_allow_blank(
                        "  Manual power-meter reading [dBm] (Enter to skip): "
                    )
                finally:
                    synth.rf_off()

            capture_path = run_capture_point(
                point_id=point_id,
                lo_mhz=lo_mhz,
                sdr_gain_db=sdr_gain_db,
                siggen_amp_dbm=siggen_amp_dbm,
                siggen_freq_mhz=SIGGEN_FREQ_MHZ,
                outdir=outdir,
                sdr=sdr,
                synth=synth,
            )

            metrics = compute_capture_metrics(capture_path)
            print_capture_metrics(lo_mhz, metrics, include_total_power_db=True)

            row = {
                "point_id": point_id,
                "repeat_idx": repeat_idx,
                "lo_mhz": lo_mhz,
                "center_freq_hz": lo_mhz * 1e6,
                "sdr_gain_db": sdr_gain_db,
                "siggen_freq_mhz": SIGGEN_FREQ_MHZ,
                "siggen_amp_dbm": siggen_amp_dbm,
                "manual_meter_dbm": manual_meter_dbm,
                "capture_path": capture_path,
                "total_power": metrics["total_power"],
                "total_power_db": metrics["total_power_db"],
                "i_min": metrics["i_min"],
                "i_max": metrics["i_max"],
                "i_median": metrics["i_median"],
                "i_rms": metrics["i_rms"],
                "i_clip_frac": metrics["i_clip_frac"],
                "q_min": metrics["q_min"],
                "q_max": metrics["q_max"],
                "q_median": metrics["q_median"],
                "q_rms": metrics["q_rms"],
                "q_clip_frac": metrics["q_clip_frac"],
            }
            append_csv_row(manifest_path, MANIFEST_FIELDS, row)
            completed += 1
            point_id += 1

    finally:
        if synth is not None:
            synth.close()
        if sdr is not None:
            sdr.close()

    manifest_row_count = count_csv_rows(manifest_path)

    print("\nSession complete")
    print(f"  completed points : {completed}")
    print(f"  manifest rows    : {manifest_row_count}")
    print(f"  manifest         : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
