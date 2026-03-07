#!/usr/bin/env python3
"""Collect SDR gain-sweep calibration data and fit linear-response summaries.

This script sweeps over:
- LO frequency (MHz)
- requested SDR gain (dB)
- signal-generator amplitude (dBm)

For each point it captures one calibrated SDR file, logs capture metrics to a
manifest CSV, and then writes an analysis summary CSV with linear-fit diagnostics
for:
1) response versus siggen amplitude (fixed LO, fixed SDR gain), and
2) response versus SDR gain (fixed LO, fixed siggen amplitude).
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from ugradio.sdr import SDR

from ugradiolab import Record, SignalGenerator, Spectrum
from ugradiolab.run import CalExperiment

DEFAULT_SIGGEN_FREQ_MHZ = 1420.405751768
DEFAULT_LO_LIST_MHZ = (1420.0, 1421.0)
DEFAULT_SIGGEN_AMP_LIST_DBM = (
    10.0,
    7.5,
    5.0,
    2.5,
    0.0,
    -2.5,
    -5.0,
    -7.5,
    -10.0,
    -12.5,
    -15.0,
    -17.5,
    -20.0,
    -22.5,
    -25.0,
    -27.5,
    -30.0,
)
DEFAULT_SDR_GAIN_LIST_DB = (0.0,)

# ---------------------------------------------------------------------------
# Session configuration
OUTDIR = "data/lab02/sdr_gain_sweep/raw"
MANIFEST_PATH = "data/lab02/sdr_gain_sweep/manifest.csv"
SUMMARY_PATH = "data/lab02/sdr_gain_sweep/summary.csv"
SIGGEN_DEVICE = "/dev/usbtmc0"
SIGGEN_FREQ_MHZ = DEFAULT_SIGGEN_FREQ_MHZ
LO_LIST_MHZ = DEFAULT_LO_LIST_MHZ
SIGGEN_AMP_LIST_DBM = DEFAULT_SIGGEN_AMP_LIST_DBM
SDR_GAIN_LIST_DB = DEFAULT_SDR_GAIN_LIST_DB
REPEATS = 1
SAMPLE_RATE_HZ = 2.56e6
NSAMPLES = 8192
NBLOCKS = 2048
SDR_DIRECT = False
MANUAL_METER = True
START_POINT_ID: int | None = None
DRY_RUN = False

MANIFEST_FIELDS = [
    "point_id",
    "session_start_iso",
    "point_start_iso",
    "point_end_iso",
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

SUMMARY_FIELDS = [
    "analysis_type",
    "lo_mhz",
    "fixed_sdr_gain_db",
    "fixed_siggen_amp_dbm",
    "n_points",
    "x_min",
    "x_max",
    "slope_db_per_db",
    "intercept_db",
    "rmse_db",
    "r2",
    "mean_total_power_db",
    "max_i_clip_frac",
    "max_q_clip_frac",
]

def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def next_id_from_manifest(path: Path, field: str) -> int:
    if not path.is_file():
        return 1
    max_id = 0
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = row.get(field, "").strip()
            if not raw:
                continue
            try:
                max_id = max(max_id, int(raw))
            except ValueError:
                continue
    return max_id + 1


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    total_power_db = float(10.0 * np.log10(total_power)) if total_power > 0 else math.nan
    return {
        "total_power": total_power,
        "total_power_db": total_power_db,
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


def print_capture_metrics(metrics: dict[str, float]) -> None:
    print(
        "    total_power={total_power:.6g} ({total_power_db:.3f} dB)  "
        "I[rms,clip]={i_rms:.3f},{i_clip_frac:.4f}  "
        "Q[rms,clip]={q_rms:.3f},{q_clip_frac:.4f}".format(**metrics)
    )


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
    sample_rate: float,
    nsamples: int,
    nblocks: int,
    direct: bool,
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
        nsamples=nsamples,
        nblocks=nblocks,
        sample_rate=sample_rate,
        center_freq=lo_hz,
        gain=sdr_gain_db,
        direct=direct,
        outdir=str(outdir),
        prefix=prefix,
        alt_deg=0.0,
        az_deg=0.0,
        siggen_freq_mhz=siggen_freq_mhz,
        siggen_amp_dbm=siggen_amp_dbm,
    )
    return exp.run(sdr, synth=synth)


def safe_float(raw: str | float | None) -> float:
    if raw is None:
        return math.nan
    if isinstance(raw, (float, int)):
        return float(raw)
    text = str(raw).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def load_manifest_rows(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "lo_mhz": safe_float(row.get("lo_mhz")),
                    "sdr_gain_db": safe_float(row.get("sdr_gain_db")),
                    "siggen_amp_dbm": safe_float(row.get("siggen_amp_dbm")),
                    "total_power_db": safe_float(row.get("total_power_db")),
                    "i_clip_frac": safe_float(row.get("i_clip_frac")),
                    "q_clip_frac": safe_float(row.get("q_clip_frac")),
                    "manual_meter_dbm": safe_float(row.get("manual_meter_dbm")),
                }
            )
    return rows


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    if xv.size < 2 or np.unique(xv).size < 2:
        return {
            "n_points": int(xv.size),
            "x_min": math.nan,
            "x_max": math.nan,
            "slope": math.nan,
            "intercept": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "mean_y": float(np.nanmean(yv)) if yv.size else math.nan,
        }

    slope, intercept = np.polyfit(xv, yv, 1)
    yhat = slope * xv + intercept
    resid = yv - yhat
    rmse = float(np.sqrt(np.mean(np.square(resid))))
    ss_res = float(np.sum(np.square(resid)))
    ss_tot = float(np.sum(np.square(yv - np.mean(yv))))
    r2 = math.nan if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)

    return {
        "n_points": int(xv.size),
        "x_min": float(np.min(xv)),
        "x_max": float(np.max(xv)),
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse": rmse,
        "r2": r2,
        "mean_y": float(np.mean(yv)),
    }


def build_summary_rows(rows: list[dict[str, float]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []

    # 1) Linearity vs siggen amplitude (fixed LO + fixed SDR gain)
    groups_a: dict[tuple[float, float], list[dict[str, float]]] = {}
    for row in rows:
        key = (row["lo_mhz"], row["sdr_gain_db"])
        groups_a.setdefault(key, []).append(row)

    for (lo_mhz, sdr_gain_db), grows in sorted(groups_a.items()):
        x = np.array([g["siggen_amp_dbm"] for g in grows], float)
        y = np.array([g["total_power_db"] for g in grows], float)
        fit = linear_fit(x, y)
        out.append(
            {
                "analysis_type": "linearity_vs_siggen_amp",
                "lo_mhz": lo_mhz,
                "fixed_sdr_gain_db": sdr_gain_db,
                "fixed_siggen_amp_dbm": math.nan,
                "n_points": fit["n_points"],
                "x_min": fit["x_min"],
                "x_max": fit["x_max"],
                "slope_db_per_db": fit["slope"],
                "intercept_db": fit["intercept"],
                "rmse_db": fit["rmse"],
                "r2": fit["r2"],
                "mean_total_power_db": fit["mean_y"],
                "max_i_clip_frac": float(np.nanmax([g["i_clip_frac"] for g in grows])),
                "max_q_clip_frac": float(np.nanmax([g["q_clip_frac"] for g in grows])),
            }
        )

    # 2) Response vs SDR gain (fixed LO + fixed siggen amplitude)
    groups_b: dict[tuple[float, float], list[dict[str, float]]] = {}
    for row in rows:
        key = (row["lo_mhz"], row["siggen_amp_dbm"])
        groups_b.setdefault(key, []).append(row)

    for (lo_mhz, siggen_amp_dbm), grows in sorted(groups_b.items()):
        x = np.array([g["sdr_gain_db"] for g in grows], float)
        y = np.array([g["total_power_db"] for g in grows], float)
        fit = linear_fit(x, y)
        out.append(
            {
                "analysis_type": "response_vs_sdr_gain",
                "lo_mhz": lo_mhz,
                "fixed_sdr_gain_db": math.nan,
                "fixed_siggen_amp_dbm": siggen_amp_dbm,
                "n_points": fit["n_points"],
                "x_min": fit["x_min"],
                "x_max": fit["x_max"],
                "slope_db_per_db": fit["slope"],
                "intercept_db": fit["intercept"],
                "rmse_db": fit["rmse"],
                "r2": fit["r2"],
                "mean_total_power_db": fit["mean_y"],
                "max_i_clip_frac": float(np.nanmax([g["i_clip_frac"] for g in grows])),
                "max_q_clip_frac": float(np.nanmax([g["q_clip_frac"] for g in grows])),
            }
        )

    return out


def prompt_float_allow_blank(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return math.nan
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, please try again.")


def close_synth(synth) -> None:
    try:
        synth.rf_off()
    except Exception:
        pass

    dev = getattr(synth, "_dev", None)
    if dev is not None:
        try:
            dev.close()
        except Exception:
            pass


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
    summary_path = Path(SUMMARY_PATH)

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
        f"  SDR profile       : sample_rate={SAMPLE_RATE_HZ/1e6:.3f} MHz, "
        f"nsamples={NSAMPLES}, nblocks={NBLOCKS}, direct={SDR_DIRECT}"
    )
    print(f"  manual meter      : {MANUAL_METER}")
    print(f"  points            : {len(points)}")
    print(f"  outdir            : {outdir}")
    print(f"  manifest          : {manifest_path}")
    print(f"  summary           : {summary_path}")

    if DRY_RUN:
        print("\nDry run enabled; no hardware capture performed.")
        for idx, point in enumerate(points, start=1):
            repeat_idx, lo_mhz, sdr_gain_db, siggen_amp_dbm = point
            print(
                f"  [{idx:03d}] rep={repeat_idx} LO={lo_mhz:g} MHz "
                f"gain={sdr_gain_db:g} dB amp={siggen_amp_dbm:g} dBm"
            )
        return 0

    session_start_iso = iso_now()
    outdir.mkdir(parents=True, exist_ok=True)

    sdr = None
    synth = None
    completed = 0
    try:
        sdr = SDR(
            direct=SDR_DIRECT,
            center_freq=lo_list_mhz[0] * 1e6,
            sample_rate=SAMPLE_RATE_HZ,
            gain=sdr_gain_list_db[0],
        )
        synth = SignalGenerator(device=SIGGEN_DEVICE)
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

            point_start_iso = iso_now()
            capture_path = run_capture_point(
                point_id=point_id,
                lo_mhz=lo_mhz,
                sdr_gain_db=sdr_gain_db,
                siggen_amp_dbm=siggen_amp_dbm,
                siggen_freq_mhz=SIGGEN_FREQ_MHZ,
                outdir=outdir,
                sample_rate=SAMPLE_RATE_HZ,
                nsamples=NSAMPLES,
                nblocks=NBLOCKS,
                direct=SDR_DIRECT,
                sdr=sdr,
                synth=synth,
            )
            point_end_iso = iso_now()

            metrics = compute_capture_metrics(capture_path)
            print_capture_metrics(metrics)

            row = {
                "point_id": point_id,
                "session_start_iso": session_start_iso,
                "point_start_iso": point_start_iso,
                "point_end_iso": point_end_iso,
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
            close_synth(synth)
        if sdr is not None:
            sdr.close()

    all_rows = load_manifest_rows(manifest_path)
    summary_rows = build_summary_rows(all_rows)
    write_csv_rows(summary_path, SUMMARY_FIELDS, summary_rows)

    print("\nSession complete")
    print(f"  completed points : {completed}")
    print(f"  manifest rows    : {len(all_rows)}")
    print(f"  summary rows     : {len(summary_rows)}")
    print(f"  manifest         : {manifest_path}")
    print(f"  summary          : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
