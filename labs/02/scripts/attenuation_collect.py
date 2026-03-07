#!/usr/bin/env python3
"""Collect fixed-tone attenuation data across cable-length sets.

Workflow per set:
1. Enter cable length (m).
2. Turn RF on and enter manual power meter reading (dBm).
3. Turn RF off, switch cable path to SDR, and confirm.
4. Capture LO=1420 MHz and LO=1421 MHz.
5. Print SDR metrics, then choose to save or discard the set.
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

SIGGEN_FREQ_MHZ = 1420.405751768
LO_1420_HZ = 1420.0e6
LO_1421_HZ = 1421.0e6
LO_FREQS_HZ = (LO_1420_HZ, LO_1421_HZ)

# ---------------------------------------------------------------------------
# Session configuration
OUTDIR = "data/lab02/attenuation/raw"
MANIFEST_PATH = "data/lab02/attenuation/manifest.csv"
SIGGEN_DEVICE = "/dev/usbtmc0"
SIGGEN_AMP_DBM: float | None = None
SAMPLE_RATE_HZ = 2.56e6
NSAMPLES = 8192
NBLOCKS = 2048
SDR_GAIN_DB = 0.0
SDR_DIRECT = False
START_SET_ID: int | None = None


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

def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def prompt_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, please try again.")


def prompt_length_or_quit() -> tuple[float | None, bool]:
    while True:
        raw = input("Cable length [m] (or q to quit): ").strip()
        if raw.lower() == "q":
            return None, True
        try:
            value = float(raw)
        except ValueError:
            print("  Invalid number, please try again.")
            continue
        if value < 0:
            print("  Cable length must be non-negative.")
            continue
        return value, False


def prompt_yes_no(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Please enter y/yes or n/no.")


def sanitize_length_tag(length_m: float) -> str:
    text = f"{length_m:.3f}"
    return text.replace("-", "m").replace(".", "p")


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
    cable_length_m: float,
    lo_hz: float,
    outdir: str | Path,
    sample_rate: float,
    nsamples: int,
    nblocks: int,
    gain: float,
    direct: bool,
    siggen_amp_dbm: float,
    sdr,
    synth,
) -> str:
    lo_mhz = int(round(lo_hz / 1e6))
    length_tag = sanitize_length_tag(cable_length_m)
    prefix = f"ATTEN-set{set_id:04d}-LO{lo_mhz}-L{length_tag}m"

    exp = CalExperiment(
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
        siggen_freq_mhz=SIGGEN_FREQ_MHZ,
        siggen_amp_dbm=siggen_amp_dbm,
    )
    return exp.run(sdr, synth=synth)


def append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def remove_manifest_rows_for_paths(
    manifest_path: Path,
    *,
    lo1420_path: str,
    lo1421_path: str,
) -> int:
    if not manifest_path.is_file():
        return 0

    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or MANIFEST_FIELDS
        kept_rows: list[dict[str, str]] = []
        removed = 0
        for row in reader:
            if (
                row.get("lo1420_path", "") == lo1420_path
                and row.get("lo1421_path", "") == lo1421_path
            ):
                removed += 1
                continue
            kept_rows.append(row)

    if removed == 0:
        return 0

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    return removed


def delete_capture_file(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    try:
        target.unlink()
        print(f"  deleted {target}")
    except OSError as exc:
        print(f"  warning: failed to delete {target}: {exc}")


def _ratio(num: float, den: float) -> float:
    if den == 0.0:
        return math.nan
    return float(num / den)


def _close_synth(synth) -> None:
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
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(MANIFEST_PATH)

    session_start_iso = iso_now()
    set_id = (
        START_SET_ID
        if START_SET_ID is not None
        else next_set_id_from_manifest(manifest_path)
    )

    siggen_amp_dbm = SIGGEN_AMP_DBM
    if siggen_amp_dbm is None:
        siggen_amp_dbm = prompt_float("Signal generator amplitude [dBm]: ")

    print("Attenuation collection session")
    print(f"  Siggen frequency: {SIGGEN_FREQ_MHZ:.9f} MHz")
    print(f"  Siggen amplitude: {siggen_amp_dbm:.2f} dBm")
    print(
        f"  SDR profile: sample_rate={SAMPLE_RATE_HZ/1e6:.3f} MHz, "
        f"nsamples={NSAMPLES}, nblocks={NBLOCKS}, gain={SDR_GAIN_DB}, "
        f"direct={SDR_DIRECT}"
    )
    print(f"  Output directory: {outdir}")
    print(f"  Manifest: {manifest_path}")
    print()

    completed = 0

    sdr = None
    synth = None
    try:
        sdr = SDR(
            direct=SDR_DIRECT,
            center_freq=LO_1420_HZ,
            sample_rate=SAMPLE_RATE_HZ,
            gain=SDR_GAIN_DB,
        )
        synth = SignalGenerator(device=SIGGEN_DEVICE)
        synth.set_freq_mhz(SIGGEN_FREQ_MHZ)
        synth.set_ampl_dbm(siggen_amp_dbm)

        while True:
            print(f"Set {set_id:04d}")
            cable_length_m, quit_now = prompt_length_or_quit()
            if quit_now:
                break
            assert cable_length_m is not None

            synth.rf_on()
            try:
                power_meter_dbm = prompt_float(
                    "RF is ON. Manual power meter reading [dBm]: "
                )
            finally:
                try:
                    synth.rf_off()
                except Exception:
                    pass

            switch_confirm = input(
                "RF is OFF. Switch cable path to SDR, then press Enter "
                "(or q to quit): "
            ).strip().lower()
            if switch_confirm == "q":
                break

            set_start_iso = iso_now()
            path_1420 = run_capture_for_lo(
                set_id=set_id,
                cable_length_m=cable_length_m,
                lo_hz=LO_1420_HZ,
                outdir=outdir,
                sample_rate=SAMPLE_RATE_HZ,
                nsamples=NSAMPLES,
                nblocks=NBLOCKS,
                gain=SDR_GAIN_DB,
                direct=SDR_DIRECT,
                siggen_amp_dbm=siggen_amp_dbm,
                sdr=sdr,
                synth=synth,
            )
            metrics_1420 = compute_capture_metrics(path_1420)
            print_capture_metrics(1420, metrics_1420)

            path_1421 = run_capture_for_lo(
                set_id=set_id,
                cable_length_m=cable_length_m,
                lo_hz=LO_1421_HZ,
                outdir=outdir,
                sample_rate=SAMPLE_RATE_HZ,
                nsamples=NSAMPLES,
                nblocks=NBLOCKS,
                gain=SDR_GAIN_DB,
                direct=SDR_DIRECT,
                siggen_amp_dbm=siggen_amp_dbm,
                sdr=sdr,
                synth=synth,
            )
            metrics_1421 = compute_capture_metrics(path_1421)
            print_capture_metrics(1421, metrics_1421)

            set_end_iso = iso_now()
            row = {
                "set_id": set_id,
                "session_start_iso": session_start_iso,
                "set_start_iso": set_start_iso,
                "set_end_iso": set_end_iso,
                "cable_length_m": cable_length_m,
                "power_meter_dbm": power_meter_dbm,
                "siggen_freq_mhz": SIGGEN_FREQ_MHZ,
                "siggen_amp_dbm": siggen_amp_dbm,
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
            should_save = prompt_yes_no("Save this run? [y/n]: ")
            if should_save:
                append_manifest_row(manifest_path, row)
                completed += 1
                print(f"  saved set {set_id:04d}")
                print()
                set_id += 1
                continue

            delete_capture_file(path_1420)
            delete_capture_file(path_1421)
            remove_manifest_rows_for_paths(
                manifest_path,
                lo1420_path=path_1420,
                lo1421_path=path_1421,
            )
            print(f"  discarded set {set_id:04d}; manifest/data removed")
            print()
    finally:
        if synth is not None:
            _close_synth(synth)
        if sdr is not None:
            sdr.close()

    print("Session complete")
    print(f"  completed sets: {completed}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
