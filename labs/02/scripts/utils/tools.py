"""Shared schema, metrics, and manifest helpers for Lab 2 capture scripts."""

import csv
import math
from pathlib import Path

import numpy as np

from ugradiolab import Record, Spectrum

# ---------------------------------------------------------------------------
# LO frequencies — single source of truth for both scripts

LO_FREQS_HZ = (1420.0e6, 1421.0e6)
LO_FREQS_MHZ = tuple(int(lo / 1e6) for lo in LO_FREQS_HZ)  # (1420, 1421)

# ---------------------------------------------------------------------------
# Manifest schema

_METRIC_KEYS = ["i_min", "i_max", "i_median", "i_rms", "i_clip_frac",
                "q_min", "q_max", "q_median", "q_rms", "q_clip_frac"]

MANIFEST_FIELDS = [
    "set_id",
    "cable_length_m", "power_meter_dbm", "siggen_freq_mhz", "siggen_amp_dbm",
    *[f"lo{lo}_path"        for lo in LO_FREQS_MHZ],
    *[f"lo{lo}_total_power" for lo in LO_FREQS_MHZ],
    "total_power_ratio_1420_over_1421",
    *[f"lo{lo}_{key}" for lo in LO_FREQS_MHZ for key in _METRIC_KEYS],
]

# ---------------------------------------------------------------------------
# Manifest I/O

def next_id_from_manifest(manifest_path: Path, field: str) -> int:
    if not manifest_path.is_file():
        return 1
    max_id = 0
    with manifest_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
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


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    append_csv_row(manifest_path, MANIFEST_FIELDS, row)


def remove_manifest_rows_for_paths(manifest_path: Path, paths: dict[int, str]) -> int:
    """Remove rows whose lo<N>_path columns all match the given paths dict."""
    if not manifest_path.is_file():
        return 0
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or MANIFEST_FIELDS
        kept_rows: list[dict[str, str]] = []
        removed = 0
        for row in reader:
            if all(row.get(f"lo{lo}_path", "") == paths[lo] for lo in LO_FREQS_MHZ):
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


# ---------------------------------------------------------------------------
# Capture metrics

def _channel_stats(channel: np.ndarray) -> dict[str, float]:
    flat = np.asarray(channel, dtype=float).ravel()
    return {
        "min":       float(np.min(flat)),
        "max":       float(np.max(flat)),
        "median":    float(np.median(flat)),
        "rms":       float(np.sqrt(np.mean(np.square(flat)))),
        "clip_frac": float(np.mean(np.abs(flat) >= 127.0)),
    }


def compute_capture_metrics(path: str | Path) -> dict[str, float]:
    record = Record.load(path)
    i_stats = _channel_stats(record.data.real)
    q_stats = _channel_stats(record.data.imag)
    total_power = float(Spectrum.from_data(path).total_power)
    return {
        "total_power": total_power,
        "total_power_db": float(10.0 * np.log10(total_power)) if total_power > 0 else math.nan,
        "i_min":       i_stats["min"],   "i_max":    i_stats["max"],
        "i_median":    i_stats["median"],"i_rms":    i_stats["rms"],
        "i_clip_frac": i_stats["clip_frac"],
        "q_min":       q_stats["min"],   "q_max":    q_stats["max"],
        "q_median":    q_stats["median"],"q_rms":    q_stats["rms"],
        "q_clip_frac": q_stats["clip_frac"],
    }


def print_capture_metrics(
    lo_mhz: int | float,
    metrics: dict[str, float],
    *,
    include_total_power_db: bool = False,
) -> None:
    lo_label = int(round(lo_mhz))
    print(f"  LO={lo_label} MHz metrics:")
    if include_total_power_db:
        print(
            "    total_power={total_power:.6g} ({total_power_db:.3f} dB)  "
            "I[min,max,median,rms,clip]={i_min:.1f},{i_max:.1f},{i_median:.1f},{i_rms:.3f},{i_clip_frac:.4f}  "
            "Q[min,max,median,rms,clip]={q_min:.1f},{q_max:.1f},{q_median:.1f},{q_rms:.3f},{q_clip_frac:.4f}".format(
                **metrics
            )
        )
        return
    print(
        "    total_power={total_power:.6g}  "
        "I[min,max,median,rms,clip]={i_min:.1f},{i_max:.1f},{i_median:.1f},{i_rms:.3f},{i_clip_frac:.4f}  "
        "Q[min,max,median,rms,clip]={q_min:.1f},{q_max:.1f},{q_median:.1f},{q_rms:.3f},{q_clip_frac:.4f}".format(
            **metrics
        )
    )


def build_manifest_row(
    *,
    set_id: int,
    paths: dict[int, str],          # {1420: path_str, 1421: path_str}
    metrics: dict[int, dict],       # {1420: metrics_dict, 1421: metrics_dict}
    cable_length_m: float = math.nan,
    power_meter_dbm: float = math.nan,
    siggen_freq_mhz: float = math.nan,
    siggen_amp_dbm: float = math.nan,
) -> dict[str, object]:
    powers = {lo: metrics[lo]["total_power"] for lo in LO_FREQS_MHZ}
    p0, p1 = powers[LO_FREQS_MHZ[0]], powers[LO_FREQS_MHZ[1]]
    row: dict[str, object] = {
        "set_id":             set_id,
        "cable_length_m":     cable_length_m,
        "power_meter_dbm":    power_meter_dbm,
        "siggen_freq_mhz":    siggen_freq_mhz,
        "siggen_amp_dbm":     siggen_amp_dbm,
        "total_power_ratio_1420_over_1421": p0 / p1 if p1 != 0.0 else math.nan,
    }
    for lo in LO_FREQS_MHZ:
        row[f"lo{lo}_path"]        = paths[lo]
        row[f"lo{lo}_total_power"] = metrics[lo]["total_power"]
        for key in _METRIC_KEYS:
            row[f"lo{lo}_{key}"] = metrics[lo][key]
    return row


# ---------------------------------------------------------------------------
# File helpers

def delete_capture_file(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    try:
        target.unlink()
        print(f"  deleted {target}")
    except OSError as exc:
        print(f"  warning: failed to delete {target}: {exc}")
