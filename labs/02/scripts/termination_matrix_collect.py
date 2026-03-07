#!/usr/bin/env python3
"""Collect termination-state timing picks and derive cable propagation metrics.

This script standardizes manual reflectometry logging for a matrix of:
- termination state (open / 50ohm / SDR unpowered / SDR powered)
- test frequency
- repeat index

For each trial, user-entered waveform timing picks are logged to a manifest CSV,
and summary statistics are written to a summary CSV.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

C_LIGHT_MPS = 299_792_458.0
DEFAULT_STATES = ("open", "50ohm", "sdr_unpowered", "sdr_powered")
DEFAULT_FREQS_MHZ = (0.5, 2.5, 5.0, 10.0)

# ---------------------------------------------------------------------------
# Session configuration
OUTDIR = "data/lab02/termination_matrix"
MANIFEST_PATH = "data/lab02/termination_matrix/manifest.csv"
SUMMARY_PATH = "data/lab02/termination_matrix/summary.csv"
STATES = DEFAULT_STATES
FREQ_LIST_MHZ = DEFAULT_FREQS_MHZ
REPEATS = 1
CABLE_LENGTH_SDR_M: float | None = None
CABLE_LENGTH_METER_M = math.nan
START_TRIAL_ID: int | None = None
DRY_RUN = False

MANIFEST_FIELDS = [
    "trial_id",
    "session_start_iso",
    "trial_start_iso",
    "trial_end_iso",
    "state",
    "test_freq_mhz",
    "repeat_idx",
    "delta_t_ns",
    "incident_vpp",
    "reflected_vpp",
    "gamma_mag",
    "vswr",
    "cable_length_sdr_m",
    "cable_length_meter_m",
    "v_sdr_mps",
    "v_sdr_over_c",
    "v_meter_mps",
    "v_meter_over_c",
    "notes",
]

SUMMARY_FIELDS = [
    "state",
    "test_freq_mhz",
    "n_trials",
    "delta_t_ns_mean",
    "delta_t_ns_std",
    "delta_t_ns_p16",
    "delta_t_ns_p84",
    "gamma_mag_mean",
    "gamma_mag_std",
    "vswr_median",
    "v_sdr_mps_mean",
    "v_sdr_mps_std",
    "v_sdr_over_c_mean",
    "v_meter_mps_mean",
    "v_meter_mps_std",
    "v_meter_over_c_mean",
]

def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_float(raw: str | float | None) -> float:
    if raw is None:
        return math.nan
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text == "":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


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


def prompt_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, please try again.")


def prompt_float_allow_blank(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return math.nan
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, please try again.")


def derive_gamma_and_vswr(incident_vpp: float, reflected_vpp: float) -> tuple[float, float]:
    if not np.isfinite(incident_vpp) or not np.isfinite(reflected_vpp):
        return math.nan, math.nan
    if incident_vpp <= 0:
        return math.nan, math.nan

    gamma = abs(reflected_vpp / incident_vpp)
    if gamma >= 1.0:
        return float(gamma), math.inf
    return float(gamma), float((1.0 + gamma) / (1.0 - gamma))


def derive_speed(length_m: float, delta_t_ns: float) -> tuple[float, float]:
    if not np.isfinite(length_m) or not np.isfinite(delta_t_ns):
        return math.nan, math.nan
    if length_m <= 0 or delta_t_ns <= 0:
        return math.nan, math.nan

    v = 2.0 * length_m / (delta_t_ns * 1e-9)
    return float(v), float(v / C_LIGHT_MPS)


def load_manifest_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.is_file():
        return []

    rows: list[dict[str, float | str]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "state": row.get("state", "").strip(),
                    "test_freq_mhz": safe_float(row.get("test_freq_mhz")),
                    "delta_t_ns": safe_float(row.get("delta_t_ns")),
                    "gamma_mag": safe_float(row.get("gamma_mag")),
                    "vswr": safe_float(row.get("vswr")),
                    "v_sdr_mps": safe_float(row.get("v_sdr_mps")),
                    "v_sdr_over_c": safe_float(row.get("v_sdr_over_c")),
                    "v_meter_mps": safe_float(row.get("v_meter_mps")),
                    "v_meter_over_c": safe_float(row.get("v_meter_over_c")),
                }
            )
    return rows


def summarize_array(values: np.ndarray) -> tuple[float, float, float, float]:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
    p16, p84 = np.percentile(vals, [16.0, 84.0])
    return mean, std, float(p16), float(p84)


def build_summary_rows(rows: list[dict[str, float | str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[dict[str, float | str]]] = {}
    for row in rows:
        state = str(row["state"])
        freq = float(row["test_freq_mhz"])
        groups.setdefault((state, freq), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (state, freq), grows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        dt = np.array([safe_float(g["delta_t_ns"]) for g in grows], float)
        gm = np.array([safe_float(g["gamma_mag"]) for g in grows], float)
        vs = np.array([safe_float(g["vswr"]) for g in grows], float)
        vsdr = np.array([safe_float(g["v_sdr_mps"]) for g in grows], float)
        vsdr_c = np.array([safe_float(g["v_sdr_over_c"]) for g in grows], float)
        vmeter = np.array([safe_float(g["v_meter_mps"]) for g in grows], float)
        vmeter_c = np.array([safe_float(g["v_meter_over_c"]) for g in grows], float)

        dt_mean, dt_std, dt_p16, dt_p84 = summarize_array(dt)
        gm_mean, gm_std, _, _ = summarize_array(gm)
        vswr_finite = vs[np.isfinite(vs)]
        vswr_median = float(np.median(vswr_finite)) if vswr_finite.size else math.nan
        vsdr_mean, vsdr_std, _, _ = summarize_array(vsdr)
        vsdr_c_mean, _, _, _ = summarize_array(vsdr_c)
        vmeter_mean, vmeter_std, _, _ = summarize_array(vmeter)
        vmeter_c_mean, _, _, _ = summarize_array(vmeter_c)

        summary_rows.append(
            {
                "state": state,
                "test_freq_mhz": freq,
                "n_trials": len(grows),
                "delta_t_ns_mean": dt_mean,
                "delta_t_ns_std": dt_std,
                "delta_t_ns_p16": dt_p16,
                "delta_t_ns_p84": dt_p84,
                "gamma_mag_mean": gm_mean,
                "gamma_mag_std": gm_std,
                "vswr_median": vswr_median,
                "v_sdr_mps_mean": vsdr_mean,
                "v_sdr_mps_std": vsdr_std,
                "v_sdr_over_c_mean": vsdr_c_mean,
                "v_meter_mps_mean": vmeter_mean,
                "v_meter_mps_std": vmeter_std,
                "v_meter_over_c_mean": vmeter_c_mean,
            }
        )

    return summary_rows


def main() -> int:
    states = [str(state).strip() for state in STATES if str(state).strip()]
    freqs_mhz = [float(freq) for freq in FREQ_LIST_MHZ]
    if not states:
        raise ValueError("STATES must contain at least one state name.")
    if not freqs_mhz:
        raise ValueError("FREQ_LIST_MHZ must contain at least one frequency.")
    if REPEATS < 1:
        raise ValueError("REPEATS must be >= 1.")

    cable_length_sdr_m = (
        prompt_float("Cable length for SDR prior [m]: ")
        if CABLE_LENGTH_SDR_M is None
        else float(CABLE_LENGTH_SDR_M)
    )
    if not np.isfinite(cable_length_sdr_m) or cable_length_sdr_m <= 0.0:
        raise ValueError("CABLE_LENGTH_SDR_M must be a positive finite value.")
    cable_length_meter_m = float(CABLE_LENGTH_METER_M)

    outdir = Path(OUTDIR)
    manifest_path = Path(MANIFEST_PATH)
    summary_path = Path(SUMMARY_PATH)

    trial_id = (
        START_TRIAL_ID
        if START_TRIAL_ID is not None
        else next_id_from_manifest(manifest_path, "trial_id")
    )

    trials = [
        (state, freq_mhz, repeat_idx)
        for state in states
        for freq_mhz in freqs_mhz
        for repeat_idx in range(1, REPEATS + 1)
    ]

    print("Termination matrix collection")
    print(f"  states             : {', '.join(states)}")
    print(f"  freqs [MHz]        : {', '.join(f'{x:g}' for x in freqs_mhz)}")
    print(f"  repeats            : {REPEATS}")
    print(f"  cable length SDR   : {cable_length_sdr_m:.6g} m")
    if np.isfinite(cable_length_meter_m):
        print(f"  cable length meter : {cable_length_meter_m:.6g} m")
    else:
        print("  cable length meter : (not provided)")
    print(f"  trials             : {len(trials)}")
    print(f"  outdir             : {outdir}")
    print(f"  manifest           : {manifest_path}")
    print(f"  summary            : {summary_path}")

    if DRY_RUN:
        print("\nDry run enabled; no prompts/capture logging performed.")
        for idx, (state, freq_mhz, repeat_idx) in enumerate(trials, start=1):
            print(
                f"  [{idx:03d}] state={state} freq={freq_mhz:g} MHz "
                f"repeat={repeat_idx}"
            )
        return 0

    session_start_iso = iso_now()
    outdir.mkdir(parents=True, exist_ok=True)

    completed = 0
    for state, freq_mhz, repeat_idx in trials:
        print(
            f"\nTrial {trial_id:04d}: state={state} "
            f"freq={freq_mhz:g} MHz repeat={repeat_idx}"
        )

        ready = input("  Configure setup, then press Enter (or q to quit): ").strip().lower()
        if ready == "q":
            print("Aborted by user.")
            break

        trial_start_iso = iso_now()
        delta_t_ns = prompt_float("  Round-trip delay delta_t [ns]: ")
        incident_vpp = prompt_float_allow_blank("  Incident amplitude [Vpp] (Enter to skip): ")
        reflected_vpp = prompt_float_allow_blank("  Reflected amplitude [Vpp] (Enter to skip): ")
        notes = input("  Notes (optional): ").strip()
        trial_end_iso = iso_now()

        gamma_mag, vswr = derive_gamma_and_vswr(incident_vpp, reflected_vpp)
        v_sdr_mps, v_sdr_over_c = derive_speed(cable_length_sdr_m, delta_t_ns)
        v_meter_mps, v_meter_over_c = derive_speed(cable_length_meter_m, delta_t_ns)

        row = {
            "trial_id": trial_id,
            "session_start_iso": session_start_iso,
            "trial_start_iso": trial_start_iso,
            "trial_end_iso": trial_end_iso,
            "state": state,
            "test_freq_mhz": freq_mhz,
            "repeat_idx": repeat_idx,
            "delta_t_ns": delta_t_ns,
            "incident_vpp": incident_vpp,
            "reflected_vpp": reflected_vpp,
            "gamma_mag": gamma_mag,
            "vswr": vswr,
            "cable_length_sdr_m": cable_length_sdr_m,
            "cable_length_meter_m": cable_length_meter_m,
            "v_sdr_mps": v_sdr_mps,
            "v_sdr_over_c": v_sdr_over_c,
            "v_meter_mps": v_meter_mps,
            "v_meter_over_c": v_meter_over_c,
            "notes": notes,
        }

        append_csv_row(manifest_path, MANIFEST_FIELDS, row)
        completed += 1
        trial_id += 1

        print(
            f"  saved: delta_t={delta_t_ns:.3f} ns, "
            f"v_sdr={v_sdr_mps:.3e} m/s ({v_sdr_over_c:.4f} c)"
        )

    all_rows = load_manifest_rows(manifest_path)
    summary_rows = build_summary_rows(all_rows)
    write_csv_rows(summary_path, SUMMARY_FIELDS, summary_rows)

    print("\nSession complete")
    print(f"  completed trials : {completed}")
    print(f"  manifest rows    : {len(all_rows)}")
    print(f"  summary rows     : {len(summary_rows)}")
    print(f"  manifest         : {manifest_path}")
    print(f"  summary          : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
