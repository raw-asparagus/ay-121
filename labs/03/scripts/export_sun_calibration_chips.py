"""Collate sanitized Sun calibration captures into one processed NPZ per chip."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from astropy.coordinates import get_sun
from astropy.time import Time

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ugradiolab import NCH_LON_DEG

LAB03_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = LAB03_DIR / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from constants import F_RF0_HZ, F_S_HZ, N_FFT, PLOT_BAND_GHZ
from captures import load_capture_series
from chips import segment_capture_times_by_gap
from dc import local_real_dc_correction

BAD_CHANNELS = np.array([0, 256, 512, 768], dtype=int)
NOMINAL_FRINGE_PERIOD_SEC = 40.0
LOCAL_DC_PERIODS = 2.5


def _lst_deg(unix_t: np.ndarray) -> np.ndarray:
    jd = np.asarray(unix_t, dtype=float) / 86400.0 + 2440587.5
    t_century = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + t_century**2 * 0.000387933
        - t_century**3 / 38710000.0
    )
    return (gmst_deg + NCH_LON_DEG) % 360.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one processed Sun-calibration NPZ per chip.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/lab03/sun_calibration"),
        help="Directory containing sanitized top-level raw Sun calibration NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to receive chip NPZ files. Defaults to <data-dir>/chips.",
    )
    parser.add_argument(
        "--chip-offset",
        type=int,
        default=0,
        help="Starting chip index. Use to append new chips alongside existing ones.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or (data_dir / "chips")).resolve()
    chip_offset = args.chip_offset

    paths = sorted(data_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No top-level NPZ files found in {data_dir}.")

    captures = load_capture_series(paths)
    order = np.argsort(captures.unix_mid)
    captures_s = captures.take(order)

    unix_mid = captures_s.unix_mid
    sun = get_sun(Time(unix_mid, format="unix"))
    sun_ra_deg = np.asarray(sun.ra.deg, dtype=float)
    sun_dec_deg = np.asarray(sun.dec.deg, dtype=float)
    lst_deg = _lst_deg(unix_mid)
    ha_deg = (lst_deg - sun_ra_deg) % 360.0
    ha_deg[ha_deg > 180.0] -= 360.0

    chip_info = segment_capture_times_by_gap(
        unix_time_start_sorted=captures_s.unix_time_start,
        unix_time_end_sorted=captures_s.unix_time_end,
        unix_sorted=unix_mid,
        ha_deg=ha_deg,
    )
    chip_slices = chip_info.chip_slices
    corr_chips = [captures_s.corr[chip_slice] for chip_slice in chip_slices]

    dc_result = local_real_dc_correction(
        corr_chips=corr_chips,
        unix_chips=chip_info.unix_chips,
        bad_channels=BAD_CHANNELS,
        nominal_fringe_period_sec=NOMINAL_FRINGE_PERIOD_SEC,
        window_periods=LOCAL_DC_PERIODS,
    )

    n_ch = captures_s.N_CH
    df_hz = F_S_HZ / N_FFT
    f_sky_ghz = (F_RF0_HZ + np.arange(n_ch) * df_hz) / 1e9
    band_mask = (f_sky_ghz >= PLOT_BAND_GHZ[0]) & (f_sky_ghz <= PLOT_BAND_GHZ[1])
    band_mask[BAD_CHANNELS] = False
    amp_dc = np.abs(dc_result.corr_dc).astype(float)
    amp_peak_global = float(np.nanmax(amp_dc))
    band_indices = np.flatnonzero(band_mask)
    band_mean = np.nanmean(amp_dc[:, band_indices], axis=0)
    finite_band = np.isfinite(band_mean)
    if not np.any(finite_band):
        raise ValueError("No finite DC-corrected channels found inside the analysis band.")
    k_peak_global = int(band_indices[finite_band][np.nanargmax(band_mean[finite_band])])
    f_peak_ghz_global = float(f_sky_ghz[k_peak_global])

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        "Processed Sun calibration chip export",
        f"source_dir={data_dir}",
        f"n_input_files={len(paths)}",
        f"n_sorted_captures={captures_s.N_cap}",
        f"n_chips={len(chip_slices)}",
        f"split_gap_threshold_sec={chip_info.split_gap_threshold_sec:.6f}",
        f"bad_channels={BAD_CHANNELS.tolist()}",
        f"nominal_fringe_period_sec={NOMINAL_FRINGE_PERIOD_SEC:.6f}",
        f"local_dc_periods={LOCAL_DC_PERIODS:.6f}",
        f"amp_peak_global={amp_peak_global:.12g}",
        f"k_peak_global={k_peak_global}",
        f"f_peak_ghz_global={f_peak_ghz_global:.12g}",
        "",
    ]

    for raw_idx, chip_slice in enumerate(chip_slices):
        chip_idx = raw_idx + chip_offset
        idx = np.arange(chip_slice.start, chip_slice.stop, dtype=int)
        gap_before = (
            float(chip_info.gap_s[chip_slice.start - 1])
            if chip_slice.start > 0
            else float("nan")
        )
        gap_after = (
            float(chip_info.gap_s[chip_slice.stop - 1])
            if chip_slice.stop < captures_s.N_cap
            else float("nan")
        )
        out_path = output_dir / f"sun_calibration_chip_{chip_idx:02d}.npz"

        payload = {
            "chip_index": chip_idx,
            "n_chip_total": len(chip_slices),
            "n_cap_chip": idx.size,
            "n_cap_total": captures_s.N_cap,
            "chip_slice_start": chip_slice.start,
            "chip_slice_stop": chip_slice.stop,
            "capture_index_in_input": order[idx],
            "source_name": np.asarray([captures_s.paths[i].name for i in idx], dtype=str),
            "corr": captures_s.corr[chip_slice],
            "corr_std": captures_s.corr_std[chip_slice],
            "corr_dc": dc_result.corr_dc_chips[raw_idx],
            "real_dc_offset": dc_result.real_offset_chips[raw_idx],
            "unix_time_start": captures_s.unix_time_start[chip_slice],
            "unix_time_end": captures_s.unix_time_end[chip_slice],
            "unix_mid": unix_mid[chip_slice],
            "n_acc": captures_s.n_acc[chip_slice],
            "alt_deg": captures_s.alt_deg[chip_slice],
            "az_deg": captures_s.az_deg[chip_slice],
            "duration_sec": captures_s.duration_sec[chip_slice],
            "ra_deg_file": captures_s.ra_deg[chip_slice],
            "dec_deg_file": captures_s.dec_deg[chip_slice],
            "sun_ra_deg": sun_ra_deg[chip_slice],
            "sun_dec_deg": sun_dec_deg[chip_slice],
            "lst_deg": lst_deg[chip_slice],
            "ha_deg": ha_deg[chip_slice],
            "gap_before_sec": gap_before,
            "gap_after_sec": gap_after,
            "split_gap_threshold_sec": chip_info.split_gap_threshold_sec,
            "gap_break_indices_global": chip_info.gap_break_indices,
            "gap_break_values_sec": chip_info.gap_break_values_sec,
            "median_cadence_sec": dc_result.median_cadence_sec_chips[raw_idx],
            "window_caps": dc_result.window_caps_chips[raw_idx],
            "window_sec": dc_result.window_sec,
            "nominal_fringe_period_sec": NOMINAL_FRINGE_PERIOD_SEC,
            "local_dc_periods": LOCAL_DC_PERIODS,
            "bad_channels": BAD_CHANNELS,
            "f_s_hz": F_S_HZ,
            "n_fft": N_FFT,
            "f_rf0_hz": F_RF0_HZ,
            "df_hz": df_hz,
            "f_sky_ghz": f_sky_ghz,
            "plot_band_ghz": np.asarray(PLOT_BAND_GHZ, dtype=float),
            "nch_lon_deg": NCH_LON_DEG,
            "amp_peak_global": amp_peak_global,
            "k_peak_global": k_peak_global,
            "f_peak_ghz_global": f_peak_ghz_global,
        }
        np.savez_compressed(out_path, **payload)

        manifest_lines.append(
            " ".join(
                [
                    f"chip={chip_idx:02d}",
                    f"file={out_path.name}",
                    f"n_cap={idx.size}",
                    f"ha_min_deg={ha_deg[chip_slice].min():.6f}",
                    f"ha_max_deg={ha_deg[chip_slice].max():.6f}",
                    f"utc_start={captures_s.unix_time_start[chip_slice][0]:.6f}",
                    f"utc_end={captures_s.unix_time_end[chip_slice][-1]:.6f}",
                    f"median_cadence_sec={dc_result.median_cadence_sec_chips[raw_idx]:.6f}",
                    f"window_caps={dc_result.window_caps_chips[raw_idx]}",
                ]
            )
        )

    (output_dir / "README_chip_export.txt").write_text("\n".join(manifest_lines) + "\n")

    print(f"Exported {len(chip_slices)} chip files to {output_dir}")
    for line in manifest_lines[-len(chip_slices) :]:
        print(line)


if __name__ == "__main__":
    main()
