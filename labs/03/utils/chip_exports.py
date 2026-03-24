"""Helpers for loading processed Lab 03 Sun chip exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .captures import CaptureSeries


@dataclass(frozen=True)
class ProcessedSunChipSeries:
    chip_paths: tuple[Path, ...]
    captures: CaptureSeries
    corr_chips: list[np.ndarray]
    corr_std_chips: list[np.ndarray]
    corr_dc_chips: list[np.ndarray]
    real_dc_offset_chips: list[np.ndarray]
    unix_chips: list[np.ndarray]
    ha_chips: list[np.ndarray]
    sun_dec_chips: list[np.ndarray]
    source_name_chips: list[np.ndarray]
    chip_slices: list[slice]
    n_caps: list[int]
    cap_chip: np.ndarray
    gap_s: np.ndarray
    split_gap_threshold_sec: float
    gap_break_indices: np.ndarray
    gap_break_values_sec: np.ndarray
    lst_deg: np.ndarray
    sun_ra_deg: np.ndarray
    sun_dec_deg: np.ndarray
    ha_deg: np.ndarray
    median_cadence_sec_chips: list[float]
    window_caps_chips: list[int]
    window_sec: float
    nominal_fringe_period_sec: float
    local_dc_periods: float
    bad_channels: np.ndarray
    f_s_hz: float
    n_fft: int
    f_rf0_hz: float
    df_hz: float
    f_sky_ghz: np.ndarray
    plot_band_ghz: tuple[float, float]
    nch_lon_deg: float
    amp_peak_global: float
    k_peak_global: int
    f_peak_ghz_global: float

    @property
    def N_cap(self) -> int:
        return self.captures.N_cap

    @property
    def N_CH(self) -> int:
        return self.captures.N_CH

    @property
    def corr_dc(self) -> np.ndarray:
        return np.vstack(self.corr_dc_chips)


def _require_shared_scalar(
    chip_npz: np.lib.npyio.NpzFile,
    shared_scalar: dict[str, object],
    key: str,
) -> None:
    value = np.asarray(chip_npz[key]).item()
    if key in shared_scalar:
        if shared_scalar[key] != value:
            raise ValueError(f"Chip export mismatch for scalar field {key!r}.")
    else:
        shared_scalar[key] = value


def _require_shared_array(
    chip_npz: np.lib.npyio.NpzFile,
    shared_array: dict[str, np.ndarray],
    key: str,
) -> None:
    value = np.asarray(chip_npz[key])
    if key in shared_array:
        if not np.array_equal(shared_array[key], value, equal_nan=True):
            raise ValueError(f"Chip export mismatch for array field {key!r}.")
    else:
        shared_array[key] = value.copy()


def load_processed_sun_chip_series(paths: Sequence[Path]) -> ProcessedSunChipSeries:
    chip_paths = tuple(Path(path) for path in paths)
    if not chip_paths:
        raise ValueError("paths must contain at least one chip NPZ file.")

    shared_scalar: dict[str, object] = {}
    shared_array: dict[str, np.ndarray] = {}

    source_name_chips: list[np.ndarray] = []
    corr_chips: list[np.ndarray] = []
    corr_std_chips: list[np.ndarray] = []
    corr_dc_chips: list[np.ndarray] = []
    real_dc_offset_chips: list[np.ndarray] = []
    unix_time_start_chips: list[np.ndarray] = []
    unix_time_end_chips: list[np.ndarray] = []
    unix_mid_chips: list[np.ndarray] = []
    n_acc_chips: list[np.ndarray] = []
    alt_deg_chips: list[np.ndarray] = []
    az_deg_chips: list[np.ndarray] = []
    duration_sec_chips: list[np.ndarray] = []
    ra_deg_file_chips: list[np.ndarray] = []
    dec_deg_file_chips: list[np.ndarray] = []
    sun_ra_deg_chips: list[np.ndarray] = []
    sun_dec_chips: list[np.ndarray] = []
    lst_deg_chips: list[np.ndarray] = []
    ha_chips: list[np.ndarray] = []
    median_cadence_sec_chips: list[float] = []
    window_caps_chips: list[int] = []
    n_caps: list[int] = []

    for chip_path in chip_paths:
        with np.load(chip_path, allow_pickle=False) as chip_npz:
            for key in (
                "split_gap_threshold_sec",
                "window_sec",
                "nominal_fringe_period_sec",
                "local_dc_periods",
                "f_s_hz",
                "n_fft",
                "f_rf0_hz",
                "df_hz",
                "nch_lon_deg",
            ):
                _require_shared_scalar(chip_npz, shared_scalar, key)

            for key in ("bad_channels", "f_sky_ghz", "plot_band_ghz"):
                _require_shared_array(chip_npz, shared_array, key)

            source_name_chips.append(np.asarray(chip_npz["source_name"], dtype=str))
            corr_chips.append(np.asarray(chip_npz["corr"], dtype=complex))
            corr_std_chips.append(np.asarray(chip_npz["corr_std"], dtype=float))
            corr_dc_chips.append(np.asarray(chip_npz["corr_dc"], dtype=complex))
            real_dc_offset_chips.append(np.asarray(chip_npz["real_dc_offset"], dtype=float))
            unix_time_start_chips.append(np.asarray(chip_npz["unix_time_start"], dtype=float))
            unix_time_end_chips.append(np.asarray(chip_npz["unix_time_end"], dtype=float))
            unix_mid_chips.append(np.asarray(chip_npz["unix_mid"], dtype=float))
            n_acc_chips.append(np.asarray(chip_npz["n_acc"], dtype=int))
            alt_deg_chips.append(np.asarray(chip_npz["alt_deg"], dtype=float))
            az_deg_chips.append(np.asarray(chip_npz["az_deg"], dtype=float))
            duration_sec_chips.append(np.asarray(chip_npz["duration_sec"], dtype=float))
            ra_deg_file_chips.append(np.asarray(chip_npz["ra_deg_file"], dtype=float))
            dec_deg_file_chips.append(np.asarray(chip_npz["dec_deg_file"], dtype=float))
            sun_ra_deg_chips.append(np.asarray(chip_npz["sun_ra_deg"], dtype=float))
            sun_dec_chips.append(np.asarray(chip_npz["sun_dec_deg"], dtype=float))
            lst_deg_chips.append(np.asarray(chip_npz["lst_deg"], dtype=float))
            ha_chips.append(np.asarray(chip_npz["ha_deg"], dtype=float))
            median_cadence_sec_chips.append(float(np.asarray(chip_npz["median_cadence_sec"]).item()))
            window_caps_chips.append(int(np.asarray(chip_npz["window_caps"]).item()))
            n_caps.append(int(np.asarray(chip_npz["n_cap_chip"]).item()))

    starts = np.cumsum([0] + n_caps[:-1], dtype=int)
    stops = np.cumsum(n_caps, dtype=int)
    chip_slices = [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]
    cap_chip = np.concatenate([
        np.full(n_cap_chip, chip_idx, dtype=int)
        for chip_idx, n_cap_chip in enumerate(n_caps)
    ])

    source_paths = tuple(
        Path(name)
        for chip_source_names in source_name_chips
        for name in chip_source_names.tolist()
    )
    captures = CaptureSeries(
        paths=source_paths,
        corr=np.vstack(corr_chips),
        corr_std=np.vstack(corr_std_chips),
        unix_time_start=np.concatenate(unix_time_start_chips),
        unix_time_end=np.concatenate(unix_time_end_chips),
        n_acc=np.concatenate(n_acc_chips),
        alt_deg=np.concatenate(alt_deg_chips),
        az_deg=np.concatenate(az_deg_chips),
        duration_sec=np.concatenate(duration_sec_chips),
        ra_deg=np.concatenate(ra_deg_file_chips),
        dec_deg=np.concatenate(dec_deg_file_chips),
    )

    gap_s = captures.unix_time_start[1:] - captures.unix_time_end[:-1]
    gap_break_indices = np.asarray(starts[1:], dtype=int)
    gap_break_values_sec = gap_s[gap_break_indices - 1] if gap_break_indices.size else np.array([], dtype=float)

    # Recompute peak channel from combined DC-corrected data
    f_sky_ghz = np.asarray(shared_array["f_sky_ghz"], dtype=float)
    bad_channels = np.asarray(shared_array["bad_channels"], dtype=int)
    plot_band_ghz = np.asarray(shared_array["plot_band_ghz"], dtype=float)
    band_mask = (f_sky_ghz >= plot_band_ghz[0]) & (f_sky_ghz <= plot_band_ghz[1])
    band_mask[bad_channels] = False

    corr_dc_all = np.vstack(corr_dc_chips)
    amp_dc = np.abs(corr_dc_all).astype(float)
    amp_peak_global = float(np.nanmax(amp_dc))
    band_indices = np.flatnonzero(band_mask)
    band_mean = np.nanmean(amp_dc[:, band_indices], axis=0)
    finite_band = np.isfinite(band_mean)
    k_peak_global = int(band_indices[finite_band][np.nanargmax(band_mean[finite_band])])
    f_peak_ghz_global = float(f_sky_ghz[k_peak_global])

    return ProcessedSunChipSeries(
        chip_paths=chip_paths,
        captures=captures,
        corr_chips=corr_chips,
        corr_std_chips=corr_std_chips,
        corr_dc_chips=corr_dc_chips,
        real_dc_offset_chips=real_dc_offset_chips,
        unix_chips=unix_mid_chips,
        ha_chips=ha_chips,
        sun_dec_chips=sun_dec_chips,
        source_name_chips=source_name_chips,
        chip_slices=chip_slices,
        n_caps=n_caps,
        cap_chip=cap_chip,
        gap_s=gap_s,
        split_gap_threshold_sec=float(shared_scalar["split_gap_threshold_sec"]),
        gap_break_indices=gap_break_indices,
        gap_break_values_sec=np.asarray(gap_break_values_sec, dtype=float),
        lst_deg=np.concatenate(lst_deg_chips),
        sun_ra_deg=np.concatenate(sun_ra_deg_chips),
        sun_dec_deg=np.concatenate(sun_dec_chips),
        ha_deg=np.concatenate(ha_chips),
        median_cadence_sec_chips=median_cadence_sec_chips,
        window_caps_chips=window_caps_chips,
        window_sec=float(shared_scalar["window_sec"]),
        nominal_fringe_period_sec=float(shared_scalar["nominal_fringe_period_sec"]),
        local_dc_periods=float(shared_scalar["local_dc_periods"]),
        bad_channels=np.asarray(shared_array["bad_channels"], dtype=int),
        f_s_hz=float(shared_scalar["f_s_hz"]),
        n_fft=int(shared_scalar["n_fft"]),
        f_rf0_hz=float(shared_scalar["f_rf0_hz"]),
        df_hz=float(shared_scalar["df_hz"]),
        f_sky_ghz=np.asarray(shared_array["f_sky_ghz"], dtype=float),
        plot_band_ghz=tuple(np.asarray(shared_array["plot_band_ghz"], dtype=float).tolist()),
        nch_lon_deg=float(shared_scalar["nch_lon_deg"]),
        amp_peak_global=amp_peak_global,
        k_peak_global=k_peak_global,
        f_peak_ghz_global=f_peak_ghz_global,
    )
