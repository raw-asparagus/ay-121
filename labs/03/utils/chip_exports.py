"""Loader for processed Sun-calibration chip exports.

The chip NPZ files are written by ``scripts/export_sun_calibration_chips.py``;
all required keys and shared scalars are guaranteed to be present and
consistent across files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .captures import CaptureSeries


@dataclass(frozen=True)
class ProcessedSunChipSeries:
    captures: CaptureSeries
    chip_slices: list[slice]
    ha_deg: np.ndarray
    sun_ra_deg: np.ndarray
    sun_dec_deg: np.ndarray
    f_sky_ghz: np.ndarray
    df_hz: float


def load_processed_sun_chip_series(paths: Sequence[Path]) -> ProcessedSunChipSeries:
    chip_paths = [Path(p) for p in paths]

    corr_chips: list[np.ndarray]            = []
    corr_std_chips: list[np.ndarray]        = []
    unix_time_start_chips: list[np.ndarray] = []
    unix_time_end_chips: list[np.ndarray]   = []
    n_acc_chips: list[np.ndarray]           = []
    alt_deg_chips: list[np.ndarray]         = []
    az_deg_chips: list[np.ndarray]          = []
    duration_sec_chips: list[np.ndarray]    = []
    ra_deg_chips: list[np.ndarray]          = []
    dec_deg_chips: list[np.ndarray]         = []
    sun_ra_chips: list[np.ndarray]          = []
    sun_dec_chips: list[np.ndarray]         = []
    ha_chips: list[np.ndarray]              = []
    source_name_chips: list[np.ndarray]     = []
    n_caps: list[int] = []

    f_sky_ghz: np.ndarray | None = None
    df_hz: float | None = None

    for chip_path in chip_paths:
        with np.load(chip_path, allow_pickle=False) as npz:
            corr_chips.append(np.asarray(npz["corr"], dtype=complex))
            corr_std_chips.append(np.asarray(npz["corr_std"], dtype=float))
            unix_time_start_chips.append(np.asarray(npz["unix_time_start"], dtype=float))
            unix_time_end_chips.append(np.asarray(npz["unix_time_end"], dtype=float))
            n_acc_chips.append(np.asarray(npz["n_acc"], dtype=int))
            alt_deg_chips.append(np.asarray(npz["alt_deg"], dtype=float))
            az_deg_chips.append(np.asarray(npz["az_deg"], dtype=float))
            duration_sec_chips.append(np.asarray(npz["duration_sec"], dtype=float))
            ra_deg_chips.append(np.asarray(npz["ra_deg_file"], dtype=float))
            dec_deg_chips.append(np.asarray(npz["dec_deg_file"], dtype=float))
            sun_ra_chips.append(np.asarray(npz["sun_ra_deg"], dtype=float))
            sun_dec_chips.append(np.asarray(npz["sun_dec_deg"], dtype=float))
            ha_chips.append(np.asarray(npz["ha_deg"], dtype=float))
            source_name_chips.append(np.asarray(npz["source_name"], dtype=str))
            n_caps.append(int(npz["n_cap_chip"].item()))
            if f_sky_ghz is None:
                f_sky_ghz = np.asarray(npz["f_sky_ghz"], dtype=float)
                df_hz     = float(npz["df_hz"].item())

    starts = np.cumsum([0] + n_caps[:-1], dtype=int)
    stops  = np.cumsum(n_caps, dtype=int)
    chip_slices = [slice(int(a), int(b)) for a, b in zip(starts, stops)]

    source_paths = tuple(
        Path(name)
        for chip_names in source_name_chips
        for name in chip_names.tolist()
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
        ra_deg=np.concatenate(ra_deg_chips),
        dec_deg=np.concatenate(dec_deg_chips),
    )

    return ProcessedSunChipSeries(
        captures=captures,
        chip_slices=chip_slices,
        ha_deg=np.concatenate(ha_chips),
        sun_ra_deg=np.concatenate(sun_ra_chips),
        sun_dec_deg=np.concatenate(sun_dec_chips),
        f_sky_ghz=f_sky_ghz,
        df_hz=df_hz,
    )
