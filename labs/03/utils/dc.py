"""Local DC correction helpers for the Lab 03 Sun notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LocalDCResult:
    corr_dc_chips: list[np.ndarray]
    corr_dc: np.ndarray
    real_offset_chips: list[np.ndarray]
    window_sec: float
    median_cadence_sec_chips: list[float]
    window_caps_chips: list[int]


def _odd_window_caps(target_window_caps: int, min_window_caps: int) -> int:
    window_caps = max(int(target_window_caps), int(min_window_caps))
    if window_caps % 2 == 0:
        window_caps += 1
    return window_caps


def _centered_rolling_nanmedian(real_mat: np.ndarray, window_caps: int) -> np.ndarray:
    n_cap, _n_ch = real_mat.shape
    if n_cap == 0:
        return np.empty_like(real_mat)

    if window_caps >= n_cap:
        full_chip_median = np.nanmedian(real_mat, axis=0)
        return np.broadcast_to(full_chip_median, real_mat.shape).copy()

    half = window_caps // 2
    rolling_median = np.empty_like(real_mat, dtype=float)

    windows = np.lib.stride_tricks.sliding_window_view(real_mat, window_shape=window_caps, axis=0)
    rolling_median[half : n_cap - half] = np.nanmedian(windows, axis=2)

    for idx in range(half):
        rolling_median[idx] = np.nanmedian(real_mat[: idx + half + 1], axis=0)
        rolling_median[n_cap - idx - 1] = np.nanmedian(real_mat[n_cap - (idx + half + 1) :], axis=0)

    return rolling_median


def local_real_dc_correction(
    *,
    files_chips: list[list[Any]],
    unix_chips: list[np.ndarray],
    bad_channels: list[int] | tuple[int, ...],
    nominal_fringe_period_sec: float = 40.0,
    window_periods: float = 2.5,
    min_window_caps: int = 5,
) -> LocalDCResult:
    """Subtract a chip-local rolling real pedestal from each channel."""
    if len(files_chips) != len(unix_chips):
        raise ValueError("files_chips and unix_chips must have the same length.")

    bad_channels_arr = np.asarray(bad_channels, dtype=int)
    window_sec = float(nominal_fringe_period_sec * window_periods)
    corr_dc_chips: list[np.ndarray] = []
    real_offset_chips: list[np.ndarray] = []
    median_cadence_sec_chips: list[float] = []
    window_caps_chips: list[int] = []

    for fchip, unix_chip in zip(files_chips, unix_chips):
        mat = np.array([np.asarray(f["corr"], dtype=complex) for f in fchip], dtype=complex)

        n_cap = mat.shape[0]
        if n_cap == 0:
            raise ValueError("Encountered an empty chip while computing local DC correction.")

        if n_cap > 1:
            cadence_sec = float(np.median(np.diff(np.asarray(unix_chip, dtype=float))))
        else:
            cadence_sec = float("nan")

        median_cadence_sec_chips.append(cadence_sec)

        if np.isfinite(cadence_sec) and cadence_sec > 0:
            target_window_caps = int(round(window_sec / cadence_sec))
        else:
            target_window_caps = min_window_caps

        effective_window_caps = _odd_window_caps(target_window_caps, min_window_caps)
        if n_cap <= effective_window_caps:
            effective_window_caps = n_cap

        window_caps_chips.append(int(effective_window_caps))

        real_mat = mat.real.astype(float)
        if bad_channels_arr.size:
            real_mat = real_mat.copy()
            real_mat[:, bad_channels_arr] = 0.0

        real_offset = _centered_rolling_nanmedian(real_mat, effective_window_caps)
        if bad_channels_arr.size:
            mat[:, bad_channels_arr] = np.nan
            real_offset[:, bad_channels_arr] = np.nan

        corr_dc = mat - real_offset
        if bad_channels_arr.size:
            corr_dc[:, bad_channels_arr] = np.nan

        real_offset_chips.append(real_offset)
        corr_dc_chips.append(corr_dc)

    return LocalDCResult(
        corr_dc_chips=corr_dc_chips,
        corr_dc=np.vstack(corr_dc_chips),
        real_offset_chips=real_offset_chips,
        window_sec=window_sec,
        median_cadence_sec_chips=median_cadence_sec_chips,
        window_caps_chips=window_caps_chips,
    )
