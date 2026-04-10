"""Local DC-pedestal removal for Lab 03 Sun visibilities.

Two correction strategies are provided:

* :func:`local_real_dc_correction` — fixed-width centred rolling median.
  Used by ``scripts/export_sun_calibration_chips.py``.
* :func:`adaptive_real_dc_correction` — per-capture window width that
  scales with the local fringe period.  Used by the analysis notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalDCResult:
    corr_dc_chips: list[np.ndarray]
    corr_dc: np.ndarray
    real_offset_chips: list[np.ndarray]
    window_sec: float
    median_cadence_sec_chips: list[float]
    window_caps_chips: list[int]


@dataclass(frozen=True)
class AdaptiveDCResult:
    corr_dc_chips: list[np.ndarray]
    corr_dc: np.ndarray
    real_offset_chips: list[np.ndarray]
    window_caps_chips: list[np.ndarray]
    window_caps: np.ndarray


def _centered_rolling_nanmedian(real_mat: np.ndarray, window_caps: int) -> np.ndarray:
    """Centred rolling nanmedian along axis 0 with edge clipping."""
    n_cap = real_mat.shape[0]
    if window_caps >= n_cap:
        return np.broadcast_to(np.nanmedian(real_mat, axis=0), real_mat.shape).copy()

    half = window_caps // 2
    out = np.empty_like(real_mat, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(real_mat, window_shape=window_caps, axis=0)
    out[half : n_cap - half] = np.nanmedian(windows, axis=2)
    for i in range(half):
        out[i]               = np.nanmedian(real_mat[: i + half + 1], axis=0)
        out[n_cap - 1 - i]   = np.nanmedian(real_mat[n_cap - (i + half + 1) :], axis=0)
    return out


def local_real_dc_correction(
    corr_chips: list[np.ndarray],
    unix_chips: list[np.ndarray],
    bad_channels: np.ndarray,
    nominal_fringe_period_sec: float,
    window_periods: float,
) -> LocalDCResult:
    """Subtract a per-chip rolling-median pedestal from the real part."""
    window_sec = nominal_fringe_period_sec * window_periods
    bad = np.asarray(bad_channels, dtype=int)

    corr_dc_chips: list[np.ndarray] = []
    real_offset_chips: list[np.ndarray] = []
    median_cadence_sec_chips: list[float] = []
    window_caps_chips: list[int] = []

    for raw, unix in zip(corr_chips, unix_chips):
        mat = np.asarray(raw, dtype=complex).copy()
        cadence = float(np.median(np.diff(unix)))
        win = max(int(round(window_sec / cadence)), 5)
        if win % 2 == 0:
            win += 1
        win = min(win, mat.shape[0])

        real_mat = mat.real.astype(float)
        real_mat[:, bad] = 0.0
        offset = _centered_rolling_nanmedian(real_mat, win)

        mat[:, bad] = np.nan
        offset[:, bad] = np.nan
        corr_dc = mat - offset

        median_cadence_sec_chips.append(cadence)
        window_caps_chips.append(win)
        real_offset_chips.append(offset)
        corr_dc_chips.append(corr_dc)

    return LocalDCResult(
        corr_dc_chips=corr_dc_chips,
        corr_dc=np.vstack(corr_dc_chips),
        real_offset_chips=real_offset_chips,
        window_sec=window_sec,
        median_cadence_sec_chips=median_cadence_sec_chips,
        window_caps_chips=window_caps_chips,
    )


def _adaptive_rolling_median(real_mat: np.ndarray, window_caps: np.ndarray) -> np.ndarray:
    """Per-capture rolling nanmedian with a variable window width."""
    n_cap = real_mat.shape[0]
    out = np.empty_like(real_mat, dtype=float)
    for i in range(n_cap):
        half = int(window_caps[i]) // 2
        out[i] = np.nanmedian(real_mat[max(0, i - half) : min(n_cap, i + half + 1)], axis=0)
    return out


def adaptive_real_dc_correction(
    corr_chips: list[np.ndarray],
    unix_chips: list[np.ndarray],
    ha_rad_chips: list[np.ndarray],
    bad_channels: np.ndarray,
    b_ew: float,
    freq_hz: float,
    dec_rad: float,
    n_periods: float,
    min_window_caps: int,
    max_window_caps: int,
) -> AdaptiveDCResult:
    r"""Adaptive rolling-median DC correction on the real part.

    The window width at each capture scales with the local fringe period
    :math:`P_f(h) \propto 1/|\cos h|`, keeping a constant number of fringe
    cycles inside the window regardless of hour angle.
    """
    from .geometry import fringe_period_s  # deferred so export script can import dc as a top-level module

    bad = np.asarray(bad_channels, dtype=int)

    dc_chips: list[np.ndarray] = []
    offset_chips: list[np.ndarray] = []
    wcaps_chips: list[np.ndarray] = []

    for raw, unix, ha_rad in zip(corr_chips, unix_chips, ha_rad_chips):
        mat = np.asarray(raw, dtype=complex).copy()
        n_cap = mat.shape[0]

        local_period = fringe_period_s(ha_rad, dec_rad, b_ew, 0.0, freq_hz)
        window_sec = n_periods * local_period

        dt = np.diff(unix)
        cadence = np.empty(n_cap)
        cadence[0] = dt[0]
        cadence[-1] = dt[-1]
        if n_cap > 2:
            cadence[1:-1] = 0.5 * (dt[:-1] + dt[1:])

        win = np.round(window_sec / cadence).astype(int)
        win = np.clip(win, min_window_caps, max_window_caps) | 1  # ensure odd

        real_mat = mat.real.astype(float)
        real_mat[:, bad] = 0.0
        offset = _adaptive_rolling_median(real_mat, win)

        mat[:, bad] = np.nan
        offset[:, bad] = np.nan
        corr_dc = mat - offset

        dc_chips.append(corr_dc)
        offset_chips.append(offset)
        wcaps_chips.append(win)

    return AdaptiveDCResult(
        corr_dc_chips=dc_chips,
        corr_dc=np.vstack(dc_chips),
        real_offset_chips=offset_chips,
        window_caps_chips=wcaps_chips,
        window_caps=np.concatenate(wcaps_chips),
    )
