"""Chip segmentation helpers for the Lab 03 Sun notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ChipSegmentation:
    chip_slices: list[slice]
    unix_chips: list[np.ndarray]
    ha_chips: list[np.ndarray]
    N_caps: list[int]
    cap_chip: np.ndarray
    gap_s: np.ndarray
    split_gap_threshold_sec: float
    gap_break_indices: np.ndarray
    gap_break_values_sec: np.ndarray


def _single_chip_segmentation(
    unix_sorted: np.ndarray,
    ha_deg: np.ndarray,
    *,
    min_split_gap_sec: float,
) -> ChipSegmentation:
    chip_slices = [slice(0, 1)]
    return ChipSegmentation(
        chip_slices=chip_slices,
        unix_chips=[unix_sorted.copy()],
        ha_chips=[ha_deg.copy()],
        N_caps=[1],
        cap_chip=np.zeros(1, dtype=int),
        gap_s=np.array([], dtype=float),
        split_gap_threshold_sec=float(min_split_gap_sec),
        gap_break_indices=np.array([], dtype=int),
        gap_break_values_sec=np.array([], dtype=float),
    )


def _split_gap_threshold(
    gap_s: np.ndarray,
    *,
    gap_multiplier: float,
    min_split_gap_sec: float,
) -> float:
    positive_gap_s = gap_s[np.isfinite(gap_s) & (gap_s > 0)]
    if positive_gap_s.size:
        median_gap_sec = float(np.median(positive_gap_s))
        lower_half_gap_s = positive_gap_s[positive_gap_s <= median_gap_sec]
        nominal_gap_sec = float(np.median(lower_half_gap_s)) if lower_half_gap_s.size else median_gap_sec
    else:
        nominal_gap_sec = 0.0
    return float(max(gap_multiplier * nominal_gap_sec, min_split_gap_sec))


def segment_capture_times_by_gap(
    unix_time_start_sorted: np.ndarray,
    unix_time_end_sorted: np.ndarray,
    unix_sorted: np.ndarray,
    ha_deg: np.ndarray,
    *,
    gap_multiplier: float = 10.0,
    min_split_gap_sec: float = 5.0,
) -> ChipSegmentation:
    """Split an array-backed capture sequence into chips using inferred cadence."""
    unix_time_start_sorted = np.asarray(unix_time_start_sorted, dtype=float)
    unix_time_end_sorted = np.asarray(unix_time_end_sorted, dtype=float)
    unix_sorted = np.asarray(unix_sorted, dtype=float)
    ha_deg = np.asarray(ha_deg, dtype=float)

    n_cap = int(unix_sorted.size)
    if (
        unix_time_start_sorted.size != n_cap
        or unix_time_end_sorted.size != n_cap
        or ha_deg.size != n_cap
    ):
        raise ValueError(
            "unix_time_start_sorted, unix_time_end_sorted, unix_sorted, and ha_deg must have the same length."
        )
    if n_cap == 0:
        raise ValueError("Cannot segment an empty capture sequence.")

    if n_cap == 1:
        return _single_chip_segmentation(unix_sorted, ha_deg, min_split_gap_sec=min_split_gap_sec)

    gap_s = unix_time_start_sorted[1:] - unix_time_end_sorted[:-1]
    split_gap_threshold_sec = _split_gap_threshold(
        gap_s,
        gap_multiplier=gap_multiplier,
        min_split_gap_sec=min_split_gap_sec,
    )

    gap_break_indices = np.flatnonzero(gap_s > split_gap_threshold_sec).astype(int) + 1
    gap_break_values_sec = gap_s[gap_break_indices - 1] if gap_break_indices.size else np.array([], dtype=float)

    starts = np.concatenate(([0], gap_break_indices))
    stops = np.concatenate((gap_break_indices, [n_cap]))
    chip_slices = [slice(int(start), int(stop)) for start, stop in zip(starts, stops) if stop > start]
    unix_chips = [unix_sorted[s].copy() for s in chip_slices]
    ha_chips = [ha_deg[s].copy() for s in chip_slices]
    n_caps = [int(s.stop - s.start) for s in chip_slices]

    cap_chip = np.empty(n_cap, dtype=int)
    for chip_idx, chip_slice in enumerate(chip_slices):
        cap_chip[chip_slice] = chip_idx

    return ChipSegmentation(
        chip_slices=chip_slices,
        unix_chips=unix_chips,
        ha_chips=ha_chips,
        N_caps=n_caps,
        cap_chip=cap_chip,
        gap_s=gap_s,
        split_gap_threshold_sec=split_gap_threshold_sec,
        gap_break_indices=gap_break_indices,
        gap_break_values_sec=np.asarray(gap_break_values_sec, dtype=float),
    )
