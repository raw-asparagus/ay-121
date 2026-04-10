"""Chip segmentation for the Lab 03 Sun calibration export."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChipSegmentation:
    chip_slices: list[slice]
    unix_chips: list[np.ndarray]
    gap_s: np.ndarray
    split_gap_threshold_sec: float
    gap_break_indices: np.ndarray
    gap_break_values_sec: np.ndarray


def segment_capture_times_by_gap(
    unix_time_start_sorted: np.ndarray,
    unix_time_end_sorted: np.ndarray,
    unix_sorted: np.ndarray,
    gap_multiplier: float,
    min_split_gap_sec: float,
) -> ChipSegmentation:
    """Split a sorted capture sequence into chips using inferred cadence.

    A chip break is inserted whenever an inter-capture gap exceeds
    ``max(gap_multiplier * nominal_cadence, min_split_gap_sec)``.
    """
    n_cap = unix_sorted.size
    gap_s = unix_time_start_sorted[1:] - unix_time_end_sorted[:-1]

    positive_gap_s  = gap_s[gap_s > 0]
    median_gap_sec  = float(np.median(positive_gap_s))
    lower_half      = positive_gap_s[positive_gap_s <= median_gap_sec]
    nominal_gap_sec = float(np.median(lower_half)) if lower_half.size else median_gap_sec
    split_gap_threshold_sec = max(gap_multiplier * nominal_gap_sec, min_split_gap_sec)

    gap_break_indices    = np.flatnonzero(gap_s > split_gap_threshold_sec).astype(int) + 1
    gap_break_values_sec = gap_s[gap_break_indices - 1]

    starts = np.concatenate(([0], gap_break_indices))
    stops  = np.concatenate((gap_break_indices, [n_cap]))
    chip_slices = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
    unix_chips  = [unix_sorted[s] for s in chip_slices]

    return ChipSegmentation(
        chip_slices=chip_slices,
        unix_chips=unix_chips,
        gap_s=gap_s,
        split_gap_threshold_sec=split_gap_threshold_sec,
        gap_break_indices=gap_break_indices,
        gap_break_values_sec=gap_break_values_sec,
    )
