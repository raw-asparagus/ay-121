"""Chip segmentation helpers for the Lab 03 Sun notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SortedCaptures:
    files_sorted: list[Any]
    paths_sorted: list[Path]
    unix_sorted: np.ndarray
    order: np.ndarray


@dataclass(frozen=True)
class ChipSegmentation:
    chip_slices: list[slice]
    files_chips: list[list[Any]]
    unix_chips: list[np.ndarray]
    ha_chips: list[np.ndarray]
    N_caps: list[int]
    cap_chip: np.ndarray
    gap_s: np.ndarray
    nominal_gap_sec: float
    split_gap_threshold_sec: float
    gap_break_indices: np.ndarray
    gap_break_values_sec: np.ndarray


def mid_unix(f: Any) -> float:
    """Midpoint unix timestamp; falls back to legacy 'unix_time' key."""
    if "unix_time_start" in f:
        return (float(f["unix_time_start"]) + float(f["unix_time_end"])) / 2
    return float(f["unix_time"])


def sort_captures_by_mid_unix(files: list[Any], paths: list[Path]) -> SortedCaptures:
    unix_mid = np.array([mid_unix(f) for f in files], dtype=float)
    order = np.argsort(unix_mid)
    return SortedCaptures(
        files_sorted=[files[idx] for idx in order],
        paths_sorted=[paths[idx] for idx in order],
        unix_sorted=unix_mid[order],
        order=order,
    )


def segment_captures_by_gap(
    files_sorted: list[Any],
    unix_sorted: np.ndarray,
    ha_deg: np.ndarray,
    *,
    gap_multiplier: float = 10.0,
    min_split_gap_sec: float = 5.0,
) -> ChipSegmentation:
    """Split a sorted capture sequence into chips using inferred cadence."""
    if len(files_sorted) != len(unix_sorted) or len(files_sorted) != len(ha_deg):
        raise ValueError("files_sorted, unix_sorted, and ha_deg must have the same length.")

    n_cap = len(files_sorted)
    if n_cap == 0:
        raise ValueError("Cannot segment an empty capture sequence.")

    if n_cap == 1:
        chip_slices = [slice(0, 1)]
        cap_chip = np.zeros(1, dtype=int)
        return ChipSegmentation(
            chip_slices=chip_slices,
            files_chips=[[files_sorted[0]]],
            unix_chips=[np.asarray(unix_sorted, dtype=float)],
            ha_chips=[np.asarray(ha_deg, dtype=float)],
            N_caps=[1],
            cap_chip=cap_chip,
            gap_s=np.array([], dtype=float),
            nominal_gap_sec=0.0,
            split_gap_threshold_sec=float(min_split_gap_sec),
            gap_break_indices=np.array([], dtype=int),
            gap_break_values_sec=np.array([], dtype=float),
        )

    t_end = np.array([float(f["unix_time_end"]) for f in files_sorted], dtype=float)
    t_start = np.array([float(f["unix_time_start"]) for f in files_sorted], dtype=float)
    gap_s = t_start[1:] - t_end[:-1]

    positive_gap_s = gap_s[np.isfinite(gap_s) & (gap_s > 0)]
    if positive_gap_s.size:
        median_gap_sec = float(np.median(positive_gap_s))
        lower_half_gap_s = positive_gap_s[positive_gap_s <= median_gap_sec]
        nominal_gap_sec = float(np.median(lower_half_gap_s)) if lower_half_gap_s.size else median_gap_sec
    else:
        nominal_gap_sec = 0.0
    split_gap_threshold_sec = float(max(gap_multiplier * nominal_gap_sec, min_split_gap_sec))

    gap_break_indices = np.flatnonzero(gap_s > split_gap_threshold_sec).astype(int) + 1
    gap_break_values_sec = gap_s[gap_break_indices - 1] if gap_break_indices.size else np.array([], dtype=float)

    starts = np.concatenate(([0], gap_break_indices))
    stops = np.concatenate((gap_break_indices, [n_cap]))
    chip_slices = [slice(int(start), int(stop)) for start, stop in zip(starts, stops) if stop > start]

    files_chips = [[files_sorted[idx] for idx in range(s.start, s.stop)] for s in chip_slices]
    unix_chips = [np.asarray(unix_sorted[s], dtype=float) for s in chip_slices]
    ha_chips = [np.asarray(ha_deg[s], dtype=float) for s in chip_slices]
    N_caps = [int(s.stop - s.start) for s in chip_slices]

    cap_chip = np.empty(n_cap, dtype=int)
    for chip_idx, chip_slice in enumerate(chip_slices):
        cap_chip[chip_slice] = chip_idx

    return ChipSegmentation(
        chip_slices=chip_slices,
        files_chips=files_chips,
        unix_chips=unix_chips,
        ha_chips=ha_chips,
        N_caps=N_caps,
        cap_chip=cap_chip,
        gap_s=gap_s,
        nominal_gap_sec=nominal_gap_sec,
        split_gap_threshold_sec=split_gap_threshold_sec,
        gap_break_indices=gap_break_indices,
        gap_break_values_sec=np.asarray(gap_break_values_sec, dtype=float),
    )
