"""Array-backed capture loading helpers for the Lab 03 Sun notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CaptureSeries:
    paths: tuple[Path, ...]
    corr: np.ndarray
    corr_std: np.ndarray
    unix_time_start: np.ndarray
    unix_time_end: np.ndarray
    n_acc: np.ndarray
    alt_deg: np.ndarray
    az_deg: np.ndarray
    duration_sec: np.ndarray
    ra_deg: np.ndarray
    dec_deg: np.ndarray

    @property
    def unix_mid(self) -> np.ndarray:
        return 0.5 * (self.unix_time_start + self.unix_time_end)

    @property
    def N_cap(self) -> int:
        return int(self.corr.shape[0])

    @property
    def N_CH(self) -> int:
        return int(self.corr.shape[1])

    def take(self, order: np.ndarray | Sequence[int] | slice) -> "CaptureSeries":
        if isinstance(order, slice):
            idx = np.arange(self.N_cap)[order]
        else:
            idx = np.asarray(order, dtype=int)

        return CaptureSeries(
            paths=tuple(self.paths[int(i)] for i in idx.tolist()),
            corr=self.corr[idx],
            corr_std=self.corr_std[idx],
            unix_time_start=self.unix_time_start[idx],
            unix_time_end=self.unix_time_end[idx],
            n_acc=self.n_acc[idx],
            alt_deg=self.alt_deg[idx],
            az_deg=self.az_deg[idx],
            duration_sec=self.duration_sec[idx],
            ra_deg=self.ra_deg[idx],
            dec_deg=self.dec_deg[idx],
        )

    def sorted_by_unix_mid(self) -> "CaptureSeries":
        return self.take(np.argsort(self.unix_mid))


def _scalar_or_nan(npz: np.lib.npyio.NpzFile, key: str, dtype) -> float | int:
    if key not in npz:
        return dtype(np.nan)
    return dtype(np.asarray(npz[key]).item())


def load_capture_series(paths: Sequence[Path]) -> CaptureSeries:
    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("paths must contain at least one capture file.")

    with np.load(path_list[0]) as first:
        first_corr = np.asarray(first["corr"], dtype=complex)
        n_ch = int(first_corr.shape[0])

    n_cap = len(path_list)
    corr = np.empty((n_cap, n_ch), dtype=complex)
    corr_std = np.empty((n_cap, n_ch), dtype=float)
    unix_time_start = np.empty(n_cap, dtype=float)
    unix_time_end = np.empty(n_cap, dtype=float)
    n_acc = np.empty(n_cap, dtype=int)
    alt_deg = np.empty(n_cap, dtype=float)
    az_deg = np.empty(n_cap, dtype=float)
    duration_sec = np.empty(n_cap, dtype=float)
    ra_deg = np.empty(n_cap, dtype=float)
    dec_deg = np.empty(n_cap, dtype=float)

    for idx, path in enumerate(path_list):
        with np.load(path) as npz:
            corr[idx] = np.asarray(npz["corr"], dtype=complex)
            corr_std[idx] = np.asarray(npz["corr_std"], dtype=float)
            unix_time_start[idx] = float(np.asarray(npz["unix_time_start"]).item())
            unix_time_end[idx] = float(np.asarray(npz["unix_time_end"]).item())
            n_acc[idx] = int(np.asarray(npz["n_acc"]).item())
            alt_deg[idx] = float(_scalar_or_nan(npz, "alt_deg", float))
            az_deg[idx] = float(_scalar_or_nan(npz, "az_deg", float))
            duration_sec[idx] = float(_scalar_or_nan(npz, "duration_sec", float))
            ra_deg[idx] = float(_scalar_or_nan(npz, "ra_deg", float))
            dec_deg[idx] = float(_scalar_or_nan(npz, "dec_deg", float))

    return CaptureSeries(
        paths=tuple(path_list),
        corr=corr,
        corr_std=corr_std,
        unix_time_start=unix_time_start,
        unix_time_end=unix_time_end,
        n_acc=n_acc,
        alt_deg=alt_deg,
        az_deg=az_deg,
        duration_sec=duration_sec,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
