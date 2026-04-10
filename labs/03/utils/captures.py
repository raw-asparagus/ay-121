"""Array-backed capture loading for the Lab 03 Sun notebooks.

The capture NPZ files are produced by ``labs/03/scripts/multi_calibration.py``;
every key referenced here is guaranteed to be present.
"""

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

    def take(self, order: np.ndarray) -> "CaptureSeries":
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


def load_capture_series(paths: Sequence[Path]) -> CaptureSeries:
    path_list = [Path(p) for p in paths]
    n_cap = len(path_list)

    with np.load(path_list[0]) as first:
        n_ch = int(np.asarray(first["corr"]).shape[0])

    corr            = np.empty((n_cap, n_ch), dtype=complex)
    corr_std        = np.empty((n_cap, n_ch), dtype=float)
    unix_time_start = np.empty(n_cap, dtype=float)
    unix_time_end   = np.empty(n_cap, dtype=float)
    n_acc           = np.empty(n_cap, dtype=int)
    alt_deg         = np.empty(n_cap, dtype=float)
    az_deg          = np.empty(n_cap, dtype=float)
    duration_sec    = np.empty(n_cap, dtype=float)
    ra_deg          = np.empty(n_cap, dtype=float)
    dec_deg         = np.empty(n_cap, dtype=float)

    for i, path in enumerate(path_list):
        with np.load(path) as npz:
            corr[i]            = npz["corr"]
            corr_std[i]        = npz["corr_std"]
            unix_time_start[i] = npz["unix_time_start"].item()
            unix_time_end[i]   = npz["unix_time_end"].item()
            n_acc[i]           = npz["n_acc"].item()
            alt_deg[i]         = npz["alt_deg"].item()
            az_deg[i]          = npz["az_deg"].item()
            duration_sec[i]    = npz["duration_sec"].item()
            ra_deg[i]          = npz["ra_deg"].item()
            dec_deg[i]         = npz["dec_deg"].item()

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
