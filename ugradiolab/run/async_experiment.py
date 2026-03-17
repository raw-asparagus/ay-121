from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..drivers.interferometer import (
    compute_moon_pointing,
    compute_radec_pointing,
    compute_sun_pointing,
)
from .interf_experiment import InterfExperiment


@dataclass
class AsyncInterfExperiment(InterfExperiment):
    """InterfExperiment that saves asynchronously via a background thread pool.

    run() blocks only on pointing + SNAP collection (_collect()), then hands the
    data dict to a background thread for np.savez() and returns the path
    immediately.  This removes the file-write latency from the inter-capture gap,
    allowing the next point+collect cycle to begin right away.

    Usage
    -----
    Subclass exactly like InterfExperiment (see AsyncSunExperiment etc.).
    Before the script exits, call::

        AsyncInterfExperiment.flush()

    to drain the save queue and surface any I/O errors.
    """

    # Class-level thread pool and pending-future list shared across all instances.
    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=2)
    _futures:  ClassVar[list[Future]]       = []

    def run(self) -> str:
        """Point, collect, then save asynchronously.

        Returns the destination path immediately after handing data to the
        background thread — the file may not yet exist on disk.

        Returns
        -------
        str
            Path to the .npz file (being written in background).
        """
        path, data = self._collect()
        future = self._executor.submit(np.savez, path, **data)
        AsyncInterfExperiment._futures.append(future)
        return path

    @classmethod
    def flush(cls) -> None:
        """Block until all pending async saves complete.

        Re-raises the first I/O error encountered (if any), after waiting for
        all futures to finish.  Clears the future list on exit.
        """
        errors = []
        for f in cls._futures:
            try:
                f.result()
            except Exception as exc:
                errors.append(exc)
        cls._futures.clear()
        if errors:
            raise RuntimeError(
                f'{len(errors)} async save(s) failed:\n'
                + '\n'.join(str(e) for e in errors)
            )


@dataclass
class AsyncSunExperiment(AsyncInterfExperiment):
    """Async interferometric observation of the Sun."""

    prefix: str = 'sun'

    def run(self) -> str:
        alt, az, *_ = compute_sun_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return super().run()


@dataclass
class AsyncMoonExperiment(AsyncInterfExperiment):
    """Async interferometric observation of the Moon."""

    prefix: str = 'moon'

    def run(self) -> str:
        alt, az, *_ = compute_moon_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return super().run()


@dataclass
class AsyncRadecExperiment(AsyncInterfExperiment):
    """Async interferometric observation of a fixed J2000 (RA, Dec) target."""

    ra_deg:  float = 0.0
    dec_deg: float = 0.0

    def run(self) -> str:
        alt, az, _ = compute_radec_pointing(
            self.ra_deg, self.dec_deg, self.lat, self.lon, self.obs_alt,
        )
        self.alt_deg, self.az_deg = alt, az
        return super().run()
