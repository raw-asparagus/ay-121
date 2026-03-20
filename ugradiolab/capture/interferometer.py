import time
from dataclasses import dataclass, field

import numpy as np

from ..astronomy.ephemeris import (
    compute_sun_pointing, compute_moon_pointing, compute_radec_pointing,
)
from ..io.paths import make_path
from .base import Experiment


class PointingError(RuntimeError):
    """Raised when the interferometer is measurably off the requested target."""


@dataclass
class InterfExperiment(Experiment):
    """Base class for interferometric captures.

    Attributes
    ----------
    interferometer : object
        Pointing controller used to slew and query antenna positions.
    snap : object
        SNAP correlator interface used to read correlation spectra.
    duration_sec : float
        Integration time in seconds for each capture.
    pointing_tol_deg : float
        Maximum allowed angular error, in degrees, before a capture is
        rejected as off target.
    """
    interferometer: object = field(default=None, repr=False, compare=False)
    snap:           object = field(default=None, repr=False, compare=False)
    duration_sec:   float  = 10.0
    pointing_tol_deg: float = 1.0   # reject capture if either antenna drifts beyond this

    def _run_summary(self) -> list[str]:
        """Return status lines for interactive runner output."""
        return [f'  duration={self.duration_sec}s']

    def _verify_on_target(self, when: str) -> None:
        """Validate that both antennas remain on target.

        Raises
        ------
        PointingError
            If either antenna is farther than ``pointing_tol_deg`` from the
            requested target.
        """
        pos = self.interferometer.get_pointing()
        for name, (alt, az) in pos.items():
            # Great-circle separation: azimuthal contribution scales as cos(alt)
            # to avoid over-rejection at high elevations (Issue 4).
            cos_alt = np.cos(np.radians(self.alt_deg))
            az_delta = (az - self.az_deg + 180.0) % 360.0 - 180.0
            err = float(np.hypot(alt - self.alt_deg, az_delta * cos_alt))
            if err > self.pointing_tol_deg:
                raise PointingError(
                    f'{when}: antenna {name} is {err:.2f}° off-target '
                    f'(tolerance {self.pointing_tol_deg}°) — telescope may have been slewed.'
                )

    def _prepare(self) -> None:
        """Refresh the target position before a slew or capture."""

    def _read_data(self) -> dict:
        """Collect and average correlation spectra from the SNAP board.

        Notes
        -----
        ``AssertionError`` from ``snap.read_data`` is handled internally as a
        transient board-conflict signal. On each such failure this helper drops
        all partially collected spectra, resets the accumulator state, and
        retries from scratch. After three consecutive failures it stops retrying
        and raises ``RuntimeError``.

        Raises
        ------
        RuntimeError
            If the SNAP board appears to be in use by another process for three
            consecutive reads.
        """
        _MAX_RETRIES = 3
        t_end              = time.time() + self.duration_sec
        spectra            = []
        d                  = None
        prev_cnt           = None
        unix_time_start    = None
        consecutive_errors = 0
        while time.time() < t_end:
            try:
                d = self.snap.read_data(prev_cnt=prev_cnt)
                if unix_time_start is None:
                    unix_time_start = d['time']
                spectra.append(d['corr01'])
                prev_cnt = d['acc_cnt']
                consecutive_errors = 0
            except AssertionError:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_RETRIES:
                    raise RuntimeError(
                        f'SNAP board interference: {consecutive_errors} consecutive '
                        'read_data() failures — another process may hold the board.'
                    )
                spectra.clear()
                prev_cnt        = None
                unix_time_start = None

        corr          = np.mean(spectra, axis=0)   # complex128, all 1024 channels
        corr_std      = np.std(spectra,  axis=0)   # per-channel scatter across dumps
        unix_time_end = d['time']

        return dict(
            corr            = corr,
            corr_std        = corr_std,
            unix_time_start = unix_time_start,
            unix_time_end   = unix_time_end,
            n_acc           = len(spectra),
            alt_deg         = self.alt_deg,
            az_deg          = self.az_deg,
            ra_deg          = getattr(self, 'ra_deg',  np.nan),
            dec_deg         = getattr(self, 'dec_deg', np.nan),
            duration_sec    = self.duration_sec,
        )

    def _collect(self) -> tuple[str, dict]:
        """Point the dishes, collect data, and build the save payload.

        Raises
        ------
        PointingError
            If the post-slew or post-collect target verification fails.
        RuntimeError
            If the slew fails or the SNAP read loop fails.
        """
        self._prepare()
        try:
            self.interferometer.point(self.alt_deg, self.az_deg)
        except (AssertionError, TimeoutError, OSError) as exc:
            raise RuntimeError(f'pointing failed: {exc}') from exc
        self._verify_on_target('post-slew')
        data = self._read_data()
        self._verify_on_target('post-collect')
        path = make_path(self.outdir, self.prefix, 'corr')
        return path, data

    def run(self) -> str:
        """Execute the interferometric observation and save the result.

        Returns
        -------
        path : str
            Path to the saved ``.npz`` file.

        Raises
        ------
        PointingError
            If target verification fails after the slew or after data
            collection.
        RuntimeError
            If pointing or data collection fails.
        OSError
            If writing the output file fails.
        """
        path, data = self._collect()
        np.savez(path, **data)
        return path


@dataclass
class SunExperiment(InterfExperiment):
    """Interferometric observation of the Sun.

    Attributes
    ----------
    prefix : str
        Filename prefix for saved Sun observations.
    """
    prefix: str = 'sun'

    def _prepare(self) -> None:
        """Refresh the current Sun pointing.

        Notes
        -----
        This helper performs no local exception handling. Any exception raised
        by the ephemeris lookup propagates to the caller.
        """
        alt, az, *_ = compute_sun_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az


@dataclass
class MoonExperiment(InterfExperiment):
    """Interferometric observation of the Moon.

    Attributes
    ----------
    prefix : str
        Filename prefix for saved Moon observations.
    """
    prefix: str = 'moon'

    def _prepare(self) -> None:
        """Refresh the current Moon pointing.

        Notes
        -----
        This helper performs no local exception handling. Any exception raised
        by the ephemeris lookup propagates to the caller.
        """
        alt, az, *_ = compute_moon_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az


@dataclass
class RadecExperiment(InterfExperiment):
    """Interferometric observation of a fixed J2000 (RA, Dec) target.

    Attributes
    ----------
    ra_deg : float
        J2000 right ascension in degrees.
    dec_deg : float
        J2000 declination in degrees.
    """
    ra_deg:  float = 0.0
    dec_deg: float = 0.0

    def _prepare(self) -> None:
        """Refresh the current pointing for the configured equatorial target.

        Notes
        -----
        This helper performs no local exception handling. Any exception raised
        by the ephemeris lookup propagates to the caller.
        """
        alt, az, _ = compute_radec_pointing(
            self.ra_deg, self.dec_deg, self.lat, self.lon, self.obs_alt,
        )
        self.alt_deg, self.az_deg = alt, az
