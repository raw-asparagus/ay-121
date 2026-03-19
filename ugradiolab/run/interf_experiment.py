import time
from dataclasses import dataclass, field

import numpy as np

from ..drivers.interferometer import (
    compute_sun_pointing, compute_moon_pointing, compute_radec_pointing,
)
from ..utils import make_path
from .experiment import Experiment


@dataclass
class InterfExperiment(Experiment):
    """Interferometric observation: point both antennas, capture.

    Geometric delay correction is applied in post-processing using the saved
    alt_deg/az_deg and the known baseline.

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator (mode='corr'). Calls read_data(prev_cnt)
        repeatedly for duration_sec and averages the corr01 spectra.
    duration_sec : float
        Total integration window (seconds); read_data() dumps are averaged.
    """
    interferometer: object = field(default=None, repr=False, compare=False)
    snap:           object = field(default=None, repr=False, compare=False)
    duration_sec:   float  = 10.0
    pointing_tol_deg: float = 1.0   # reject capture if either antenna drifts beyond this

    def _run_summary(self) -> list[str]:
        return [f'  duration={self.duration_sec}s']

    def _verify_on_target(self, when: str) -> None:
        """Raise RuntimeError if either antenna is off-target by more than pointing_tol_deg."""
        pos = self.interferometer.get_pointing()
        for name, (alt, az) in pos.items():
            # Great-circle separation: azimuthal contribution scales as cos(alt)
            # to avoid over-rejection at high elevations (Issue 4).
            cos_alt = np.cos(np.radians(self.alt_deg))
            az_delta = (az - self.az_deg + 180.0) % 360.0 - 180.0
            err = float(np.hypot(alt - self.alt_deg, az_delta * cos_alt))
            if err > self.pointing_tol_deg:
                raise RuntimeError(
                    f'{when}: antenna {name} is {err:.2f}° off-target '
                    f'(tolerance {self.pointing_tol_deg}°) — telescope may have been slewed.'
                )

    def _prepare(self) -> tuple[float, float]:
        """Ephemeris lookup: set self.alt_deg/az_deg and return (alt, az).

        Base implementation returns the already-set values unchanged.
        Subclasses override to compute current pointing from source coordinates.
        Called by _collect() before every slew, and by ContinuousCapture before
        firing the non-blocking point().
        """
        return self.alt_deg, self.az_deg

    def _read_data(self) -> dict:
        """SNAP accumulation loop — return data dict without saving.

        Accumulates read_data() dumps for duration_sec, then averages.
        read_data(prev_cnt) blocks until acc_cnt changes, then asserts the counter
        did not advance again *during* the read.  If it did (another process is
        using the board), an AssertionError is raised: discard all spectra collected
        so far (the ADC state is unknown), reset prev_cnt, and retry from scratch.
        Three consecutive failures raises RuntimeError so the caller can skip the
        capture entirely rather than spinning indefinitely.

        Returns
        -------
        dict
            Keys ready to pass directly to np.savez(path, **data).
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
                d        = self.snap.read_data(prev_cnt=prev_cnt)
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

        if not spectra:
            raise RuntimeError('No valid SNAP dumps collected within the capture window.')

        corr          = np.mean(spectra, axis=0)   # complex128, all 1024 channels
        corr_std      = np.std(spectra,  axis=0)   # per-channel scatter across dumps
        unix_time_end = d['time']

        return dict(
            corr            = corr,
            corr_std        = corr_std,
            unix_time_start = unix_time_start,
            unix_time_end   = unix_time_end,
            n_acc           = len(spectra),
            f_s_hz          = 500e6,
            # SNAP uses a 2048-point real FFT → 1024 unique positive-frequency channels.
            # Channel spacing: Δf = f_s / n_fft = 500/2048 = 244.1 kHz/channel.
            # All 1024 channels are unique (not hermitian mirrors).
            # Frequency axis: f_sky(k) = f_rf0_hz + k * f_s/n_fft
            #
            # f_rf0_hz derivation (sky RF at channel 0):
            # LO chain: LO1=8750 MHz, LO2=1540 MHz, f_s=500 MHz
            #   IF2 = LO1 + LO2 - f_sky  (e.g. 10 GHz → IF2 = 290 MHz)
            #   IF2=290 MHz is in 2nd Nyquist zone (250–500 MHz); aliases to f_s−IF2
            #   Channel 0: f_alias=0 → IF2=f_s → f_sky = LO1+LO2−f_s = 9790 MHz
            #   10 GHz sky → channel ≈ (10000−9790)/0.2441 ≈ 860
            n_fft           = 2048,
            f_rf0_hz        = 9790e6,
            alt_deg         = self.alt_deg,
            az_deg          = self.az_deg,
            ra_deg          = getattr(self, 'ra_deg',  np.nan),
            dec_deg         = getattr(self, 'dec_deg', np.nan),
            duration_sec    = self.duration_sec,
            lat             = self.lat,
            lon             = self.lon,
            obs_alt         = self.obs_alt,
        )

    def _collect(self) -> tuple[str, dict]:
        """Point, capture, verify — return (path, data_dict) without saving."""
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
        """Execute the interferometric observation using self.interferometer and self.snap.

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        path, data = self._collect()
        np.savez(path, **data)
        return path


@dataclass
class SunExperiment(InterfExperiment):
    """Interferometric observation of the Sun.

    Computes current Sun position at run time and delegates to InterfExperiment.
    """
    prefix: str = 'sun'

    def _prepare(self) -> tuple[float, float]:
        alt, az, *_ = compute_sun_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return alt, az


@dataclass
class MoonExperiment(InterfExperiment):
    """Interferometric observation of the Moon.

    Computes current Moon position at run time and delegates to InterfExperiment.
    """
    prefix: str = 'moon'

    def _prepare(self) -> tuple[float, float]:
        alt, az, *_ = compute_moon_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return alt, az


@dataclass
class RadecExperiment(InterfExperiment):
    """Interferometric observation of a fixed J2000 (RA, Dec) target.

    Computes current alt/az from the supplied equatorial coordinates at run time
    and delegates to InterfExperiment.  Use this for catalog sources such as
    Cas A, Tau A, Cyg A, or Orion A.

    Parameters
    ----------
    ra_deg : float
        J2000 right ascension in degrees.
    dec_deg : float
        J2000 declination in degrees.
    """
    ra_deg:  float = 0.0
    dec_deg: float = 0.0

    def _prepare(self) -> tuple[float, float]:
        alt, az, _ = compute_radec_pointing(
            self.ra_deg, self.dec_deg, self.lat, self.lon, self.obs_alt,
        )
        self.alt_deg, self.az_deg = alt, az
        return alt, az
