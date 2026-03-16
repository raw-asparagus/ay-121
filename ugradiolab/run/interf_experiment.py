from dataclasses import dataclass

import numpy as np
import ugradio.nch as nch

from ..drivers.interferometer import compute_sun_pointing, compute_moon_pointing, geometric_delay_ns
from ..utils import make_path


@dataclass
class InterfExperiment:
    """Interferometric observation: point both antennas, optionally set delay, capture.

    Delay compensation requires a calibrated baseline. Leave baseline_ew_m=None
    (the default) to skip delay entirely — the appropriate mode when running the
    initial Sun calibration to determine the baseline from fringe data.

    Parameters
    ----------
    alt_deg, az_deg : float
        Pointing direction in degrees (horizontal coordinates).
    duration_sec : float
        Integration time passed to snap.get_corr().
    outdir : str
        Output directory (created if absent).
    prefix : str
        Filename prefix for the saved .npz file.
    baseline_ew_m : float or None
        East-west baseline in metres, fit from fringe data. None disables delay.
    baseline_ns_m : float
        North-south baseline in metres, fit from fringe data. Ignored when
        baseline_ew_m is None.
    delay_max_ns : float or None
        Hardware delay limit; clips the computed delay when provided. Omit
        until confirmed from DelayClient documentation.
    lat, lon : float
        Observer latitude/longitude in degrees. Defaults to NCH.
    obs_alt : float
        Observer altitude in metres. Defaults to NCH.
    """
    alt_deg:       float        = 0.0
    az_deg:        float        = 0.0
    duration_sec:  float        = 10.0
    outdir:        str          = 'data/'
    prefix:        str          = 'interf'
    baseline_ew_m: float | None = None
    baseline_ns_m: float        = 0.0
    delay_max_ns:  float | None = None
    lat:           float        = nch.lat
    lon:           float        = nch.lon
    obs_alt:       float        = nch.alt

    def run(self, interferometer, snap, delay_line=None) -> str:
        """Execute the interferometric observation.

        Parameters
        ----------
        interferometer : ugradio.interf.Interferometer
            Connected interferometer controller.
        snap : snap_spec object
            SNAP correlator; must provide get_corr(duration_sec).
        delay_line : ugradio.interf_delay.DelayClient, optional
            Connected delay-line client. Used only when baseline_ew_m is set.

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        interferometer.point(self.alt_deg, self.az_deg)

        tau = None
        if self.baseline_ew_m is not None and delay_line is not None:
            tau = geometric_delay_ns(
                self.alt_deg, self.az_deg,
                self.baseline_ew_m, self.baseline_ns_m,
                lat=self.lat,
                delay_max_ns=self.delay_max_ns,
            )
            delay_line.delay_ns(tau)

        # NOTE: snap.get_corr() is a placeholder — confirm snap_spec API before use.
        data = snap.get_corr(self.duration_sec)

        path = make_path(self.outdir, self.prefix, 'corr')
        np.savez(
            path,
            data          = data,
            alt_deg       = self.alt_deg,
            az_deg        = self.az_deg,
            duration_sec  = self.duration_sec,
            baseline_ew_m = np.nan if self.baseline_ew_m is None else self.baseline_ew_m,
            baseline_ns_m = self.baseline_ns_m,
            delay_ns      = np.nan if tau is None else tau,
            lat           = self.lat,
            lon           = self.lon,
            obs_alt       = self.obs_alt,
        )
        return path


@dataclass
class SunExperiment(InterfExperiment):
    """Interferometric observation of the Sun.

    Computes current Sun position at run time and delegates to InterfExperiment.
    """
    prefix: str = 'sun'

    def run(self, interferometer, snap, delay_line=None) -> str:
        alt, az, *_ = compute_sun_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return super().run(interferometer, snap, delay_line)


@dataclass
class MoonExperiment(InterfExperiment):
    """Interferometric observation of the Moon.

    Computes current Moon position at run time and delegates to InterfExperiment.
    """
    prefix: str = 'moon'

    def run(self, interferometer, snap, delay_line=None) -> str:
        alt, az, *_ = compute_moon_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return super().run(interferometer, snap, delay_line)
