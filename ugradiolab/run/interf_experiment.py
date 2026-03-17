import time
from dataclasses import dataclass, field

import numpy as np

from ..drivers.interferometer import (
    compute_sun_pointing, compute_moon_pointing, compute_radec_pointing, geometric_delay_ns,
)
from ..utils import make_path
from .experiment import Experiment


@dataclass
class InterfExperiment(Experiment):
    """Interferometric observation: point both antennas, optionally set delay, capture.

    Delay compensation requires a calibrated baseline. Leave baseline_ew_m=None
    (the default) to skip delay entirely — the appropriate mode when running the
    initial Sun calibration to determine the baseline from fringe data.

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator (mode='corr'). Calls read_data(prev_cnt)
        repeatedly for duration_sec and averages the corr01 spectra.
    delay_line : ugradio.interf_delay.DelayClient, optional
        Connected delay-line client. Used only when baseline_ew_m is set.
    duration_sec : float
        Total integration window (seconds); read_data() dumps are averaged.
    baseline_ew_m : float or None
        East-west baseline in metres, fit from fringe data. None disables delay.
    baseline_ns_m : float
        North-south baseline in metres. Ignored when baseline_ew_m is None.
    delay_max_ns : float or None
        Hardware delay limit in ns; clips the computed delay. Confirmed value:
        64.8 ns (ugradio.interf_delay.MAX_DELAY, calibrated 2019-03-21).
    """
    interferometer: object      = field(default=None, repr=False, compare=False)
    snap:           object      = field(default=None, repr=False, compare=False)
    delay_line:     object      = field(default=None, repr=False, compare=False)
    duration_sec:   float       = 10.0
    baseline_ew_m:  float | None = None
    baseline_ns_m:  float       = 0.0
    delay_max_ns:   float | None = 64.8  # hardware limit confirmed: ugradio.interf_delay.MAX_DELAY

    def _run_summary(self) -> list[str]:
        return [f'  duration={self.duration_sec}s']

    def run(self) -> str:
        """Execute the interferometric observation using self.interferometer, self.snap, self.delay_line.

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        self.interferometer.point(self.alt_deg, self.az_deg)

        tau = None
        if self.baseline_ew_m is not None and self.delay_line is not None:
            tau = geometric_delay_ns(
                self.alt_deg, self.az_deg,
                self.baseline_ew_m, self.baseline_ns_m,
                lat=self.lat,
                delay_max_ns=self.delay_max_ns,
            )
            self.delay_line.delay_ns(tau)

        # Accumulate snap_spec read_data() dumps for duration_sec, then average.
        # read_data(prev_cnt) blocks until acc_cnt changes, then asserts the counter
        # did not advance again *during* the read.  If it did (another process is
        # using the board), an AssertionError is raised: discard all spectra collected
        # so far (the ADC state is unknown), reset prev_cnt, and retry from scratch.
        # Three consecutive failures raises RuntimeError so the caller can skip the
        # capture entirely rather than spinning indefinitely.
        _MAX_RETRIES = 3
        t_end             = time.time() + self.duration_sec
        spectra           = []
        d                 = None
        prev_cnt          = None
        consecutive_errors = 0
        while time.time() < t_end:
            try:
                d        = self.snap.read_data(prev_cnt=prev_cnt)
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
                prev_cnt = None

        if not spectra:
            raise RuntimeError('No valid SNAP dumps collected within the capture window.')

        corr      = np.mean(spectra, axis=0)   # complex128, all 1024 channels
        unix_time = d['time']

        path = make_path(self.outdir, self.prefix, 'corr')
        np.savez(
            path,
            corr          = corr,
            unix_time     = unix_time,
            n_acc         = len(spectra),
            f_s_hz        = 500e6,
            # Sky RF frequency at SNAP channel 0.
            # LO chain: LO1=8750 MHz, LO2=1540 MHz, f_s=500 MHz
            #   IF2 = LO1 + LO2 - f_sky  (e.g. 10 GHz → IF2 = 290 MHz)
            #   ADC Nyquist = 250 MHz; IF2 in 2nd zone aliases: f_dig = f_s - IF2
            #   Channel 0 has f_dig = 0  →  IF2 = f_s  →  f_sky = LO1+LO2-f_s
            #   f_sky(ch=0) = 8750 + 1540 - 500 = 9790 MHz
            f_rf0_hz      = 9790e6,
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

    def run(self) -> str:
        alt, az, *_ = compute_sun_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return super().run()


@dataclass
class MoonExperiment(InterfExperiment):
    """Interferometric observation of the Moon.

    Computes current Moon position at run time and delegates to InterfExperiment.
    """
    prefix: str = 'moon'

    def run(self) -> str:
        alt, az, *_ = compute_moon_pointing(self.lat, self.lon, self.obs_alt)
        self.alt_deg, self.az_deg = alt, az
        return super().run()


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

    def run(self) -> str:
        alt, az, _ = compute_radec_pointing(
            self.ra_deg, self.dec_deg, self.lat, self.lon, self.obs_alt,
        )
        self.alt_deg, self.az_deg = alt, az
        return super().run()
