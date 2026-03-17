import time
from dataclasses import dataclass, field

import numpy as np
import ugradio.nch as nch

from ..drivers.interferometer import compute_sun_pointing, compute_moon_pointing, geometric_delay_ns
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
        Hardware delay limit; clips the computed delay when provided.
    """
    interferometer: object      = field(default=None, repr=False, compare=False)
    snap:           object      = field(default=None, repr=False, compare=False)
    delay_line:     object      = field(default=None, repr=False, compare=False)
    duration_sec:   float       = 10.0
    baseline_ew_m:  float | None = None
    baseline_ns_m:  float       = 0.0
    delay_max_ns:   float | None = None

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
        # did not advance again *during* the read.  If it did (accumulation period
        # shorter than transfer time), an AssertionError is raised — discard that
        # dump, reset prev_cnt so the next call reads whatever is freshest, and retry.
        t_end    = time.time() + self.duration_sec
        spectra  = []
        d        = None
        prev_cnt = None
        while time.time() < t_end:
            d        = self.snap.read_data(prev_cnt=prev_cnt)
            spectra.append(d['corr01'])
            prev_cnt = d['acc_cnt']

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
            #   IF2 = LO1 + LO2 - f_sky  →  aliases from 2nd Nyquist zone
            #   f_sky(ch=0) = LO1 + LO2 - f_s = 8750 + 1540 - 500 = 9790 MHz
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
