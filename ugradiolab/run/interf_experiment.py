from dataclasses import dataclass, field

import numpy as np
import ugradio.nch as nch

from ..drivers.interferometer import compute_sun_pointing, compute_moon_pointing, geometric_delay_ns
from ..utils import make_path
from .experiment import Experiment


@dataclass
class InterfExperiment(Experiment):
    """Interferometric observation: point both antennas, optionally set delay, capture one SNAP dump.

    Each call to run() reads exactly one accumulation from the SNAP correlator
    (~0.625 s at ACC_LEN=38150, SPEC_PER_ACC=8, f_s=500 MHz) and saves it as a
    separate .npz file.  Passing prev_cnt between successive run() calls ensures
    each dump is a fresh accumulation — the SNAP blocks until the next one is
    ready rather than returning the same buffer twice.

    Delay compensation requires a calibrated baseline. Leave baseline_ew_m=None
    (the default) to skip delay entirely — the appropriate mode when running the
    initial Sun calibration to determine the baseline from fringe data.

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator (mode='corr').
    delay_line : ugradio.interf_delay.DelayClient, optional
        Connected delay-line client. Used only when baseline_ew_m is set.
    baseline_ew_m : float or None
        East-west baseline in metres, fit from fringe data. None disables delay.
    baseline_ns_m : float
        North-south baseline in metres. Ignored when baseline_ew_m is None.
    delay_max_ns : float or None
        Hardware delay limit; clips the computed delay when provided.
    _prev_cnt : int or None
        Last acc_cnt returned by the SNAP; updated automatically after each run().
        Do not set manually — it is managed by successive run() calls.
    """
    interferometer: object      = field(default=None, repr=False, compare=False)
    snap:           object      = field(default=None, repr=False, compare=False)
    delay_line:     object      = field(default=None, repr=False, compare=False)
    baseline_ew_m:  float | None = None
    baseline_ns_m:  float       = 0.0
    delay_max_ns:   float | None = None
    _prev_cnt:      int | None  = field(default=None, repr=False, compare=False)

    def _run_summary(self) -> list[str]:
        return []

    def run(self) -> str:
        """Capture one SNAP dump, save to .npz, and return the file path.

        Blocks until a new accumulation is available (via prev_cnt handshake).

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

        # Read exactly one new accumulation; blocks until acc_cnt changes.
        d = self.snap.read_data(prev_cnt=self._prev_cnt)
        self._prev_cnt = d['acc_cnt']

        corr_full = d['corr01']                        # complex128, shape (1024,)
        corr      = corr_full[:len(corr_full) // 2]   # positive-frequency half (0–511)

        path = make_path(self.outdir, self.prefix, 'corr')
        np.savez(
            path,
            corr_i        = corr.real,
            corr_q        = corr.imag,
            unix_time     = d['time'],
            f_s_hz        = 500e6,
            f_rf0_hz      = 10.0e9,
            alt_deg       = self.alt_deg,
            az_deg        = self.az_deg,
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
