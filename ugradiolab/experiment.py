"""Experiment specification and sequential queue runner.

Define collection parameters as dataclass instances, then execute them
sequentially with ``run_queue`` to produce one .npz file per experiment.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
import ugradio.nch as nch

from .drivers.siggen import set_signal
from .data.schema import save_cal, save_obs


def _make_path(outdir, prefix, tag):
    """Generate a timestamped output filepath."""
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(outdir, f'{prefix}_{tag}_{ts}.npz')


@dataclass
class Experiment:
    """Base experiment specification (SDR parameters + output settings).

    Not intended to be run directly — use CalExperiment or ObsExperiment.
    """
    nsamples: int = 2048
    nblocks: int = 1
    sample_rate: float = 2.56e6
    center_freq: float = 0.0
    gain: float = 0.0
    direct: bool = True
    outdir: str = '.'
    prefix: str = 'exp'

    def _configure_sdr(self, sdr):
        """Apply this experiment's SDR parameters to an existing SDR object."""
        if self.direct:
            sdr.set_direct_sampling('q')
            sdr.set_center_freq(0)
        else:
            sdr.set_direct_sampling(0)
            sdr.set_center_freq(self.center_freq)
        sdr.direct = self.direct
        sdr.set_sample_rate(self.sample_rate)
        sdr.set_gain(self.gain)

    def run(self, sdr, synth=None):
        raise NotImplementedError


@dataclass
class CalExperiment(Experiment):
    """Calibration experiment with signal generator.

    Parameters
    ----------
    siggen_freq_mhz : float
        Signal generator CW frequency in MHz.
    siggen_amp_dbm : float
        Signal generator amplitude in dBm.
    """
    siggen_freq_mhz: float = 0.0
    siggen_amp_dbm: float = -10.0
    alt_deg: float = 0.0
    az_deg: float = 0.0
    lat: float = field(default_factory=lambda: nch.lat)
    lon: float = field(default_factory=lambda: nch.lon)
    observer_alt: float = field(default_factory=lambda: nch.alt)

    def run(self, sdr, synth=None):
        """Execute the calibration experiment.

        Parameters
        ----------
        sdr : ugradio.sdr.SDR
            Initialized SDR (will be reconfigured to match experiment params).
        synth : ugradio.agilent.SynthDirect
            Connected signal generator.

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        self._configure_sdr(sdr)
        set_signal(synth, self.siggen_freq_mhz, self.siggen_amp_dbm)
        from .drivers.sdr_utils import _capture
        data = _capture(sdr, self.nsamples, self.nblocks)
        synth.rf_off()
        path = _make_path(self.outdir, self.prefix, 'cal')
        save_cal(path, data, sdr, synth, alt_deg=self.alt_deg,
                 az_deg=self.az_deg, lat=self.lat, lon=self.lon,
                 observer_alt=self.observer_alt)
        return path


@dataclass
class ObsExperiment(Experiment):
    """Sky observation experiment.

    Parameters
    ----------
    alt_deg : float
        Telescope altitude/elevation in degrees.
    az_deg : float
        Telescope azimuth in degrees.
    lat : float
        Observer latitude in degrees.
    lon : float
        Observer longitude in degrees.
    observer_alt : float
        Observer altitude in meters.
    """
    alt_deg: float = 0.0
    az_deg: float = 0.0
    lat: float = field(default_factory=lambda: nch.lat)
    lon: float = field(default_factory=lambda: nch.lon)
    observer_alt: float = field(default_factory=lambda: nch.alt)

    def run(self, sdr, synth=None):
        """Execute the sky observation experiment.

        Parameters
        ----------
        sdr : ugradio.sdr.SDR
            Initialized SDR (will be reconfigured to match experiment params).
        synth : ignored.

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        self._configure_sdr(sdr)
        from .drivers.sdr_utils import _capture
        data = _capture(sdr, self.nsamples, self.nblocks)
        path = _make_path(self.outdir, self.prefix, 'obs')
        save_obs(path, data, sdr, self.alt_deg, self.az_deg,
                 lat=self.lat, lon=self.lon, observer_alt=self.observer_alt)
        return path


def _format_experiment(exp, index, total):
    """Format experiment details for display."""
    tag = type(exp).__name__
    lines = [f'[{index}/{total}] {exp.prefix} ({tag})']
    lines.append(f'  alt={exp.alt_deg}  az={exp.az_deg}')
    lines.append(f'  nsamples={exp.nsamples}  nblocks={exp.nblocks}  '
                 f'sample_rate={exp.sample_rate/1e6:.2f} MHz')
    if isinstance(exp, CalExperiment):
        lines.append(f'  siggen: {exp.siggen_freq_mhz} MHz, '
                     f'{exp.siggen_amp_dbm} dBm')
    else:
        lines.append(f'  siggen: OFF')
    return '\n'.join(lines)


def run_queue(experiments, sdr, synth=None, confirm=True, cadence_sec=None):
    """Execute a list of experiments sequentially.

    Parameters
    ----------
    experiments : list of Experiment
        Experiments to run in order.
    sdr : ugradio.sdr.SDR
        Initialized SDR object (reused across experiments).
    synth : ugradio.agilent.SynthDirect, optional
        Signal generator (required for CalExperiment entries).
    confirm : bool
        If True, prompt for confirmation before each experiment.
        Enter=run, s=skip, q=quit. Default: True.
    cadence_sec : float, optional
        When set, enforce a minimum interval (in seconds) between the
        start of consecutive ObsExperiment runs.

    Returns
    -------
    list of str
        Paths to the saved .npz files, one per experiment.
    """
    n = len(experiments)
    paths = []
    obs_start = None
    for i, exp in enumerate(experiments):
        # Sleep until next cadence boundary (before starting an ObsExperiment)
        if cadence_sec and isinstance(exp, ObsExperiment) and obs_start is not None:
            elapsed = time.time() - obs_start
            wait = cadence_sec - elapsed
            if wait > 0:
                print(f'  sleeping {wait:.1f}s until next cadence...')
                time.sleep(wait)

        print(_format_experiment(exp, i + 1, n))
        if confirm:
            resp = input('  [Enter]=run  s=skip  q=quit: ').strip().lower()
            if resp == 'q':
                print('Queue aborted.')
                break
            if resp == 's':
                print('  skipped.')
                continue

        if isinstance(exp, ObsExperiment):
            obs_start = time.time()

        path = exp.run(sdr, synth=synth)
        paths.append(path)
        print(f'  -> {path}')
    return paths