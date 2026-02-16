"""Experiment specification for SDR data collection workflows.

Define collection parameters as dataclass instances, then execute them
sequentially with ``QueueRunner`` to produce one .npz file per experiment.
"""

import os
import time
from dataclasses import dataclass, field

import ugradio.nch as nch

from .drivers.siggen import set_signal
from .data.schema import build_record, save_record


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
        if synth is None:
            raise ValueError('CalExperiment requires a connected signal generator (synth).')

        self._configure_sdr(sdr)
        path = _make_path(self.outdir, self.prefix, 'cal')
        try:
            set_signal(synth, self.siggen_freq_mhz, self.siggen_amp_dbm)
            # Discard the first block, which may contain stale ring-buffer data.
            data = sdr.capture_data(nsamples=self.nsamples, nblocks=self.nblocks + 1)[1:]
            record = build_record(
                data, sdr, alt_deg=self.alt_deg, az_deg=self.az_deg,
                lat=self.lat, lon=self.lon, observer_alt=self.observer_alt,
                synth=synth
            )
            save_record(path, record)
        finally:
            synth.rf_off()
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
        # Discard the first block, which may contain stale ring-buffer data.
        data = sdr.capture_data(nsamples=self.nsamples, nblocks=self.nblocks + 1)[1:]
        path = _make_path(self.outdir, self.prefix, 'obs')
        record = build_record(
            data, sdr, alt_deg=self.alt_deg, az_deg=self.az_deg,
            lat=self.lat, lon=self.lon, observer_alt=self.observer_alt,
            synth=None
        )
        save_record(path, record)
        return path
