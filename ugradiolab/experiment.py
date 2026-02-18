import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import ugradio.nch as nch

from .data import Record


def _make_path(outdir, prefix, tag):
    """Generate a timestamped output filepath."""
    os.makedirs(outdir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(outdir, f'{prefix}_{tag}_{ts}.npz')


@dataclass
class Experiment(ABC):
    """Base experiment specification (SDR parameters + output settings)."""
    nsamples: int = 32768
    nblocks: int = 1
    sample_rate: float = 2.56e6  # Hz
    center_freq: float = 1420e6  # Hz
    gain: float = 0.0
    direct: bool = False
    outdir: str = 'data/'
    prefix: str = 'exp'
    alt_deg: float = 0.0
    az_deg: float = 0.0
    lat: float = nch.lat
    lon: float = nch.lon
    observer_alt: float = nch.alt

    def _configure_sdr(self, sdr):
        """Apply this experiment's SDR parameters to an existing SDR object."""
        # Using the same statefull reinstantiation as in ugradio.sdr
        sdr.direct = self.direct
        if self.direct:
            sdr.set_direct_sampling('q')
            sdr.set_center_freq(0)
        else:
            sdr.set_direct_sampling(0)
            sdr.set_center_freq(self.center_freq)
        sdr.set_gain(self.gain)
        sdr.set_sample_rate(self.sample_rate)

    def _capture(self, sdr, synth=None):
        """Capture data and build a Record.

        Discards the first block to flush stale buffer.
        """
        raw_data = sdr.capture_data(
            nsamples=self.nsamples,
            nblocks=self.nblocks + 1,
        )
        return Record.from_sdr(
            raw_data[1:],
            sdr,
            alt_deg=self.alt_deg,
            az_deg=self.az_deg,
            lat=self.lat,
            lon=self.lon,
            observer_alt=self.observer_alt,
            synth=synth,
        )

    @property
    def counts_for_cadence(self) -> bool:
        return True

    def siggen_summary(self) -> str:
        return 'OFF'

    @abstractmethod
    def run(self, sdr, synth=None):
        ...


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
    siggen_freq_mhz: float = 1420.405751768
    siggen_amp_dbm: float = -80.0

    @property
    def counts_for_cadence(self) -> bool:
        return False

    def siggen_summary(self) -> str:
        return f'{self.siggen_freq_mhz} MHz, {self.siggen_amp_dbm} dBm'

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
            raise ValueError(
                'CalExperiment requires a connected signal generator (synth).'
            )

        self._configure_sdr(sdr)
        path = _make_path(self.outdir, self.prefix, 'cal')
        try:
            synth.set_freq_mhz(self.siggen_freq_mhz)
            synth.set_ampl_dbm(self.siggen_amp_dbm)
            synth.rf_on()
            record = self._capture(sdr, synth=synth)
            record.save(path)
        finally:
            synth.rf_off()
        return path


@dataclass
class ObsExperiment(Experiment):
    """Sky observation experiment."""

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
        record = self._capture(sdr)
        path = _make_path(self.outdir, self.prefix, 'obs')
        record.save(path)
        return path
