from dataclasses import dataclass, field

from ..data import Record
from ..io.paths import make_path
from .base import Experiment


@dataclass
class SDRExperiment(Experiment):
    """Base class for SDR-backed captures.

    Attributes
    ----------
    sdr : object
        SDR device used to collect raw I/Q samples.
    nsamples : int
        Number of samples captured per block.
    nblocks : int
        Number of blocks retained for each saved capture.
    sample_rate : float
        SDR sample rate in Hz.
    center_freq : float
        SDR center frequency in Hz.
    gain : float
        SDR gain setting.
    direct : bool
        Whether direct sampling mode is enabled.
    """
    sdr:         object = field(default=None, repr=False, compare=False)
    nsamples:    int    = 32768
    nblocks:     int    = 1
    sample_rate: float  = 2.56e6
    center_freq: float  = 1420e6
    gain:        float  = 0.0
    direct:      bool   = False

    def _run_summary(self) -> list[str]:
        """Return status lines for interactive runner output."""
        return [
            f'  nsamples={self.nsamples}  nblocks={self.nblocks}'
            f'  sample_rate={self.sample_rate / 1e6:.2f} MHz',
            f'  siggen: {self.siggen_summary()}',
        ]

    def _configure_sdr(self):
        """Apply the configured tuning and gain settings to the SDR.

        Notes
        -----
        This helper does not catch hardware-control exceptions. Any exception
        raised by the SDR driver propagates to the caller unchanged.
        """
        sdr = self.sdr
        sdr.direct = self.direct
        if self.direct:
            sdr.set_direct_sampling('q')
            sdr.set_center_freq(0)
        else:
            sdr.set_direct_sampling(0)
            sdr.set_center_freq(self.center_freq)
        sdr.set_gain(self.gain)
        sdr.set_sample_rate(self.sample_rate)

    def _capture(self, synth=None):
        """Capture one record from the SDR and package it as a ``Record``.

        Notes
        -----
        This helper performs no local exception handling. Exceptions from the
        SDR driver and from ``Record.from_sdr`` propagate to the caller.
        """
        sdr = self.sdr
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
            obs_alt=self.obs_alt,
            synth=synth,
        )

    def siggen_summary(self) -> str:
        """Return a short signal-generator status summary.

        Returns
        -------
        summary : str
            Human-readable summary text for runner output.
        """
        return 'OFF'


@dataclass
class CalExperiment(SDRExperiment):
    """Calibration experiment with signal generator.

    Parameters
    ----------
    siggen_freq_mhz : float
        Signal generator CW frequency in MHz.
    siggen_amp_dbm : float
        Signal generator amplitude in dBm.
    """
    synth:           object = field(default=None, repr=False, compare=False)
    siggen_freq_mhz: float  = 1420.405751768
    siggen_amp_dbm:  float  = -80.0

    def siggen_summary(self) -> str:
        """Return the configured signal-generator settings.

        Returns
        -------
        summary : str
            Frequency and amplitude summary for runner output.
        """
        return f'{self.siggen_freq_mhz} MHz, {self.siggen_amp_dbm} dBm'

    def run(self) -> str:
        """Execute the calibration capture and save the resulting record.

        Returns
        -------
        path : str
            Path to the saved ``.npz`` file.

        Raises
        ------
        ValueError
            If ``synth`` is not configured, or if the captured data cannot be
            sanitized into a ``Record``.
        OSError
            If saving the record fails.
        """
        if self.synth is None:
            raise ValueError(
                'CalExperiment requires a connected signal generator (synth).'
            )
        self._configure_sdr()
        path = make_path(self.outdir, self.prefix, 'cal')
        try:
            self.synth.set_freq_mhz(self.siggen_freq_mhz)
            self.synth.set_ampl_dbm(self.siggen_amp_dbm)
            self.synth.rf_on()
            record = self._capture(synth=self.synth)
            record.save(path)
        finally:
            self.synth.rf_off()
        return path


@dataclass
class ObsExperiment(SDRExperiment):
    """Sky observation experiment."""

    def run(self) -> str:
        """Execute the sky observation and save the resulting record.

        Returns
        -------
        path : str
            Path to the saved ``.npz`` file.

        Raises
        ------
        ValueError
            If the captured data cannot be sanitized into a ``Record``.
        OSError
            If saving the record fails.
        """
        self._configure_sdr()
        path   = make_path(self.outdir, self.prefix, 'obs')
        record = self._capture()
        record.save(path)
        return path
