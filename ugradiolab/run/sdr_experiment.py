from dataclasses import dataclass, field

from ..models import Record
from ..utils import make_path
from .experiment import Experiment


@dataclass
class SDRExperiment(Experiment):
    """Abstract SDR-based experiment. Adds SDR hardware fields and capture helpers."""
    sdr:         object = field(default=None, repr=False, compare=False)
    nsamples:    int    = 32768
    nblocks:     int    = 1
    sample_rate: float  = 2.56e6
    center_freq: float  = 1420e6
    gain:        float  = 0.0
    direct:      bool   = False

    def _configure_sdr(self):
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
        return 'OFF'

    def _run_summary(self) -> list[str]:
        return [
            f'  nsamples={self.nsamples}  nblocks={self.nblocks}'
            f'  sample_rate={self.sample_rate / 1e6:.2f} MHz',
            f'  siggen: {self.siggen_summary()}',
        ]


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
        return f'{self.siggen_freq_mhz} MHz, {self.siggen_amp_dbm} dBm'

    def run(self) -> str:
        """Executes the calibration experiment using self.sdr and self.synth.

        Returns
        -------
        str
            Path to the saved .npz file.
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
        """Executes the sky observation experiment using self.sdr.

        Returns
        -------
        str
            Path to the saved .npz file.
        """
        self._configure_sdr()
        path   = make_path(self.outdir, self.prefix, 'obs')
        record = self._capture()
        record.save(path)
        return path
