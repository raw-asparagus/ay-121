from dataclasses import dataclass

import numpy as np

from ..data import Record
from ._base import SpectrumBase


@dataclass(frozen=True)
class Spectrum(SpectrumBase):
    """Integrated power spectrum computed from a Record.

    Attributes
    ----------
    record : Record
        The raw capture this spectrum was derived from.
    psd : np.ndarray, shape (nfrequencies,)
        Normalised mean power spectrum across blocks, DC-centred.
        Each bin is ``mean(|FFT|²) / nsamples²`` — *not* raw |FFT|²  — so
        that ``sum(psd)`` equals ``mean(|IQ|²)`` (mean per-sample signal
        power in counts²) by Parseval's theorem, independent of FFT length.
    std : float
        Standard error of the mean PSD estimator: mean across frequency bins
        of ``std(|FFT|²/nsamples², axis=0) / √nblocks``.  Represents the
        uncertainty on ``psd`` and scales as ``1/√nblocks``.
        Same units as ``psd``.
    freqs : np.ndarray, shape (nfrequencies,)
        Frequency axis in Hz, DC-centred, absolute (baseband + centre_freq).
    """

    record: Record
    psd: np.ndarray
    std: float
    freqs: np.ndarray

    @property
    def nblocks(self) -> int:
        return self.record.nblocks

    @classmethod
    def from_record(cls, rec: Record) -> 'Spectrum':
        """Compute the integrated power spectrum from a Record.

        Parameters
        ----------
        rec : Record
            Captured I/Q data with shape (nblocks, nsamples).

        Returns
        -------
        Spectrum
        """
        nblocks, nsamples = rec.data.shape
        data = rec.data - rec.data.mean(axis=1, keepdims=True)
        block_psds = (
            np.abs(np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)) ** 2
            / nsamples ** 2
        )
        freqs = np.fft.fftshift(
            np.fft.fftfreq(nsamples, d=1.0 / rec.sample_rate)
        ) + rec.center_freq
        return cls(
            record=rec,
            psd=np.mean(block_psds, axis=0),
            std=float(np.mean(np.std(block_psds, axis=0))) / np.sqrt(nblocks),
            freqs=freqs,
        )