from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from scipy.signal import find_peaks as _find_peaks
from scipy.stats import gamma as _gamma

from ..data import Record


@dataclass(frozen=True)
class Spectrum:
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
        block_psds = (
            np.abs(np.fft.fftshift(np.fft.fft(rec.data, axis=1), axes=1)) ** 2
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

    @property
    def freqs_mhz(self) -> np.ndarray:
        """Frequency axis in MHz."""
        return self.freqs / 1e6

    @property
    def total_power(self) -> float:
        """Total integrated power: sum of normalised PSD across all frequency bins.

        Because ``psd`` is normalised by ``nsamples²``, this equals
        ``mean(|IQ|²)`` (mean per-sample signal power in counts²) by
        Parseval's theorem, and is directly comparable across captures with
        different ``nsamples``.
        """
        return float(np.sum(self.psd))

    def bin_at(self, freq_hz: float) -> int:
        """Index of the frequency bin closest to freq_hz (in Hz)."""
        return int(np.argmin(np.abs(self.freqs - freq_hz)))

    def find_peaks(
        self,
        p_false_alarm: float = 1e-3,
        min_distance: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find peaks that stand significantly above the noise floor.

        The detection threshold is derived from the Gamma distribution of
        the averaged periodogram.  Under H₀ (noise only), each bin of the
        mean-of-N-blocks periodogram follows Gamma(N, μ/N) exactly for
        complex Gaussian noise.  A Bonferroni correction across all
        frequency bins controls the spectrum-wide false-alarm rate.

        Parameters
        ----------
        p_false_alarm : float
            Target probability of *any* false detection across the whole
            spectrum (default 1e-3).
        min_distance : int
            Minimum separation between peaks in frequency bins.

        Returns
        -------
        List of dicts with keys: freq_hz, freq_mhz, psd, prominence,
        snr_over_median_dB.
        """
        nblocks = self.record.data.shape[0]
        nfreqs = len(self.psd)

        noise_mean = np.median(self.psd)

        p_per_bin = p_false_alarm / nfreqs
        threshold = _gamma.ppf(
            1.0 - p_per_bin, a=nblocks, scale=noise_mean / nblocks
        )

        peak_idx, props = _find_peaks(
            self.psd,
            height=threshold,
            prominence=threshold - noise_mean,
            distance=min_distance,
        )
        return [
            {
                'freq_hz': self.freqs[i],
                'freq_mhz': self.freqs_mhz[i],
                'psd': self.psd[i],
                'prominence': props['prominences'][k],
                'snr_over_median_dB': 10 * np.log10(self.psd[i] / noise_mean),
            }
            for k, i in enumerate(peak_idx)
        ]
