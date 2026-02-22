from typing import Any, Dict, List

import numpy as np
from scipy.signal import find_peaks as _find_peaks
from scipy.stats import gamma as _gamma


class SpectrumBase:
    """Base class providing common spectrum analysis methods.

    Subclasses must expose ``psd`` (ndarray), ``freqs`` (ndarray),
    and ``nblocks`` (int) as attributes or properties.
    """

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

    def smooth(self, method: str = 'gaussian', **kwargs) -> np.ndarray:
        """Return a smoothed copy of psd. The original ``psd`` is unchanged.

        All three methods convolve the spectrum with a kernel in the frequency
        domain.  Because PSD bins are independent for white noise, smoothing the
        stored PSD gives the same result as recomputing at lower resolution from
        raw data.  Noise std scales as ``1/sqrt(M_eff)`` where ``M_eff`` is the
        effective kernel width in bins.

        Parameters
        ----------
        method : {'gaussian', 'savgol', 'boxcar'}
        **kwargs
            gaussian : sigma (float, default 32) — kernel width in bins.
            savgol   : window_length (int, default 129), polyorder (int, default 3).
            boxcar   : M (int, default 64) — number of bins to average.

        Returns
        -------
        np.ndarray, shape (nfrequencies,)
        """
        if method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(self.psd, sigma=kwargs.get('sigma', 32))
        elif method == 'savgol':
            from scipy.signal import savgol_filter
            return savgol_filter(
                self.psd,
                window_length=kwargs.get('window_length', 129),
                polyorder=kwargs.get('polyorder', 3),
            )
        elif method == 'boxcar':
            M = kwargs.get('M', 64)
            return np.convolve(self.psd, np.ones(M) / M, mode='same')
        else:
            raise ValueError(
                f"Unknown method {method!r}. Choose 'gaussian', 'savgol', or 'boxcar'."
            )

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
        nfreqs = len(self.psd)
        noise_mean = np.median(self.psd)
        p_per_bin = p_false_alarm / nfreqs
        threshold = _gamma.ppf(
            1.0 - p_per_bin, a=self.nblocks, scale=noise_mean / self.nblocks
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