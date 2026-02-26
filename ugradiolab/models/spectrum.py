import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from .record import Record

SmoothMethod = Literal['gaussian', 'savgol', 'boxcar']

_REQUIRED_KEYS = frozenset({
    'psd', 'std', 'freqs',
    'sample_rate', 'center_freq', 'gain', 'direct',
    'unix_time', 'jd', 'lst', 'alt', 'az',
    'obs_lat', 'obs_lon', 'obs_alt',
    'nblocks', 'nsamples',
})


@dataclass(frozen=True)
class Spectrum:
    """Integrated power spectrum with observation metadata.

    Attributes
    ----------
    psd : np.ndarray, shape (nfrequencies,)
        Normalised mean power spectrum across blocks, DC-centred.
    std : np.ndarray, shape (nfrequencies,)
        Per-frequency standard error of the mean PSD estimator.
    freqs : np.ndarray, shape (nfrequencies,)
        Frequency axis in Hz, DC-centred, absolute (baseband + centre_freq).
    sample_rate : float
        Sample rate in Hz.
    center_freq : float
        Centre frequency in Hz.
    gain : float
        SDR gain value.
    direct : bool
        Direct sampling mode flag.
    unix_time : float
        Unix timestamp of the capture.
    jd : float
        Julian Date at capture time.
    lst : float
        Local Sidereal Time at capture.
    alt : float
        Telescope altitude in degrees.
    az : float
        Telescope azimuth in degrees.
    obs_lat : float
        Observer latitude in degrees.
    obs_lon : float
        Observer longitude in degrees.
    obs_alt : float
        Observer altitude in metres.
    nblocks : int
        Number of captured blocks.
    nsamples : int
        Number of samples per block.
    siggen_freq : float or None
        Signal generator frequency in Hz, if applicable.
    siggen_amp : float or None
        Signal generator amplitude in dBm, if applicable.
    siggen_rf_on : bool or None
        Signal generator RF output state, if applicable.
    """

    psd: np.ndarray
    std: np.ndarray
    freqs: np.ndarray
    sample_rate: float
    center_freq: float
    gain: float
    direct: bool
    unix_time: float
    jd: float
    lst: float
    alt: float
    az: float
    obs_lat: float
    obs_lon: float
    obs_alt: float
    nblocks: int
    nsamples: int
    siggen_freq: float | None = None
    siggen_amp: float | None  = None
    siggen_rf_on: bool | None = None

    @property
    def uses_synth(self) -> bool:
        """True if all signal generator fields are populated."""
        return (
            self.siggen_freq is not None
            and self.siggen_amp is not None
            and self.siggen_rf_on is not None
        )

    @property
    def freqs_mhz(self) -> np.ndarray:
        """Frequency axis in MHz."""
        return self.freqs / 1e6

    @property
    def total_power(self) -> float:
        """Total integrated power."""
        return float(np.sum(self.psd))

    def bin_at(self, freq_hz: float) -> int:
        """Index of the frequency bin closest to freq_hz (in Hz)."""
        return int(np.argmin(np.abs(self.freqs - freq_hz)))

    def smooth(
            self,
            method: SmoothMethod = 'gaussian',
            **kwargs,
    ) -> np.ndarray:
        """Returns a smoothed copy of psd. The original ``psd`` is unchanged.

        Parameters
        ----------
        method : {'gaussian', 'savgol', 'boxcar'}
        **kwargs
            gaussian : sigma (float, default 32) — kernel width in bins.
            savgol   : window_length (int, default 129),
                       polyorder (int, default 3).
            boxcar   : M (int, default 64) — number of bins to average.

        Returns
        -------
        np.ndarray, shape (nfrequencies,)
        """
        if method == 'gaussian':
            return gaussian_filter1d(self.psd, sigma=kwargs.get('sigma', 32))
        elif method == 'savgol':
            return savgol_filter(
                self.psd,
                window_length = kwargs.get('window_length', 129),
                polyorder     = kwargs.get('polyorder', 3),
            )
        elif method == 'boxcar':
            M = kwargs.get('M', 64)
            return np.convolve(self.psd, np.ones(M) / M, mode='same')
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Choose 'gaussian', 'savgol', or 'boxcar'."
            )

    @classmethod
    def from_data(cls, filepath) -> 'Spectrum':
        """Computes a Spectrum from a Record specified by filepath.

        Parameters
        ----------
        filepath : str or Path
            Path to a Record .npz file written by ``Record.save``.

        Returns
        -------
        Spectrum
        """
        rec               = Record.load(filepath)
        nblocks, nsamples = rec.data.shape
        data              = rec.data - rec.data.mean(axis=1, keepdims=True)
        block_psds        = np.abs(np.fft.fftshift(
            np.fft.fft(data, axis=1),
            axes=1
        )) ** 2 / nsamples ** 2
        return cls(
            psd          = np.mean(block_psds, axis=0),
            std          = np.std(block_psds, axis=0) / np.sqrt(nblocks),
            freqs        = np.fft.fftshift(
                np.fft.fftfreq(nsamples, d=1.0 / rec.sample_rate)
            ) + rec.center_freq,
            sample_rate  = rec.sample_rate,
            center_freq  = rec.center_freq,
            gain         = rec.gain,
            direct       = rec.direct,
            unix_time    = rec.unix_time,
            jd           = rec.jd,
            lst          = rec.lst,
            alt          = rec.alt,
            az           = rec.az,
            obs_lat      = rec.obs_lat,
            obs_lon      = rec.obs_lon,
            obs_alt      = rec.obs_alt,
            nblocks      = rec.nblocks,
            nsamples     = rec.nsamples,
            siggen_freq  = rec.siggen_freq,
            siggen_amp   = rec.siggen_amp,
            siggen_rf_on = rec.siggen_rf_on,
        )

    def save(self, filepath):
        """Saves this Spectrum to a .npz file.

        Parameters
        ----------
        filepath : str or Path
            Destination path.
        """
        np.savez(os.fspath(filepath), **self._to_npz_dict())

    @classmethod
    def load(cls, filepath) -> 'Spectrum':
        """Loads a .npz file written by ``save`` and return a Spectrum.

        Parameters
        ----------
        filepath : str or Path
            Path to a .npz file written by ``save``.

        Returns
        -------
        Spectrum

        Raises
        ------
        ValueError
            If required keys are missing or arrays have unexpected shapes.
        """
        with np.load(os.fspath(filepath), allow_pickle=False) as f:
            missing = _REQUIRED_KEYS - f.keys()
            if missing:
                raise ValueError(
                    f'{filepath}: missing required keys: {missing}'
                )

            psd   = f['psd']
            freqs = f['freqs']

            if psd.ndim != 1:
                raise ValueError(
                    f'{filepath}: psd must be 1-D, got shape {psd.shape}'
                )
            if freqs.shape != psd.shape:
                raise ValueError(
                    f'{filepath}: freqs shape {freqs.shape} does not match '
                    f'psd shape {psd.shape}'
                )

            return cls(
                psd          = psd,
                std          = f['std'],
                freqs        = freqs,
                sample_rate  = float(f['sample_rate']),
                center_freq  = float(f['center_freq']),
                gain         = float(f['gain']),
                direct       = bool(f['direct']),
                unix_time    = float(f['unix_time']),
                jd           = float(f['jd']),
                lst          = float(f['lst']),
                alt          = float(f['alt']),
                az           = float(f['az']),
                obs_lat      = float(f['obs_lat']),
                obs_lon      = float(f['obs_lon']),
                obs_alt      = float(f['obs_alt']),
                nblocks      = int(f['nblocks']),
                nsamples     = int(f['nsamples']),
                siggen_freq  = (
                    float(f['siggen_freq']) if 'siggen_freq' in f else None
                ),
                siggen_amp   = (
                    float(f['siggen_amp']) if 'siggen_amp' in f else None
                ),
                siggen_rf_on = (
                    bool(f['siggen_rf_on']) if 'siggen_rf_on' in f else None
                ),
            )

    def _to_npz_dict(self):
        """Converts to dtype-stable kwargs for ``np.savez``."""
        out = dict(
            psd         = self.psd,
            std         = self.std,
            freqs       = self.freqs,
            sample_rate = np.float64(self.sample_rate),
            center_freq = np.float64(self.center_freq),
            gain        = np.float64(self.gain),
            direct      = np.bool_(self.direct),
            unix_time   = np.float64(self.unix_time),
            jd          = np.float64(self.jd),
            lst         = np.float64(self.lst),
            alt         = np.float64(self.alt),
            az          = np.float64(self.az),
            obs_lat     = np.float64(self.obs_lat),
            obs_lon     = np.float64(self.obs_lon),
            obs_alt     = np.float64(self.obs_alt),
            nblocks     = np.int64(self.nblocks),
            nsamples    = np.int64(self.nsamples),
        )
        if self.uses_synth:
            out.update(
                siggen_freq  = np.float64(self.siggen_freq),
                siggen_amp   = np.float64(self.siggen_amp),
                siggen_rf_on = np.bool_(self.siggen_rf_on),
            )
        return out
