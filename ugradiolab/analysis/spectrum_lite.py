import os
from dataclasses import dataclass

import numpy as np

from ._base import SpectrumBase


_RECORD_REQUIRED_KEYS = frozenset({
    'data', 'sample_rate', 'center_freq', 'gain', 'direct',
    'unix_time', 'jd', 'lst', 'alt', 'az',
    'observer_lat', 'observer_lon', 'observer_alt',
    'nblocks', 'nsamples',
})

_REQUIRED_KEYS = frozenset({
    'psd', 'std', 'freqs',
    'sample_rate', 'center_freq', 'gain', 'direct',
    'unix_time', 'jd', 'lst', 'alt', 'az',
    'observer_lat', 'observer_lon', 'observer_alt',
    'nblocks', 'nsamples',
})


@dataclass(frozen=True)
class SpectrumLite(SpectrumBase):
    """Integrated power spectrum with observation metadata but without raw I/Q data.

    Analogous to combining Spectrum and Record, but omits the raw data array
    to reduce memory and storage footprint for large captures.

    Attributes
    ----------
    psd : np.ndarray, shape (nfrequencies,)
        Normalised mean power spectrum across blocks, DC-centred.
        Each bin is ``mean(|FFT|²) / nsamples²`` so that ``sum(psd)`` equals
        ``mean(|IQ|²)`` (Parseval's theorem), independent of FFT length.
    std : float
        Standard error of the mean PSD estimator: mean across frequency bins
        of ``std(|FFT|²/nsamples², axis=0) / √nblocks``.  Scales as
        ``1/√nblocks``.  Same units as ``psd``.
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
    observer_lat : float
        Observer latitude in degrees.
    observer_lon : float
        Observer longitude in degrees.
    observer_alt : float
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
    std: float
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
    observer_lat: float
    observer_lon: float
    observer_alt: float
    nblocks: int
    nsamples: int
    siggen_freq: float | None = None
    siggen_amp: float | None = None
    siggen_rf_on: bool | None = None

    @property
    def uses_synth(self) -> bool:
        """True if all signal generator fields are populated."""
        return (
            self.siggen_freq is not None
            and self.siggen_amp is not None
            and self.siggen_rf_on is not None
        )

    @classmethod
    def from_data(cls, filepath) -> 'SpectrumLite':
        """Load a Record .npz file, compute the power spectrum, and discard raw data.

        Parameters
        ----------
        filepath : str or Path
            Path to a Record .npz file written by ``Record.save``.

        Returns
        -------
        SpectrumLite

        Raises
        ------
        ValueError
            If required Record keys are missing or the data array is invalid.
        """
        with np.load(os.fspath(filepath), allow_pickle=False) as f:
            missing = _RECORD_REQUIRED_KEYS - f.keys()
            if missing:
                raise ValueError(f'{filepath}: missing required keys: {missing}')

            raw = f['data']
            nblocks_stored = int(f['nblocks'])
            nsamples_stored = int(f['nsamples'])

            if raw.ndim != 3 or raw.shape[-1] != 2:
                raise ValueError(
                    f'{filepath}: data must have shape (nblocks, nsamples, 2), '
                    f'got {raw.shape}'
                )
            if raw.dtype != np.dtype(np.int8):
                raise ValueError(
                    f'{filepath}: data must be int8, got dtype {raw.dtype}'
                )
            if raw.shape[:2] != (nblocks_stored, nsamples_stored):
                raise ValueError(
                    f'{filepath}: data shape {raw.shape[:2]} inconsistent with '
                    f'nblocks={nblocks_stored}, nsamples={nsamples_stored}'
                )

            nblocks, nsamples = raw.shape[:2]
            data = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)
            sample_rate = float(f['sample_rate'])
            center_freq = float(f['center_freq'])

            data = data - data.mean(axis=1, keepdims=True)
            block_psds = (
                np.abs(np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)) ** 2
                / nsamples ** 2
            )
            freqs = np.fft.fftshift(
                np.fft.fftfreq(nsamples, d=1.0 / sample_rate)
            ) + center_freq
            psd = np.mean(block_psds, axis=0)
            std = float(np.mean(np.std(block_psds, axis=0))) / np.sqrt(nblocks)

            return cls(
                psd=psd,
                std=std,
                freqs=freqs,
                sample_rate=sample_rate,
                center_freq=center_freq,
                gain=float(f['gain']),
                direct=bool(f['direct']),
                unix_time=float(f['unix_time']),
                jd=float(f['jd']),
                lst=float(f['lst']),
                alt=float(f['alt']),
                az=float(f['az']),
                observer_lat=float(f['observer_lat']),
                observer_lon=float(f['observer_lon']),
                observer_alt=float(f['observer_alt']),
                nblocks=nblocks,
                nsamples=nsamples,
                siggen_freq=float(f['siggen_freq']) if 'siggen_freq' in f else None,
                siggen_amp=float(f['siggen_amp']) if 'siggen_amp' in f else None,
                siggen_rf_on=bool(f['siggen_rf_on']) if 'siggen_rf_on' in f else None,
            )

    def save(self, filepath):
        """Save this SpectrumLite to a .npz file.

        Parameters
        ----------
        filepath : str or Path
            Destination path.
        """
        np.savez(os.fspath(filepath), **self._to_npz_dict())

    @classmethod
    def load(cls, filepath) -> 'SpectrumLite':
        """Load a .npz file written by ``save`` and return a SpectrumLite.

        Parameters
        ----------
        filepath : str or Path
            Path to a .npz file written by ``save``.

        Returns
        -------
        SpectrumLite

        Raises
        ------
        ValueError
            If required keys are missing or arrays have unexpected shapes.
        """
        with np.load(os.fspath(filepath), allow_pickle=False) as f:
            missing = _REQUIRED_KEYS - f.keys()
            if missing:
                raise ValueError(f'{filepath}: missing required keys: {missing}')

            psd = f['psd']
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
                psd=psd,
                std=float(f['std']),
                freqs=freqs,
                sample_rate=float(f['sample_rate']),
                center_freq=float(f['center_freq']),
                gain=float(f['gain']),
                direct=bool(f['direct']),
                unix_time=float(f['unix_time']),
                jd=float(f['jd']),
                lst=float(f['lst']),
                alt=float(f['alt']),
                az=float(f['az']),
                observer_lat=float(f['observer_lat']),
                observer_lon=float(f['observer_lon']),
                observer_alt=float(f['observer_alt']),
                nblocks=int(f['nblocks']),
                nsamples=int(f['nsamples']),
                siggen_freq=float(f['siggen_freq']) if 'siggen_freq' in f else None,
                siggen_amp=float(f['siggen_amp']) if 'siggen_amp' in f else None,
                siggen_rf_on=bool(f['siggen_rf_on']) if 'siggen_rf_on' in f else None,
            )

    def _to_npz_dict(self):
        """Convert to dtype-stable kwargs for ``np.savez``."""
        out = dict(
            psd=self.psd,
            std=np.float64(self.std),
            freqs=self.freqs,
            sample_rate=np.float64(self.sample_rate),
            center_freq=np.float64(self.center_freq),
            gain=np.float64(self.gain),
            direct=np.bool_(self.direct),
            unix_time=np.float64(self.unix_time),
            jd=np.float64(self.jd),
            lst=np.float64(self.lst),
            alt=np.float64(self.alt),
            az=np.float64(self.az),
            observer_lat=np.float64(self.observer_lat),
            observer_lon=np.float64(self.observer_lon),
            observer_alt=np.float64(self.observer_alt),
            nblocks=np.int64(self.nblocks),
            nsamples=np.int64(self.nsamples),
        )
        if self.uses_synth:
            out.update(
                siggen_freq=np.float64(self.siggen_freq),
                siggen_amp=np.float64(self.siggen_amp),
                siggen_rf_on=np.bool_(self.siggen_rf_on),
            )
        return out