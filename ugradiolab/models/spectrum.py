import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from .record import Record

SmoothMethod = Literal['gaussian', 'savgol', 'boxcar']
FrequencyAxis = Literal['absolute', 'baseband']
PlotScale = Literal['linear', 'log']

_REQUIRED_KEYS = frozenset({
    'psd', 'std', 'freqs',
    'sample_rate', 'center_freq', 'gain', 'direct',
    'unix_time', 'jd', 'lst', 'alt', 'az',
    'obs_lat', 'obs_lon', 'obs_alt',
    'nblocks', 'nsamples',
})

_SCALAR_FLOAT_FIELDS = (
    'sample_rate',
    'center_freq',
    'gain',
    'unix_time',
    'jd',
    'lst',
    'alt',
    'az',
    'obs_lat',
    'obs_lon',
    'obs_alt',
)
_POSITIVE_FLOAT_FIELDS = frozenset({'sample_rate'})
_OPTIONAL_FLOAT_FIELDS = ('siggen_freq', 'siggen_amp')
_OPTIONAL_BOOL_FIELDS = ('siggen_rf_on',)


def _as_scalar(name: str, value, *, kind: str) -> float | int | bool:
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f'{name} must be a scalar, got shape {arr.shape}')
    item = arr.item()
    if kind == 'float':
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f'{name} must be a real scalar, got boolean {item!r}')
        try:
            out = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a real scalar, got {item!r}') from exc
        if not np.isfinite(out):
            raise ValueError(f'{name} must be finite, got {out!r}')
        return out
    if kind == 'int':
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f'{name} must be a positive integer, got boolean {item!r}')
        if isinstance(item, (int, np.integer)):
            out = int(item)
        else:
            try:
                numeric = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'{name} must be a positive integer, got {item!r}'
                ) from exc
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(
                    f'{name} must be a positive integer, got {item!r}'
                )
            out = int(numeric)
        if out <= 0:
            raise ValueError(f'{name} must be > 0, got {out!r}')
        return out
    if kind == 'bool':
        if isinstance(item, (bool, np.bool_)):
            return bool(item)
        if isinstance(item, (int, np.integer)) and item in (0, 1):
            return bool(item)
        raise ValueError(f'{name} must be boolean, got {item!r}')
    raise ValueError(f'Unknown scalar kind {kind!r}')


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

    def __post_init__(self):
        psd = np.asarray(self.psd, dtype=float)
        std = np.asarray(self.std, dtype=float)
        freqs = np.asarray(self.freqs, dtype=float)
        if psd.ndim != 1:
            raise ValueError(f'psd must be 1-D, got shape {psd.shape}')
        if std.ndim != 1:
            raise ValueError(f'std must be 1-D, got shape {std.shape}')
        if freqs.ndim != 1:
            raise ValueError(f'freqs must be 1-D, got shape {freqs.shape}')
        if std.shape != psd.shape:
            raise ValueError(
                f'std shape {std.shape} does not match psd shape {psd.shape}'
            )
        if freqs.shape != psd.shape:
            raise ValueError(
                f'freqs shape {freqs.shape} does not match psd shape {psd.shape}'
            )
        if not np.isfinite(psd).all():
            raise ValueError('psd must be finite.')
        if not np.isfinite(std).all():
            raise ValueError('std must be finite.')
        if not np.isfinite(freqs).all():
            raise ValueError('freqs must be finite.')
        if np.any(std < 0):
            raise ValueError('std must be non-negative.')

        nblocks = _as_scalar('nblocks', self.nblocks, kind='int')
        nsamples = _as_scalar('nsamples', self.nsamples, kind='int')
        if psd.size != nsamples:
            raise ValueError(
                f'psd length {psd.size} inconsistent with nsamples={nsamples}'
            )

        object.__setattr__(self, 'psd', np.asarray(psd, dtype=np.float64))
        object.__setattr__(self, 'std', np.asarray(std, dtype=np.float64))
        object.__setattr__(self, 'freqs', np.asarray(freqs, dtype=np.float64))
        object.__setattr__(self, 'nblocks', nblocks)
        object.__setattr__(self, 'nsamples', nsamples)
        object.__setattr__(self, 'direct', _as_scalar('direct', self.direct, kind='bool'))

        for name in _SCALAR_FLOAT_FIELDS:
            value = _as_scalar(name, getattr(self, name), kind='float')
            if name in _POSITIVE_FLOAT_FIELDS and value <= 0:
                raise ValueError(f'{name} must be > 0, got {value!r}')
            object.__setattr__(
                self,
                name,
                value,
            )

        for name in _OPTIONAL_FLOAT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _as_scalar(name, value, kind='float'))

        for name in _OPTIONAL_BOOL_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(self, name, _as_scalar(name, value, kind='bool'))

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
    def bin_width(self) -> float:
        """Frequency resolution per bin in Hz (sample_rate / nsamples)."""
        return self.sample_rate / self.nsamples

    @property
    def total_power(self) -> float:
        """Total integrated power (mean square of time samples).

        Computed as ``sum(psd)`` where ``psd`` is per-bin (not per-Hz), so no
        bin-width factor is needed.  By Parseval's theorem this equals
        ``mean(|x[n]|²)`` and is independent of ``nsamples``.
        """
        return float(np.sum(self.psd))

    @property
    def total_power_db(self) -> float:
        """Total integrated power in dB."""
        power = self.total_power
        if not np.isfinite(power) or power <= 0:
            raise ValueError('total_power must be finite and > 0 for dB conversion.')
        return float(10.0 * np.log10(power))

    @property
    def total_power_sigma(self) -> float:
        """Uncertainty on the total integrated power."""
        return float(np.sqrt(np.sum(np.square(self.std))))

    def bin_at(self, freq_hz: float) -> int:
        """Index of the frequency bin closest to freq_hz (in Hz)."""
        return int(np.argmin(np.abs(self.freqs - freq_hz)))

    def frequency_axis_mhz(
            self,
            mode: FrequencyAxis = 'absolute',
    ) -> np.ndarray:
        """Frequency axis in MHz.

        Parameters
        ----------
        mode : {'absolute', 'baseband'}
            ``'absolute'`` returns the sky frequency axis. ``'baseband'``
            returns the offset from the local oscillator centre frequency.
        """
        if mode == 'absolute':
            return self.freqs_mhz
        if mode == 'baseband':
            return (self.freqs - self.center_freq) / 1e6
        raise ValueError(
            f'Unknown frequency axis mode {mode!r}. '
            f"Choose 'absolute' or 'baseband'."
        )

    def velocity_axis_kms(
            self,
            rest_freq_hz: float,
            velocity_shift_kms: float = 0.0,
    ) -> np.ndarray:
        """Radio-definition Doppler velocity axis in km/s."""
        c_light_kms = 2.99792458e5
        return (
            c_light_kms * (rest_freq_hz - self.freqs) / rest_freq_hz
            + velocity_shift_kms
        )

    def mask_dc_bin(
            self,
            values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a copy with the DC-centre bin masked as ``NaN``."""
        masked = np.array(
            self.psd if values is None else values,
            dtype=float,
            copy=True,
        )
        i0 = self.bin_at(self.center_freq)
        masked[max(0, i0):min(masked.size, i0 + 1)] = np.nan
        return masked

    def psd_values(
            self,
            *,
            smooth_kwargs: dict | None = None,
            mask_dc: bool = False,
    ) -> np.ndarray:
        """Return raw or smoothed PSD values as a float array copy."""
        if smooth_kwargs is None:
            values = np.array(self.psd, dtype=float, copy=True)
        else:
            values = np.array(self.smooth(**smooth_kwargs), dtype=float, copy=True)
        return self.mask_dc_bin(values) if mask_dc else values

    def std_bounds(
            self,
            values: np.ndarray | None = None,
            *,
            floor: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper one-sigma envelopes for ``values``."""
        center = self.psd_values() if values is None else np.array(
            values,
            dtype=float,
            copy=True,
        )
        lo = center - self.std
        hi = center + self.std
        if floor is not None:
            lo = np.clip(lo, floor, None)
            hi = np.clip(hi, floor, None)
        return lo, hi

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

    def ratio_to(
            self,
            other: 'Spectrum',
            *,
            smooth_kwargs: dict | None = None,
    ) -> np.ndarray:
        """Return the channel-wise ratio ``self / other``."""
        if self.psd.shape != other.psd.shape:
            raise ValueError(
                'Spectrum ratios require matching PSD shapes, got '
                f'{self.psd.shape} and {other.psd.shape}.'
            )
        num = self.psd_values(smooth_kwargs=smooth_kwargs)
        den = other.psd_values(smooth_kwargs=smooth_kwargs)
        with np.errstate(divide='ignore', invalid='ignore'):
            return num / den

    def ratio_std_to(self, other: 'Spectrum') -> np.ndarray:
        """Propagate raw-PSD SEM values into the ratio uncertainty."""
        if self.psd.shape != other.psd.shape:
            raise ValueError(
                'Spectrum ratios require matching PSD shapes, got '
                f'{self.psd.shape} and {other.psd.shape}.'
            )
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = self.psd / other.psd
            return ratio * np.sqrt(
                (self.std / self.psd) ** 2
                + (other.std / other.psd) ** 2
            )

    @classmethod
    def from_record(cls, record: Record) -> 'Spectrum':
        """Computes a Spectrum from a sanitized Record."""
        if not isinstance(record, Record):
            raise TypeError(f'record must be a Record, got {type(record)!r}')
        nblocks, nsamples = record.data.shape
        data = record.data - record.data.mean(axis=1, keepdims=True)
        block_psds = np.abs(np.fft.fftshift(
            np.fft.fft(data, axis=1),
            axes=1
        )) ** 2 / nsamples ** 2
        return cls(
            psd          = np.mean(block_psds, axis=0),
            std          = np.std(block_psds, axis=0) / np.sqrt(nblocks),
            freqs        = np.fft.fftshift(
                np.fft.fftfreq(nsamples, d=1.0 / record.sample_rate)
            ) + record.center_freq,
            sample_rate  = record.sample_rate,
            center_freq  = record.center_freq,
            gain         = record.gain,
            direct       = record.direct,
            unix_time    = record.unix_time,
            jd           = record.jd,
            lst          = record.lst,
            alt          = record.alt,
            az           = record.az,
            obs_lat      = record.obs_lat,
            obs_lon      = record.obs_lon,
            obs_alt      = record.obs_alt,
            nblocks      = record.nblocks,
            nsamples     = record.nsamples,
            siggen_freq  = record.siggen_freq,
            siggen_amp   = record.siggen_amp,
            siggen_rf_on = record.siggen_rf_on,
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
        return cls.from_record(Record.load(filepath))

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

            psd = f['psd']
            std = f['std']
            freqs = f['freqs']

            if psd.ndim != 1:
                raise ValueError(
                    f'{filepath}: psd must be 1-D, got shape {psd.shape}'
                )
            if std.ndim != 1:
                raise ValueError(
                    f'{filepath}: std must be 1-D, got shape {std.shape}'
                )
            if freqs.ndim != 1:
                raise ValueError(
                    f'{filepath}: freqs must be 1-D, got shape {freqs.shape}'
                )
            if std.shape != psd.shape:
                raise ValueError(
                    f'{filepath}: std shape {std.shape} does not match '
                    f'psd shape {psd.shape}'
                )
            if freqs.shape != psd.shape:
                raise ValueError(
                    f'{filepath}: freqs shape {freqs.shape} does not match '
                    f'psd shape {psd.shape}'
                )

            return cls(
                psd          = psd,
                std          = std,
                freqs        = freqs,
                sample_rate  = f['sample_rate'],
                center_freq  = f['center_freq'],
                gain         = f['gain'],
                direct       = f['direct'],
                unix_time    = f['unix_time'],
                jd           = f['jd'],
                lst          = f['lst'],
                alt          = f['alt'],
                az           = f['az'],
                obs_lat      = f['obs_lat'],
                obs_lon      = f['obs_lon'],
                obs_alt      = f['obs_alt'],
                nblocks      = f['nblocks'],
                nsamples     = f['nsamples'],
                siggen_freq  = f['siggen_freq'] if 'siggen_freq' in f else None,
                siggen_amp   = f['siggen_amp'] if 'siggen_amp' in f else None,
                siggen_rf_on = f['siggen_rf_on'] if 'siggen_rf_on' in f else None,
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
