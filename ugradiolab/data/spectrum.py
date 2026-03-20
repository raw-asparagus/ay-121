import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from .record import Record
from .schema import (
    COMMON_REQUIRED_METADATA_KEYS,
    as_scalar,
    missing_required_keys,
    optional_npz_value,
    set_common_metadata_fields,
)

SmoothMethod = Literal['gaussian', 'savgol', 'boxcar']
FrequencyAxis = Literal['absolute', 'baseband']
PlotScale = Literal['linear', 'log']

_REQUIRED_KEYS = frozenset({'psd', 'std', 'freqs'}) | COMMON_REQUIRED_METADATA_KEYS


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
        """Validate array shapes and normalize shared metadata after construction.

        Notes
        -----
        Validation failures are not handled locally. All detected schema issues
        are reported as ``ValueError``.

        Raises
        ------
        ValueError
            If the spectrum arrays are not finite one-dimensional arrays with
            matching shapes, if ``std`` contains negative values, if ``psd``
            length disagrees with ``nsamples``, or if shared metadata validation
            fails.
        """
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

        nblocks = as_scalar('nblocks', self.nblocks, kind='int')
        nsamples = as_scalar('nsamples', self.nsamples, kind='int')
        if psd.size != nsamples:
            raise ValueError(
                f'psd length {psd.size} inconsistent with nsamples={nsamples}'
            )

        object.__setattr__(self, 'psd', np.asarray(psd, dtype=np.float64))
        object.__setattr__(self, 'std', np.asarray(std, dtype=np.float64))
        object.__setattr__(self, 'freqs', np.asarray(freqs, dtype=np.float64))
        object.__setattr__(self, 'nblocks', nblocks)
        object.__setattr__(self, 'nsamples', nsamples)
        set_common_metadata_fields(self)

    @property
    def uses_synth(self) -> bool:
        """Whether all signal-generator metadata fields are populated.

        Returns
        -------
        uses_synth : bool
            ``True`` when the frequency, amplitude, and RF-state fields are all
            present.
        """
        return (
            self.siggen_freq is not None
            and self.siggen_amp is not None
            and self.siggen_rf_on is not None
        )

    @property
    def freqs_mhz(self) -> np.ndarray:
        """Return the absolute frequency axis in MHz.

        Returns
        -------
        freqs_mhz : np.ndarray
            Frequency axis converted from Hz to MHz.
        """
        return self.freqs / 1e6

    @property
    def bin_width(self) -> float:
        """Return the frequency spacing between adjacent bins.

        Returns
        -------
        bin_width : float
            Frequency resolution per bin in Hz.
        """
        return self.sample_rate / self.nsamples

    @property
    def total_power(self) -> float:
        """Return the total integrated power.

        Returns
        -------
        total_power : float
            Sum of the per-bin PSD values. Because ``psd`` is stored per bin
            rather than per Hz, no bin-width factor is required.
        """
        return float(np.sum(self.psd))

    @property
    def total_power_db(self) -> float:
        """Return the total integrated power in dB.

        Returns
        -------
        total_power_db : float
            ``10 * log10(total_power)``.

        Raises
        ------
        ValueError
            If ``total_power`` is not finite and strictly positive.
        """
        power = self.total_power
        if not np.isfinite(power) or power <= 0:
            raise ValueError('total_power must be finite and > 0 for dB conversion.')
        return float(10.0 * np.log10(power))

    @property
    def total_power_sigma(self) -> float:
        """Return the propagated uncertainty on the total power.

        Returns
        -------
        total_power_sigma : float
            Quadrature sum of the per-bin standard errors.
        """
        return float(np.sqrt(np.sum(np.square(self.std))))

    def bin_at(self, freq_hz: float) -> int:
        """Return the index of the closest frequency bin.

        Parameters
        ----------
        freq_hz : float
            Target frequency in Hz.

        Returns
        -------
        index : int
            Index of the bin nearest to ``freq_hz``.
        """
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

        Returns
        -------
        axis_mhz : np.ndarray
            Frequency axis in MHz in the requested frame.

        Raises
        ------
        ValueError
            If ``mode`` is not one of ``'absolute'`` or ``'baseband'``.
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
        """Return the radio-definition Doppler velocity axis.

        Parameters
        ----------
        rest_freq_hz : float
            Rest frequency in Hz.
        velocity_shift_kms : float, optional
            Constant velocity offset to add to the axis, in km/s.

        Returns
        -------
        velocity_kms : np.ndarray
            Doppler velocity axis in km/s.
        """
        c_light_kms = 2.99792458e5
        return (
            c_light_kms * (rest_freq_hz - self.freqs) / rest_freq_hz
            + velocity_shift_kms
        )

    def mask_dc_bin(
            self,
            values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an array with the DC bin masked as ``NaN``.

        Parameters
        ----------
        values : np.ndarray or None, optional
            Values to mask. If omitted, ``self.psd`` is copied and masked.

        Returns
        -------
        masked : np.ndarray
            Float array copy with the DC-centered bin replaced by ``NaN``.
        """
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
        """Return raw or smoothed PSD values as a float array copy.

        Parameters
        ----------
        smooth_kwargs : dict or None, optional
            Keyword arguments forwarded to ``smooth``. If omitted, no smoothing
            is applied.
        mask_dc : bool, optional
            If ``True``, replace the DC-centered bin with ``NaN``.

        Returns
        -------
        values : np.ndarray
            PSD values after optional smoothing and masking.
        """
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
        """Return lower and upper one-sigma envelopes.

        Parameters
        ----------
        values : np.ndarray or None, optional
            Central values for the interval. If omitted, raw PSD values are
            used.
        floor : float or None, optional
            Minimum allowed value for both envelopes after propagation.

        Returns
        -------
        lo : np.ndarray
            Lower one-sigma envelope.
        hi : np.ndarray
            Upper one-sigma envelope.
        """
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
        """Return a smoothed copy of ``psd``.

        Parameters
        ----------
        method : {'gaussian', 'savgol', 'boxcar'}
            Smoothing algorithm to apply.
        **kwargs
            gaussian : sigma (float, default 32) — kernel width in bins.
            savgol   : window_length (int, default 129),
                       polyorder (int, default 3).
            boxcar   : M (int, default 64) — number of bins to average.

        Returns
        -------
        smoothed : np.ndarray
            Smoothed PSD values with the same shape as ``psd``.

        Raises
        ------
        ValueError
            If ``method`` is unknown.
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
        """Return the channel-wise PSD ratio ``self / other``.

        Parameters
        ----------
        other : Spectrum
            Denominator spectrum.
        smooth_kwargs : dict or None, optional
            Optional smoothing configuration applied to both spectra before the
            ratio is formed.

        Returns
        -------
        ratio : np.ndarray
            Channel-wise ratio of the selected PSD values.

        Raises
        ------
        ValueError
            If the spectra do not have matching PSD shapes.
        """
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
        """Propagate PSD standard errors into the ratio uncertainty.

        Parameters
        ----------
        other : Spectrum
            Denominator spectrum.

        Returns
        -------
        ratio_std : np.ndarray
            Propagated one-sigma uncertainty on ``self.psd / other.psd``.

        Raises
        ------
        ValueError
            If the spectra do not have matching PSD shapes.
        """
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
        """Compute a ``Spectrum`` from a sanitized ``Record``.

        Parameters
        ----------
        record : Record
            Input record containing time-domain samples and metadata.

        Returns
        -------
        spectrum : Spectrum
            Integrated power spectrum derived from ``record``.

        Raises
        ------
        TypeError
            If ``record`` is not a ``Record`` instance.
        ValueError
            If the derived arrays or metadata fail ``Spectrum`` validation.
        """
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
        """Compute a ``Spectrum`` from a serialized ``Record`` file.

        Parameters
        ----------
        filepath : str or Path
            Path to a Record .npz file written by ``Record.save``.

        Returns
        -------
        spectrum : Spectrum
            Spectrum derived from the serialized record.

        Raises
        ------
        ValueError
            If the record file is malformed or the derived spectrum fails
            validation.
        OSError
            If ``filepath`` cannot be opened.
        """
        return cls.from_record(Record.load(filepath))

    def save(self, filepath):
        """Save this spectrum to a ``.npz`` file.

        Parameters
        ----------
        filepath : str or Path
            Destination path.

        Returns
        -------
        None
            The spectrum is written to ``filepath``.

        Raises
        ------
        OSError
            If the destination cannot be opened or written.
        """
        np.savez(os.fspath(filepath), **self._to_npz_dict())

    @classmethod
    def load(cls, filepath) -> 'Spectrum':
        """Load a ``Spectrum`` from a ``.npz`` file.

        Parameters
        ----------
        filepath : str or Path
            Path to a .npz file written by ``save``.

        Returns
        -------
        spectrum : Spectrum
            Reconstructed spectrum instance.

        Raises
        ------
        ValueError
            If required keys are missing or arrays have unexpected shapes.
        OSError
            If ``filepath`` cannot be opened.
        """
        with np.load(os.fspath(filepath), allow_pickle=False) as f:
            missing = missing_required_keys(f.keys(), _REQUIRED_KEYS)
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
                siggen_freq  = optional_npz_value(f, 'siggen_freq'),
                siggen_amp   = optional_npz_value(f, 'siggen_amp'),
                siggen_rf_on = optional_npz_value(f, 'siggen_rf_on'),
            )

    def _to_npz_dict(self):
        """Build dtype-stable keyword arguments for ``np.savez``.

        Notes
        -----
        This helper relies on prior validation in ``__post_init__`` and does
        not perform additional error handling. Any unexpected NumPy conversion
        failure propagates to the caller.
        """
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
