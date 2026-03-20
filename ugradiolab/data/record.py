import os
from dataclasses import dataclass

import numpy as np
import ugradio.nch as nch
import ugradio.timing as timing

from ..io.clock import get_unix_time
from .schema import (
    COMMON_REQUIRED_METADATA_KEYS,
    as_scalar,
    missing_required_keys,
    optional_npz_value,
    set_common_metadata_fields,
)

_REQUIRED_KEYS = frozenset({'data'}) | COMMON_REQUIRED_METADATA_KEYS
_INT8_MIN = np.iinfo(np.int8).min
_INT8_MAX = np.iinfo(np.int8).max


def _validate_int8_capture(data: np.ndarray) -> None:
    """Validate that a complex capture can round-trip through int8 storage.

    Notes
    -----
    This helper does not modify ``data``. It raises ``ValueError`` if either
    the real or imaginary component contains non-integer values or falls
    outside the representable int8 range.

    Raises
    ------
    ValueError
        If either component of ``data`` is not integer-valued or exceeds the
        int8 range.
    """
    for label, values in (('real', data.real), ('imag', data.imag)):
        rounded = np.rint(values)
        if np.any(values != rounded):
            raise ValueError(f'data {label} component must be integer-valued.')
        if np.any((rounded < _INT8_MIN) | (rounded > _INT8_MAX)):
            raise ValueError(
                f'data {label} component must stay within int8 range '
                f'[{_INT8_MIN}, {_INT8_MAX}].'
            )


@dataclass(frozen=True)
class Record:
    """Unified capture metadata record for both observation and calibration files.

    Attributes
    ----------
    data : np.ndarray
        Complex I/Q samples with shape ``(nblocks, nsamples)``.
    sample_rate : float
        Sample rate in Hz.
    center_freq : float
        Center frequency in Hz.
    gain : float
        SDR gain setting.
    direct : bool
        Direct sampling mode flag.
    unix_time : float
        Unix timestamp of the capture.
    jd : float
        Julian Date at capture time.
    lst : float
        Local sidereal time at capture time.
    alt : float
        Telescope altitude in degrees.
    az : float
        Telescope azimuth in degrees.
    obs_lat : float
        Observer latitude in degrees.
    obs_lon : float
        Observer longitude in degrees.
    obs_alt : float
        Observer altitude in meters.
    nblocks : int
        Number of captured blocks.
    nsamples : int
        Number of samples per block.
    siggen_freq : float or None
        Signal-generator frequency in Hz, if present.
    siggen_amp : float or None
        Signal-generator amplitude in dBm, if present.
    siggen_rf_on : bool or None
        Signal-generator RF state, if present.
    """

    data: np.ndarray
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
        """Validate shapes, dtypes, and shared metadata after construction.

        Notes
        -----
        Validation failures are not handled locally. All detected schema issues
        are reported as ``ValueError``.

        Raises
        ------
        ValueError
            If ``data`` is not a finite numeric ``(nblocks, nsamples)`` array,
            if it cannot be represented as int8-backed complex samples, or if
            any shared metadata field fails scalar validation.
        """
        data = np.asarray(self.data)
        if data.ndim != 2:
            raise ValueError(
                f'data must have shape (nblocks, nsamples), got {data.shape}'
            )
        if not (
            np.issubdtype(data.dtype, np.complexfloating)
            or np.issubdtype(data.dtype, np.floating)
            or np.issubdtype(data.dtype, np.integer)
        ):
            raise ValueError(
                f'data must be numeric, got dtype {data.dtype}'
            )
        data = np.asarray(data, dtype=np.complex64)
        if not (np.isfinite(data.real).all() and np.isfinite(data.imag).all()):
            raise ValueError('data must be finite.')
        _validate_int8_capture(data)

        nblocks = as_scalar('nblocks', self.nblocks, kind='int')
        nsamples = as_scalar('nsamples', self.nsamples, kind='int')
        if data.shape != (nblocks, nsamples):
            raise ValueError(
                f'data shape {data.shape} inconsistent with '
                f'nblocks={nblocks}, nsamples={nsamples}'
            )

        object.__setattr__(self, 'data', data)
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

    @classmethod
    def from_sdr(
        cls,
        data,
        sdr,
        alt_deg,
        az_deg,
        lat     = nch.lat,
        lon     = nch.lon,
        obs_alt = nch.alt,
        synth   = None,
    ):
        """Build a ``Record`` from raw SDR output and current hardware state.

        Parameters
        ----------
        data : array-like
            Raw I/Q samples from the SDR, shape (nblocks, nsamples, 2)
            where the last axis is [I, Q] as int8.
        sdr : ugradio.sdr.SDR
            Configured SDR instance.
        alt_deg : float
            Telescope altitude in degrees.
        az_deg : float
            Telescope azimuth in degrees.
        lat : float
            Observer latitude in degrees.
        lon : float
            Observer longitude in degrees.
        obs_alt : float
            Observer altitude in metres.
        synth : SignalGenerator, optional
            Connected signal generator.

        Returns
        -------
        record : Record
            Sanitized record containing the captured complex samples and
            current hardware metadata.

        Raises
        ------
        ValueError
            If ``data`` does not have shape ``(nblocks, nsamples, 2)``, is not
            ``int8``, or if the resulting record fails metadata validation.
        """
        raw = np.asarray(data)
        if raw.ndim != 3 or raw.shape[-1] != 2:
            raise ValueError('data must have shape (nblocks, nsamples, 2)')
        if raw.dtype != np.dtype(np.int8):
            raise ValueError(f'data must be int8, got dtype {raw.dtype}')
        iq = (raw[..., 0].astype(np.float32)
              + 1j * raw[..., 1].astype(np.float32))

        t   = get_unix_time(local=True)
        jd  = timing.julian_date(t)

        kwargs = dict(
            data        = iq,
            sample_rate = sdr.get_sample_rate(),
            center_freq = sdr.get_center_freq(),
            gain        = sdr.get_gain(),
            direct      = sdr.direct,
            unix_time   = t,
            jd          = jd,
            lst         = timing.lst(jd, lon),
            alt         = alt_deg,
            az          = az_deg,
            obs_lat     = lat,
            obs_lon     = lon,
            obs_alt     = obs_alt,
            nblocks     = iq.shape[0],
            nsamples    = iq.shape[1],
        )
        if synth is not None:
            kwargs.update(
                siggen_freq  = synth.get_freq(),
                siggen_amp   = synth.get_ampl(),
                siggen_rf_on = synth.rf_state(),
            )
        return cls(**kwargs)

    def save(self, filepath):
        """Save this record to a ``.npz`` file.

        Parameters
        ----------
        filepath : str or Path
            Destination path.

        Returns
        -------
        None
            The record is written to ``filepath``.

        Raises
        ------
        OSError
            If the destination cannot be opened or written.
        """
        np.savez(os.fspath(filepath), **self._to_npz_dict())

    @classmethod
    def load(cls, filepath):
        """Load a ``Record`` from a ``.npz`` file.

        Parameters
        ----------
        filepath : str or Path
            Path to a .npz file written by ``save``.

        Returns
        -------
        record : Record
            Reconstructed record instance.

        Raises
        ------
        ValueError
            If required keys are missing, the data array has an unexpected
            shape or dtype, or stored nblocks/nsamples are inconsistent with
            the data array dimensions.
        OSError
            If ``filepath`` cannot be opened.
        """
        with np.load(os.fspath(filepath), allow_pickle=False) as f:
            missing = missing_required_keys(f.keys(), _REQUIRED_KEYS)
            if missing:
                raise ValueError(
                    f'{filepath}: missing required keys: {missing}'
                )

            data = f['data']
            nblocks = as_scalar('nblocks', f['nblocks'], kind='int')
            nsamples = as_scalar('nsamples', f['nsamples'], kind='int')

            if data.ndim != 3 or data.shape[-1] != 2:
                raise ValueError(
                    f'{filepath}: data must have shape '
                    f'(nblocks, nsamples, 2), got {data.shape}'
                )
            if data.dtype != np.dtype(np.int8):
                raise ValueError(
                    f'{filepath}: data must be int8, got dtype {data.dtype}'
                )
            if data.shape[:2] != (nblocks, nsamples):
                raise ValueError(
                    f'{filepath}: data shape {data.shape[:2]} inconsistent '
                    f'with nblocks={nblocks}, nsamples={nsamples}'
                )

            iq = (data[..., 0].astype(np.float32)
                  + 1j * data[..., 1].astype(np.float32))

            return cls(
                data         = iq,
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
                nblocks      = nblocks,
                nsamples     = nsamples,
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
            data        = np.stack([
                self.data.real.astype(np.int8),
                self.data.imag.astype(np.int8)
            ], axis=-1),
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
