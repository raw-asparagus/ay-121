import os
from dataclasses import dataclass

import numpy as np
import ugradio.nch as nch
import ugradio.timing as timing

from ..utils import get_unix_time

_REQUIRED_KEYS = frozenset({
    'data', 'sample_rate', 'center_freq', 'gain', 'direct',
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
_INT8_MIN = np.iinfo(np.int8).min
_INT8_MAX = np.iinfo(np.int8).max


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


def _validate_int8_capture(data: np.ndarray) -> None:
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
    """Unified capture metadata record for both obs and cal files."""

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

        nblocks = _as_scalar('nblocks', self.nblocks, kind='int')
        nsamples = _as_scalar('nsamples', self.nsamples, kind='int')
        if data.shape != (nblocks, nsamples):
            raise ValueError(
                f'data shape {data.shape} inconsistent with '
                f'nblocks={nblocks}, nsamples={nsamples}'
            )

        object.__setattr__(self, 'data', data)
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
        """Builds a Record from hardware state and raw captured data.

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
        Record
        """
        raw = np.asarray(data)
        if raw.ndim != 3 or raw.shape[-1] != 2:
            raise ValueError('data must have shape (nblocks, nsamples, 2)')
        if raw.dtype != np.dtype(np.int8):
            raise ValueError(f'data must be int8, got dtype {raw.dtype}')
        iq = (raw[..., 0].astype(np.float32)
              + 1j * raw[..., 1].astype(np.float32))

        t   = get_unix_time()
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
        """Saves this record to a .npz file.

        Parameters
        ----------
        filepath : str or Path
            Destination path.
        """
        np.savez(os.fspath(filepath), **self._to_npz_dict())

    @classmethod
    def load(cls, filepath):
        """Loads a .npz file and return a Record.

        Parameters
        ----------
        filepath : str or Path
            Path to a .npz file written by ``save``.

        Returns
        -------
        Record

        Raises
        ------
        ValueError
            If required keys are missing, the data array has an unexpected
            shape or dtype, or stored nblocks/nsamples are inconsistent with
            the data array dimensions.
        """
        with np.load(os.fspath(filepath), allow_pickle=False) as f:
            missing = _REQUIRED_KEYS - f.keys()
            if missing:
                raise ValueError(
                    f'{filepath}: missing required keys: {missing}'
                )

            data = f['data']
            nblocks = _as_scalar('nblocks', f['nblocks'], kind='int')
            nsamples = _as_scalar('nsamples', f['nsamples'], kind='int')

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
                siggen_freq  = f['siggen_freq'] if 'siggen_freq' in f else None,
                siggen_amp   = f['siggen_amp'] if 'siggen_amp' in f else None,
                siggen_rf_on = f['siggen_rf_on'] if 'siggen_rf_on' in f else None,
            )

    def _to_npz_dict(self):
        """Converts this record to dtype-stable kwargs for ``np.savez``."""
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
