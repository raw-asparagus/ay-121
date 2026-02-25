import os
from dataclasses import dataclass

import numpy as np

_REQUIRED_KEYS = frozenset({
    'data', 'sample_rate', 'center_freq', 'gain', 'direct',
    'unix_time', 'jd', 'lst', 'alt', 'az',
    'observer_lat', 'observer_lon', 'observer_alt',
    'nblocks', 'nsamples',
})
import ugradio.nch as nch
import ugradio.timing as timing

from ..utils import get_unix_time


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
    def from_sdr(
        cls,
        data,
        sdr,
        alt_deg,
        az_deg,
        lat=nch.lat,
        lon=nch.lon,
        observer_alt=nch.alt,
        synth=None,
    ):
        """Build a Record from hardware state and raw captured data.

        Parameters
        ----------
        data : array-like
            Raw I/Q samples from the SDR, shape (nblocks, nsamples, 2)
            where the last axis is [I, Q] as int8.  Stored internally as
            complex64 with shape (nblocks, nsamples).
        sdr : ugradio.sdr.SDR
            Configured SDR instance; queried for sample_rate, center_freq, gain.
        alt_deg : float
            Telescope altitude in degrees.
        az_deg : float
            Telescope azimuth in degrees.
        lat : float
            Observer latitude in degrees.
        lon : float
            Observer longitude in degrees.
        observer_alt : float
            Observer altitude in metres.
        synth : SignalGenerator, optional
            Connected signal generator; if provided, siggen fields are populated.

        Returns
        -------
        Record
        """
        raw = np.asarray(data, dtype=np.int8)
        if raw.ndim != 3 or raw.shape[-1] != 2:
            raise ValueError(
                'data must have shape (nblocks, nsamples, 2)'
            )
        iq = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)

        t = get_unix_time()
        jd = timing.julian_date(t)
        lst = timing.lst(jd, lon)

        kwargs = dict(
            data=iq,
            sample_rate=sdr.get_sample_rate(),
            center_freq=sdr.get_center_freq(),
            gain=sdr.get_gain(),
            direct=sdr.direct,
            unix_time=t,
            jd=jd,
            lst=lst,
            alt=alt_deg,
            az=az_deg,
            observer_lat=lat,
            observer_lon=lon,
            observer_alt=observer_alt,
            nblocks=iq.shape[0],
            nsamples=iq.shape[1],
        )
        if synth is not None:
            kwargs.update(
                siggen_freq=synth.get_freq(),
                siggen_amp=synth.get_ampl(),
                siggen_rf_on=synth.rf_state(),
            )
        return cls(**kwargs)

    def save(self, filepath):
        """Save this record to a .npz file.

        Parameters
        ----------
        filepath : str or Path
            Destination path.
        """
        np.savez(os.fspath(filepath), **self._to_npz_dict())

    @classmethod
    def load(cls, filepath):
        """Load a .npz file and return a Record.

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
        with np.load(filepath, allow_pickle=False) as f:
            missing = _REQUIRED_KEYS - f.keys()
            if missing:
                raise ValueError(
                    f'{filepath}: missing required keys: {missing}'
                )

            raw = f['data']
            nblocks = int(f['nblocks'])
            nsamples = int(f['nsamples'])

            if raw.ndim != 3 or raw.shape[-1] != 2:
                raise ValueError(
                    f'{filepath}: data must have shape (nblocks, nsamples, 2), '
                    f'got {raw.shape}'
                )
            if raw.dtype != np.dtype(np.int8):
                raise ValueError(
                    f'{filepath}: data must be int8, got dtype {raw.dtype}'
                )
            if raw.shape[:2] != (nblocks, nsamples):
                raise ValueError(
                    f'{filepath}: data shape {raw.shape[:2]} inconsistent with '
                    f'nblocks={nblocks}, nsamples={nsamples}'
                )

            data = raw[..., 0].astype(np.float32) + 1j * raw[..., 1].astype(np.float32)

            return cls(
                data=data,
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
                nblocks=nblocks,
                nsamples=nsamples,
                siggen_freq=(
                    float(f['siggen_freq']) if 'siggen_freq' in f else None
                ),
                siggen_amp=(
                    float(f['siggen_amp']) if 'siggen_amp' in f else None
                ),
                siggen_rf_on=(
                    bool(f['siggen_rf_on']) if 'siggen_rf_on' in f else None
                ),
            )

    def _to_npz_dict(self):
        """Convert this record to dtype-stable kwargs for ``np.savez``."""
        out = dict(
            data=np.stack(
                [self.data.real.astype(np.int8), self.data.imag.astype(np.int8)],
                axis=-1,
            ),
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
