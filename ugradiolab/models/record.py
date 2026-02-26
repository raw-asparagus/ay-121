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
        if data.ndim != 3 or data.shape[-1] != 2:
            raise ValueError('data must have shape (nblocks, nsamples, 2)')
        iq = (data[..., 0].astype(np.float32)
              + 1j * data[..., 1].astype(np.float32))

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
        with np.load(filepath, allow_pickle=False) as f:
            missing = _REQUIRED_KEYS - f.keys()
            if missing:
                raise ValueError(
                    f'{filepath}: missing required keys: {missing}'
                )

            data     = f['data']
            nblocks  = int(f['nblocks'])
            nsamples = int(f['nsamples'])

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
                nblocks      = nblocks,
                nsamples     = nsamples,
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
