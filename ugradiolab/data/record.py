from dataclasses import dataclass

import ntplib
import numpy as np
import ugradio.nch as nch
import ugradio.timing as timing


def _get_unix_time() -> float:
    try:
        c = ntplib.NTPClient()
        return c.request('pool.ntp.org', version=3).tx_time
    except ntplib.NTPException:
        return timing.unix_time()


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
            complex128 with shape (nblocks, nsamples).
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
        iq = raw[..., 0].astype(np.float64) + 1j * raw[..., 1].astype(np.float64)

        t = _get_unix_time()
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
        np.savez(filepath, **self._to_npz_dict())

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
        """
        with np.load(filepath, allow_pickle=False) as f:
            return cls(
                data=f['data'],
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
            data=self.data,
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
        if (
            self.siggen_freq is not None
            and self.siggen_amp is not None
            and self.siggen_rf_on is not None
        ):
            out.update(
                siggen_freq=np.float64(self.siggen_freq),
                siggen_amp=np.float64(self.siggen_amp),
                siggen_rf_on=np.bool_(self.siggen_rf_on),
            )
        return out
