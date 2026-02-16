"""Standardized .npz schemas for SDR data collection.

Two schemas:
- Calibration (cal): signal-generator-driven measurements
- Observation (obs): sky observations with pointing metadata
"""

from dataclasses import dataclass

import numpy as np
import ugradio.timing as timing
import ugradio.nch as nch

@dataclass(frozen=True)
class CaptureRecord:
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

    def to_npz_dict(self):
        """Convert this record to dtype-stable kwargs for ``np.savez``."""
        out = dict(
            data=self.data.astype(np.int8),
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
        if self.siggen_freq is not None:
            out['siggen_freq'] = np.float64(self.siggen_freq)
        if self.siggen_amp is not None:
            out['siggen_amp'] = np.float64(self.siggen_amp)
        if self.siggen_rf_on is not None:
            out['siggen_rf_on'] = np.bool_(self.siggen_rf_on)
        return out


def build_record(data, sdr, alt_deg, az_deg, lat=nch.lat, lon=nch.lon,
                 observer_alt=nch.alt, synth=None, unix_time=None):
    """Build a unified capture record from hardware state + raw data."""
    data = np.asarray(data, dtype=np.int8)
    if data.ndim < 2:
        raise ValueError('data must have shape (nblocks, nsamples[, ...])')

    t = timing.unix_time() if unix_time is None else float(unix_time)
    jd = timing.julian_date(t)
    lst = timing.lst(jd, lon)

    kwargs = dict(
        data=data,
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
        nblocks=data.shape[0],
        nsamples=data.shape[1],
    )
    if synth is not None:
        kwargs.update(
            siggen_freq=synth.get_freq(),
            siggen_amp=synth.get_ampl(),
            siggen_rf_on=synth.rf_state(),
        )
    return CaptureRecord(**kwargs)


def save_record(filepath, record):
    """Save a ``CaptureRecord`` to a .npz file."""
    np.savez(filepath, **record.to_npz_dict())


def save_cal(filepath, data, sdr, synth, alt_deg=0.0, az_deg=0.0,
             lat=nch.lat, lon=nch.lon, observer_alt=nch.alt):
    """Save a calibration capture to .npz."""
    record = build_record(
        data, sdr, alt_deg=alt_deg, az_deg=az_deg, lat=lat, lon=lon,
        observer_alt=observer_alt, synth=synth
    )
    save_record(filepath, record)


def save_obs(filepath, data, sdr, alt_deg, az_deg,
             lat=nch.lat, lon=nch.lon, observer_alt=nch.alt):
    """Save a sky observation capture to .npz."""
    record = build_record(
        data, sdr, alt_deg=alt_deg, az_deg=az_deg, lat=lat, lon=lon,
        observer_alt=observer_alt, synth=None
    )
    save_record(filepath, record)


def load(filepath):
    """Load a .npz data file.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.

    Returns
    -------
    np.lib.npyio.NpzFile
        Dict-like object; access fields by key (e.g., f['data'], f['alt']).
    """
    return np.load(filepath, allow_pickle=False)
