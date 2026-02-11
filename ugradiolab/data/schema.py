"""Standardized .npz schemas for SDR data collection.

Two schemas:
- Calibration (cal): signal-generator-driven measurements
- Observation (obs): sky observations with pointing metadata
"""

import numpy as np
import ugradio.timing as timing
import ugradio.nch as nch

def save_cal(filepath, data, sdr, synth, alt_deg=0.0, az_deg=0.0,
             lat=nch.lat, lon=nch.lon, observer_alt=nch.alt):
    """Save a calibration capture to .npz.

    Parameters
    ----------
    filepath : str
        Output file path (extension added automatically if missing).
    data : np.ndarray
        Raw int8 samples, shape (nblocks, nsamples) or (nblocks, nsamples, 2).
    sdr : ugradio.sdr.SDR
        The SDR object used for capture.
    synth : SignalGenerator
        The signal generator object (queried for current state).
    alt_deg : float
        Telescope altitude/elevation in degrees.
    az_deg : float
        Telescope azimuth in degrees.
    lat : float
        Observer latitude in degrees. Default: nch.lat.
    lon : float
        Observer longitude in degrees. Default: nch.lon.
    observer_alt : float
        Observer altitude in meters. Default: nch.alt.
    """
    t = timing.unix_time()
    jd = timing.julian_date(t)
    lst = timing.lst(jd, lon)

    nblocks = data.shape[0]
    nsamples = data.shape[1]

    np.savez(filepath,
             data=data.astype(np.int8),
             sample_rate=np.float64(sdr.get_sample_rate()),
             center_freq=np.float64(sdr.get_center_freq()),
             gain=np.float64(sdr.get_gain()),
             direct=np.bool_(sdr.direct),
             siggen_freq=np.float64(synth.get_freq()),
             siggen_amp=np.float64(synth.get_ampl()),
             siggen_rf_on=np.bool_(synth.rf_state()),
             unix_time=np.float64(t),
             jd=np.float64(jd),
             lst=np.float64(lst),
             alt=np.float64(alt_deg),
             az=np.float64(az_deg),
             observer_lat=np.float64(lat),
             observer_lon=np.float64(lon),
             observer_alt=np.float64(observer_alt),
             nblocks=np.int64(nblocks),
             nsamples=np.int64(nsamples))


def save_obs(filepath, data, sdr, alt_deg, az_deg,
             lat=nch.lat, lon=nch.lon, observer_alt=nch.alt):
    """Save a sky observation capture to .npz.

    Parameters
    ----------
    filepath : str
        Output file path (extension added automatically if missing).
    data : np.ndarray
        Raw int8 samples, shape (nblocks, nsamples) or (nblocks, nsamples, 2).
    sdr : ugradio.sdr.SDR
        The SDR object used for capture.
    alt_deg : float
        Telescope altitude/elevation in degrees.
    az_deg : float
        Telescope azimuth in degrees.
    lat : float
        Observer latitude in degrees. Default: nch.lat.
    lon : float
        Observer longitude in degrees. Default: nch.lon.
    observer_alt : float
        Observer altitude in meters. Default: nch.alt.
    """
    t = timing.unix_time()
    jd = timing.julian_date(t)
    lst = timing.lst(jd, lon)

    nblocks = data.shape[0]
    nsamples = data.shape[1]

    np.savez(filepath,
             data=data.astype(np.int8),
             sample_rate=np.float64(sdr.get_sample_rate()),
             center_freq=np.float64(sdr.get_center_freq()),
             gain=np.float64(sdr.get_gain()),
             direct=np.bool_(sdr.direct),
             unix_time=np.float64(t),
             jd=np.float64(jd),
             lst=np.float64(lst),
             alt=np.float64(alt_deg),
             az=np.float64(az_deg),
             observer_lat=np.float64(lat),
             observer_lon=np.float64(lon),
             observer_alt=np.float64(observer_alt),
             nblocks=np.int64(nblocks),
             nsamples=np.int64(nsamples))


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