"""LSR velocity correction for Leuschner Observatory."""

from __future__ import annotations

import numpy as np
import astropy.coordinates as ac
import astropy.units as u
from astropy.time import Time

# Leuschner Observatory
LEO_LOCATION = ac.EarthLocation(
    lat=37.9183 * u.deg, lon=-122.1067 * u.deg, height=304 * u.m,
)

# Standard kinematic LSR: solar motion 20 km/s toward (18h, +30 deg) B1900
_APEX = ac.SkyCoord(
    ra=270 * u.deg, dec=30 * u.deg,
    frame='fk4', equinox='B1900.0',
).transform_to('icrs')
V_SUN_KMS = 20.0


def vlsr_correction(ra_deg: float, dec_deg: float,
                    unix_time: float) -> float:
    """Return v_corr [km/s] such that v_LSR = v_topo + v_corr.

    Combines the heliocentric correction (Earth's orbital + rotational
    motion) with the solar motion projection onto the line of sight.

    Parameters
    ----------
    ra_deg, dec_deg : float
        ICRS coordinates of the source in degrees.
    unix_time : float
        Observation time as a Unix timestamp.

    Returns
    -------
    float
        Velocity correction in km/s.
    """
    t = Time(unix_time, format='unix')
    sc = ac.SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    v_helio = sc.radial_velocity_correction(
        kind='heliocentric', obstime=t, location=LEO_LOCATION,
    ).to(u.km / u.s).value
    v_sun = V_SUN_KMS * np.cos(sc.separation(_APEX).rad)
    return v_helio + v_sun


def resample_records_to_lsr(
    records: list[dict],
    *,
    nfft: int,
    sample_rate_hz: float,
    hi_rest_mhz: float = 1420.40575,
) -> None:
    """Resample each record's ``corr00`` / ``corr11`` onto the LSR frame in place.

    Computes ``v_corr`` per dump from the dump's ``(ra, dec, time)`` once
    (vectorised SkyCoord + ``radial_velocity_correction``), converts to a
    channel shift via ``dvch_kms``, and applies linear interpolation on
    the 1024-channel axis so the resulting spectra share a common LSR
    grid even though each dump has a different per-dump ``v_corr``.

    Side effects: sets ``r['v_corr']`` on each record and overwrites
    ``r['corr00']`` / ``r['corr11']`` with their LSR-resampled copies.
    Off-grid channels are NaN-filled.  The channel shift is LO-agnostic
    because ``df_mhz`` is the same for both LOs in a freq-switched pair.

    Parameters
    ----------
    records : list of dict
        Each record must have ``ra``, ``dec``, ``time``, ``corr00``,
        ``corr11`` keys.  Modified in place.
    nfft : int
        Number of channels in the spectra.
    sample_rate_hz : float
        SDR sample rate in Hz.
    hi_rest_mhz : float
        HI rest frequency used to convert ``df`` -> ``dv``.
    """
    if not records:
        return
    ra_arr  = np.array([r['ra']  for r in records])
    dec_arr = np.array([r['dec'] for r in records])
    t_arr   = np.array([r['time'] for r in records])
    sc      = ac.SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame='icrs')
    t_astro = Time(t_arr, format='unix')
    v_helio = sc.radial_velocity_correction(
        kind='heliocentric', obstime=t_astro, location=LEO_LOCATION,
    ).to(u.km / u.s).value
    v_sun = V_SUN_KMS * np.cos(sc.separation(_APEX).rad)
    v_corr = v_helio + v_sun

    dvch_kms = (sample_rate_hz / nfft / 1e6) * 299792.458 / hi_rest_mhz
    channels = np.arange(nfft, dtype=np.float64)
    for r, vc in zip(records, v_corr):
        r['v_corr'] = float(vc)
        shift_ch = vc / dvch_kms
        r['corr00'] = np.interp(channels + shift_ch, channels, r['corr00'],
                                left=np.nan, right=np.nan)
        r['corr11'] = np.interp(channels + shift_ch, channels, r['corr11'],
                                left=np.nan, right=np.nan)
