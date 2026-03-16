"""Interferometry utilities for Lab 3 (NCH X-band interferometer).

Provides solar system body pointing and geometric delay computation for use
with ugradio.interf.Interferometer and ugradio.interf_delay.DelayClient.

Baseline constants (BASELINE_EW_M, BASELINE_NS_M) and the hardware delay
limit (DELAY_MAX_NS) are NOT defined here — they are fit from fringe data and
must be supplied explicitly by the caller.
"""

import numpy as np
import ugradio.coord as coord
import ugradio.nch as nch
import ugradio.timing as timing

from ..utils import get_unix_time

_C_M_PER_NS = 0.299792458  # speed of light in m/ns


# ---------------------------------------------------------------------------
# Body Pointing
# ---------------------------------------------------------------------------

def compute_sun_pointing(
    lat: float = nch.lat,
    lon: float = nch.lon,
    obs_alt: float = nch.alt,
) -> tuple[float, float, float, float, float]:
    """Return (alt, az, ra, dec, jd) for the Sun at the current time.

    Parameters
    ----------
    lat, lon : float
        Observer latitude/longitude in degrees. Defaults to NCH.
    obs_alt : float
        Observer altitude in metres. Defaults to NCH.

    Returns
    -------
    (alt_deg, az_deg, ra_deg, dec_deg, jd)
    """
    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)
    ra, dec = coord.sunpos(jd)
    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd


def compute_moon_pointing(
    lat: float = nch.lat,
    lon: float = nch.lon,
    obs_alt: float = nch.alt,
) -> tuple[float, float, float, float, float]:
    """Return (alt, az, ra, dec, jd) for the Moon at the current time.

    Parameters
    ----------
    lat, lon : float
        Observer latitude/longitude in degrees. Defaults to NCH.
    obs_alt : float
        Observer altitude in metres. Defaults to NCH.

    Returns
    -------
    (alt_deg, az_deg, ra_deg, dec_deg, jd)
    """
    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)
    ra, dec = coord.moonpos(jd, lat, lon, obs_alt)
    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd


# ---------------------------------------------------------------------------
# Geometric Delay
# ---------------------------------------------------------------------------

def geometric_delay_ns(
    alt_deg: float,
    az_deg: float,
    baseline_ew_m: float,
    baseline_ns_m: float,
    lat: float = nch.lat,
    delay_max_ns: float | None = None,
) -> float:
    """Compute geometric path-length delay (ns) for the east antenna relative
    to the west antenna given a pointing direction (alt, az).

    Uses the formula:
        τ_g = (B_ew/c) cos(dec) sin(ha) + (B_ns/c) [sin(lat) cos(dec) cos(ha)
              - cos(lat) sin(dec)]

    where ha and dec are derived from (alt, az) at the observer latitude.

    Parameters
    ----------
    alt_deg, az_deg : float
        Pointing direction in degrees (horizontal coordinates).
    baseline_ew_m : float
        East-west baseline length in metres (fit from fringe data).
    baseline_ns_m : float
        North-south baseline length in metres (fit from fringe data).
    lat : float
        Observer latitude in degrees. Defaults to NCH.
    delay_max_ns : float or None
        If given, clip the result to ±delay_max_ns before returning.
        Omit until the hardware limit is confirmed from DelayClient docs.

    Returns
    -------
    float
        Geometric delay in nanoseconds.
    """
    alt = np.radians(alt_deg)
    az  = np.radians(az_deg)
    phi = np.radians(lat)

    # Convert (alt, az) → (ha, dec) at observer latitude
    sin_dec = np.sin(phi) * np.sin(alt) + np.cos(phi) * np.cos(alt) * np.cos(az)
    sin_dec = np.clip(sin_dec, -1.0, 1.0)
    dec = np.arcsin(sin_dec)
    cos_dec = np.cos(dec)

    if cos_dec < 1e-9:
        ha = 0.0
    else:
        sin_ha = -np.cos(alt) * np.sin(az) / cos_dec
        cos_ha = (np.sin(alt) - np.sin(phi) * sin_dec) / (np.cos(phi) * cos_dec + 1e-30)
        ha = np.arctan2(sin_ha, cos_ha)

    tau_ew = (baseline_ew_m / _C_M_PER_NS) * cos_dec * np.sin(ha)
    tau_ns = (baseline_ns_m / _C_M_PER_NS) * (
        np.sin(phi) * cos_dec * np.cos(ha) - np.cos(phi) * sin_dec
    )
    tau = tau_ew + tau_ns

    if delay_max_ns is not None:
        tau = np.clip(tau, -delay_max_ns, delay_max_ns)

    return float(tau)
