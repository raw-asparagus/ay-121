"""Interferometry pointing utilities for Lab 3 (NCH X-band interferometer)."""

import ugradio.nch as nch
import ugradio.timing as timing

from ..utils import get_unix_time


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
    import ugradio.coord as coord

    unix_t = get_unix_time(skip_net=True)
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
    import ugradio.coord as coord

    unix_t = get_unix_time(skip_net=True)
    jd = timing.julian_date(unix_t)
    ra, dec = coord.moonpos(jd, lat, lon, obs_alt)
    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd


def compute_radec_pointing(
    ra_deg: float,
    dec_deg: float,
    lat: float = nch.lat,
    lon: float = nch.lon,
    obs_alt: float = nch.alt,
) -> tuple[float, float, float]:
    """Return (alt, az, jd) for a fixed J2000 (RA, Dec) source at the current time.

    Parameters
    ----------
    ra_deg, dec_deg : float
        J2000 equatorial coordinates in degrees.
    lat, lon : float
        Observer latitude/longitude in degrees. Defaults to NCH.
    obs_alt : float
        Observer altitude in metres. Defaults to NCH.

    Returns
    -------
    (alt_deg, az_deg, jd)
    """
    import ugradio.coord as coord

    unix_t = get_unix_time(skip_net=True)
    jd = timing.julian_date(unix_t)
    alt, az = coord.get_altaz(ra_deg, dec_deg, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, jd
