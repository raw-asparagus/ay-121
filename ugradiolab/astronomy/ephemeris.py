import astropy.coordinates as ac
import astropy.units as u
import ugradio.timing as timing

from ..io.clock import get_unix_time
from .site import NCH_LAT_DEG, NCH_LON_DEG, NCH_OBS_ALT_M


def compute_sun_pointing(
    lat: float = NCH_LAT_DEG,
    lon: float = NCH_LON_DEG,
    obs_alt: float = NCH_OBS_ALT_M,
) -> tuple[float, float, float, float, float]:
    """Return the Sun pointing solution at the current time.

    Parameters
    ----------
    lat : float, optional
        Observer latitude in degrees.
    lon : float, optional
        Observer longitude in degrees.
    obs_alt : float, optional
        Observer altitude in meters.

    Returns
    -------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees.
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    jd : float
        Julian Date used for the coordinate evaluation.
    """
    import ugradio.coord as coord

    unix_t = get_unix_time(local=True)
    jd = timing.julian_date(unix_t)
    ra, dec = coord.sunpos(jd)
    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd


def compute_moon_pointing(
    lat: float = NCH_LAT_DEG,
    lon: float = NCH_LON_DEG,
    obs_alt: float = NCH_OBS_ALT_M,
) -> tuple[float, float, float, float, float]:
    """Return the Moon pointing solution at the current time.

    Parameters
    ----------
    lat : float, optional
        Observer latitude in degrees.
    lon : float, optional
        Observer longitude in degrees.
    obs_alt : float, optional
        Observer altitude in meters.

    Returns
    -------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees.
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    jd : float
        Julian Date used for the coordinate evaluation.
    """
    import ugradio.coord as coord

    unix_t = get_unix_time(local=True)
    jd = timing.julian_date(unix_t)
    ra, dec = coord.moonpos(jd, lat, lon, obs_alt)
    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd


def compute_radec_pointing(
    ra_deg: float,
    dec_deg: float,
    lat: float = NCH_LAT_DEG,
    lon: float = NCH_LON_DEG,
    obs_alt: float = NCH_OBS_ALT_M,
) -> tuple[float, float, float]:
    """Return the horizontal pointing for a fixed equatorial target.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    lat : float, optional
        Observer latitude in degrees.
    lon : float, optional
        Observer longitude in degrees.
    obs_alt : float, optional
        Observer altitude in meters.

    Returns
    -------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees.
    jd : float
        Julian Date used for the coordinate evaluation.
    """
    import ugradio.coord as coord

    unix_t = get_unix_time(local=True)
    jd = timing.julian_date(unix_t)
    alt, az = coord.get_altaz(ra_deg, dec_deg, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, jd


def compute_gal_pointing(
    gal_l: float,
    gal_b: float,
    lat: float = NCH_LAT_DEG,
    lon: float = NCH_LON_DEG,
    obs_alt: float = NCH_OBS_ALT_M,
) -> tuple[float, float, float, float, float]:
    """Return the pointing solution for a galactic target.

    Parameters
    ----------
    gal_l : float
        Galactic longitude in degrees.
    gal_b : float
        Galactic latitude in degrees.
    lat : float, optional
        Observer latitude in degrees.
    lon : float, optional
        Observer longitude in degrees.
    obs_alt : float, optional
        Observer altitude in meters.

    Returns
    -------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees.
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    jd : float
        Julian Date used for the coordinate evaluation.
    """
    import ugradio.coord as coord

    unix_t = get_unix_time(local=True)
    jd = timing.julian_date(unix_t)

    gc = ac.SkyCoord(l=gal_l * u.deg, b=gal_b * u.deg, frame="galactic")
    ra, dec = gc.icrs.ra.deg, gc.icrs.dec.deg

    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd
