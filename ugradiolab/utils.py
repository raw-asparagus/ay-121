import ntplib
import astropy.coordinates as ac
import astropy.units as u
import ugradio.coord as coord
import ugradio.nch as nch
import ugradio.timing as timing


def get_unix_time() -> float:
    """Return current Unix time from NTP, falling back to system time."""
    try:
        c = ntplib.NTPClient()
        return c.request('pool.ntp.org', version=3).tx_time
    except ntplib.NTPException:
        print("Unable to connect to NTP! Using system time.")
        return timing.unix_time()


def compute_pointing(
    gal_l: float,
    gal_b: float,
    lat: float = nch.lat,
    lon: float = nch.lon,
    observer_alt: float = nch.alt,
) -> tuple[float, float, float, float, float]:
    """Return (alt_deg, az_deg, ra_deg, dec_deg, jd) for a galactic coordinate now.

    Parameters
    ----------
    gal_l : float
        Galactic longitude in degrees.
    gal_b : float
        Galactic latitude in degrees.
    lat : float
        Observer latitude in degrees. Defaults to NCH.
    lon : float
        Observer longitude in degrees. Defaults to NCH.
    observer_alt : float
        Observer altitude in metres. Defaults to NCH.
    """
    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)

    gc = ac.SkyCoord(l=gal_l * u.deg, b=gal_b * u.deg, frame='galactic')
    ra = gc.icrs.ra.deg
    dec = gc.icrs.dec.deg

    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=observer_alt)
    return alt, az, ra, dec, jd
