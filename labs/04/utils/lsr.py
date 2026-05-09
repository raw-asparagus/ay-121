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
