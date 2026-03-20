import numpy as np

# ---------------------------------------------------------------------------
# IAU 1958 / Hipparcos rotation matrix  (equatorial ICRS → galactic)
# ---------------------------------------------------------------------------
_EQ_TO_GAL = np.array([
    [-0.054875539726, -0.873437108010, -0.483834985808],
    [0.494109453312, -0.444829589425, 0.746982251810],
    [-0.867666135858, -0.198076386122, 0.455983795705],
])
_GAL_TO_EQ = _EQ_TO_GAL.T


def _unit_gal(l_deg: float, b_deg: float) -> np.ndarray:
    """Build a galactic Cartesian unit vector."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def _unit_eq(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Build an equatorial Cartesian unit vector."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _unit_altaz(alt_deg: float, az_deg: float) -> np.ndarray:
    """Build a North-East-Up unit vector from altitude and azimuth."""
    alt, az = np.radians(alt_deg), np.radians(az_deg)
    return np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])


def _eq_from_unit(n: np.ndarray) -> tuple[float, float]:
    """Recover equatorial coordinates from a Cartesian unit vector."""
    dec_rad = np.arcsin(np.clip(n[2], -1.0, 1.0))
    ra_rad = np.arctan2(n[1], n[0]) % (2 * np.pi)
    return float(np.degrees(ra_rad)), float(np.degrees(dec_rad))


def _gal_from_unit(n: np.ndarray) -> tuple[float, float]:
    """Recover galactic coordinates from a Cartesian unit vector."""
    b_rad = np.arcsin(np.clip(n[2], -1.0, 1.0))
    l_rad = np.arctan2(n[1], n[0]) % (2 * np.pi)
    return float(np.degrees(l_rad)), float(np.degrees(b_rad))


def _altaz_from_unit(n: np.ndarray) -> tuple[float, float]:
    """Recover altitude and azimuth from a North-East-Up unit vector."""
    alt_rad = np.arcsin(np.clip(n[2], -1.0, 1.0))
    az_rad = np.arctan2(n[1], n[0]) % (2 * np.pi)
    return float(np.degrees(alt_rad)), float(np.degrees(az_rad))


def galactic_to_equatorial_matrix() -> np.ndarray:
    """Return the fixed galactic-to-equatorial rotation matrix."""
    return _GAL_TO_EQ.copy()


def equatorial_to_altaz_matrix(lst_rad: float, lat_deg: float) -> np.ndarray:
    """Return the equatorial-to-horizontal rotation matrix.

    Parameters
    ----------
    lst_rad : float
        Local sidereal time in radians.
    lat_deg : float
        Observer latitude in degrees.

    Returns
    -------
    matrix : np.ndarray
        A ``(3, 3)`` rotation matrix that maps ICRS equatorial Cartesian unit
        vectors to North-East-Up horizontal vectors.
    """
    lat = np.radians(lat_deg)
    sl, cl = np.sin(lst_rad), np.cos(lst_rad)
    sp, cp = np.sin(lat), np.cos(lat)

    rz = np.array([
        [cl, sl, 0.0],
        [-sl, cl, 0.0],
        [0.0, 0.0, 1.0],
    ])

    m_lat = np.array([
        [sp, 0.0, -cp],
        [0.0, 1.0, 0.0],
        [cp, 0.0, sp],
    ])

    return m_lat @ rz


def galactic_to_equatorial(l_deg: float, b_deg: float) -> tuple[float, float]:
    """Convert galactic coordinates to ICRS equatorial coordinates.

    Parameters
    ----------
    l_deg : float
        Galactic longitude in degrees.
    b_deg : float
        Galactic latitude in degrees.

    Returns
    -------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    """
    return _eq_from_unit(_GAL_TO_EQ @ _unit_gal(l_deg, b_deg))


def equatorial_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert ICRS equatorial coordinates to galactic coordinates.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.

    Returns
    -------
    l_deg : float
        Galactic longitude in degrees.
    b_deg : float
        Galactic latitude in degrees.
    """
    return _gal_from_unit(_EQ_TO_GAL @ _unit_eq(ra_deg, dec_deg))


def equatorial_to_altaz(
    ra_deg: float, dec_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert ICRS equatorial coordinates to horizontal coordinates.

    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    lst_rad : float
        Local sidereal time in radians.
    lat_deg : float
        Observer latitude in degrees.

    Returns
    -------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees, measured from north through east.
    """
    matrix = equatorial_to_altaz_matrix(lst_rad, lat_deg)
    return _altaz_from_unit(matrix @ _unit_eq(ra_deg, dec_deg))


def altaz_to_equatorial(
    alt_deg: float, az_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert horizontal coordinates to ICRS equatorial coordinates.

    Parameters
    ----------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees, measured from north through east.
    lst_rad : float
        Local sidereal time in radians.
    lat_deg : float
        Observer latitude in degrees.

    Returns
    -------
    ra_deg : float
        Right ascension in degrees.
    dec_deg : float
        Declination in degrees.
    """
    matrix = equatorial_to_altaz_matrix(lst_rad, lat_deg)
    return _eq_from_unit(matrix.T @ _unit_altaz(alt_deg, az_deg))


def galactic_to_altaz(
    l_deg: float, b_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert galactic coordinates to horizontal coordinates.

    Parameters
    ----------
    l_deg : float
        Galactic longitude in degrees.
    b_deg : float
        Galactic latitude in degrees.
    lst_rad : float
        Local sidereal time in radians.
    lat_deg : float
        Observer latitude in degrees.

    Returns
    -------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees, measured from north through east.
    """
    ra_deg, dec_deg = galactic_to_equatorial(l_deg, b_deg)
    return equatorial_to_altaz(ra_deg, dec_deg, lst_rad, lat_deg)


def altaz_to_galactic(
    alt_deg: float, az_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert horizontal coordinates to galactic coordinates.

    Parameters
    ----------
    alt_deg : float
        Altitude in degrees.
    az_deg : float
        Azimuth in degrees, measured from north through east.
    lst_rad : float
        Local sidereal time in radians.
    lat_deg : float
        Observer latitude in degrees.

    Returns
    -------
    l_deg : float
        Galactic longitude in degrees.
    b_deg : float
        Galactic latitude in degrees.
    """
    ra_deg, dec_deg = altaz_to_equatorial(alt_deg, az_deg, lst_rad, lat_deg)
    return equatorial_to_galactic(ra_deg, dec_deg)
