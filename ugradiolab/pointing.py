from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ugradio.coord as coord
import ugradio.nch as nch
import ugradio.timing as timing
import astropy.coordinates as ac
import astropy.units as u

from .utils import get_unix_time

# ---------------------------------------------------------------------------
# IAU 1958 / Hipparcos rotation matrix  (equatorial ICRS → galactic)
# ---------------------------------------------------------------------------
_EQ_TO_GAL = np.array([
    [-0.054875539726, -0.873437108010, -0.483834985808],
    [ 0.494109453312, -0.444829589425,  0.746982251810],
    [-0.867666135858, -0.198076386122,  0.455983795705],
])
_GAL_TO_EQ = _EQ_TO_GAL.T


# ---------------------------------------------------------------------------
# Unit-vector helpers
# ---------------------------------------------------------------------------

def _unit_gal(l_deg: float, b_deg: float) -> np.ndarray:
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def _unit_eq(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _unit_altaz(alt_deg: float, az_deg: float) -> np.ndarray:
    """North-East-Up unit vector from altitude and azimuth (N→E convention)."""
    alt, az = np.radians(alt_deg), np.radians(az_deg)
    return np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])


def _eq_from_unit(n: np.ndarray) -> tuple[float, float]:
    dec_rad = np.arcsin(np.clip(n[2], -1.0, 1.0))
    ra_rad = np.arctan2(n[1], n[0]) % (2 * np.pi)
    return float(np.degrees(ra_rad)), float(np.degrees(dec_rad))


def _gal_from_unit(n: np.ndarray) -> tuple[float, float]:
    b_rad = np.arcsin(np.clip(n[2], -1.0, 1.0))
    l_rad = np.arctan2(n[1], n[0]) % (2 * np.pi)
    return float(np.degrees(l_rad)), float(np.degrees(b_rad))


def _altaz_from_unit(n: np.ndarray) -> tuple[float, float]:
    """Return (alt_deg, az_deg) from a North-East-Up unit vector."""
    alt_rad = np.arcsin(np.clip(n[2], -1.0, 1.0))
    az_rad = np.arctan2(n[1], n[0]) % (2 * np.pi)
    return float(np.degrees(alt_rad)), float(np.degrees(az_rad))


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _gmst_rad(jd: float) -> float:
    """Greenwich Mean Sidereal Time in radians (IAU 1982)."""
    D = jd - 2451545.0
    gmst_hours = (18.697374558 + 24.06570982441908 * D) % 24.0
    return gmst_hours * np.pi / 12.0


def _lst_rad(jd: float, lon_deg: float) -> float:
    """Local Sidereal Time in radians (East-positive longitude)."""
    return (_gmst_rad(jd) + np.radians(lon_deg)) % (2 * np.pi)


# ---------------------------------------------------------------------------
# Matrix-returning functions
# ---------------------------------------------------------------------------

def galactic_to_equatorial_matrix() -> np.ndarray:
    """Return the 3×3 rotation matrix from galactic to ICRS equatorial unit vectors."""
    return _GAL_TO_EQ.copy()


def equatorial_to_altaz_matrix(lst_rad: float, lat_deg: float) -> np.ndarray:
    """Return the 3×3 rotation matrix from ICRS equatorial to North-East-Up alt/az.

    Pipeline: M_lat @ Rz(-LST), where Rz(-LST) converts RA→HA and M_lat
    converts the HA-Dec frame to North-East-Up horizontal coordinates.
    """
    lat = np.radians(lat_deg)
    sl, cl = np.sin(lst_rad), np.cos(lst_rad)
    sp, cp = np.sin(lat), np.cos(lat)

    # Rz(-LST): rotates equatorial unit vector so x-axis points toward HA=0
    rz = np.array([
        [ cl,  sl, 0.0],
        [-sl,  cl, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # M_lat: HA-Dec → North-East-Up
    # [N, E, U] = M_lat @ [cos(dec)cos(H), -cos(dec)sin(H), sin(dec)]
    m_lat = np.array([
        [sp,  0.0, -cp],
        [0.0, 1.0,  0.0],
        [cp,  0.0,  sp],
    ])

    return m_lat @ rz


# ---------------------------------------------------------------------------
# 6 pairwise coordinate conversions
# ---------------------------------------------------------------------------

def galactic_to_equatorial(l_deg: float, b_deg: float) -> tuple[float, float]:
    """Convert galactic (l, b) to ICRS equatorial (RA, Dec), all in degrees."""
    return _eq_from_unit(_GAL_TO_EQ @ _unit_gal(l_deg, b_deg))


def equatorial_to_galactic(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert ICRS equatorial (RA, Dec) to galactic (l, b), all in degrees."""
    return _gal_from_unit(_EQ_TO_GAL @ _unit_eq(ra_deg, dec_deg))


def equatorial_to_altaz(
    ra_deg: float, dec_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert ICRS equatorial (RA, Dec) to horizontal (alt, az), all in degrees."""
    M = equatorial_to_altaz_matrix(lst_rad, lat_deg)
    return _altaz_from_unit(M @ _unit_eq(ra_deg, dec_deg))


def altaz_to_equatorial(
    alt_deg: float, az_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert horizontal (alt, az) to ICRS equatorial (RA, Dec), all in degrees."""
    M = equatorial_to_altaz_matrix(lst_rad, lat_deg)
    return _eq_from_unit(M.T @ _unit_altaz(alt_deg, az_deg))


def galactic_to_altaz(
    l_deg: float, b_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert galactic (l, b) to horizontal (alt, az), all in degrees."""
    ra_deg, dec_deg = galactic_to_equatorial(l_deg, b_deg)
    return equatorial_to_altaz(ra_deg, dec_deg, lst_rad, lat_deg)


def altaz_to_galactic(
    alt_deg: float, az_deg: float, lst_rad: float, lat_deg: float
) -> tuple[float, float]:
    """Convert horizontal (alt, az) to galactic (l, b), all in degrees."""
    ra_deg, dec_deg = altaz_to_equatorial(alt_deg, az_deg, lst_rad, lat_deg)
    return equatorial_to_galactic(ra_deg, dec_deg)


# ---------------------------------------------------------------------------
# High-level pointing
# ---------------------------------------------------------------------------

def compute_pointing_matrix(
    gal_l: float,
    gal_b: float,
    lat: float = nch.lat,
    lon: float = nch.lon,
    obs_alt: float = nch.alt,
) -> tuple[float, float, float, float, float]:
    """Compute pointing using the explicit rotation-matrix backend.

    Returns (alt_deg, az_deg, ra_deg, dec_deg, jd).
    """
    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)
    lst = _lst_rad(jd, lon)

    ra, dec = galactic_to_equatorial(gal_l, gal_b)
    alt, az = equatorial_to_altaz(ra, dec, lst, lat)
    return alt, az, ra, dec, jd


def compute_pointing(
    gal_l: float,
    gal_b: float,
    lat: float = nch.lat,
    lon: float = nch.lon,
    obs_alt: float = nch.alt,
    *,
    backend: str = "astropy",
) -> tuple[float, float, float, float, float]:
    """Convert galactic coordinates to horizontal/equatorial coordinates.

    Parameters
    ----------
    gal_l, gal_b : float
        Galactic longitude and latitude in degrees.
    lat, lon : float
        Observer latitude/longitude in degrees. Defaults to NCH.
    obs_alt : float
        Observer altitude in metres. Defaults to NCH.
    backend : str
        ``"astropy"`` (default) uses ugradio/astropy; ``"matrix"`` uses the
        explicit rotation-matrix implementation in this module.

    Returns
    -------
    (alt_deg, az_deg, ra_deg, dec_deg, jd)
    """
    if backend == "matrix":
        return compute_pointing_matrix(
            gal_l=gal_l, gal_b=gal_b, lat=lat, lon=lon, obs_alt=obs_alt
        )
    if backend != "astropy":
        raise ValueError(
            f"Unknown backend={backend!r}. Expected 'astropy' or 'matrix'."
        )

    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)

    gc = ac.SkyCoord(l=gal_l * u.deg, b=gal_b * u.deg, frame="galactic")
    ra, dec = gc.icrs.ra.deg, gc.icrs.dec.deg

    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=lat, lon=lon, alt=obs_alt)
    return alt, az, ra, dec, jd


# ---------------------------------------------------------------------------
# Backend comparison
# ---------------------------------------------------------------------------

@dataclass
class PointingComparison:
    """Residuals from comparing the matrix backend against astropy."""

    ra_matrix_deg: float
    dec_matrix_deg: float
    alt_matrix_deg: float
    az_matrix_deg: float

    ra_astropy_deg: float
    dec_astropy_deg: float
    alt_astropy_deg: float
    az_astropy_deg: float

    d_ra_deg: float
    d_dec_deg: float
    d_alt_deg: float
    d_az_deg: float

    # Residuals vs a recorded pointing (None when not supplied)
    d_alt_recorded_deg: float | None
    d_az_recorded_deg: float | None


def compare_pointing_backends(
    gal_l_deg: float,
    gal_b_deg: float,
    *,
    jd: float,
    lst_rad: float,
    lat_deg: float = nch.lat,
    lon_deg: float = nch.lon,
    obs_alt_m: float = nch.alt,
    recorded_alt_deg: float | None = None,
    recorded_az_deg: float | None = None,
) -> PointingComparison:
    """Compare matrix and astropy pointing backends at a given JD/LST.

    Parameters
    ----------
    gal_l_deg, gal_b_deg : float
        Target in galactic coordinates.
    jd : float
        Julian date of the observation.
    lst_rad : float
        Local Sidereal Time in radians.
    lat_deg, lon_deg, obs_alt_m : float
        Observer location. Defaults to NCH.
    recorded_alt_deg, recorded_az_deg : float or None
        Optional actually-recorded pointing for residual comparison.
    """
    # Matrix backend
    ra_m, dec_m = galactic_to_equatorial(gal_l_deg, gal_b_deg)
    alt_m, az_m = equatorial_to_altaz(ra_m, dec_m, lst_rad, lat_deg)

    # Astropy backend
    gc = ac.SkyCoord(l=gal_l_deg * u.deg, b=gal_b_deg * u.deg, frame="galactic")
    ra_a, dec_a = gc.icrs.ra.deg, gc.icrs.dec.deg
    alt_a, az_a = coord.get_altaz(ra_a, dec_a, jd=jd, lat=lat_deg, lon=lon_deg, alt=obs_alt_m)

    d_alt_rec = (alt_a - recorded_alt_deg) if recorded_alt_deg is not None else None
    d_az_rec = (az_a - recorded_az_deg) if recorded_az_deg is not None else None

    return PointingComparison(
        ra_matrix_deg=ra_m,
        dec_matrix_deg=dec_m,
        alt_matrix_deg=alt_m,
        az_matrix_deg=az_m,
        ra_astropy_deg=ra_a,
        dec_astropy_deg=dec_a,
        alt_astropy_deg=alt_a,
        az_astropy_deg=az_a,
        d_ra_deg=ra_m - ra_a,
        d_dec_deg=dec_m - dec_a,
        d_alt_deg=alt_m - float(alt_a),
        d_az_deg=az_m - float(az_a),
        d_alt_recorded_deg=d_alt_rec,
        d_az_recorded_deg=d_az_rec,
    )
