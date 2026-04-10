"""Interferometer geometry, timing, and coordinate utilities.

All angles are in **radians** unless the function name contains ``_deg``.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    C_LIGHT_MS,
    NCH_LAT_DEG,
    NCH_LON_DEG,
    OMEGA_EARTH_RAD_S,
)

__all__ = [
    "jd_from_unix",
    "lst_deg",
    "hour_angle_deg",
    "hour_angle_rad",
    "geometric_delay_s",
    "effective_geometric_delay_s",
    "delay_baseline_lambda",
    "sky_baseline_lambda",
    "fringe_frequency_hz",
    "fringe_period_s",
    "x_coordinate",
]

# ---------------------------------------------------------------------------
# Time conversions
# ---------------------------------------------------------------------------

_JD_UNIX_EPOCH = 2_440_587.5  # JD at 1970-01-01T00:00:00 UTC


def jd_from_unix(unix_t: np.ndarray | float) -> np.ndarray | float:
    """Convert Unix timestamp(s) to Julian Date."""
    return np.asarray(unix_t) / 86_400.0 + _JD_UNIX_EPOCH


def lst_deg(
    unix_t: np.ndarray | float,
    lon_deg: float = NCH_LON_DEG,
) -> np.ndarray | float:
    """Local Sidereal Time in degrees.

    Uses the IAU GMST polynomial (same formula as
    ``labs/03/scripts/utils.py:lst_deg``).

    Parameters
    ----------
    unix_t : array_like
        Unix timestamp(s) in seconds.
    lon_deg : float
        Observer east longitude in degrees (default: NCH).
    """
    jd = jd_from_unix(unix_t)
    T = (jd - 2_451_545.0) / 36_525.0
    gmst = (
        280.460_618_37
        + 360.985_647_366_29 * (jd - 2_451_545.0)
        + T**2 * 0.000_387_933
        - T**3 / 38_710_000.0
    )
    return (gmst + lon_deg) % 360.0


# ---------------------------------------------------------------------------
# Hour angle
# ---------------------------------------------------------------------------


def hour_angle_deg(
    lst_deg_val: np.ndarray | float,
    ra_deg: np.ndarray | float,
) -> np.ndarray | float:
    """Hour angle in degrees, wrapped to (-180, 180].

    HA = LST - RA.
    """
    ha = np.asarray(lst_deg_val) - np.asarray(ra_deg)
    return ((ha + 180.0) % 360.0) - 180.0


def hour_angle_rad(
    lst_deg_val: np.ndarray | float,
    ra_deg: np.ndarray | float,
) -> np.ndarray | float:
    """Hour angle in radians, wrapped to (-pi, pi]."""
    return np.deg2rad(hour_angle_deg(lst_deg_val, ra_deg))


# ---------------------------------------------------------------------------
# Geometric delay
# ---------------------------------------------------------------------------


def geometric_delay_s(
    ha_rad: np.ndarray | float,
    dec_rad: float,
    b_ew: float,
    b_ns: float = 0.0,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
) -> np.ndarray | float:
    r"""Geometric path delay between east and west antennas [seconds].

    .. math::

        \tau_g(h) = \frac{b_{\rm ew}}{c}\cos\delta\,\sin h
                  + \frac{b_{\rm ns}}{c}\sin L\,\cos\delta\,\cos h

    Parameters
    ----------
    ha_rad : array_like
        Source hour angle in radians.
    dec_rad : float
        Source declination in radians.
    b_ew, b_ns : float
        East-west and north-south baseline components in metres.
    lat_rad : float
        Observatory latitude in radians.
    """
    ha = np.asarray(ha_rad)
    cos_dec = np.cos(dec_rad)
    return (
        (b_ew / C_LIGHT_MS) * cos_dec * np.sin(ha)
        + (b_ns / C_LIGHT_MS) * np.sin(lat_rad) * cos_dec * np.cos(ha)
    )


def effective_geometric_delay_s(
    ha_rad: np.ndarray | float,
    dec_rad: float,
    b_ew: float,
    b_ns: float = 0.0,
    tau_c: float = 0.0,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
) -> np.ndarray | float:
    r"""Total delay including cable/instrumental offset.

    .. math::

        \tau_{\rm tot}(h) = \tau_g(h) + \tau_c
                          - \frac{b_{\rm ns}}{c}\cos L\,\sin\delta

    The last term folds the constant NS contribution into an effective cable
    delay (see lab manual).
    """
    tau_g = geometric_delay_s(ha_rad, dec_rad, b_ew, b_ns, lat_rad)
    tau_c_eff = tau_c - (b_ns / C_LIGHT_MS) * np.cos(lat_rad) * np.sin(dec_rad)
    return tau_g + tau_c_eff


# ---------------------------------------------------------------------------
# Projected baseline / spatial frequency
# ---------------------------------------------------------------------------


def delay_baseline_lambda(
    ha_rad: np.ndarray | float,
    dec_rad: float,
    b_ew: float,
    b_ns: float,
    freq_hz: float | np.ndarray,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
) -> np.ndarray | float:
    r"""Delay in units of wavelength (the *w*-component).

    .. math::

        w = \nu\,\tau_g(h)

    This is the component of the baseline **along** the line of sight.
    It is zero at transit (for a pure EW baseline) and maximum at the horizon.

    .. note::

        For the Bessel-function visibility of an extended source, use
        :func:`sky_baseline_lambda` instead — that gives the sky-plane
        component perpendicular to the line of sight.
    """
    tau = geometric_delay_s(ha_rad, dec_rad, b_ew, b_ns, lat_rad)
    return np.asarray(freq_hz) * tau


def sky_baseline_lambda(
    ha_rad: np.ndarray | float,
    dec_rad: float,
    b_ew: float,
    b_ns: float = 0.0,
    freq_hz: float = 10.5e9,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
) -> np.ndarray | float:
    r"""Full 2-D projected baseline on the sky plane, in wavelengths.

    The sky-plane spatial frequency relevant to the Bessel-function
    modulation of an extended source has **two** components:

    .. math::

        u(h) = \frac{|f_f|}{\omega_\oplus}
             = \left|\frac{b_{\rm ew}}{\lambda}\cos\delta\,\cos h
               - \frac{b_{\rm ns}}{\lambda}\sin L\,\cos\delta\,\sin h\right|

    (the time-varying component that produces fringes), and

    .. math::

        v_0 = \frac{b_{\rm ns}\cos L}{\lambda}

    (a constant component from the NS baseline projected perpendicular to
    the Earth's rotation axis).  The total sky-plane spatial frequency is

    .. math::

        q(h) = \sqrt{u(h)^2 + v_0^2}

    For a purely EW baseline (:math:`b_{\rm ns} = 0`), :math:`v_0 = 0` and
    :math:`q = |u|` — the 1-D approximation used in many textbooks.
    """
    ff = fringe_frequency_hz(ha_rad, dec_rad, b_ew, b_ns, freq_hz, lat_rad)
    u = np.abs(ff) / OMEGA_EARTH_RAD_S
    lam = C_LIGHT_MS / freq_hz
    v0 = b_ns * np.cos(lat_rad) / lam
    return np.sqrt(u**2 + v0**2)


# ---------------------------------------------------------------------------
# Fringe frequency
# ---------------------------------------------------------------------------


def fringe_frequency_hz(
    ha_rad: np.ndarray | float,
    dec_rad: float,
    b_ew: float,
    b_ns: float = 0.0,
    freq_hz: float = 10.5e9,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
) -> np.ndarray | float:
    r"""Instantaneous fringe frequency in Hz.

    .. math::

        f_f = \omega_\oplus \left[
            \frac{b_{\rm ew}}{\lambda}\cos\delta\,\cos h
          - \frac{b_{\rm ns}}{\lambda}\sin L\,\cos\delta\,\sin h
        \right]

    This is :math:`d\phi / dt` where :math:`\phi = 2\pi\nu\tau_g`.
    """
    ha = np.asarray(ha_rad)
    lam = C_LIGHT_MS / freq_hz
    cos_dec = np.cos(dec_rad)
    return OMEGA_EARTH_RAD_S * (
        (b_ew / lam) * cos_dec * np.cos(ha)
        - (b_ns / lam) * np.sin(lat_rad) * cos_dec * np.sin(ha)
    )


def fringe_period_s(
    ha_rad: np.ndarray | float,
    dec_rad: float,
    b_ew: float,
    b_ns: float = 0.0,
    freq_hz: float = 10.5e9,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
) -> np.ndarray | float:
    """Instantaneous fringe period in seconds (1 / |f_f|)."""
    ff = fringe_frequency_hz(ha_rad, dec_rad, b_ew, b_ns, freq_hz, lat_rad)
    return 1.0 / np.abs(ff)


# ---------------------------------------------------------------------------
# x-coordinate for FFT baseline method
# ---------------------------------------------------------------------------


def x_coordinate(
    ha_rad: np.ndarray | float,
    dec_rad: float,
) -> np.ndarray | float:
    r"""Dimensionless coordinate :math:`x = \cos\delta\,\sin h`.

    The visibility phase is :math:`\phi = 2\pi (b_{\rm ew}/\lambda)\,x` (for
    ``b_ns = 0``), so the fringe is a pure sinusoid in *x*, making FFT-based
    baseline extraction straightforward.
    """
    return np.cos(dec_rad) * np.sin(np.asarray(ha_rad))
