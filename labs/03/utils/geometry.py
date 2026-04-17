"""Interferometer geometry, timing, and coordinate utilities.

Angles are in radians unless the function name contains ``_deg``.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    C_LIGHT_MS,
    NCH_LAT_DEG,
    NCH_LON_DEG,
    OMEGA_EARTH_RAD_S,
)

_LAT_RAD_NCH = np.deg2rad(NCH_LAT_DEG)


def lst_deg(unix_t):
    """Local Sidereal Time at NCH in degrees (IAU GMST polynomial)."""
    jd = np.asarray(unix_t) / 86_400.0 + 2_440_587.5
    T = (jd - 2_451_545.0) / 36_525.0
    gmst = (
        280.460_618_37
        + 360.985_647_366_29 * (jd - 2_451_545.0)
        + T**2 * 0.000_387_933
        - T**3 / 38_710_000.0
    )
    return (gmst + NCH_LON_DEG) % 360.0


def hour_angle_deg(lst_deg_val, ra_deg):
    """Hour angle in degrees, wrapped to (-180, 180]."""
    ha = np.asarray(lst_deg_val) - np.asarray(ra_deg)
    return ((ha + 180.0) % 360.0) - 180.0


def hour_angle_rad(lst_deg_val, ra_deg):
    """Hour angle in radians, wrapped to (-pi, pi]."""
    return np.deg2rad(hour_angle_deg(lst_deg_val, ra_deg))


def geometric_delay_s(ha_rad, dec_rad, b_ew, b_ns):
    r"""Geometric path delay between east and west antennas [seconds].

    .. math::

        \tau_g(h) = \frac{b_{\rm ew}}{c}\cos\delta\,\sin h
                  + \frac{b_{\rm ns}}{c}\sin L\,\cos\delta\,\cos h
    """
    cos_dec = np.cos(dec_rad)
    return (
        (b_ew / C_LIGHT_MS) * cos_dec * np.sin(ha_rad)
        + (b_ns / C_LIGHT_MS) * np.sin(_LAT_RAD_NCH) * cos_dec * np.cos(ha_rad)
    )


def fringe_frequency_hz(ha_rad, dec_rad, b_ew, b_ns, freq_hz):
    r"""Instantaneous fringe frequency in Hz.

    .. math::

        f_f = \omega_\oplus \left[
            \frac{b_{\rm ew}}{\lambda}\cos\delta\,\cos h
          - \frac{b_{\rm ns}}{\lambda}\sin L\,\cos\delta\,\sin h
        \right]
    """
    lam = C_LIGHT_MS / freq_hz
    cos_dec = np.cos(dec_rad)
    return OMEGA_EARTH_RAD_S * (
        (b_ew / lam) * cos_dec * np.cos(ha_rad)
        - (b_ns / lam) * np.sin(_LAT_RAD_NCH) * cos_dec * np.sin(ha_rad)
    )


def fringe_period_s(ha_rad, dec_rad, b_ew, b_ns, freq_hz):
    """Instantaneous fringe period in seconds (1 / |f_f|)."""
    return 1.0 / np.abs(fringe_frequency_hz(ha_rad, dec_rad, b_ew, b_ns, freq_hz))


def sky_baseline_lambda(ha_rad, dec_rad, b_ew, b_ns, freq_hz):
    r"""Full 2-D projected baseline on the sky plane, in wavelengths.

    Returns :math:`q(h) = \sqrt{u(h)^2 + v_0^2}`, the magnitude of the
    sky-plane spatial frequency relevant to the Bessel-function modulation
    of an extended source.
    """
    ff = fringe_frequency_hz(ha_rad, dec_rad, b_ew, b_ns, freq_hz)
    u = np.abs(ff) / OMEGA_EARTH_RAD_S
    lam = C_LIGHT_MS / freq_hz
    v0 = b_ns * np.cos(_LAT_RAD_NCH) * np.sin(dec_rad) / lam
    return np.sqrt(u**2 + v0**2)


def x_coordinate(ha_rad, dec_rad):
    r"""Dimensionless coordinate :math:`x = \cos\delta\,\sin h`."""
    return np.cos(dec_rad) * np.sin(np.asarray(ha_rad))
