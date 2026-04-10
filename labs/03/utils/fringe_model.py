"""Phenomenological models of interferometric fringe patterns.

Provides forward models for:
* Point-source fringes (complex visibility and real fringe pattern).
* Uniform-disk visibility amplitude (Bessel-function modulation).
* Combined solar visibility (disk + optional sunspot component).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1, jn_zeros

from .constants import C_LIGHT_MS, NCH_LAT_DEG
from .geometry import geometric_delay_s, sky_baseline_lambda

__all__ = [
    "FringeModelParams",
    "SolarDiskParams",
    "SunspotParams",
    "point_source_visibility",
    "point_source_fringes",
    "uniform_disk_visibility_amplitude",
    "uniform_disk_visibility_signed",
    "uniform_disk_zeros",
    "solar_visibility",
    "fringe_envelope",
]

# ============================================================================
# Parameter containers
# ============================================================================


@dataclass(frozen=True)
class FringeModelParams:
    """Interferometer + source parameters for point-source fringe models."""

    b_ew: float  # east-west baseline [m]
    b_ns: float  # north-south baseline [m]
    freq_hz: float  # observing frequency [Hz]
    dec_rad: float  # source declination [rad]
    lat_rad: float = np.deg2rad(NCH_LAT_DEG)  # observatory latitude [rad]
    amplitude: float = 1.0  # overall amplitude scale
    phase_offset: float = 0.0  # instrumental phase offset [rad]


@dataclass(frozen=True)
class SolarDiskParams:
    """Uniform-disk model of the Sun."""

    angular_radius_rad: float  # angular radius [rad]


@dataclass(frozen=True)
class SunspotParams:
    """Point-source sunspot model (delta-function offset from disk centre)."""

    flux_fraction: float  # fraction of total solar flux in the spot
    offset_ha_rad: float  # offset from disk centre in HA direction [rad]
    offset_dec_rad: float  # offset from disk centre in Dec direction [rad]


# ============================================================================
# Point-source models
# ============================================================================


def point_source_visibility(
    ha_rad: np.ndarray,
    params: FringeModelParams,
) -> np.ndarray:
    r"""Complex visibility for a point source.

    .. math::

        V(h) = A\,\exp\!\bigl[i\,(2\pi\nu\,\tau_g(h) + \phi_0)\bigr]

    Returns complex-valued array of shape ``ha_rad.shape``.
    """
    tau = geometric_delay_s(
        ha_rad, params.dec_rad, params.b_ew, params.b_ns, params.lat_rad
    )
    phase = 2.0 * np.pi * params.freq_hz * tau + params.phase_offset
    return params.amplitude * np.exp(1j * phase)


def point_source_fringes(
    ha_rad: np.ndarray,
    params: FringeModelParams,
) -> np.ndarray:
    r"""Real-valued fringe pattern for a point source.

    .. math::

        F(h) = A\cos(2\pi\nu\,\tau_g + \phi_0)
             + B\sin(2\pi\nu\,\tau_g + \phi_0)

    This is simply ``Re[point_source_visibility]``.
    """
    return point_source_visibility(ha_rad, params).real


# ============================================================================
# Uniform-disk (Bessel) modulation
# ============================================================================


def _jinc(x: np.ndarray) -> np.ndarray:
    """Evaluate 2*J_1(x)/x with the x = 0 singularity handled."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0.0
    out[nz] = 2.0 * j1(x[nz]) / x[nz]
    return out


def uniform_disk_visibility_amplitude(
    u_lambda: np.ndarray | float,
    angular_radius_rad: float,
) -> np.ndarray:
    r"""Normalised visibility amplitude for a uniform disk.

    .. math::

        \frac{|V|}{V(0)} = \left|\frac{2\,J_1(x)}{x}\right|,
        \quad x = 2\pi\,|u|\,R

    where *u* is the projected baseline in wavelengths and *R* is the angular
    radius in radians.

    Parameters
    ----------
    u_lambda : array_like
        Projected baseline in wavelengths (spatial frequency).
    angular_radius_rad : float
        Angular radius of the disk in radians.
    """
    x = 2.0 * np.pi * np.abs(np.asarray(u_lambda)) * angular_radius_rad
    return np.abs(_jinc(x))


def uniform_disk_visibility_signed(
    u_lambda: np.ndarray | float,
    angular_radius_rad: float,
) -> np.ndarray:
    r"""Signed (not absolute-valued) visibility for a uniform disk.

    .. math::

        \frac{V}{V(0)} = \frac{2\,J_1(x)}{x},
        \quad x = 2\pi\,|u|\,R

    Useful for detecting sign changes (Bessel zero crossings).
    """
    x = 2.0 * np.pi * np.abs(np.asarray(u_lambda)) * angular_radius_rad
    return _jinc(x)


def uniform_disk_zeros(n: int = 5) -> np.ndarray:
    r"""Projected-baseline values at Bessel-function zero crossings.

    Returns the first *n* values of :math:`|u|\,R` at which
    :math:`J_1(2\pi u R) = 0`, i.e.

    .. math::

        (|u|\,R)_k = \frac{j_{1,k}}{2\pi}

    where :math:`j_{1,k}` is the *k*-th positive zero of :math:`J_1`.

    To obtain the angular radius from an observed zero:
    ``R = (u_R)_k / u_observed_k``.
    """
    return jn_zeros(1, n) / (2.0 * np.pi)


# ============================================================================
# Solar visibility (disk + optional sunspot)
# ============================================================================


def solar_visibility(
    ha_rad: np.ndarray,
    fringe_params: FringeModelParams,
    disk_params: SolarDiskParams,
    sunspot_params: SunspotParams | None = None,
) -> np.ndarray:
    r"""Full complex visibility model for the Sun.

    Combines the point-source fringe oscillation with the uniform-disk
    amplitude modulation:

    .. math::

        V_{\rm disk}(h) = V_{\rm pt}(h)\;\frac{2\,J_1(x)}{x}

    If *sunspot_params* is given, adds a point-source component offset from
    disk centre:

    .. math::

        V_{\rm total} = (1 - f)\,V_{\rm disk} + f\,V_{\rm spot}

    where *f* is the flux fraction and the spot has an additional phase
    proportional to its angular offset projected onto the baseline.
    """
    # Disk component
    u = sky_baseline_lambda(
        ha_rad,
        fringe_params.dec_rad,
        fringe_params.b_ew,
        fringe_params.b_ns,
        fringe_params.freq_hz,
        fringe_params.lat_rad,
    )
    disk_mod = uniform_disk_visibility_signed(u, disk_params.angular_radius_rad)
    v_disk = point_source_visibility(ha_rad, fringe_params) * disk_mod

    if sunspot_params is None:
        return v_disk

    # Sunspot: point source with additional phase from angular offset
    f = sunspot_params.flux_fraction
    # Phase offset from spot position projected onto baseline vector
    # delta_phi = 2*pi * (b/lambda) . delta_theta
    wavelength = C_LIGHT_MS / fringe_params.freq_hz
    delta_phase = 2.0 * np.pi * (
        (fringe_params.b_ew / wavelength) * sunspot_params.offset_ha_rad
        + (fringe_params.b_ns / wavelength)
        * np.sin(fringe_params.lat_rad)
        * sunspot_params.offset_dec_rad
    )
    v_spot = point_source_visibility(ha_rad, fringe_params) * np.exp(1j * delta_phase)

    return (1.0 - f) * v_disk + f * v_spot


def fringe_envelope(
    ha_rad: np.ndarray,
    fringe_params: FringeModelParams,
    disk_params: SolarDiskParams,
) -> np.ndarray:
    r"""Amplitude envelope of the solar fringe pattern.

    .. math::

        |V_{\rm disk}(h)| = A\,\left|\frac{2\,J_1(x)}{x}\right|

    where :math:`x = 2\pi\,|u(h)|\,R`.
    """
    u = sky_baseline_lambda(
        ha_rad,
        fringe_params.dec_rad,
        fringe_params.b_ew,
        fringe_params.b_ns,
        fringe_params.freq_hz,
        fringe_params.lat_rad,
    )
    return fringe_params.amplitude * uniform_disk_visibility_amplitude(
        u, disk_params.angular_radius_rad
    )
