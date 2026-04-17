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

_SIN_LAT_NCH = np.sin(np.deg2rad(NCH_LAT_DEG))


# ============================================================================
# Parameter containers
# ============================================================================


@dataclass(frozen=True)
class FringeModelParams:
    """Interferometer + source parameters for point-source fringe models."""

    b_ew: float          # east-west baseline [m]
    b_ns: float          # north-south baseline [m]
    freq_hz: float       # observing frequency [Hz]
    dec_rad: float       # source declination [rad]
    amplitude: float     # overall amplitude scale
    phase_offset: float  # instrumental phase offset [rad]


@dataclass(frozen=True)
class SolarDiskParams:
    """Uniform-disk model of the Sun."""

    angular_radius_rad: float


@dataclass(frozen=True)
class SunspotParams:
    """Point-source sunspot model offset from disk centre."""

    flux_fraction: float
    offset_ha_rad: float
    offset_dec_rad: float


# ============================================================================
# Point-source models
# ============================================================================


def point_source_visibility(ha_rad: np.ndarray, params: FringeModelParams) -> np.ndarray:
    r"""Complex visibility for a point source.

    .. math::

        V(h) = A\,\exp\!\bigl[i\,(2\pi\nu\,\tau_g(h) + \phi_0)\bigr]
    """
    tau = geometric_delay_s(ha_rad, params.dec_rad, params.b_ew, params.b_ns)
    phase = 2.0 * np.pi * params.freq_hz * tau + params.phase_offset
    return params.amplitude * np.exp(1j * phase)


def point_source_fringes(ha_rad: np.ndarray, params: FringeModelParams) -> np.ndarray:
    """Real-valued fringe pattern for a point source."""
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


def uniform_disk_visibility_amplitude(u_lambda, angular_radius_rad: float) -> np.ndarray:
    r"""Normalised visibility amplitude for a uniform disk.

    .. math::

        \frac{|V|}{V(0)} = \left|\frac{2\,J_1(x)}{x}\right|, \quad x = 2\pi |u| R
    """
    x = 2.0 * np.pi * np.abs(np.asarray(u_lambda)) * angular_radius_rad
    return np.abs(_jinc(x))


def uniform_disk_visibility_signed(u_lambda, angular_radius_rad: float) -> np.ndarray:
    r"""Signed visibility for a uniform disk: :math:`2\,J_1(x)/x`."""
    x = 2.0 * np.pi * np.abs(np.asarray(u_lambda)) * angular_radius_rad
    return _jinc(x)


def uniform_disk_zeros(n: int) -> np.ndarray:
    r"""Projected-baseline values :math:`(|u|R)_k = j_{1,k}/(2\pi)` at the
    first *n* Bessel-function zero crossings."""
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

    Combines the point-source fringe with uniform-disk amplitude modulation.
    If *sunspot_params* is supplied, adds a point-source contribution offset
    from disk centre.
    """
    u = sky_baseline_lambda(
        ha_rad, fringe_params.dec_rad, fringe_params.b_ew, fringe_params.b_ns, fringe_params.freq_hz
    )
    disk_mod = uniform_disk_visibility_signed(u, disk_params.angular_radius_rad)
    v_disk = point_source_visibility(ha_rad, fringe_params) * disk_mod

    if sunspot_params is None:
        return v_disk

    f = sunspot_params.flux_fraction
    lam = C_LIGHT_MS / fringe_params.freq_hz
    cos_dec = np.cos(fringe_params.dec_rad)
    sin_dec = np.sin(fringe_params.dec_rad)
    sin_h = np.sin(ha_rad)
    cos_h = np.cos(ha_rad)
    # Differential phase for offset source: 2π ν Δτ where
    # Δτ = (∂τ_g/∂h)·Δh + (∂τ_g/∂δ)·Δδ
    b_ew, b_ns = fringe_params.b_ew, fringe_params.b_ns
    dtau_dh = (b_ew / C_LIGHT_MS) * cos_dec * cos_h - (b_ns / C_LIGHT_MS) * _SIN_LAT_NCH * cos_dec * sin_h
    dtau_ddec = -(b_ew / C_LIGHT_MS) * sin_dec * sin_h - (b_ns / C_LIGHT_MS) * _SIN_LAT_NCH * sin_dec * cos_h
    delta_phase = 2.0 * np.pi * fringe_params.freq_hz * (
        dtau_dh * sunspot_params.offset_ha_rad
        + dtau_ddec * sunspot_params.offset_dec_rad
    )
    v_spot = point_source_visibility(ha_rad, fringe_params) * np.exp(1j * delta_phase)
    return (1.0 - f) * v_disk + f * v_spot


def fringe_envelope(
    ha_rad: np.ndarray,
    fringe_params: FringeModelParams,
    disk_params: SolarDiskParams,
) -> np.ndarray:
    """Amplitude envelope :math:`A\\,|2 J_1(x)/x|` of the solar fringe pattern."""
    u = sky_baseline_lambda(
        ha_rad, fringe_params.dec_rad, fringe_params.b_ew, fringe_params.b_ns, fringe_params.freq_hz
    )
    return fringe_params.amplitude * uniform_disk_visibility_amplitude(
        u, disk_params.angular_radius_rad
    )
