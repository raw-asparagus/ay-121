"""Solar diameter measurement and sunspot characterisation.

Uses the fringe amplitude envelope modulated by the uniform-disk Bessel
function to determine the angular diameter of the Sun and to detect
sunspot contributions at visibility nulls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import jn_zeros

from .constants import C_LIGHT_MS, NCH_LAT_DEG
from .fringe_model import (
    uniform_disk_visibility_amplitude,
    uniform_disk_visibility_signed,
)
from .geometry import projected_baseline_lambda

__all__ = [
    "SolarDiameterResult",
    "SunspotDetection",
    "extract_fringe_envelope",
    "find_bessel_zeros_in_envelope",
    "solar_diameter_from_zeros",
    "fit_solar_diameter_bessel",
    "detect_sunspot_anomalies",
    "characterize_sunspot_flux",
]

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolarDiameterResult:
    """Result of a solar angular diameter measurement."""

    diameter_arcmin: float
    diameter_err_arcmin: float
    method: str  # "bessel_zeros" or "bessel_fit"
    zero_crossings_u_R: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SunspotDetection:
    """A sunspot signature detected as a residual at a Bessel null."""

    null_index: int  # which Bessel null (1st, 2nd, ...)
    u_lambda_at_null: float  # projected baseline at the null
    ha_rad_at_null: float  # hour angle at the null
    residual_amplitude: float  # amplitude where it should be zero
    significance_sigma: float  # detection significance
    estimated_flux_fraction: float  # f_spot ~ |V_null| / |V(0)|


# ---------------------------------------------------------------------------
# Envelope extraction
# ---------------------------------------------------------------------------


def extract_fringe_envelope(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float,
    b_ew: float,
    freq_hz: float | np.ndarray,
    *,
    b_ns: float = 0.0,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
    band_mask: np.ndarray | None = None,
    smooth_window: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Extract the fringe amplitude envelope from DC-corrected visibilities.

    Parameters
    ----------
    corr_dc : (N_cap, N_CH) or (N_cap,) complex array
        DC-corrected visibility.
    freq_hz : float or (N_CH,) array
        Observing frequency (or per-channel frequencies).
    band_mask : (N_CH,) bool array, optional
        Channels to include in band averaging (True = include).
    smooth_window : int
        Running-mean smoothing kernel width (1 = no smoothing).

    Returns
    -------
    u_lambda : (N_cap,) array
        Projected baseline in wavelengths at each capture.
    envelope : (N_cap,) array
        Amplitude envelope (band-averaged |V|).
    envelope_std : (N_cap,) array
        Standard error of the band average at each capture.
    """
    corr_dc = np.atleast_2d(corr_dc)
    freq_hz = np.atleast_1d(freq_hz)

    if band_mask is not None:
        corr_dc = corr_dc[:, band_mask]
        freq_hz = freq_hz[band_mask]

    # Band-averaged amplitude per capture
    amp = np.abs(corr_dc)  # (N_cap, N_ch)
    envelope = np.nanmean(amp, axis=1)
    envelope_std = np.nanstd(amp, axis=1) / np.sqrt(np.sum(np.isfinite(amp), axis=1))

    # Projected baseline at band-centre frequency
    freq_center = np.mean(freq_hz)
    u_lambda = projected_baseline_lambda(
        ha_rad, dec_rad, b_ew, b_ns, freq_center, lat_rad
    )

    # Optional smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        envelope = np.convolve(envelope, kernel, mode="same")
        envelope_std = np.convolve(envelope_std, kernel, mode="same")

    return u_lambda, envelope, envelope_std


# ---------------------------------------------------------------------------
# Bessel zero detection
# ---------------------------------------------------------------------------


def find_bessel_zeros_in_envelope(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    *,
    smooth_window: int = 5,
) -> np.ndarray:
    """Locate projected-baseline values where the fringe envelope crosses zero.

    Uses sign changes in the smoothed, Hilbert-demodulated envelope. The
    envelope is first smoothed to suppress noise, then local minima below a
    threshold are identified as approximate null locations.

    Returns
    -------
    u_at_zeros : array
        Projected baseline values at each detected zero crossing.
    """
    # Sort by |u|
    order = np.argsort(np.abs(u_lambda))
    u_sorted = u_lambda[order]
    env_sorted = envelope[order]

    # Smooth
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        env_smooth = np.convolve(env_sorted, kernel, mode="same")
    else:
        env_smooth = env_sorted

    # Normalise to [0, 1]
    peak = np.nanmax(env_smooth)
    if peak <= 0:
        return np.array([])
    env_norm = env_smooth / peak

    # Find local minima below threshold (candidate nulls)
    threshold = 0.15
    zeros = []
    for i in range(1, len(env_norm) - 1):
        if (
            env_norm[i] < env_norm[i - 1]
            and env_norm[i] < env_norm[i + 1]
            and env_norm[i] < threshold
        ):
            zeros.append(u_sorted[i])

    return np.array(zeros)


# ---------------------------------------------------------------------------
# Solar diameter from Bessel zeros
# ---------------------------------------------------------------------------


def solar_diameter_from_zeros(
    zero_u_lambda: np.ndarray,
    n_zeros_to_use: int | None = None,
) -> SolarDiameterResult:
    r"""Angular diameter from Bessel-function zero crossings.

    For a uniform disk the first zero of :math:`J_1` occurs at:

    .. math::

        |u|\,R = \frac{j_{1,k}}{2\pi}

    so :math:`R = j_{1,k} / (2\pi\,|u_k|)`.

    Parameters
    ----------
    zero_u_lambda : array_like
        Projected baselines at observed nulls (in wavelengths).
    n_zeros_to_use : int, optional
        How many zeros to use (default: all).
    """
    zeros = np.abs(np.asarray(zero_u_lambda))
    if len(zeros) == 0:
        return SolarDiameterResult(
            diameter_arcmin=np.nan,
            diameter_err_arcmin=np.inf,
            method="bessel_zeros",
        )

    if n_zeros_to_use is not None:
        zeros = zeros[: n_zeros_to_use]

    j1_zeros = jn_zeros(1, len(zeros))
    u_R_theory = j1_zeros / (2.0 * np.pi)  # theoretical u*R at each null

    # Each null gives an estimate of R
    R_estimates = u_R_theory / zeros  # angular radius in radians

    R_mean = np.mean(R_estimates)
    R_err = np.std(R_estimates) / np.sqrt(len(R_estimates)) if len(R_estimates) > 1 else np.inf

    diameter_arcmin = np.rad2deg(2.0 * R_mean) * 60.0
    diameter_err = np.rad2deg(2.0 * R_err) * 60.0

    return SolarDiameterResult(
        diameter_arcmin=diameter_arcmin,
        diameter_err_arcmin=diameter_err,
        method="bessel_zeros",
        zero_crossings_u_R=u_R_theory,
        fitted_params={
            "R_rad_per_null": R_estimates,
            "u_observed": zeros,
            "j1_zeros_used": j1_zeros,
        },
    )


# ---------------------------------------------------------------------------
# Solar diameter from Bessel envelope fit
# ---------------------------------------------------------------------------


def _bessel_envelope_model(
    u_lambda: np.ndarray,
    amplitude: float,
    angular_radius_rad: float,
) -> np.ndarray:
    """Model function for curve_fit: ``A * |2 J_1(x) / x|``."""
    return amplitude * uniform_disk_visibility_amplitude(u_lambda, angular_radius_rad)


def fit_solar_diameter_bessel(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    *,
    envelope_std: np.ndarray | None = None,
    initial_diameter_arcmin: float = 32.0,
) -> SolarDiameterResult:
    r"""Nonlinear least-squares fit of the Bessel envelope.

    Fits ``A * |2 J_1(2*pi*|u|*R) / (2*pi*|u|*R)|`` to the observed
    amplitude envelope using :func:`scipy.optimize.curve_fit`.

    Parameters
    ----------
    u_lambda : array_like
        Projected baseline in wavelengths.
    envelope : array_like
        Measured amplitude envelope (same length as *u_lambda*).
    envelope_std : array_like, optional
        Per-point uncertainties (used as sigma in curve_fit).
    initial_diameter_arcmin : float
        Starting guess for the angular diameter.
    """
    u = np.asarray(u_lambda)
    env = np.asarray(envelope)

    valid = np.isfinite(env) & np.isfinite(u)
    u_fit = u[valid]
    env_fit = env[valid]
    sigma = envelope_std[valid] if envelope_std is not None else None

    R0 = np.deg2rad(initial_diameter_arcmin / 2.0 / 60.0)
    A0 = np.nanmax(env_fit)

    try:
        popt, pcov = curve_fit(
            _bessel_envelope_model,
            u_fit,
            env_fit,
            p0=[A0, R0],
            sigma=sigma,
            absolute_sigma=sigma is not None,
            bounds=([0, 0], [np.inf, np.deg2rad(1.0)]),
            maxfev=10000,
        )
    except RuntimeError:
        return SolarDiameterResult(
            diameter_arcmin=np.nan,
            diameter_err_arcmin=np.inf,
            method="bessel_fit",
        )

    amplitude_fit, R_fit = popt
    perr = np.sqrt(np.diag(pcov))
    R_err = perr[1]

    diameter_arcmin = np.rad2deg(2.0 * R_fit) * 60.0
    diameter_err = np.rad2deg(2.0 * R_err) * 60.0

    # Reduced chi-squared
    model = _bessel_envelope_model(u_fit, *popt)
    resid = env_fit - model
    if sigma is not None:
        chi2 = np.sum((resid / sigma) ** 2)
    else:
        chi2 = np.sum(resid**2) / (np.var(resid) if np.var(resid) > 0 else 1.0)
    dof = max(1, len(u_fit) - 2)
    chi2_red = chi2 / dof

    return SolarDiameterResult(
        diameter_arcmin=diameter_arcmin,
        diameter_err_arcmin=diameter_err,
        method="bessel_fit",
        fitted_params={
            "amplitude": amplitude_fit,
            "R_rad": R_fit,
            "R_err_rad": R_err,
            "chi2_reduced": chi2_red,
            "popt": popt,
            "pcov": pcov,
        },
    )


# ---------------------------------------------------------------------------
# Sunspot detection
# ---------------------------------------------------------------------------


def detect_sunspot_anomalies(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    disk_model: np.ndarray,
    noise_std: np.ndarray | float,
    *,
    sigma_threshold: float = 3.0,
    null_search_radius_frac: float = 0.1,
) -> list[SunspotDetection]:
    r"""Detect sunspot signatures as excess amplitude at Bessel nulls.

    At a Bessel null, the uniform-disk visibility is zero, so any residual
    amplitude comes from asymmetric structure (sunspots).  The spot flux
    fraction is approximately:

    .. math::

        f_{\rm spot} \approx \frac{|V_{\rm null}|}{|V(0)|}

    Parameters
    ----------
    disk_model : array_like
        Expected amplitude from the uniform-disk model (same shape as
        *envelope*).
    noise_std : float or array_like
        Noise estimate per point.
    null_search_radius_frac : float
        Fraction of the null spacing to search around each predicted null.
    """
    u = np.abs(np.asarray(u_lambda))
    env = np.asarray(envelope)
    model = np.asarray(disk_model)
    noise = np.broadcast_to(np.asarray(noise_std), env.shape)

    residual = env - model

    # Find model nulls (local minima in model)
    null_indices = []
    for i in range(1, len(model) - 1):
        if model[i] < model[i - 1] and model[i] < model[i + 1]:
            null_indices.append(i)

    v0 = np.nanmax(env)  # V(0) normalization
    detections = []

    for null_idx in null_indices:
        # Search region around null
        u_null = u[null_idx]
        radius = null_search_radius_frac * u_null if u_null > 0 else 1.0
        mask = np.abs(u - u_null) <= radius

        if not mask.any():
            continue

        # Peak residual in the search window
        region_resid = residual[mask]
        region_noise = noise[mask]
        region_env = env[mask]

        peak_in_region = np.argmax(np.abs(region_resid))
        peak_resid = region_resid[peak_in_region]
        peak_noise = region_noise[peak_in_region]
        peak_env = region_env[peak_in_region]

        significance = np.abs(peak_resid) / peak_noise if peak_noise > 0 else 0.0

        if significance >= sigma_threshold:
            # Map back to global index
            global_indices = np.where(mask)[0]
            gi = global_indices[peak_in_region]

            flux_frac = peak_env / v0 if v0 > 0 else np.nan

            detections.append(
                SunspotDetection(
                    null_index=len(detections) + 1,
                    u_lambda_at_null=u[gi],
                    ha_rad_at_null=np.nan,  # caller should fill from context
                    residual_amplitude=peak_resid,
                    significance_sigma=significance,
                    estimated_flux_fraction=flux_frac,
                )
            )

    return detections


def characterize_sunspot_flux(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    disk_diameter_arcmin: float,
    *,
    n_nulls: int = 3,
) -> dict:
    """Estimate sunspot flux fraction from visibility at Bessel nulls.

    A simpler interface than :func:`detect_sunspot_anomalies` — computes the
    theoretical null locations and measures the envelope amplitude there.

    Returns
    -------
    dict with keys:
        ``null_u_R`` — theoretical u*R values,
        ``null_u_observed`` — u_lambda at each null,
        ``null_amplitude`` — measured envelope amplitude at each null,
        ``flux_fraction`` — estimated spot flux fraction per null.
    """
    R_rad = np.deg2rad(disk_diameter_arcmin / 2.0 / 60.0)
    j1_zeros = jn_zeros(1, n_nulls)
    u_R_theory = j1_zeros / (2.0 * np.pi)
    u_null_theory = u_R_theory / R_rad  # projected baseline at each null

    u_abs = np.abs(np.asarray(u_lambda))
    env = np.asarray(envelope)

    v0 = np.nanmax(env)
    null_amps = []
    null_u_obs = []

    for u_null in u_null_theory:
        idx = np.argmin(np.abs(u_abs - u_null))
        null_amps.append(env[idx])
        null_u_obs.append(u_abs[idx])

    null_amps = np.array(null_amps)
    flux_fracs = null_amps / v0 if v0 > 0 else np.full_like(null_amps, np.nan)

    return {
        "null_u_R": u_R_theory,
        "null_u_observed": np.array(null_u_obs),
        "null_amplitude": null_amps,
        "flux_fraction": flux_fracs,
    }
