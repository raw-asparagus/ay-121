"""Solar diameter measurement and sunspot characterisation.

Uses the fringe amplitude envelope modulated by the uniform-disk Bessel
function to determine the angular diameter of the Sun and to detect
sunspot contributions at visibility nulls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import find_peaks
from scipy.special import jn_zeros

from .fringe_model import uniform_disk_visibility_amplitude, uniform_disk_visibility_signed
from .geometry import sky_baseline_lambda


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolarDiameterResult:
    diameter_arcmin: float
    diameter_err_arcmin: float
    method: str  # "bessel_zeros" or "bessel_fit"
    zero_crossings_u_R: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SunspotDetection:
    null_index: int
    u_lambda_at_null: float
    ha_rad_at_null: float
    residual_amplitude: float
    significance_sigma: float
    estimated_flux_fraction: float


@dataclass(frozen=True)
class SunspotLocalization:
    """Result of fitting a single offset point source on top of a uniform disk.

    The model fitted is, for an EW baseline projected onto the sky,

        V(u) = (1 - f) * V_disk(u; R) + f * exp(-2 pi i * u * delta_alpha_rad)

    where ``delta_alpha_rad`` is the angular offset along the EW direction
    from the disk centre. The NS offset is unconstrained for a single
    EW baseline (it would require an NS baseline component or a different
    parallactic angle), so we report only the EW offset.
    """
    flux_fraction: float
    flux_fraction_err: float
    delta_alpha_arcmin: float
    delta_alpha_err_arcmin: float
    chi2_reduced: float
    n_points: int
    null_index: int


# ---------------------------------------------------------------------------
# Envelope extraction
# ---------------------------------------------------------------------------


def extract_fringe_envelope(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float,
    b_ew: float,
    f_sky_hz: np.ndarray,
    b_ns: float,
    band_mask: np.ndarray,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Band-averaged amplitude envelope vs projected baseline.

    Returns ``(u_lambda, envelope, envelope_std)``, where ``envelope`` is the
    band-averaged ``|V|`` per capture and ``u_lambda`` is the projected
    baseline at the band-centre frequency.

    The per-capture uncertainty is estimated from the across-channel scatter
    of |V| inside the analysis band, divided by ``sqrt(N_eff)``. Smoothing
    correlates samples; the returned ``envelope_std`` is rescaled by
    ``sqrt(smooth_window)`` to approximately compensate, so that downstream
    least-squares fits do not under-estimate the per-point uncertainty.
    """
    band_corr = corr_dc[:, band_mask]
    band_freq = f_sky_hz[band_mask]

    amp = np.abs(band_corr)
    envelope = np.nanmean(amp, axis=1)
    n_good = np.sum(np.isfinite(amp), axis=1).astype(float)
    n_good[n_good == 0] = np.nan
    envelope_std = np.nanstd(amp, axis=1) / np.sqrt(n_good)

    u_lambda = sky_baseline_lambda(ha_rad, dec_rad, b_ew, b_ns, float(np.mean(band_freq)))

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        envelope = np.convolve(envelope, kernel, mode="same")
        # Smoothing correlates neighbouring points; inflate sigma so that the
        # *effective* number of independent samples in any subsequent
        # least-squares fit is roughly N_caps / smooth_window rather than N_caps.
        envelope_std = np.convolve(envelope_std, kernel, mode="same") * np.sqrt(smooth_window)

    return u_lambda, envelope, envelope_std


# ---------------------------------------------------------------------------
# Bessel zero detection
# ---------------------------------------------------------------------------


def find_bessel_zeros_in_envelope(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    smooth_window: int,
    expected_diameter_arcmin: float,
    threshold: float = 0.15,
    prominence_frac: float = 0.05,
    n_max: int = 8,
) -> np.ndarray:
    """Locate projected-baseline values where the fringe envelope nulls.

    Sorts the envelope by ``|u|``, smooths it, and uses
    ``scipy.signal.find_peaks`` on the *negated* normalised envelope with
    a minimum-distance constraint set from the expected Bessel-null spacing
    (``Δu ≈ π / R``) and a minimum prominence of ``prominence_frac`` of the
    peak. This eliminates the noise-dominated point-wise local minima the
    naive 3-point test would otherwise return.

    Parameters
    ----------
    threshold : maximum normalised envelope value at an accepted null.
    prominence_frac : minimum prominence (fraction of peak envelope).
    n_max : at most this many strongest nulls are returned.
    """
    R_approx = np.deg2rad(expected_diameter_arcmin / 2.0 / 60.0)
    j1 = jn_zeros(1, 4)
    min_u_lambda = 0.5 * j1[0] / (2.0 * np.pi * R_approx)
    # Spacing between successive Bessel-J1 zeros: ~π in argument 2π R u,
    # so in u-space the spacing is ~1/(2 R).
    expected_null_spacing = 0.5 / R_approx  # in wavelengths

    u_abs = np.abs(u_lambda)
    finite = np.isfinite(u_abs) & np.isfinite(envelope)
    if not np.any(finite):
        return np.array([])

    order = np.argsort(u_abs[finite])
    u_sorted     = u_lambda[finite][order]
    u_abs_sorted = u_abs[finite][order]
    env_sorted   = envelope[finite][order]

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        env_smooth = np.convolve(env_sorted, kernel, mode="same")
    else:
        env_smooth = env_sorted

    peak = np.nanmax(env_smooth)
    if not np.isfinite(peak) or peak <= 0:
        return np.array([])
    env_norm = env_smooth / peak

    # Convert expected null spacing to a sample-count distance.
    if len(u_abs_sorted) > 1:
        median_du = float(np.median(np.diff(u_abs_sorted)))
        min_distance = max(1, int(0.5 * expected_null_spacing / max(median_du, 1e-12)))
    else:
        min_distance = 1

    # find_peaks operates on positive peaks: negate the envelope.
    neg = -env_norm
    peaks, props = find_peaks(
        neg,
        height=-threshold,            # env_norm < threshold
        prominence=prominence_frac,   # noise rejection
        distance=min_distance,        # one peak per Bessel sidelobe
    )

    # Skip peaks below the inner-edge guard.
    peaks = peaks[u_abs_sorted[peaks] >= min_u_lambda]

    if len(peaks) == 0:
        return np.array([])

    # Keep the n_max most prominent.
    if "prominences" in props:
        prom = props["prominences"]
        # Re-extract prominences for the surviving peaks (after the guard cut)
        # by recomputing on the same neg signal.
        from scipy.signal import peak_prominences
        prom_kept = peak_prominences(neg, peaks)[0]
        order_p = np.argsort(prom_kept)[::-1][:n_max]
        peaks = np.sort(peaks[order_p])

    return u_sorted[peaks]


# ---------------------------------------------------------------------------
# Solar diameter from Bessel zeros
# ---------------------------------------------------------------------------


def solar_diameter_from_zeros(
    zero_u_lambda: np.ndarray,
    expected_diameter_arcmin: float,
    n_candidate_nulls: int,
) -> SolarDiameterResult:
    r"""Angular diameter from Bessel-function zero crossings.

    Each detected null is matched to the nearest theoretical null (computed
    from the expected diameter) so the index assignment stays consistent
    even when the first Bessel null falls outside the observed baseline range.
    """
    zeros = np.abs(zero_u_lambda)

    R_expected = np.deg2rad(expected_diameter_arcmin / 2.0 / 60.0)
    j1_all = jn_zeros(1, n_candidate_nulls)
    u_null_theory = j1_all / (2.0 * np.pi * R_expected)
    spacing = u_null_theory[1] - u_null_theory[0]

    matched_j1 = []
    matched_u  = []
    used_theory: set[int] = set()
    for u_det in np.sort(zeros):
        dists = np.abs(u_null_theory - u_det)
        best = int(np.argmin(dists))
        # Accept only within 40% of the null spacing to avoid mis-assignment.
        if dists[best] < 0.4 * spacing and best not in used_theory:
            matched_j1.append(j1_all[best])
            matched_u.append(u_det)
            used_theory.add(best)

    matched_j1 = np.array(matched_j1)
    matched_u  = np.array(matched_u)

    u_R_theory   = matched_j1 / (2.0 * np.pi)
    R_estimates  = u_R_theory / matched_u
    R_mean       = float(np.mean(R_estimates))
    R_err        = float(np.std(R_estimates) / np.sqrt(len(R_estimates))) if len(R_estimates) > 1 else float("inf")

    return SolarDiameterResult(
        diameter_arcmin=np.rad2deg(2.0 * R_mean) * 60.0,
        diameter_err_arcmin=np.rad2deg(2.0 * R_err) * 60.0,
        method="bessel_zeros",
        zero_crossings_u_R=u_R_theory,
        fitted_params={
            "R_rad_per_null": R_estimates,
            "u_observed": matched_u,
            "j1_zeros_used": matched_j1,
        },
    )


# ---------------------------------------------------------------------------
# Solar diameter from Bessel envelope fit
# ---------------------------------------------------------------------------


def fit_solar_diameter_bessel(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    envelope_std: np.ndarray,
    initial_diameter_arcmin: float,
) -> SolarDiameterResult:
    r"""Nonlinear least-squares fit of the Bessel envelope.

    Fits ``A * |2 J_1(2*pi*|u|*R) / (2*pi*|u|*R)|`` to the observed envelope.

    Uncertainties are computed in two ways and the *larger* of the two is
    reported as the conservative diameter error:

    1. The formal covariance from ``curve_fit`` with the supplied per-point
       sigmas (``absolute_sigma=True``).
    2. The same covariance rescaled by ``sqrt(chi2_reduced)`` so that
       under-estimated input sigmas (or unmodelled systematics that inflate
       the residuals) are absorbed into a residual-based error.

    A floor of ``0.05 arcmin`` is applied to guard against pathological
    near-singular covariance matrices that previously produced ``±0.00``
    diameters.
    """
    def model(u, amplitude, R_rad):
        return amplitude * uniform_disk_visibility_amplitude(u, R_rad)

    valid = (
        np.isfinite(envelope)
        & np.isfinite(u_lambda)
        & np.isfinite(envelope_std)
        & (envelope_std > 0)
    )
    u_fit = u_lambda[valid]
    env_fit = envelope[valid]
    sigma = envelope_std[valid]

    if u_fit.size < 10:
        raise ValueError("fit_solar_diameter_bessel: not enough valid points to fit.")

    R0 = np.deg2rad(initial_diameter_arcmin / 2.0 / 60.0)
    A0 = np.nanmax(env_fit)

    popt, pcov = curve_fit(
        model,
        u_fit,
        env_fit,
        p0=[A0, R0],
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0, 0], [np.inf, np.deg2rad(1.0)]),
        maxfev=10000,
    )
    amplitude_fit, R_fit = popt

    diag = np.diag(pcov)
    if np.any(~np.isfinite(diag)) or np.any(diag < 0):
        raise RuntimeError("fit_solar_diameter_bessel: covariance non-finite or negative.")

    R_err_formal = float(np.sqrt(diag[1]))

    resid = env_fit - model(u_fit, *popt)
    dof = max(1, len(u_fit) - 2)
    chi2 = float(np.sum((resid / sigma) ** 2))
    chi2_red = chi2 / dof
    R_err_rescaled = R_err_formal * np.sqrt(max(chi2_red, 1.0))

    R_err = max(R_err_formal, R_err_rescaled)

    # Sanity floor: a baseline-uncertainty-limited diameter error is
    #   dR/R ≈ d|u|/|u| ≈ d b_ew / b_ew ≈ 0.03/15 ≈ 2e-3
    # giving ~0.06 arcmin for a 32 arcmin disk. Below this, the formal
    # error is almost certainly under-estimated.
    R_err_floor = np.deg2rad(0.05 / 2.0 / 60.0)
    R_err = max(R_err, R_err_floor)

    return SolarDiameterResult(
        diameter_arcmin=np.rad2deg(2.0 * R_fit) * 60.0,
        diameter_err_arcmin=np.rad2deg(2.0 * R_err) * 60.0,
        method="bessel_fit",
        fitted_params={
            "amplitude": amplitude_fit,
            "R_rad": R_fit,
            "R_err_rad": R_err,
            "R_err_formal_rad": R_err_formal,
            "R_err_rescaled_rad": R_err_rescaled,
            "chi2": chi2,
            "chi2_reduced": chi2_red,
            "dof": dof,
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
    noise_std: np.ndarray,
    sigma_threshold: float,
    null_search_radius_frac: float,
) -> list[SunspotDetection]:
    r"""Detect sunspot signatures as excess amplitude at uniform-disk nulls.

    The spot flux fraction is approximately
    :math:`f_{\rm spot} \approx |V_{\rm null}| / |V(0)|`.
    """
    u = np.abs(u_lambda)
    residual = envelope - disk_model

    # Local minima of the disk model are the predicted nulls.
    null_indices = [
        i
        for i in range(1, len(disk_model) - 1)
        if disk_model[i] < disk_model[i - 1] and disk_model[i] < disk_model[i + 1]
    ]

    v0 = float(np.nanmax(envelope))
    detections: list[SunspotDetection] = []

    for null_idx in null_indices:
        u_null = u[null_idx]
        radius = null_search_radius_frac * u_null
        mask = np.abs(u - u_null) <= radius
        if not mask.any():
            continue

        region_resid = residual[mask]
        region_noise = noise_std[mask]

        peak_in_region = int(np.argmax(np.abs(region_resid)))
        peak_resid = region_resid[peak_in_region]
        peak_noise = region_noise[peak_in_region]
        significance = float(np.abs(peak_resid) / peak_noise)

        if significance >= sigma_threshold:
            gi = int(np.where(mask)[0][peak_in_region])
            detections.append(
                SunspotDetection(
                    null_index=len(detections) + 1,
                    u_lambda_at_null=float(u[gi]),
                    ha_rad_at_null=float("nan"),
                    residual_amplitude=float(peak_resid),
                    significance_sigma=significance,
                    estimated_flux_fraction=float(np.abs(peak_resid) / v0),
                )
            )
    return detections


def localize_sunspot_phase(
    u_lambda: np.ndarray,
    visibility_complex: np.ndarray,
    disk_radius_rad: float,
    null_index: int,
    n_nulls: int = 6,
    half_window_frac: float = 0.25,
    sigma: np.ndarray | None = None,
) -> SunspotLocalization | None:
    """Fit (flux fraction, EW offset) of a sunspot from complex visibility.

    Selects the points in a ``±half_window_frac`` window around the
    ``null_index``-th theoretical Bessel null of the uniform-disk model and
    fits, in the complex plane,

        V(u) = (1 - f) * V_disk_signed(u; R)
             + f * exp(-2 pi i * u * delta_alpha_rad)

    treating ``f`` and ``delta_alpha_rad`` as nonlinear parameters.

    Parameters
    ----------
    u_lambda
        Signed projected baseline at each capture (wavelengths).
    visibility_complex
        Complex visibility at each capture, normalised so that ``|V(u→0)| ≈ 1``.
    disk_radius_rad
        Angular radius of the (best-fit uniform) solar disk, in radians.
    null_index
        1-based Bessel null index. ``1`` selects the *first* null.
    n_nulls
        Total number of theoretical nulls to compute (must be ≥ ``null_index``).
    half_window_frac
        Half-width of the |u|-window around the null, as a fraction of the
        local null spacing ``Δu = 1/(2R)``.
    sigma
        Optional per-point uncertainty on |V|. If ``None``, an unweighted
        fit is used and the chi^2 reported is the unweighted residual sum.
    """
    j1 = jn_zeros(1, max(n_nulls, null_index))
    u_null = j1[null_index - 1] / (2.0 * np.pi * disk_radius_rad)
    half_window = half_window_frac * 0.5 / disk_radius_rad

    u_abs = np.abs(u_lambda)
    mask = (
        np.isfinite(u_abs)
        & np.isfinite(visibility_complex.real)
        & np.isfinite(visibility_complex.imag)
        & (np.abs(u_abs - u_null) <= half_window)
    )
    if mask.sum() < 8:
        return None

    u_w = u_lambda[mask].astype(float)
    V_w = visibility_complex[mask].astype(complex)
    if sigma is None:
        sig = np.ones(u_w.size)
    else:
        sig = np.where(np.isfinite(sigma[mask]) & (sigma[mask] > 0), sigma[mask], np.nan)
        if not np.all(np.isfinite(sig)):
            sig = np.ones(u_w.size)

    def residuals(params):
        f, delta_alpha = params
        V_disk = (1.0 - f) * uniform_disk_visibility_signed(u_w, disk_radius_rad)
        V_spot = f * np.exp(-2.0j * np.pi * u_w * delta_alpha)
        V_model = V_disk + V_spot
        r = (V_w - V_model) / sig
        return np.concatenate([r.real, r.imag])

    p0 = [0.05, 0.0]  # 5% spot, on-axis
    bounds = ([0.0, -np.deg2rad(0.5)], [0.5, np.deg2rad(0.5)])
    try:
        result = least_squares(residuals, p0, bounds=bounds, max_nfev=2000)
    except Exception:
        return None
    if not result.success:
        return None

    f_fit, delta_alpha_fit = result.x
    n_dof = max(1, 2 * u_w.size - 2)
    chi2_red = float(np.sum(result.fun ** 2) / n_dof)

    # Approximate covariance from the Jacobian (Gauss-Newton).
    try:
        J = result.jac
        JTJ = J.T @ J
        cov = np.linalg.inv(JTJ) * chi2_red
        f_err = float(np.sqrt(cov[0, 0]))
        da_err = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        f_err = float("nan")
        da_err = float("nan")

    return SunspotLocalization(
        flux_fraction=float(f_fit),
        flux_fraction_err=f_err,
        delta_alpha_arcmin=float(np.rad2deg(delta_alpha_fit) * 60.0),
        delta_alpha_err_arcmin=float(np.rad2deg(da_err) * 60.0),
        chi2_reduced=chi2_red,
        n_points=int(u_w.size),
        null_index=int(null_index),
    )


def characterize_sunspot_flux(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    disk_diameter_arcmin: float,
    n_nulls: int,
) -> dict:
    """Estimate sunspot flux fraction from envelope amplitude at theoretical nulls."""
    R_rad = np.deg2rad(disk_diameter_arcmin / 2.0 / 60.0)
    j1_zeros = jn_zeros(1, n_nulls)
    u_R_theory = j1_zeros / (2.0 * np.pi)
    u_null_theory = u_R_theory / R_rad

    u_abs = np.abs(u_lambda)
    v0 = float(np.nanmax(envelope))

    null_amps  = np.empty(n_nulls)
    null_u_obs = np.empty(n_nulls)
    for k, u_null in enumerate(u_null_theory):
        idx = int(np.argmin(np.abs(u_abs - u_null)))
        null_amps[k]  = envelope[idx]
        null_u_obs[k] = u_abs[idx]

    return {
        "null_u_R":         u_R_theory,
        "null_u_observed":  null_u_obs,
        "null_amplitude":   null_amps,
        "flux_fraction":    null_amps / v0,
    }
