from dataclasses import dataclass
from typing import Sequence

import numpy as np
import ugradio.doppler
from scipy.optimize import curve_fit

from ..models import Spectrum

HI_REST_FREQ_HZ = 1420.405751768e6
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class ZenithLSRCorrection:
    ra_deg: float
    dec_deg: float
    jd: float
    velocity_kms: float


@dataclass(frozen=True)
class HIRatioProfile:
    rest_freq_hz: float
    numerator: Spectrum
    denominator: Spectrum
    baseband_mhz: np.ndarray
    raw_ratio: np.ndarray
    ratio_sigma: np.ndarray
    smooth_ratio: np.ndarray
    numerator_topo_velocity_kms: np.ndarray
    denominator_topo_velocity_kms: np.ndarray
    numerator_velocity_kms: np.ndarray
    denominator_velocity_kms: np.ndarray
    velocity_shift_kms: float
    peak_profile: np.ndarray
    peak_profile_smooth: np.ndarray
    peak_sigma: np.ndarray
    dip_profile: np.ndarray
    dip_profile_smooth: np.ndarray
    dip_sigma: np.ndarray


@dataclass(frozen=True)
class GaussianComponentGuess:
    amplitude: float
    center_kms: float
    sigma_kms: float


@dataclass(frozen=True)
class GaussianComponentFit:
    amplitude: float
    center_kms: float
    sigma_kms: float
    amplitude_err: float
    center_err: float
    sigma_err: float

    @property
    def fwhm_kms(self) -> float:
        return FWHM_FACTOR * self.sigma_kms

    @property
    def fwhm_err_kms(self) -> float:
        return FWHM_FACTOR * self.sigma_err


@dataclass(frozen=True)
class HIProfileFit:
    label: str
    component_guesses: tuple[GaussianComponentGuess, ...]
    baseline_poly_order: int
    fit_min_kms: float
    fit_max_kms: float
    popt: np.ndarray
    perr: np.ndarray
    mask: np.ndarray
    chi2_red: float

    @property
    def n_components(self) -> int:
        return len(self.component_guesses)

    @property
    def n_baseline_coeffs(self) -> int:
        return self.baseline_poly_order + 1

    @property
    def components(self) -> tuple[GaussianComponentFit, ...]:
        return tuple(
            GaussianComponentFit(
                amplitude=float(self.popt[3 * i]),
                center_kms=float(self.popt[3 * i + 1]),
                sigma_kms=float(self.popt[3 * i + 2]),
                amplitude_err=float(self.perr[3 * i]),
                center_err=float(self.perr[3 * i + 1]),
                sigma_err=float(self.perr[3 * i + 2]),
            )
            for i in range(self.n_components)
        )

    @property
    def baseline_coeffs(self) -> np.ndarray:
        return self.popt[3 * self.n_components:]

    @property
    def baseline_coeff_errors(self) -> np.ndarray:
        return self.perr[3 * self.n_components:]

    def model(self, vel_kms: np.ndarray) -> np.ndarray:
        return _profile_model(
            vel_kms,
            self.popt,
            self.n_components,
        )

    def baseline(self, vel_kms: np.ndarray) -> np.ndarray:
        return polynomial_baseline(vel_kms, *self.baseline_coeffs)

    def component(self, vel_kms: np.ndarray, index: int) -> np.ndarray:
        component = self.components[index]
        return gaussian_mixture(
            vel_kms,
            component.amplitude,
            component.center_kms,
            component.sigma_kms,
        )

    def print_summary(self) -> None:
        print(
            f'\n{self.label}  ({self.n_components}-component fit + '
            f'poly{self.baseline_poly_order}, {int(self.mask.sum())} channels, '
            f'chi^2_r = {self.chi2_red:.3f})'
        )
        print(
            f'  {"Comp":>4}  {"A":>10}  {"v0_topo (km/s)":>15}  '
            f'{"sigma (km/s)":>13}  {"FWHM (km/s)":>12}'
        )
        print(f'  {"-" * 4}  {"-" * 10}  {"-" * 15}  {"-" * 13}  {"-" * 12}')
        for idx, component in enumerate(self.components, start=1):
            print(
                f'  {idx:>4}  '
                f'{component.amplitude:>8.5f}+/-{component.amplitude_err:<8.5f}  '
                f'{component.center_kms:>+12.2f}+/-{component.center_err:<7.2f}  '
                f'{component.sigma_kms:>10.2f}+/-{component.sigma_err:<7.2f}  '
                f'{component.fwhm_kms:>9.2f}+/-{component.fwhm_err_kms:<7.2f}'
            )
        for idx, (coeff, coeff_err) in enumerate(
                zip(self.baseline_coeffs, self.baseline_coeff_errors)
        ):
            print(f'  baseline c{idx}: {coeff:+.6f} +/- {coeff_err:.6f}  (x = v/100)')


@dataclass(frozen=True)
class ToyHIRatioSimulation:
    signal_velocity_kms: np.ndarray
    reference_velocity_kms: np.ndarray
    continuum_signal: np.ndarray
    continuum_reference: np.ndarray
    line_signal: np.ndarray
    line_reference: np.ndarray
    sky_signal: np.ndarray
    sky_reference: np.ndarray
    power_signal: np.ndarray
    power_reference: np.ndarray
    ratio: np.ndarray
    inverse_ratio: np.ndarray
    lo_separation_kms: float


def zenith_lsr_correction(spectrum: Spectrum) -> ZenithLSRCorrection:
    """Projected LSR correction for a zenith observation."""
    ra_deg = float(np.degrees(spectrum.lst))
    dec_deg = float(spectrum.obs_lat)
    velocity_ms = ugradio.doppler.get_projected_velocity(
        ra=ra_deg,
        dec=dec_deg,
        jd=spectrum.jd,
        obs_lat=spectrum.obs_lat,
        obs_lon=spectrum.obs_lon,
        obs_alt=spectrum.obs_alt,
    )
    return ZenithLSRCorrection(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        jd=float(spectrum.jd),
        velocity_kms=float(velocity_ms / 1e3),
    )


def extract_hi_ratio_profile(
        numerator: Spectrum,
        denominator: Spectrum,
        *,
        rest_freq_hz: float = HI_REST_FREQ_HZ,
        smooth_kwargs: dict | None = None,
        velocity_shift_kms: float = 0.0,
) -> HIRatioProfile:
    """Extract raw and smoothed ratio-space HI profiles from a dual-LO pair."""
    if smooth_kwargs is None:
        smooth_kwargs = dict(method='savgol', window_length=257, polyorder=3)

    raw_ratio = numerator.ratio_to(denominator)
    ratio_sigma = numerator.ratio_std_to(denominator)
    smooth_ratio = numerator.ratio_to(denominator, smooth_kwargs=smooth_kwargs)

    with np.errstate(divide='ignore', invalid='ignore'):
        dip_profile = 1.0 / raw_ratio - 1.0
        dip_sigma = ratio_sigma / raw_ratio**2
        dip_profile_smooth = 1.0 / smooth_ratio - 1.0

    return HIRatioProfile(
        rest_freq_hz=float(rest_freq_hz),
        numerator=numerator,
        denominator=denominator,
        baseband_mhz=numerator.frequency_axis_mhz(mode='baseband'),
        raw_ratio=raw_ratio,
        ratio_sigma=ratio_sigma,
        smooth_ratio=smooth_ratio,
        numerator_topo_velocity_kms=numerator.velocity_axis_kms(rest_freq_hz),
        denominator_topo_velocity_kms=denominator.velocity_axis_kms(rest_freq_hz),
        numerator_velocity_kms=numerator.velocity_axis_kms(
            rest_freq_hz,
            velocity_shift_kms=velocity_shift_kms,
        ),
        denominator_velocity_kms=denominator.velocity_axis_kms(
            rest_freq_hz,
            velocity_shift_kms=velocity_shift_kms,
        ),
        velocity_shift_kms=float(velocity_shift_kms),
        peak_profile=raw_ratio - 1.0,
        peak_profile_smooth=smooth_ratio - 1.0,
        peak_sigma=ratio_sigma,
        dip_profile=dip_profile,
        dip_profile_smooth=dip_profile_smooth,
        dip_sigma=dip_sigma,
    )


def gaussian_mixture(vel_kms: np.ndarray, *params: float) -> np.ndarray:
    """Sum of Gaussian components stored as [A1, v1, s1, A2, v2, s2, ...]."""
    result = np.zeros_like(vel_kms, dtype=float)
    for idx in range(len(params) // 3):
        amplitude = params[3 * idx]
        center = params[3 * idx + 1]
        sigma = params[3 * idx + 2]
        result += amplitude * np.exp(-0.5 * ((vel_kms - center) / sigma) ** 2)
    return result


def polynomial_baseline(vel_kms: np.ndarray, *coeffs: float) -> np.ndarray:
    """Polynomial baseline in x = v / 100 for stable coefficients."""
    scaled_vel = vel_kms / 100.0
    result = np.zeros_like(vel_kms, dtype=float)
    for order, coeff in enumerate(coeffs):
        result += coeff * scaled_vel**order
    return result


def fit_hi_profile(
        vel_kms: np.ndarray,
        profile_raw: np.ndarray,
        sigma_profile: np.ndarray,
        *,
        initial_guesses: Sequence[GaussianComponentGuess | tuple[float, float, float]],
        baseline_poly_order: int = 1,
        fit_min_kms: float = -120.0,
        fit_max_kms: float = 120.0,
        label: str = '',
) -> HIProfileFit:
    """Fit Gaussian components plus a low-order polynomial baseline."""
    component_guesses = tuple(_coerce_guess(guess) for guess in initial_guesses)
    mask = (
        np.isfinite(vel_kms)
        & np.isfinite(profile_raw)
        & np.isfinite(sigma_profile)
        & (sigma_profile > 0)
        & (vel_kms >= fit_min_kms)
        & (vel_kms <= fit_max_kms)
    )
    v_fit = vel_kms[mask]
    p_fit = profile_raw[mask]
    s_fit = sigma_profile[mask]

    n_components = len(component_guesses)
    n_base = baseline_poly_order + 1
    p0_gauss = [
        value
        for guess in component_guesses
        for value in (guess.amplitude, guess.center_kms, guess.sigma_kms)
    ]

    outer = np.abs(v_fit) > 80.0
    if outer.sum() <= baseline_poly_order:
        outer = np.ones_like(v_fit, dtype=bool)
    coeff_high_to_low = np.polyfit(
        v_fit[outer] / 100.0,
        p_fit[outer],
        deg=baseline_poly_order,
        w=1.0 / np.maximum(s_fit[outer], 1e-9),
    )
    p0_base = coeff_high_to_low[::-1].tolist()
    p0 = p0_gauss + p0_base

    lower_gauss = [0.0, -200.0, 1.0] * n_components
    upper_gauss = [np.inf, 200.0, 200.0] * n_components
    lower_base = [-np.inf] * n_base
    upper_base = [np.inf] * n_base

    popt, pcov = curve_fit(
        lambda vel, *params: _profile_model(vel, np.asarray(params), n_components),
        v_fit,
        p_fit,
        p0=p0,
        sigma=s_fit,
        absolute_sigma=True,
        bounds=(lower_gauss + lower_base, upper_gauss + upper_base),
        maxfev=50000,
    )
    perr = np.sqrt(np.diag(pcov))
    residuals = p_fit - _profile_model(v_fit, popt, n_components)
    chi2_red = float(np.sum((residuals / s_fit) ** 2) / max(mask.sum() - len(p0), 1))

    return HIProfileFit(
        label=label,
        component_guesses=component_guesses,
        baseline_poly_order=baseline_poly_order,
        fit_min_kms=float(fit_min_kms),
        fit_max_kms=float(fit_max_kms),
        popt=np.asarray(popt, dtype=float),
        perr=np.asarray(perr, dtype=float),
        mask=mask,
        chi2_red=chi2_red,
    )


def print_lsr_fit_summary(
        fits: Sequence[tuple[str, HIProfileFit]],
        *,
        velocity_shift_kms: float,
) -> None:
    """Print topocentric and LSR centroid summaries for fitted components."""
    print(
        f'\nLSR correction: Delta v = {velocity_shift_kms:+.4f} km/s  '
        f'(v_LSR = v_topo + Delta v)'
    )
    print(f'  {"Band":<12}  {"Comp":>4}  {"v0_topo (km/s)":>16}  {"v0_LSR (km/s)":>15}')
    print(f'  {"-" * 12}  {"-" * 4}  {"-" * 16}  {"-" * 15}')
    for band_label, fit in fits:
        for idx, component in enumerate(fit.components, start=1):
            v0_lsr = component.center_kms + velocity_shift_kms
            print(
                f'  {band_label:<12}  {idx:>4}  '
                f'{component.center_kms:>+13.2f}+/-{component.center_err:<5.2f}  '
                f'{v0_lsr:>+12.2f}+/-{component.center_err:<5.2f}'
            )


def simulate_hi_ratio_signature(
        template_spectrum: Spectrum,
        component_params: Sequence[float],
        *,
        signal_center_freq_hz: float = 1420e6,
        reference_center_freq_hz: float = 1421e6,
        rest_freq_hz: float = HI_REST_FREQ_HZ,
) -> ToyHIRatioSimulation:
    """Toy continuum plus line simulation for the two-LO ratio method."""
    component_params = np.asarray(component_params, dtype=float)
    delta_f_hz = template_spectrum.freqs - template_spectrum.center_freq

    signal_freqs = signal_center_freq_hz + delta_f_hz
    reference_freqs = reference_center_freq_hz + delta_f_hz

    signal_velocity = 2.99792458e5 * (rest_freq_hz - signal_freqs) / rest_freq_hz
    reference_velocity = 2.99792458e5 * (rest_freq_hz - reference_freqs) / rest_freq_hz

    continuum_signal = 1.0 + 0.05 * (signal_velocity / 250.0) + 0.03 * (signal_velocity / 250.0) ** 2
    continuum_reference = 1.0 + 0.05 * (reference_velocity / 250.0) + 0.03 * (reference_velocity / 250.0) ** 2
    line_signal = gaussian_mixture(signal_velocity, *component_params)
    line_reference = gaussian_mixture(reference_velocity, *component_params)
    sky_signal = continuum_signal + line_signal
    sky_reference = continuum_reference + line_reference

    x = delta_f_hz / np.max(np.abs(delta_f_hz))
    bandpass = 1.0 - 0.30 * x**2 + 0.05 * x**4

    power_signal = bandpass * sky_signal
    power_reference = bandpass * sky_reference
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = power_signal / power_reference
        inverse_ratio = power_reference / power_signal

    lo_separation_kms = float(
        2.99792458e5 * (reference_center_freq_hz - signal_center_freq_hz) / rest_freq_hz
    )
    return ToyHIRatioSimulation(
        signal_velocity_kms=signal_velocity,
        reference_velocity_kms=reference_velocity,
        continuum_signal=continuum_signal,
        continuum_reference=continuum_reference,
        line_signal=line_signal,
        line_reference=line_reference,
        sky_signal=sky_signal,
        sky_reference=sky_reference,
        power_signal=power_signal,
        power_reference=power_reference,
        ratio=ratio,
        inverse_ratio=inverse_ratio,
        lo_separation_kms=lo_separation_kms,
    )


def _coerce_guess(
        guess: GaussianComponentGuess | tuple[float, float, float],
) -> GaussianComponentGuess:
    if isinstance(guess, GaussianComponentGuess):
        return guess
    amplitude, center_kms, sigma_kms = guess
    return GaussianComponentGuess(
        amplitude=float(amplitude),
        center_kms=float(center_kms),
        sigma_kms=float(sigma_kms),
    )


def _profile_model(
        vel_kms: np.ndarray,
        params: np.ndarray,
        n_components: int,
) -> np.ndarray:
    gauss_params = params[: 3 * n_components]
    base_params = params[3 * n_components:]
    return gaussian_mixture(vel_kms, *gauss_params) + polynomial_baseline(
        vel_kms,
        *base_params,
    )
