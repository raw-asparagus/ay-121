from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal as sig
import ugradio.doppler

from ugradiolab import Spectrum

from .constants import C_LIGHT_KMS, HARDWARE_RESPONSE_MIN, HI_REST_FREQ_HZ, RFI_SIGMA, ROLLING_WIDTH, SAVGOL


def load_lo_pair(spectra_dir: Path) -> dict[int, Spectrum]:
    files = sorted(Path(spectra_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {spectra_dir}")
    pairs: dict[int, Spectrum] = {}
    for path in files:
        spec = Spectrum.load(path)
        pairs[int(round(spec.center_freq / 1e6))] = spec
    if 1420 not in pairs or 1421 not in pairs:
        raise ValueError(f"Expected LO 1420 and 1421 in {spectra_dir}, got {list(pairs)}")
    return pairs


def lo_center_bin_index(spectrum: Spectrum) -> int:
    return int(spectrum.bin_at(float(spectrum.center_freq)))


def lo_analysis_mask(spectrum: Spectrum) -> np.ndarray:
    mask = np.ones(np.asarray(spectrum.psd).shape, dtype=bool)
    mask[lo_center_bin_index(spectrum)] = False
    if not np.any(mask):
        raise ValueError("LO-analysis mask removed every channel.")
    return mask


def combine_spectrum_mask(
    spectrum: Spectrum,
    *masks: np.ndarray,
    require_nonempty: bool = False,
) -> np.ndarray:
    combined = lo_analysis_mask(spectrum)
    for mask in masks:
        arr = np.asarray(mask, bool)
        if arr.shape != combined.shape:
            raise ValueError("Mask shape mismatch.")
        combined &= arr
    if require_nonempty and not np.any(combined):
        raise ValueError("Combined mask is empty.")
    return combined


def masked_spectrum_values(
    spectrum: Spectrum,
    values: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    base = np.asarray(spectrum.psd if values is None else values, float)
    arr = np.array(base, dtype=float, copy=True)
    use_mask = combine_spectrum_mask(spectrum, mask) if mask is not None else lo_analysis_mask(spectrum)
    arr[~use_mask] = fill_value
    return arr


def fill_masked_spectrum_values(
    spectrum: Spectrum,
    values: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = masked_spectrum_values(spectrum, values=values, mask=mask)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise ValueError("Need at least two finite channels to interpolate across the LO bin.")
    idx = np.arange(arr.size, dtype=float)
    filled = arr.copy()
    filled[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return filled


def smooth_series(values: np.ndarray, smooth_kwargs: dict) -> np.ndarray:
    arr = np.asarray(values, float)
    method = smooth_kwargs.get("method", "savgol")
    if method == "savgol":
        window_length = int(smooth_kwargs.get("window_length", 129))
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= arr.size:
            window_length = arr.size - 1 if arr.size % 2 == 0 else arr.size
        if window_length < 3:
            return arr.copy()
        polyorder = min(int(smooth_kwargs.get("polyorder", 3)), window_length - 1)
        return sig.savgol_filter(arr, window_length=window_length, polyorder=polyorder, mode="interp")
    if method == "boxcar":
        width = int(smooth_kwargs.get("M", 64))
        if width <= 1:
            return arr.copy()
        kernel = np.ones(width, float) / width
        return np.convolve(arr, kernel, mode="same")
    if method == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(arr, sigma=float(smooth_kwargs.get("sigma", 32)))
    raise ValueError(f"Unknown smoothing method {method!r}")


def interp_mono(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_new: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_new = np.asarray(x_new, float)
    finite = np.isfinite(x_src) & np.isfinite(y_src)
    if finite.sum() < 2:
        return np.full_like(x_new, fill_value, dtype=float)
    xs = x_src[finite]
    ys = y_src[finite]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    out = np.interp(x_new, xs, ys)
    out[(x_new < xs[0]) | (x_new > xs[-1])] = fill_value
    return out


def interp_bool_nearest(
    x_src: np.ndarray,
    mask_src: np.ndarray,
    x_new: np.ndarray,
    default: bool = False,
) -> np.ndarray:
    xs = np.asarray(x_src, float)
    ms = np.asarray(mask_src, bool)
    xn = np.asarray(x_new, float)
    finite = np.isfinite(xs)
    if finite.sum() == 0:
        return np.full_like(xn, default, dtype=bool)
    xs = xs[finite]
    ms = ms[finite]
    order = np.argsort(xs)
    xs = xs[order]
    ms = ms[order]
    idx = np.searchsorted(xs, xn, side="left")
    idx = np.clip(idx, 0, len(xs) - 1)
    prev_idx = np.clip(idx - 1, 0, len(xs) - 1)
    choose_prev = np.abs(xn - xs[prev_idx]) <= np.abs(xn - xs[idx])
    nearest = np.where(choose_prev, prev_idx, idx)
    out = ms[nearest]
    out[(xn < xs[0]) | (xn > xs[-1])] = default
    return out


def sigma_clip_rfi_mask(
    spectrum: Spectrum,
    n_sigma: float = RFI_SIGMA,
    smooth_kwargs: dict = SAVGOL,
) -> np.ndarray:
    psd = masked_spectrum_values(spectrum)
    baseline_seed = fill_masked_spectrum_values(spectrum)
    baseline = masked_spectrum_values(spectrum, smooth_series(baseline_seed, smooth_kwargs))
    resid = psd - baseline

    w = ROLLING_WIDTH
    half = w // 2
    n = len(resid)
    local_std = np.full(n, np.nan)
    for idx in range(n):
        lo = max(0, idx - half)
        hi = min(n, idx + half + 1)
        window = resid[lo:hi]
        finite = np.isfinite(window)
        if np.any(finite):
            local_std[idx] = np.nanstd(window[finite])

    good = np.isfinite(resid) & np.isfinite(local_std)
    good &= np.abs(resid) <= n_sigma * np.maximum(local_std, 1e-12)
    return combine_spectrum_mask(spectrum, good)


def apply_mask_to_psd(spectrum: Spectrum, mask: np.ndarray) -> np.ndarray:
    return masked_spectrum_values(spectrum, mask=mask)


def masked_total_power(
    spectrum: Spectrum,
    mask: np.ndarray,
    psd_override: np.ndarray | None = None,
    std_override: np.ndarray | None = None,
) -> tuple[float, float]:
    combined = combine_spectrum_mask(spectrum, mask, require_nonempty=True)
    psd_full = (
        masked_spectrum_values(spectrum, values=psd_override, mask=combined)
        if psd_override is not None
        else masked_spectrum_values(spectrum, mask=combined)
    )
    std_values = spectrum.std if std_override is None else std_override
    std_full = masked_spectrum_values(spectrum, values=std_values, mask=combined)
    psd = psd_full[combined]
    std = std_full[combined]
    return float(np.sum(psd)), float(np.sqrt(np.sum(std**2)))


def masked_mean_power(
    spectrum: Spectrum,
    mask: np.ndarray,
    psd_override: np.ndarray | None = None,
    std_override: np.ndarray | None = None,
) -> tuple[float, float]:
    combined = combine_spectrum_mask(spectrum, mask, require_nonempty=True)
    psd_full = (
        masked_spectrum_values(spectrum, values=psd_override, mask=combined)
        if psd_override is not None
        else masked_spectrum_values(spectrum, mask=combined)
    )
    std_values = spectrum.std if std_override is None else std_override
    std_full = masked_spectrum_values(spectrum, values=std_values, mask=combined)
    psd = psd_full[combined]
    std = std_full[combined]
    sigma_sum = float(np.sqrt(np.sum(std**2)))
    return float(np.mean(psd)), float(sigma_sum / max(int(psd.size), 1))


def omit_lo_center_bin_mask(spectrum: Spectrum, mask: np.ndarray) -> tuple[np.ndarray, int]:
    return combine_spectrum_mask(spectrum, mask, require_nonempty=True), lo_center_bin_index(spectrum)


def velocity_axis(freqs_hz: np.ndarray) -> np.ndarray:
    return C_LIGHT_KMS * (HI_REST_FREQ_HZ - np.asarray(freqs_hz, float)) / HI_REST_FREQ_HZ


def lsr_correction_kms(spectrum: Spectrum) -> float:
    has_altaz = hasattr(spectrum, "az") and hasattr(spectrum, "alt")
    if not has_altaz:
        raise ValueError("Spectrum is missing az/alt metadata for LSR correction.")
    v_ms = ugradio.doppler.get_projected_velocity(
        spectrum.jd,
        spectrum.alt,
        spectrum.az,
        spectrum.obs_lat,
        spectrum.obs_lon,
    )
    return float(v_ms / 1e3)


def apply_hardware_response_correction(
    values: np.ndarray,
    response: np.ndarray,
    response_floor: float,
) -> np.ndarray:
    arr = np.asarray(values, float)
    resp = np.asarray(response, float)
    if arr.shape != resp.shape:
        raise ValueError("values and response must have matching shapes.")
    floor = max(float(response_floor), HARDWARE_RESPONSE_MIN)
    resp_safe = np.clip(resp, floor, np.inf)
    out = arr / resp_safe
    out[~np.isfinite(resp)] = np.nan
    return out
