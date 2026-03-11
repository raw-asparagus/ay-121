from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import astropy.coordinates as astro_coord
import astropy.time as astro_time
import astropy.units as astro_u
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.signal as sig
import ugradio.coord
import ugradio.doppler

from ugradiolab import Spectrum

from .common import (
    combine_spectrum_mask,
    fill_masked_spectrum_values,
    interp_bool_nearest,
    interp_mono,
    load_lo_pair,
    masked_spectrum_values,
    smooth_series,
    sigma_clip_rfi_mask,
)
from .constants import (
    C_LIGHT_KMS,
    FIT_SEED,
    FIT_WINDOWS_KMS,
    FWHM_PHYS_MAX_KMS,
    FWHM_PHYS_MIN_KMS,
    HI_REST_FREQ_HZ,
    LO_FREQS_MHZ,
    MAX_SMOOTH_KMS,
    N_GAUSS_GRID,
    NAIVE_MULTI_STARTS,
    NAIVE_SIG_MAX_KMS,
    NAIVE_SIG_MIN_KMS,
    OFFLINE_MIN_KMS,
    PHYSICS_WIDTH_EDGE_MARGIN_FRAC,
    PHYSICS_WIDTH_PENALTY_LAMBDA,
    POLY_GRID,
    SIGMA_PHYS_MAX_KMS,
    SIGMA_PHYS_MIN_KMS,
    SMOOTH_METHOD,
    SMOOTH_NCHAN,
)
from .contracts import AnalysisResult
from .io import load_equipment_artifact, load_temperature_artifact
from .paths import CYGNUS_X_SPECTRA_DIR, ETA_EFF_ESTIMATE_PATH, STANDARD_SPECTRA_DIR, ensure_output_dirs
from .plotting import (
    plot_dataset_fits,
    plot_hyperfine,
    plot_lsr_geometry,
    plot_mean_vs_median,
    plot_ratio_profile,
)


def velocity_axis(freqs_hz: np.ndarray, rest_freq_hz: float = HI_REST_FREQ_HZ) -> np.ndarray:
    return C_LIGHT_KMS * (rest_freq_hz - np.asarray(freqs_hz, float)) / rest_freq_hz


def lsr_correction_kms(spectrum: Spectrum) -> float:
    ra_deg = np.nan
    dec_deg = np.nan
    has_altaz = hasattr(spectrum, "az") and hasattr(spectrum, "alt")
    if has_altaz:
        az_deg = float(getattr(spectrum, "az"))
        alt_deg = float(getattr(spectrum, "alt"))
        if np.isfinite(az_deg) and np.isfinite(alt_deg):
            location = astro_coord.EarthLocation(
                lat=float(spectrum.obs_lat) * astro_u.deg,
                lon=float(spectrum.obs_lon) * astro_u.deg,
                height=float(spectrum.obs_alt) * astro_u.m,
            )
            obstime = astro_time.Time(float(spectrum.jd), format="jd")
            frame_altaz = astro_coord.AltAz(obstime=obstime, location=location)
            sky_altaz = astro_coord.SkyCoord(az=az_deg * astro_u.deg, alt=alt_deg * astro_u.deg, frame=frame_altaz)
            sky_icrs = sky_altaz.icrs
            ra_deg = float(sky_icrs.ra.deg)
            dec_deg = float(sky_icrs.dec.deg)
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        ra_deg = float(np.degrees(float(spectrum.lst)))
        dec_deg = float(spectrum.obs_lat)
    v_ms = ugradio.doppler.get_projected_velocity(
        ra=ra_deg,
        dec=dec_deg,
        jd=spectrum.jd,
        obs_lat=spectrum.obs_lat,
        obs_lon=spectrum.obs_lon,
        obs_alt=spectrum.obs_alt,
    )
    return float(v_ms / 1e3)


def robust_mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-12)


def smooth_nanboxcar(y: np.ndarray, nchan: int) -> np.ndarray:
    y = np.asarray(y, float)
    if nchan <= 1:
        return y.copy()
    if nchan % 2 == 0:
        nchan += 1
    good = np.isfinite(y).astype(float)
    y0 = np.where(np.isfinite(y), y, 0.0)
    kernel = np.ones(nchan, float)
    num = np.convolve(y0, kernel, mode="same")
    den = np.convolve(good, kernel, mode="same")
    out = np.full_like(y, np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def smooth_nanmedian(y: np.ndarray, nchan: int) -> np.ndarray:
    y = np.asarray(y, float)
    if nchan <= 1:
        return y.copy()
    if nchan % 2 == 0:
        nchan += 1
    half = nchan // 2
    out = np.full_like(y, np.nan, dtype=float)
    for idx in range(y.size):
        seg = y[max(0, idx - half) : min(y.size, idx + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[idx] = np.nanmedian(seg)
    return out


def smooth_profile_with_sigma(y: np.ndarray, sigma: np.ndarray, nchan: int, method: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    if nchan <= 1:
        return y.copy(), sigma.copy()
    if method != "mean":
        raise ValueError(f"Only mean smoothing is supported, got {method!r}")
    y_sm = smooth_nanboxcar(y, nchan)
    var = np.where(np.isfinite(sigma), sigma**2, np.nan)
    var_sm = smooth_nanboxcar(var, nchan)
    sig_sm = np.sqrt(np.clip(var_sm / max(nchan, 1), 0.0, np.inf))
    return y_sm, sig_sm


def _eval_poly(v: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = np.asarray(v, float) / 100.0
    out = np.zeros_like(x)
    for idx, coeff in enumerate(np.asarray(coeffs, float)):
        out += coeff * x**idx
    return out


@dataclass(frozen=True)
class FitResult:
    n_gauss: int
    poly_order: int
    popt: np.ndarray
    perr: np.ndarray
    chi2: float
    dof: int
    chi2_red: float
    vel_fit: np.ndarray
    data_fit: np.ndarray
    sigma_fit: np.ndarray

    def model(self, vel: np.ndarray) -> np.ndarray:
        return gauss_poly_model(vel, self.popt, self.n_gauss, self.poly_order)


def gauss_poly_model(vel: np.ndarray, params: np.ndarray, n_gauss: int, poly_order: int) -> np.ndarray:
    vel = np.asarray(vel, float)
    params = np.asarray(params, float)
    out = np.zeros_like(vel)
    for idx in range(n_gauss):
        A = params[3 * idx]
        mu = params[3 * idx + 1]
        sig_k = max(params[3 * idx + 2], 1e-6)
        out += A * np.exp(-0.5 * ((vel - mu) / sig_k) ** 2)
    out += _eval_poly(vel, params[3 * n_gauss : 3 * n_gauss + poly_order + 1])
    return out


def fit_weighted_baseline_seed(v: np.ndarray, y: np.ndarray, s: np.ndarray, poly_order: int, outer_kms: float = OFFLINE_MIN_KMS) -> np.ndarray:
    use_outer = np.abs(v) > outer_kms
    if use_outer.sum() <= poly_order:
        use_outer = np.ones_like(v, dtype=bool)
    coeff_high_to_low = np.polyfit(v[use_outer] / 100.0, y[use_outer], deg=poly_order, w=1.0 / np.maximum(s[use_outer], 1e-9))
    return coeff_high_to_low[::-1]


def find_gaussian_seeds(v: np.ndarray, y_line: np.ndarray, n_gauss: int, vel_min: float, vel_max: float):
    finite = np.isfinite(v) & np.isfinite(y_line) & (v >= vel_min) & (v <= vel_max)
    v_use = np.asarray(v[finite], float)
    y_use = np.asarray(y_line[finite], float)
    if v_use.size < max(8, n_gauss):
        raise ValueError("Insufficient channels for seed finding.")
    order = np.argsort(v_use)
    v_use = v_use[order]
    y_use = y_use[order]
    n = v_use.size
    win = max(7, min(61, (n // 6) * 2 + 1))
    if win >= n:
        win = n - 1 if n % 2 == 0 else n
    if win % 2 == 0:
        win -= 1
    if win < 5:
        win = 5 if n >= 5 else max(3, n | 1)
    y_smooth = sig.savgol_filter(y_use, window_length=win, polyorder=2, mode="interp") if 5 <= win < n else y_use.copy()
    resid_scale = robust_mad_sigma(y_use - y_smooth)
    y_scale = max(float(np.nanmax(np.abs(y_smooth))), resid_scale, 1e-6)
    dv = np.nanmedian(np.diff(v_use))
    dv = float(abs(dv)) if np.isfinite(dv) and dv != 0 else 1.0
    min_dist = max(2, int(round(12.0 / dv)))
    peaks, props = sig.find_peaks(y_smooth, prominence=max(2.0 * resid_scale, 0.05 * y_scale), distance=min_dist)
    amps: list[float] = []
    mus: list[float] = []
    sigmas: list[float] = []
    if peaks.size > 0:
        ord2 = np.argsort(props["prominences"])[::-1]
        peaks = peaks[ord2]
        widths, _, _, _ = sig.peak_widths(y_smooth, peaks, rel_height=0.5)
        for pidx, width in zip(peaks, widths):
            if len(amps) >= n_gauss:
                break
            amps.append(float(max(y_smooth[pidx], 0.01 * y_scale)))
            mus.append(float(v_use[pidx]))
            sigmas.append(float(np.clip((width * dv) / 2.355, NAIVE_SIG_MIN_KMS, NAIVE_SIG_MAX_KMS)))
    if len(amps) < n_gauss:
        fallback_mu = np.linspace(vel_min + 0.2 * (vel_max - vel_min), vel_max - 0.2 * (vel_max - vel_min), n_gauss)
        base_amp = max(np.percentile(np.maximum(y_use, 0.0), 90), 0.02 * y_scale, 1e-4)
        while len(amps) < n_gauss:
            idx = len(amps)
            amps.append(float(base_amp * (0.9**idx)))
            mus.append(float(fallback_mu[idx]))
            sigmas.append(float(0.5 * (SIGMA_PHYS_MIN_KMS + SIGMA_PHYS_MAX_KMS)))
    A0 = np.asarray(amps[:n_gauss], float)
    mu0 = np.asarray(mus[:n_gauss], float)
    sig0 = np.asarray(sigmas[:n_gauss], float)
    order = np.argsort(mu0)
    return A0[order], mu0[order], sig0[order]


def _reorder_component_params(params: np.ndarray, errors: np.ndarray, n_gauss: int, poly_order: int):
    mu = np.array([params[3 * idx + 1] for idx in range(n_gauss)])
    order = np.argsort(mu)
    p2 = params.copy()
    e2 = errors.copy()
    for out_k, src_k in enumerate(order):
        p2[3 * out_k : 3 * out_k + 3] = params[3 * src_k : 3 * src_k + 3]
        e2[3 * out_k : 3 * out_k + 3] = errors[3 * src_k : 3 * src_k + 3]
    p2[3 * n_gauss : 3 * n_gauss + poly_order + 1] = params[3 * n_gauss : 3 * n_gauss + poly_order + 1]
    e2[3 * n_gauss : 3 * n_gauss + poly_order + 1] = errors[3 * n_gauss : 3 * n_gauss + poly_order + 1]
    return p2, e2


def _physics_width_penalty_terms(theta: np.ndarray, n_gauss: int) -> np.ndarray:
    sigmas = np.array([theta[3 * idx + 2] for idx in range(n_gauss)], dtype=float)
    if sigmas.size == 0:
        return np.array([], dtype=float)
    scale = max(SIGMA_PHYS_MAX_KMS - SIGMA_PHYS_MIN_KMS, 1e-6)
    edge_margin = max(PHYSICS_WIDTH_EDGE_MARGIN_FRAC * scale, 1e-6)
    lower_soft = SIGMA_PHYS_MIN_KMS + edge_margin
    upper_soft = SIGMA_PHYS_MAX_KMS - edge_margin
    near_low = np.clip(lower_soft - sigmas, 0.0, np.inf)
    near_high = np.clip(sigmas - upper_soft, 0.0, np.inf)
    return np.concatenate([near_low / edge_margin, near_high / edge_margin])


def _fit_n_phys_viol(params: np.ndarray, n_gauss: int) -> int:
    sigmas = np.array([params[3 * idx + 2] for idx in range(n_gauss)], dtype=float)
    bad = (~np.isfinite(sigmas)) | (sigmas < SIGMA_PHYS_MIN_KMS) | (sigmas > SIGMA_PHYS_MAX_KMS)
    return int(np.sum(bad))


def gauss_poly_fit(
    vel: np.ndarray,
    profile: np.ndarray,
    sigma: np.ndarray,
    n_gauss: int,
    poly_order: int,
    vel_min: float,
    vel_max: float,
    n_multistart: int = NAIVE_MULTI_STARTS,
    random_seed: int = FIT_SEED,
) -> FitResult:
    mask = np.isfinite(vel) & np.isfinite(profile) & np.isfinite(sigma) & (sigma > 0) & (vel >= vel_min) & (vel <= vel_max)
    v = np.asarray(vel[mask], float)
    y = np.asarray(profile[mask], float)
    s = np.asarray(sigma[mask], float)
    k_params = 3 * n_gauss + poly_order + 1
    if v.size <= k_params + 2:
        raise ValueError("Insufficient channels for fit complexity.")
    ordv = np.argsort(v)
    v, y, s = v[ordv], y[ordv], s[ordv]
    baseline_seed = fit_weighted_baseline_seed(v, y, s, poly_order=poly_order)
    baseline_seed_eval = _eval_poly(v, baseline_seed)
    A0, mu0, sig0 = find_gaussian_seeds(v, y - baseline_seed_eval, n_gauss, vel_min, vel_max)
    sig0 = np.clip(sig0, SIGMA_PHYS_MIN_KMS, SIGMA_PHYS_MAX_KMS)
    p0 = np.concatenate([np.column_stack([A0, mu0, sig0]).ravel(), baseline_seed])
    lb = np.array([0.0, vel_min, SIGMA_PHYS_MIN_KMS] * n_gauss + [-np.inf] * (poly_order + 1), dtype=float)
    ub = np.array([np.inf, vel_max, SIGMA_PHYS_MAX_KMS] * n_gauss + [np.inf] * (poly_order + 1), dtype=float)

    def resid(theta: np.ndarray) -> np.ndarray:
        data_resid = (y - gauss_poly_model(v, theta, n_gauss, poly_order)) / s
        if PHYSICS_WIDTH_PENALTY_LAMBDA <= 0:
            return data_resid
        width_pen = _physics_width_penalty_terms(theta, n_gauss)
        if width_pen.size == 0:
            return data_resid
        return np.concatenate([data_resid, np.sqrt(PHYSICS_WIDTH_PENALTY_LAMBDA) * width_pen])

    rng = np.random.default_rng(random_seed)
    starts = [p0]
    for _ in range(n_multistart):
        trial = p0.copy()
        for idx in range(n_gauss):
            trial[3 * idx] *= np.exp(rng.normal(0.0, 0.25))
            trial[3 * idx + 1] += rng.normal(0.0, 8.0)
            trial[3 * idx + 2] *= np.exp(rng.normal(0.0, 0.20))
        trial[3 * n_gauss : 3 * n_gauss + poly_order + 1] += rng.normal(0.0, 0.2 * robust_mad_sigma(y), size=poly_order + 1)
        trial = np.clip(trial, lb + 1e-9, ub - 1e-9)
        starts.append(trial)

    best = None
    best_cost = np.inf
    for start in starts:
        try:
            res = opt.least_squares(resid, start, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=60000)
        except Exception:
            continue
        if res.cost < best_cost:
            best = res
            best_cost = float(res.cost)
    if best is None:
        raise RuntimeError("All least-squares starts failed.")

    popt = np.asarray(best.x, float)
    r_data = (y - gauss_poly_model(v, popt, n_gauss, poly_order)) / s
    dof = max(v.size - popt.size, 1)
    chi2 = float(np.sum(r_data**2))
    chi2_red = chi2 / dof
    try:
        jtj_inv = np.linalg.inv(best.jac.T @ best.jac)
        pcov = jtj_inv * chi2_red
        perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    except np.linalg.LinAlgError:
        perr = np.full_like(popt, np.nan)
    popt, perr = _reorder_component_params(popt, perr, n_gauss, poly_order)
    return FitResult(n_gauss, poly_order, popt, perr, chi2, dof, chi2_red, v, y, s)


def _fallback_fit_result(vel: np.ndarray, profile: np.ndarray, sigma: np.ndarray, vel_min: float, vel_max: float) -> FitResult:
    try:
        return gauss_poly_fit(vel, profile, sigma, n_gauss=1, poly_order=0, vel_min=vel_min, vel_max=vel_max, n_multistart=max(2, NAIVE_MULTI_STARTS), random_seed=FIT_SEED + 101)
    except Exception:
        pass
    mask = np.isfinite(vel) & np.isfinite(profile) & np.isfinite(sigma) & (sigma > 0) & (vel >= vel_min) & (vel <= vel_max)
    v = np.asarray(vel[mask], float)
    y = np.asarray(profile[mask], float)
    s = np.asarray(sigma[mask], float)
    if v.size < 8:
        raise RuntimeError("Fallback fit failed: insufficient finite channels in fit window.")
    order = np.argsort(v)
    v, y, s = v[order], y[order], s[order]
    outer = np.abs(v) > 80.0
    if outer.sum() < 5:
        outer = np.ones_like(v, dtype=bool)
    baseline_seed = float(np.nanmedian(y[outer]))
    y_line = y - baseline_seed
    w = np.clip(y_line, 0.0, np.inf)
    if np.nansum(w) <= 0:
        w = np.abs(y_line)
    if np.nansum(w) <= 0:
        w = np.ones_like(v, float)
    mu = float(np.nansum(w * v) / np.nansum(w))
    var = float(np.nansum(w * (v - mu) ** 2) / np.nansum(w))
    sigma_k = float(np.clip(np.sqrt(max(var, 1e-12)), SIGMA_PHYS_MIN_KMS, SIGMA_PHYS_MAX_KMS))
    amp = float(np.nanmax(y_line[np.isfinite(y_line)])) if np.isfinite(y_line).any() else 0.0
    if not np.isfinite(amp) or amp <= 0:
        amp = float(max(np.nanmax(np.abs(y_line)) if y_line.size else 0.0, 1e-6))
    p0 = np.array([amp, mu, sigma_k, baseline_seed], dtype=float)
    lb = np.array([0.0, vel_min, SIGMA_PHYS_MIN_KMS, -np.inf], dtype=float)
    ub = np.array([np.inf, vel_max, SIGMA_PHYS_MAX_KMS, np.inf], dtype=float)
    p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)

    def resid(theta: np.ndarray) -> np.ndarray:
        data_resid = (y - gauss_poly_model(v, theta, n_gauss=1, poly_order=0)) / s
        width_pen = _physics_width_penalty_terms(theta, 1)
        if width_pen.size == 0:
            return data_resid
        return np.concatenate([data_resid, np.sqrt(PHYSICS_WIDTH_PENALTY_LAMBDA) * width_pen])

    try:
        best = opt.least_squares(resid, p0, bounds=(lb, ub), loss="soft_l1", f_scale=1.0, max_nfev=20000)
        popt = np.asarray(best.x, float)
    except Exception:
        best = None
        popt = p0
    model = gauss_poly_model(v, popt, n_gauss=1, poly_order=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (y - model) / s
    chi2 = float(np.nansum(r**2))
    dof = max(v.size - popt.size, 1)
    chi2_red = chi2 / dof
    if best is not None:
        try:
            jtj_inv = np.linalg.inv(best.jac.T @ best.jac)
            pcov = jtj_inv * chi2_red
            perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
        except np.linalg.LinAlgError:
            perr = np.full_like(popt, np.nan, dtype=float)
    else:
        perr = np.full_like(popt, np.nan, dtype=float)
    return FitResult(1, 0, popt, perr, chi2, dof, chi2_red, v, y, s)


def select_model_grid(vel: np.ndarray, profile: np.ndarray, sigma: np.ndarray, vel_min: float, vel_max: float, n_grid=N_GAUSS_GRID, poly_grid=POLY_GRID):
    rows = []
    fits = {}
    for n_gauss in n_grid:
        for poly_order in poly_grid:
            try:
                fit = gauss_poly_fit(vel, profile, sigma, n_gauss=n_gauss, poly_order=poly_order, vel_min=vel_min, vel_max=vel_max)
            except Exception as exc:
                rows.append(dict(n_gauss=n_gauss, poly_order=poly_order, chi2_red=np.nan, aic=np.nan, aicc=np.nan, bic=np.nan, n_phys_viol=np.nan, status=f"fail: {exc}"))
                continue
            k = 3 * n_gauss + poly_order + 1
            nobs = fit.vel_fit.size
            if nobs > k + 1:
                aic = fit.chi2 + 2 * k
                aicc = aic + (2 * k * (k + 1)) / max(nobs - k - 1, 1)
                bic = fit.chi2 + k * np.log(nobs)
            else:
                aic = aicc = bic = np.nan
            n_phys_viol = _fit_n_phys_viol(fit.popt, n_gauss)
            rows.append(dict(n_gauss=n_gauss, poly_order=poly_order, chi2_red=fit.chi2_red, aic=aic, aicc=aicc, bic=bic, n_phys_viol=n_phys_viol, status="ok"))
            fits[(n_gauss, poly_order)] = fit
    table = pd.DataFrame(rows)
    ok = table[table["status"] == "ok"].copy()
    if ok.empty:
        fallback_fit = _fallback_fit_result(vel, profile, sigma, vel_min, vel_max)
        fallback_row = dict(n_gauss=fallback_fit.n_gauss, poly_order=fallback_fit.poly_order, chi2_red=fallback_fit.chi2_red, aic=np.nan, aicc=np.nan, bic=np.nan, n_phys_viol=_fit_n_phys_viol(fallback_fit.popt, fallback_fit.n_gauss), status="fallback: joint_1gauss_poly0", selected=True)
        table_out = pd.concat([table, pd.DataFrame([fallback_row])], ignore_index=True)
        if "selected" not in table_out.columns:
            table_out["selected"] = False
        table_out["selected"] = table_out["selected"].fillna(False).astype(bool)
        return fallback_fit, table_out.sort_values(["selected", "status"], ascending=[False, True]).reset_index(drop=True)
    in_band = ok[(ok["chi2_red"] >= 0.3) & (ok["chi2_red"] <= 5.0)].copy()
    pref = in_band if not in_band.empty else ok.copy()
    min_viol = np.nanmin(np.asarray(pref["n_phys_viol"], float))
    pref = pref[np.asarray(pref["n_phys_viol"], float) == min_viol].copy()
    pref["_aicc_sort"] = np.where(np.isfinite(pref["aicc"]), pref["aicc"], np.inf)
    pref["_chi2_sort"] = np.where(np.isfinite(pref["chi2_red"]), pref["chi2_red"], np.inf)
    pref = pref.sort_values(["_aicc_sort", "_chi2_sort"], ascending=[True, True]).reset_index(drop=True)
    best_row = pref.iloc[0]
    key = (int(best_row["n_gauss"]), int(best_row["poly_order"]))
    table["selected"] = False
    table.loc[(table["status"] == "ok") & (table["n_gauss"] == key[0]) & (table["poly_order"] == key[1]), "selected"] = True
    return fits[key], table


def fit_summary_metrics(fit: FitResult):
    p = np.asarray(fit.popt, float)
    areas = []
    mus = []
    sigs = []
    for idx in range(fit.n_gauss):
        A = p[3 * idx]
        mu = p[3 * idx + 1]
        sig_k = p[3 * idx + 2]
        area = float(A * sig_k * np.sqrt(2 * np.pi))
        areas.append(max(area, 0.0))
        mus.append(mu)
        sigs.append(sig_k)
    areas = np.asarray(areas, float)
    mus = np.asarray(mus, float)
    sigs = np.asarray(sigs, float)
    wsum = float(np.sum(areas))
    if wsum <= 0:
        return {"area": np.nan, "centroid": np.nan, "fwhm_eff": np.nan}
    centroid = float(np.sum(areas * mus) / wsum)
    second = float(np.sum(areas * (sigs**2 + (mus - centroid) ** 2)) / wsum)
    sigma_eff = np.sqrt(max(second, 0.0))
    return {"area": wsum, "centroid": centroid, "fwhm_eff": float(2.355 * sigma_eff)}


def fit_metric_uncertainty_mc(fit: FitResult, n_draw: int = 300, seed: int = 0):
    p = np.asarray(fit.popt, float)
    pe = np.asarray(fit.perr, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draw):
        theta = p + rng.normal(0.0, np.where(np.isfinite(pe), pe, 0.0))
        for idx in range(fit.n_gauss):
            theta[3 * idx] = max(theta[3 * idx], 0.0)
            theta[3 * idx + 2] = np.clip(theta[3 * idx + 2], SIGMA_PHYS_MIN_KMS, SIGMA_PHYS_MAX_KMS)
        vals.append(
            fit_summary_metrics(
                FitResult(
                    fit.n_gauss,
                    fit.poly_order,
                    theta,
                    fit.perr,
                    fit.chi2,
                    fit.dof,
                    fit.chi2_red,
                    fit.vel_fit,
                    fit.data_fit,
                    fit.sigma_fit,
                )
            )
        )
    df = pd.DataFrame(vals)
    return {"sigma_area": float(np.nanstd(df["area"])), "sigma_centroid": float(np.nanstd(df["centroid"])), "sigma_fwhm_eff": float(np.nanstd(df["fwhm_eff"]))}


def gal_to_equatorial(l_deg: float, b_deg: float) -> tuple[float, float]:
    R_eq_to_gal_2000 = np.array(
        [[-0.054876, -0.873437, -0.483835], [0.494109, -0.444830, 0.746982], [-0.867666, -0.198076, 0.455984]],
        dtype=float,
    )
    M_gal2eq = R_eq_to_gal_2000.T
    l = np.radians(float(l_deg))
    b = np.radians(float(b_deg))
    xyz_g = np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])
    xyz_e = M_gal2eq @ xyz_g
    ra_deg = float(np.degrees(np.arctan2(xyz_e[1], xyz_e[0])) % 360.0)
    dec_deg = float(np.degrees(np.arcsin(np.clip(xyz_e[2], -1.0, 1.0))))
    return ra_deg, dec_deg


def hadec_to_altaz(ha_deg: float, dec_deg: float, lat_deg: float) -> tuple[float, float]:
    ha = np.radians(float(ha_deg))
    dec = np.radians(float(dec_deg))
    lat = np.radians(float(lat_deg))
    sin_alt = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha)
    alt_rad = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    cos_alt = np.cos(alt_rad)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = (np.sin(dec) - np.sin(lat) * sin_alt) / (np.cos(lat) * cos_alt)
    az_rad = np.arccos(np.clip(cos_az, -1.0, 1.0))
    if np.sin(ha) > 0:
        az_rad = 2.0 * np.pi - az_rad
    return float(np.degrees(alt_rad)), float(np.degrees(az_rad))


def run_analysis() -> AnalysisResult:
    ensure_output_dirs()
    tables: dict[str, object] = {}
    figures: dict[str, object] = {}
    values: dict[str, object] = {}

    std_pair = load_lo_pair(STANDARD_SPECTRA_DIR)
    cyg_pair = load_lo_pair(CYGNUS_X_SPECTRA_DIR)
    datasets = {
        "standard": {"pair": std_pair, "masks": {lo: sigma_clip_rfi_mask(std_pair[lo]) for lo in LO_FREQS_MHZ}},
        "cygnus-x": {"pair": cyg_pair, "masks": {lo: sigma_clip_rfi_mask(cyg_pair[lo]) for lo in LO_FREQS_MHZ}},
    }
    _, cal = load_temperature_artifact()
    _, eq = load_equipment_artifact()

    figures["hyperfine"] = plot_hyperfine()
    figures["lsr_geometry"] = plot_lsr_geometry()

    spec_ref = std_pair[1420]
    ra_manual, dec_manual = gal_to_equatorial(120.0, 0.0)
    lst_rad = float(spec_ref.lst)
    ha_deg = (np.degrees(lst_rad) - ra_manual) % 360.0
    alt_manual, az_manual = hadec_to_altaz(ha_deg, dec_manual, float(spec_ref.obs_lat))
    tables["coordinate_check"] = pd.DataFrame([{"ra_deg": ra_manual, "dec_deg": dec_manual, "ha_deg": ha_deg, "alt_deg": alt_manual, "az_deg": az_manual}])

    dv_chan = C_LIGHT_KMS * (std_pair[1420].sample_rate / std_pair[1420].psd.size) / HI_REST_FREQ_HZ
    if SMOOTH_NCHAN > int(np.floor(MAX_SMOOTH_KMS / dv_chan)):
        raise ValueError("Configured smoothing width violates the <=1 km/s policy.")

    # Mean vs median figure on the requested single trace.
    raw_dir = STANDARD_SPECTRA_DIR.parent / "standard"
    candidates = sorted(raw_dir.glob("*-1420-0_obs_*.npz"))
    if candidates:
        data = np.load(candidates[0], allow_pickle=False)
        iq_raw = data["data"]
        sr = float(data["sample_rate"])
        fc = float(data["center_freq"])
        nsamples = int(iq_raw.shape[1])
        iq = iq_raw[..., 0].astype(np.float32) + 1j * iq_raw[..., 1].astype(np.float32)
        iq -= iq.mean(axis=1, keepdims=True)
        psd_blocks = np.abs(np.fft.fftshift(np.fft.fft(iq, axis=1), axes=1)) ** 2 / nsamples**2
        freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1.0 / sr)) + fc
        freqs_mhz = freqs / 1e6
        center_idx = int(np.argmin(np.abs(freqs - fc)))
        analysis_mask = np.ones(nsamples, dtype=bool)
        analysis_mask[center_idx] = False
        psd_blocks[:, ~analysis_mask] = np.nan
        psd = np.full(nsamples, np.nan, dtype=float)
        valid_cols = np.any(np.isfinite(psd_blocks), axis=0)
        if np.any(valid_cols):
            psd[valid_cols] = np.nanmean(psd_blocks[:, valid_cols], axis=0)
        psd[~analysis_mask] = np.nan

        def _fill_nan_linear(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, float)
            finite = np.isfinite(arr)
            if finite.sum() < 2:
                raise ValueError("Need at least two finite channels to interpolate across the LO bin.")
            x = np.arange(arr.size, dtype=float)
            out = arr.copy()
            out[~finite] = np.interp(x[~finite], x[finite], arr[finite])
            return out

        window_neighbors = 4
        window_size = 1 + 2 * window_neighbors
        if window_size > nsamples:
            window_size = nsamples if nsamples % 2 == 1 else nsamples - 1
        if window_size < 3:
            window_size = 3
        if window_size % 2 == 0:
            window_size -= 1

        mean_slide = smooth_series(_fill_nan_linear(psd), {"method": "boxcar", "M": window_size})
        median_slide = smooth_nanmedian(psd, window_size)
        mean_slide[~analysis_mask] = np.nan
        median_slide[~analysis_mask] = np.nan
        focus = analysis_mask
        norm_ref = np.nanmedian(mean_slide[focus])
        figures["mean_vs_median"] = plot_mean_vs_median(
            freqs_mhz=freqs_mhz,
            focus=focus,
            psd=psd,
            median_slide=median_slide,
            mean_slide=mean_slide,
            window_size=window_size,
            norm_ref=norm_ref,
        )

    ratio_profiles = {}
    ratio_fits = {}
    ratio_model_tables = {}
    for ds_name, ds in datasets.items():
        pair = ds["pair"]
        masks = ds["masks"]
        p0 = masked_spectrum_values(pair[1420])
        p1 = masked_spectrum_values(pair[1421])
        s0 = masked_spectrum_values(pair[1420], pair[1420].std)
        s1 = masked_spectrum_values(pair[1421], pair[1421].std)
        mask = combine_spectrum_mask(pair[1420], masks[1420], masks[1421], np.isfinite(p0), np.isfinite(p1), np.isfinite(s0), np.isfinite(s1), (p0 > 0), (p1 > 0), require_nonempty=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = p0 / p1
            Rinv = p1 / p0
            R_sigma = np.abs(R) * np.sqrt((s0 / p0) ** 2 + (s1 / p1) ** 2)
            Rinv_sigma = np.abs(Rinv) * np.sqrt((s1 / p1) ** 2 + (s0 / p0) ** 2)
        y_R = np.where(mask, R - 1.0, np.nan)
        y_inv = np.where(mask, Rinv - 1.0, np.nan)
        y_R_fit, s_R_fit = smooth_profile_with_sigma(y_R, np.where(mask, R_sigma, np.nan), SMOOTH_NCHAN, method=SMOOTH_METHOD)
        y_inv_fit, s_inv_fit = smooth_profile_with_sigma(y_inv, np.where(mask, Rinv_sigma, np.nan), SMOOTH_NCHAN, method=SMOOTH_METHOD)
        v0 = velocity_axis(pair[1420].freqs) + lsr_correction_kms(pair[1420])
        v1 = velocity_axis(pair[1421].freqs) + lsr_correction_kms(pair[1421])
        ratio_profiles[ds_name] = {"v0": v0, "v1": v1, "y_R": y_R, "y_inv": y_inv, "s_R": np.where(mask, R_sigma, np.nan), "s_inv": np.where(mask, Rinv_sigma, np.nan), "y_R_fit": y_R_fit, "y_inv_fit": y_inv_fit, "s_R_fit": s_R_fit, "s_inv_fit": s_inv_fit}
        vel_min, vel_max = FIT_WINDOWS_KMS[ds_name]
        for tag, vel, y, sy in [("R", v0, y_R_fit, s_R_fit), ("Rinv", v1, y_inv_fit, s_inv_fit)]:
            fit, table = select_model_grid(vel, y, sy, vel_min, vel_max)
            ratio_fits[(ds_name, tag)] = fit
            ratio_model_tables[(ds_name, tag)] = table
    figures["ratio_profile"] = plot_ratio_profile(
        standard=ratio_profiles["standard"],
        cygnus_x=ratio_profiles["cygnus-x"],
        smooth_nchan=SMOOTH_NCHAN,
    )
    tables["ratio_model_tables"] = {f"{key[0]}:{key[1]}": value for key, value in ratio_model_tables.items()}

    eq_offset = np.asarray(eq["freq_offset_mhz"], float)
    eq_resp_floor = float(eq["response_floor"])
    eq_resp = np.asarray(eq["fir_response_norm"], float) * np.asarray(eq["sum_response_norm"], float)
    eq_pass = np.asarray(eq["passband_mask"], bool)
    eq_eval = np.asarray(eq["combined_eval_mask"], bool)
    alpha = float(eq["alpha_db_per_m"])
    sigma_alpha = float(eq["sigma_alpha_db_per_m"])
    L_unknown = float(eq["unknown_cable_length_m"])
    att_frac_raw = np.log(10) / 10.0 * abs(sigma_alpha) * abs(L_unknown) if np.isfinite(sigma_alpha) and np.isfinite(L_unknown) else 0.0
    rmse_arr = np.asarray(eq["sweep_rmse_db"], float)
    rmse_db = float(np.nanmedian(rmse_arr)) if rmse_arr.size else 0.2
    lin_frac = np.log(10) / 20.0 * abs(rmse_db)
    sys_frac = float(cal["sigma_hw_fraction"]) if "sigma_hw_fraction" in cal else float(lin_frac)
    tables["linearity_headroom"] = pd.DataFrame(
        [
            {
                "target_100mVpp_dbm": 10.0 * np.log10(((0.100 / (2.0 * np.sqrt(2.0))) ** 2 / 50.0) / 1e-3),
                "highest_unclipped_setpoint_dbm": float(eq["highest_unclipped_setpoint_dbm"]),
                "first_clipped_setpoint_dbm": float(eq["first_clipped_setpoint_dbm"]),
                "linearity_rmse_db": rmse_db,
            }
        ]
    )

    def response_on_axis(freq_axis_hz: np.ndarray, center_freq_hz: float):
        x_new = (np.asarray(freq_axis_hz, float) - float(center_freq_hz)) / 1e6
        resp_interp = interp_mono(eq_offset, eq_resp, x_new, fill_value=np.nan)
        pass_interp = interp_bool_nearest(eq_offset, eq_pass, x_new, default=False)
        eval_interp = interp_bool_nearest(eq_offset, eq_eval, x_new, default=False)
        resp_safe = np.clip(resp_interp, eq_resp_floor, np.inf)
        return resp_safe, pass_interp, eval_interp

    def cold_profile_on_axis(cal_npz: dict[str, np.ndarray], lo: int, freq_axis_hz: np.ndarray):
        fsrc = np.asarray(cal_npz[f"freq_hz_{lo}"], float)
        psrc = np.asarray(cal_npz[f"cold_ref_profile_{lo}"], float)
        msrc = np.asarray(cal_npz[f"cold_ref_mask_{lo}"], bool)
        return (
            interp_mono(fsrc, psrc, freq_axis_hz, fill_value=np.nan),
            interp_bool_nearest(fsrc, msrc, freq_axis_hz, default=False),
        )

    t_rx_1420 = float(cal["t_rx_1420"])
    t_rx_1421 = float(cal["t_rx_1421"])
    s_t_rx_1420 = float(cal["sigma_t_rx_1420"])
    s_t_rx_1421 = float(cal["sigma_t_rx_1421"])
    T_cold = float(cal["t_cold"])
    cold_ref_method = str(cal["cold_ref_method"].item() if np.asarray(cal["cold_ref_method"]).ndim == 0 else cal["cold_ref_method"])
    cold_ref_is_hw_corrected = "hw_corrected" in cold_ref_method.lower()

    temp_profiles = {}
    temp_fits = {}
    temp_model_tables = {}
    for ds_name, ds in datasets.items():
        pair = ds["pair"]
        masks = ds["masks"]
        p0 = masked_spectrum_values(pair[1420])
        p1 = masked_spectrum_values(pair[1421])
        s0 = masked_spectrum_values(pair[1420], pair[1420].std)
        s1 = masked_spectrum_values(pair[1421], pair[1421].std)
        f0 = np.asarray(pair[1420].freqs, float)
        f1 = np.asarray(pair[1421].freqs, float)
        resp0, _, _ = response_on_axis(f0, pair[1420].center_freq)
        resp1, _, _ = response_on_axis(f1, pair[1421].center_freq)
        c0, cm0 = cold_profile_on_axis(cal, 1420, f0)
        c1, cm1 = cold_profile_on_axis(cal, 1421, f1)
        hw_good = combine_spectrum_mask(pair[1420], masks[1420], masks[1421], cm0, cm1, np.isfinite(p0), np.isfinite(p1), np.isfinite(s0), np.isfinite(s1), np.isfinite(resp0), np.isfinite(resp1), np.isfinite(c0), np.isfinite(c1), (p0 > 0), (p1 > 0), (c0 > 0), (c1 > 0), require_nonempty=True)
        p0h = np.where(hw_good, p0 / resp0, np.nan)
        p1h = np.where(hw_good, p1 / resp1, np.nan)
        s0h = np.where(hw_good, s0 / resp0, np.nan)
        s1h = np.where(hw_good, s1 / resp1, np.nan)
        c0h = np.where(hw_good, c0, np.nan) if cold_ref_is_hw_corrected else np.where(hw_good, c0 / resp0, np.nan)
        c1h = np.where(hw_good, c1, np.nan) if cold_ref_is_hw_corrected else np.where(hw_good, c1 / resp1, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = p0h / p1h
            Rinv = p1h / p0h
            sR = np.abs(R) * np.sqrt((s0h / p0h) ** 2 + (s1h / p1h) ** 2)
            sRinv = np.abs(Rinv) * np.sqrt((s1h / p1h) ** 2 + (s0h / p0h) ** 2)
        Tsys1 = p1h * (T_cold + t_rx_1421) / c1h
        Tsys0 = p0h * (T_cold + t_rx_1420) / c0h
        with np.errstate(divide="ignore", invalid="ignore"):
            sTsys1 = np.abs(Tsys1) * np.sqrt((s1h / p1h) ** 2 + (s_t_rx_1421 / (T_cold + t_rx_1421)) ** 2)
            sTsys0 = np.abs(Tsys0) * np.sqrt((s0h / p0h) ** 2 + (s_t_rx_1420 / (T_cold + t_rx_1420)) ** 2)
        yR = R - 1.0
        yinv = Rinv - 1.0
        Tline_R = yR * Tsys1
        with np.errstate(divide="ignore", invalid="ignore"):
            sTline_R_stat = np.sqrt((Tsys1 * sR) ** 2 + (yR * sTsys1) ** 2)
            sTline_R_total = np.sqrt(sTline_R_stat**2 + (sys_frac * np.abs(Tline_R)) ** 2)
        denom_inv = 1.0 + yinv
        valid_inv = np.isfinite(denom_inv) & (np.abs(denom_inv) > 1e-6)
        Tline_inv = np.full_like(yinv, np.nan, dtype=float)
        Tline_inv[valid_inv] = -(yinv[valid_inv] / denom_inv[valid_inv]) * Tsys0[valid_inv]
        dT_dy = np.full_like(yinv, np.nan, dtype=float)
        dT_dTsys = np.full_like(yinv, np.nan, dtype=float)
        dT_dy[valid_inv] = -Tsys0[valid_inv] / (denom_inv[valid_inv] ** 2)
        dT_dTsys[valid_inv] = -yinv[valid_inv] / denom_inv[valid_inv]
        sTline_inv_stat = np.full_like(yinv, np.nan, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sTline_inv_stat[valid_inv] = np.sqrt((dT_dy[valid_inv] * sRinv[valid_inv]) ** 2 + (dT_dTsys[valid_inv] * sTsys0[valid_inv]) ** 2)
        sTline_inv_total = np.sqrt(sTline_inv_stat**2 + (sys_frac * np.abs(Tline_inv)) ** 2)
        Tline_R_fit, sTline_R_fit = smooth_profile_with_sigma(Tline_R, sTline_R_stat, SMOOTH_NCHAN, method=SMOOTH_METHOD)
        Tline_inv_fit, sTline_inv_fit = smooth_profile_with_sigma(Tline_inv, sTline_inv_stat, SMOOTH_NCHAN, method=SMOOTH_METHOD)
        v0 = velocity_axis(f0) + lsr_correction_kms(pair[1420])
        v1 = velocity_axis(f1) + lsr_correction_kms(pair[1421])
        temp_profiles[ds_name] = {"v0": v0, "v1": v1, "Tline_R": Tline_R, "Tline_inv": Tline_inv, "Tline_R_fit": Tline_R_fit, "Tline_inv_fit": Tline_inv_fit, "sTline_R_stat": sTline_R_stat, "sTline_inv_stat": sTline_inv_stat, "sTline_R_fit": sTline_R_fit, "sTline_inv_fit": sTline_inv_fit, "sTline_R_total": sTline_R_total, "sTline_inv_total": sTline_inv_total}
        vel_min, vel_max = FIT_WINDOWS_KMS[ds_name]
        for tag, vel, y, sy in [("R", v0, Tline_R_fit, sTline_R_fit), ("Rinv", v1, Tline_inv_fit, sTline_inv_fit)]:
            fit, table = select_model_grid(vel, y, sy, vel_min, vel_max)
            temp_fits[(ds_name, tag)] = fit
            temp_model_tables[(ds_name, tag)] = table
    tables["temp_model_tables"] = {f"{key[0]}:{key[1]}": value for key, value in temp_model_tables.items()}

    baseline_sys_rows = []
    for ds_name in ["standard", "cygnus-x"]:
        for tag in ["R", "Rinv"]:
            best_fit = ratio_fits[(ds_name, tag)]
            n_g_best = best_fit.n_gauss
            m_best = best_fit.poly_order
            vel_min, vel_max = FIT_WINDOWS_KMS[ds_name]
            if tag == "R":
                vel = ratio_profiles[ds_name]["v0"]
                y = ratio_profiles[ds_name]["y_R_fit"]
                sy = ratio_profiles[ds_name]["s_R_fit"]
            else:
                vel = ratio_profiles[ds_name]["v1"]
                y = ratio_profiles[ds_name]["y_inv_fit"]
                sy = ratio_profiles[ds_name]["s_inv_fit"]
            centroid_best = fit_summary_metrics(best_fit)["centroid"]
            fwhm_best = fit_summary_metrics(best_fit)["fwhm_eff"]
            centroid_deltas = []
            fwhm_deltas = []
            for dm in [-1, +1]:
                m_try = m_best + dm
                if m_try < 0:
                    continue
                try:
                    alt_fit, _ = select_model_grid(vel, y, sy, vel_min, vel_max, n_grid=(n_g_best,), poly_grid=(m_try,))
                    centroid_deltas.append(abs(fit_summary_metrics(alt_fit)["centroid"] - centroid_best))
                    fwhm_deltas.append(abs(fit_summary_metrics(alt_fit)["fwhm_eff"] - fwhm_best))
                except Exception:
                    pass
            baseline_sys_rows.append(
                {
                    "dataset": ds_name,
                    "profile": tag,
                    "poly_best": m_best,
                    "n_gauss": n_g_best,
                    "sys_centroid_kms": round(max(centroid_deltas) if centroid_deltas else 0.0, 4),
                    "sys_fwhm_kms": round(max(fwhm_deltas) if fwhm_deltas else 0.0, 4),
                }
            )
    df_baseline_sys = pd.DataFrame(baseline_sys_rows)
    tables["baseline_systematics"] = df_baseline_sys

    summary_rows = []
    for ds_name in ["standard", "cygnus-x"]:
        ds_label = {"standard": "HI profile at (l=120°, b=0°)", "cygnus-x": "Cygnus-X"}[ds_name]
        fit_b = temp_fits[(ds_name, "R")]
        m_b = fit_summary_metrics(fit_b)
        u_b = fit_metric_uncertainty_mc(fit_b, n_draw=250, seed=29)
        vel_min, vel_max = FIT_WINDOWS_KMS[ds_name]
        vgrid = np.linspace(vel_min, vel_max, 2000)
        profile_b = temp_profiles[ds_name]["Tline_R_fit"]
        sigma_b = temp_profiles[ds_name]["sTline_R_fit"]
        vel_b = temp_profiles[ds_name]["v0"]
        finite_b = np.isfinite(vel_b) & np.isfinite(profile_b) & (vel_b >= vel_min) & (vel_b <= vel_max)
        figures[f"{ds_name}_fits"], resid_b = plot_dataset_fits(
            ds_name=ds_name,
            fit_b=fit_b,
            vel_min=vel_min,
            vel_max=vel_max,
            vgrid=vgrid,
            profile_b=profile_b,
            sigma_b=sigma_b,
            vel_b=vel_b,
            finite_b=finite_b,
        )
        summary_rows.append(
            {
                "dataset": ds_label,
                "profile": "R",
                "n_gauss": fit_b.n_gauss,
                "poly_order": fit_b.poly_order,
                "chi2_red": fit_b.chi2_red,
                "centroid": m_b["centroid"],
                "sigma_centroid": u_b["sigma_centroid"],
                "fwhm_eff": m_b["fwhm_eff"],
                "sigma_fwhm_eff": u_b["sigma_fwhm_eff"],
                "area": m_b["area"],
                "sigma_area": u_b["sigma_area"],
                "residual_rms": float(np.nanstd(resid_b)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    tables["summary"] = summary_df

    component_tables = {}
    for label, fit_dict in [("Path A (ratio-domain)", ratio_fits), ("Path B (calibrated)", temp_fits)]:
        rows = []
        for (ds_name, tag), fit in fit_dict.items():
            for idx in range(fit.n_gauss):
                rows.append(
                    {
                        "dataset": ds_name,
                        "profile": tag,
                        "comp": idx + 1,
                        "A": fit.popt[3 * idx],
                        "sigma_A": fit.perr[3 * idx],
                        "v0_kms": fit.popt[3 * idx + 1],
                        "sigma_v0_kms": fit.perr[3 * idx + 1],
                        "fwhm_kms": 2.355 * fit.popt[3 * idx + 2],
                        "sigma_fwhm_kms": 2.355 * fit.perr[3 * idx + 2],
                        "chi2_red": fit.chi2_red,
                    }
                )
        component_tables[label] = pd.DataFrame(rows).sort_values(["dataset", "profile", "comp"]).reset_index(drop=True)
    tables["component_tables"] = component_tables

    if ETA_EFF_ESTIMATE_PATH.exists():
        with ETA_EFF_ESTIMATE_PATH.open() as handle:
            eta_row = next(csv.DictReader(handle))
        tables["eta_efficiency"] = pd.DataFrame([eta_row])

    values.update(
        {
            "ratio_profiles": ratio_profiles,
            "ratio_fits": ratio_fits,
            "temp_profiles": temp_profiles,
            "temp_fits": temp_fits,
            "sys_frac": sys_frac,
            "att_frac_raw": att_frac_raw,
            "lin_frac": lin_frac,
        }
    )
    return AnalysisResult(artifact={}, artifact_path=None, tables=tables, figures=figures, values=values)
