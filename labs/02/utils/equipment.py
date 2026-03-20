from datetime import datetime, timezone
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from ugradiolab.data import Spectrum

from .constants import (
    ATTENUATOR_SDR_DB,
    BENCH_KL_FILTER_LOSS_DB,
    BENCH_LAB_AMPS_NET_GAIN_DB,
    BENCH_ROOF_CHAIN_NET_GAIN_DB,
    BENCH_UNKNOWN_CABLE_LOSS_DB,
    G_SUM_EST,
    H_FIR,
    INFLUENCE_ALPHA_DELTA_THRESHOLD,
    LENGTH_UNCERTAINTY_MC_SAMPLES,
    LENGTH_UNCERTAINTY_MC_SEED,
    LO_FREQS_MHZ,
    METER_RULE_DIVISION_M,
    MIN_POINTS_AFTER_SCREEN,
    OUTLIER_Z_THRESHOLD,
    PASSBAND_DB_THRESHOLD,
    PORT2_CABLE_LEN_M,
    POWER_METER_DIVISION_DB,
    POWER_METER_UNCERTAINTY_DB,
    PRIMARY_RULE_MAX_RELATIVE_CI95_WIDTH,
    PRIMARY_RULE_REQUIRE_POSITIVE_ALPHA_CI95,
    RESPONSE_FLOOR,
    RTL_INTERNAL_SAMPLE_RATE_HZ,
    SIGGEN_FREQ_MHZ,
    SIGMA_CAL_LENGTH_M,
    SIGMA_LEAD_LENGTH_M,
    SIGMA_LENGTH_READ_M,
    SPLITTER_S1_DB,
    SPLITTER_S2_DB,
    UNKNOWN_LEAD_LENGTH_M,
)
from .contracts import EquipmentCalibrationResult
from .io import attenuation_manifest, save_npz, sdr_gain_sweep_manifest, unknown_length_manifest
from .paths import (
    COLD_REF_1420_PATH,
    EQUIPMENT_ARTIFACT_PATH,
    ensure_output_dirs,
)
from .plotting import (
    plot_cable_attenuation_lo,
    plot_cable_attenuation_power_meter,
    plot_reflectometry,
    plot_sdr_fir_summing_correction,
    plot_sdr_gain_response_clipping,
    plot_signal_chain,
)


def to_normalised_db(total_power: float, siggen_amp_dbm: float) -> float:
    return 10.0 * np.log10(float(total_power)) - float(siggen_amp_dbm)


def _aic_bic(rss: float, n_obs: int, n_params: int) -> tuple[float, float]:
    if rss <= 0 or n_obs <= n_params:
        return np.nan, np.nan
    aic = n_obs * np.log(rss / n_obs) + 2 * n_params
    bic = n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)
    return float(aic), float(bic)


def _mc_cov_shared_length(
    L: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    sigma_L: float,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    if not np.isfinite(sigma_L) or sigma_L <= 0 or n_mc < 4:
        return np.zeros((3, 3), dtype=float)
    rng = np.random.default_rng(seed)
    betas = np.full((int(n_mc), 3), np.nan, dtype=float)
    for idx in range(int(n_mc)):
        Lj = L + rng.normal(0.0, sigma_L, size=len(L))
        if np.unique(np.round(Lj, 9)).size < 2:
            continue
        n = Lj.size
        y = np.concatenate([y0, y1])
        X = np.zeros((2 * n, 3), dtype=float)
        X[:n, 0] = 1.0
        X[:n, 2] = -Lj
        X[n:, 1] = 1.0
        X[n:, 2] = -Lj
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas[idx, :] = beta
    valid = betas[np.isfinite(betas).all(axis=1)]
    if valid.shape[0] < 4:
        return np.zeros((3, 3), dtype=float)
    return np.cov(valid, rowvar=False, ddof=1).astype(float, copy=False)


def _mc_cov_single_length(
    L: np.ndarray,
    y: np.ndarray,
    sigma_L: float,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    if not np.isfinite(sigma_L) or sigma_L <= 0 or n_mc < 4:
        return np.zeros((2, 2), dtype=float)
    rng = np.random.default_rng(seed)
    betas = np.full((int(n_mc), 2), np.nan, dtype=float)
    for idx in range(int(n_mc)):
        Lj = L + rng.normal(0.0, sigma_L, size=len(L))
        if np.unique(np.round(Lj, 9)).size < 2:
            continue
        X = np.column_stack([np.ones_like(Lj), -Lj])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        betas[idx, :] = beta
    valid = betas[np.isfinite(betas).all(axis=1)]
    if valid.shape[0] < 4:
        return np.zeros((2, 2), dtype=float)
    return np.cov(valid, rowvar=False, ddof=1).astype(float, copy=False)


def fit_shared_linear(
    L: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    *,
    sigma_L: float = 0.0,
    mc_samples: int | None = None,
    mc_seed: int | None = None,
) -> dict[str, object]:
    L = np.asarray(L, float)
    y0 = np.asarray(y0, float)
    y1 = np.asarray(y1, float)
    n = L.size
    y = np.concatenate([y0, y1])
    X = np.zeros((2 * n, 3))
    X[:n, 0] = 1.0
    X[:n, 2] = -L
    X[n:, 1] = 1.0
    X[n:, 2] = -L

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    B0, B1, alpha = beta
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    dof = y.size - 3
    sigma2 = rss / dof if dof > 0 else np.nan
    cov_ols = sigma2 * np.linalg.inv(X.T @ X) if np.isfinite(sigma2) else np.full((3, 3), np.nan)

    cov_len = np.zeros((3, 3), dtype=float)
    if np.isfinite(sigma_L) and sigma_L > 0 and np.all(np.isfinite(cov_ols)):
        n_mc_use = int(mc_samples) if mc_samples is not None else LENGTH_UNCERTAINTY_MC_SAMPLES
        seed_use = int(mc_seed) if mc_seed is not None else LENGTH_UNCERTAINTY_MC_SEED
        cov_len = _mc_cov_shared_length(L, y0, y1, float(sigma_L), n_mc_use, seed_use)
    cov = cov_ols + cov_len
    sig = np.sqrt(np.diag(cov))
    sig_ols = np.sqrt(np.diag(cov_ols)) if np.all(np.isfinite(cov_ols)) else np.full(3, np.nan)
    sig_len = np.sqrt(np.clip(np.diag(cov_len), 0.0, None))
    var_dB = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    aic, bic = _aic_bic(rss, y.size, 3)

    row_resid_1420 = y0 - (B0 - alpha * L)
    row_resid_1421 = y1 - (B1 - alpha * L)
    row_resid_norm = np.sqrt(row_resid_1420**2 + row_resid_1421**2)

    return {
        "B1420": float(B0),
        "B1421": float(B1),
        "alpha": float(alpha),
        "sigma_B1420": float(sig[0]),
        "sigma_B1421": float(sig[1]),
        "sigma_alpha": float(sig[2]),
        "sigma_B1420_ols": float(sig_ols[0]),
        "sigma_B1421_ols": float(sig_ols[1]),
        "sigma_alpha_ols": float(sig_ols[2]),
        "sigma_B1420_len": float(sig_len[0]),
        "sigma_B1421_len": float(sig_len[1]),
        "sigma_alpha_len": float(sig_len[2]),
        "sigma_deltaB": float(np.sqrt(var_dB)) if var_dB >= 0 else np.nan,
        "deltaB": float(B0 - B1),
        "rss": rss,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "aic": aic,
        "bic": bic,
        "residuals": resid,
        "row_resid_1420": row_resid_1420,
        "row_resid_1421": row_resid_1421,
        "row_resid_norm": row_resid_norm,
        "cov": cov,
        "cov_ols": cov_ols,
        "cov_len": cov_len,
    }


def fit_single_linear(
    L: np.ndarray,
    y: np.ndarray,
    *,
    sigma_L: float = 0.0,
    mc_samples: int | None = None,
    mc_seed: int | None = None,
) -> dict[str, object]:
    L = np.asarray(L, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(L), -L])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    B, alpha = beta
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    dof = y.size - 2
    sigma2 = rss / dof if dof > 0 else np.nan
    cov_ols = sigma2 * np.linalg.inv(X.T @ X) if np.isfinite(sigma2) else np.full((2, 2), np.nan)
    cov_len = np.zeros((2, 2), dtype=float)
    if np.isfinite(sigma_L) and sigma_L > 0 and np.all(np.isfinite(cov_ols)):
        n_mc_use = int(mc_samples) if mc_samples is not None else LENGTH_UNCERTAINTY_MC_SAMPLES
        seed_use = int(mc_seed) if mc_seed is not None else LENGTH_UNCERTAINTY_MC_SEED
        cov_len = _mc_cov_single_length(L, y, float(sigma_L), n_mc_use, seed_use)
    cov = cov_ols + cov_len
    sig = np.sqrt(np.diag(cov))
    sig_ols = np.sqrt(np.diag(cov_ols)) if np.all(np.isfinite(cov_ols)) else np.full(2, np.nan)
    sig_len = np.sqrt(np.clip(np.diag(cov_len), 0.0, None))
    aic, bic = _aic_bic(rss, y.size, 2)
    return {
        "B": float(B),
        "alpha": float(alpha),
        "sigma_B": float(sig[0]),
        "sigma_alpha": float(sig[1]),
        "sigma_B_ols": float(sig_ols[0]),
        "sigma_alpha_ols": float(sig_ols[1]),
        "sigma_B_len": float(sig_len[0]),
        "sigma_alpha_len": float(sig_len[1]),
        "rss": rss,
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "aic": aic,
        "bic": bic,
        "residuals": resid,
        "cov": cov,
        "cov_ols": cov_ols,
        "cov_len": cov_len,
    }


def robust_row_outlier_diagnostics(fit: dict[str, object], z_thresh: float = OUTLIER_Z_THRESHOLD) -> dict[str, object]:
    r = fit["row_resid_norm"]
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    robust_z = np.zeros_like(r) if mad <= 1e-12 else 0.67448975 * (r - med) / mad
    inlier_mask = np.abs(robust_z) <= z_thresh
    return {
        "row_resid_norm": r,
        "robust_z": robust_z,
        "median": med,
        "mad": mad,
        "inlier_mask": inlier_mask,
        "outlier_mask": ~inlier_mask,
        "n_outliers": int(np.sum(~inlier_mask)),
    }


def leave_one_out_alpha(L: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> np.ndarray:
    L = np.asarray(L, float)
    y0 = np.asarray(y0, float)
    y1 = np.asarray(y1, float)
    out = np.full(L.size, np.nan)
    for idx in range(L.size):
        keep = np.ones(L.size, dtype=bool)
        keep[idx] = False
        if keep.sum() < 3 or np.unique(L[keep]).size < 2:
            continue
        out[idx] = fit_shared_linear(L[keep], y0[keep], y1[keep])["alpha"]
    return out


def propagate_length_sigma(
    B: float,
    y_obs: float,
    alpha: float,
    sigma_B: float,
    sigma_alpha: float,
    cov_B_alpha: float = 0.0,
    sigma_y: float = 0.0,
) -> float:
    L = (B - y_obs) / alpha
    dLdB = 1.0 / alpha
    dLdy = -1.0 / alpha
    dLda = -L / alpha
    var = (
        (dLdB**2) * (sigma_B**2)
        + (dLdy**2) * (sigma_y**2)
        + (dLda**2) * (sigma_alpha**2)
        + 2.0 * dLdB * dLda * cov_B_alpha
    )
    return float(np.sqrt(max(var, 0.0)))


def infer_unknown_length_linear(y0_obs: float, y1_obs: float, fit_linear: dict[str, object]) -> dict[str, float]:
    alpha = float(fit_linear["alpha"])
    L0 = (float(fit_linear["B1420"]) - y0_obs) / alpha
    L1 = (float(fit_linear["B1421"]) - y1_obs) / alpha
    return {"L0": float(L0), "L1": float(L1), "L_total": float(0.5 * (L0 + L1))}


def dbm_to_watts(p_dbm: float) -> float:
    return 1e-3 * (10.0 ** (p_dbm / 10.0))


def watts_to_vrms(p_w: float, r_ohm: float = 50.0) -> float:
    return float(np.sqrt(max(p_w, 0.0) * r_ohm))


def vrms_to_vpp(v_rms: float) -> float:
    return float(2.0 * np.sqrt(2.0) * v_rms)


def power_response_on_output_axis(
    coeffs: np.ndarray,
    freq_hz: np.ndarray,
    internal_sample_rate_hz: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    c = np.asarray(coeffs, float)
    f = np.asarray(freq_hz, float)
    n = np.arange(c.size, dtype=float)
    out = np.empty(f.size, dtype=float)
    fs = float(internal_sample_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("internal_sample_rate_hz must be finite and > 0.")
    for idx in range(0, f.size, chunk_size):
        fb = f[idx : idx + chunk_size]
        phase = np.exp(-2j * np.pi * np.outer(fb, n) / fs)
        H = phase @ c
        out[idx : idx + chunk_size] = np.abs(H) ** 2
    peak = float(np.nanmax(out))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Filter response peak is not finite/positive.")
    return out / peak


def _fill_nan_linear(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, float)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        raise ValueError("Need at least two finite channels to interpolate across the LO bin.")
    idx = np.arange(arr.size, dtype=float)
    out = arr.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return out


def _normalise_in_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, float)
    valid = arr[mask]
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size == 0:
        return arr
    return arr / float(np.median(valid))


def _ripple_db(values: np.ndarray, mask: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    valid = np.asarray(values, float)[mask]
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size < 10:
        return np.nan
    qlo, qhi = np.percentile(valid, [lo, hi])
    return float(10.0 * np.log10(qhi / qlo))


def _pct_ripple_db(values: np.ndarray, mask: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> float:
    return _ripple_db(values, mask, lo=lo, hi=hi)


def _frac_std(values: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(values, float)[mask]
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size < 10:
        return np.nan
    valid = valid / np.median(valid)
    return float(np.std(valid - 1.0))


def make_symmetric_sum_filter(theta6: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = np.asarray(theta6, float)
    return np.array([a, b, c, d, e, f, e, d, c, b, a], dtype=float)


def _git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_equipment_calibration() -> EquipmentCalibrationResult:
    ensure_output_dirs()

    figures: dict[str, object] = {}
    tables: dict[str, object] = {}
    values: dict[str, object] = {}

    blocks = [
        ("Horn", 0.0, "roof"),
        ("Roof chain\n(ZKL×3+Reactel)", BENCH_ROOF_CHAIN_NET_GAIN_DB, "roof"),
        ("Cable", -BENCH_UNKNOWN_CABLE_LOSS_DB, "cable"),
        ("K&L BPF", -BENCH_KL_FILTER_LOSS_DB, "lab"),
        ("Lab amps\n(WB+NB)", BENCH_LAB_AMPS_NET_GAIN_DB, "lab"),
        ("SDR", 0.0, "lab"),
    ]
    G_cum_db = [0.0]
    for _, gain_db, _ in blocks[1:]:
        G_cum_db.append(G_cum_db[-1] + gain_db)
    x = np.arange(len(blocks))
    labels = [block[0] for block in blocks]
    regions = [block[2] for block in blocks]
    region_color = {"roof": "C0", "cable": "C4", "lab": "C1"}
    region_label = {"roof": "Roof (LNA chain)", "cable": "Cable run", "lab": "Lab rack"}
    figures["signal_chain"] = plot_signal_chain(
        x=x,
        G_cum_db=G_cum_db,
        labels=labels,
        regions=regions,
        region_color=region_color,
        region_label=region_label,
    )
    values["net_gain_db"] = float(G_cum_db[-1])

    df_att = attenuation_manifest()
    df_unk = unknown_length_manifest()
    for df in (df_att, df_unk):
        df["y_lo1420_db"] = df.apply(lambda row: to_normalised_db(row["lo1420_total_power"], row["siggen_amp_dbm"]), axis=1)
        df["y_lo1421_db"] = df.apply(lambda row: to_normalised_db(row["lo1421_total_power"], row["siggen_amp_dbm"]), axis=1)
        df["power_meter_norm_db"] = df["power_meter_dbm"] - df["siggen_amp_dbm"]
    df_att_all = df_att.copy().sort_values("cable_length_m").reset_index(drop=True)
    tables["attenuation_all"] = df_att_all.copy()
    tables["unknown_manifest"] = df_unk.copy()

    L_all = df_att_all["cable_length_m"].to_numpy(float)
    y1420_all = df_att_all["y_lo1420_db"].to_numpy(float)
    y1421_all = df_att_all["y_lo1421_db"].to_numpy(float)
    meter_all = df_att_all["power_meter_norm_db"].to_numpy(float)

    fit_lin_all = fit_shared_linear(
        L_all,
        y1420_all,
        y1421_all,
        sigma_L=SIGMA_CAL_LENGTH_M,
        mc_samples=LENGTH_UNCERTAINTY_MC_SAMPLES,
        mc_seed=LENGTH_UNCERTAINTY_MC_SEED + 11,
    )
    outlier_diag = robust_row_outlier_diagnostics(fit_lin_all, z_thresh=OUTLIER_Z_THRESHOLD)
    loo_alpha = leave_one_out_alpha(L_all, y1420_all, y1421_all)
    alpha_delta_loo = np.abs(loo_alpha - float(fit_lin_all["alpha"]))
    influence_mask = np.isfinite(alpha_delta_loo) & (alpha_delta_loo > INFLUENCE_ALPHA_DELTA_THRESHOLD)

    df_diag = df_att_all[["set_id", "cable_length_m", "siggen_amp_dbm"]].copy()
    df_diag["row_resid_norm_db"] = outlier_diag["row_resid_norm"]
    df_diag["robust_z"] = outlier_diag["robust_z"]
    df_diag["inlier_resid"] = outlier_diag["inlier_mask"]
    df_diag["alpha_if_row_dropped"] = loo_alpha
    df_diag["delta_alpha_loo"] = alpha_delta_loo
    df_diag["influence_flag"] = influence_mask
    tables["screening_diagnostics"] = df_diag.copy()

    drop_mask = (~outlier_diag["inlier_mask"]) | influence_mask
    can_screen = drop_mask.any() and np.sum(~drop_mask) >= MIN_POINTS_AFTER_SCREEN and np.unique(L_all[~drop_mask]).size >= 2
    screening_applied = bool(can_screen)
    df_att_used = df_att_all.loc[~drop_mask].copy().reset_index(drop=True) if screening_applied else df_att_all.copy().reset_index(drop=True)
    tables["attenuation_primary"] = df_att_used.copy()

    L = df_att_used["cable_length_m"].to_numpy(float)
    y1420 = df_att_used["y_lo1420_db"].to_numpy(float)
    y1421 = df_att_used["y_lo1421_db"].to_numpy(float)
    meter = df_att_used["power_meter_norm_db"].to_numpy(float)
    fit_lin_screened = fit_shared_linear(
        L,
        y1420,
        y1421,
        sigma_L=SIGMA_CAL_LENGTH_M,
        mc_samples=LENGTH_UNCERTAINTY_MC_SAMPLES,
        mc_seed=LENGTH_UNCERTAINTY_MC_SEED + 37,
    )
    fit_lin = fit_lin_screened if screening_applied else fit_lin_all

    SDR_FIXED_CORRECTION_DB = SPLITTER_S2_DB + ATTENUATOR_SDR_DB
    for df in (df_att_all, df_att_used, df_unk):
        df["y_lo1420_corrected_db"] = df["y_lo1420_db"] + SDR_FIXED_CORRECTION_DB
        df["y_lo1421_corrected_db"] = df["y_lo1421_db"] + SDR_FIXED_CORRECTION_DB
        df["power_meter_corrected_db"] = df["power_meter_norm_db"] + SPLITTER_S1_DB

    fit_compare = pd.DataFrame(
        [
            {
                "fit_case": "all-point",
                "n_rows": len(L_all),
                "alpha [dB/m]": fit_lin_all["alpha"],
                "sigma_alpha": fit_lin_all["sigma_alpha"],
                "B1420 [dB]": fit_lin_all["B1420"],
                "B1421 [dB]": fit_lin_all["B1421"],
                "RMSE [dB]": fit_lin_all["rmse"],
                "AIC": fit_lin_all["aic"],
                "BIC": fit_lin_all["bic"],
            },
            {
                "fit_case": "primary-screened" if screening_applied else "primary-all-point",
                "n_rows": len(L),
                "alpha [dB/m]": fit_lin["alpha"],
                "sigma_alpha": fit_lin["sigma_alpha"],
                "B1420 [dB]": fit_lin["B1420"],
                "B1421 [dB]": fit_lin["B1421"],
                "RMSE [dB]": fit_lin["rmse"],
                "AIC": fit_lin["aic"],
                "BIC": fit_lin["bic"],
            },
        ]
    ).set_index("fit_case")
    tables["fit_compare"] = fit_compare

    L_line = np.linspace(np.min(L_all), np.max(L_all), 500)
    figures["cable_attenuation_lo"] = plot_cable_attenuation_lo(
        L_all=L_all,
        y1420_all=y1420_all,
        y1421_all=y1421_all,
        drop_mask=drop_mask,
        fit_lin_all=fit_lin_all,
        fit_lin=fit_lin,
        L=L,
        L_line=L_line,
    )

    fit_meter_all = fit_single_linear(
        L_all,
        meter_all,
        sigma_L=SIGMA_CAL_LENGTH_M,
        mc_samples=LENGTH_UNCERTAINTY_MC_SAMPLES,
        mc_seed=LENGTH_UNCERTAINTY_MC_SEED + 101,
    )
    fit_meter_screened = fit_single_linear(
        L,
        meter,
        sigma_L=SIGMA_CAL_LENGTH_M,
        mc_samples=LENGTH_UNCERTAINTY_MC_SAMPLES,
        mc_seed=LENGTH_UNCERTAINTY_MC_SEED + 131,
    )
    fit_meter = fit_meter_screened if screening_applied else fit_meter_all
    meter_compare = pd.DataFrame(
        [
            {
                "fit_case": "all-point",
                "n_rows": len(L_all),
                "alpha [dB/m]": fit_meter_all["alpha"],
                "sigma_alpha": fit_meter_all["sigma_alpha"],
                "RMSE [dB]": fit_meter_all["rmse"],
                "AIC": fit_meter_all["aic"],
                "BIC": fit_meter_all["bic"],
            },
            {
                "fit_case": "primary-screened" if screening_applied else "primary-all-point",
                "n_rows": len(L),
                "alpha [dB/m]": fit_meter["alpha"],
                "sigma_alpha": fit_meter["sigma_alpha"],
                "RMSE [dB]": fit_meter["rmse"],
                "AIC": fit_meter["aic"],
                "BIC": fit_meter["bic"],
            },
        ]
    ).set_index("fit_case")
    tables["meter_compare"] = meter_compare
    sdr_line_n = -fit_lin["alpha"] * L_line
    meter_line_n = -fit_meter["alpha"] * L_line
    y1420_n = y1420 - fit_lin["B1420"]
    meter_n = meter - fit_meter["B"]
    meter_resid = meter - (fit_meter["B"] - fit_meter["alpha"] * L)
    figures["cable_attenuation_power_meter"] = plot_cable_attenuation_power_meter(
        L=L,
        y1420_n=y1420_n,
        L_line=L_line,
        sdr_line_n=sdr_line_n,
        meter_n=meter_n,
        meter_line_n=meter_line_n,
        fit_lin=fit_lin,
        meter_resid=meter_resid,
    )

    meter_df = df_att_used[["set_id", "cable_length_m", "power_meter_dbm"]].copy()
    meter_df["power_w"] = meter_df["power_meter_dbm"].map(dbm_to_watts)
    meter_df["v_rms_v"] = meter_df["power_w"].map(watts_to_vrms)
    meter_df["v_pp_v"] = meter_df["v_rms_v"].map(vrms_to_vpp)
    tables["meter_voltage"] = meter_df.sort_values(["cable_length_m", "set_id"]).reset_index(drop=True)

    ANALOG_SIGGEN_DBM = 20.0
    CHAIN_DELTA_UNCERTAINTY_DB = np.sqrt(2.0) * POWER_METER_UNCERTAINTY_DB
    bench_rows = [
        {"setup_id": "A1_ref_2x6ft", "purpose": "Reference path (two 6-ft cables)", "path": "siggen -> 6-ft -> 6-ft -> power meter", "p_in_dbm": ANALOG_SIGGEN_DBM, "p_out_dbm": 15.7},
        {"setup_id": "A2_unknown_only", "purpose": "Unknown cable loss measurement", "path": "siggen -> 6-ft -> 6-ft -> unknown cable -> power meter", "p_in_dbm": ANALOG_SIGGEN_DBM, "p_out_dbm": -13.6},
        {"setup_id": "A3_roof_chain", "purpose": "Unknown cable + (ZKL + Reactel + ZKL + ZKL)", "path": "A2 path + 6-ft + ZKL -> Reactel -> ZKL -> ZKL -> power meter", "p_in_dbm": ANALOG_SIGGEN_DBM, "p_out_dbm": 17.9},
        {"setup_id": "A4_lab_chain", "purpose": "K&L + wideband amp + narrowband amp", "path": "siggen -> 6-ft -> 6-ft -> K&L -> wideband amp -> narrowband amp -> power meter", "p_in_dbm": ANALOG_SIGGEN_DBM, "p_out_dbm": 7.5},
        {"setup_id": "A5_lab_filter_only", "purpose": "K&L insertion loss only", "path": "siggen -> 6-ft -> 6-ft -> K&L -> power meter", "p_in_dbm": ANALOG_SIGGEN_DBM, "p_out_dbm": -4.6},
    ]
    df_bench = pd.DataFrame(bench_rows)
    df_bench["p_in_sigma_dbm"] = POWER_METER_UNCERTAINTY_DB
    df_bench["p_out_sigma_dbm"] = POWER_METER_UNCERTAINTY_DB
    df_bench["chain_delta_db"] = df_bench["p_out_dbm"] - df_bench["p_in_dbm"]
    df_bench["chain_delta_uncertainty_db"] = CHAIN_DELTA_UNCERTAINTY_DB
    tables["bench_log"] = df_bench.copy()

    def _pout(setup_id: str) -> float:
        return float(df_bench.loc[df_bench["setup_id"] == setup_id, "p_out_dbm"].iloc[0])

    m_ref = _pout("A1_ref_2x6ft")
    m_unknown = _pout("A2_unknown_only")
    m_roof = _pout("A3_roof_chain")
    m_lab_full = _pout("A4_lab_chain")
    m_kl_only = _pout("A5_lab_filter_only")
    bench_metrics = {
        "baseline_loss_db": ANALOG_SIGGEN_DBM - m_ref,
        "unknown_incremental_loss_db": m_ref - m_unknown,
        "roof_chain_net_gain_db": m_roof - m_unknown,
        "kl_filter_incremental_loss_db": m_ref - m_kl_only,
        "wide_narrow_net_gain_db": m_lab_full - m_kl_only,
        "lab_net_vs_baseline_db": m_lab_full - m_ref,
    }
    df_bench_metrics = pd.DataFrame(
        [
            {
                "metric": key,
                "value_db": value,
                "sigma_db": CHAIN_DELTA_UNCERTAINTY_DB,
                "value_pm_sigma_db": f"{value:.3f} +/- {CHAIN_DELTA_UNCERTAINTY_DB:.3f}",
            }
            for key, value in bench_metrics.items()
        ]
    )
    tables["bench_metrics"] = df_bench_metrics

    alpha_m = float(fit_meter["alpha"])
    sigma_alpha_m = float(fit_meter["sigma_alpha"])
    delta_unknown_db = bench_metrics["unknown_incremental_loss_db"]
    sigma_delta_unknown_db = CHAIN_DELTA_UNCERTAINTY_DB
    L_unknown_bench_meter_m = delta_unknown_db / alpha_m
    sigma_L_alpha = abs(delta_unknown_db / (alpha_m**2)) * sigma_alpha_m
    sigma_L_measurement = sigma_delta_unknown_db / abs(alpha_m)
    L_unknown_bench_meter_sigma_m = float(np.sqrt(sigma_L_alpha**2 + sigma_L_measurement**2))

    per_6ft_loss_est_db = (ANALOG_SIGGEN_DBM - m_ref) / 2.0
    sigma_per_6ft_loss_db = POWER_METER_UNCERTAINTY_DB / 2.0
    p_stage1_in_est_dbm = m_unknown - per_6ft_loss_est_db
    sigma_p_stage1_dbm = np.sqrt(POWER_METER_UNCERTAINTY_DB**2 + sigma_per_6ft_loss_db**2)
    ZKL_GAIN_TYP_DB_1420 = 26.8
    ZKL_POUT_1DB_DBM_1420 = 36.8
    ZKL_PIN_1DB_DBM_1420 = ZKL_POUT_1DB_DBM_1420 - ZKL_GAIN_TYP_DB_1420
    ZKL_PIN_NO_DAMAGE_LIMIT_DBM_1420 = 0.0
    REACTEL_IL_ASSUMED_DB = 2.0
    p_stage2_in_est_dbm = p_stage1_in_est_dbm + ZKL_GAIN_TYP_DB_1420 - REACTEL_IL_ASSUMED_DB
    over_no_damage_db = max(0.0, p_stage2_in_est_dbm - ZKL_PIN_NO_DAMAGE_LIMIT_DBM_1420)
    safe_siggen_dbm_est = ANALOG_SIGGEN_DBM - over_no_damage_db
    safe_siggen_sigma_db = float(sigma_p_stage1_dbm) if over_no_damage_db > 0 else np.nan

    y0_obs = float(df_unk["y_lo1420_db"].iloc[0])
    y1_obs = float(df_unk["y_lo1421_db"].iloc[0])
    ym_obs = float(df_unk["power_meter_norm_db"].iloc[0])
    lin_result = infer_unknown_length_linear(y0_obs, y1_obs, fit_lin)
    L0_total = float(lin_result["L0"])
    L1_total = float(lin_result["L1"])
    L_total_lin = float(lin_result["L_total"])
    L_meter_total = (float(fit_meter["B"]) - ym_obs) / float(fit_meter["alpha"])
    L_unknown_lin = L_total_lin - UNKNOWN_LEAD_LENGTH_M
    L_meter_unknown = L_meter_total - UNKNOWN_LEAD_LENGTH_M

    sigma_y_sdr_obs = float(fit_lin["rmse"])
    sigma_y_meter_obs = float(fit_meter["rmse"])
    cov_m = np.asarray(fit_meter["cov"], float)
    sigma_L_meter_total = propagate_length_sigma(
        float(fit_meter["B"]),
        ym_obs,
        float(fit_meter["alpha"]),
        float(fit_meter["sigma_B"]),
        float(fit_meter["sigma_alpha"]),
        cov_B_alpha=float(cov_m[0, 1]),
        sigma_y=sigma_y_meter_obs,
    )
    sigma_L_meter_unknown = float(np.sqrt(sigma_L_meter_total**2 + SIGMA_LEAD_LENGTH_M**2))
    cov_s = np.asarray(fit_lin["cov"], float)
    sigma_L0_total = propagate_length_sigma(
        float(fit_lin["B1420"]),
        y0_obs,
        float(fit_lin["alpha"]),
        float(fit_lin["sigma_B1420"]),
        float(fit_lin["sigma_alpha"]),
        cov_B_alpha=float(cov_s[0, 2]),
        sigma_y=sigma_y_sdr_obs,
    )
    sigma_L1_total = propagate_length_sigma(
        float(fit_lin["B1421"]),
        y1_obs,
        float(fit_lin["alpha"]),
        float(fit_lin["sigma_B1421"]),
        float(fit_lin["sigma_alpha"]),
        cov_B_alpha=float(cov_s[1, 2]),
        sigma_y=sigma_y_sdr_obs,
    )
    alpha = float(fit_lin["alpha"])
    J_theta = np.array([1.0 / (2.0 * alpha), 1.0 / (2.0 * alpha), -(L0_total + L1_total) / (2.0 * alpha)], dtype=float)
    var_theta = float(J_theta @ cov_s @ J_theta)
    dLdy = -1.0 / (2.0 * alpha)
    var_y = (dLdy**2) * (sigma_y_sdr_obs**2 + sigma_y_sdr_obs**2)
    sigma_L_lin_total = float(np.sqrt(max(var_theta + var_y, 0.0)))
    sigma_L_lin_analytic = float(np.sqrt(sigma_L_lin_total**2 + SIGMA_LEAD_LENGTH_M**2))
    L_unknown_ci95 = (L_unknown_lin - 1.96 * sigma_L_lin_analytic, L_unknown_lin + 1.96 * sigma_L_lin_analytic)
    q025 = float(fit_lin["alpha"]) - 1.96 * float(fit_lin["sigma_alpha"])
    q975 = float(fit_lin["alpha"]) + 1.96 * float(fit_lin["sigma_alpha"])

    sdr_rule_reasons: list[str] = []
    if PRIMARY_RULE_REQUIRE_POSITIVE_ALPHA_CI95 and (q025 <= 0.0 <= q975):
        sdr_rule_reasons.append("SDR analytic alpha 95% CI crosses zero")
    rel_ci95_width = (L_unknown_ci95[1] - L_unknown_ci95[0]) / max(abs(L_unknown_lin), 1e-9)
    if np.isfinite(rel_ci95_width) and rel_ci95_width > PRIMARY_RULE_MAX_RELATIVE_CI95_WIDTH:
        sdr_rule_reasons.append(
            f"SDR analytic L_unknown 95% CI width/|estimate| = {rel_ci95_width:.2f} > {PRIMARY_RULE_MAX_RELATIVE_CI95_WIDTH:.2f}"
        )
    if sdr_rule_reasons:
        primary_length_source = "manifest_meter_linear_fallback"
        L_unknown_primary = float(L_meter_unknown)
        L_unknown_primary_sigma = float(sigma_L_meter_unknown)
    else:
        primary_length_source = "manifest_sdr_linear"
        L_unknown_primary = float(L_unknown_lin)
        L_unknown_primary_sigma = float(sigma_L_lin_analytic)

    crosscheck_rows = [
        {"source": "Manifest SDR", "L_unknown_m": float(L_unknown_lin), "sigma_m": float(sigma_L_lin_analytic)},
        {"source": "Manifest meter", "L_unknown_m": float(L_meter_unknown), "sigma_m": float(sigma_L_meter_unknown)},
        {"source": "Bench analog", "L_unknown_m": float(L_unknown_bench_meter_m), "sigma_m": float(L_unknown_bench_meter_sigma_m)},
        {"source": "Selected primary", "L_unknown_m": float(L_unknown_primary), "sigma_m": float(L_unknown_primary_sigma)},
    ]
    crosscheck_df = pd.DataFrame(crosscheck_rows)
    crosscheck_df["value_pm_sigma_m"] = crosscheck_df.apply(lambda row: f"{row['L_unknown_m']:.4f} +/- {row['sigma_m']:.4f}", axis=1)
    tables["unknown_crosscheck"] = crosscheck_df

    fit_quality = pd.DataFrame(
        [
            {"Fit": "SDR all-point", "n_rows": len(L_all), "alpha [dB/m]": f"{fit_lin_all['alpha']:.4f} +/- {fit_lin_all['sigma_alpha']:.4f}", "RMSE [dB]": f"{fit_lin_all['rmse']:.4f}", "AIC": f"{fit_lin_all['aic']:.2f}", "BIC": f"{fit_lin_all['bic']:.2f}"},
            {"Fit": "SDR primary", "n_rows": len(L), "alpha [dB/m]": f"{fit_lin['alpha']:.4f} +/- {fit_lin['sigma_alpha']:.4f}", "RMSE [dB]": f"{fit_lin['rmse']:.4f}", "AIC": f"{fit_lin['aic']:.2f}", "BIC": f"{fit_lin['bic']:.2f}"},
            {"Fit": "Meter all-point", "n_rows": len(L_all), "alpha [dB/m]": f"{fit_meter_all['alpha']:.4f} +/- {fit_meter_all['sigma_alpha']:.4f}", "RMSE [dB]": f"{fit_meter_all['rmse']:.4f}", "AIC": f"{fit_meter_all['aic']:.2f}", "BIC": f"{fit_meter_all['bic']:.2f}"},
            {"Fit": "Meter primary", "n_rows": len(L), "alpha [dB/m]": f"{fit_meter['alpha']:.4f} +/- {fit_meter['sigma_alpha']:.4f}", "RMSE [dB]": f"{fit_meter['rmse']:.4f}", "AIC": f"{fit_meter['aic']:.2f}", "BIC": f"{fit_meter['bic']:.2f}"},
        ]
    ).set_index("Fit")
    estimate_summary = pd.DataFrame(
        [
            {"Estimate": "Manifest SDR", "L_unknown [m]": f"{L_unknown_lin:.3f}", "Uncertainty": f"[{L_unknown_lin - sigma_L_lin_analytic:.2f}, {L_unknown_lin + sigma_L_lin_analytic:.2f}]", "Notes": "analytic +/-1sigma"},
            {"Estimate": "Manifest meter", "L_unknown [m]": f"{L_meter_unknown:.3f}", "Uncertainty": f"+/- {sigma_L_meter_unknown:.3f} m", "Notes": "analytic propagation"},
            {"Estimate": "Selected primary", "L_unknown [m]": f"{L_unknown_primary:.3f}", "Uncertainty": f"+/- {L_unknown_primary_sigma:.3f} m", "Notes": primary_length_source},
            {"Estimate": "Bench analog", "L_unknown [m]": f"{L_unknown_bench_meter_m:.3f}", "Uncertainty": f"+/- {L_unknown_bench_meter_sigma_m:.3f} m", "Notes": "consistency cross-check"},
        ]
    ).set_index("Estimate")
    tables["fit_quality"] = fit_quality
    tables["estimate_summary"] = estimate_summary

    F_HZ = 2.5e6
    T_NS = 1e9 / F_HZ
    C_MPS = 299_792_458.0
    VF_TARGET = 0.66
    T_FIRST_PLATEAU_START_NS = -402.0
    T_MAX_PLATEAU_START_NS = -304.0
    TAU_MOD_NS = T_MAX_PLATEAU_START_NS - T_FIRST_PLATEAU_START_NS
    L_SDR_M = float(L_unknown_lin)
    L_METER_M = float(L_meter_unknown)
    rows = []
    for n in range(4):
        dt_ns = TAU_MOD_NS + n * T_NS
        dt_s = dt_ns * 1e-9
        v_sdr = 2 * L_SDR_M / dt_s
        v_meter = 2 * L_METER_M / dt_s
        sigma_v_sdr = abs(v_sdr) * (sigma_L_lin_analytic / abs(L_SDR_M))
        sigma_v_meter = abs(v_meter) * (sigma_L_meter_unknown / abs(L_METER_M))
        rows.append(
            {
                "n": n,
                "delta_t_ns": dt_ns,
                "v_sdr_mps": v_sdr,
                "v_sdr_sigma_mps": sigma_v_sdr,
                "v_sdr_over_c": v_sdr / C_MPS,
                "v_sdr_over_c_sigma": sigma_v_sdr / C_MPS,
                "v_meter_mps": v_meter,
                "v_meter_sigma_mps": sigma_v_meter,
                "v_meter_over_c": v_meter / C_MPS,
                "v_meter_over_c_sigma": sigma_v_meter / C_MPS,
            }
        )
    branches = pd.DataFrame(rows)
    tables["reflectometry_branches"] = branches

    def pick_branch(v_over_c_col: str) -> int:
        sub = branches[branches[v_over_c_col] < 1.0].copy()
        sub["vf_dist"] = (sub[v_over_c_col] - VF_TARGET).abs()
        return int(sub.sort_values("vf_dist").iloc[0]["n"])

    n_sdr = pick_branch("v_sdr_over_c")
    n_meter = pick_branch("v_meter_over_c")
    sel_sdr = branches[branches["n"] == n_sdr].iloc[0]
    sel_meter = branches[branches["n"] == n_meter].iloc[0]
    if primary_length_source.startswith("manifest_sdr"):
        v_primary_mps = float(sel_sdr["v_sdr_mps"])
        sigma_v_primary_mps = float(sel_sdr["v_sdr_sigma_mps"])
        v_primary_over_c = float(sel_sdr["v_sdr_over_c"])
        sigma_v_primary_over_c = float(sel_sdr["v_sdr_over_c_sigma"])
    else:
        v_primary_mps = float(sel_meter["v_meter_mps"])
        sigma_v_primary_mps = float(sel_meter["v_meter_sigma_mps"])
        v_primary_over_c = float(sel_meter["v_meter_over_c"])
        sigma_v_primary_over_c = float(sel_meter["v_meter_over_c_sigma"])

    t_grid = np.linspace(-460, 20, 2400)
    anchor_t = np.array([-460, -428, -402, -324, -304, -230, -202, -124, -104, -26, 20], dtype=float)
    anchor_y = np.array([0.00, 0.00, 0.55, 0.55, 1.00, 1.00, 0.55, 0.55, 0.00, 0.00, 0.55], dtype=float)
    wave = np.interp(t_grid, anchor_t, anchor_y)
    TIMES_NS = [-428, -402, -324, -304, -230, -202, -124, -104, -26]
    figures["reflectometry"] = plot_reflectometry(
        t_grid=t_grid,
        wave=wave,
        TIMES_NS=TIMES_NS,
        T_FIRST_PLATEAU_START_NS=T_FIRST_PLATEAU_START_NS,
        T_MAX_PLATEAU_START_NS=T_MAX_PLATEAU_START_NS,
        TAU_MOD_NS=TAU_MOD_NS,
    )

    df_sweep = sdr_gain_sweep_manifest().copy()
    df_sweep["clip_max_frac"] = df_sweep[["i_clip_frac", "q_clip_frac"]].max(axis=1)
    df_sweep["meter_minus_set_db"] = df_sweep["manual_meter_dbm"] - df_sweep["siggen_amp_dbm"]
    df_sweep["is_clipped"] = df_sweep["clip_max_frac"] >= 1e-3
    fit_rows = []
    fig_sweep = None
    for lo_mhz, group in df_sweep.groupby("lo_mhz", sort=True):
        g = group.sort_values("siggen_amp_dbm").reset_index(drop=True)
        unclipped = g[g["is_clipped"] == False]
        clipped = g[g["is_clipped"]]
        slope = intercept = rmse = r2 = np.nan
        if len(unclipped) >= 2:
            xfit = unclipped["siggen_amp_dbm"].to_numpy(dtype=float)
            yfit = unclipped["total_power_db"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(xfit, yfit, 1)
            yhat = slope * xfit + intercept
            resid = yfit - yhat
            rmse = float(np.sqrt(np.mean(resid**2)))
            ss_res = float(np.sum(resid**2))
            ss_tot = float(np.sum((yfit - np.mean(yfit)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        fit_rows.append(
            {
                "lo_mhz": float(lo_mhz),
                "n_points": int(len(g)),
                "n_unclipped": int(len(unclipped)),
                "n_clipped": int(len(clipped)),
                "highest_unclipped_setpoint_dbm": float(np.max(unclipped["siggen_amp_dbm"])) if len(unclipped) else np.nan,
                "first_clipped_setpoint_dbm": float(np.min(clipped["siggen_amp_dbm"])) if len(clipped) else np.nan,
                "slope_total_power_db_per_dbm": slope,
                "rmse_db": rmse,
                "r2": r2,
                "meter_minus_set_mean_db": float(np.nanmean(g["meter_minus_set_db"])),
                "meter_minus_set_std_db": float(np.nanstd(g["meter_minus_set_db"])),
            }
        )
        if lo_mhz == 1420.0:
            fig_sweep = plot_sdr_gain_response_clipping(
                g=g,
                clipped=clipped,
                unclipped=unclipped,
                slope=slope,
                intercept=intercept,
            )
    fit_df = pd.DataFrame(fit_rows).sort_values("lo_mhz").reset_index(drop=True)
    required_sweep_cols = [
        "highest_unclipped_setpoint_dbm",
        "first_clipped_setpoint_dbm",
        "rmse_db",
    ]
    if fit_df[required_sweep_cols].isna().any(axis=1).any():
        raise ValueError(
            "SDR gain sweep must provide unclipped headroom, clipped onset, "
            "and RMSE for every LO before writing the equipment artifact."
        )
    tables["sweep_fit"] = fit_df
    if fig_sweep is not None:
        figures["sdr_gain_response_clipping"] = fig_sweep

    spec_noise = Spectrum.load(COLD_REF_1420_PATH)
    N_FFT = int(spec_noise.psd.size)
    freq_offset_hz = spec_noise.freqs - float(spec_noise.center_freq)
    freq_offset_mhz = freq_offset_hz / 1e6
    P_fir_norm_shifted = power_response_on_output_axis(H_FIR, freq_offset_hz, RTL_INTERNAL_SAMPLE_RATE_HZ)
    P_sum_est_norm_shifted = power_response_on_output_axis(G_SUM_EST, freq_offset_hz, RTL_INTERNAL_SAMPLE_RATE_HZ)
    P_combined_est_norm_shifted = P_fir_norm_shifted * P_sum_est_norm_shifted
    combined_peak = float(np.nanmax(P_combined_est_norm_shifted))
    if np.isfinite(combined_peak) and combined_peak > 0:
        P_combined_est_norm_shifted = P_combined_est_norm_shifted / combined_peak
    passband_floor = 10 ** (PASSBAND_DB_THRESHOLD / 10.0)
    passband_mask = P_fir_norm_shifted >= passband_floor
    noise_center_idx = int(spec_noise.bin_at(float(spec_noise.center_freq)))
    noise_analysis_mask = np.ones(N_FFT, dtype=bool)
    noise_analysis_mask[noise_center_idx] = False
    noise_psd_shifted = np.array(spec_noise.psd_values(mask_dc=True), float, copy=True)
    window = min(257, N_FFT - (1 - N_FFT % 2))
    if window < 7:
        window = 7 if N_FFT >= 7 else (N_FFT // 2) * 2 + 1
    baseline_seed = _fill_nan_linear(noise_psd_shifted)
    baseline = np.array(
        spec_noise.mask_dc_bin(savgol_filter(baseline_seed, window_length=window, polyorder=3, mode="interp")),
        float,
        copy=True,
    )
    resid = noise_psd_shifted - baseline
    finite_resid = resid[np.isfinite(resid)]
    med_resid = float(np.median(finite_resid))
    mad = float(np.median(np.abs(finite_resid - med_resid)))
    scale = 1.4826 * mad if mad > 0 else float(np.nanstd(finite_resid))
    rfi_mask = noise_analysis_mask if scale <= 0 else noise_analysis_mask & (np.abs(resid - med_resid) < 3.0 * scale)
    combined_mask = passband_mask & noise_analysis_mask & rfi_mask
    if np.sum(combined_mask) < 64:
        combined_mask = passband_mask & noise_analysis_mask
    noise_norm = _normalise_in_mask(noise_psd_shifted, combined_mask)
    after_fir = noise_norm / np.clip(P_fir_norm_shifted, RESPONSE_FLOOR, None)
    after_fir_n = _normalise_in_mask(after_fir, combined_mask)

    def summing_response_norm_shifted(g_coeffs: np.ndarray) -> np.ndarray:
        return power_response_on_output_axis(g_coeffs, freq_offset_hz, RTL_INTERNAL_SAMPLE_RATE_HZ)

    def corrected_with_summing(P_sum_norm_shifted: np.ndarray) -> np.ndarray:
        return noise_norm / np.clip(P_fir_norm_shifted * P_sum_norm_shifted, RESPONSE_FLOOR, None)

    def objective(theta6: np.ndarray) -> float:
        g = make_symmetric_sum_filter(theta6)
        P_sum = summing_response_norm_shifted(g)
        corr = corrected_with_summing(P_sum)
        x = np.asarray(corr, float)[combined_mask]
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < 64:
            return 1e10
        x = x / np.median(x)
        return float(np.var(x))

    theta_init = np.array([-1 / 8, -1 / 4, -3 / 4, -1 / 2, -1, 8], dtype=float)
    g_init = make_symmetric_sum_filter(theta_init)
    P_sum_init_norm_shifted = summing_response_norm_shifted(g_init)
    after_init_n = _normalise_in_mask(corrected_with_summing(P_sum_init_norm_shifted), combined_mask)
    optimization = minimize(
        objective,
        theta_init,
        method="Nelder-Mead",
        options={"maxiter": 60_000, "xatol": 1e-9, "fatol": 1e-12, "adaptive": True},
    )
    theta_opt = np.asarray(optimization.x, float)
    g_opt = make_symmetric_sum_filter(theta_opt)
    P_sum_opt_norm_shifted = summing_response_norm_shifted(g_opt)
    after_opt_n = _normalise_in_mask(corrected_with_summing(P_sum_opt_norm_shifted), combined_mask)
    figures["sdr_fir_summing_correction"] = plot_sdr_fir_summing_correction(
        freq_offset_mhz=freq_offset_mhz,
        combined_mask=combined_mask,
        noise_norm=noise_norm,
        after_init_n=after_init_n,
        after_opt_n=after_opt_n,
    )
    tables["response_summary"] = pd.DataFrame(
        [
            {"metric": "FIR passband ripple [dB]", "value": _pct_ripple_db(P_fir_norm_shifted, passband_mask)},
            {"metric": "Combined passband ripple [dB]", "value": _pct_ripple_db(P_combined_est_norm_shifted, passband_mask)},
            {"metric": "Whitened frac std after FIR", "value": _frac_std(after_fir_n, combined_mask)},
            {"metric": "Whitened frac std after optimized FIR+sum", "value": _frac_std(after_opt_n, combined_mask)},
        ]
    )

    highest_unclipped_setpoint_dbm = float(fit_df["highest_unclipped_setpoint_dbm"].min())
    first_clipped_setpoint_dbm = float(fit_df["first_clipped_setpoint_dbm"].min())
    sweep_rmse_db = fit_df["rmse_db"].to_numpy(dtype=float)
    sweep_lo_mhz = fit_df["lo_mhz"].to_numpy(dtype=float)
    clip_threshold = 1e-3
    artifact = {
        "schema_version": np.str_("2.0.0"),
        "model.alpha_db_per_m": np.float64(float(fit_lin["alpha"])),
        "model.sigma_alpha_db_per_m": np.float64(float(fit_lin["sigma_alpha"])),
        "model.fit_method": np.str_("shared_linear_screened" if screening_applied else "shared_linear_all_points"),
        "length.unknown_m": np.float64(L_unknown_primary),
        "length.sigma_unknown_m": np.float64(L_unknown_primary_sigma),
        "length.method": np.str_(primary_length_source),
        "length.lead_subtracted_m": np.float64(UNKNOWN_LEAD_LENGTH_M),
        "instrument.ruler_division_m": np.float64(METER_RULE_DIVISION_M),
        "instrument.ruler_sigma_read_m": np.float64(SIGMA_LENGTH_READ_M),
        "instrument.power_meter_division_dbm": np.float64(POWER_METER_DIVISION_DB),
        "instrument.power_meter_sigma_read_dbm": np.float64(POWER_METER_UNCERTAINTY_DB),
        "reflectometry.v_primary_mps": np.float64(v_primary_mps),
        "reflectometry.sigma_v_primary_mps": np.float64(sigma_v_primary_mps),
        "reflectometry.v_primary_over_c": np.float64(v_primary_over_c),
        "reflectometry.sigma_v_primary_over_c": np.float64(sigma_v_primary_over_c),
        "response.freq_offset_mhz": np.asarray(freq_offset_mhz, dtype=np.float64),
        "response.fir_power_norm": np.asarray(P_fir_norm_shifted, dtype=np.float64),
        "response.sum_power_norm": np.asarray(P_sum_opt_norm_shifted, dtype=np.float64),
        "response.combined_power_norm": np.asarray(_normalise_in_mask(P_fir_norm_shifted * P_sum_opt_norm_shifted, np.isfinite(P_fir_norm_shifted)), dtype=np.float64),
        "response.passband_mask": np.asarray(passband_mask, dtype=bool),
        "response.eval_mask": np.asarray(combined_mask, dtype=bool),
        "response.floor": np.float64(RESPONSE_FLOOR),
        "linearity.highest_unclipped_setpoint_dbm": np.float64(highest_unclipped_setpoint_dbm),
        "linearity.first_clipped_setpoint_dbm": np.float64(first_clipped_setpoint_dbm),
        "linearity.lo_mhz": np.asarray(sweep_lo_mhz, dtype=np.float64),
        "linearity.sweep_rmse_db": np.asarray(sweep_rmse_db, dtype=np.float64),
        "linearity.clip_threshold": np.float64(clip_threshold),
        "provenance.source_notebook": np.str_("labs/02/equipment_calibration.ipynb"),
        "provenance.created_utc": np.str_(datetime.now(timezone.utc).isoformat()),
        "provenance.git_commit": np.str_(_git_commit_short()),
        "trace.requirement_ids": np.array(
            ["R-SC-001", "R-SC-002", "R-SC-003", "R-SC-004", "R-SC-005", "R-SC-006", "R-SC-007", "R-CAL-001", "R-CAL-002", "R-COORD-001"],
            dtype="U32",
        ),
    }
    save_npz(EQUIPMENT_ARTIFACT_PATH, artifact)

    values.update(
        {
            "fit_lin": fit_lin,
            "fit_lin_all": fit_lin_all,
            "fit_meter": fit_meter,
            "fit_meter_all": fit_meter_all,
            "screening_applied": screening_applied,
            "primary_length_source": primary_length_source,
            "L_unknown_primary": L_unknown_primary,
            "L_unknown_primary_sigma": L_unknown_primary_sigma,
            "safe_siggen_dbm_est": safe_siggen_dbm_est,
            "safe_siggen_sigma_db": safe_siggen_sigma_db,
        }
    )
    return EquipmentCalibrationResult(
        artifact=artifact,
        artifact_path=EQUIPMENT_ARTIFACT_PATH,
        tables=tables,
        figures=figures,
        values=values,
    )
