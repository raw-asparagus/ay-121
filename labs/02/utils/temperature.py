from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ugradiolab import Spectrum

from .common import (
    apply_hardware_response_correction,
    combine_spectrum_mask,
    fill_masked_spectrum_values,
    interp_bool_nearest,
    interp_mono,
    load_lo_pair,
    lo_center_bin_index,
    lo_analysis_mask,
    masked_spectrum_values,
    masked_total_power,
    omit_lo_center_bin_mask,
    sigma_clip_rfi_mask,
    smooth_series,
)
from .constants import (
    BEAM_FILLING_FACTOR_DEFAULT,
    C_LIGHT_M_S,
    HARDWARE_RESPONSE_COLLAPSE_THRESH_FRAC,
    HARDWARE_RESPONSE_MIN,
    HARDWARE_RESPONSE_RELATIVE_SUPPORT_THRESH,
    HORN_APERTURE_AREA_M2,
    HORN_PARALLEL_M,
    HORN_PERP_HORIZON_M,
    LO_FREQS_MHZ,
    OMEGA_BEAM_APPROX_SR_HI,
    SAVGOL,
    SIGMA_T_COLD,
    SIGMA_T_HOT,
    T_COLD,
    T_HOT,
)
from .contracts import TemperatureCalibrationResult
from .io import load_equipment_artifact, save_npz
from .paths import DATA_ROOT, HUMAN_SPECTRA_DIR, COLD_REF_SPECTRA_DIR, TEMPERATURE_ARTIFACT_PATH, ensure_output_dirs
from .plotting import plot_per_frequency_trx, plot_sigma_masking


def hardware_systematic_fraction(eq: dict[str, object]) -> tuple[float, float, float, float]:
    sigma_alpha = float(eq["sigma_alpha_db_per_m"])
    L_unknown = float(eq["unknown_cable_length_m"])
    att_frac_raw = np.log(10.0) / 10.0 * abs(sigma_alpha) * abs(L_unknown)
    att_frac = 0.0
    rmse_arr = np.asarray(eq["sweep_rmse_db"], float)
    rmse_db = float(np.nanmedian(rmse_arr)) if rmse_arr.size and np.isfinite(np.nanmedian(rmse_arr)) else 0.2
    lin_frac = np.log(10.0) / 20.0 * abs(rmse_db)
    frac = float(np.sqrt(att_frac**2 + lin_frac**2))
    return frac, att_frac_raw, att_frac, rmse_db


def _response_support_fraction(resp: np.ndarray, eval_mask: np.ndarray | None = None) -> float:
    arr = np.asarray(resp, float)
    finite = np.isfinite(arr)
    if eval_mask is not None:
        finite = finite & np.asarray(eval_mask, bool)
    if not np.any(finite):
        return np.nan
    peak = float(np.nanmax(arr[finite]))
    if not np.isfinite(peak) or peak <= 0:
        return np.nan
    return float(np.mean((arr >= 0.1 * peak) & finite))


def select_equipment_response_model(eq: dict[str, object]) -> tuple[np.ndarray, str, dict[str, float | bool]]:
    eq_fir = np.asarray(eq["fir_response_norm"], float)
    eq_sum = np.asarray(eq["sum_response_norm"], float)
    eq_eval = np.asarray(eq["combined_eval_mask"], bool)
    eq_combined = eq_fir * eq_sum
    support_combined = _response_support_fraction(eq_combined, eq_eval)
    support_fir = _response_support_fraction(eq_fir, eq_eval)
    collapsed = (
        np.isfinite(support_combined)
        and np.isfinite(support_fir)
        and support_combined < HARDWARE_RESPONSE_COLLAPSE_THRESH_FRAC
        and support_combined < HARDWARE_RESPONSE_RELATIVE_SUPPORT_THRESH * support_fir
    )
    if collapsed:
        chosen = eq_fir
        variant = "fir_only_fallback_combined_collapsed"
    else:
        chosen = eq_combined
        variant = "fir_times_sum"
    diagnostics = {
        "support_combined": float(support_combined) if np.isfinite(support_combined) else np.nan,
        "support_fir": float(support_fir) if np.isfinite(support_fir) else np.nan,
        "collapsed": bool(collapsed),
    }
    return np.asarray(chosen, float), variant, diagnostics


def hardware_response_on_axis(
    eq: dict[str, object],
    spectrum: Spectrum,
    eq_resp_model: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    eq_offset = np.asarray(eq["freq_offset_mhz"], float)
    eq_pass = np.asarray(eq["passband_mask"], bool)
    eq_eval = np.asarray(eq["combined_eval_mask"], bool)
    eq_resp = (
        np.asarray(eq["fir_response_norm"], float) * np.asarray(eq["sum_response_norm"], float)
        if eq_resp_model is None
        else np.asarray(eq_resp_model, float)
    )
    eq_floor_nom = max(float(eq["response_floor"]), HARDWARE_RESPONSE_MIN)
    eval_finite = eq_eval & np.isfinite(eq_resp)
    if np.any(eval_finite):
        eval_floor = float(np.nanmin(eq_resp[eval_finite]))
    else:
        eval_floor = float(np.nanmin(eq_resp[np.isfinite(eq_resp)]))
    eq_floor_eff = max(HARDWARE_RESPONSE_MIN, min(eq_floor_nom, eval_floor))
    offset_mhz = (np.asarray(spectrum.freqs, float) - float(spectrum.center_freq)) / 1e6
    resp = interp_mono(eq_offset, eq_resp, offset_mhz, fill_value=np.nan)
    pass_mask = interp_bool_nearest(eq_offset, eq_pass, offset_mhz, default=False)
    eval_mask = interp_bool_nearest(eq_offset, eq_eval, offset_mhz, default=False)
    hw_mask = np.isfinite(resp) & (resp >= HARDWARE_RESPONSE_MIN)
    return resp, hw_mask, eq_floor_eff, pass_mask, eval_mask


def build_cold_reference_profile(
    spectrum: Spectrum,
    mask: np.ndarray,
    smooth_kwargs: dict = SAVGOL,
    min_floor_frac: float = 1e-6,
    response: np.ndarray | None = None,
    response_floor: float = HARDWARE_RESPONSE_MIN,
) -> np.ndarray:
    good = combine_spectrum_mask(spectrum, mask, require_nonempty=True)
    ref_seed = fill_masked_spectrum_values(spectrum, mask=good)
    ref = masked_spectrum_values(spectrum, smooth_series(ref_seed, smooth_kwargs), mask=good)
    if response is not None:
        ref = apply_hardware_response_correction(ref, response, response_floor)
        ref = masked_spectrum_values(spectrum, ref, mask=good)
    good_stats = combine_spectrum_mask(spectrum, good, np.isfinite(ref), require_nonempty=True)
    med = float(np.nanmedian(ref[good_stats]))
    floor = max(med * float(min_floor_frac), np.finfo(float).tiny)
    ref = np.where(np.isfinite(ref), ref, med)
    ref = np.clip(ref, floor, None)
    ref[~good] = np.nan
    return ref


@dataclass(frozen=True)
class YFactorResult:
    lo_mhz: int
    Y: float
    sigma_Y: float
    Y_dB: float
    sigma_Y_dB: float
    T_rx: float
    sigma_T_rx_meas: float
    sigma_T_rx_loads: float
    sigma_T_rx_hw: float
    sigma_hw_frac: float
    sigma_T_rx_total: float
    T_hot: float
    T_cold: float
    P_hot: float
    sigma_P_hot: float
    P_cold: float
    sigma_P_cold: float


def measure_y_factor(
    p_hot: float,
    sigma_hot: float,
    p_cold: float,
    sigma_cold: float,
    t_hot: float,
    t_cold: float,
    hardware_frac: float = 0.0,
    lo_mhz: int = 0,
) -> YFactorResult:
    Y = p_hot / p_cold
    sigma_Y = Y * np.sqrt((sigma_hot / p_hot) ** 2 + (sigma_cold / p_cold) ** 2)
    Y_dB = 10.0 * np.log10(Y)
    sigma_Y_dB = (10.0 / np.log(10.0)) * (sigma_Y / Y)
    T_rx = (t_hot - Y * t_cold) / (Y - 1.0)
    dT_dY = (t_cold - t_hot) / (Y - 1.0) ** 2
    dT_dTh = 1.0 / (Y - 1.0)
    dT_dTc = -Y / (Y - 1.0)
    sigma_T_meas = abs(dT_dY) * sigma_Y
    sigma_T_loads = float(np.sqrt((dT_dTh * SIGMA_T_HOT) ** 2 + (dT_dTc * SIGMA_T_COLD) ** 2))
    sigma_hw_frac = max(float(hardware_frac), 0.0)
    sigma_T_hw = abs(float(T_rx)) * sigma_hw_frac
    sigma_T_total = float(np.sqrt(sigma_T_meas**2 + sigma_T_loads**2 + sigma_T_hw**2))
    return YFactorResult(
        lo_mhz=lo_mhz,
        Y=float(Y),
        sigma_Y=float(sigma_Y),
        Y_dB=float(Y_dB),
        sigma_Y_dB=float(sigma_Y_dB),
        T_rx=float(T_rx),
        sigma_T_rx_meas=float(sigma_T_meas),
        sigma_T_rx_loads=float(sigma_T_loads),
        sigma_T_rx_hw=float(sigma_T_hw),
        sigma_hw_frac=float(sigma_hw_frac),
        sigma_T_rx_total=float(sigma_T_total),
        T_hot=float(t_hot),
        T_cold=float(t_cold),
        P_hot=float(p_hot),
        sigma_P_hot=float(sigma_hot),
        P_cold=float(p_cold),
        sigma_P_cold=float(sigma_cold),
    )


def run_temperature_calibration() -> TemperatureCalibrationResult:
    ensure_output_dirs()
    tables: dict[str, object] = {}
    figures: dict[str, object] = {}
    values: dict[str, object] = {}

    human_pair = load_lo_pair(HUMAN_SPECTRA_DIR)
    cold_ref_pair = load_lo_pair(COLD_REF_SPECTRA_DIR)
    loaded_spectra = {}
    loaded_lo_masks = {}
    for label, pair in [("human", human_pair), ("cold_ref", cold_ref_pair)]:
        for lo in LO_FREQS_MHZ:
            loaded_spectra[(label, lo)] = pair[lo]
            loaded_lo_masks[(label, lo)] = lo_analysis_mask(pair[lo])

    eq_path, eq = load_equipment_artifact()
    hardware_systematic_frac, hardware_att_frac_raw, hardware_att_frac_used, hardware_rmse_db = hardware_systematic_fraction(eq)
    eq_response_model, eq_response_variant, eq_response_diag = select_equipment_response_model(eq)
    hardware = {
        "path": str(eq_path),
        "schema_version": str(eq["schema_version"].item() if np.asarray(eq["schema_version"]).ndim == 0 else eq["schema_version"]),
        "alpha_db_per_m": float(eq["alpha_db_per_m"]),
        "sigma_alpha_db_per_m": float(eq["sigma_alpha_db_per_m"]),
        "unknown_cable_length_m": float(eq["unknown_cable_length_m"]),
        "unknown_cable_length_sigma_m": float(eq["unknown_cable_length_sigma_m"]),
        "highest_unclipped_setpoint_dbm": float(eq["highest_unclipped_setpoint_dbm"]),
        "first_clipped_setpoint_dbm": float(eq["first_clipped_setpoint_dbm"]),
        "clip_threshold": float(eq["clip_threshold"]),
        "response_floor": max(float(eq["response_floor"]), HARDWARE_RESPONSE_MIN),
        "linearity_rmse_db": float(hardware_rmse_db),
        "att_frac_raw": float(hardware_att_frac_raw),
        "att_frac_used": float(hardware_att_frac_used),
        "systematic_fraction": float(hardware_systematic_frac),
        "att_model": "two_load_common_mode_cancelled",
        "response_variant": str(eq_response_variant),
        "response_support_combined": float(eq_response_diag["support_combined"]),
        "response_support_fir": float(eq_response_diag["support_fir"]),
    }

    masks_rfi = {}
    hardware_masks = {}
    hardware_response = {}
    passband_masks = {}
    eval_masks = {}
    hardware_floors = []
    for label, pair in [("human", human_pair), ("cold_ref", cold_ref_pair)]:
        for lo in LO_FREQS_MHZ:
            spec = pair[lo]
            rfi_mask = sigma_clip_rfi_mask(spec)
            resp, hw_mask, eff_floor, pass_mask, eval_mask = hardware_response_on_axis(eq, spec, eq_resp_model=eq_response_model)
            masks_rfi[(label, lo)] = rfi_mask
            hardware_masks[(label, lo)] = hw_mask
            hardware_response[(label, lo)] = resp
            passband_masks[(label, lo)] = pass_mask
            eval_masks[(label, lo)] = eval_mask
            hardware_floors.append(float(eff_floor))
    hardware["response_floor"] = float(np.nanmin(hardware_floors)) if hardware_floors else hardware["response_floor"]
    masks = {
        key: combine_spectrum_mask(loaded_spectra[key], loaded_lo_masks[key], masks_rfi[key], hardware_masks[key], require_nonempty=True)
        for key in masks_rfi
    }

    worst_info = None
    for data_dir in sorted(DATA_ROOT.glob("*_combined_spectra")):
        try:
            pair = load_lo_pair(data_dir)
        except Exception:
            continue
        for lo, spec in pair.items():
            mask = sigma_clip_rfi_mask(spec)
            bad = int(np.sum(~mask))
            total = len(mask)
            if worst_info is None or bad > worst_info["bad"]:
                worst_info = {
                    "label": data_dir.name,
                    "lo": lo,
                    "spec": spec,
                    "mask": mask,
                    "bad": bad,
                    "total": total,
                }
    if worst_info is not None:
        worst_freqs = np.asarray(worst_info["spec"].freqs, float) / 1e6
        worst_psd = masked_spectrum_values(worst_info["spec"])
        worst_mask = combine_spectrum_mask(worst_info["spec"], worst_info["mask"])
        center_idx = lo_center_bin_index(worst_info["spec"])
        flagged_idxs = np.where(~worst_mask)[0]
        non_center_flagged = flagged_idxs[flagged_idxs != center_idx]
        if non_center_flagged.size >= 5:
            worst_mask[non_center_flagged[4]] = True
        figures["sigma_masking"] = plot_sigma_masking(
            worst_freqs=worst_freqs,
            worst_psd=worst_psd,
            worst_mask=worst_mask,
        )

    rows = []
    yfactor_results = {}
    yfactor_common_masks = {}
    for lo in LO_FREQS_MHZ:
        hot_mask = np.asarray(masks[("human", lo)], bool)
        cold_mask = np.asarray(masks[("cold_ref", lo)], bool)
        common_mask = hot_mask & cold_mask
        yfactor_common_masks[lo] = common_mask
        p_hot, s_hot = masked_total_power(human_pair[lo], common_mask)
        p_cold, s_cold = masked_total_power(cold_ref_pair[lo], common_mask)
        yfactor_results[lo] = measure_y_factor(
            p_hot,
            s_hot,
            p_cold,
            s_cold,
            T_HOT,
            T_COLD,
            hardware_frac=hardware["systematic_fraction"],
            lo_mhz=lo,
        )
        r = yfactor_results[lo]
        rows.append(
            {
                "LO [MHz]": lo,
                "N_common": int(np.sum(yfactor_common_masks[lo])),
                "P_hot": f"{r.P_hot:.4f}",
                "P_cold": f"{r.P_cold:.4f}",
                "Y": f"{r.Y:.6f}",
                "Y [dB]": f"{r.Y_dB:.4f} +/- {r.sigma_Y_dB:.4f}",
                "T_rx [K]": f"{r.T_rx:.2f}",
                "sigma_meas [K]": f"{r.sigma_T_rx_meas:.2f}",
                "sigma_loads [K]": f"{r.sigma_T_rx_loads:.2f}",
                "sigma_hw [K]": f"{r.sigma_T_rx_hw:.2f}",
                "sigma_total [K]": f"{r.sigma_T_rx_total:.2f}",
            }
        )
    tables["yfactor_summary"] = pd.DataFrame(rows).set_index("LO [MHz]")

    y_spec = {}
    sy_spec = {}
    trx_spec = {}
    for lo in LO_FREQS_MHZ:
        mask = np.asarray(yfactor_common_masks[lo], bool)
        p_h = masked_spectrum_values(human_pair[lo], mask=mask)
        s_h = masked_spectrum_values(human_pair[lo], human_pair[lo].std, mask=mask)
        p_c = masked_spectrum_values(cold_ref_pair[lo], mask=mask)
        s_c = masked_spectrum_values(cold_ref_pair[lo], cold_ref_pair[lo].std, mask=mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = p_h / p_c
            sy = y * np.sqrt((s_h / p_h) ** 2 + (s_c / p_c) ** 2)
            trx = (T_HOT - y * T_COLD) / (y - 1)
        y_spec[lo] = y
        sy_spec[lo] = sy
        trx_spec[lo] = trx

    figures["per_frequency_trx"] = plot_per_frequency_trx(
        human_pair=human_pair,
        yfactor_common_masks=yfactor_common_masks,
        trx_spec=trx_spec,
        yfactor_results=yfactor_results,
    )

    cross_rows = []
    for lo in LO_FREQS_MHZ:
        hot_mask = np.asarray(masks[("human", lo)], bool)
        cold_mask = np.asarray(masks[("cold_ref", lo)], bool)
        common_mask = hot_mask & cold_mask
        p_hot_common, _ = masked_total_power(human_pair[lo], common_mask)
        p_cold_common, _ = masked_total_power(cold_ref_pair[lo], common_mask)
        denom_common = p_hot_common - p_cold_common
        y_common = p_hot_common / p_cold_common
        t_rx_y_common = (T_HOT - y_common * T_COLD) / (y_common - 1.0)
        t_sys_y_common = float(T_COLD + t_rx_y_common)
        t_sys_cool_common = float((p_cold_common / denom_common) * (T_HOT - T_COLD))
        t_sys_y_production = float(yfactor_results[lo].T_rx + yfactor_results[lo].T_cold)
        p_hot_sep, _ = masked_total_power(human_pair[lo], hot_mask)
        p_cold_sep, _ = masked_total_power(cold_ref_pair[lo], cold_mask)
        t_sys_cool_sep = float((p_cold_sep / (p_hot_sep - p_cold_sep)) * (T_HOT - T_COLD))
        cross_rows.append(
            {
                "LO [MHz]": lo,
                "N_common": int(np.sum(common_mask)),
                "T_sys (Y, common mask) [K]": t_sys_y_common,
                "T_sys (cool, common mask) [K]": t_sys_cool_common,
                "Delta(common cool - common Y) [K]": t_sys_cool_common - t_sys_y_common,
                "T_sys (Y, production masks) [K]": t_sys_y_production,
                "Delta(prod Y - common Y) [K]": t_sys_y_production - t_sys_y_common,
                "T_sys (cool, separate masks) [K]": t_sys_cool_sep,
                "Delta(separate cool - common Y) [K]": t_sys_cool_sep - t_sys_y_common,
            }
        )
    cross_df = pd.DataFrame(cross_rows).set_index("LO [MHz]")
    tables["crosscheck"] = cross_df

    r1420 = yfactor_results[1420]
    r1421 = yfactor_results[1421]
    p_cold_total_1420, _ = masked_total_power(cold_ref_pair[1420], yfactor_common_masks[1420])
    p_cold_total_1421, _ = masked_total_power(cold_ref_pair[1421], yfactor_common_masks[1421])
    resp_1420 = hardware_response[("cold_ref", 1420)]
    resp_1421 = hardware_response[("cold_ref", 1421)]
    cold_ref_profile_1420 = build_cold_reference_profile(cold_ref_pair[1420], yfactor_common_masks[1420], smooth_kwargs=SAVGOL, response=resp_1420, response_floor=hardware["response_floor"])
    cold_ref_profile_1421 = build_cold_reference_profile(cold_ref_pair[1421], yfactor_common_masks[1421], smooth_kwargs=SAVGOL, response=resp_1421, response_floor=hardware["response_floor"])
    cold_ref_mask_1420 = np.asarray(yfactor_common_masks[1420], bool)
    cold_ref_mask_1421 = np.asarray(yfactor_common_masks[1421], bool)
    cold_ref_mask_stats_1420, lo_center_bin_index_1420 = omit_lo_center_bin_mask(cold_ref_pair[1420], cold_ref_mask_1420)
    cold_ref_mask_stats_1421, lo_center_bin_index_1421 = omit_lo_center_bin_mask(cold_ref_pair[1421], cold_ref_mask_1421)
    freq_hz_1420 = np.asarray(cold_ref_pair[1420].freqs, float)
    freq_hz_1421 = np.asarray(cold_ref_pair[1421].freqs, float)
    delta_nu_hz_1420 = float(cold_ref_pair[1420].bin_width)
    delta_nu_hz_1421 = float(cold_ref_pair[1421].bin_width)
    tau_s_1420 = float(cold_ref_pair[1420].nblocks * cold_ref_pair[1420].nsamples / cold_ref_pair[1420].sample_rate)
    tau_s_1421 = float(cold_ref_pair[1421].nblocks * cold_ref_pair[1421].nsamples / cold_ref_pair[1421].sample_rate)
    omega_beam_approx_sr_1420 = float((C_LIGHT_M_S / cold_ref_pair[1420].center_freq) ** 2 / HORN_APERTURE_AREA_M2)
    omega_beam_approx_sr_1421 = float((C_LIGHT_M_S / cold_ref_pair[1421].center_freq) ** 2 / HORN_APERTURE_AREA_M2)

    artifact = {
        "t_rx_1420": np.float64(r1420.T_rx),
        "sigma_t_rx_1420": np.float64(r1420.sigma_T_rx_total),
        "sigma_t_rx_meas_1420": np.float64(r1420.sigma_T_rx_meas),
        "sigma_t_rx_loads_1420": np.float64(r1420.sigma_T_rx_loads),
        "sigma_t_rx_hw_1420": np.float64(r1420.sigma_T_rx_hw),
        "t_rx_1421": np.float64(r1421.T_rx),
        "sigma_t_rx_1421": np.float64(r1421.sigma_T_rx_total),
        "sigma_t_rx_meas_1421": np.float64(r1421.sigma_T_rx_meas),
        "sigma_t_rx_loads_1421": np.float64(r1421.sigma_T_rx_loads),
        "sigma_t_rx_hw_1421": np.float64(r1421.sigma_T_rx_hw),
        "sigma_hw_fraction": np.float64(hardware["systematic_fraction"]),
        "t_cold": np.float64(T_COLD),
        "t_cold_1420": np.float64(T_COLD),
        "t_cold_1421": np.float64(T_COLD),
        "t_hot": np.float64(T_HOT),
        "p_cold_total_1420": np.float64(p_cold_total_1420),
        "p_cold_total_1421": np.float64(p_cold_total_1421),
        "cold_ref_profile_1420": np.asarray(cold_ref_profile_1420, dtype=np.float64),
        "cold_ref_profile_1421": np.asarray(cold_ref_profile_1421, dtype=np.float64),
        "cold_ref_mask_1420": np.asarray(cold_ref_mask_1420, dtype=bool),
        "cold_ref_mask_1421": np.asarray(cold_ref_mask_1421, dtype=bool),
        "freq_hz_1420": np.asarray(freq_hz_1420, dtype=np.float64),
        "freq_hz_1421": np.asarray(freq_hz_1421, dtype=np.float64),
        "cold_ref_method": np.str_("savgol_profile_from_cold_ref_sky_l165_b36_hw_corrected_lo_masked"),
        "calibration_mask_method": np.str_("rfi_sigma_clip_and_lax_hardware_response_support_lo_masked"),
        "cold_ref_savgol_window_length": np.int64(SAVGOL["window_length"]),
        "cold_ref_savgol_polyorder": np.int64(SAVGOL["polyorder"]),
        "equipment_artifact_path": np.str_(hardware["path"]),
        "equipment_schema_version": np.str_(hardware["schema_version"]),
        "alpha_db_per_m": np.float64(hardware["alpha_db_per_m"]),
        "sigma_alpha_db_per_m": np.float64(hardware["sigma_alpha_db_per_m"]),
        "unknown_cable_length_m": np.float64(hardware["unknown_cable_length_m"]),
        "unknown_cable_length_sigma_m": np.float64(hardware["unknown_cable_length_sigma_m"]),
        "highest_unclipped_setpoint_dbm": np.float64(hardware["highest_unclipped_setpoint_dbm"]),
        "first_clipped_setpoint_dbm": np.float64(hardware["first_clipped_setpoint_dbm"]),
        "clip_threshold": np.float64(hardware["clip_threshold"]),
        "hardware_response_floor": np.float64(hardware["response_floor"]),
        "hardware_linearity_rmse_db": np.float64(hardware["linearity_rmse_db"]),
        "hardware_att_frac_raw": np.float64(hardware["att_frac_raw"]),
        "hardware_att_frac_used": np.float64(hardware["att_frac_used"]),
        "hardware_mask_1420": np.asarray(hardware_masks[("cold_ref", 1420)], dtype=bool),
        "hardware_mask_1421": np.asarray(hardware_masks[("cold_ref", 1421)], dtype=bool),
        "hardware_response_1420": np.asarray(resp_1420, dtype=np.float64),
        "hardware_response_1421": np.asarray(resp_1421, dtype=np.float64),
        "hardware_response_variant": np.str_(hardware["response_variant"]),
        "hardware_response_support_combined": np.float64(hardware["response_support_combined"]),
        "hardware_response_support_fir": np.float64(hardware["response_support_fir"]),
        "horn_parallel_m": np.float64(HORN_PARALLEL_M),
        "horn_perpendicular_horizon_m": np.float64(HORN_PERP_HORIZON_M),
        "horn_aperture_area_m2": np.float64(HORN_APERTURE_AREA_M2),
        "beam_filling_factor_default": np.float64(BEAM_FILLING_FACTOR_DEFAULT),
        "omega_beam_approx_sr_hi": np.float64(OMEGA_BEAM_APPROX_SR_HI),
        "delta_nu_hz_1420": np.float64(delta_nu_hz_1420),
        "delta_nu_hz_1421": np.float64(delta_nu_hz_1421),
        "tau_s_1420": np.float64(tau_s_1420),
        "tau_s_1421": np.float64(tau_s_1421),
        "omega_beam_approx_sr_1420": np.float64(omega_beam_approx_sr_1420),
        "omega_beam_approx_sr_1421": np.float64(omega_beam_approx_sr_1421),
        "temperature_scale_mode": np.str_("antenna_temperature_yfactor_freq_ref_profile_hw_aware"),
        "hardware_mask_strategy": np.str_("lax_response_support_with_passband_eval_diagnostics"),
        "yfactor_mask_strategy": np.str_("common_hot_cold_intersection"),
        "theory_traceability_version": np.str_("temperature_calibration_trace_v3_lo_mask_loadtime"),
        "stats_exclude_lo_center_bin": np.bool_(True),
        "lo_center_bin_index_1420": np.int64(lo_center_bin_index_1420),
        "lo_center_bin_index_1421": np.int64(lo_center_bin_index_1421),
        "hardware_eval_mask_fraction_1420": np.float64(np.mean(eval_masks[("cold_ref", 1420)])),
        "hardware_eval_mask_fraction_1421": np.float64(np.mean(eval_masks[("cold_ref", 1421)])),
        "hardware_passband_mask_fraction_1420": np.float64(np.mean(passband_masks[("cold_ref", 1420)])),
        "hardware_passband_mask_fraction_1421": np.float64(np.mean(passband_masks[("cold_ref", 1421)])),
    }
    save_npz(TEMPERATURE_ARTIFACT_PATH, artifact)

    summary = pd.DataFrame(
        [
            {
                "LO [MHz]": 1420,
                "T_rx [K]": f"{r1420.T_rx:.2f} +/- {r1420.sigma_T_rx_meas:.2f} (meas) +/- {r1420.sigma_T_rx_loads:.2f} (loads) +/- {r1420.sigma_T_rx_hw:.2f} (hw) +/- {r1420.sigma_T_rx_total:.2f} (total)",
                "T_cold [K]": T_COLD,
                "T_hot [K]": T_HOT,
                "Y": f"{r1420.Y:.4f}",
                "Y_dB": f"{r1420.Y_dB:.3f} dB",
                "P_cold_ref med": f"{np.nanmedian(cold_ref_profile_1420[cold_ref_mask_stats_1420]):.6f}",
                "P_cold_total": f"{p_cold_total_1420:.4f}",
                "HW frac": f"{hardware['systematic_fraction']:.4f}",
            },
            {
                "LO [MHz]": 1421,
                "T_rx [K]": f"{r1421.T_rx:.2f} +/- {r1421.sigma_T_rx_meas:.2f} (meas) +/- {r1421.sigma_T_rx_loads:.2f} (loads) +/- {r1421.sigma_T_rx_hw:.2f} (hw) +/- {r1421.sigma_T_rx_total:.2f} (total)",
                "T_cold [K]": T_COLD,
                "T_hot [K]": T_HOT,
                "Y": f"{r1421.Y:.4f}",
                "Y_dB": f"{r1421.Y_dB:.3f} dB",
                "P_cold_ref med": f"{np.nanmedian(cold_ref_profile_1421[cold_ref_mask_stats_1421]):.6f}",
                "P_cold_total": f"{p_cold_total_1421:.4f}",
                "HW frac": f"{hardware['systematic_fraction']:.4f}",
            },
        ]
    ).set_index("LO [MHz]")
    tables["summary"] = summary

    values.update(
        {
            "hardware": hardware,
            "yfactor_results": yfactor_results,
            "hardware_response": hardware_response,
            "masks": masks,
        }
    )
    return TemperatureCalibrationResult(
        artifact=artifact,
        artifact_path=TEMPERATURE_ARTIFACT_PATH,
        tables=tables,
        figures=figures,
        values=values,
    )
