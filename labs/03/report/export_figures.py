#!/usr/bin/env python3
"""Export report figures that require data loading.

Most figures are saved as vector PDFs directly by the notebooks via
``savefig(fig, name)``.  This script handles only the figures that
need raw chip data loaded outside the notebook context.

Run from the report/ directory:
    /path/to/.venv/bin/python export_figures.py
"""
import sys
import os
from pathlib import Path

_LAB03_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LAB03_DIR))
os.chdir(str(_LAB03_DIR))

import numpy as np
import pickle

from plotters import (
    savefig,
    plot_dc_before_after,
    plot_phase_slope,
)
from utils import (
    PLOT_BAND_GHZ, NOMINAL_B_EW_M, NOMINAL_B_NS_M, BAD_CHANNELS,
    C_LIGHT_MS,
    load_processed_sun_chip_series,
    adaptive_real_dc_correction,
    fringe_frequency_hz,
    geometric_delay_s,
    FringeModelParams,
)


# ============================================================
# Load data
# ============================================================

def _load_data():
    DATA_DIR = _LAB03_DIR / "data" / "sun_calibration" / "chips"
    chip_paths = sorted(DATA_DIR.resolve().glob("sun_calibration_chip_*.npz"))
    chip_data = load_processed_sun_chip_series(chip_paths)

    captures = chip_data.captures
    corr_raw = captures.corr
    unix_mid = captures.unix_mid
    ha_deg = chip_data.ha_deg
    ha_rad = np.deg2rad(ha_deg)
    sun_dec = chip_data.sun_dec_deg
    F_SKY_GHZ = chip_data.f_sky_ghz
    F_SKY_HZ = F_SKY_GHZ * 1e9
    chip_slices = chip_data.chip_slices

    band_mask = (F_SKY_GHZ >= PLOT_BAND_GHZ[0]) & (F_SKY_GHZ <= PLOT_BAND_GHZ[1])
    for bc in BAD_CHANNELS:
        band_mask[bc] = False
    band_center_hz = np.mean(F_SKY_HZ[band_mask])
    band_indices = np.where(band_mask)[0]
    k_mid = band_indices[len(band_indices) // 2]
    dec_rad_mean = np.deg2rad(np.nanmean(sun_dec))

    dc_result = adaptive_real_dc_correction(
        corr_chips=[corr_raw[sl] for sl in chip_slices],
        unix_chips=[unix_mid[sl] for sl in chip_slices],
        ha_rad_chips=[ha_rad[sl] for sl in chip_slices],
        bad_channels=np.array(BAD_CHANNELS, dtype=int),
        b_ew=NOMINAL_B_EW_M, freq_hz=band_center_hz, dec_rad=dec_rad_mean,
        n_periods=3.0, min_window_caps=7, max_window_caps=201,
    )

    return dict(
        corr_raw=corr_raw, unix_mid=unix_mid, ha_deg=ha_deg, ha_rad=ha_rad,
        sun_dec=sun_dec, F_SKY_GHZ=F_SKY_GHZ, F_SKY_HZ=F_SKY_HZ,
        chip_slices=chip_slices, dec_rad_mean=dec_rad_mean,
        band_mask=band_mask, band_center_hz=band_center_hz,
        band_indices=band_indices, k_mid=k_mid,
        corr_dc=dc_result.corr_dc,
        real_offset_chips=dc_result.real_offset_chips,
        window_caps=dc_result.window_caps,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exporting Lab 3 report figures...\n")

    # --- Direct-render figures ---
    print("DC correction (direct render):")
    d = _load_data()
    k = d["k_mid"]

    fig, _ = plot_dc_before_after(
        ha_deg=d["ha_deg"], raw_re=d["corr_raw"][:, k].real,
        dc_re=d["corr_dc"][:, k].real,
        dc_offset=np.vstack(d["real_offset_chips"])[:, k],
        chip_slices=d["chip_slices"],
        channel_idx=None,
        freq_ghz=None,
        height_ratios=(1, 1, 0.5),
    )
    savefig(fig, "fig_dc_before_after.pdf")

    # Pre-compute phase slope data for three HAs
    ha_targets = [-60.0, -30.0, 5.0]
    freqs_ghz = d["F_SKY_HZ"][d["band_mask"]] / 1e9
    phase_list, coeffs_list, ha_list, tau_list = [], [], [], []
    for ha_t in ha_targets:
        idx = np.argmin(np.abs(d["ha_deg"] - ha_t))
        ph = np.angle(d["corr_dc"][idx, :][d["band_mask"]])
        good = np.isfinite(ph)
        if good.sum() < 10:
            continue
        ph_uw = np.unwrap(ph[good])
        f_good = d["F_SKY_HZ"][d["band_mask"]][good]
        coeffs = np.polyfit(f_good, ph_uw, 1)
        tau_fit = coeffs[0] / (2.0 * np.pi) * 1e9
        phase_list.append(ph_uw)
        coeffs_list.append(coeffs)
        ha_list.append(float(d["ha_deg"][idx]))
        tau_list.append(tau_fit)

    fig, _ = plot_phase_slope(
        freqs_ghz_band=freqs_ghz[np.isfinite(np.angle(d["corr_dc"][0, :][d["band_mask"]]))],
        phase_unwrapped=phase_list,
        fit_coeffs=coeffs_list,
        ha_actual_deg=ha_list,
        tau_ns=tau_list,
    )
    savefig(fig, "fig_phase_slope.pdf")
    print()

    # All other figures are saved as vector PDFs directly by the
    # notebooks (02b, 04, 05) via savefig(fig, name).

    print(f"Done. All figures in {_LAB03_DIR / 'report' / 'figures'}/")
