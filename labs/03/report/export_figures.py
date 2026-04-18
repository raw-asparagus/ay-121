#!/usr/bin/env python3
"""Export all report figures as publication-quality PDFs.

Follows the same pattern as ~/projects/ay-128/labs/01/plotters.py:
each figure is a dedicated function, all output as PDF via savefig().

Run from the lab03 directory (or any directory — paths are absolute):
    /path/to/.venv/bin/python report/export_figures.py
"""
import sys
import os
from pathlib import Path

# --- Path setup (must come before any local imports) ---
_LAB03_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_LAB03_DIR))
os.chdir(str(_LAB03_DIR / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    PLOT_BAND_GHZ, NOMINAL_B_EW_M, NOMINAL_B_NS_M, BAD_CHANNELS,
    C_LIGHT_MS, OMEGA_EARTH_RAD_S,
    load_processed_sun_chip_series,
    adaptive_real_dc_correction,
    fringe_frequency_hz, fringe_period_s, geometric_delay_s,
    sky_baseline_lambda,
    point_source_fringes,
    FringeModelParams,
    uniform_disk_visibility_signed,
    # plotting
    TEXTWIDTH_IN,
    LW_FINE, LW_LIGHT,
    NEUTRAL_COLOR,
    TICK_SIZE,
    textwidth_figure, columnwidth_figure,
)
from scipy.special import j1, jn_zeros
from scipy.signal import stft as _stft, welch as _welch

# ============================================================
# Output directory and savefig
# ============================================================

_FIGURES_DIR = _LAB03_DIR / "report" / "figures"

def savefig(fig, name):
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)
    plt.close(fig)
    print(f"  {name}")

CHIP_COLORS = [f"C{i}" for i in range(10)]


# ============================================================
# Data loading (lazy, shared across figures)
# ============================================================

_data = {}

def _load_data():
    """Load and cache all shared data."""
    if _data:
        return _data

    import pickle

    DATA_DIR = _LAB03_DIR / ".." / ".." / "data" / "lab03" / "sun_calibration" / "chips"
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
    N_chips = len(chip_slices)
    dec_rad_mean = np.deg2rad(np.nanmean(sun_dec))

    band_mask = (F_SKY_GHZ >= PLOT_BAND_GHZ[0]) & (F_SKY_GHZ <= PLOT_BAND_GHZ[1])
    for bc in BAD_CHANNELS:
        band_mask[bc] = False
    band_center_hz = np.mean(F_SKY_HZ[band_mask])
    band_indices = np.where(band_mask)[0]
    k_mid = band_indices[len(band_indices) // 2]

    # DC correction
    dc_result = adaptive_real_dc_correction(
        corr_chips=[corr_raw[sl] for sl in chip_slices],
        unix_chips=[unix_mid[sl] for sl in chip_slices],
        ha_rad_chips=[ha_rad[sl] for sl in chip_slices],
        bad_channels=np.array(BAD_CHANNELS, dtype=int),
        b_ew=NOMINAL_B_EW_M, freq_hz=band_center_hz, dec_rad=dec_rad_mean,
        n_periods=3.0, min_window_caps=7, max_window_caps=201,
    )

    # Adopted baseline
    bl_path = _LAB03_DIR / "notebooks" / "_baseline_adopted.pkl"
    with open(bl_path, "rb") as f:
        bl = pickle.load(f)

    _data.update(dict(
        corr_raw=corr_raw, unix_mid=unix_mid, ha_deg=ha_deg, ha_rad=ha_rad,
        sun_dec=sun_dec, F_SKY_GHZ=F_SKY_GHZ, F_SKY_HZ=F_SKY_HZ,
        chip_slices=chip_slices, N_chips=N_chips,
        dec_rad_mean=dec_rad_mean, band_mask=band_mask,
        band_center_hz=band_center_hz, band_indices=band_indices, k_mid=k_mid,
        corr_dc=dc_result.corr_dc,
        real_offset_chips=dc_result.real_offset_chips,
        window_caps=dc_result.window_caps,
        B_EW=bl["b_ew_m"], B_NS=bl["b_ns_m"],
    ))
    return _data


# ============================================================
# Figure 1: Synthetic point-source fringe
# ============================================================

def fig_fringe_synthetic():
    ha = np.linspace(np.deg2rad(-80), np.deg2rad(25), 5000)
    params = FringeModelParams(
        b_ew=NOMINAL_B_EW_M, b_ns=NOMINAL_B_NS_M,
        freq_hz=10.45e9, dec_rad=np.deg2rad(-0.08),
        amplitude=1.0, phase_offset=0.0,
    )
    fringe = point_source_fringes(ha, params)

    fig, ax = textwidth_figure(3)
    ax.plot(np.rad2deg(ha), fringe, lw=0.3, color="C0")
    ax.set_xlabel("Hour angle [deg]")
    ax.set_ylabel(r"$F(h)$")
    ax.set_xlim(-80, 25)
    savefig(fig, "fig_fringe_synthetic.pdf")


# ============================================================
# Figure 2: Bessel envelope (jinc) with zeros
# ============================================================

def fig_bessel_envelope_theory():
    x = np.linspace(0, 16, 2000)
    jinc = np.where(x == 0, 1.0, 2 * j1(2 * np.pi * x) / (2 * np.pi * x))

    fig, ax = columnwidth_figure(4)
    ax.plot(x, jinc, "C0", lw=1.2, label=r"$2J_1(2\pi x)/(2\pi x)$")
    ax.plot(x, np.abs(jinc), "C0", lw=0.6, alpha=0.4, label=r"$|2J_1(2\pi x)/(2\pi x)|$")
    zeros = jn_zeros(1, 4) / (2 * np.pi)
    for i, z in enumerate(zeros):
        ax.axvline(z, color="C3", ls="--", lw=0.6, alpha=0.7)
        ax.text(z, 0.85 - 0.15 * i, rf"$j_{{1,{i+1}}}$", fontsize=7, ha="center", color="C3")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel(r"$|u|\,R$")
    ax.set_ylabel(r"$V/V(0)$")
    ax.legend(fontsize=7, loc="upper right")
    savefig(fig, "fig_bessel_envelope_theory.pdf")


# ============================================================
# Figure 3: Raw visibility (4-panel)
# ============================================================

def fig_raw_visibility():
    d = _load_data()
    vis = d["corr_raw"][:, d["k_mid"]]
    ha_deg = d["ha_deg"]
    chip_slices = d["chip_slices"]

    fig, axes = plt.subplots(4, 1, figsize=(TEXTWIDTH_IN, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.06})
    panels = [vis.real, vis.imag, np.abs(vis), np.angle(vis, deg=True)]
    labels = [r"Re$(V)$", r"Im$(V)$", r"$|V|$", r"arg$(V)$ [deg]"]
    for ax, y, lbl in zip(axes, panels, labels):
        for ci, sl in enumerate(chip_slices):
            ax.scatter(ha_deg[sl], y[sl], s=0.1, alpha=0.3,
                       color=CHIP_COLORS[ci], rasterized=True)
        ax.set_ylabel(lbl)
        if "Re" in lbl or "Im" in lbl:
            ax.axhline(0, color=NEUTRAL_COLOR, lw=LW_LIGHT, ls="--")
    axes[-1].set_xlabel("Hour angle [deg]")
    axes[0].set_title(
        f"Raw visibility (ch {d['k_mid']}, {d['F_SKY_GHZ'][d['k_mid']]:.3f} GHz)",
        fontsize=TICK_SIZE)
    fig.tight_layout()
    savefig(fig, "fig_raw_visibility.pdf")


# ============================================================
# Figure 4: Amplitude spectrum
# ============================================================

def fig_amplitude_spectrum():
    d = _load_data()
    fig, ax = textwidth_figure(3)
    for ci, sl in enumerate(d["chip_slices"]):
        amp_mean = np.nanmean(np.abs(d["corr_raw"][sl]), axis=0)
        ax.plot(d["F_SKY_GHZ"], amp_mean, lw=0.5, alpha=0.7,
                color=CHIP_COLORS[ci], label=f"Chip {ci}")
    ax.axvspan(PLOT_BAND_GHZ[0], PLOT_BAND_GHZ[1], alpha=0.1, color="C1",
               label="Analysis band")
    for bc in BAD_CHANNELS:
        ax.axvline(d["F_SKY_GHZ"][bc], color="red", lw=0.4, ls=":", alpha=0.5)
    ax.set_xlabel("Sky frequency [GHz]")
    ax.set_ylabel(r"$\langle|V|\rangle$")
    ax.legend(fontsize=7, ncol=4)
    savefig(fig, "fig_amplitude_spectrum.pdf")


# ============================================================
# Figure 5: Before/after DC correction
# ============================================================

def fig_dc_before_after():
    d = _load_data()
    k = d["k_mid"]
    raw_re = d["corr_raw"][:, k].real
    dc_re = d["corr_dc"][:, k].real
    offset = np.vstack(d["real_offset_chips"])[:, k]

    fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH_IN, 3.2), sharex=True,
                             gridspec_kw={"hspace": 0.06, "height_ratios": [1, 1, 0.5]})
    for ci, sl in enumerate(d["chip_slices"]):
        axes[0].scatter(d["ha_deg"][sl], raw_re[sl], s=0.1, alpha=0.3,
                        color=CHIP_COLORS[ci], rasterized=True)
        axes[1].scatter(d["ha_deg"][sl], dc_re[sl], s=0.1, alpha=0.3,
                        color=CHIP_COLORS[ci], rasterized=True)
        axes[2].plot(d["ha_deg"][sl], offset[sl], lw=LW_FINE,
                     color=CHIP_COLORS[ci])
    for ax in axes[:2]:
        ax.axhline(0, color=NEUTRAL_COLOR, lw=LW_LIGHT, ls="--")
    axes[0].set_ylabel(r"Re$(V)$ raw")
    axes[1].set_ylabel(r"Re$(V_{\rm dc})$")
    axes[2].set_ylabel("Diff")
    axes[-1].set_xlabel("Hour angle [deg]")
    fig.tight_layout()
    savefig(fig, "fig_dc_before_after.pdf")


# ============================================================
# Figure 6: Window adaptation
# ============================================================

def fig_window_adaptation():
    d = _load_data()
    period = fringe_period_s(d["ha_rad"], d["dec_rad_mean"],
                             NOMINAL_B_EW_M, 0.0, d["band_center_hz"])
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN * 0.48, 3.0),
                             sharex=True, gridspec_kw={"hspace": 0.08})
    for ci, sl in enumerate(d["chip_slices"]):
        axes[0].scatter(d["ha_deg"][sl], period[sl], s=0.2,
                        color=CHIP_COLORS[ci], alpha=0.5, rasterized=True)
        axes[1].scatter(d["ha_deg"][sl], d["window_caps"][sl], s=0.2,
                        color=CHIP_COLORS[ci], alpha=0.5, rasterized=True)
    axes[0].set_ylabel("Fringe period [s]")
    axes[1].set_ylabel("Window [captures]")
    axes[1].set_xlabel("Hour angle [deg]")
    axes[1].axhline(7, color=NEUTRAL_COLOR, lw=0.5, ls=":")
    axes[1].axhline(201, color=NEUTRAL_COLOR, lw=0.5, ls="--")
    fig.tight_layout()
    savefig(fig, "fig_window_adaptation.pdf")


# ============================================================
# Figure 7: STFT sanity check
# ============================================================

def fig_stft_sanity():
    d = _load_data()
    vis_ch = d["corr_dc"][:, d["k_mid"]].real
    sl0 = d["chip_slices"][0]
    vis0 = vis_ch[sl0]
    ha0 = d["ha_rad"][sl0]
    t0 = d["unix_mid"][sl0]
    valid0 = np.isfinite(vis0)
    dt_med = np.median(np.diff(t0[valid0]))
    t_uni = np.arange(t0[valid0][0], t0[valid0][-1], dt_med)
    vis_uni = np.interp(t_uni, t0[valid0], vis0[valid0])
    ha_uni = np.interp(t_uni, t0[valid0], ha0[valid0])

    f_stft, t_stft, Zxx = _stft(vis_uni, fs=1.0 / dt_med,
                                 nperseg=128, noverlap=112, window="hann")
    ha_stft = np.interp(t_stft + t_uni[0], t_uni, ha_uni)

    ha_pred = np.linspace(ha0[valid0].min(), ha0[valid0].max(), 500)
    ff_pred = np.abs(fringe_frequency_hz(ha_pred, d["dec_rad_mean"],
                                          NOMINAL_B_EW_M, NOMINAL_B_NS_M,
                                          d["band_center_hz"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.8),
                                    gridspec_kw={"width_ratios": [2, 1], "wspace": 0.3})
    ax1.pcolormesh(np.rad2deg(ha_stft), f_stft,
                   10 * np.log10(np.abs(Zxx) ** 2 + 1e-30),
                   cmap="inferno", shading="auto")
    ax1.plot(np.rad2deg(ha_pred), ff_pred, "w--", lw=1.0,
             label=r"Predicted $|f_f(h)|$")
    ax1.set_ylim(0, 0.08)
    ax1.set_xlabel("Hour angle [deg]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.legend(fontsize=7, loc="upper right")

    f_psd, Pxx = _welch(vis_uni, fs=1.0 / dt_med, nperseg=256, window="hann")
    ff_transit = abs(fringe_frequency_hz(0.0, d["dec_rad_mean"],
                                          NOMINAL_B_EW_M, NOMINAL_B_NS_M,
                                          d["band_center_hz"]))
    ff_min = abs(fringe_frequency_hz(ha0[valid0].max(), d["dec_rad_mean"],
                                      NOMINAL_B_EW_M, NOMINAL_B_NS_M,
                                      d["band_center_hz"]))
    ax2.semilogy(f_psd, Pxx, "C0", lw=0.8)
    ax2.axvspan(ff_min, ff_transit, alpha=0.15, color="C1",
                label=r"Expected $f_f$ range")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("PSD")
    ax2.set_xlim(0, 0.08)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig_stft_sanity.pdf")


# ============================================================
# Figure 8: Phase slope
# ============================================================

def fig_phase_slope():
    d = _load_data()
    ha_test = [-60.0, -30.0, 5.0]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.5), sharey=True)
    for ax, ha_target in zip(axes, ha_test):
        idx = np.argmin(np.abs(d["ha_deg"] - ha_target))
        vis_spec = d["corr_dc"][idx, :]
        ph = np.angle(vis_spec[d["band_mask"]])
        freqs = d["F_SKY_HZ"][d["band_mask"]]
        good = np.isfinite(ph)
        if good.sum() < 10:
            continue
        ph_uw = np.unwrap(ph[good])
        f_g = freqs[good]
        coeffs = np.polyfit(f_g, ph_uw, 1)
        tau_ns = coeffs[0] / (2.0 * np.pi) * 1e9
        ax.scatter(f_g / 1e9, ph_uw, s=0.5, alpha=0.5, color="C0", rasterized=True)
        ax.plot(f_g / 1e9, np.polyval(coeffs, f_g), "C1", lw=1.0,
                label=rf"$\tau$ = {tau_ns:.1f} ns")
        ax.set_xlabel("Freq [GHz]")
        ax.set_title(rf"HA = {d['ha_deg'][idx]:.0f}$^\circ$", fontsize=9)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Unwrapped phase [rad]")
    fig.tight_layout()
    savefig(fig, "fig_phase_slope.pdf")


# ============================================================
# Figures 9-12: Baseline determination (extracted from nb04)
# ============================================================

def _extract_notebook_figures(nb_path, cell_fig_map):
    """Extract the first PNG output from each listed cell and save as PDF."""
    import json
    import base64
    from io import BytesIO
    from PIL import Image

    with open(nb_path) as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        cid = cell.get("id", "")
        if cid not in cell_fig_map:
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "display_data":
                img_b64 = out.get("data", {}).get("image/png")
                if img_b64:
                    img = Image.open(BytesIO(base64.b64decode(img_b64)))
                    # Convert RGBA to RGB if needed
                    if img.mode == "RGBA":
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    outpath = _FIGURES_DIR / cell_fig_map[cid]
                    img.save(outpath, "PDF", resolution=300)
                    print(f"  {cell_fig_map[cid]}  (from cell {cid})")
                    break


def fig_data_reduction_from_nb02b():
    nb02b = _LAB03_DIR / "notebooks" / "02b_corrected_data_inspection.ipynb"
    _extract_notebook_figures(nb02b, {
        "waterfall_combined": "fig_waterfall.pdf",
    })


def fig_baseline_from_nb04():
    nb04 = _LAB03_DIR / "notebooks" / "04_baseline_determination.ipynb"
    _extract_notebook_figures(nb04, {
        "16zdvbp5u3u": "fig_fft_per_channel.pdf",
        "24b71d7aa2a9cd48": "fig_lag_delay_vs_ha.pdf",
        "4620ef3329c789fa": "fig_stft_baseline_fit.pdf",
        "9f878ac7f3b8b976": "fig_brute_1d.pdf",
        "98e3746baf56b065": "fig_brute_2d.pdf",
        "grid_windowed_code": "fig_brute_windowed.pdf",
    })


# ============================================================
# Figures 13-18: Solar analysis (extracted from nb05)
# ============================================================

def fig_solar_from_nb05():
    nb05 = _LAB03_DIR / "notebooks" / "05_solar_analysis.ipynb"
    _extract_notebook_figures(nb05, {
        "254b54bb": "fig_bessel_extrema.pdf",
        "d15ec7cd": "fig_diameter_vs_freq.pdf",
        "jinc_fit_free_R": "fig_jinc_fit.pdf",
        "e5ddff70": "fig_lb_fit_vs_data.pdf",
        "results_plot": "fig_fit_params_vs_freq.pdf",
        "sunspot_lb_results": "fig_eps_f_correlation.pdf",
    })


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Exporting Lab 3 report figures...")
    print()

    print("Theory figures (direct render):")
    fig_fringe_synthetic()
    fig_bessel_envelope_theory()
    print()

    print("Data reduction figures (direct render):")
    fig_raw_visibility()
    fig_amplitude_spectrum()
    fig_dc_before_after()
    fig_window_adaptation()
    fig_stft_sanity()
    fig_phase_slope()
    print()

    print("Data reduction figures (from nb02b):")
    fig_data_reduction_from_nb02b()
    print()

    print("Baseline determination figures (from nb04):")
    fig_baseline_from_nb04()
    print()

    print("Solar analysis figures (from nb05):")
    fig_solar_from_nb05()
    print()

    print(f"Done. All figures in {_FIGURES_DIR}/")
