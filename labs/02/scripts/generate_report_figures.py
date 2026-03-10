#!/usr/bin/env python3
"""
Generate missing report figures for AY121 Lab 2 HI analysis.

Figures produced (all saved to labs/02/report/figures/):
  ratio_profile.pdf     - Dual-LO ratio r(ν) = R-1 in LSR velocity frame
  tline_calibrated.pdf  - Calibrated T_line(K) vs LSR velocity (standard field)
  pointing_track.pdf    - Alt/Az vs LST for (l,b)=(120°,0°)
  cygnus_x_spectrum.pdf - Calibrated T_line(K) vs LSR velocity (Cygnus-X field)

Run from repo root:
  .venv/bin/python labs/02/scripts/generate_report_figures.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # ay-121/
sys.path.insert(0, str(REPO_ROOT))

NB_DIR = REPO_ROOT / 'labs/02'
CACHE_DIR = NB_DIR / 'cache'
DATA_ROOT = REPO_ROOT / 'data/lab02'
FIG_DIR = NB_DIR / 'report/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

from ugradiolab import Spectrum  # noqa: E402
import ugradio.doppler  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HI_REST_FREQ_HZ = 1_420_405_751.768
C_LIGHT_KMS = 2.99792458e5
SMOOTH_NCHAN = 31  # 31-channel mean → 0.51 km/s kernel

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 1.2,
})

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def velocity_axis_kms(freqs_hz: np.ndarray) -> np.ndarray:
    """Radio-definition Doppler velocity for given frequency array."""
    return C_LIGHT_KMS * (HI_REST_FREQ_HZ - np.asarray(freqs_hz, float)) / HI_REST_FREQ_HZ


def lsr_correction_kms(spectrum: 'Spectrum') -> float:
    """Return LSR correction Δv_LSR in km/s for a Spectrum."""
    dv_ms = ugradio.doppler.get_projected_velocity(
        spectrum.jd, spectrum.alt, spectrum.az,
        spectrum.obs_lat, spectrum.obs_lon
    )
    return float(dv_ms) / 1000.0


def smooth_mean(arr: np.ndarray, nchan: int) -> np.ndarray:
    """Running mean smoothing, preserving NaN positions."""
    if nchan <= 1:
        return arr.copy()
    kernel = np.ones(nchan) / nchan
    finite = np.isfinite(arr)
    filled = arr.copy()
    if finite.sum() > 1:
        idx = np.arange(len(arr))
        filled[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    out = np.convolve(filled, kernel, mode='same')
    out[~finite] = np.nan
    return out


def remove_dc_bin(psd: np.ndarray, center_freq_hz: float, freqs_hz: np.ndarray) -> np.ndarray:
    """Zero the bin closest to center frequency (LO leakage)."""
    idx = int(np.argmin(np.abs(freqs_hz - center_freq_hz)))
    psd = psd.copy()
    psd[idx] = np.nan
    return psd


def interp_mono(xsrc, ysrc, xnew, fill_value=np.nan):
    """Linear interpolation with fill_value outside bounds."""
    xsrc = np.asarray(xsrc, float)
    ysrc = np.asarray(ysrc, float)
    xnew = np.asarray(xnew, float)
    out = np.interp(xnew, xsrc, ysrc, left=np.nan, right=np.nan)
    return out


def sigma_clip_mask(arr, nsigma=5.0):
    """Return boolean mask True = keep, with simple iterative sigma clipping."""
    mask = np.isfinite(arr)
    for _ in range(3):
        mu = np.nanmedian(arr[mask])
        sg = np.nanstd(arr[mask])
        if sg == 0:
            break
        mask &= np.abs(arr - mu) < nsigma * sg
    return mask


def load_spectrum_pair(spectra_dir: Path) -> dict:
    """Load LO1420 and LO1421 combined spectra from directory."""
    pair = {}
    for f in sorted(spectra_dir.glob('*.npz')):
        sp = Spectrum.load(f)
        lo_mhz = int(round(sp.center_freq / 1e6))
        pair[lo_mhz] = sp
    if 1420 not in pair or 1421 not in pair:
        raise ValueError(f'Expected LO 1420+1421 in {spectra_dir}, got {list(pair.keys())}')
    return pair


def psd_clean(spectrum: 'Spectrum') -> np.ndarray:
    """Return PSD array with DC bin zeroed and RFI clipped."""
    psd = np.array(spectrum.psd, dtype=float)
    freqs = np.array(spectrum.freqs, dtype=float)
    psd = remove_dc_bin(psd, spectrum.center_freq, freqs)
    clip_mask = sigma_clip_mask(psd)
    psd[~clip_mask] = np.nan
    return psd


def compute_ratio_profile(pair: dict) -> dict:
    """Compute frequency-switched ratio profiles R-1 and 1/R-1."""
    sp0 = pair[1420]
    sp1 = pair[1421]
    p0 = psd_clean(sp0)
    p1 = psd_clean(sp1)

    # Good channel mask (both finite and positive)
    good = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0)
    p0 = np.where(good, p0, np.nan)
    p1 = np.where(good, p1, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        R = p0 / p1
        Rinv = p1 / p0

    y_R = R - 1.0
    y_inv = Rinv - 1.0

    y_R_sm = smooth_mean(np.where(good, y_R, np.nan), SMOOTH_NCHAN)
    y_inv_sm = smooth_mean(np.where(good, y_inv, np.nan), SMOOTH_NCHAN)

    dv0 = lsr_correction_kms(sp0)
    dv1 = lsr_correction_kms(sp1)
    v0 = velocity_axis_kms(sp0.freqs) + dv0
    v1 = velocity_axis_kms(sp1.freqs) + dv1

    return dict(v0=v0, v1=v1,
                y_R=y_R, y_inv=y_inv,
                y_R_sm=y_R_sm, y_inv_sm=y_inv_sm,
                good=good)


def compute_tline_profile(pair: dict, cal: dict) -> dict:
    """Compute calibrated T_line profiles (Path B)."""
    sp0 = pair[1420]
    sp1 = pair[1421]
    f0 = np.array(sp0.freqs, dtype=float)
    f1 = np.array(sp1.freqs, dtype=float)
    p0 = psd_clean(sp0)
    p1 = psd_clean(sp1)

    t_rx_1420 = float(cal['t_rx_1420'])
    t_rx_1421 = float(cal['t_rx_1421'])
    T_cold = float(cal['t_cold'])

    # Cold reference profiles interpolated onto spectrum frequency axis
    fsrc_1420 = np.asarray(cal['freq_hz_1420'], float)
    fsrc_1421 = np.asarray(cal['freq_hz_1421'], float)
    c0 = interp_mono(fsrc_1420, np.asarray(cal['cold_ref_profile_1420'], float), f0)
    c1 = interp_mono(fsrc_1421, np.asarray(cal['cold_ref_profile_1421'], float), f1)

    # Hardware response (FIR × sum correction)
    hw_resp_1420 = np.asarray(cal['hardware_response_1420'], float)
    hw_resp_1421 = np.asarray(cal['hardware_response_1421'], float)
    hw_f_1420 = np.asarray(cal['freq_hz_1420'], float)
    hw_f_1421 = np.asarray(cal['freq_hz_1421'], float)
    resp0 = interp_mono(hw_f_1420, hw_resp_1420, f0)
    resp1 = interp_mono(hw_f_1421, hw_resp_1421, f1)

    hw_floor = float(cal.get('hardware_response_floor', 0.01))
    resp0 = np.where(np.isfinite(resp0) & (resp0 > hw_floor), resp0, np.nan)
    resp1 = np.where(np.isfinite(resp1) & (resp1 > hw_floor), resp1, np.nan)

    # Good mask
    good = (np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0) &
            np.isfinite(c0) & np.isfinite(c1) & (c0 > 0) & (c1 > 0) &
            np.isfinite(resp0) & np.isfinite(resp1))

    # Hardware-corrected PSDs
    p0h = np.where(good, p0 / resp0, np.nan)
    p1h = np.where(good, p1 / resp1, np.nan)
    c0h = np.where(good, c0 / resp0, np.nan)
    c1h = np.where(good, c1 / resp1, np.nan)

    # T_sys reference
    Tsys0 = p0h * (T_cold + t_rx_1420) / c0h
    Tsys1 = p1h * (T_cold + t_rx_1421) / c1h

    with np.errstate(divide='ignore', invalid='ignore'):
        R = p0h / p1h
    y_R = R - 1.0

    Tline_R = y_R * Tsys1

    # Smooth
    Tline_sm = smooth_mean(np.where(good, Tline_R, np.nan), SMOOTH_NCHAN)

    dv0 = lsr_correction_kms(sp0)
    v0 = velocity_axis_kms(f0) + dv0

    return dict(v0=v0, freqs=f0,
                Tline=Tline_R, Tline_sm=Tline_sm,
                good=good)


# ---------------------------------------------------------------------------
# Figure 1: ratio_profile.pdf
# ---------------------------------------------------------------------------

def plot_ratio_profile(std_ratio: dict, cyg_ratio: dict, outpath: Path):
    """4-panel figure: R-1 and 1/R-1 for standard and Cygnus-X fields."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex='col')
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    vmin, vmax = -200, 200

    configs = [
        # (ax, data_dict, x_key, y_key, title, ylabel)
        (axes[0, 0], std_ratio, 'v0', 'y_R_sm',
         r'Standard Field $(l,b)=(120°,0°)$: $R-1$', r'$R - 1$'),
        (axes[0, 1], std_ratio, 'v1', 'y_inv_sm',
         r'Standard Field: $1/R - 1$', r'$1/R - 1$'),
        (axes[1, 0], cyg_ratio, 'v0', 'y_R_sm',
         r'Cygnus-X Field: $R-1$', r'$R - 1$'),
        (axes[1, 1], cyg_ratio, 'v1', 'y_inv_sm',
         r'Cygnus-X Field: $1/R - 1$', r'$1/R - 1$'),
    ]

    for ax, data, xk, yk, title, ylabel in configs:
        v = data[xk]
        y_raw = data[yk.replace('_sm', '')]
        y_sm = data[yk]
        sel = (v > vmin) & (v < vmax)
        ax.plot(v[sel], y_raw[sel], color='C0', lw=0.3, alpha=0.35, label='raw')
        ax.plot(v[sel], y_sm[sel], color='C0', lw=1.1, alpha=0.9,
                label=f'smoothed (n={SMOOTH_NCHAN})')
        ax.axhline(0, color='gray', lw=0.7, ls='--')
        ax.set_xlim(vmin, vmax)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('$v_\\mathrm{LSR}$ [km/s]')
        ax.legend(fontsize=7, loc='upper left')
        ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))

    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {outpath}')


# ---------------------------------------------------------------------------
# Figure 2: tline_calibrated.pdf
# ---------------------------------------------------------------------------

def plot_tline_calibrated(tline_data: dict, outpath: Path):
    """Calibrated T_line vs LSR velocity with secondary frequency axis."""
    v = tline_data['v0']
    freqs = tline_data['freqs']
    T_raw = tline_data['Tline']
    T_sm = tline_data['Tline_sm']

    vmin, vmax = -200, 80

    fig, ax = plt.subplots(figsize=(8, 4))

    sel = (v > vmin) & (v < vmax)
    ax.plot(v[sel], T_raw[sel], color='C0', lw=0.3, alpha=0.3, label='raw')
    ax.fill_between(v[sel], 0, T_sm[sel],
                    where=T_sm[sel] > 0, alpha=0.15, color='C0')
    ax.plot(v[sel], T_sm[sel], color='C0', lw=1.3,
            label=f'smoothed (n={SMOOTH_NCHAN}, $\\approx$0.51 km/s)')
    ax.axhline(0, color='gray', lw=0.7, ls='--')

    ax.set_xlim(vmin, vmax)
    ax.set_xlabel('$v_\\mathrm{LSR}$ [km/s]')
    ax.set_ylabel('$T_\\mathrm{line}$ [K]')
    ax.set_title(r'Calibrated HI Spectrum: Standard Field $(l,b)=(120°, 0°)$')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

    # Secondary x-axis: RF frequency in MHz
    ax2 = ax.twiny()
    ax2.set_xlim(vmin, vmax)
    # Velocity → frequency: ν = ν0 * (1 - v/c)
    v_ticks = np.array([-200, -150, -100, -50, 0, 50])
    v_ticks_in = v_ticks[(v_ticks >= vmin) & (v_ticks <= vmax)]
    f_ticks = HI_REST_FREQ_HZ * (1 - v_ticks_in / C_LIGHT_KMS) / 1e6
    ax2.set_xticks(v_ticks_in)
    ax2.set_xticklabels([f'{f:.2f}' for f in f_ticks], fontsize=8)
    ax2.set_xlabel('Frequency [MHz]', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {outpath}')


# ---------------------------------------------------------------------------
# Figure 3: pointing_track.pdf
# ---------------------------------------------------------------------------

def galactic_to_equatorial(l_deg: float, b_deg: float) -> tuple:
    """Convert Galactic (l,b) to equatorial (RA, Dec) J2000 in degrees."""
    l = np.radians(l_deg)
    b = np.radians(b_deg)

    # IAU Galactic ↔ J2000 rotation matrix (standard)
    # North Galactic Pole at J2000: RA=192.85948°, Dec=27.12825°
    # Galactic longitude of NCP: l=122.93192°
    RA_NGP = np.radians(192.85948)
    DEC_NGP = np.radians(27.12825)
    L_NCP = np.radians(122.93192)

    sin_d = (np.cos(b) * np.cos(DEC_NGP) * np.cos(l - L_NCP)
             + np.sin(b) * np.sin(DEC_NGP))
    dec = np.arcsin(np.clip(sin_d, -1, 1))

    cos_d = np.cos(dec)
    if abs(cos_d) < 1e-10:
        ra = 0.0
    else:
        cos_ra = (np.cos(b) * np.sin(l - L_NCP)) / cos_d
        sin_ra = (-np.cos(b) * np.sin(DEC_NGP) * np.cos(l - L_NCP)
                  + np.sin(b) * np.cos(DEC_NGP)) / cos_d
        ra = np.arctan2(sin_ra, cos_ra) + RA_NGP
    ra = ra % (2 * np.pi)
    return np.degrees(ra), np.degrees(dec)


def equatorial_to_altaz(ra_deg: float, dec_deg: float,
                        lst_h: float, lat_deg: float) -> tuple:
    """Convert equatorial (RA, Dec) to topocentric (Alt, Az) in degrees."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    lat = np.radians(lat_deg)
    lst = np.radians(lst_h * 15.0)  # hours → radians
    H = lst - ra  # hour angle

    sin_alt = (np.sin(dec) * np.sin(lat)
               + np.cos(dec) * np.cos(lat) * np.cos(H))
    alt = np.arcsin(np.clip(sin_alt, -1, 1))

    cos_alt = np.cos(alt)
    if np.isscalar(H):
        if abs(cos_alt) < 1e-10:
            az = 0.0
        else:
            cos_az = (np.sin(dec) - np.sin(alt) * np.sin(lat)) / (cos_alt * np.cos(lat))
            sin_az = -np.cos(dec) * np.sin(H) / cos_alt
            az = np.arctan2(sin_az, cos_az)
            az = az % (2 * np.pi)
    else:
        az = np.where(
            np.abs(cos_alt) < 1e-10,
            0.0,
            np.arctan2(
                -np.cos(dec) * np.sin(H) / np.where(np.abs(cos_alt) < 1e-10, 1, cos_alt),
                (np.sin(dec) - np.sin(alt) * np.sin(lat)) /
                (np.where(np.abs(cos_alt) < 1e-10, 1, cos_alt) * np.cos(lat))
            ) % (2 * np.pi)
        )
    return np.degrees(alt), np.degrees(az)


def plot_pointing_track(outpath: Path):
    """Alt/Az vs LST for (l,b)=(120°,0°) at Berkeley."""
    lat_deg = 37.873199
    l_deg, b_deg = 120.0, 0.0
    ra_deg, dec_deg = galactic_to_equatorial(l_deg, b_deg)
    print(f'  (l,b)=({l_deg},{b_deg}) → (RA,Dec) = ({ra_deg:.2f}°, {dec_deg:.2f}°)')

    lst_h = np.linspace(0, 24, 1000)
    alt, az = equatorial_to_altaz(ra_deg, dec_deg, lst_h, lat_deg)

    # Observation window (horizon limit = 15° Alt)
    observable = alt > 15.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0.15)

    # Altitude
    ax1.plot(lst_h, alt, 'C0', lw=1.3, label=r'Altitude $(l,b)=(120°,0°)$')
    ax1.fill_between(lst_h, 15, alt, where=observable, alpha=0.18, color='C0',
                     label='Observable (Alt > 15°)')
    ax1.axhline(15, color='gray', lw=0.8, ls='--', label='Horizon limit (15°)')
    ax1.axhline(90, color='gray', lw=0.5, ls=':')
    ax1.set_ylim(-10, 95)
    ax1.set_ylabel('Altitude [deg]')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(30))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax1.set_title(r'Pointing Track: $(l,b)=(120°,\,0°)$ from Campbell Hall, Berkeley '
                  r'($\phi=37.87°$N)', fontsize=10)

    # Mark observation epochs
    obs_jd = 2461105.386  # from standard spectrum metadata
    # LST at obs: compute from JD
    obs_lst_fraction = 0.007181977903727467  # in hours (from spectrum)
    ax1.axvline(obs_lst_fraction, color='red', lw=1.5, ls='-', alpha=0.7,
                label=f'Observation LST = {obs_lst_fraction:.2f} h')
    ax1.legend(fontsize=8, loc='upper right')

    # Azimuth
    ax2.plot(lst_h[observable], az[observable], 'C1', lw=1.3)
    ax2.plot(lst_h[~observable], az[~observable], 'C1', lw=0.5, alpha=0.3)
    ax2.set_ylabel('Azimuth [deg]')
    ax2.set_xlabel('Local Sidereal Time [h]')
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(90))
    ax2.yaxis.set_minor_locator(mticker.MultipleLocator(30))
    ax2.axvline(obs_lst_fraction, color='red', lw=1.5, ls='-', alpha=0.7,
                label=f'Obs. LST = {obs_lst_fraction:.2f} h')
    ax2.legend(fontsize=8)

    ax2.set_xlim(0, 24)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {outpath}')


# ---------------------------------------------------------------------------
# Figure 4: cygnus_x_spectrum.pdf
# ---------------------------------------------------------------------------

def plot_cygnus_x_spectrum(tline_data: dict, outpath: Path):
    """Calibrated T_line vs LSR velocity for Cygnus-X."""
    v = tline_data['v0']
    T_raw = tline_data['Tline']
    T_sm = tline_data['Tline_sm']

    vmin, vmax = -200, 80

    fig, ax = plt.subplots(figsize=(8, 4))

    sel = (v > vmin) & (v < vmax)
    ax.plot(v[sel], T_raw[sel], color='C1', lw=0.3, alpha=0.3, label='raw')
    ax.fill_between(v[sel], 0, T_sm[sel],
                    where=T_sm[sel] > 0, alpha=0.15, color='C1')
    ax.plot(v[sel], T_sm[sel], color='C1', lw=1.3,
            label=f'smoothed (n={SMOOTH_NCHAN}, $\\approx$0.51 km/s)')
    ax.axhline(0, color='gray', lw=0.7, ls='--')

    ax.set_xlim(vmin, vmax)
    ax.set_xlabel('$v_\\mathrm{LSR}$ [km/s]')
    ax.set_ylabel('$T_\\mathrm{line}$ [K]')
    ax.set_title(r'Calibrated HI Spectrum: Cygnus-X Cross-Validation Field')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))

    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(vmin, vmax)
    v_ticks = np.array([-200, -150, -100, -50, 0, 50])
    v_ticks_in = v_ticks[(v_ticks >= vmin) & (v_ticks <= vmax)]
    f_ticks = HI_REST_FREQ_HZ * (1 - v_ticks_in / C_LIGHT_KMS) / 1e6
    ax2.set_xticks(v_ticks_in)
    ax2.set_xticklabels([f'{f:.2f}' for f in f_ticks], fontsize=8)
    ax2.set_xlabel('Frequency [MHz]', fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {outpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading calibration data...')
    cal_path = CACHE_DIR / 'calibration_results_v2.npz'
    with np.load(cal_path, allow_pickle=False) as _f:
        cal = {k: _f[k] for k in _f.files}
    print(f'  T_rx(1420) = {float(cal["t_rx_1420"]):.2f} K')
    print(f'  T_rx(1421) = {float(cal["t_rx_1421"]):.2f} K')

    print('\nLoading standard field spectra...')
    std_pair = load_spectrum_pair(DATA_ROOT / 'standard_combined_spectra')
    print(f'  LO1420: {len(std_pair[1420].freqs)} channels, '
          f'jd={std_pair[1420].jd:.4f}, lst={std_pair[1420].lst:.4f}h')

    print('\nLoading Cygnus-X spectra...')
    cyg_pair = load_spectrum_pair(DATA_ROOT / 'cygnus-x_combined_spectra')
    print(f'  LO1420: {len(cyg_pair[1420].freqs)} channels, '
          f'jd={cyg_pair[1420].jd:.4f}, lst={cyg_pair[1420].lst:.4f}h')

    print('\nComputing ratio profiles...')
    std_ratio = compute_ratio_profile(std_pair)
    cyg_ratio = compute_ratio_profile(cyg_pair)

    print('\nComputing calibrated T_line profiles...')
    std_tline = compute_tline_profile(std_pair, cal)
    cyg_tline = compute_tline_profile(cyg_pair, cal)

    print(f'\nStandard field T_line peak: {np.nanmax(std_tline["Tline_sm"]):.2f} K')
    print(f'Cygnus-X field T_line peak: {np.nanmax(cyg_tline["Tline_sm"]):.2f} K')

    # ---- Generate figures ----
    print('\nGenerating figures...')

    plot_ratio_profile(std_ratio, cyg_ratio,
                       FIG_DIR / 'ratio_profile.pdf')

    plot_tline_calibrated(std_tline,
                          FIG_DIR / 'tline_calibrated.pdf')

    plot_pointing_track(FIG_DIR / 'pointing_track.pdf')

    plot_cygnus_x_spectrum(cyg_tline,
                           FIG_DIR / 'cygnus_x_spectrum.pdf')

    print('\nDone. All figures saved to', FIG_DIR)


if __name__ == '__main__':
    main()
