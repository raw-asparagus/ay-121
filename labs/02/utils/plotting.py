from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.text import Text

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10

_SUBSCRIPT_MAP = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
}
_SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
}
_UNICODE_LATEX_MAP = {
    "±": r"$\pm$",
    "σ": r"$\sigma$",
    "Δ": r"$\Delta$",
    "ν": r"$\nu$",
    "λ": r"$\lambda$",
    "α": r"$\alpha$",
    "β": r"$\beta$",
    "η": r"$\eta$",
    "μ": r"$\mu$",
    "°": r"$^\circ$",
    "×": r"$\times$",
    "↑": r"$\uparrow$",
    "↓": r"$\downarrow$",
    "−": "-",
    "~": r"$\sim$",
}


def _replace_script_runs(text: str, mapping: dict[str, str], kind: str) -> str:
    if not text:
        return text
    chars = set(mapping)
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] not in chars:
            out.append(text[i])
            i += 1
            continue
        j = i
        while j < len(text) and text[j] in chars:
            j += 1
        inner = "".join(mapping[ch] for ch in text[i:j])
        if kind == "sub":
            out.append(f"$_{{{inner}}}$")
        else:
            out.append(f"$^{{{inner}}}$")
        i = j
    return "".join(out)


def _latex_safe_text(text: str) -> str:
    safe = _replace_script_runs(text, _SUBSCRIPT_MAP, "sub")
    safe = _replace_script_runs(safe, _SUPERSCRIPT_MAP, "sup")
    for raw, repl in _UNICODE_LATEX_MAP.items():
        safe = safe.replace(raw, repl)
    safe = re.sub(r"(?<!\\)&", r"\\&", safe)
    safe = re.sub(r"(?<!\\)%", r"\\%", safe)
    safe = re.sub(r"(?<!\\)#", r"\\#", safe)
    return safe


def _sanitize_figure_text(fig: Figure) -> None:
    for artist in fig.findobj(match=lambda obj: isinstance(obj, Text)):
        raw = artist.get_text()
        safe = _latex_safe_text(raw)
        if safe != raw:
            artist.set_text(safe)


def _tight_layout(fig: Figure, *, use_pyplot: bool = False, **kwargs) -> None:
    _sanitize_figure_text(fig)
    if use_pyplot:
        plt.tight_layout(**kwargs)
    else:
        fig.tight_layout(**kwargs)


def _save_figure(fig: Figure, path: Path, **kwargs) -> None:
    _sanitize_figure_text(fig)
    fig.savefig(path, **kwargs)


mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.titlesize": EMPHASIS_SIZE,
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
    }
)


def figure(result, name: str):
    return result.figures[name]


def figure_names(result) -> list[str]:
    return sorted(result.figures)


def _eval_poly(v: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = np.asarray(v, float) / 100.0
    out = np.zeros_like(x)
    for idx, coeff in enumerate(np.asarray(coeffs, float)):
        out += coeff * x**idx
    return out


def signal_chain(
    *,
    x: np.ndarray,
    G_cum_db: list[float] | np.ndarray,
    labels: list[str],
    regions: list[str],
    region_color: dict[str, str],
    region_label: dict[str, str],
):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.85))

    # Region shading
    prev_reg = regions[0]; seg_start = 0; _seen_lbl = set()
    for j in range(1, len(regions) + 1):
        cur_reg = regions[j] if j < len(regions) else None
        if cur_reg != prev_reg or j == len(regions):
            lbl = region_label[prev_reg] if prev_reg not in _seen_lbl else '_nolegend_'
            _seen_lbl.add(prev_reg)
            ax.axvspan(seg_start - 0.4, j - 1 + 0.4,
                       color=region_color[prev_reg], alpha=0.12, label=lbl)
            seg_start = j; prev_reg = cur_reg

    ax.plot(x, G_cum_db, 'o-', color='C2', lw=1.8, zorder=3)

    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_ylabel('Cumulative gain [dB]')
    ax.set_xticks(x)
    ax.set_xticklabels([_latex_safe_text(label) for label in labels], rotation=20, ha='right')
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(loc='upper right')

    _tight_layout(fig, rect=[0, 0.05, 1, 1])
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'signal_chain.pdf', bbox_inches='tight')
    plt.show()
    return fig


def cable_attenuation_lo(
    *,
    L_all: np.ndarray,
    y1420_all: np.ndarray,
    y1421_all: np.ndarray,
    drop_mask: np.ndarray,
    fit_lin_all: dict[str, Any],
    fit_lin: dict[str, Any],
    L: np.ndarray,
    L_line: np.ndarray,
):
    def _line_y(B, alpha, x):
        return B - alpha * x

    fig, axes = plt.subplots(
        3, 1, figsize=(TEXTWIDTH_IN, 4.74),
        height_ratios=[5, 1, 1],
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)

    # ── Top panel: data + fits + RG-58 reference lines ───────────────────────────
    ax = axes[0]
    all_inlier = ~drop_mask
    ax.scatter(L_all[all_inlier], y1420_all[all_inlier], color='C0', s=10, label='LO1420')
    ax.scatter(L_all[all_inlier], y1421_all[all_inlier], color='C1', s=10, label='LO1421')
    if np.any(~all_inlier):
        ax.scatter(L_all[~all_inlier], y1420_all[~all_inlier], color='C0', s=20, marker='x',
                   lw=1.0, label='omitted')
        ax.scatter(L_all[~all_inlier], y1421_all[~all_inlier], color='C1', s=20, marker='x',
                   lw=1.0)

    ax.plot(L_line, _line_y(fit_lin_all['B1420'], fit_lin_all['alpha'], L_line),
            color='C0', lw=1.0, ls=':', label='all-point fit')
    ax.plot(L_line, _line_y(fit_lin_all['B1421'], fit_lin_all['alpha'], L_line),
            color='C1', lw=1.0, ls=':')
    ax.plot(L_line, _line_y(fit_lin['B1420'], fit_lin['alpha'], L_line),
            color='C0', lw=1.0, ls='--', label='subset fit')
    ax.plot(L_line, _line_y(fit_lin['B1421'], fit_lin['alpha'], L_line),
            color='C1', lw=1.0, ls='--')

    # RG-58 reference lines (same intercepts as primary fit, published slopes)
    coax_refs = [
        ('RG-58 typical',     0.440, 'C4', '-.'),
        ('RG-58 weather', 0.748, 'C6', '-.'),
    ]
    for label, alpha_ref, color, ls in coax_refs:
        ax.plot(L_line, _line_y(fit_lin['B1420'], alpha_ref, L_line),
                color=color, lw=0.9, ls=ls, label=label)

    ax.set_ylabel('Normalised power [dB]')
    ax.tick_params(labelbottom=False)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(ncols=3)

    # ── Middle panel: all-point fit residuals (includes screened points) ──────────
    ax = axes[1]
    ax.scatter(L_all[all_inlier], fit_lin_all['row_resid_1420'][all_inlier],
               color='C0', s=10, alpha=0.5)
    ax.scatter(L_all[all_inlier], fit_lin_all['row_resid_1421'][all_inlier],
               color='C1', s=10, alpha=0.5)
    if np.any(drop_mask):
        ax.scatter(L_all[drop_mask], fit_lin_all['row_resid_1420'][drop_mask],
                   color='C0', s=20, alpha=0.6, marker='x', lw=1.5)
        ax.scatter(L_all[drop_mask], fit_lin_all['row_resid_1421'][drop_mask],
                   color='C1', s=20, alpha=0.6, marker='x', lw=1.5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.tick_params(labelbottom=False, labelsize=TICK_SIZE)
    ax.set_ylabel('Resid.\n\n[dB]', fontsize=LABEL_SIZE)
    ax.grid(True, lw=0.4, alpha=0.5)

    # ── Bottom panel: primary (screened) fit residuals ────────────────────────────
    ax = axes[2]
    ax.scatter(L, fit_lin['row_resid_1420'], color='C0', s=10, alpha=0.5)
    ax.scatter(L, fit_lin['row_resid_1421'], color='C1', s=10, alpha=0.5)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel('Cable length [m]')
    ax.set_ylabel('Resid.\n(omitted)\n[dB]', fontsize=LABEL_SIZE)
    ax.grid(True, lw=0.4, alpha=0.5)

    # Shared y-limits for both residual panels (use larger range)
    all_resid = np.concatenate([
        fit_lin_all['row_resid_1420'], fit_lin_all['row_resid_1421'],
        fit_lin['row_resid_1420'],     fit_lin['row_resid_1421'],
    ])
    rmax = np.nanmax(np.abs(all_resid)) * 1.25
    for ax in (axes[1], axes[2]):
        ax.set_ylim(-rmax, rmax)

    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'cable_attenuation_lo.pdf', bbox_inches='tight')
    plt.show()
    return fig


def cable_attenuation_power_meter(
    *,
    L: np.ndarray,
    y1420_n: np.ndarray,
    L_line: np.ndarray,
    sdr_line_n: np.ndarray,
    meter_n: np.ndarray,
    meter_line_n: np.ndarray,
    fit_lin: dict[str, Any],
    meter_resid: np.ndarray,
):
    fig, axes = plt.subplots(
        2, 1, figsize=(TEXTWIDTH_IN, 3.32),
        height_ratios=[3, 1],
        sharex=True,
    )
    fig.subplots_adjust(hspace=0)

    # ── Top panel: normalised SDR 1420 and power meter overlaid ──────────────────
    ax = axes[0]
    ax.scatter(L, y1420_n, color='C0', s=30, label='SDR 1420', zorder=4)
    ax.plot(L_line, sdr_line_n, color='C0', lw=1.5, ls='--', label='SDR 1420 fit')
    ax.scatter(L, meter_n, color='C2', s=30, marker='^', label='Power meter', zorder=4)
    ax.plot(L_line, meter_line_n, color='C2', lw=1.5, ls='--', label='Power meter fit')
    ax.set_ylabel('Relative attenuation\n[dB]')
    ax.tick_params(labelbottom=False)
    ax.legend(ncols=2)
    ax.grid(alpha=0.3)

    # ── Bottom panel: residuals ───────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(L, fit_lin['row_resid_1420'], color='C0', s=20)
    ax.scatter(L, meter_resid, color='C2', s=20, marker='^')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Cable length [m]')
    ax.set_ylabel('Residual\n[dB]')
    ax.grid(alpha=0.3)

    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'cable_attenuation_power_meter.pdf', bbox_inches='tight')
    plt.show()
    return fig


def reflectometry(
    *,
    t_grid: np.ndarray,
    wave: np.ndarray,
    TIMES_NS: list[float] | np.ndarray,
    T_FIRST_PLATEAU_START_NS: float,
    T_MAX_PLATEAU_START_NS: float,
    TAU_MOD_NS: float,
):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.85))
    ax.plot(t_grid, wave, lw=2.0, color='C0')

    # Keep timing labels horizontal below the top border, and stagger y-levels to avoid overlap.
    label_levels = [0.25, 0.1]
    label_spacing_ns = 30.0
    last_x_for_level = [-np.inf] * len(label_levels)

    for t in sorted(TIMES_NS):
        ax.axvline(t, color='0.45', lw=1.0, ls='--', alpha=0.8)

        level_idx = None
        for i, last_x in enumerate(last_x_for_level):
            if (t - last_x) >= label_spacing_ns:
                level_idx = i
                break
        if level_idx is None:
            level_idx = int(np.argmin(last_x_for_level))

        y_label = label_levels[level_idx]
        last_x_for_level[level_idx] = t
        ax.text(
            t,
            y_label,
            f"{int(t)} ns",
            rotation=0,
            va='top',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85),
        )

    ax.axvline(T_FIRST_PLATEAU_START_NS, color='C3', lw=2.2)
    ax.axvline(T_MAX_PLATEAU_START_NS, color='C2', lw=2.2)
    ax.annotate(
        r"$\Delta t_{mod}$ =" + f" {TAU_MOD_NS:.0f} ns",
        xy=((T_FIRST_PLATEAU_START_NS + T_MAX_PLATEAU_START_NS) / 2, 0.88),
        xytext=(-360, 0.95),
        ha='center',
    )

    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Normalized amplitude')
    ax.grid(alpha=0.25)
    _tight_layout(fig, use_pyplot=True)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'reflectometry.pdf', bbox_inches='tight')
    plt.show()
    return fig


def sdr_gain_response_clipping(
    *,
    g,
    clipped,
    unclipped,
    slope: float,
    intercept: float,
):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 2.85), constrained_layout=True)

    ax.scatter(g['siggen_amp_dbm'][:-2], g['total_power_db'][:-2], color='C0', alpha=0.7, label='all points')
    if len(clipped):
        ax.scatter(clipped['siggen_amp_dbm'], clipped['total_power_db'],
                   marker='x', s=60, color='C3', label='clipped')
    if len(unclipped) >= 2:
        xfit = np.linspace(unclipped['siggen_amp_dbm'].min(), unclipped['siggen_amp_dbm'].max(), 200)
        ax.plot(xfit, slope * xfit + intercept, '--', color='k',
                label=r'unclipped fit: $' + f'y={slope:.3f}x+{intercept:.2f}' + r'$')
    ax.set_xlabel('Signal Generator setpoint [dBm]')
    ax.set_ylabel('SDR total power [dB]')
    ax.grid(alpha=0.3)
    ax.legend()

    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'sdr_gain_response_clipping.pdf', bbox_inches='tight')
    plt.show()
    return fig


def sdr_fir_summing_correction(
    *,
    freq_offset_mhz: np.ndarray,
    combined_mask: np.ndarray,
    noise_norm: np.ndarray,
    after_init_n: np.ndarray,
    after_opt_n: np.ndarray,
):
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, 2.85))
    ax.plot(freq_offset_mhz[combined_mask], noise_norm[combined_mask],
            lw=0.9, color='C0', alpha=0.7, label='raw data')
    ax.plot(freq_offset_mhz[combined_mask], after_init_n[combined_mask],
            lw=0.9, color='C1', alpha=0.7, label='after FIR + summing (guess)')
    ax.plot(freq_offset_mhz[combined_mask], after_opt_n[combined_mask],
            lw=0.9, color='C2', alpha=0.8, label='after FIR + summing (optimized)')
    ax.axhline(1.0, color='k', lw=0.7, ls='--', alpha=0.5, label='y = 1 (ideal)')
    ax.set_xlabel('Frequency offset from LO [MHz]')
    ax.set_ylabel('Normalised power')
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.set_ylim(0.4, 1.4)
    _tight_layout(fig)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'sdr_fir_summing_correction.pdf', bbox_inches='tight')
    plt.show()
    return fig


def sigma_masking(
    *,
    worst_freqs: np.ndarray,
    worst_psd: np.ndarray,
    worst_mask: np.ndarray,
):
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, 2.37))
    ax.plot(
        worst_freqs[~worst_mask],
        worst_psd[~worst_mask],
        marker='x',
        ms=10,
        alpha=0.80,
        lw=0.0,
        color='tab:red',
        label='flagged (removed)',
    )
    ax.plot(
        worst_freqs[worst_mask],
        worst_psd[worst_mask],
        '.',
        ms=1.6,
        alpha=0.35,
        color='tab:blue',
        label='kept after sigma clip',
    )
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('PSD')
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.legend(loc='best')
    _tight_layout(fig)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'sigma_masking.pdf', bbox_inches='tight')
    plt.show()
    return fig


def per_frequency_trx(
    *,
    human_pair,
    yfactor_common_masks: dict[int, np.ndarray],
    trx_spec: dict[int, np.ndarray],
    yfactor_results: dict[int, Any],
):
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, 3.32))
    COLORS = {1420: '#1f77b4', 1421: '#ff7f0e'}
    HI_FREQ_MHZ = 1420.405751768
    ax.axvline(HI_FREQ_MHZ, ls='--', lw=0.9, color='grey', alpha=0.8, label='HI rest freq')
    for lo in [1420, 1421]:
        freqs_mhz = human_pair[lo].freqs / 1e6
        mask = yfactor_common_masks[lo]
        trx = trx_spec[lo]
        ax.plot(freqs_mhz[mask], trx[mask], lw=0.9, alpha=0.8,
                color=COLORS[lo], label=r'Per channel $T_{\rm rx}$' + f' LO {lo} MHz')
        scalar_trx = yfactor_results[lo].T_rx
        ax.axhline(scalar_trx, ls='--', lw=1.3, color=COLORS[lo], alpha=0.6, label=r'$T_{\rm rx}$' + f' LO {lo} MHz')

    ax.set_ylabel(r'$T_{\mathrm{rx}}(\nu)$ [K]')
    ax.set_xlabel('Frequency [MHz]')
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.legend(ncols=2, loc='upper left')
    _tight_layout(fig)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    out_pdf = _nb_dir / 'report' / 'figures' / 'per_frequency_trx.pdf'
    _save_figure(fig, out_pdf, bbox_inches='tight')
    plt.show()
    return fig


def ratio_profile(
    *,
    standard: dict[str, np.ndarray],
    cygnus_x: dict[str, np.ndarray],
    smooth_nchan: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 4.56), sharex="col")
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    vmin, vmax = -200, 200
    configs = [
        (axes[0, 0], standard, "v0", "y_R", "y_R_fit", r"Standard Field $(l,b)=(120^\circ,0^\circ)$: $R-1$", r"$R - 1$"),
        (axes[0, 1], standard, "v1", "y_inv", "y_inv_fit", r"Standard Field: $1/R - 1$", r"$1/R - 1$"),
        (axes[1, 0], cygnus_x, "v0", "y_R", "y_R_fit", r"Cygnus-X Field: $R-1$", r"$R - 1$"),
        (axes[1, 1], cygnus_x, "v1", "y_inv", "y_inv_fit", r"Cygnus-X Field: $1/R - 1$", r"$1/R - 1$"),
    ]

    for ax, data, xkey, raw_key, smooth_key, title, ylabel in configs:
        vel = np.asarray(data[xkey], float)
        y_raw = np.asarray(data[raw_key], float)
        y_smooth = np.asarray(data[smooth_key], float)
        sel = (vel > vmin) & (vel < vmax)

        ax.plot(vel[sel], y_raw[sel], color="C0", lw=0.3, alpha=0.35, label="raw")
        ax.plot(
            vel[sel],
            y_smooth[sel],
            color="C0",
            lw=1.1,
            alpha=0.9,
            label=f"smoothed (n={smooth_nchan})",
        )
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xlim(vmin, vmax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"$v_\mathrm{LSR}$ [km/s]")
        ax.legend(loc="lower left")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.grid(True, alpha=0.7)

    _nb_dir = Path("labs/02") if Path("labs/02").exists() else Path(".")
    _save_figure(fig, _nb_dir / "report" / "figures" / "ratio_profile.pdf", bbox_inches="tight")
    plt.show()
    return fig


def hyperfine():
    fig, ax = plt.subplots(figsize=(COLUMNWIDTH_IN, 2.0))
    ax.set_xlim(0.2, 8.4)
    ax.set_ylim(2.8, 8.2)
    ax.axis('off')

    # Energy levels
    ax.hlines(6.8, 1.1, 7.6, colors='#2166ac', linewidths=2.6)
    ax.hlines(4.0, 1.1, 7.6, colors='#d6604d', linewidths=2.6)

    # Labels
    ax.text(0.85, 6.8, 'F = 1\n(triplet)', va='center', ha='right',
            color='#2166ac', fontweight='bold', fontsize=EMPHASIS_SIZE)
    ax.text(0.85, 4.0, 'F = 0\n(singlet)', va='center', ha='right',
            color='#d6604d', fontweight='bold', fontsize=EMPHASIS_SIZE)

    # ── Spin arrows F=1: electron (left) and proton (right), BOTH pointing up ──
    ax.annotate('', xy=(2.0, 7.7), xytext=(2.0, 6.2),
                arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.8))
    ax.text(2.0, 6.0, 'e⁻', ha='center', va='top', color='#2166ac', fontsize=ANNOTATION_SIZE)

    ax.annotate('', xy=(2.8, 7.7), xytext=(2.8, 6.2),
                arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.8))
    ax.text(2.8, 6.0, 'p', ha='center', va='top', color='#2166ac', fontsize=ANNOTATION_SIZE)

    ax.text(2.4, 7.95, 'parallel spins  ↑↑', ha='center', va='bottom',
            color='#2166ac', fontsize=ANNOTATION_SIZE)

    # ── Spin arrows F=0: electron pointing UP, proton pointing DOWN ──
    ax.annotate('', xy=(2.0, 4.9), xytext=(2.0, 3.4),   # electron: up
                arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.8))
    ax.text(2.0, 3.2, 'e⁻', ha='center', va='top', color='#d6604d', fontsize=ANNOTATION_SIZE)

    ax.annotate('', xy=(2.8, 3.1), xytext=(2.8, 4.6),   # proton: down
                arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.8))
    ax.text(2.8, 4.75, 'p', ha='center', va='bottom', color='#d6604d', fontsize=ANNOTATION_SIZE)

    ax.text(2.4, 2.9, 'anti-parallel spins  ↑↓', ha='center', va='top',
            color='#d6604d', fontsize=ANNOTATION_SIZE)

    # ── Spontaneous emission arrow (downward) with label to the LEFT ──
    ax.annotate('', xy=(5.35, 4.2), xytext=(5.35, 6.6),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.2))
    ax.text(5.1, 5.4, 'Spontaneous\nemission\nA₁₀ = 2.869×10⁻¹⁵ s⁻¹\n(~11 Myr)',
            ha='right', va='center', fontsize=ANNOTATION_SIZE)

    # ── Energy gap annotation — dark grey ──
    ax.annotate('', xy=(6.1, 4.15), xytext=(6.1, 6.65),
                arrowprops=dict(arrowstyle='<->', color='#444444', lw=1.4))
    ax.text(6.28, 5.4, 'ΔE = hν\nν = 1420.406 MHz\nλ = 21.1 cm',
            ha='left', va='center', color='#333333', style='italic', fontsize=ANNOTATION_SIZE)

    _tight_layout(fig, use_pyplot=True, pad=0.2)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'hyperfine.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()
    return fig


def lsr_geometry():
    _bbox = dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.15')

    fig, ax = plt.subplots(figsize=(COLUMNWIDTH_IN, 2.49))
    ax.set_xlim(-3.9, 4.9)
    ax.set_ylim(-3.7, 4.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Sun ──────────────────────────────────────────────────────────────────────
    ax.plot(0, 0, 'o', color='gold', markersize=16, zorder=5,
            markeredgecolor='orange', markeredgewidth=2)
    ax.text(0, -0.52, 'Sun\n(LSR origin)', ha='center', va='top', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── Earth orbit ───────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(3.8*np.cos(theta), 3.3*np.sin(theta), 'b--', alpha=0.3, linewidth=1.3,
            label='Earth orbit')

    # ── Earth ─────────────────────────────────────────────────────────────────────
    earth_angle = np.radians(50)
    ex, ey = 3.8*np.cos(earth_angle), 3.3*np.sin(earth_angle)
    ax.plot(ex, ey, 'o', color='steelblue', markersize=10.5, zorder=5,
            markeredgecolor='navy', markeredgewidth=1.5)
    ax.text(ex + 0.22, ey + 0.18, 'Earth', ha='left', va='bottom', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── v_orb: tangent to orbit ────────────────────────────────────────────────────
    v_orb_angle = earth_angle + np.pi/2
    vox, voy = 1.25*np.cos(v_orb_angle), 1.25*np.sin(v_orb_angle)
    ax.annotate('', xy=(ex + vox, ey + voy), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.2))
    ax.text(ex + vox, ey + voy + 0.15,
            r'$v_\mathrm{orb}\approx 30\,\mathrm{km\,s^{-1}}$',
            ha='right', va='bottom', color='steelblue', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── v_spin: diurnal spin (small, roughly along local east) ────────────────────
    spin_angle = earth_angle + np.pi/2   # same direction, smaller magnitude
    spx, spy = 0.42*np.cos(spin_angle), 0.42*np.sin(spin_angle)
    ax.annotate('', xy=(ex + spx, ey + spy), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color='teal', lw=1.5))
    ax.text(ex + spx + 0.2, ey + spy - 0.6,
            r'$v_\mathrm{spin}\approx 0.46\,\mathrm{km\,s^{-1}}$',
            ha='left', va='top', color='teal', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── v_sun: solar motion toward Galactic apex ──────────────────────────────────
    apex_angle = np.radians(56)
    sax, say = 1.65*np.cos(apex_angle), 1.65*np.sin(apex_angle)
    ax.annotate('', xy=(sax, say), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.2))
    ax.text(-0.5, 0.02,
            r'$v_\mathrm{sun}\approx 20\,\mathrm{km\,s^{-1}}$' + '\n' + r'$(\ell\approx56^\circ,\ b\approx23^\circ)$',
            ha='right', va='center', color='darkorange', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── Target direction ──────────────────────────────────────────────────────────
    target_angle = np.radians(210)
    tx = ex + 1.8*np.cos(target_angle)
    ty = ey + 1.8*np.sin(target_angle)
    ax.annotate('', xy=(tx, ty), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8))
    ax.text(tx - 0.02, ty + 0.18, r'$\hat{n}$ target',
            ha='right', va='top', color='crimson', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── Formula box (bottom left, clear of all arrows) ────────────────────────────
    ax.text(-3.75, -3.55,
            r'$v_{\rm LSR} = v_{\rm obs} - \vec{v}_{\rm helio} \cdot \hat{n}$' + '\n' +
            r'$\vec{v}_{\rm helio} = \vec{v}_{\rm orb} + \vec{v}_{\rm spin}$', va='bottom',
            bbox=dict(facecolor='lightyellow', edgecolor='#888888',
                      boxstyle='round,pad=0.25', alpha=0.95), fontsize=ANNOTATION_SIZE)

    ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False)
    _tight_layout(fig, use_pyplot=True, pad=0.2)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'lsr_geometry.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()
    return fig


def mean_vs_median(
    *,
    freqs_mhz: np.ndarray,
    focus: np.ndarray,
    psd: np.ndarray,
    median_slide: np.ndarray,
    mean_slide: np.ndarray,
    window_size: int,
    norm_ref: float,
):
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 2.37), sharex=True, height_ratios=(5, 1))
    top_ax, bottom_ax = axes

    for line, label, color in [
        (psd, 'Raw PSD', 'gray'),
        (median_slide, f'Median sliding ({window_size} ch)', 'C0'),
        (mean_slide, f'Mean sliding ({window_size} ch)', 'C1'),
    ]:
        alpha = 0.2 if label == 'Raw PSD' else 0.4
        lw = 0.6 if label == 'Raw PSD' else 1.2
        top_ax.plot(freqs_mhz[focus], line[focus] / norm_ref, color=color, lw=lw, alpha=alpha, label=label)

    top_ax.set_ylabel('Normalized PSD')
    top_ax.legend()
    top_ax.grid(alpha=0.3)

    bottom_ax.plot(freqs_mhz[focus], (mean_slide - median_slide)[focus] / norm_ref, color='C2', lw=1.1)
    bottom_ax.axhline(0, color='gray', lw=0.8, ls='--')
    bottom_ax.set_xlabel('Frequency [MHz]')
    bottom_ax.set_ylabel('Diff.')
    bottom_ax.grid(alpha=0.3)

    _tight_layout(fig)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / 'mean_vs_median.pdf', bbox_inches='tight')
    plt.show()
    return fig


def dataset_fits(
    *,
    ds_name: str,
    fit_b,
    vel_min: float,
    vel_max: float,
    vgrid: np.ndarray,
    profile_b: np.ndarray,
    sigma_b: np.ndarray,
    vel_b: np.ndarray,
    finite_b: np.ndarray,
):
    cal_label = ''

    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 3.8), sharex=True,
                             gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.0})
    axis_cal, axis_cal_resid = axes

    axis_cal.plot(vel_b[finite_b], profile_b[finite_b], lw=0.6, alpha=0.45, color='C2', label=f'{cal_label}data')
    axis_cal.fill_between(vel_b[finite_b], profile_b[finite_b] - sigma_b[finite_b],
                          profile_b[finite_b] + sigma_b[finite_b], color='C2', alpha=0.25, label=f'{cal_label}±1σ')
    model_b = fit_b.model(vgrid)
    p_b = fit_b.popt
    base_b = _eval_poly(vgrid, p_b[3 * fit_b.n_gauss:3 * fit_b.n_gauss + fit_b.poly_order + 1])
    axis_cal.plot(vgrid, model_b, color='C3', lw=1.6,
                  label=rf'{cal_label}fit (n={fit_b.n_gauss}, poly={fit_b.poly_order}, $\chi^2_r$={fit_b.chi2_red:.3f})')
    axis_cal.plot(vgrid, base_b, color='k', lw=1.0, ls=':', label='Continuum baseline')
    comp_colors_b = plt.cm.Oranges(np.linspace(0.45, 0.9, fit_b.n_gauss))
    for k in range(fit_b.n_gauss):
        gk = p_b[3 * k] * np.exp(-0.5 * ((vgrid - p_b[3 * k + 1]) / p_b[3 * k + 2]) ** 2)
        fwhm_kms = 2.355 * p_b[3 * k + 2]
        comp_label = f'v={p_b[3 * k + 1]:.1f}' + r'kms$^{-1}$' + f', A={p_b[3 * k]:.2f} K, FWHM={fwhm_kms:.1f} ' + r'kms$^{-1}$'
        axis_cal.plot(vgrid, base_b + gk, lw=1.0, ls='--', color=comp_colors_b[k], label=comp_label)
    axis_cal.axhline(0, color='gray', lw=0.8, ls='--')
    axis_cal.set_ylabel(r'Calibrated $T_{\mathrm{line}}$ [K]')
    axis_cal.legend(fontsize=LEGEND_SIZE)
    axis_cal.tick_params(axis='x', which='both', labelbottom=False)
    axis_cal.grid(True, alpha=0.6, ls='--')

    resid_b = (profile_b[finite_b] - fit_b.model(vel_b[finite_b])) / np.maximum(sigma_b[finite_b], 1e-9)
    axis_cal_resid.plot(vel_b[finite_b], resid_b, color='C4', lw=0.9)
    axis_cal_resid.axhline(0, color='black', lw=0.8, ls='--')
    axis_cal_resid.set_ylabel('Residuals')
    axis_cal_resid.set_xlabel('LSR velocity [km/s]')
    axis_cal_resid.set_xlim(vel_min, vel_max)
    axis_cal_resid.grid(True, alpha=0.6, ls='--')

    fig.subplots_adjust(hspace=0.0)
    _nb_dir = Path('labs/02') if Path('labs/02').exists() else Path('.')
    _save_figure(fig, _nb_dir / 'report' / 'figures' / f'{ds_name}_fits.pdf', bbox_inches='tight')
    plt.show()
    return fig, resid_b


plot_signal_chain = signal_chain
plot_cable_attenuation_lo = cable_attenuation_lo
plot_cable_attenuation_power_meter = cable_attenuation_power_meter
plot_reflectometry = reflectometry
plot_sdr_gain_response_clipping = sdr_gain_response_clipping
plot_sdr_fir_summing_correction = sdr_fir_summing_correction
plot_sigma_masking = sigma_masking
plot_per_frequency_trx = per_frequency_trx
plot_ratio_profile = ratio_profile
plot_hyperfine = hyperfine
plot_lsr_geometry = lsr_geometry
plot_mean_vs_median = mean_vs_median
plot_dataset_fits = dataset_fits
