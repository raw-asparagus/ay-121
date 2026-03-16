import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .paths import FIGURES_DIR, ensure_output_dirs

TEXTWIDTH_IN = 7.59
COLUMNWIDTH_IN = 3.73
LABEL_SIZE = 9
TICK_SIZE = 8
LEGEND_SIZE = 8
ANNOTATION_SIZE = 9
EMPHASIS_SIZE = 10

# Visual weight scale tuned for the standard figure sizes defined above.
LW_NONE = 0.0
LW_GRID = 0.4
LW_FINE = 0.6
LW_GUIDE = 0.8
LW_LIGHT = 0.9
LW_STANDARD = 1.0
LW_MEDIUM = 1.1
LW_STRONG = 1.3
LW_FIT = 1.5
LW_MODEL = 1.6
LW_EMPHASIS = 1.8
LW_CALLOUT = 2.2
LW_LEVEL = 2.6

SCATTER_S_FINE = 10
SCATTER_S_STANDARD = 20
SCATTER_S_EMPHASIS = 30
SCATTER_S_CALLOUT = 60

MARKER_MS_FINE = 1.6
MARKER_MS_STANDARD = 8
MARKER_MS_MEDIUM = 10.5
MARKER_MS_LARGE = 16

PRIMARY_COLOR = "C0"
SECONDARY_COLOR = "C1"
TERTIARY_COLOR = "C2"
QUATERNARY_COLOR = "C3"
QUINARY_COLOR = "C4"
SENARY_COLOR = "C5"
SEPTENARY_COLOR = "C6"
NEUTRAL_COLOR = "C7"
LIGHT_NEUTRAL_COLOR = "C8"
NONARY_COLOR = "C9"
COMPONENT_COLORS = (QUINARY_COLOR, SENARY_COLOR, SEPTENARY_COLOR, LIGHT_NEUTRAL_COLOR, NONARY_COLOR)


mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": LABEL_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.grid": True,
        "axes.titlesize": EMPHASIS_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "grid.linewidth": LW_GRID,
        "grid.alpha": 0.5,
        "legend.fontsize": LEGEND_SIZE,
        "axes.unicode_minus": False,
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}",
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _escape_latex_text(text: str) -> str:
    safe = text
    safe = re.sub(r"(?<!\\)&", r"\\&", safe)
    safe = re.sub(r"(?<!\\)%", r"\\%", safe)
    safe = re.sub(r"(?<!\\)#", r"\\#", safe)
    return safe


def _tight_layout(fig: Figure, *, use_pyplot: bool = False, **kwargs) -> None:
    if use_pyplot:
        plt.tight_layout(**kwargs)
    else:
        fig.tight_layout(**kwargs)


def _save_figure(fig: Figure, path: Path, **kwargs) -> None:
    fig.savefig(path, **kwargs)


def _apply_grid(ax) -> None:
    ax.grid(True)


def _save_lab02_figure(fig: Figure, filename: str, **kwargs) -> None:
    ensure_output_dirs()
    _save_figure(fig, FIGURES_DIR / filename, **kwargs)


def _single_panel(figsize: tuple[float, float], *, constrained_layout: bool = False):
    return plt.subplots(figsize=figsize, constrained_layout=constrained_layout)


def _textwidth_figsize(height_out_of_8: float) -> tuple[float, float]:
    return (TEXTWIDTH_IN, height_out_of_8 / 8 * TEXTWIDTH_IN)


def _columnwidth_figsize(height_out_of_3_5: float) -> tuple[float, float]:
    return (COLUMNWIDTH_IN, height_out_of_3_5 / 3.5 * COLUMNWIDTH_IN)


def _stacked_panels(
    nrows: int,
    *,
    figsize: tuple[float, float],
    height_ratios: list[int] | tuple[int, ...],
    hspace: float = 0.0,
):
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": hspace},
    )
    if hspace == 0.0:
        fig.subplots_adjust(hspace=0.0)
    return fig, axes


def _grid_2x2(
    *,
    figsize: tuple[float, float],
    sharex: str | bool = "col",
):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=sharex)
    return fig, axes


def _zero_line(ax) -> None:
    ax.axhline(0.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--")


def _unity_line(ax, *, label: str | None = None) -> None:
    ax.axhline(1.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--", alpha=0.7, label=label)


def _reference_vline(ax, x: float, *, label: str | None = None, color: str = NEUTRAL_COLOR, lw: float = LW_LIGHT, ls: str = "--", alpha: float = 0.8) -> None:
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def figure(result, name: str):
    return result.figures[name]


def figure_names(result) -> list[str]:
    return sorted(result.figures)


def _eval_poly(v: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = v / 100.0
    out = np.zeros_like(x)
    for idx, coeff in enumerate(coeffs):
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
    fig, ax = _single_panel(_textwidth_figsize(3))

    # Region shading
    prev_reg = regions[0]; seg_start = 0; _seen_lbl = set()
    for j in range(1, len(regions) + 1):
        cur_reg = regions[j] if j < len(regions) else None
        if cur_reg != prev_reg or j == len(regions):
            lbl = _escape_latex_text(region_label[prev_reg]) if prev_reg not in _seen_lbl else '_nolegend_'
            _seen_lbl.add(prev_reg)
            ax.axvspan(seg_start - 0.4, j - 1 + 0.4,
                       color=region_color[prev_reg], alpha=0.12, label=lbl)
            seg_start = j; prev_reg = cur_reg

    ax.plot(x, G_cum_db, 'o-', color=TERTIARY_COLOR, lw=LW_EMPHASIS, zorder=3)

    _zero_line(ax)
    ax.set_ylabel(r'Cumulative gain [$\mathrm{dB}$]')
    ax.set_xticks(x)
    ax.set_xticklabels([_escape_latex_text(label) for label in labels], rotation=20, ha='right')
    _apply_grid(ax)
    ax.legend(loc='upper right')

    _tight_layout(fig, rect=[0, 0.05, 1, 1])
    _save_lab02_figure(fig, 'signal_chain.pdf')
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

    fig, axes = _stacked_panels(3, figsize=_textwidth_figsize(5), height_ratios=[5, 1, 1])

    # ── Top panel: data + fits + RG-58 reference lines ───────────────────────────
    ax = axes[0]
    all_inlier = ~drop_mask
    ax.scatter(L_all[all_inlier], y1420_all[all_inlier], color=PRIMARY_COLOR, s=SCATTER_S_FINE, label=r'LO 1420')
    ax.scatter(L_all[all_inlier], y1421_all[all_inlier], color=SECONDARY_COLOR, s=SCATTER_S_FINE, label=r'LO 1421')
    if np.any(~all_inlier):
        ax.scatter(L_all[~all_inlier], y1420_all[~all_inlier], color=PRIMARY_COLOR, s=SCATTER_S_STANDARD, marker='x',
                   lw=LW_STANDARD, label=r'omitted')
        ax.scatter(L_all[~all_inlier], y1421_all[~all_inlier], color=SECONDARY_COLOR, s=SCATTER_S_STANDARD, marker='x',
                   lw=LW_STANDARD)

    ax.plot(L_line, _line_y(fit_lin_all['B1420'], fit_lin_all['alpha'], L_line),
            color=PRIMARY_COLOR, lw=LW_STANDARD, ls=':', label=r'all-point fit')
    ax.plot(L_line, _line_y(fit_lin_all['B1421'], fit_lin_all['alpha'], L_line),
            color=SECONDARY_COLOR, lw=LW_STANDARD, ls=':')
    ax.plot(L_line, _line_y(fit_lin['B1420'], fit_lin['alpha'], L_line),
            color=PRIMARY_COLOR, lw=LW_STANDARD, ls='--', label=r'subset fit')
    ax.plot(L_line, _line_y(fit_lin['B1421'], fit_lin['alpha'], L_line),
            color=SECONDARY_COLOR, lw=LW_STANDARD, ls='--')

    # RG-58 reference lines (same intercepts as primary fit, published slopes)
    coax_refs = [
        (r'RG-58 typical', 0.440, QUINARY_COLOR, '-.'),
        (r'RG-58 weather', 0.748, SEPTENARY_COLOR, '-.'),
    ]
    for label, alpha_ref, color, ls in coax_refs:
        ax.plot(L_line, _line_y(fit_lin['B1420'], alpha_ref, L_line),
                color=color, lw=LW_LIGHT, ls=ls, label=label)

    ax.set_ylabel(r'Normalised power [$\mathrm{dB}$]')
    ax.tick_params(labelbottom=False)
    _apply_grid(ax)
    ax.legend(ncols=3)

    # ── Middle panel: all-point fit residuals (includes screened points) ──────────
    ax = axes[1]
    ax.scatter(L_all[all_inlier], fit_lin_all['row_resid_1420'][all_inlier],
               color=PRIMARY_COLOR, s=SCATTER_S_FINE, alpha=0.5)
    ax.scatter(L_all[all_inlier], fit_lin_all['row_resid_1421'][all_inlier],
               color=SECONDARY_COLOR, s=SCATTER_S_FINE, alpha=0.5)
    if np.any(drop_mask):
        ax.scatter(L_all[drop_mask], fit_lin_all['row_resid_1420'][drop_mask],
                   color=PRIMARY_COLOR, s=SCATTER_S_STANDARD, alpha=0.6, marker='x', lw=LW_FIT)
        ax.scatter(L_all[drop_mask], fit_lin_all['row_resid_1421'][drop_mask],
                   color=SECONDARY_COLOR, s=SCATTER_S_STANDARD, alpha=0.6, marker='x', lw=LW_FIT)
    _zero_line(ax)
    ax.tick_params(labelbottom=False, labelsize=TICK_SIZE)
    ax.set_ylabel('Resid.\n\n' + r'[$\mathrm{dB}$]', fontsize=LABEL_SIZE)
    _apply_grid(ax)

    # ── Bottom panel: primary (screened) fit residuals ────────────────────────────
    ax = axes[2]
    ax.scatter(L, fit_lin['row_resid_1420'], color=PRIMARY_COLOR, s=SCATTER_S_FINE, alpha=0.5)
    ax.scatter(L, fit_lin['row_resid_1421'], color=SECONDARY_COLOR, s=SCATTER_S_FINE, alpha=0.5)
    _zero_line(ax)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlabel(r'Cable length [$\mathrm{m}$]')
    ax.set_ylabel('Resid.\n(omitted)\n' + r'[$\mathrm{dB}$]', fontsize=LABEL_SIZE)
    _apply_grid(ax)

    # Shared y-limits for both residual panels (use larger range)
    all_resid = np.concatenate([
        fit_lin_all['row_resid_1420'], fit_lin_all['row_resid_1421'],
        fit_lin['row_resid_1420'],     fit_lin['row_resid_1421'],
    ])
    rmax = np.nanmax(np.abs(all_resid)) * 1.25
    for ax in (axes[1], axes[2]):
        ax.set_ylim(-rmax, rmax)

    _save_lab02_figure(fig, 'cable_attenuation_lo.pdf')
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
    fig, axes = _stacked_panels(2, figsize=_textwidth_figsize(3.5), height_ratios=[3, 1])

    # ── Top panel: normalised SDR 1420 and power meter overlaid ──────────────────
    ax = axes[0]
    ax.scatter(L, y1420_n, color=PRIMARY_COLOR, s=SCATTER_S_EMPHASIS, label=r'SDR 1420', zorder=4)
    ax.plot(L_line, sdr_line_n, color=PRIMARY_COLOR, lw=LW_FIT, ls='--', label=r'SDR 1420 fit')
    ax.scatter(L, meter_n, color=TERTIARY_COLOR, s=SCATTER_S_EMPHASIS, marker='^', label=r'Power meter', zorder=4)
    ax.plot(L_line, meter_line_n, color=TERTIARY_COLOR, lw=LW_FIT, ls='--', label=r'Power meter fit')
    ax.set_ylabel('Relative attenuation\n' + r'[$\mathrm{dB}$]')
    ax.tick_params(labelbottom=False)
    ax.legend(ncols=2)
    _apply_grid(ax)

    # ── Bottom panel: residuals ───────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(L, fit_lin['row_resid_1420'], color=PRIMARY_COLOR, s=SCATTER_S_STANDARD)
    ax.scatter(L, meter_resid, color=TERTIARY_COLOR, s=SCATTER_S_STANDARD, marker='^')
    _zero_line(ax)
    ax.set_xlabel(r'Cable length [$\mathrm{m}$]')
    ax.set_ylabel('Residual\n' + r'[$\mathrm{dB}$]')
    _apply_grid(ax)

    _save_lab02_figure(fig, 'cable_attenuation_power_meter.pdf')
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
    fig, ax = _single_panel(_textwidth_figsize(3))
    ax.plot(t_grid, wave, lw=LW_CALLOUT, color=PRIMARY_COLOR)

    # Keep timing labels horizontal below the top border, and stagger y-levels to avoid overlap.
    label_levels = [0.25, 0.1]
    label_spacing_ns = 30.0
    last_x_for_level = [-np.inf] * len(label_levels)

    for t in sorted(TIMES_NS):
        _reference_vline(ax, t, color=NEUTRAL_COLOR, lw=LW_STANDARD)

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
            rf"${int(t)}\,\mathrm{{ns}}$",
            rotation=0,
            va='top',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.85),
        )

    _reference_vline(ax, T_FIRST_PLATEAU_START_NS, color=QUATERNARY_COLOR, lw=LW_CALLOUT, alpha=1.0)
    _reference_vline(ax, T_MAX_PLATEAU_START_NS, color=TERTIARY_COLOR, lw=LW_CALLOUT, alpha=1.0)
    ax.annotate(
        rf"$\Delta t_{{\mathrm{{mod}}}} = {TAU_MOD_NS:.0f}\,\mathrm{{ns}}$",
        xy=((T_FIRST_PLATEAU_START_NS + T_MAX_PLATEAU_START_NS) / 2, 0.88),
        xytext=(-360, 0.95),
        ha='center',
    )

    ax.set_xlabel(r'Time [$\mathrm{ns}$]')
    ax.set_ylabel(r'Normalized amplitude')
    _apply_grid(ax)
    _tight_layout(fig, use_pyplot=True)
    _save_lab02_figure(fig, 'reflectometry.pdf')
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
    fig, ax = _single_panel(_textwidth_figsize(3), constrained_layout=True)

    ax.scatter(g['siggen_amp_dbm'][:-2], g['total_power_db'][:-2], color=PRIMARY_COLOR, alpha=0.7, label=r'all points')
    if len(clipped):
        ax.scatter(clipped['siggen_amp_dbm'], clipped['total_power_db'],
                   marker='x', s=SCATTER_S_CALLOUT, color=QUATERNARY_COLOR, label=r'clipped')
    if len(unclipped) >= 2:
        xfit = np.linspace(unclipped['siggen_amp_dbm'].min(), unclipped['siggen_amp_dbm'].max(), 200)
        ax.plot(xfit, slope * xfit + intercept, '--', color=NEUTRAL_COLOR,
                label=rf'unclipped fit: $y={slope:.3f}x+{intercept:.2f}$')
    ax.set_xlabel(r'Signal Generator setpoint [$\mathrm{dBm}$]')
    ax.set_ylabel(r'SDR total power [$\mathrm{dB}$]')
    _apply_grid(ax)
    ax.legend()

    _save_lab02_figure(fig, 'sdr_gain_response_clipping.pdf')
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
    fig, ax = _single_panel(_textwidth_figsize(3))
    ax.plot(freq_offset_mhz[combined_mask], noise_norm[combined_mask],
            lw=LW_LIGHT, color=PRIMARY_COLOR, alpha=0.7, label=r'raw data')
    ax.plot(freq_offset_mhz[combined_mask], after_init_n[combined_mask],
            lw=LW_LIGHT, color=SECONDARY_COLOR, alpha=0.7, label=r'after FIR + summing (guess)')
    ax.plot(freq_offset_mhz[combined_mask], after_opt_n[combined_mask],
            lw=LW_LIGHT, color=TERTIARY_COLOR, alpha=0.8, label=r'after FIR + summing (optimized)')
    _unity_line(ax, label=r'$y = 1$ (ideal)')
    ax.set_xlabel(r'Frequency offset from LO [$\mathrm{MHz}$]')
    ax.set_ylabel(r'Normalised power')
    ax.legend()
    _apply_grid(ax)
    ax.set_ylim(0.4, 1.4)
    _tight_layout(fig)
    _save_lab02_figure(fig, 'sdr_fir_summing_correction.pdf')
    plt.show()
    return fig


def sigma_masking(
    *,
    worst_freqs: np.ndarray,
    worst_psd: np.ndarray,
    worst_mask: np.ndarray,
):
    fig, ax = _single_panel(_textwidth_figsize(3))
    ax.plot(
        worst_freqs[~worst_mask],
        worst_psd[~worst_mask],
        marker='x',
        ms=MARKER_MS_STANDARD,
        alpha=0.80,
        lw=LW_NONE,
        color=QUATERNARY_COLOR,
        label=r'flagged (removed)',
    )
    ax.plot(
        worst_freqs[worst_mask],
        worst_psd[worst_mask],
        '.',
        ms=MARKER_MS_FINE,
        alpha=0.35,
        color=PRIMARY_COLOR,
        label=r'kept after sigma clip',
    )
    ax.set_xlabel(r'Frequency [$\mathrm{MHz}$]')
    ax.set_ylabel(r'PSD')
    _apply_grid(ax)
    ax.legend(loc='best')
    _tight_layout(fig)
    _save_lab02_figure(fig, 'sigma_masking.pdf')
    plt.show()
    return fig


def per_frequency_trx(
    *,
    human_pair,
    yfactor_common_masks: dict[int, np.ndarray],
    trx_spec: dict[int, np.ndarray],
    yfactor_results: dict[int, Any],
):
    fig, ax = _single_panel(_textwidth_figsize(3.5))
    colors = {1420: PRIMARY_COLOR, 1421: SECONDARY_COLOR}
    HI_FREQ_MHZ = 1420.405751768
    _reference_vline(ax, HI_FREQ_MHZ, color=NEUTRAL_COLOR, label=r'HI rest freq')
    for lo in [1420, 1421]:
        freqs_mhz = human_pair[lo].freqs / 1e6
        mask = yfactor_common_masks[lo]
        trx = trx_spec[lo]
        ax.plot(freqs_mhz[mask], trx[mask], lw=LW_LIGHT, alpha=0.8,
                color=colors[lo], label=rf'Per-channel $T_{{\rm rx}}$, LO ${lo}\,\mathrm{{MHz}}$')
        scalar_trx = yfactor_results[lo].T_rx
        ax.axhline(scalar_trx, ls='--', lw=LW_STRONG, color=colors[lo], alpha=0.6, label=rf'$T_{{\rm rx}}$, LO ${lo}\,\mathrm{{MHz}}$')

    ax.set_ylabel(r'$T_{\mathrm{rx}}(\nu)$ [$\mathrm{K}$]')
    ax.set_xlabel(r'Frequency [$\mathrm{MHz}$]')
    _apply_grid(ax)
    ax.legend(ncols=2, loc='upper left')
    _tight_layout(fig)
    _save_lab02_figure(fig, 'per_frequency_trx.pdf')
    plt.show()
    return fig


def ratio_profile(
    *,
    standard: dict[str, np.ndarray],
    cygnus_x: dict[str, np.ndarray],
    smooth_nchan: int,
):
    fig, axes = _grid_2x2(figsize=_textwidth_figsize(5), sharex="col")

    vmin, vmax = -200, 200
    configs = [
        (axes[0, 0], standard, "v0", "y_R", "y_R_fit", r"Standard Field $(l,b)=(120^\circ,0^\circ)$: $R-1$", r"$R - 1$"),
        (axes[0, 1], standard, "v1", "y_inv", "y_inv_fit", r"Standard Field: $1/R - 1$", r"$1/R - 1$"),
        (axes[1, 0], cygnus_x, "v0", "y_R", "y_R_fit", r"Cygnus-X Field: $R-1$", r"$R - 1$"),
        (axes[1, 1], cygnus_x, "v1", "y_inv", "y_inv_fit", r"Cygnus-X Field: $1/R - 1$", r"$1/R - 1$"),
    ]

    for ax, data, xkey, raw_key, smooth_key, title, ylabel in configs:
        vel = data[xkey]
        y_raw = data[raw_key]
        y_smooth = data[smooth_key]
        sel = (vel > vmin) & (vel < vmax)

        ax.plot(vel[sel], y_raw[sel], color=PRIMARY_COLOR, lw=LW_GRID, alpha=0.35, label=r"raw")
        ax.plot(
            vel[sel],
            y_smooth[sel],
            color=PRIMARY_COLOR,
            lw=LW_MEDIUM,
            alpha=0.9,
            label=rf"smoothed ($n={smooth_nchan}$)",
        )
        _zero_line(ax)
        ax.set_xlim(vmin, vmax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"$v_\mathrm{LSR}$ [$\mathrm{km\,s^{-1}}$]")
        ax.legend(loc="lower left")
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        _apply_grid(ax)

    _save_lab02_figure(fig, "ratio_profile.pdf")
    plt.show()
    return fig


def hyperfine():
    fig, ax = _single_panel(_columnwidth_figsize(2))
    ax.set_xlim(0.2, 8.4)
    ax.set_ylim(2.8, 8.2)
    ax.axis('off')

    triplet_color = PRIMARY_COLOR
    singlet_color = QUATERNARY_COLOR
    emission_color = NEUTRAL_COLOR
    gap_color = SENARY_COLOR

    # Energy levels
    ax.hlines(6.8, 1.1, 7.6, colors=triplet_color, linewidths=LW_LEVEL)
    ax.hlines(4.0, 1.1, 7.6, colors=singlet_color, linewidths=LW_LEVEL)

    # Labels
    ax.text(0.85, 6.8, r'$F = 1$' + '\n' + r'(triplet)', va='center', ha='right',
            color=triplet_color, fontweight='bold', fontsize=EMPHASIS_SIZE)
    ax.text(0.85, 4.0, r'$F = 0$' + '\n' + r'(singlet)', va='center', ha='right',
            color=singlet_color, fontweight='bold', fontsize=EMPHASIS_SIZE)

    # ── Spin arrows F=1: electron (left) and proton (right), BOTH pointing up ──
    ax.annotate('', xy=(2.0, 7.7), xytext=(2.0, 6.2),
                arrowprops=dict(arrowstyle='->', color=triplet_color, lw=LW_EMPHASIS))
    ax.text(2.0, 6.0, r'$e^{-}$', ha='center', va='top', color=triplet_color, fontsize=ANNOTATION_SIZE)

    ax.annotate('', xy=(2.8, 7.7), xytext=(2.8, 6.2),
                arrowprops=dict(arrowstyle='->', color=triplet_color, lw=LW_EMPHASIS))
    ax.text(2.8, 6.0, r'$p$', ha='center', va='top', color=triplet_color, fontsize=ANNOTATION_SIZE)

    ax.text(2.4, 7.95, r'parallel spins $\uparrow\uparrow$', ha='center', va='bottom',
            color=triplet_color, fontsize=ANNOTATION_SIZE)

    # ── Spin arrows F=0: electron pointing UP, proton pointing DOWN ──
    ax.annotate('', xy=(2.0, 4.9), xytext=(2.0, 3.4),   # electron: up
                arrowprops=dict(arrowstyle='->', color=singlet_color, lw=LW_EMPHASIS))
    ax.text(2.0, 3.2, r'$e^{-}$', ha='center', va='top', color=singlet_color, fontsize=ANNOTATION_SIZE)

    ax.annotate('', xy=(2.8, 3.1), xytext=(2.8, 4.6),   # proton: down
                arrowprops=dict(arrowstyle='->', color=singlet_color, lw=LW_EMPHASIS))
    ax.text(2.8, 4.75, r'$p$', ha='center', va='bottom', color=singlet_color, fontsize=ANNOTATION_SIZE)

    ax.text(2.4, 2.9, r'anti-parallel spins $\uparrow\downarrow$', ha='center', va='top',
            color=singlet_color, fontsize=ANNOTATION_SIZE)

    # ── Spontaneous emission arrow (downward) with label to the LEFT ──
    ax.annotate('', xy=(5.35, 4.2), xytext=(5.35, 6.6),
                arrowprops=dict(arrowstyle='->', color=emission_color, lw=LW_CALLOUT))
    ax.text(5.1, 5.4, r'Spontaneous' + '\n' + r'emission' + '\n' + r'$A_{10} = 2.869 \times 10^{-15}\,\mathrm{s^{-1}}$' + '\n' + r'($\sim 11\,\mathrm{Myr}$)',
            ha='right', va='center', color=emission_color, fontsize=ANNOTATION_SIZE)

    # ── Energy gap annotation ────────────────────────────────────────────────────
    ax.annotate('', xy=(6.1, 4.15), xytext=(6.1, 6.65),
                arrowprops=dict(arrowstyle='<->', color=gap_color, lw=LW_STRONG))
    ax.text(6.28, 5.4, r'$\Delta E = h\nu$' + '\n' + r'$\nu \sim 1420.406\,\mathrm{MHz}$' + '\n' + r'$\lambda = 21.1\,\mathrm{cm}$',
            ha='left', va='center', color=gap_color, style='italic', fontsize=ANNOTATION_SIZE)

    _tight_layout(fig, use_pyplot=True, pad=0.2)
    _save_lab02_figure(fig, 'hyperfine.pdf', pad_inches=0.02)
    plt.show()
    return fig


def lsr_geometry():
    _bbox = dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.15')
    orbit_color = PRIMARY_COLOR
    earth_color = PRIMARY_COLOR
    spin_color = TERTIARY_COLOR
    sun_color = SECONDARY_COLOR
    target_color = QUATERNARY_COLOR

    fig, ax = _single_panel(_columnwidth_figsize(2.5))
    ax.set_xlim(-3.9, 4.9)
    ax.set_ylim(-3.7, 4.0)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Sun ──────────────────────────────────────────────────────────────────────
    ax.plot(0, 0, 'o', color=sun_color, markersize=MARKER_MS_LARGE, zorder=5,
            markeredgecolor=QUINARY_COLOR, markeredgewidth=LW_CALLOUT)
    ax.text(0, -0.52, r'Sun' + '\n' + r'(LSR origin)', ha='center', va='top', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── Earth orbit ───────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(3.8*np.cos(theta), 3.3*np.sin(theta), ls='--', color=orbit_color, alpha=0.3, linewidth=LW_STRONG,
            label=r'Earth orbit')

    # ── Earth ─────────────────────────────────────────────────────────────────────
    earth_angle = np.radians(50)
    ex, ey = 3.8*np.cos(earth_angle), 3.3*np.sin(earth_angle)
    ax.plot(ex, ey, 'o', color=earth_color, markersize=MARKER_MS_MEDIUM, zorder=5,
            markeredgecolor=QUINARY_COLOR, markeredgewidth=LW_FIT)
    ax.text(ex + 0.22, ey + 0.18, r'Earth', ha='left', va='bottom', fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── v_orb: tangent to orbit ────────────────────────────────────────────────────
    v_orb_angle = earth_angle + np.pi/2
    vox, voy = 1.25*np.cos(v_orb_angle), 1.25*np.sin(v_orb_angle)
    ax.annotate('', xy=(ex + vox, ey + voy), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color=orbit_color, lw=LW_CALLOUT))
    ax.text(ex + vox, ey + voy + 0.15,
            r'$v_\mathrm{orb}\approx 30\,\mathrm{km\,s^{-1}}$',
            ha='right', va='bottom', color=orbit_color, fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── v_spin: diurnal spin (small, roughly along local east) ────────────────────
    spin_angle = earth_angle + np.pi/2   # same direction, smaller magnitude
    spx, spy = 0.42*np.cos(spin_angle), 0.42*np.sin(spin_angle)
    ax.annotate('', xy=(ex + spx, ey + spy), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color=spin_color, lw=LW_FIT))
    ax.text(ex + spx + 0.2, ey + spy - 0.6,
            r'$v_\mathrm{spin}\approx 0.46\,\mathrm{km\,s^{-1}}$',
            ha='left', va='top', color=spin_color, fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── v_sun: solar motion toward Galactic apex ──────────────────────────────────
    apex_angle = np.radians(56)
    sax, say = 1.65*np.cos(apex_angle), 1.65*np.sin(apex_angle)
    ax.annotate('', xy=(sax, say), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=sun_color, lw=LW_CALLOUT))
    ax.text(-0.5, 0.02,
            r'$v_\mathrm{sun}\approx 20\,\mathrm{km\,s^{-1}}$' + '\n' + r'$(\ell\approx56^\circ,\ b\approx23^\circ)$',
            ha='right', va='center', color=sun_color, fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── Target direction ──────────────────────────────────────────────────────────
    target_angle = np.radians(210)
    tx = ex + 1.8*np.cos(target_angle)
    ty = ey + 1.8*np.sin(target_angle)
    ax.annotate('', xy=(tx, ty), xytext=(ex, ey),
                arrowprops=dict(arrowstyle='->', color=target_color, lw=LW_EMPHASIS))
    ax.text(tx - 0.02, ty + 0.18, r'$\hat{n}$ target',
            ha='right', va='top', color=target_color, fontsize=ANNOTATION_SIZE, bbox=_bbox)

    # ── Formula box (bottom left, clear of all arrows) ────────────────────────────
    ax.text(-3.75, -3.55,
            r'$v_{\rm LSR} = v_{\rm obs} - \vec{v}_{\rm helio} \cdot \hat{n}$' + '\n' +
            r'$\vec{v}_{\rm helio} = \vec{v}_{\rm orb} + \vec{v}_{\rm spin}$', va='bottom',
            bbox=dict(facecolor='white', edgecolor=NEUTRAL_COLOR,
                      boxstyle='round,pad=0.25', alpha=0.95), fontsize=ANNOTATION_SIZE)

    ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False)
    _tight_layout(fig, use_pyplot=True, pad=0.2)
    _save_lab02_figure(fig, 'lsr_geometry.pdf', pad_inches=0.02)
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
    fig, axes = _stacked_panels(2, figsize=_textwidth_figsize(2.5), height_ratios=[5, 1])
    top_ax, bottom_ax = axes

    for line, label, color in [
        (psd, r'Raw PSD', NEUTRAL_COLOR),
        (median_slide, rf'Median sliding ($N={window_size}$ channels)', PRIMARY_COLOR),
        (mean_slide, rf'Mean sliding ($N={window_size}$ channels)', SECONDARY_COLOR),
    ]:
        alpha = 0.2 if label == r'Raw PSD' else 0.4
        lw = LW_FINE if label == r'Raw PSD' else LW_STRONG
        top_ax.plot(freqs_mhz[focus], line[focus] / norm_ref, color=color, lw=lw, alpha=alpha, label=label)

    top_ax.set_ylabel(r'Normalized PSD')
    top_ax.legend()
    _apply_grid(top_ax)
    top_ax.tick_params(axis='x', which='both', labelbottom=False)

    bottom_ax.plot(freqs_mhz[focus], (mean_slide - median_slide)[focus] / norm_ref, color=TERTIARY_COLOR, lw=LW_MEDIUM)
    _zero_line(bottom_ax)
    bottom_ax.set_xlabel(r'Frequency [$\mathrm{MHz}$]')
    bottom_ax.set_ylabel(r'Diff.')
    _apply_grid(bottom_ax)

    fig.subplots_adjust(hspace=0.0)
    _save_lab02_figure(fig, 'mean_vs_median.pdf')
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

    fig, axes = _stacked_panels(2, figsize=_textwidth_figsize(4), height_ratios=[5, 1])
    axis_cal, axis_cal_resid = axes

    axis_cal.plot(vel_b[finite_b], profile_b[finite_b], lw=LW_FINE, alpha=0.45, color=TERTIARY_COLOR, label=rf'{cal_label}data')
    axis_cal.fill_between(vel_b[finite_b], profile_b[finite_b] - sigma_b[finite_b],
                          profile_b[finite_b] + sigma_b[finite_b], color=TERTIARY_COLOR, alpha=0.25, label=rf'{cal_label}$\pm 1\sigma$')
    model_b = fit_b.model(vgrid)
    p_b = fit_b.popt
    base_b = _eval_poly(vgrid, p_b[3 * fit_b.n_gauss:3 * fit_b.n_gauss + fit_b.poly_order + 1])
    axis_cal.plot(vgrid, model_b, color=QUATERNARY_COLOR, lw=LW_MODEL,
                  label=rf'{cal_label}fit ($n={fit_b.n_gauss}$, poly={fit_b.poly_order}, $\chi_r^2={fit_b.chi2_red:.3f}$)')
    axis_cal.plot(vgrid, base_b, color=NEUTRAL_COLOR, lw=LW_STANDARD, ls=':', label=r'Continuum baseline')
    comp_colors_b = [COMPONENT_COLORS[k % len(COMPONENT_COLORS)] for k in range(fit_b.n_gauss)]
    for k in range(fit_b.n_gauss):
        gk = p_b[3 * k] * np.exp(-0.5 * ((vgrid - p_b[3 * k + 1]) / p_b[3 * k + 2]) ** 2)
        fwhm_kms = 2.355 * p_b[3 * k + 2]
        comp_label = (
            rf'$v={p_b[3 * k + 1]:.1f}\,\mathrm{{km\,s^{{-1}}}}$, '
            rf'$A={p_b[3 * k]:.2f}\,\mathrm{{K}}$, '
            rf'$\mathrm{{FWHM}}={fwhm_kms:.1f}\,\mathrm{{km\,s^{{-1}}}}$'
        )
        axis_cal.plot(vgrid, base_b + gk, lw=LW_STANDARD, ls='--', color=comp_colors_b[k], label=comp_label)
    _zero_line(axis_cal)
    axis_cal.set_ylabel(r'Calibrated $T_{\mathrm{line}}$ [$\mathrm{K}$]')
    axis_cal.legend(fontsize=LEGEND_SIZE)
    axis_cal.tick_params(axis='x', which='both', labelbottom=False)
    _apply_grid(axis_cal)

    resid_b = (profile_b[finite_b] - fit_b.model(vel_b[finite_b])) / np.maximum(sigma_b[finite_b], 1e-9)
    axis_cal_resid.plot(vel_b[finite_b], resid_b, color=QUINARY_COLOR, lw=LW_LIGHT)
    _zero_line(axis_cal_resid)
    axis_cal_resid.set_ylabel(r'Residuals')
    axis_cal_resid.set_xlabel(r'LSR velocity [$\mathrm{km\,s^{-1}}$]')
    axis_cal_resid.set_xlim(vel_min, vel_max)
    _apply_grid(axis_cal_resid)

    fig.subplots_adjust(hspace=0.0)
    _save_lab02_figure(fig, f'{ds_name}_fits.pdf')
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
