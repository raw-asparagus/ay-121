"""Lab 3 -- reusable plotting functions for solar interferometry data.

All functions use the style constants from ``ugradiolab/plotting.py`` and return
``(fig, ax)`` or ``(fig, axes)`` for interactive use.  Notebooks import
individual functions::

    from plotters import plot_dc_before_after, plot_phase_slope

For report export, use :func:`savefig` which writes to ``report/figures/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1, jn_zeros

from ugradiolab.plotting import (
    TEXTWIDTH_IN,
    COLUMNWIDTH_IN,
    LABEL_SIZE,
    TICK_SIZE,
    LEGEND_SIZE,
    EMPHASIS_SIZE,
    LW_NONE,
    LW_FINE,
    LW_LIGHT,
    LW_STANDARD,
    MS_MICRO,
    MS_FINE,
    MS_STANDARD,
    SS_FINE,
    ALPHA_EXTRA_LIGHT,
    ALPHA_FAINT,
    ALPHA_LIGHT,
    ALPHA_STANDARD,
    NEUTRAL_COLOR,
    GRID_STYLE,
    GUIDE_STYLE,
    FIT_STYLE,
    ERRORBAR_STYLE,
    SCATTER_STYLE,
    FILL_STYLE,
    textwidth_figure,
    columnwidth_figure,
    subpanels,
    zero_line,
)


# ---------------------------------------------------------------------------
# Output directory and savefig
# ---------------------------------------------------------------------------

_LAB03_DIR = Path(__file__).resolve().parent
_FIGURES_DIR = _LAB03_DIR / "report" / "figures"

CHIP_COLORS = [f"C{i}" for i in range(10)]


def savefig(fig: plt.Figure, name: str) -> None:
    """Save *fig* as a PDF in ``report/figures/``."""
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / name)
    print(f"  {name}")



# ===================================================================
# Section 1: Theory figures (notebook 01)
# ===================================================================


def plot_fringe_synthetic(
    ha_deg: np.ndarray,
    fringe: np.ndarray,
    *,
    ha_lim: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    """Synthetic point-source fringe pattern vs hour angle.

    Parameters
    ----------
    ha_deg : array
        Hour-angle grid in degrees.
    fringe : array
        Fringe amplitude ``F(h)`` (from :func:`point_source_fringes`).
    ha_lim : tuple
        Hour-angle axis limits.

    Returns
    -------
    fig, ax
    """
    fig, ax = textwidth_figure(3)
    ax.plot(ha_deg, fringe, lw=LW_FINE, color="C0", zorder=2)
    ax.set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$F(h)$", fontsize=LABEL_SIZE)
    ax.set_xlim(ha_lim)
    return fig, ax


def plot_bessel_envelope_theory(
    *,
    x_max: float,
    n_zeros: int,
) -> tuple[plt.Figure, plt.Axes]:
    """Theoretical jinc (Bessel) envelope with labelled zeros.

    Parameters
    ----------
    x_max : float
        Upper limit of the horizontal axis (``|u| R``).
    n_zeros : int
        Number of ``J_1`` zeros to annotate.

    Returns
    -------
    fig, ax
    """
    x = np.linspace(0, x_max, 2000)
    jinc = np.where(x == 0, 1.0, 2 * j1(2 * np.pi * x) / (2 * np.pi * x))

    fig, ax = columnwidth_figure(4)
    ax.plot(x, jinc, "C0", lw=LW_STANDARD,
            label=r"$2J_1(2\pi x)/(2\pi x)$", zorder=2)
    ax.plot(x, np.abs(jinc), "C0", lw=LW_FINE, alpha=ALPHA_FAINT,
            label=r"$|2J_1(2\pi x)/(2\pi x)|$", zorder=2)

    zeros = jn_zeros(1, n_zeros) / (2 * np.pi)
    for i, z in enumerate(zeros):
        ax.axvline(z, color="C3", ls="--", lw=LW_FINE, alpha=ALPHA_LIGHT,
                   zorder=1)
        ax.text(z, 0.85 - 0.15 * i, rf"$j_{{1,{i + 1}}}$",
                fontsize=LEGEND_SIZE, ha="center", color="C3")

    zero_line(ax)
    ax.set_xlabel(r"$|u|\,R$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$V/V(0)$", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    return fig, ax


# ===================================================================
# Section 2a: Raw data inspection (notebook 02a)
# ===================================================================


def plot_raw_visibility_4panel(
    ha_deg: np.ndarray,
    vis_ch: np.ndarray,
    chip_slices: Sequence[slice],
    *,
    channel_idx: int | None,
    freq_ghz: float | None,
) -> tuple[plt.Figure, np.ndarray]:
    """Four-panel raw visibility vs hour angle (Re, Im, |V|, arg).

    Parameters
    ----------
    ha_deg : array, shape (N_cap,)
        Hour-angle per capture in degrees.
    vis_ch : complex array, shape (N_cap,)
        Complex visibility at one representative channel.
    chip_slices : sequence of slice
        Per-chip index slices into the capture axis.
    channel_idx : int or None
        Channel index (for title annotation).
    freq_ghz : float or None
        Sky frequency in GHz (for title annotation).

    Returns
    -------
    fig, axes : (Figure, ndarray of 4 Axes)
    """
    panels = [vis_ch.real, vis_ch.imag, np.abs(vis_ch), np.angle(vis_ch, deg=True)]
    labels = [r"Re$(V)$", r"Im$(V)$", r"$|V|$", r"$\arg(V)$ [deg]"]

    fig, _ax = textwidth_figure(17)
    _ax.remove()
    axes = subpanels(fig, 4)

    for ax, y, lbl in zip(axes, panels, labels):
        for ci, sl in enumerate(chip_slices):
            ax.scatter(
                ha_deg[sl], y[sl], s=0.15, color=CHIP_COLORS[ci],
                alpha=ALPHA_FAINT, rasterized=True, zorder=2,
                label=f"chip {ci}" if ax is axes[0] else None,
            )
        ax.set_ylabel(lbl, fontsize=TICK_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        if "Re" in lbl or "Im" in lbl:
            ax.axhline(0, color=NEUTRAL_COLOR, lw=LW_LIGHT, ls="--",
                       zorder=1)

    axes[0].legend(
        fontsize=TICK_SIZE, markerscale=8,
        ncol=len(chip_slices), loc="upper right",
    )
    title_parts = ["Raw visibility vs hour angle"]
    if channel_idx is not None and freq_ghz is not None:
        title_parts.append(f"(ch {channel_idx}, {freq_ghz:.3f} GHz)")
    axes[0].set_title(" ".join(title_parts), fontsize=TICK_SIZE)
    axes[-1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    return fig, axes


def plot_amplitude_spectrum(
    f_sky_ghz: np.ndarray,
    corr_raw: np.ndarray,
    chip_slices: Sequence[slice],
    *,
    plot_band_ghz: tuple[float, float] | None,
    bad_channels: Sequence[int],
) -> tuple[plt.Figure, plt.Axes]:
    """Time-averaged amplitude spectrum per chip across the IF band.

    Parameters
    ----------
    f_sky_ghz : array, shape (N_ch,)
        Sky frequency axis.
    corr_raw : complex array, shape (N_cap, N_ch)
        Raw complex visibility.
    chip_slices : sequence of slice
        Per-chip index slices.
    plot_band_ghz : tuple or None
        (lo, hi) of analysis band to shade.
    bad_channels : sequence of int
        Bad channel indices to mark.

    Returns
    -------
    fig, ax
    """
    fig, ax = textwidth_figure(3)

    for ci, sl in enumerate(chip_slices):
        amp_mean = np.nanmean(np.abs(corr_raw[sl]), axis=0)
        ax.plot(f_sky_ghz, amp_mean, lw=LW_FINE, alpha=ALPHA_STANDARD,
                color=CHIP_COLORS[ci], label=f"Chip {ci}", zorder=2)

    if plot_band_ghz is not None:
        ax.axvspan(plot_band_ghz[0], plot_band_ghz[1],
                   alpha=0.08, color="C0",
                   label="Analysis band", zorder=1)

    for i, bc in enumerate(bad_channels):
        ax.axvline(f_sky_ghz[bc], color=NEUTRAL_COLOR, lw=LW_FINE, ls=":",
                   alpha=ALPHA_LIGHT, zorder=1,
                   label="Bad ch" if i == 0 else None)

    ax.set_xlabel("Sky frequency [GHz]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$\langle|V|\rangle$", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, ncol=4)
    return fig, ax


def plot_observation_timeline(
    ha_deg: np.ndarray,
    duration: np.ndarray,
    gaps: np.ndarray,
    chip_slices: Sequence[slice],
) -> tuple[plt.Figure, np.ndarray]:
    """Three-panel observation timeline: duration, gap, and cumulative captures.

    Parameters
    ----------
    ha_deg : array, shape (N_cap,)
        Hour-angle per capture.
    duration : array, shape (N_cap,)
        Integration duration per capture in seconds.
    gaps : array, shape (N_cap - 1,)
        Inter-capture dead time in seconds.
    chip_slices : sequence of slice
        Per-chip index slices.

    Returns
    -------
    fig, axes : (Figure, ndarray of 3 Axes)
    """
    fig, _ax = textwidth_figure(11)
    _ax.remove()
    axes = subpanels(fig, 3)

    # (a) Integration duration
    for ci, sl in enumerate(chip_slices):
        axes[0].scatter(ha_deg[sl], duration[sl], s=0.3,
                        color=CHIP_COLORS[ci], alpha=ALPHA_LIGHT,
                        rasterized=True, zorder=2)
    axes[0].set_ylabel("Duration [s]", fontsize=LABEL_SIZE)
    axes[0].set_title("Observation timeline", fontsize=TICK_SIZE)

    # (b) Inter-capture gap
    for ci, sl in enumerate(chip_slices):
        idx = slice(sl.start, sl.stop - 1)
        axes[1].scatter(ha_deg[idx], gaps[idx], s=0.3,
                        color=CHIP_COLORS[ci], alpha=ALPHA_LIGHT,
                        rasterized=True, zorder=2)
    axes[1].set_ylabel("Gap [s]", fontsize=LABEL_SIZE)
    axes[1].set_yscale("log")

    # (c) Cumulative captures
    for ci, sl in enumerate(chip_slices):
        n = sl.stop - sl.start
        axes[2].plot(ha_deg[sl], np.arange(n), lw=LW_FINE,
                     color=CHIP_COLORS[ci], zorder=2,
                     label=f"chip {ci} ({n} cap)")
    axes[2].set_ylabel("Cumulative captures", fontsize=LABEL_SIZE)
    axes[2].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    axes[2].legend(fontsize=TICK_SIZE, ncol=len(chip_slices))

    return fig, axes


# ===================================================================
# Section 2b: DC-corrected data inspection (notebook 02b)
# ===================================================================


def plot_dc_before_after(
    ha_deg: np.ndarray,
    raw_re: np.ndarray,
    dc_re: np.ndarray,
    dc_offset: np.ndarray,
    chip_slices: Sequence[slice],
    *,
    channel_idx: int | None,
    freq_ghz: float | None,
    height_ratios: tuple[float, ...],
) -> tuple[plt.Figure, np.ndarray]:
    """Three-panel DC correction comparison: raw Re(V), corrected, and pedestal.

    Parameters
    ----------
    ha_deg : array, shape (N_cap,)
        Hour-angle per capture.
    raw_re : array, shape (N_cap,)
        Real part of raw visibility at a single channel.
    dc_re : array, shape (N_cap,)
        Real part after DC correction.
    dc_offset : array, shape (N_cap,)
        Subtracted DC pedestal (raw_re - dc_re).
    chip_slices : sequence of slice
        Per-chip index slices.
    channel_idx : int or None
        Channel index for the title.
    freq_ghz : float or None
        Sky frequency in GHz for the title.
    height_ratios : tuple
        Relative heights of the three panels.

    Returns
    -------
    fig, axes : (Figure, ndarray of 3 Axes)
    """
    fig, _ax = textwidth_figure(7)
    _ax.remove()
    axes = subpanels(fig, 3, height_ratios=list(height_ratios))

    for ci, sl in enumerate(chip_slices):
        axes[0].scatter(ha_deg[sl], raw_re[sl], s=0.15,
                        alpha=ALPHA_FAINT,
                        color=CHIP_COLORS[ci], rasterized=True, zorder=2)
        axes[1].scatter(ha_deg[sl], dc_re[sl], s=0.15,
                        alpha=ALPHA_FAINT,
                        color=CHIP_COLORS[ci], rasterized=True, zorder=2)
        axes[2].plot(ha_deg[sl], dc_offset[sl], lw=LW_FINE,
                     color=CHIP_COLORS[ci], zorder=2)

    for ax in axes[:2]:
        ax.axhline(0, color=NEUTRAL_COLOR, lw=LW_LIGHT, ls="--", zorder=1)

    axes[0].set_ylabel(r"Re$(V)$ raw", fontsize=LABEL_SIZE)
    axes[1].set_ylabel(r"Re$(V_{\rm dc})$", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("Diff", fontsize=LABEL_SIZE)
    axes[-1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)

    return fig, axes


def plot_dc_corrected_4panel(
    ha_deg: np.ndarray,
    vis_dc_ch: np.ndarray,
    chip_slices: Sequence[slice],
    *,
    channel_idx: int | None,
    freq_ghz: float | None,
) -> tuple[plt.Figure, np.ndarray]:
    """Four-panel DC-corrected visibility vs hour angle (Re, Im, |V|, arg).

    Parameters
    ----------
    ha_deg : array, shape (N_cap,)
        Hour-angle per capture in degrees.
    vis_dc_ch : complex array, shape (N_cap,)
        DC-corrected complex visibility at one channel.
    chip_slices : sequence of slice
        Per-chip index slices.
    channel_idx : int or None
        Channel index (for title).
    freq_ghz : float or None
        Sky frequency in GHz (for title).

    Returns
    -------
    fig, axes : (Figure, ndarray of 4 Axes)
    """
    panels = [vis_dc_ch.real, vis_dc_ch.imag,
              np.abs(vis_dc_ch), np.angle(vis_dc_ch, deg=True)]
    labels = [r"Re$(V_{\rm dc})$", r"Im$(V_{\rm dc})$",
              r"$|V_{\rm dc}|$", r"$\arg(V_{\rm dc})$ [deg]"]

    fig, _ax = textwidth_figure(17)
    _ax.remove()
    axes = subpanels(fig, 4)

    for ax, y, lbl in zip(axes, panels, labels):
        for ci, sl in enumerate(chip_slices):
            ax.scatter(
                ha_deg[sl], y[sl], s=0.15, color=CHIP_COLORS[ci],
                alpha=ALPHA_FAINT, rasterized=True, zorder=2,
                label=f"chip {ci}" if ax is axes[0] else None,
            )
        ax.set_ylabel(lbl, fontsize=TICK_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        if "Re" in lbl or "Im" in lbl:
            ax.axhline(0, color=NEUTRAL_COLOR, lw=LW_LIGHT, ls="--",
                       zorder=1)

    axes[0].legend(
        fontsize=TICK_SIZE, markerscale=8,
        ncol=len(chip_slices), loc="upper right",
    )
    title_parts = ["DC-corrected visibility"]
    if channel_idx is not None and freq_ghz is not None:
        title_parts.append(f"(ch {channel_idx}, {freq_ghz:.3f} GHz)")
    axes[0].set_title(" ".join(title_parts), fontsize=TICK_SIZE)
    axes[-1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    return fig, axes


def plot_waterfall(
    band_freqs_ghz: np.ndarray,
    ha_deg: np.ndarray,
    corr_dc: np.ndarray,
    band_mask: np.ndarray,
    *,
    ha_windows: Sequence[tuple[float, float]],
    amp_cmap: str,
    phase_cmap: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Waterfall plot: visibility amplitude and phase vs HA and frequency.

    Shows broken-axis panels for three HA windows, with amplitude on the
    left and phase on the right.

    Parameters
    ----------
    band_freqs_ghz : array, shape (N_band,)
        Sky frequency axis restricted to the analysis band.
    ha_deg : array, shape (N_cap,)
        Hour-angle per capture in degrees.
    corr_dc : complex array, shape (N_cap, N_ch)
        Full DC-corrected visibility array.
    band_mask : bool array, shape (N_ch,)
        Channel mask for the analysis band.
    ha_windows : sequence of (lo, hi)
        HA windows to display.
    amp_cmap, phase_cmap : str
        Colormaps for amplitude and phase panels.

    Returns
    -------
    fig, axes : (Figure, ndarray shape (n_windows, 2))
    """
    # Collect data per window
    panels = []
    for ha_lo, ha_hi in ha_windows:
        ha_sel = (ha_deg >= ha_lo) & (ha_deg <= ha_hi)
        if ha_sel.sum() > 0:
            panels.append((
                ha_deg[ha_sel],
                np.abs(corr_dc[ha_sel][:, band_mask]),
                np.angle(corr_dc[ha_sel][:, band_mask]),
            ))
        else:
            panels.append(None)

    vmax = np.nanpercentile(
        np.concatenate([p[1].ravel() for p in panels if p is not None]), 98,
    )
    spans = [ha_hi - ha_lo for ha_lo, ha_hi in ha_windows]
    n_panels = len(panels)

    fig, _ax = textwidth_figure(67 / 4)
    _ax.remove()
    all_axes = np.empty((n_panels, 2), dtype=object)
    gs = fig.add_gridspec(n_panels, 2, height_ratios=spans, hspace=0.07,
                          wspace=0.14)
    for row in range(n_panels):
        for col in range(2):
            share = all_axes[0, col] if row > 0 else None
            all_axes[row, col] = fig.add_subplot(gs[row, col], sharex=share)

    im_amp = im_ph = None
    for row, (panel, (ha_lo, ha_hi)) in enumerate(zip(panels, ha_windows)):
        ax_amp = all_axes[row, 0]
        ax_ph = all_axes[row, 1]

        if panel is None:
            ax_amp.set_visible(False)
            ax_ph.set_visible(False)
            continue

        ha_w, amp_w, phase_w = panel

        im_amp = ax_amp.pcolormesh(
            band_freqs_ghz, ha_w, amp_w,
            cmap=amp_cmap, shading="auto", vmin=0, vmax=vmax,
        )
        ax_amp.set_ylim(ha_hi, ha_lo)
        ax_amp.set_ylabel("HA [deg]", fontsize=LABEL_SIZE)

        im_ph = ax_ph.pcolormesh(
            band_freqs_ghz, ha_w, phase_w,
            cmap=phase_cmap, shading="auto",
            vmin=-np.pi, vmax=np.pi,
        )
        ax_ph.set_ylim(ha_hi, ha_lo)
        ax_ph.set_yticklabels([])

        # Broken-axis styling
        for ax in [ax_amp, ax_ph]:
            if row < n_panels - 1:
                ax.spines["bottom"].set_visible(False)
                ax.tick_params(bottom=False, labelbottom=False)
            if row > 0:
                ax.spines["top"].set_visible(False)

            d = 0.5
            kw = dict(transform=ax.transAxes, color="k", clip_on=False,
                      lw=LW_LIGHT * 0.8)
            if row > 0:
                ax.plot((-0.01, 0.01), (1 - d * 0.02, 1 + d * 0.02), **kw)
                ax.plot((0.99, 1.01), (1 - d * 0.02, 1 + d * 0.02), **kw)
            if row < n_panels - 1:
                ax.plot((-0.01, 0.01), (-d * 0.02, d * 0.02), **kw)
                ax.plot((0.99, 1.01), (-d * 0.02, d * 0.02), **kw)

    all_axes[-1, 0].set_xlabel("Sky frequency [GHz]", fontsize=LABEL_SIZE)
    all_axes[-1, 1].set_xlabel("Sky frequency [GHz]", fontsize=LABEL_SIZE)
    all_axes[0, 0].set_title("$|V|$", fontsize=TICK_SIZE)
    all_axes[0, 1].set_title(r"arg($V$)", fontsize=TICK_SIZE)

    if im_amp is not None:
        fig.colorbar(im_amp, ax=all_axes[:, 0], label="$|V|$",
                     orientation="horizontal", shrink=0.8, pad=0.08,
                     aspect=20)
    if im_ph is not None:
        fig.colorbar(im_ph, ax=all_axes[:, 1], label=r"arg($V$) [rad]",
                     orientation="horizontal", shrink=0.8, pad=0.08,
                     aspect=20)

    return fig, all_axes


def plot_window_adaptation(
    ha_deg: np.ndarray,
    fringe_period_s: np.ndarray,
    window_caps: np.ndarray,
    chip_slices: Sequence[slice],
    *,
    min_window: int,
    max_window: int,
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel plot of fringe period and adaptive window size vs HA.

    Parameters
    ----------
    ha_deg : array, shape (N_cap,)
        Hour-angle per capture.
    fringe_period_s : array, shape (N_cap,)
        Local fringe period in seconds.
    window_caps : array, shape (N_cap,)
        Window width in captures.
    chip_slices : sequence of slice
        Per-chip index slices.
    min_window, max_window : int
        Window clamp limits (shown as reference lines).

    Returns
    -------
    fig, axes : (Figure, ndarray of 2 Axes)
    """
    fig, _ax = columnwidth_figure(6)
    _ax.remove()
    axes = subpanels(fig, 2)

    for ci, sl in enumerate(chip_slices):
        axes[0].scatter(ha_deg[sl], fringe_period_s[sl], s=0.2,
                        color=CHIP_COLORS[ci], alpha=ALPHA_LIGHT,
                        rasterized=True, zorder=2)
        axes[1].scatter(ha_deg[sl], window_caps[sl], s=0.2,
                        color=CHIP_COLORS[ci], alpha=ALPHA_LIGHT,
                        rasterized=True, zorder=2)

    axes[0].set_ylabel("Fringe period [s]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Window [captures]", fontsize=LABEL_SIZE)
    axes[1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    axes[1].axhline(min_window, color=NEUTRAL_COLOR, lw=LW_FINE, ls=":",
                    zorder=1)
    axes[1].axhline(max_window, color=NEUTRAL_COLOR, lw=LW_FINE, ls="--",
                    zorder=1)

    return fig, axes


def plot_fringe_frequency_stft(
    ha_stft_deg: np.ndarray,
    f_stft: np.ndarray,
    power_db: np.ndarray,
    ha_pred_deg: np.ndarray,
    ff_pred_hz: np.ndarray,
    f_psd: np.ndarray,
    psd: np.ndarray,
    *,
    ff_range_hz: tuple[float, float] | None,
    f_ylim: float,
    channel_idx: int | None,
    freq_ghz: float | None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """STFT spectrogram with predicted fringe frequency, plus Welch PSD.

    Parameters
    ----------
    ha_stft_deg : array
        Hour-angle axis for the STFT columns.
    f_stft : array
        Frequency axis from STFT.
    power_db : 2-D array
        STFT power in dB, shape ``(len(f_stft), len(ha_stft_deg))``.
    ha_pred_deg : array
        HA grid for the predicted fringe frequency curve.
    ff_pred_hz : array
        Predicted |f_f(h)| at each ``ha_pred_deg``.
    f_psd : array
        Frequency axis from Welch PSD.
    psd : array
        Power spectral density from Welch.
    ff_range_hz : tuple or None
        (f_min, f_max) of the expected fringe-frequency range (shaded).
    f_ylim : float
        Upper frequency limit for both panels.
    channel_idx, freq_ghz
        Annotation for the suptitle.

    Returns
    -------
    fig, (ax_stft, ax_psd)
    """
    fig, _ax = textwidth_figure(7)
    _ax.remove()
    axes = subpanels(fig, 1, 2, width_ratios=[2, 1], sharex=False)
    ax_stft, ax_psd = axes.flat[0], axes.flat[1]

    ax_stft.pcolormesh(
        ha_stft_deg, f_stft, power_db,
        cmap="inferno", shading="auto",
    )
    ax_stft.plot(ha_pred_deg, ff_pred_hz, "w--", lw=LW_STANDARD,
                 label=r"predicted $|f_f(h)|$", zorder=3)
    ax_stft.set_ylim(0, f_ylim)
    ax_stft.set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    ax_stft.set_ylabel("Frequency [Hz]", fontsize=LABEL_SIZE)
    ax_stft.legend(fontsize=LEGEND_SIZE, loc="upper right")

    ax_psd.semilogy(f_psd, psd, "C0", lw=LW_LIGHT * 0.8, zorder=2)
    if ff_range_hz is not None:
        ax_psd.axvspan(ff_range_hz[0], ff_range_hz[1],
                       alpha=ALPHA_EXTRA_LIGHT * 0.75, color="C1",
                       label=r"expected $f_f$ range", zorder=1)
    ax_psd.set_xlabel("Frequency [Hz]", fontsize=LABEL_SIZE)
    ax_psd.set_ylabel("PSD", fontsize=LABEL_SIZE)
    ax_psd.set_xlim(0, f_ylim)
    ax_psd.legend(fontsize=LEGEND_SIZE)

    title_parts = ["Fringe-frequency sanity check"]
    if channel_idx is not None and freq_ghz is not None:
        title_parts.append(f"(ch {channel_idx}, {freq_ghz:.3f} GHz)")
    fig.suptitle(" ".join(title_parts), fontsize=TICK_SIZE, y=1.02)
    return fig, (ax_stft, ax_psd)


def plot_phase_slope(
    freqs_ghz_band: np.ndarray,
    phase_unwrapped: Sequence[np.ndarray],
    fit_coeffs: Sequence[np.ndarray],
    ha_actual_deg: Sequence[float],
    tau_ns: Sequence[float],
) -> tuple[plt.Figure, np.ndarray]:
    """Three-panel phase vs frequency at different hour angles.

    Parameters
    ----------
    freqs_ghz_band : array
        Frequency grid in GHz (analysis band, finite entries only).
    phase_unwrapped : sequence of arrays
        Unwrapped phase at each HA slice (only good channels).
    fit_coeffs : sequence of arrays
        ``np.polyfit`` coefficients (slope, intercept) per panel.
    ha_actual_deg : sequence of float
        Actual HA in degrees for each panel title.
    tau_ns : sequence of float
        Fitted geometric delay in nanoseconds per panel.

    Returns
    -------
    fig, axes : (Figure, ndarray of Axes)
    """
    n = len(phase_unwrapped)
    fig, _ax = textwidth_figure(3)
    _ax.remove()
    axes = subpanels(fig, 1, n, sharex=False, sharey=True, wspace=0.07)
    if n == 1:
        axes = np.atleast_1d(axes)

    for ax, ph, coeffs, ha, tau in zip(
        axes.flat, phase_unwrapped, fit_coeffs, ha_actual_deg, tau_ns,
    ):
        ax.scatter(freqs_ghz_band, ph, s=1, alpha=ALPHA_LIGHT,
                   color="C0", rasterized=True, zorder=2)
        f_hz = freqs_ghz_band * 1e9
        ax.plot(freqs_ghz_band, np.polyval(coeffs, f_hz), "C1",
                lw=LW_LIGHT, zorder=3,
                label=rf"$\tau$ = {tau:.1f} ns")
        ax.set_xlabel("Freq [GHz]", fontsize=LABEL_SIZE)
        ax.set_title(rf"HA = {ha:.0f}$^\circ$", fontsize=TICK_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)

    axes.flat[0].set_ylabel("Unwrapped\nphase [rad]", fontsize=LABEL_SIZE)
    return fig, axes


# ===================================================================
# Section 3: Baseline determination (notebook 04)
# ===================================================================

# Figures from notebook 04 are complex multi-step computations best
# extracted directly from notebook cell outputs.  The extraction
# helper below is shared with export_figures.py.


def extract_notebook_figures(
    nb_path: str | Path,
    cell_fig_map: dict[str, str],
    output_dir: str | Path | None = None,
) -> None:
    """Extract the first PNG output from listed notebook cells and save as PDF.

    Parameters
    ----------
    nb_path : path
        Path to the ``.ipynb`` file.
    cell_fig_map : dict
        Mapping ``{cell_id: output_filename}``.
    output_dir : path, optional
        Target directory (default: ``report/figures/``).
    """
    import base64
    import json
    from io import BytesIO

    from PIL import Image

    if output_dir is None:
        output_dir = _FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
                    if img.mode == "RGBA":
                        bg = Image.new("RGB", img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[3])
                        img = bg
                    outpath = output_dir / cell_fig_map[cid]
                    img.save(outpath, "PDF", resolution=300)
                    print(f"  {cell_fig_map[cid]}  (from cell {cid})")
                    break


# ===================================================================
# Section 4: Solar analysis (notebook 05)
# ===================================================================

# Most nb05 figures are also extracted from notebook outputs because
# they depend on iterative fit state.  The functions below cover the
# two theory/diagnostic plots that CAN be rendered standalone.


def plot_bessel_extrema(
    u_lambda: np.ndarray,
    envelope: np.ndarray,
    *,
    zero_crossings: np.ndarray | None,
    extrema_u: np.ndarray | None,
    extrema_val: np.ndarray | None,
    model_u: np.ndarray | None,
    model_env: np.ndarray | None,
    diameter_arcmin: float | None,
) -> tuple[plt.Figure, plt.Axes]:
    """Fringe amplitude envelope with Bessel zeros and extrema marked.

    Parameters
    ----------
    u_lambda : array
        Projected baseline in wavelengths.
    envelope : array
        Observed fringe amplitude envelope.
    zero_crossings : array or None
        Baseline values of Bessel zeros.
    extrema_u, extrema_val : arrays or None
        Locations and values of envelope extrema.
    model_u, model_env : arrays or None
        Model envelope curve for overlay.
    diameter_arcmin : float or None
        Fitted diameter for annotation.

    Returns
    -------
    fig, ax
    """
    fig, ax = textwidth_figure(4)

    ax.scatter(u_lambda, envelope, **SCATTER_STYLE, color="C0",
               label="Envelope", zorder=2)

    if model_u is not None and model_env is not None:
        order = np.argsort(model_u)
        ax.plot(model_u[order], model_env[order], lw=LW_STANDARD,
                color="C2", alpha=ALPHA_STANDARD,
                label="Bessel fit", zorder=3)

    if zero_crossings is not None:
        for k, u_z in enumerate(zero_crossings):
            ax.axvline(u_z, color="C3", ls="--", lw=LW_FINE,
                       alpha=ALPHA_LIGHT, zorder=1,
                       label="Null" if k == 0 else None)

    if extrema_u is not None and extrema_val is not None:
        ax.scatter(extrema_u, extrema_val, s=SS_FINE * 3,
                   color="C1", marker="D", zorder=4,
                   label="Extrema")

    if diameter_arcmin is not None:
        ax.annotate(
            rf"$\varnothing = {diameter_arcmin:.2f}$ arcmin",
            xy=(0.98, 0.92), xycoords="axes fraction",
            ha="right", fontsize=TICK_SIZE, color="C2",
        )

    ax.set_xlabel(r"Projected baseline [$\lambda$]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Amplitude envelope", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    return fig, ax


def plot_jinc_extrema(
    ha_deg: np.ndarray,
    envelope: np.ndarray,
    ha_model: np.ndarray,
    model_curve: np.ndarray,
    *,
    extrema_ha_min: np.ndarray | None,
    extrema_val_min: np.ndarray | None,
    extrema_ha_max: np.ndarray | None,
    extrema_val_max: np.ndarray | None,
    feature_windows: Sequence[tuple] | None = None,
    residuals: np.ndarray | None = None,
    diameter_arcmin: float | None,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Combined jinc fit + Bessel extrema vs hour angle.

    Parameters
    ----------
    ha_deg : array
        Hour angle of each data point (degrees).
    envelope : array
        Normalised observed envelope.
    ha_model : array
        Hour-angle grid for the model curve (degrees).
    model_curve : array
        Jinc model evaluated on *ha_model*.
    extrema_ha_min, extrema_val_min : arrays or None
        HA and normalised value of identified minima.
    extrema_ha_max, extrema_val_max : arrays or None
        HA and normalised value of identified maxima.
    feature_windows : list of (ha_lo, ha_hi, kind, root, label) or None
        HA search windows; shaded green for maxima, red for minima.
    residuals : array or None
        Fit residuals (same length as *ha_deg*). Adds a bottom panel.
    diameter_arcmin : float or None
        Fitted diameter for annotation.
    title : str
        Axes title.

    Returns
    -------
    fig, ax_or_axes
    """
    if residuals is not None:
        fig, _ax = textwidth_figure(6)
        _ax.remove()
        axes = subpanels(fig, 2, height_ratios=[3, 1])
        ax_top, ax_bot = axes
    else:
        fig, ax_top = textwidth_figure(4)

    # Shade feature windows first (behind everything)
    if feature_windows is not None:
        for ha_lo, ha_hi, kind, _root, _label in feature_windows:
            color = "C2" if kind == "max" else "C3"
            ax_top.axvspan(ha_lo, ha_hi, alpha=0.08, color=color, zorder=0)

    ax_top.scatter(ha_deg, envelope, s=0.3, alpha=ALPHA_EXTRA_LIGHT,
                   color="C0", rasterized=True, label="Data", zorder=2)

    order = np.argsort(ha_model)
    ax_top.plot(ha_model[order], model_curve[order], lw=LW_STANDARD,
                color="C1", alpha=ALPHA_STANDARD, label="Jinc fit", zorder=3)

    if extrema_ha_min is not None and extrema_val_min is not None:
        ax_top.scatter(extrema_ha_min, extrema_val_min, s=SS_FINE * 3,
                       marker="v", color="C3", zorder=5, label="Minima")

    if extrema_ha_max is not None and extrema_val_max is not None:
        ax_top.scatter(extrema_ha_max, extrema_val_max, s=SS_FINE * 3,
                       marker="^", color="C2", zorder=5, label="Maxima")

    ax_top.set_ylabel(r"$|V| / V_0$", fontsize=LABEL_SIZE)
    ax_top.set_ylim(-0.05, 1.15)
    ax_top.legend(fontsize=LEGEND_SIZE, loc="upper right")

    if residuals is not None:
        ax_bot.scatter(ha_deg, residuals, s=0.3, alpha=ALPHA_EXTRA_LIGHT,
                       color="C1", rasterized=True, zorder=2)
        zero_line(ax_bot)
        ax_bot.set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
        ax_bot.set_ylabel("Residual", fontsize=LABEL_SIZE)
        return fig, axes
    else:
        ax_top.set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
        return fig, ax_top


def plot_diameter_vs_freq(
    freq_ghz: np.ndarray,
    diameter_arcmin: np.ndarray,
    *,
    diameter_err: np.ndarray | None,
    nominal_arcmin: float | None,
    ylabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Solar diameter estimate vs frequency channel.

    Parameters
    ----------
    freq_ghz : array
        Frequency axis.
    diameter_arcmin : array
        Diameter estimate per channel.
    diameter_err : array or None
        Error bars on diameter.
    nominal_arcmin : float or None
        Nominal optical diameter for reference line.

    Returns
    -------
    fig, ax
    """
    fig, ax = textwidth_figure(3.5)

    if diameter_err is not None:
        ax.errorbar(freq_ghz, diameter_arcmin, yerr=diameter_err,
                     **ERRORBAR_STYLE, color="C0", label="Per-channel fit",
                     zorder=2)
    else:
        ax.scatter(freq_ghz, diameter_arcmin, **SCATTER_STYLE,
                   color="C0", label="Per-channel fit", zorder=2)

    if nominal_arcmin is not None:
        ax.axhline(nominal_arcmin, color=NEUTRAL_COLOR, lw=LW_LIGHT,
                   ls="--", label=f"Optical ({nominal_arcmin:.1f}')",
                   zorder=1)

    ax.set_xlabel("Sky frequency [GHz]", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    return fig, ax


def plot_jinc_fit(
    u_lambda: np.ndarray,
    vis_observed: np.ndarray,
    u_model: np.ndarray,
    vis_model: np.ndarray,
    *,
    residuals: np.ndarray | None,
    title: str,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    """Jinc (uniform-disk) fit overlay on observed visibility.

    If *residuals* is provided, a second panel is added below.

    Parameters
    ----------
    u_lambda : array
        Projected baseline for observed data.
    vis_observed : array
        Observed visibility amplitude (or signed).
    u_model : array
        Baseline grid for model curve.
    vis_model : array
        Model visibility curve.
    residuals : array or None
        Fit residuals (same length as ``u_lambda``).
    title : str
        Figure title.

    Returns
    -------
    fig, ax_or_axes
    """
    if residuals is not None:
        fig, _ax = textwidth_figure(10)
        _ax.remove()
        axes = subpanels(fig, 2, height_ratios=[3, 1])
        ax_top, ax_bot = axes

        ax_top.scatter(u_lambda, vis_observed, s=0.3,
                       alpha=ALPHA_EXTRA_LIGHT, color="C0",
                       rasterized=True, label="Data", zorder=2)
        order = np.argsort(u_model)
        ax_top.plot(u_model[order], vis_model[order], **FIT_STYLE,
                    color="C2", label="Jinc fit", zorder=3)
        ax_top.set_ylabel(r"$V / V(0)$", fontsize=LABEL_SIZE)
        ax_top.legend(fontsize=LEGEND_SIZE)
        if title:
            ax_top.set_title(title, fontsize=TICK_SIZE)

        ax_bot.scatter(u_lambda, residuals, s=0.3,
                       alpha=ALPHA_EXTRA_LIGHT, color="C1",
                       rasterized=True, zorder=2)
        zero_line(ax_bot)
        ax_bot.set_xlabel(r"Projected baseline [$\lambda$]",
                          fontsize=LABEL_SIZE)
        ax_bot.set_ylabel("Residual", fontsize=LABEL_SIZE)

        return fig, axes
    else:
        fig, ax = textwidth_figure(4)
        ax.scatter(u_lambda, vis_observed, s=0.3,
                   alpha=ALPHA_EXTRA_LIGHT, color="C0",
                   rasterized=True, label="Data", zorder=2)
        order = np.argsort(u_model)
        ax.plot(u_model[order], vis_model[order], **FIT_STYLE,
                color="C2", label="Jinc fit", zorder=3)
        ax.set_xlabel(r"Projected baseline [$\lambda$]", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"$V / V(0)$", fontsize=LABEL_SIZE)
        ax.legend(fontsize=LEGEND_SIZE)
        if title:
            ax.set_title(title, fontsize=TICK_SIZE)
        return fig, ax


def plot_lb_fit_vs_data(
    ha_deg_raw: np.ndarray,
    raw_envelope: np.ndarray,
    ha_deg_interp: np.ndarray,
    interp_envelope: np.ndarray,
    fit_curve: np.ndarray,
    *,
    fit_label: str,
    title: str,
    ylim: tuple[float, float],
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel comparison: data + fit overlay, and residuals.

    Parameters
    ----------
    ha_deg_raw : array
        Hour-angle axis for raw scatter data.
    raw_envelope : array
        Raw (normalised) fringe amplitude for scatter.
    ha_deg_interp : array
        Hour-angle axis for interpolated + fit curves.
    interp_envelope : array
        Interpolated (normalised) fringe amplitude.
    fit_curve : array
        Best-fit model curve (same grid as *ha_deg_interp*).
    fit_label : str
        Legend label for the fit curve (includes parameter values).
    title : str
        Figure title.
    ylim : tuple
        Y-axis limits for the top panel.

    Returns
    -------
    fig, axes : (Figure, ndarray of 2 Axes)
    """
    residual = interp_envelope - fit_curve

    fig, _ax = textwidth_figure(11)
    _ax.remove()
    axes = subpanels(fig, 2, height_ratios=[3, 1])

    finite = np.isfinite(raw_envelope) & np.isfinite(ha_deg_raw)
    axes[0].scatter(ha_deg_raw[finite], raw_envelope[finite],
                    s=0.3, alpha=0.05, color="C0", rasterized=True,
                    label=r"raw $|V|$", zorder=2)
    axes[0].plot(ha_deg_interp, interp_envelope, lw=LW_FINE, color="C0",
                 alpha=ALPHA_FAINT, label="interpolated", zorder=2)
    axes[0].plot(ha_deg_interp, fit_curve, lw=LW_STANDARD, color="C1",
                 label=fit_label, zorder=3)
    axes[0].set_ylabel(r"Normalised $|V|$", fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=TICK_SIZE, loc="upper right")
    axes[0].set_ylim(ylim)

    axes[1].plot(ha_deg_interp, residual, lw=LW_FINE, color="C0",
                 alpha=ALPHA_LIGHT, zorder=2)
    axes[1].axhline(0, color="0.4", lw=LW_LIGHT, ls="--", zorder=1)
    axes[1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Residual", fontsize=LABEL_SIZE)

    if title:
        axes[0].set_title(title, fontsize=TICK_SIZE)

    return fig, axes


def plot_fit_params_vs_freq(
    freq_ghz: np.ndarray,
    params: dict[str, np.ndarray],
    *,
    param_errs: dict[str, np.ndarray] | None,
    labels: dict[str, str] | None,
) -> tuple[plt.Figure, np.ndarray]:
    """Multi-panel plot of fit parameters vs frequency.

    Parameters
    ----------
    freq_ghz : array
        Frequency axis.
    params : dict
        ``{name: values}`` for each parameter to plot.
    param_errs : dict or None
        ``{name: uncertainties}`` for error bars.
    labels : dict or None
        ``{name: y_label}`` overrides.

    Returns
    -------
    fig, axes
    """
    n = len(params)
    fig, _ax = textwidth_figure(max(4, round(n * 4)))
    _ax.remove()
    axes = subpanels(fig, n)
    if n == 1:
        axes = np.atleast_1d(axes)

    if labels is None:
        labels = {}
    if param_errs is None:
        param_errs = {}

    for ax, (name, vals) in zip(axes.flat, params.items()):
        err = param_errs.get(name)
        if err is not None:
            ax.errorbar(freq_ghz, vals, yerr=err,
                        **ERRORBAR_STYLE, color="C0", zorder=2)
        else:
            ax.scatter(freq_ghz, vals, **SCATTER_STYLE, color="C0", zorder=2)
        ax.set_ylabel(labels.get(name, name), fontsize=LABEL_SIZE)

    axes.flat[-1].set_xlabel("Sky frequency [GHz]", fontsize=LABEL_SIZE)
    return fig, axes


def plot_eps_f_correlation(
    freq_ghz: np.ndarray,
    epsilon: np.ndarray,
    f_sunspot: np.ndarray,
    *,
    epsilon_err: np.ndarray | None,
    f_err: np.ndarray | None,
) -> tuple[plt.Figure, plt.Axes]:
    """Sunspot flux fraction vs limb-brightening parameter correlation.

    Parameters
    ----------
    freq_ghz : array
        Frequency per data point (used for colour).
    epsilon : array
        Limb-brightening parameter.
    f_sunspot : array
        Sunspot flux fraction.
    epsilon_err, f_err : arrays or None
        Error bars.

    Returns
    -------
    fig, ax
    """
    fig, ax = columnwidth_figure(5)

    sc = ax.scatter(epsilon, f_sunspot, c=freq_ghz, cmap="viridis",
                    s=SS_FINE, alpha=ALPHA_LIGHT, edgecolors="none",
                    zorder=3)
    if epsilon_err is not None and f_err is not None:
        ax.errorbar(epsilon, f_sunspot, xerr=epsilon_err, yerr=f_err,
                    fmt="none", ecolor=NEUTRAL_COLOR, elinewidth=LW_FINE,
                    alpha=ALPHA_FAINT, zorder=2)

    plt.colorbar(sc, ax=ax, label="Frequency [GHz]")
    ax.set_xlabel(r"$\epsilon$ (limb brightening)", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$f_{\rm spot}$", fontsize=LABEL_SIZE)
    return fig, ax


# ===================================================================
# Section 5: Fringe fit quality (notebook 04)
# ===================================================================


def plot_fringe_fit(
    ha_deg: np.ndarray,
    vis_re: np.ndarray,
    model_re: np.ndarray,
    *,
    rms: float,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel fringe-fit comparison: data + model overlay, residuals.

    Parameters
    ----------
    ha_deg : array
        Hour-angle axis in degrees.
    vis_re : array
        Observed Re(V) data.
    model_re : array
        Best-fit model Re(V).
    rms : float
        RMS of the residuals (for annotation).
    title : str
        Legend label for the model curve.

    Returns
    -------
    fig, axes : (Figure, ndarray of 2 Axes)
    """
    residual = vis_re - model_re

    fig, _ax = columnwidth_figure(5)
    _ax.remove()
    axes = subpanels(fig, 2, height_ratios=[3, 1])

    axes[0].scatter(ha_deg, vis_re, s=SS_FINE, alpha=ALPHA_EXTRA_LIGHT,
                    color="C0", rasterized=True, zorder=2)
    axes[0].plot(ha_deg, model_re, lw=LW_LIGHT, color="C1",
                 label=title, zorder=3)
    axes[0].set_ylabel(r"Re$(V)$", fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=TICK_SIZE, loc="upper right")

    axes[1].scatter(ha_deg, residual, s=SS_FINE, alpha=ALPHA_EXTRA_LIGHT,
                    color="C0", rasterized=True, zorder=2)
    axes[1].axhline(0, color="k", lw=LW_LIGHT, ls="--", zorder=1)
    axes[1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Residual", fontsize=LABEL_SIZE)

    return fig, axes


# ===================================================================
# Section 6: Brute-force baseline search (notebook 04, diagnostic)
# ===================================================================


def plot_brute_1d(
    b_ew_grid: np.ndarray,
    cost: np.ndarray,
    *,
    b_ew_best: float | None,
    xlabel: str,
    ylabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    """1-D brute-force baseline cost curve.

    Parameters
    ----------
    b_ew_grid : array
        East-west baseline grid values.
    cost : array
        Cost (or chi-squared) at each grid point.
    b_ew_best : float or None
        Best-fit value to mark.

    Returns
    -------
    fig, ax
    """
    fig, ax = textwidth_figure(3)
    ax.plot(b_ew_grid, cost, lw=LW_LIGHT, color="C0", zorder=2)

    if b_ew_best is not None:
        ax.axvline(b_ew_best, **GUIDE_STYLE, label=f"Best: {b_ew_best:.3f} m",
                   zorder=1)
        ax.legend(fontsize=LEGEND_SIZE)

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    return fig, ax


def plot_brute_2d(
    b_ew_full: np.ndarray,
    b_ns_full: np.ndarray,
    chi2_nu_full: np.ndarray,
    chi2_nu_min_full: float,
    b_ew_win: np.ndarray,
    b_ns_win: np.ndarray,
    chi2_nu_win: np.ndarray,
    chi2_nu_min_win: float,
    *,
    sigma_factors: dict[float, str],
) -> tuple[plt.Figure, np.ndarray]:
    """Stacked reduced-χ² grid search: full track (top) and windowed (bottom).

    Parameters
    ----------
    b_ew_full, b_ns_full : arrays
        Baseline grid axes for the full-track search (metres).
    chi2_nu_full : 2-D array
        Reduced χ² surface for the full-track search.
    chi2_nu_min_full : float
        Minimum reduced χ² of the full-track search.
    b_ew_win, b_ns_win : arrays
        Baseline grid axes for the windowed search (metres).
    chi2_nu_win : 2-D array
        Reduced χ² surface for the windowed search.
    chi2_nu_min_win : float
        Minimum reduced χ² of the windowed search.
    sigma_factors : dict
        Mapping ``{multiplicative_factor: label}`` for confidence
        contours, e.g. ``{2.30: r"1$\\sigma$", 6.17: r"2$\\sigma$"}``.

    Returns
    -------
    fig, axes : (Figure, ndarray of 2 Axes)
    """
    fig, subfigs = columnwidth_figure(19 / 2, subfigures=(2, 1))
    subfigs = np.atleast_1d(subfigs)

    axes = np.empty(2, dtype=object)

    for i, (sf, b_ew, b_ns, chi2, chi2_min) in enumerate([
        (subfigs[0], b_ew_full, b_ns_full, chi2_nu_full, chi2_nu_min_full),
        (subfigs[1], b_ew_win, b_ns_win, chi2_nu_win, chi2_nu_min_win),
    ]):
        ax = sf.subplots()
        axes[i] = ax

        chi2_finite = chi2[np.isfinite(chi2)]
        vmax = np.nanpercentile(chi2_finite, 95) if chi2_finite.size else 10.0

        # Downsample to 256×256 for faster rendering
        MAX_PIX = 256
        step_ew = max(1, len(b_ew) // MAX_PIX)
        step_ns = max(1, len(b_ns) // MAX_PIX)
        b_ew_ds = b_ew[::step_ew]
        b_ns_ds = b_ns[::step_ns]
        chi2_ds = chi2[::step_ns, ::step_ew]

        im = ax.pcolormesh(
            b_ew_ds, b_ns_ds, chi2_ds,
            cmap="inferno_r", shading="auto",
            vmin=chi2_min, vmax=vmax,
        )

        # Sigma contours (on full-res data for accuracy)
        levels = [f * chi2_min for f in sigma_factors]
        chi2_filled = np.where(np.isfinite(chi2), chi2,
                               np.nanmax(chi2_finite))
        cs = ax.contour(
            b_ew, b_ns, chi2_filled,
            levels=levels, colors=["white", "white"],
            linewidths=[LW_STANDARD, LW_LIGHT],
            linestyles=["solid", "dashed"],
        )
        ax.clabel(cs, fmt=dict(zip(levels, sigma_factors.values())),
                  fontsize=TICK_SIZE)

        ax.set_ylabel(r"$b_{\rm NS}$ [m]", fontsize=LABEL_SIZE)
        if i == 1:
            ax.set_xlabel(r"$b_{\rm EW}$ [m]", fontsize=LABEL_SIZE)

        sf.colorbar(im, ax=ax, orientation="vertical",
                    label=r"$\chi^2_\nu$", shrink=0.8, pad=0.03)

    return fig, axes


# ===================================================================
# Section 6: Lag / delay diagnostics (notebook 04)
# ===================================================================


def plot_lag_delay_vs_ha(
    ha_deg: np.ndarray,
    tau_disp_ns: np.ndarray,
    lag_power_db: np.ndarray,
    tau_model_ns: np.ndarray,
    resid_kept_ns: np.ndarray,
    ha_kept_deg: np.ndarray,
    *,
    ha_clipped_deg: np.ndarray | None,
    resid_clipped_ns: np.ndarray | None,
    n_kept: int,
    n_clipped: int,
    rms_ns: float,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel lag-delay figure: spectrogram + residuals.

    Parameters
    ----------
    ha_deg : array
        Hour-angle axis for the lag spectrogram columns.
    tau_disp_ns : array
        Delay axis in nanoseconds (rows of lag spectrogram).
    lag_power_db : 2-D array
        Lag spectrogram power in dB, shape ``(len(ha_deg), len(tau_disp_ns))``.
    tau_model_ns : array
        Fitted model delay curve, same length as *ha_deg*.
    resid_kept_ns : array
        Residuals (measured − model) for kept points.
    ha_kept_deg : array
        HA values for kept points.
    ha_clipped_deg : array or None
        HA values for sigma-clipped (rejected) points.
    resid_clipped_ns : array or None
        Residuals for sigma-clipped points.
    n_kept, n_clipped : int
        Counts for the legend.
    rms_ns : float
        RMS of kept residuals (for 3σ band).
    title : str
        Figure title.

    Returns
    -------
    fig, axes : (Figure, ndarray of 2 Axes)
    """
    fig, _ax = textwidth_figure(12)
    _ax.remove()
    axes = subpanels(fig, 2, height_ratios=[4.5, 1])

    # Top: lag spectrogram with model overlay
    pm = axes[0].pcolormesh(
        ha_deg, tau_disp_ns, lag_power_db.T,
        shading="auto", cmap="magma",
    )
    axes[0].plot(ha_deg, tau_model_ns, color="cyan", lw=LW_STANDARD,
                 label=title, zorder=3)
    axes[0].set_ylabel(r"Delay $\tau$ [ns]", fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=TICK_SIZE, loc="upper right")

    # Colorbar above spectrogram
    fig.colorbar(pm, ax=axes[0], orientation="horizontal",
                 label="power [dB, arb.]", location="top",
                 shrink=0.8, pad=0.02, aspect=25)

    # Bottom: residuals
    if ha_clipped_deg is not None and resid_clipped_ns is not None and n_clipped > 0:
        axes[1].scatter(ha_clipped_deg, resid_clipped_ns, s=SS_FINE,
                        marker="x", color="C3", alpha=ALPHA_FAINT, zorder=2,
                        label=f"Clipped ({n_clipped})")
    axes[1].scatter(ha_kept_deg, resid_kept_ns, s=SS_FINE,
                    color="C0", alpha=ALPHA_FAINT, rasterized=True, zorder=2,
                    label=f"Kept ({n_kept})")
    axes[1].axhline(0, color="k", lw=LW_LIGHT, ls="--", zorder=1)
    axes[1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Residual [ns]", fontsize=LABEL_SIZE)
    axes[1].legend(fontsize=TICK_SIZE, ncol=3)

    ha_xlim = (ha_deg.min(), ha_deg.max())
    axes[0].set_xlim(ha_xlim)
    axes[1].set_xlim(ha_xlim)

    return fig, axes


def plot_stft_baseline_fit(
    ha_stft_deg: np.ndarray,
    f_stft_mhz: np.ndarray,
    power_db: np.ndarray,
    ha_model_deg: np.ndarray,
    ff_model_mhz: np.ndarray,
    resid_kept_mhz: np.ndarray,
    ha_kept_deg: np.ndarray,
    *,
    ha_clipped_deg: np.ndarray | None,
    resid_clipped_mhz: np.ndarray | None,
    n_kept: int,
    n_clipped: int,
    rms_mhz: float,
    f_ylim_mhz: float,
    title: str,
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel STFT baseline fit: spectrogram + residuals.

    Parameters
    ----------
    ha_stft_deg : array
        HA axis for STFT spectrogram columns.
    f_stft_mhz : array
        Frequency axis in mHz.
    power_db : 2-D array
        STFT power in dB, shape ``(len(f_stft_mhz), len(ha_stft_deg))``.
    ha_model_deg : array
        HA grid for the smooth fitted-model curve.
    ff_model_mhz : array
        Fitted fringe-frequency model in mHz.
    resid_kept_mhz : array
        Residuals (measured − model) for kept windows in mHz.
    ha_kept_deg : array
        HA values for kept windows.
    ha_clipped_deg : array or None
        HA values for sigma-clipped windows.
    resid_clipped_mhz : array or None
        Residuals for sigma-clipped windows in mHz.
    n_kept, n_clipped : int
        Counts for the legend.
    rms_mhz : float
        RMS of kept residuals (for 3σ band).
    f_ylim_mhz : float
        Upper frequency limit in mHz.
    title : str
        Legend label for the model curve (includes baseline values).

    Returns
    -------
    fig, axes : (Figure, ndarray of 2 Axes)
    """
    fig, _ax = textwidth_figure(12)
    _ax.remove()
    axes = subpanels(fig, 2, height_ratios=[4.5, 1])

    # Top: STFT spectrogram with model overlay
    pm = axes[0].pcolormesh(
        ha_stft_deg, f_stft_mhz, power_db,
        shading="auto", cmap="magma",
    )
    axes[0].plot(ha_model_deg, ff_model_mhz, color="cyan", lw=LW_STANDARD,
                 label=title, zorder=3)
    axes[0].set_ylim(0.0, f_ylim_mhz)
    axes[0].set_ylabel("Fringe frequency [mHz]", fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=TICK_SIZE, loc="upper right")

    # Colorbar above spectrogram
    fig.colorbar(pm, ax=axes[0], orientation="horizontal",
                 label="power [dB, arb.]", location="top",
                 shrink=0.8, pad=0.02, aspect=25)

    # Bottom: residuals
    axes[1].scatter(ha_kept_deg, resid_kept_mhz,
                    s=SS_FINE, alpha=ALPHA_LIGHT, color="C0", zorder=2,
                    label=f"Kept ({n_kept})")
    axes[1].axhline(0, color="k", lw=LW_LIGHT, ls="--", zorder=1)

    # Clamp y-range to kept data before adding clipped outliers
    kept_ypad = 4.0 * rms_mhz
    axes[1].set_ylim(-kept_ypad, kept_ypad)

    if ha_clipped_deg is not None and resid_clipped_mhz is not None and n_clipped > 0:
        clipped_y = np.clip(resid_clipped_mhz, -kept_ypad, kept_ypad)
        axes[1].scatter(ha_clipped_deg, clipped_y,
                        marker="x", s=SS_FINE * 2, color="C3",
                        alpha=ALPHA_STANDARD, zorder=4,
                        label=f"Clipped ({n_clipped})")

    axes[1].set_xlabel("Hour angle [deg]", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Residual [mHz]", fontsize=LABEL_SIZE)
    axes[1].legend(fontsize=TICK_SIZE, ncol=3)

    ha_xlim = (ha_stft_deg.min(), ha_stft_deg.max())
    axes[0].set_xlim(ha_xlim)
    axes[1].set_xlim(ha_xlim)

    return fig, axes


# ===================================================================
# Section 7: Literature comparison (notebook 05)
# ===================================================================

# Menezes et al. 2021, ApJ 910, 77, Table 1 — literature compilation
# (frequency_GHz, radius_arcsec, radius_err_arcsec)
_LITERATURE_TABLE = [
    ( 3,   1070,  17),
    ( 5,   1020,   9),
    ( 9,    989,   2),
    (11,    991,   5),
    (13,    989,   2),
    (16,    990,   4),
    (17,  976.6, 1.5),
    (22,  981.7, 0.8),
    (25,    979,   4),
    (30,    979,   4),
    (35,    979,   3),
    (37,    979,   5),
    (44,  978.1, 1.3),
    (48,  983.6, 1.9),
    (48,  973.1, 2.9),
    (70,    969,   5),
    (74,    967,   4),
    (94,    972,   5),
    (100, 964.1, 4.5),
    (100,   966,   1),
    (100, 965.9, 3.2),
    (115, 969.3, 1.6),
    (212, 966.5, 2.8),
    (230, 961.1, 2.5),
    (230, 961.6, 2.1),
    (231, 968.2, 1.0),
    (405, 966.5, 2.7),
]

# Menezes+ 2021 own measurements (Table 2, equatorial mean)
_MENEZES2021_TABLE = [
    (100, 968,   3),
    (212, 963,   4),
    (230, 963.7, 1.8),
    (405, 963,   5),
]

# Marongiu+ 2024 (18–26 GHz, approximate from their paper)
_MARONGIU2024_TABLE = [
    (18, 983, 3),
    (26, 980, 3),
]


def _radii_to_diameters(table):
    """Convert (freq, R_arcsec, err_arcsec) tuples to diameter in arcmin."""
    f = np.array([e[0] for e in table], dtype=float)
    d = np.array([2 * e[1] / 60 for e in table])
    e = np.array([2 * e[2] / 60 for e in table])
    return f, d, e


def plot_diameter_vs_freq_literature(
    this_work: Sequence[tuple[float, float, float, str]],
    *,
    optical_arcmin: float = 31.97,
) -> tuple[plt.Figure, plt.Axes]:
    """Solar diameter vs radio frequency with literature comparison.

    Parameters
    ----------
    this_work : sequence of (freq_GHz, D_arcmin, D_err_arcmin, label)
        Measurements from this work. Plotted with C0, C1, … in order.
    optical_arcmin : float
        Optical photospheric diameter reference line.

    Returns
    -------
    fig, ax
    """
    lit_f, lit_d, lit_e = _radii_to_diameters(
        _LITERATURE_TABLE + _MENEZES2021_TABLE + _MARONGIU2024_TABLE
    )

    fig, ax = columnwidth_figure(5)

    # Literature
    ax.errorbar(
        lit_f, lit_d, yerr=lit_e,
        fmt="x", color="k", ms=MS_FINE, lw=LW_FINE,
        capsize=1.5, capthick=LW_FINE,
        label="Literature", zorder=2,
    )

    # This work
    colors = ["C0", "C1", "C2", "C3"]
    for i, (freq, diam, err, label) in enumerate(this_work):
        ax.errorbar(
            freq, diam, yerr=err,
            fmt="x", color=colors[i % len(colors)],
            ms=MS_STANDARD, lw=LW_STANDARD,
            capsize=2.5, capthick=LW_LIGHT,
            label=label, zorder=4,
        )

    # Optical reference
    ax.axhline(
        optical_arcmin, color="C2", lw=LW_LIGHT, ls="--",
        alpha=ALPHA_LIGHT,
        label="Photosphere",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Frequency [GHz]", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Solar diameter [$'$]", fontsize=LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    ax.set_xlim(2, 500)

    return fig, ax
