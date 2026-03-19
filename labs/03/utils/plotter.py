"""Shared plotting helpers for the Lab 03 Sun notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .plotting import (
    ALPHA_BAR,
    ALPHA_FILL,
    ALPHA_SHADE_LIGHT,
    ALPHA_SHADE_STANDARD,
    ALPHA_SPAN_LIGHT,
    BAR_HEIGHT_STANDARD,
    ERRORBAR_CAPSIZE_SMALL,
    LW_FINE,
    LW_GUIDE,
    LW_HAIRLINE,
    MARKER_MS_SMALL,
    NEUTRAL_COLOR,
    PRIMARY_COLOR,
    SCATTER_S_FINE,
    SECONDARY_COLOR,
    TEXTWIDTH_IN,
    TICK_SIZE,
    WATERFALL_HA_MIN_PER_IN,
    WATERFALL_PANEL_HEIGHT_IN,
    _single_panel,
    _stacked_panels,
    _tight_layout,
    _zero_line,
)

CHIP_COLORS = (PRIMARY_COLOR, SECONDARY_COLOR)


def _ha_fmt(deg: float, pos: float | None) -> str:
    del pos
    sign = "-" if deg < 0 else "+"
    hour_abs = abs(deg) / 15.0
    hh = int(hour_abs)
    mm = round((hour_abs - hh) * 60)
    if mm == 60:
        hh += 1
        mm = 0
    return rf"${sign}{hh}^h\,{mm:02d}^m$"


def _ha_formatter() -> mticker.FuncFormatter:
    return mticker.FuncFormatter(_ha_fmt)


def _channel_secondary_xaxis(ax: Axes, f_rf0_hz: float, df_hz: float):
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda ghz: (ghz * 1e9 - f_rf0_hz) / df_hz,
            lambda chan: (chan * df_hz + f_rf0_hz) / 1e9,
        ),
    )
    ax_top.set_xlabel("Channel")
    return ax_top


def _hour_angle_degree_secondary_xaxis(ax: Axes):
    ax_top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    ax_top.set_xlabel("Hour angle [deg]")
    return ax_top


def _hour_angle_degree_secondary_yaxis(ax: Axes):
    ax_right = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    ax_right.set_ylabel("Hour angle [deg]")
    ax_right.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: rf"${y:.1f}^\circ$"))
    return ax_right


def _channel_secondary_yaxis(ax: Axes, f_rf0_hz: float, df_hz: float):
    ax_right = ax.secondary_yaxis(
        "right",
        functions=(
            lambda ghz: (ghz * 1e9 - f_rf0_hz) / df_hz,
            lambda chan: (chan * df_hz + f_rf0_hz) / 1e9,
        ),
    )
    ax_right.set_ylabel("Channel")
    return ax_right


def _waterfall_row_span_deg() -> float:
    return TEXTWIDTH_IN * WATERFALL_HA_MIN_PER_IN / 4.0


def _plot_waterfall_panels(
    f_sky_ghz: np.ndarray,
    ha_deg: np.ndarray,
    quantity_matrix: np.ndarray,
    plot_band_ghz: tuple[float, float],
    f_rf0_hz: float,
    df_hz: float,
    title: str,
    cbar_label: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[Figure, np.ndarray]:
    order = np.argsort(ha_deg)
    ha_sorted = np.asarray(ha_deg)[order]
    quantity_sorted = np.asarray(quantity_matrix)[order, :]

    row_span_deg = _waterfall_row_span_deg()
    total_span_deg = ha_sorted.max() - ha_sorted.min()
    nrows = max(1, int(np.ceil(total_span_deg / row_span_deg)))

    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(TEXTWIDTH_IN, nrows * WATERFALL_PANEL_HEIGHT_IN + 0.6),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.ravel()

    image = None
    ha_start_deg = ha_sorted.min()
    for row_idx, ax in enumerate(axes):
        row_min_deg = ha_start_deg + row_idx * row_span_deg
        row_max_deg = row_min_deg + row_span_deg
        if row_idx == nrows - 1:
            mask = (ha_sorted >= row_min_deg) & (ha_sorted <= row_max_deg)
        else:
            mask = (ha_sorted >= row_min_deg) & (ha_sorted < row_max_deg)

        if np.any(mask):
            image = ax.pcolormesh(
                ha_sorted[mask],
                f_sky_ghz,
                quantity_sorted[mask, :].T,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap_name,
                shading="auto",
            )

        ax.set_xlim(row_min_deg, row_max_deg)
        ax.set_ylim(*plot_band_ghz)
        ax.set_xlabel("Hour angle")
        ax.set_ylabel(r"$f_{\rm sky}$ [GHz]")
        ax.xaxis.set_major_formatter(_ha_formatter())

        ax_top = _hour_angle_degree_secondary_xaxis(ax)
        ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: rf"${x:.1f}^\circ$"))
        _channel_secondary_yaxis(ax, f_rf0_hz, df_hz)

        if row_idx == 0:
            ax.set_title(title, fontsize=TICK_SIZE)

    if image is None:
        raise ValueError("Waterfall input contains no finite rows to plot.")

    fig.colorbar(image, ax=axes.tolist(), label=cbar_label, location="bottom")
    return fig, axes


def plot_example_spectrum(
    f_sky_ghz: np.ndarray,
    amp_norm: np.ndarray,
    plot_band_ghz: tuple[float, float],
    peak_freq_ghz: float,
    f_rf0_hz: float,
    df_hz: float,
    capture_index: int,
    capture_count: int,
    integration_s: float,
    n_acc: int,
    ha_deg: float,
    alt_deg: float,
) -> tuple[Figure, Axes]:
    fig, ax = _single_panel((TEXTWIDTH_IN, 3))
    ax.plot(f_sky_ghz, amp_norm, lw=LW_FINE, color=PRIMARY_COLOR)
    ax.axvspan(*plot_band_ghz, color=NEUTRAL_COLOR, alpha=ALPHA_SHADE_LIGHT, label="plot band")
    ax.axvline(
        peak_freq_ghz,
        color=NEUTRAL_COLOR,
        lw=LW_GUIDE,
        ls="--",
        label=rf"peak  $f={peak_freq_ghz:.4f}$ GHz",
    )
    ax.set_xlabel(r"$f_{\rm sky}$ [GHz]")
    ax.set_ylabel(r"$|V_{12}| / |V_{12}|_{\rm peak}$")
    ax.set_title(
        rf"Sun --- example spectrum (capture {capture_index}/{capture_count - 1})"
        "\n"
        rf"$\Delta t={integration_s:.2f}$ s,  $N_{{\rm acc}}={n_acc}$,  "
        rf"HA $={ha_deg:.2f}^\circ$,  alt $={alt_deg:.1f}^\circ$",
        fontsize=TICK_SIZE,
    )
    _channel_secondary_xaxis(ax, f_rf0_hz, df_hz)
    ax.set_xlim(f_sky_ghz[0], f_sky_ghz[-1])
    ax.set_ylim(0, None)
    ax.legend(fontsize=TICK_SIZE)
    _tight_layout(fig)
    return fig, ax


def plot_waterfall_amplitude(
    f_sky_ghz: np.ndarray,
    ha_deg: np.ndarray,
    amp_matrix: np.ndarray,
    plot_band_ghz: tuple[float, float],
    f_rf0_hz: float,
    df_hz: float,
    title: str,
    cbar_label: str,
) -> tuple[Figure, np.ndarray]:
    return _plot_waterfall_panels(
        f_sky_ghz=f_sky_ghz,
        ha_deg=ha_deg,
        quantity_matrix=amp_matrix,
        plot_band_ghz=plot_band_ghz,
        f_rf0_hz=f_rf0_hz,
        df_hz=df_hz,
        title=title,
        cbar_label=cbar_label,
        cmap_name="viridis",
        vmin=0,
        vmax=1,
    )


def plot_visibility_waterfall(
    f_sky_ghz: np.ndarray,
    ha_deg: np.ndarray,
    quantity_matrix: np.ndarray,
    plot_band_ghz: tuple[float, float],
    f_rf0_hz: float,
    df_hz: float,
    title: str,
    cbar_label: str,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[Figure, np.ndarray]:
    return _plot_waterfall_panels(
        f_sky_ghz=f_sky_ghz,
        ha_deg=ha_deg,
        quantity_matrix=quantity_matrix,
        plot_band_ghz=plot_band_ghz,
        f_rf0_hz=f_rf0_hz,
        df_hz=df_hz,
        title=title,
        cbar_label=cbar_label,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
    )


def plot_capture_timeline_and_gaps(
    capture_start_s: np.ndarray,
    capture_end_s: np.ndarray,
    capture_count: int,
    duty_cycle_pct: float,
    gap_s: np.ndarray,
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 5))

    ax = axes[0]
    for idx, (start_s, end_s) in enumerate(zip(capture_start_s, capture_end_s)):
        ax.barh(
            idx,
            end_s - start_s,
            left=start_s,
            height=BAR_HEIGHT_STANDARD,
            color=PRIMARY_COLOR,
            alpha=ALPHA_BAR,
        )
    for end_s, next_start_s in zip(capture_end_s[:-1], capture_start_s[1:]):
        ax.barh(
            0,
            next_start_s - end_s,
            left=end_s,
            height=capture_count,
            color=NEUTRAL_COLOR,
            alpha=ALPHA_SHADE_STANDARD,
            zorder=0,
        )
    ax.set_xlabel("Time since first capture [s]")
    ax.set_ylabel("Capture index")
    ax.set_title(rf"Capture timeline -- {capture_count} captures, duty cycle {duty_cycle_pct:.1f}\%")
    ax.set_ylim(-0.5, capture_count - 0.5)

    ax = axes[1]
    ax.hist(gap_s, bins=20, color=PRIMARY_COLOR, edgecolor="white", linewidth=LW_HAIRLINE)
    ax.axvline(gap_s.mean(), color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--", label=rf"mean {gap_s.mean():.2f} s")
    ax.axvline(np.median(gap_s), color=NEUTRAL_COLOR, lw=LW_GUIDE, ls=":", label=rf"median {np.median(gap_s):.2f} s")
    ax.set_xlabel("Inter-capture gap [s]")
    ax.set_ylabel("Count")
    ax.set_title("Gap distribution (slew + overhead)")
    ax.legend(fontsize=TICK_SIZE)

    _tight_layout(fig)
    return fig, axes


def plot_channel_time_series(
    ha_chips: list[np.ndarray],
    amp_norm_chips: list[np.ndarray],
    phase_deg_chips: list[np.ndarray],
    real_chips: list[np.ndarray],
    imag_chips: list[np.ndarray],
    channel_index: int,
    channel_freq_ghz: float,
    ha_limits_deg: tuple[float, float],
) -> tuple[Figure, np.ndarray]:
    fig, axes = _stacked_panels(4, (TEXTWIDTH_IN, 8), (1, 1, 1, 1), 0.0)

    for chip_idx, ha_deg in enumerate(ha_chips):
        axes[0].plot(ha_deg, amp_norm_chips[chip_idx], lw=LW_FINE, color=CHIP_COLORS[chip_idx], label=f"chip {chip_idx}")
        axes[1].plot(ha_deg, phase_deg_chips[chip_idx], lw=LW_FINE, color=CHIP_COLORS[chip_idx])
        axes[2].plot(ha_deg, real_chips[chip_idx], lw=LW_FINE, color=CHIP_COLORS[chip_idx])
        axes[3].plot(ha_deg, imag_chips[chip_idx], lw=LW_FINE, color=CHIP_COLORS[chip_idx])

    axes[0].axhline(1.0, color=NEUTRAL_COLOR, lw=LW_GUIDE, ls="--")
    axes[0].set_ylabel(r"$|V_{12}|\,/\,\langle|V_{12}|\rangle_{\rm global}$")
    axes[0].legend(fontsize=TICK_SIZE)

    axes[1].set_ylim(-180, 180)
    axes[1].set_ylabel(r"$\arg(V_{12})$ [deg]")

    _zero_line(axes[2])
    axes[2].set_ylabel(r"$\mathrm{Re}(V_{12})$")

    _zero_line(axes[3])
    axes[3].set_ylabel(r"$\mathrm{Im}(V_{12})$")
    axes[3].set_xlabel("Hour angle")
    axes[3].xaxis.set_major_formatter(_ha_formatter())

    for ax in axes:
        ax.set_xlim(*ha_limits_deg)

    fig.suptitle(
        rf"Sun --- channel $k={channel_index}$,  $f_{{\rm sky}}={channel_freq_ghz:.4f}$ GHz",
        fontsize=TICK_SIZE,
    )
    _tight_layout(fig)
    return fig, axes


def plot_unwrapped_phase_vs_ha_time(
    ha_time_s_chips: list[np.ndarray],
    phase_deg_chips: list[np.ndarray],
    sidereal_rate_deg_s: float,
    channel_index: int,
    channel_freq_ghz: float,
    ha_time_limits_s: tuple[float, float],
) -> tuple[Figure, Axes]:
    fig, ax = _single_panel((TEXTWIDTH_IN, 3))
    for chip_idx, ha_time_s in enumerate(ha_time_s_chips):
        ax.plot(
            ha_time_s,
            phase_deg_chips[chip_idx],
            lw=LW_FINE,
            color=CHIP_COLORS[chip_idx],
            label=f"chip {chip_idx}",
        )
    ax.set_xlabel("Hour angle [s]")
    ax.set_ylabel(r"$\arg(V_{12} - \langle V_{12}\rangle)$ [deg, unwrapped]")
    ax.set_xlim(*ha_time_limits_s)
    ax.legend(fontsize=TICK_SIZE)
    ax_top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda seconds: seconds * sidereal_rate_deg_s,
            lambda degrees: degrees / sidereal_rate_deg_s,
        ),
    )
    ax_top.set_xlabel("Hour angle [deg]")
    ax.set_title(
        rf"Sun --- unwrapped phase per chip, $k={channel_index}$  ($f_{{\rm sky}}={channel_freq_ghz:.4f}$ GHz)",
        fontsize=TICK_SIZE,
    )
    _tight_layout(fig)
    return fig, ax


def plot_fringe_rate_vs_frequency(
    f_sky_ghz: np.ndarray,
    dphi_dt_chips: list[np.ndarray],
    dphi_dt_err_chips: list[np.ndarray],
    plot_band_ghz: tuple[float, float],
    f_rf0_hz: float,
    df_hz: float,
) -> tuple[Figure, Axes]:
    fig, ax = _single_panel((TEXTWIDTH_IN, 3))
    for chip_idx, dphi_dt in enumerate(dphi_dt_chips):
        ax.plot(f_sky_ghz, dphi_dt, lw=LW_FINE, color=CHIP_COLORS[chip_idx], label=f"chip {chip_idx}")
        ax.fill_between(
            f_sky_ghz,
            dphi_dt - dphi_dt_err_chips[chip_idx],
            dphi_dt + dphi_dt_err_chips[chip_idx],
            alpha=ALPHA_FILL,
            color=CHIP_COLORS[chip_idx],
        )
    _zero_line(ax)
    ax.set_xlim(*plot_band_ghz)
    ax.set_xlabel(r"$f_{\rm sky}$ [GHz]")
    ax.set_ylabel(r"$d\varphi/dt$ [Hz]")
    ax.legend(fontsize=TICK_SIZE)
    _channel_secondary_xaxis(ax, f_rf0_hz, df_hz)
    ax.set_title(r"Sun --- fringe rate per chip", fontsize=TICK_SIZE)
    _tight_layout(fig)
    return fig, ax


def plot_baseline_vs_frequency(
    f_sky_ghz: np.ndarray,
    baseline_chips: list[np.ndarray],
    baseline_err_chips: list[np.ndarray],
    plot_band_ghz: tuple[float, float],
    f_rf0_hz: float,
    df_hz: float,
) -> tuple[Figure, Axes]:
    fig, ax = _single_panel((TEXTWIDTH_IN, 3))
    for chip_idx, baseline in enumerate(baseline_chips):
        ax.plot(f_sky_ghz, baseline, lw=LW_FINE, color=CHIP_COLORS[chip_idx], label=f"chip {chip_idx}")
        ax.fill_between(
            f_sky_ghz,
            baseline - baseline_err_chips[chip_idx],
            baseline + baseline_err_chips[chip_idx],
            alpha=ALPHA_FILL,
            color=CHIP_COLORS[chip_idx],
        )
    _zero_line(ax)
    ax.set_xlim(*plot_band_ghz)
    ax.set_xlabel(r"$f_{\rm sky}$ [GHz]")
    ax.set_ylabel(r"$B_{\rm EW}$ [m]")
    ax.legend(fontsize=TICK_SIZE)
    _channel_secondary_xaxis(ax, f_rf0_hz, df_hz)
    ax.set_title(r"Sun --- $B_{\rm EW}$ vs $f_{\rm sky}$ (sin$\,h$ fit per chip)", fontsize=TICK_SIZE)
    _tight_layout(fig)
    return fig, ax


def plot_lag_delay_summary(
    tau_axis_ns: np.ndarray,
    lag_amp: np.ndarray,
    tau_peak_ns: float,
    lag_snr: float,
    capture_index: int,
    ha_chips: list[np.ndarray],
    tau_lag_chips: list[np.ndarray],
    baseline_lag_chips: list[float],
    sin_h_lag_chips: list[np.ndarray],
    coeffs_lag_chips: list[np.ndarray],
    ha_limits_deg: tuple[float, float],
) -> tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 6))

    ax = axes[0]
    ax.plot(tau_axis_ns, lag_amp, lw=LW_FINE, color=CHIP_COLORS[0])
    ax.axvline(
        tau_peak_ns,
        color=NEUTRAL_COLOR,
        lw=LW_GUIDE,
        ls="--",
        label=rf"$\tau={tau_peak_ns:.2f}$ ns  (chip 0, SNR$={lag_snr:.1f}$)",
    )
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"$|\mathrm{IFFT}(V_{12})|$")
    ax.set_title(rf"Lag spectrum --- chip 0, capture {capture_index}", fontsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE)

    ax = axes[1]
    for chip_idx, ha_deg in enumerate(ha_chips):
        sin_h = sin_h_lag_chips[chip_idx]
        sin_h_fine = np.linspace(sin_h.min(), sin_h.max(), 100)
        tau_fit = coeffs_lag_chips[chip_idx][0] + coeffs_lag_chips[chip_idx][1] * sin_h_fine
        ha_fine = np.degrees(np.arcsin(sin_h_fine))
        ax.scatter(
            ha_deg,
            tau_lag_chips[chip_idx],
            s=SCATTER_S_FINE,
            color=CHIP_COLORS[chip_idx],
            zorder=3,
            label=rf"chip {chip_idx}  ($B_{{EW}}={baseline_lag_chips[chip_idx]:.3f}$ m)",
        )
        ax.plot(ha_fine, tau_fit, color=CHIP_COLORS[chip_idx], lw=LW_GUIDE, ls="--")

    ax.set_xlim(*ha_limits_deg)
    ax.set_xlabel("Hour angle")
    ax.set_ylabel(r"$\tau$ [ns]")
    ax.xaxis.set_major_formatter(_ha_formatter())
    ax.legend(fontsize=TICK_SIZE)
    ax_top = _hour_angle_degree_secondary_xaxis(ax)
    ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: rf"${x:.1f}^\circ$"))
    ax.set_title(r"Lag delay sin-$h$ fit per chip", fontsize=TICK_SIZE)

    _tight_layout(fig)
    return fig, axes


def plot_interval_baseline(
    ha_chips: list[np.ndarray],
    baseline_lag_chips: list[float],
    interval_results: list[dict[str, float | int | None]],
) -> tuple[Figure, Axes]:
    fig, ax = _single_panel((TEXTWIDTH_IN, 3.5))

    for chip_idx, ha_deg in enumerate(ha_chips):
        ax.axvspan(ha_deg.min(), ha_deg.max(), alpha=ALPHA_SPAN_LIGHT, color=CHIP_COLORS[chip_idx])
        ax.axhline(
            baseline_lag_chips[chip_idx],
            color=CHIP_COLORS[chip_idx],
            lw=LW_GUIDE,
            ls="--",
            label=rf"chip {chip_idx}  $B_{{EW}}={baseline_lag_chips[chip_idx]:.3f}$ m",
        )

    offset_deg = 0.4
    for entry in interval_results:
        chip_idx = entry["chip"]
        color = CHIP_COLORS[chip_idx] if chip_idx is not None else NEUTRAL_COLOR
        peers = [peer for peer in interval_results if peer["lo"] == entry["lo"] and peer is not entry]
        x_coord = entry["ha_ctr"] + (offset_deg if peers and chip_idx == 1 else -offset_deg if peers else 0)
        ax.errorbar(
            x_coord,
            entry["B_EW"],
            yerr=entry["B_EW_err"],
            fmt="o",
            color=color,
            capsize=ERRORBAR_CAPSIZE_SMALL,
            markersize=MARKER_MS_SMALL,
            zorder=4,
        )

    ax.add_artist(ax.legend(fontsize=TICK_SIZE, loc="upper left"))
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=CHIP_COLORS[chip_idx],
            linestyle="none",
            markersize=MARKER_MS_SMALL,
            label=f"chip {chip_idx} interval",
        )
        for chip_idx in range(2)
    ]
    ax.legend(handles=handles, fontsize=TICK_SIZE, loc="upper right")
    ax.set_xlabel("Hour angle (bin centre)")
    ax.set_ylabel(r"$B_{\rm EW}$ [m]")
    ax.xaxis.set_major_formatter(_ha_formatter())
    ax_top = _hour_angle_degree_secondary_xaxis(ax)
    ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: rf"${x:.0f}^\circ$"))
    ax.set_title(r"$B_{\rm EW}$ (lag delay) --- 20-min sub-intervals", fontsize=TICK_SIZE)
    _tight_layout(fig)
    return fig, ax


def plot_drift_comparison(
    ha_chips: list[np.ndarray],
    baseline_lag_chips: list[float],
    uncorrected_results: list[dict[str, float | int | None]],
    drift_results: list[dict[str, float | int | None] | None],
) -> tuple[Figure, np.ndarray]:
    fig, axes = _stacked_panels(2, (TEXTWIDTH_IN, 6), (1, 1), 0.0)

    panel_specs = [
        (axes[0], "No drift correction", uncorrected_results),
        (axes[1], "With drift correction", [entry for entry in drift_results if entry is not None]),
    ]
    offset_deg = 0.4

    for ax, label, results in panel_specs:
        for chip_idx, ha_deg in enumerate(ha_chips):
            ax.axvspan(ha_deg.min(), ha_deg.max(), alpha=ALPHA_SPAN_LIGHT, color=CHIP_COLORS[chip_idx])
            ax.axhline(
                baseline_lag_chips[chip_idx],
                color=CHIP_COLORS[chip_idx],
                lw=LW_GUIDE,
                ls="--",
                label=rf"chip {chip_idx} $B_{{EW}}={baseline_lag_chips[chip_idx]:.3f}$ m",
            )

        for entry in results:
            chip_idx = entry["chip"]
            color = CHIP_COLORS[chip_idx] if chip_idx is not None else NEUTRAL_COLOR
            peers = [peer for peer in results if peer["lo"] == entry["lo"] and peer is not entry]
            x_coord = entry["ha_ctr"] + (offset_deg if peers and chip_idx == 1 else -offset_deg if peers else 0)
            ax.errorbar(
                x_coord,
                entry["B_EW"],
                yerr=entry["B_EW_err"],
                fmt="o",
                color=color,
                capsize=ERRORBAR_CAPSIZE_SMALL,
                markersize=MARKER_MS_SMALL,
                zorder=4,
            )

        ax.set_ylabel(r"$B_{\rm EW}$ [m]")
        ax.set_title(label, fontsize=TICK_SIZE)
        ax.legend(fontsize=TICK_SIZE, loc="upper left")

    axes[1].set_xlabel("Hour angle (bin centre)")
    axes[1].xaxis.set_major_formatter(_ha_formatter())
    ax_top = _hour_angle_degree_secondary_xaxis(axes[1])
    ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: rf"${x:.0f}^\circ$"))
    fig.suptitle(
        r"Atmospheric drift correction: $\tau = A\sin h + B\,\Delta t + \tau_{\rm inst}$",
        fontsize=TICK_SIZE,
    )
    _tight_layout(fig)
    return fig, axes
