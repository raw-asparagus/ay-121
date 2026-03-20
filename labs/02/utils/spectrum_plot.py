"""Lab-local plotting helpers for ``ugradiolab.Spectrum`` objects."""

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ugradiolab.data import FrequencyAxis, PlotScale, Spectrum

# Sentinel: ylabel was not explicitly set, derive from normalize_per_hz.
_YLABEL_AUTO = object()


def _default_ylabel(normalize_per_hz: bool) -> str:
    return "PSD (counts²/Hz)" if normalize_per_hz else "PSD (counts²/bin)"


def _default_xlabel(x_mode: FrequencyAxis) -> str:
    return "Frequency (MHz)" if x_mode == "absolute" else r"Baseband offset $\Delta f$ (MHz)"


def plot_spectrum_psd(
    spectrum: Spectrum,
    ax=None,
    title: str | None = None,
    *,
    normalize_per_hz: bool = True,
    smooth_kwargs: dict | None = None,
    show_raw: bool = False,
    show_std: bool = True,
    mask_dc: bool = False,
    x_mode: FrequencyAxis = "absolute",
    yscale: PlotScale = "linear",
    color: str = "C0",
    smooth_color: str | None = None,
    raw_label: str | None = None,
    smooth_label: str | None = None,
    xlabel: str | None = None,
    ylabel=_YLABEL_AUTO,
    legend: bool = False,
    legend_fontsize: float | None = None,
    grid: bool = True,
    title_loc: str = "center",
    raw_linewidth: float = 0.9,
    raw_alpha: float = 0.9,
    smooth_linewidth: float = 1.2,
    smooth_alpha: float = 1.0,
    std_alpha: float = 0.15,
):
    """Plot the raw and optionally smoothed PSD on ``ax``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4), dpi=300)

    scale = 1.0 / spectrum.bin_width if normalize_per_hz else 1.0

    if ylabel is _YLABEL_AUTO:
        ylabel = _default_ylabel(normalize_per_hz)

    x = spectrum.frequency_axis_mhz(mode=x_mode)
    plot_func = ax.semilogy if yscale == "log" else ax.plot

    raw = spectrum.psd_values(mask_dc=mask_dc) * scale
    if show_std:
        floor = np.finfo(float).tiny if yscale == "log" else None
        lo = raw - spectrum.std * scale
        hi = raw + spectrum.std * scale
        if floor is not None:
            lo = np.clip(lo, floor, None)
            hi = np.clip(hi, floor, None)
        ax.fill_between(x, lo, hi, color=color, alpha=std_alpha)

    if show_raw:
        plot_func(
            x,
            raw,
            color=color,
            lw=raw_linewidth,
            alpha=raw_alpha,
            label=raw_label,
        )

    if smooth_kwargs is not None:
        plot_func(
            x,
            spectrum.psd_values(smooth_kwargs=smooth_kwargs, mask_dc=mask_dc) * scale,
            color=smooth_color or color,
            lw=smooth_linewidth,
            alpha=smooth_alpha,
            label=smooth_label,
        )

    ax.set_xlabel(_default_xlabel(x_mode) if xlabel is None else xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    if legend:
        if legend_fontsize is None:
            ax.legend()
        else:
            ax.legend(fontsize=legend_fontsize)
    if grid:
        ax.grid(True, lw=0.4, alpha=0.5)
    return ax


def plot_spectrum_compare(
    spectrum_a: Spectrum,
    spectrum_b: Spectrum,
    ax=None,
    title: str | None = None,
    *,
    normalize_per_hz: bool = True,
    labels: Sequence[str] = ("A", "B"),
    colors: Sequence[str] = ("C0", "C1"),
    smooth_kwargs: dict | None = None,
    show_std: bool = True,
    mask_dc: bool = False,
    x_mode: FrequencyAxis = "absolute",
    yscale: PlotScale = "log",
    xlabel: str | None = None,
    ylabel=_YLABEL_AUTO,
    legend: bool = True,
    legend_fontsize: float | None = 8,
    grid: bool = True,
    title_loc: str = "center",
    raw_linewidth: float = 0.5,
    raw_alpha: float = 0.35,
    smooth_linewidth: float = 1.2,
    std_alpha: float = 0.15,
):
    """Overlay two spectra on a common axis."""
    if len(labels) != 2 or len(colors) != 2:
        raise ValueError("plot_spectrum_compare expects two labels and two colors.")
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4), dpi=300)

    if ylabel is _YLABEL_AUTO:
        ylabel = _default_ylabel(normalize_per_hz)

    plot_spectrum_psd(
        spectrum_a,
        ax=ax,
        normalize_per_hz=normalize_per_hz,
        smooth_kwargs=smooth_kwargs,
        show_std=show_std,
        mask_dc=mask_dc,
        x_mode=x_mode,
        yscale=yscale,
        color=colors[0],
        smooth_color=colors[0],
        raw_label=f"{labels[0]} (raw)",
        smooth_label=(f"{labels[0]} (smoothed)" if smooth_kwargs is not None else None),
        xlabel=xlabel,
        ylabel=ylabel,
        legend=False,
        grid=False,
        raw_linewidth=raw_linewidth,
        raw_alpha=raw_alpha,
        smooth_linewidth=smooth_linewidth,
        std_alpha=std_alpha,
    )
    plot_spectrum_psd(
        spectrum_b,
        ax=ax,
        normalize_per_hz=normalize_per_hz,
        smooth_kwargs=smooth_kwargs,
        show_std=show_std,
        mask_dc=mask_dc,
        x_mode=x_mode,
        yscale=yscale,
        color=colors[1],
        smooth_color=colors[1],
        raw_label=f"{labels[1]} (raw)",
        smooth_label=(f"{labels[1]} (smoothed)" if smooth_kwargs is not None else None),
        xlabel=xlabel,
        ylabel=None,
        legend=False,
        grid=False,
        raw_linewidth=raw_linewidth,
        raw_alpha=raw_alpha,
        smooth_linewidth=smooth_linewidth,
        std_alpha=std_alpha,
    )
    if title is not None:
        ax.set_title(title, loc=title_loc)
    if legend:
        if legend_fontsize is None:
            ax.legend()
        else:
            ax.legend(fontsize=legend_fontsize)
    if grid:
        ax.grid(True, lw=0.4, alpha=0.5)
    return ax


def plot_spectrum_ratio(
    spectrum_a: Spectrum,
    spectrum_b: Spectrum,
    ax=None,
    title: str | None = None,
    *,
    smooth_kwargs: dict | None = None,
    x_mode: FrequencyAxis = "baseband",
    color: str = "C0",
    smooth_color: str = "C1",
    raw_label: str = "raw",
    smooth_label: str = "smoothed",
    show_std: bool = True,
    add_unity_line: bool = True,
    reference_lines: Sequence[dict] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    grid: bool = True,
    title_loc: str = "center",
    raw_linewidth: float = 0.5,
    raw_alpha: float = 0.4,
    smooth_linewidth: float = 1.2,
    std_alpha: float = 0.15,
):
    """Plot the ratio ``spectrum_a / spectrum_b`` with propagated raw uncertainties."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4), dpi=300)

    x = spectrum_a.frequency_axis_mhz(mode=x_mode)
    ratio = spectrum_a.ratio_to(spectrum_b)

    if show_std:
        sigma = spectrum_a.ratio_std_to(spectrum_b)
        ax.fill_between(
            x,
            ratio - sigma,
            ratio + sigma,
            alpha=std_alpha,
            color=color,
        )

    ax.plot(
        x,
        ratio,
        lw=raw_linewidth,
        alpha=raw_alpha,
        color=color,
        label=raw_label,
    )

    if smooth_kwargs is not None:
        ax.plot(
            x,
            spectrum_a.ratio_to(spectrum_b, smooth_kwargs=smooth_kwargs),
            lw=smooth_linewidth,
            color=smooth_color,
            label=smooth_label,
        )

    if add_unity_line:
        ax.axhline(1.0, color="gray", lw=0.7, linestyle="--")
    for ref_line in reference_lines or ():
        params = dict(ref_line)
        x_value = params.pop("x")
        ax.axvline(x_value, **params)

    ax.set_xlabel(_default_xlabel(x_mode) if xlabel is None else xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, lw=0.4, alpha=0.5)
    return ax


def plot_spectrum_stack(
    spectra: Sequence[Spectrum],
    axes=None,
    title: str | None = None,
    *,
    normalize_per_hz: bool = True,
    smooth_kwargs: dict | None = None,
    show_std: bool = True,
    mask_dc: bool = False,
    x_mode: FrequencyAxis = "absolute",
    yscale: PlotScale = "linear",
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    color: str = "C0",
    xlabel: str | None = None,
    ylabel=_YLABEL_AUTO,
    panel_title_fn: Callable[[Spectrum], str] | None = None,
    panel_title_loc: str = "left",
    raw_linewidth: float = 0.9,
    raw_alpha: float = 0.9,
    smooth_linewidth: float = 1.2,
    std_alpha: float = 0.3,
    x_pad_frac: float = 0.05,
    y_pad_frac: float = 0.05,
):
    """Plot one spectrum per row and apply shared axis limits."""
    if not spectra:
        raise ValueError("plot_spectrum_stack requires at least one spectrum.")

    if ylabel is _YLABEL_AUTO:
        ylabel = _default_ylabel(normalize_per_hz)

    nplots = len(spectra)

    if axes is None:
        _, axes = plt.subplots(
            nplots,
            1,
            figsize=figsize or (11, 2.5 * nplots),
            dpi=dpi,
            sharex=True,
            sharey=True,
        )

    axes = np.atleast_1d(axes)
    if axes.size != nplots:
        raise ValueError(f"plot_spectrum_stack expected {nplots} axes, got {axes.size}.")

    for ax, spectrum in zip(axes, spectra):
        plot_spectrum_psd(
            spectrum,
            ax=ax,
            normalize_per_hz=normalize_per_hz,
            title=panel_title_fn(spectrum) if panel_title_fn is not None else None,
            smooth_kwargs=smooth_kwargs,
            show_std=show_std,
            mask_dc=mask_dc,
            x_mode=x_mode,
            yscale=yscale,
            color=color,
            smooth_color=color,
            xlabel=None,
            ylabel=ylabel,
            legend=False,
            title_loc=panel_title_loc,
            raw_linewidth=raw_linewidth,
            raw_alpha=raw_alpha,
            smooth_linewidth=smooth_linewidth,
            std_alpha=std_alpha,
        )

    x_min, x_max, y_min, y_max = _shared_limits(
        spectra,
        normalize_per_hz=normalize_per_hz,
        smooth_kwargs=smooth_kwargs,
        show_std=show_std,
        mask_dc=mask_dc,
        x_mode=x_mode,
        yscale=yscale,
        x_pad_frac=x_pad_frac,
        y_pad_frac=y_pad_frac,
    )
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[-1].set_xlabel(_default_xlabel(x_mode) if xlabel is None else xlabel)

    if title is not None:
        axes[0].figure.suptitle(title)
    return axes


def _shared_limits(
    spectra: Sequence[Spectrum],
    *,
    normalize_per_hz: bool = True,
    smooth_kwargs: dict | None,
    show_std: bool,
    mask_dc: bool,
    x_mode: FrequencyAxis,
    yscale: PlotScale,
    x_pad_frac: float,
    y_pad_frac: float,
) -> tuple[float, float, float, float]:
    x_all = np.concatenate([spectrum.frequency_axis_mhz(mode=x_mode) for spectrum in spectra])

    y_arrays = []
    floor = np.finfo(float).tiny if yscale == "log" else None
    for spectrum in spectra:
        scale = 1.0 / spectrum.bin_width if normalize_per_hz else 1.0
        raw = spectrum.psd_values(mask_dc=mask_dc) * scale
        y_arrays.append(raw)
        if show_std:
            lo = raw - spectrum.std * scale
            hi = raw + spectrum.std * scale
            if floor is not None:
                lo = np.clip(lo, floor, None)
                hi = np.clip(hi, floor, None)
            y_arrays.extend([lo, hi])
        if smooth_kwargs is not None:
            y_arrays.append(spectrum.psd_values(smooth_kwargs=smooth_kwargs, mask_dc=mask_dc) * scale)

    finite_x = x_all[np.isfinite(x_all)]
    x_min = float(finite_x.min())
    x_max = float(finite_x.max())
    x_span = x_max - x_min
    x_pad = x_pad_frac * (x_span if x_span > 0 else 1.0)

    if yscale == "log":
        finite_y = np.concatenate([y[np.isfinite(y) & (y > 0)] for y in y_arrays])
        y_min = float(finite_y.min())
        y_max = float(finite_y.max())
        log_span = np.log10(y_max) - np.log10(y_min)
        y_pad = y_pad_frac * (log_span if log_span > 0 else 1.0)
        return (
            x_min - x_pad,
            x_max + x_pad,
            10 ** (np.log10(y_min) - y_pad),
            10 ** (np.log10(y_max) + y_pad),
        )

    finite_y = np.concatenate([y[np.isfinite(y)] for y in y_arrays])
    y_min = float(finite_y.min())
    y_max = float(finite_y.max())
    y_span = y_max - y_min
    y_pad = y_pad_frac * (y_span if y_span > 0 else 1.0)
    return (
        x_min - x_pad,
        x_max + x_pad,
        y_min - y_pad,
        y_max + y_pad,
    )


plot_psd = plot_spectrum_psd
plot_compare = plot_spectrum_compare
plot_ratio = plot_spectrum_ratio
plot_stack = plot_spectrum_stack


__all__ = [
    "plot_spectrum_psd",
    "plot_spectrum_compare",
    "plot_spectrum_ratio",
    "plot_spectrum_stack",
    "plot_psd",
    "plot_compare",
    "plot_ratio",
    "plot_stack",
]
