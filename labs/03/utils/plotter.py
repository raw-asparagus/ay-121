"""Lab 03 analysis plots used by the Sun notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plotting import (
    ALPHA_EXTRA_LIGHT,
    ALPHA_LIGHT,
    LW_FINE,
    LW_LIGHT,
    MS_FINE,
    MS_STANDARD,
    NEUTRAL_COLOR,
    SS_FINE,
    TEXTWIDTH_IN,
    TICK_SIZE,
    zero_line,
)


def plot_fringe_model_comparison(
    ha_deg: np.ndarray,
    observed: np.ndarray,
    model: np.ndarray,
    title: str,
) -> tuple[Figure, np.ndarray]:
    """Three-panel plot: observed fringes, model, and residuals vs hour angle."""
    residual = observed - model
    fig, axes = plt.subplots(
        3, 1,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.55),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 3, 2), "hspace": 0.0},
    )

    axes[0].plot(ha_deg, observed, lw=LW_FINE, color="C0", label="observed")
    axes[0].set_ylabel("Observed")
    axes[0].legend(fontsize=TICK_SIZE, loc="upper right")
    axes[0].set_title(title, fontsize=TICK_SIZE)

    axes[1].plot(ha_deg, model, lw=LW_FINE, color="C2", label="model")
    axes[1].set_ylabel("Model")
    axes[1].legend(fontsize=TICK_SIZE, loc="upper right")

    axes[2].plot(ha_deg, residual, lw=LW_FINE, color="C1", label="residual")
    zero_line(axes[2])
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("Hour angle [deg]")
    axes[2].legend(fontsize=TICK_SIZE, loc="upper right")

    fig.tight_layout()
    return fig, axes


def plot_bessel_envelope_fit(
    u_lambda: np.ndarray,
    observed_envelope: np.ndarray,
    fitted_envelope: np.ndarray,
    zero_crossings_u: np.ndarray,
    fitted_diameter_arcmin: float,
) -> tuple[Figure, Axes]:
    """Observed amplitude envelope with Bessel-function fit overlay."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.35))

    ax.scatter(
        u_lambda,
        observed_envelope,
        s=SS_FINE,
        color="C0",
        alpha=0.5,
        label="observed",
        zorder=2,
    )

    order = np.argsort(u_lambda)
    ax.plot(
        u_lambda[order],
        fitted_envelope[order],
        lw=LW_FINE,
        color="C2",
        label="Bessel fit",
        zorder=3,
    )

    for k, u_z in enumerate(zero_crossings_u):
        ax.axvline(
            u_z,
            color="C1",
            lw=LW_LIGHT,
            ls="--",
            alpha=ALPHA_LIGHT,
            label="null" if k == 0 else None,
        )

    ax.annotate(
        rf"$\varnothing = {fitted_diameter_arcmin:.2f}$ arcmin",
        xy=(0.98, 0.92),
        xycoords="axes fraction",
        ha="right",
        fontsize=TICK_SIZE,
        color="C2",
    )

    ax.set_xlabel(r"Projected baseline [$\lambda$]")
    ax.set_ylabel("Amplitude envelope")
    ax.legend(fontsize=TICK_SIZE, loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_solar_diameter_summary(
    diameters_arcmin: np.ndarray,
    errors_arcmin: np.ndarray,
    labels: list[str],
    nominal_arcmin: float,
) -> tuple[Figure, Axes]:
    """Compare diameter estimates from different methods or chips."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.3))
    y_pos = np.arange(len(diameters_arcmin))

    ax.errorbar(
        diameters_arcmin,
        y_pos,
        xerr=errors_arcmin,
        fmt="o",
        color="C0",
        markersize=MS_FINE,
        capsize=2,
        zorder=3,
    )
    ax.axvline(
        nominal_arcmin,
        color=NEUTRAL_COLOR,
        lw=LW_LIGHT,
        ls="--",
        label=f"nominal ({nominal_arcmin:.1f}')",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Solar diameter [arcmin]")
    ax.legend(fontsize=TICK_SIZE)
    fig.tight_layout()
    return fig, ax


def plot_sunspot_residuals(
    u_lambda: np.ndarray,
    residuals: np.ndarray,
    noise_std: np.ndarray,
    detections_u: np.ndarray,
) -> tuple[Figure, Axes]:
    """Residuals from the uniform-disk model with sunspot detections marked."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.3))

    ax.scatter(
        u_lambda, residuals, s=SS_FINE, color="C0", alpha=0.5, zorder=2
    )
    zero_line(ax)

    noise = np.broadcast_to(np.asarray(noise_std), residuals.shape)
    order = np.argsort(u_lambda)
    ax.fill_between(
        u_lambda[order],
        -noise[order],
        noise[order],
        alpha=ALPHA_EXTRA_LIGHT,
        color=NEUTRAL_COLOR,
        label=r"$\pm 1\sigma$ noise",
        zorder=1,
    )

    for k, u_d in enumerate(detections_u):
        ax.axvline(
            u_d,
            color="C1",
            lw=LW_LIGHT,
            ls=":",
            alpha=0.7,
            label="detection" if k == 0 else None,
        )

    ax.set_xlabel(r"Projected baseline [$\lambda$]")
    ax.set_ylabel("Residual amplitude")
    ax.legend(fontsize=TICK_SIZE)
    fig.tight_layout()
    return fig, ax
