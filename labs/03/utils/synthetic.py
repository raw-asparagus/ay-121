"""Synthetic end-to-end data generation for pipeline validation.

This module generates a mock visibility time series with known baseline,
declination, solar angular radius, noise level, and (optionally) a DC
pedestal and a per-chip gain step. Feeding the mock data into the same
DC-correction / baseline-fit / diameter-fit pipeline used on the real
observations lets us verify that injected truth is recovered within the
reported statistical uncertainties — a single cheap check that would
otherwise require an independent observation of a source with known
diameter.

This file is **not** required by the AY 121 lab manual. It exists to
back the synthetic validation cell in notebook 05 and is also imported
by notebook 04 for cross-checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C_LIGHT_MS, NCH_LAT_DEG
from .fringe_model import (
    FringeModelParams,
    SolarDiskParams,
    solar_visibility,
)

_SIN_LAT_NCH = np.sin(np.deg2rad(NCH_LAT_DEG))


@dataclass(frozen=True)
class SyntheticDataset:
    """Synthetic Sun-like dataset with known ground-truth parameters."""

    ha_rad: np.ndarray          # (N_cap,) hour angle per capture
    visibility_clean: np.ndarray  # (N_cap, N_ch) complex — ideal visibility
    visibility_noisy: np.ndarray  # (N_cap, N_ch) complex — clean + noise + pedestal + chip gain
    f_sky_hz: np.ndarray        # (N_ch,) channel frequencies
    chip_slices: list[slice]    # row-partition of the N_cap captures into chips
    chip_gains: np.ndarray      # (N_chips,) per-chip multiplicative gain
    pedestal_real: np.ndarray   # (N_cap,) per-capture DC pedestal on Re(V)
    truth: dict                 # ground-truth parameter dict


def make_synthetic_sun_dataset(
    *,
    n_captures: int = 2000,
    ha_range_deg: tuple[float, float] = (-80.0, 80.0),
    b_ew_m: float = 15.17,
    b_ns_m: float = 1.36,
    dec_deg: float = 0.0,
    solar_diameter_arcmin: float = 32.0,
    amplitude: float = 1.0,
    phase_offset_rad: float = 0.3,
    f_band_ghz: tuple[float, float] = (10.415, 10.485),
    n_channels: int = 40,
    noise_sigma: float = 0.02,
    pedestal_amplitude: float = 0.5,
    chip_boundaries_frac: tuple[float, ...] = (0.33, 0.66),
    chip_gains: tuple[float, ...] = (1.0, 1.0, 1.0991),
    rng_seed: int | None = 42,
) -> SyntheticDataset:
    r"""Build a Sun-like synthetic visibility dataset with known truth.

    The clean visibility is a uniform-disk model (:func:`solar_visibility`)
    with the supplied baseline, declination, and angular radius. Three
    *nuisances* are then optionally added on top to mimic the real data:

    1. **Gaussian complex noise** of per-channel :math:`\sigma =` ``noise_sigma``
       (independent on the real and imaginary parts).
    2. **A slowly-varying DC pedestal** on the real part only, of the form
       :math:`p(h) = A\bigl[0.6\cos h + 0.3\cos 2h + 0.1\bigr]` — the
       rolling-median DC corrector is designed to subtract this.
    3. **Per-chip multiplicative gain step.** The captures are partitioned
       into contiguous chips whose boundaries are given as fractions of the
       total, and each chip is multiplied by its gain. The reference chip
       (gain = 1) should be index 0.

    Parameters
    ----------
    n_captures
        Total number of synthetic captures.
    ha_range_deg
        Hour-angle bracket (uniformly sampled).
    b_ew_m, b_ns_m
        East-west and north-south baseline components in metres.
    dec_deg
        Source declination in degrees.
    solar_diameter_arcmin
        Angular diameter of the synthetic disk.
    amplitude
        Point-source fringe amplitude (the normaliser :math:`V(0)`).
    phase_offset_rad
        Constant cable-delay phase.
    f_band_ghz
        Channel-frequency range, uniformly sampled into ``n_channels``
        bins.
    noise_sigma
        Gaussian noise sigma per channel per capture, applied
        independently to Re(V) and Im(V).
    pedestal_amplitude
        Amplitude of the DC pedestal added to the real part.
    chip_boundaries_frac
        Monotonically increasing fractions in ``(0, 1)`` giving the
        chip boundaries as fractions of ``n_captures``. Length
        ``len(chip_gains) - 1``.
    chip_gains
        Per-chip multiplicative gain, length ``len(chip_boundaries_frac) + 1``.
    rng_seed
        Seed for the noise generator. Set for reproducibility.

    Returns
    -------
    SyntheticDataset
        Container with clean and noisy visibilities, chip slices, and a
        ``truth`` dict of the injected parameters.
    """
    if len(chip_gains) != len(chip_boundaries_frac) + 1:
        raise ValueError(
            "chip_gains must be exactly one longer than chip_boundaries_frac"
        )

    rng = np.random.default_rng(rng_seed)

    ha_rad = np.deg2rad(
        np.linspace(ha_range_deg[0], ha_range_deg[1], n_captures)
    )
    f_sky_hz = np.linspace(
        f_band_ghz[0] * 1e9, f_band_ghz[1] * 1e9, n_channels
    )
    band_centre_hz = float(np.mean(f_sky_hz))

    dec_rad = np.deg2rad(dec_deg)
    R_rad = np.deg2rad(solar_diameter_arcmin / 2.0 / 60.0)

    # Build the clean (N_cap, N_ch) visibility by evaluating the uniform-disk
    # Sun model once per channel (the band is narrow enough that per-channel
    # evaluation is cheap).
    vis_clean = np.empty((n_captures, n_channels), dtype=complex)
    for j, f_hz in enumerate(f_sky_hz):
        params = FringeModelParams(
            b_ew=b_ew_m,
            b_ns=b_ns_m,
            freq_hz=float(f_hz),
            dec_rad=dec_rad,
            amplitude=amplitude,
            phase_offset=phase_offset_rad,
        )
        disk = SolarDiskParams(angular_radius_rad=R_rad)
        vis_clean[:, j] = solar_visibility(ha_rad, params, disk)

    # Additive noise (independent real/imag per channel per capture).
    noise_re = rng.normal(0.0, noise_sigma, size=vis_clean.shape)
    noise_im = rng.normal(0.0, noise_sigma, size=vis_clean.shape)
    vis_noisy = vis_clean + noise_re + 1j * noise_im

    # DC pedestal on the real part, channel-independent.
    pedestal_real = pedestal_amplitude * (
        0.6 * np.cos(ha_rad) + 0.3 * np.cos(2.0 * ha_rad) + 0.1
    )
    vis_noisy.real += pedestal_real[:, None]

    # Per-chip gain.
    boundaries = [0] + [
        int(round(frac * n_captures)) for frac in chip_boundaries_frac
    ] + [n_captures]
    chip_slices = [slice(boundaries[i], boundaries[i + 1]) for i in range(len(chip_gains))]
    for sl, g in zip(chip_slices, chip_gains):
        vis_noisy[sl] = vis_noisy[sl] * float(g)

    truth = {
        "b_ew_m": b_ew_m,
        "b_ns_m": b_ns_m,
        "dec_deg": dec_deg,
        "solar_diameter_arcmin": solar_diameter_arcmin,
        "R_rad": R_rad,
        "amplitude": amplitude,
        "phase_offset_rad": phase_offset_rad,
        "band_centre_hz": band_centre_hz,
        "chip_gains": np.array(chip_gains, dtype=float),
        "pedestal_amplitude": pedestal_amplitude,
        "noise_sigma": noise_sigma,
    }

    return SyntheticDataset(
        ha_rad=ha_rad,
        visibility_clean=vis_clean,
        visibility_noisy=vis_noisy,
        f_sky_hz=f_sky_hz,
        chip_slices=chip_slices,
        chip_gains=np.array(chip_gains, dtype=float),
        pedestal_real=pedestal_real,
        truth=truth,
    )
