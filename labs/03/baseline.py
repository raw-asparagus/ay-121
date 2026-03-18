"""baseline.py — Fringe MCMC implementation for AY-121 Lab 3 baseline estimation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from astropy.coordinates import get_sun
from astropy.time import Time
from ugdatalab.mcmc import NoUTurnHamiltonian

# ---------------------------------------------------------------------------
# Physical / model constants
# ---------------------------------------------------------------------------
C_LIGHT      = 2.99792458e8   # m/s
D_DISH       = 0.80           # m — dish diameter, used only as beam prior scale
LN2_4        = 4.0 * np.log(2.0)
K_CHANNEL    = 550            # spectral channel for baseline fit
NCH_LAT_RAD  = np.deg2rad(37.8732)   # Campbell Hall latitude

PARAM_NAMES  = ['b_e', 'phi0', 'A0', 'V_re', 'V_im', 'H_beam', 'theta_fwhm', 'sigma']
PARAM_LABELS = [r'$b_E$', r'$\phi_0$', r'$A_0$', r'$V_{\rm re}$', r'$V_{\rm im}$',
                r'$H_{\rm beam}$', r'$\theta_{\rm FWHM}$', r'$\sigma$']

CORNER_NAMES  = ['b_e', 'phi0', 'A0', 'theta_fwhm']
CORNER_LABELS = [r'$b_E$', r'$\phi_0$', r'$A_0$', r'$\theta_{\rm FWHM}$']
CORNER_IDX    = [PARAM_NAMES.index(n) for n in CORNER_NAMES]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_session_arrays(sd: dict) -> tuple:
    """Convert a session dict into (vis_obs, ha_rad, dec_rad, y_re, y_im)."""
    vis_obs = np.array([f['corr'][K_CHANNEL] for f in sd['files_s']])
    ha_rad  = np.deg2rad(sd['ha_deg'])
    dec_rad = np.array([
        get_sun(Time(t, format='unix')).dec.rad
        for t in sd['unix_sort']
    ])
    y_re = vis_obs.real.astype(float)
    y_im = vis_obs.imag.astype(float)
    return vis_obs, ha_rad, dec_rad, y_re, y_im


# ---------------------------------------------------------------------------
# PyMC model
# ---------------------------------------------------------------------------
def _build_fringe_model(f_k, theta_fwhm_rad_0, ha_rad, dec_rad, y_re, y_im, ha_mean, sigma0):
    with pm.Model() as model:
        b_e        = pm.Uniform('b_e',            lower=7.5,        upper=17.5)
        phi0       = pm.Uniform('phi0',           lower=-np.pi,     upper=np.pi)
        A0         = pm.HalfNormal('A0',          sigma=sigma0)
        V_re       = pm.Normal('V_re',            mu=float(np.mean(y_re)), sigma=sigma0)
        V_im       = pm.Normal('V_im',            mu=float(np.mean(y_im)), sigma=float(np.std(y_im)))
        H_beam     = pm.Normal('H_beam',          mu=ha_mean,       sigma=np.deg2rad(5.0))
        theta_fwhm = pm.HalfNormal('theta_fwhm', sigma=theta_fwhm_rad_0)
        sigma      = pm.HalfNormal('sigma',       sigma=sigma0)

        phi_i = (2.0 * np.pi * f_k / C_LIGHT) * b_e * pt.cos(dec_rad) * pt.sin(ha_rad)
        l_i   = pt.cos(dec_rad) * pt.sin(ha_rad - H_beam)   # EW direction cosine
        A_i   = A0 * pt.exp(-LN2_4 * l_i**2 / theta_fwhm**2)

        mu_re = V_re + A_i * pt.cos(phi_i + phi0)
        mu_im = V_im + A_i * pt.sin(phi_i + phi0)

        pm.Normal('obs_re', mu=mu_re, sigma=sigma, observed=y_re)
        pm.Normal('obs_im', mu=mu_im, sigma=sigma, observed=y_im)
    return model


# ---------------------------------------------------------------------------
# Result construction
# ---------------------------------------------------------------------------
def _make_result(samples_arr: np.ndarray, log_probs: np.ndarray,
                 acceptance_rate: float) -> SimpleNamespace:
    samples_dict   = {name: samples_arr[:, i] for i, name in enumerate(PARAM_NAMES)}
    display        = SimpleNamespace(
        samples      = samples_arr,
        log_probs    = log_probs,
        n_burn       = 0,
        param_labels = PARAM_LABELS,
    )
    display_corner = SimpleNamespace(
        samples      = samples_arr[:, CORNER_IDX],
        n_burn       = 0,
        param_labels = CORNER_LABELS,
    )
    return SimpleNamespace(
        samples         = samples_arr,
        log_probs       = log_probs,
        samples_dict    = samples_dict,
        display         = display,
        display_corner  = display_corner,
        param_names     = PARAM_NAMES,
        param_labels    = PARAM_LABELS,
        acceptance_rate = acceptance_rate,
    )


# ---------------------------------------------------------------------------
# MCMC sampler
# ---------------------------------------------------------------------------
def run_fringe_mcmc(vis_obs, ha_rad, dec_rad, y_re, y_im, *,
                    f_k: float, theta_fwhm_rad_0: float,
                    label: str = '', n_steps: int = 2000, n_burn: int = 1000,
                    seed: int = 42) -> SimpleNamespace:
    ha_mean = float(np.mean(ha_rad))
    sigma0  = float(np.std(y_re))

    model = _build_fringe_model(f_k, theta_fwhm_rad_0, ha_rad, dec_rad,
                                y_re, y_im, ha_mean, sigma0)

    theta0 = np.array([
        12.5,                      # b_e
        0.0,                       # phi0
        sigma0,                    # A0
        float(np.mean(y_re)),      # V_re
        float(np.mean(y_im)),      # V_im
        ha_mean,                   # H_beam
        theta_fwhm_rad_0,          # theta_fwhm
        sigma0,                    # sigma
    ])

    print(f'\n── Sampling: {label} ──')
    sampler = NoUTurnHamiltonian(
        model=model,
        var_names=PARAM_NAMES,
        theta0=theta0,
        seed=seed,
        labels=PARAM_LABELS,
    )
    sampler.run(n_steps=n_steps, n_burn=n_burn)

    samples_arr = np.asarray(sampler.samples, dtype=float)
    log_probs   = np.asarray(sampler.log_probs, dtype=float)
    print(f'  acceptance rate: {sampler.acceptance_rate:.3f}')
    return _make_result(samples_arr, log_probs, sampler.acceptance_rate)


# ---------------------------------------------------------------------------
# Cache-aware entry point
# ---------------------------------------------------------------------------
def load_or_run_mcmc(session_arrays: list, *, f_k: float, theta_fwhm_rad_0: float,
                     cache_path: Path, session_labels: list[str] | None = None,
                     n_steps: int = 2000, n_burn: int = 1000,
                     seed: int = 42) -> list[SimpleNamespace]:
    """Return one result per session, loading from cache_path if it exists."""
    cache_path = Path(cache_path)
    n_sessions = len(session_arrays)
    labels     = session_labels or [f'Session {i+1}' for i in range(n_sessions)]

    if cache_path.exists():
        print(f'Loading MCMC results from {cache_path}')
        cache   = np.load(cache_path)
        results = []
        for i, lbl in enumerate(labels):
            r = _make_result(
                cache[f's{i+1}_samples'],
                cache[f's{i+1}_log_probs'],
                float(cache[f's{i+1}_acceptance_rate']),
            )
            print(f'  {lbl}: {len(r.samples)} draws  '
                  f'(acceptance rate: {r.acceptance_rate:.3f})')
            results.append(r)
        return results

    results   = []
    save_dict = {}
    for i, (arrays, lbl) in enumerate(zip(session_arrays, labels)):
        vis_obs, ha_rad, dec_rad, y_re, y_im = arrays
        r = run_fringe_mcmc(
            vis_obs, ha_rad, dec_rad, y_re, y_im,
            f_k=f_k, theta_fwhm_rad_0=theta_fwhm_rad_0,
            label=lbl, n_steps=n_steps, n_burn=n_burn, seed=seed,
        )
        results.append(r)
        save_dict[f's{i+1}_samples']         = r.samples
        save_dict[f's{i+1}_log_probs']       = r.log_probs
        save_dict[f's{i+1}_acceptance_rate'] = r.acceptance_rate

    np.savez(cache_path, **save_dict)
    print(f'\nSaved MCMC cache → {cache_path}')
    return results


# ---------------------------------------------------------------------------
# Baseline periodogram estimator
# ---------------------------------------------------------------------------
def estimate_baseline(y_re, y_im, ha_rad, dec_rad, f_k, *,
                      b_min: float = 7.5, b_max: float = 17.5,
                      n_b: int = 50_000) -> SimpleNamespace:
    """Estimate b_e via matched-filter periodogram in direction-cosine space.

    The fringe phase φ_i = 2π f b_e u_i / c  where  u_i = cos(δ_i) sin(H_i).
    After DC subtraction, the power P(b) = |Σ z_i exp(-i φ_i(b))|² peaks at
    the true b_e without phase-wrapping ambiguity.

    Returns a SimpleNamespace with fields:
        b_est   – peak baseline estimate (m)
        b_sigma – half-power half-width converted to 1-σ (m)
        b_grid  – (n_b,) baseline grid (m)
        power   – (n_b,) matched-filter power
        u       – (N,) direction cosines used
    """
    z = (y_re - np.median(y_re)) + 1j * (y_im - np.median(y_im))
    u = np.cos(dec_rad) * np.sin(ha_rad)

    b_grid = np.linspace(b_min, b_max, n_b)
    phase  = (2.0 * np.pi * f_k / C_LIGHT) * np.outer(b_grid, u)   # (n_b, N)
    power  = np.abs(np.dot(np.exp(-1j * phase), z)) ** 2            # (n_b,)

    i_peak  = int(np.argmax(power))
    b_est   = b_grid[i_peak]

    # Half-power width → σ  (FWHM / 2.35 for a sinc-like peak)
    half    = power[i_peak] / 2.0
    left    = i_peak - np.searchsorted(power[i_peak::-1], half)
    right   = i_peak + np.searchsorted(power[i_peak:],    half)
    fwhm    = b_grid[min(right, n_b - 1)] - b_grid[max(left, 0)]
    b_sigma = fwhm / 2.35

    return SimpleNamespace(
        b_est   = b_est,
        b_sigma = b_sigma,
        b_grid  = b_grid,
        power   = power,
        u       = u,
    )


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(result: SimpleNamespace, label: str) -> None:
    print(f'\n=== {label}  (acceptance rate: {result.acceptance_rate:.3f}) ===')
    print(f"{'param':>12}  {'mean':>10}  {'std':>10}  {'hdi_3%':>10}  {'hdi_97%':>10}")
    for name in ['b_e', 'phi0', 'A0', 'V_re', 'V_im', 'theta_fwhm', 'sigma']:
        s = result.samples_dict[name]
        print(f'{name:>12}  {np.mean(s):>10.4f}  {np.std(s):>10.4f}'
              f'  {np.percentile(s, 3):>10.4f}  {np.percentile(s, 97):>10.4f}')
