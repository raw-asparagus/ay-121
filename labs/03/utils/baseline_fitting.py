"""Baseline determination from interferometric fringe observations.

Three independent methods are provided, each returning a standardised
:class:`BaselineResult`.  A combined inverse-variance estimator is also
available.

Methods
-------
1. **FFT in x-coordinate** — remap visibility to the linearised coordinate
   ``x = cos(dec) sin(h)`` and FFT to find the fringe spatial frequency.
2. **Phase slope** — fit unwrapped phase against ``sin(h)`` per channel.
3. **Lag-spectrum delay** — IFFT the cross-spectrum to obtain geometric delay,
   then fit ``tau(sin h)`` across captures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_LIGHT_MS, NCH_LAT_DEG
from .geometry import geometric_delay_s, x_coordinate

__all__ = [
    "BaselineResult",
    "fft_baseline_single_channel",
    "fft_baseline_broadband",
    "phase_slope_baseline_single_channel",
    "phase_slope_baseline_broadband",
    "lag_delay_single_capture",
    "lag_delay_baseline_series",
    "nls_baseline_single_channel",
    "nls_baseline_broadband",
    "combined_baseline",
]

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Standardised output of a baseline determination."""

    b_ew_m: float
    b_ew_err_m: float
    b_ns_m: float = 0.0
    b_ns_err_m: float = np.inf
    method: str = ""
    chi2_reduced: float = np.nan
    n_points: int = 0
    metadata: dict = field(default_factory=dict)


# ===================================================================
# Method 1 — FFT in x-coordinate
# ===================================================================

_FFT_PAD_FACTOR = 16
_FFT_MIN_SAMPLES = 16
_FFT_ZERO_EXCLUDE_CYCLES = 25.0
_FFT_SECONDARY_SEP_CYCLES = 15.0


def _fft_peak_refined(
    freq_axis: np.ndarray,
    amp: np.ndarray,
    peak_idx: int,
) -> float:
    """Parabolic (3-point Lagrange) interpolation around an FFT peak."""
    if peak_idx <= 0 or peak_idx >= len(amp) - 1:
        return freq_axis[peak_idx]
    y0, y1, y2 = amp[peak_idx - 1], amp[peak_idx], amp[peak_idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-30:
        return freq_axis[peak_idx]
    delta = 0.5 * (y0 - y2) / denom
    df = freq_axis[1] - freq_axis[0]
    return freq_axis[peak_idx] + delta * df


def _fft_sorted_local_peaks(
    amp: np.ndarray,
    search_mask: np.ndarray,
) -> np.ndarray:
    """Return indices of local maxima in *amp* within *search_mask*, sorted by
    descending amplitude."""
    candidates = []
    for i in range(1, len(amp) - 1):
        if search_mask[i] and amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1]:
            candidates.append(i)
    if not candidates:
        return np.array([], dtype=int)
    candidates = np.array(candidates)
    return candidates[np.argsort(-amp[candidates])]


def fft_baseline_single_channel(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float,
    freq_hz: float,
    *,
    pad_factor: int = _FFT_PAD_FACTOR,
    min_samples: int = _FFT_MIN_SAMPLES,
    zero_exclude_cycles: float = _FFT_ZERO_EXCLUDE_CYCLES,
) -> BaselineResult | None:
    r"""Baseline from a single spectral channel using FFT in x-space.

    .. note::

        This method only determines :math:`b_{\rm ew}`.  The coordinate
        transform :math:`x = \cos\delta\,\sin h` absorbs the :math:`\cos h`
        dependence from :math:`b_{\rm ns}`, so the north–south component
        cannot be recovered.  The returned result has
        ``b_ns_m = 0, b_ns_err_m = inf``.

    Steps
    -----
    1. Compute ``x = cos(dec) sin(h)`` for each sample.
    2. Sort and interpolate visibility onto a uniform x-grid.
    3. Subtract complex mean, apply Hanning window.
    4. Zero-pad and FFT.
    5. Find peak frequency via parabolic interpolation.
    6. Convert: ``B_EW = |f_x| * c / freq_hz``.

    Returns ``None`` if fewer than *min_samples* valid points.
    """
    ha_rad = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    # Mask NaN / inf
    valid = np.isfinite(vis)
    if valid.sum() < min_samples:
        return None

    x_vals = x_coordinate(ha_rad, dec_rad)
    x_good = x_vals[valid]
    vis_good = vis[valid]

    # Sort by x
    order = np.argsort(x_good)
    x_good = x_good[order]
    vis_good = vis_good[order]

    # Uniform grid
    n_pts = len(x_good)
    x_uniform = np.linspace(x_good[0], x_good[-1], n_pts)
    vis_re = np.interp(x_uniform, x_good, vis_good.real)
    vis_im = np.interp(x_uniform, x_good, vis_good.imag)
    vis_uniform = vis_re + 1j * vis_im

    # DC subtract + Hanning window
    vis_uniform -= vis_uniform.mean()
    window = np.hanning(n_pts)
    vis_uniform *= window

    # FFT
    n_pad = pad_factor * n_pts
    dx = np.median(np.diff(x_uniform))
    spectrum = np.fft.fftshift(np.fft.fft(vis_uniform, n=n_pad))
    fx_axis = np.fft.fftshift(np.fft.fftfreq(n_pad, d=dx))
    amp = np.abs(spectrum)

    # Exclude DC zone
    search_mask = np.abs(fx_axis) >= zero_exclude_cycles

    peaks = _fft_sorted_local_peaks(amp, search_mask)
    if len(peaks) == 0:
        return None

    fx_peak = _fft_peak_refined(fx_axis, amp, peaks[0])
    b_ew = np.abs(fx_peak) * C_LIGHT_MS / freq_hz

    # Uncertainty: max of grid resolution and span-based limit
    df_x = 1.0 / (n_pad * dx) if dx > 0 else np.inf
    x_span = x_good[-1] - x_good[0]
    err_grid = 0.5 * df_x * C_LIGHT_MS / freq_hz
    err_span = (0.5 / x_span * C_LIGHT_MS / freq_hz) if x_span > 0 else np.inf
    b_ew_err = max(err_grid, err_span)

    # SNR
    off_peak = np.ones(len(amp), dtype=bool)
    pk = peaks[0]
    lo = max(0, pk - 10)
    hi = min(len(amp), pk + 11)
    off_peak[lo:hi] = False
    snr = amp[pk] / np.median(amp[off_peak]) if off_peak.any() else np.nan

    return BaselineResult(
        b_ew_m=b_ew,
        b_ew_err_m=b_ew_err,
        method="fft",
        n_points=n_pts,
        metadata={"fx_peak": fx_peak, "snr": snr, "x_span": x_span},
    )


def fft_baseline_broadband(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float,
    f_sky_hz: np.ndarray,
    *,
    bad_channels: np.ndarray | tuple | None = None,
    band_hz: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[BaselineResult | None, np.ndarray]:
    """Run :func:`fft_baseline_single_channel` across all good channels.

    Parameters
    ----------
    corr_dc : (N_cap, N_CH) complex array
        DC-corrected visibility.
    f_sky_hz : (N_CH,) float array
        Sky frequency per channel in Hz.
    bad_channels : array_like, optional
        Channel indices to skip.
    band_hz : (lo, hi), optional
        Restrict to channels within this frequency range.

    Returns
    -------
    result : BaselineResult or None
        Band-median baseline (None if no valid channels).
    per_channel : (N_CH,) structured array
        Per-channel ``b_ew_m``, ``b_ew_err_m``, ``snr`` (NaN where invalid).
    """
    n_ch = corr_dc.shape[1]
    f_sky_hz = np.asarray(f_sky_hz)

    # Build channel mask
    good = np.ones(n_ch, dtype=bool)
    if bad_channels is not None:
        good[list(bad_channels)] = False
    if band_hz is not None:
        good &= (f_sky_hz >= band_hz[0]) & (f_sky_hz <= band_hz[1])

    per_ch = np.full(
        n_ch, np.nan, dtype=[("b_ew_m", float), ("b_ew_err_m", float), ("snr", float)]
    )

    for k in np.where(good)[0]:
        res = fft_baseline_single_channel(
            ha_rad, corr_dc[:, k], dec_rad, f_sky_hz[k], **kwargs
        )
        if res is not None:
            per_ch[k]["b_ew_m"] = res.b_ew_m
            per_ch[k]["b_ew_err_m"] = res.b_ew_err_m
            per_ch[k]["snr"] = res.metadata.get("snr", np.nan)

    valid = np.isfinite(per_ch["b_ew_m"])
    if not valid.any():
        return None, per_ch

    b_vals = per_ch["b_ew_m"][valid]
    med = np.median(b_vals)
    p16, p84 = np.percentile(b_vals, [16, 84])
    scatter = 0.5 * (p84 - p16)

    result = BaselineResult(
        b_ew_m=med,
        b_ew_err_m=scatter,
        method="fft_broadband",
        n_points=int(valid.sum()),
        metadata={
            "p16": p16,
            "p84": p84,
            "median_snr": float(np.nanmedian(per_ch["snr"][valid])),
        },
    )
    return result, per_ch


# ===================================================================
# Method 2 — Phase slope (phi vs sin h)
# ===================================================================


def phase_slope_baseline_single_channel(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float,
    freq_hz: float,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
    min_samples: int = 4,
) -> BaselineResult | None:
    r"""Baseline from unwrapped-phase slope against ``sin(h)`` and ``cos(h)``.

    Fits:

    .. math::

        \phi(h) = A\,\sin h + B\,\cos h + \phi_0

    then converts:

    .. math::

        b_{\rm ew} = \frac{-A\,c}{2\pi\,\nu\,\cos\delta}, \quad
        b_{\rm ns} = \frac{-B\,c}{2\pi\,\nu\,\sin L\,\cos\delta}.

    Returns ``None`` if fewer than *min_samples* valid points.
    """
    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    valid = np.isfinite(vis)
    if valid.sum() < min_samples:
        return None

    phase = np.unwrap(np.angle(vis[valid]))
    sin_h = np.sin(ha[valid])
    cos_h = np.cos(ha[valid])

    # Design matrix: [sin(h), cos(h), 1]
    n = int(valid.sum())
    X = np.column_stack([sin_h, cos_h, np.ones(n)])
    # Least-squares solve
    result = np.linalg.lstsq(X, phase, rcond=None)
    coeffs = result[0]  # [A, B, phi_0]
    A, B, phi_0 = coeffs

    cos_dec = np.cos(dec_rad)
    sin_lat = np.sin(lat_rad)
    scale = C_LIGHT_MS / (2.0 * np.pi * freq_hz)

    b_ew = -A * scale / cos_dec
    b_ns = -B * scale / (sin_lat * cos_dec) if abs(sin_lat * cos_dec) > 1e-15 else np.nan

    # Uncertainty from residual covariance
    resid = phase - X @ coeffs
    resid_var = np.sum(resid**2) / max(1, n - 3)
    try:
        cov = resid_var * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)

    A_err = np.sqrt(max(0, cov[0, 0]))
    B_err = np.sqrt(max(0, cov[1, 1]))
    b_ew_err = np.abs(A_err * scale / cos_dec)
    b_ns_err = (
        np.abs(B_err * scale / (sin_lat * cos_dec))
        if abs(sin_lat * cos_dec) > 1e-15
        else np.inf
    )

    chi2 = resid_var  # already divided by dof

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=b_ns,
        b_ns_err_m=b_ns_err,
        method="phase_slope",
        chi2_reduced=chi2,
        n_points=n,
        metadata={
            "A_sin_h": A,
            "B_cos_h": B,
            "phi_0": phi_0,
            "cov_AB": cov[:2, :2],
        },
    )


def phase_slope_baseline_broadband(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float,
    f_sky_hz: np.ndarray,
    *,
    bad_channels: np.ndarray | tuple | None = None,
    band_hz: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[BaselineResult | None, np.ndarray]:
    """Band-averaged phase-slope baseline across channels.

    Returns (aggregated result, per-channel structured array).
    """
    n_ch = corr_dc.shape[1]
    f_sky_hz = np.asarray(f_sky_hz)

    good = np.ones(n_ch, dtype=bool)
    if bad_channels is not None:
        good[list(bad_channels)] = False
    if band_hz is not None:
        good &= (f_sky_hz >= band_hz[0]) & (f_sky_hz <= band_hz[1])

    per_ch = np.full(
        n_ch, np.nan, dtype=[("b_ew_m", float), ("b_ew_err_m", float)]
    )

    for k in np.where(good)[0]:
        res = phase_slope_baseline_single_channel(
            ha_rad, corr_dc[:, k], dec_rad, f_sky_hz[k], **kwargs
        )
        if res is not None:
            per_ch[k]["b_ew_m"] = res.b_ew_m
            per_ch[k]["b_ew_err_m"] = res.b_ew_err_m

    valid = np.isfinite(per_ch["b_ew_m"])
    if not valid.any():
        return None, per_ch

    b_vals = per_ch["b_ew_m"][valid]
    mean_b = np.mean(b_vals)
    std_b = np.std(b_vals) / np.sqrt(len(b_vals)) if len(b_vals) > 1 else np.inf

    result = BaselineResult(
        b_ew_m=mean_b,
        b_ew_err_m=std_b,
        method="phase_slope_broadband",
        n_points=int(valid.sum()),
    )
    return result, per_ch


# ===================================================================
# Method 3 — Lag-spectrum delay
# ===================================================================


def lag_delay_single_capture(
    vis_spectrum: np.ndarray,
    df_hz: float,
    *,
    pad_factor: int = 16,
    bad_channels: np.ndarray | tuple | None = None,
) -> tuple[float, float, float]:
    """Geometric delay from IFFT of a single cross-spectrum.

    Parameters
    ----------
    vis_spectrum : (N_CH,) complex array
        Single-capture cross-correlation spectrum.
    df_hz : float
        Channel spacing in Hz.
    pad_factor : int
        Zero-padding factor.
    bad_channels : array_like, optional
        Channels to interpolate over.

    Returns
    -------
    tau_ns : float
        Peak delay in nanoseconds.
    snr : float
        Peak-to-median amplitude ratio.
    peak_amp : float
        Amplitude at the lag peak.
    """
    vis = np.array(vis_spectrum, dtype=complex)
    n_ch = len(vis)

    # Interpolate bad channels
    if bad_channels is not None:
        good = np.ones(n_ch, dtype=bool)
        good[list(bad_channels)] = False
        x_good = np.where(good)[0]
        x_bad = np.where(~good)[0]
        if len(x_good) >= 2 and len(x_bad) > 0:
            vis.real[x_bad] = np.interp(x_bad, x_good, vis.real[x_good])
            vis.imag[x_bad] = np.interp(x_bad, x_good, vis.imag[x_good])

    n_pad = n_ch * pad_factor
    lag = np.fft.fftshift(np.fft.ifft(vis, n=n_pad))
    tau_axis_ns = np.fft.fftshift(np.fft.fftfreq(n_pad, d=df_hz)) * 1e9

    amp = np.abs(lag)
    pk = np.argmax(amp)
    tau_ns = tau_axis_ns[pk]
    peak_amp = amp[pk]

    # SNR: peak / median of region away from peak
    off_mask = np.ones(len(amp), dtype=bool)
    lo = max(0, pk - 5)
    hi = min(len(amp), pk + 6)
    off_mask[lo:hi] = False
    snr = peak_amp / np.median(amp[off_mask]) if off_mask.any() else np.nan

    return tau_ns, snr, peak_amp


def lag_delay_baseline_series(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float,
    df_hz: float,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
    pad_factor: int = 16,
    bad_channels: np.ndarray | tuple | None = None,
    min_snr: float = 0.0,
) -> BaselineResult | None:
    r"""Baseline from lag-spectrum delays fitted across captures.

    For each capture, computes the IFFT delay, then fits:

    .. math::

        \tau(h) = A\,\sin h + B\,\cos h + \tau_{\rm inst}

    where :math:`A = (b_{\rm ew}/c)\cos\delta` and
    :math:`B = (b_{\rm ns}/c)\sin L\,\cos\delta`.

    Parameters
    ----------
    corr_dc : (N_cap, N_CH) complex array
        DC-corrected visibility per capture.
    dec_rad : float
        Source declination.
    df_hz : float
        Channel spacing in Hz.
    lat_rad : float
        Observatory latitude in radians.

    Returns ``None`` if fewer than 4 valid captures.
    """
    n_cap = corr_dc.shape[0]
    tau_ns = np.full(n_cap, np.nan)
    snr_arr = np.full(n_cap, np.nan)

    for i in range(n_cap):
        t, s, _ = lag_delay_single_capture(
            corr_dc[i], df_hz, pad_factor=pad_factor, bad_channels=bad_channels
        )
        tau_ns[i] = t
        snr_arr[i] = s

    # Filter by SNR
    valid = np.isfinite(tau_ns)
    if min_snr > 0:
        valid &= snr_arr >= min_snr
    n = int(valid.sum())
    if n < 4:
        return None

    sin_h = np.sin(ha_rad[valid])
    cos_h = np.cos(ha_rad[valid])
    tau_valid = tau_ns[valid]

    # Design matrix: [sin(h), cos(h), 1]
    X = np.column_stack([sin_h, cos_h, np.ones(n)])
    result = np.linalg.lstsq(X, tau_valid, rcond=None)
    coeffs = result[0]  # [A_ns, B_ns, tau_inst_ns]
    A_ns, B_ns, tau_inst_ns = coeffs

    cos_dec = np.cos(dec_rad)
    sin_lat = np.sin(lat_rad)

    # A_ns = (b_ew / c) * cos_dec  [in ns, so multiply by 1e-9]
    b_ew = A_ns * 1e-9 * C_LIGHT_MS / cos_dec
    # B_ns = (b_ns / c) * sin_lat * cos_dec  [in ns]
    b_ns = (
        B_ns * 1e-9 * C_LIGHT_MS / (sin_lat * cos_dec)
        if abs(sin_lat * cos_dec) > 1e-15
        else np.nan
    )

    # Uncertainty from residual covariance
    resid = tau_valid - X @ coeffs
    resid_var = np.sum(resid**2) / max(1, n - 3)
    try:
        cov = resid_var * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)

    A_err = np.sqrt(max(0, cov[0, 0]))
    B_err = np.sqrt(max(0, cov[1, 1]))
    b_ew_err = np.abs(A_err * 1e-9 * C_LIGHT_MS / cos_dec)
    b_ns_err = (
        np.abs(B_err * 1e-9 * C_LIGHT_MS / (sin_lat * cos_dec))
        if abs(sin_lat * cos_dec) > 1e-15
        else np.inf
    )

    chi2 = resid_var

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=b_ns,
        b_ns_err_m=b_ns_err,
        method="lag_delay",
        chi2_reduced=chi2,
        n_points=n,
        metadata={
            "tau_inst_ns": tau_inst_ns,
            "A_ns_per_sinh": A_ns,
            "B_ns_per_cosh": B_ns,
            "cov_AB": cov[:2, :2],
            "median_snr": float(np.nanmedian(snr_arr[valid])),
            "tau_ns": tau_ns,
            "snr": snr_arr,
        },
    )


# ===================================================================
# Method 4 — Nonlinear least-squares fringe fit
# ===================================================================


def _nls_residual(
    q_vec: np.ndarray,
    ha_rad: np.ndarray,
    vis_real: np.ndarray,
) -> np.ndarray:
    r"""Residual function for the NLS fringe fit.

    The model is :math:`F(h) = A\cos\psi + B\sin\psi` where
    :math:`\psi = 2\pi(Q_{\rm ew}\sin h + Q_{\rm ns}\cos h)`.

    For fixed :math:`(Q_{\rm ew}, Q_{\rm ns})`, the model is **linear** in
    :math:`(A, B)`, so we solve for them analytically and return the residual.
    """
    Q_ew, Q_ns = q_vec
    psi = 2.0 * np.pi * (Q_ew * np.sin(ha_rad) + Q_ns * np.cos(ha_rad))
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Solve the 2x2 linear system for A, B
    # [sum(cos^2)  sum(cos*sin)] [A]   [sum(y*cos)]
    # [sum(cos*sin) sum(sin^2) ] [B] = [sum(y*sin)]
    cc = np.sum(cos_psi * cos_psi)
    ss = np.sum(sin_psi * sin_psi)
    cs = np.sum(cos_psi * sin_psi)
    yc = np.sum(vis_real * cos_psi)
    ys = np.sum(vis_real * sin_psi)

    det = cc * ss - cs * cs
    if abs(det) < 1e-30:
        return vis_real  # degenerate — return raw data as residual
    A = (ss * yc - cs * ys) / det
    B = (cc * ys - cs * yc) / det

    return vis_real - (A * cos_psi + B * sin_psi)


def nls_baseline_single_channel(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float,
    freq_hz: float,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
    min_samples: int = 16,
    q_ew_init: float | None = None,
    q_ns_init: float = 0.0,
) -> BaselineResult | None:
    r"""Baseline from nonlinear least-squares fit of the full fringe equation.

    Fits the model:

    .. math::

        F(h) = A\cos(2\pi\psi) + B\sin(2\pi\psi), \quad
        \psi = Q_{\rm ew}\sin h + Q_{\rm ns}\cos h

    where :math:`Q_{\rm ew} = (b_{\rm ew}/\lambda)\cos\delta` and
    :math:`Q_{\rm ns} = (b_{\rm ns}/\lambda)\sin L\,\cos\delta`.

    The parameters :math:`(A, B)` are solved analytically at each iteration
    (they are linear), while :math:`(Q_{\rm ew}, Q_{\rm ns})` are optimised
    via ``scipy.optimize.least_squares``.

    Parameters
    ----------
    q_ew_init : float, optional
        Initial guess for :math:`Q_{\rm ew}` (default: estimated from the
        nominal 20 m baseline).
    q_ns_init : float
        Initial guess for :math:`Q_{\rm ns}` (default: 0).

    Returns ``None`` if fewer than *min_samples* valid points.
    """
    from scipy.optimize import least_squares

    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    valid = np.isfinite(vis)
    n = int(valid.sum())
    if n < min_samples:
        return None

    vis_real = vis[valid].real
    ha_valid = ha[valid]

    lam = C_LIGHT_MS / freq_hz
    cos_dec = np.cos(dec_rad)
    sin_lat = np.sin(lat_rad)

    if q_ew_init is None:
        q_ew_init = 20.0 / lam * cos_dec  # nominal 20 m baseline

    sol = least_squares(
        _nls_residual,
        x0=[q_ew_init, q_ns_init],
        args=(ha_valid, vis_real),
        method="lm",
    )

    Q_ew, Q_ns = sol.x
    b_ew = Q_ew * lam / cos_dec
    b_ns = Q_ns * lam / (sin_lat * cos_dec) if abs(sin_lat * cos_dec) > 1e-15 else np.nan

    # Uncertainty from Jacobian
    # J = d(residual)/d(Q_ew, Q_ns) at solution
    # cov ≈ (J^T J)^{-1} * s^2 where s^2 = sum(resid^2) / (n - 4)
    resid = sol.fun
    dof = max(1, n - 4)  # 4 params: Q_ew, Q_ns, A, B
    s2 = np.sum(resid**2) / dof
    try:
        JtJ = sol.jac.T @ sol.jac
        cov_Q = s2 * np.linalg.inv(JtJ)
        Q_ew_err = np.sqrt(max(0, cov_Q[0, 0]))
        Q_ns_err = np.sqrt(max(0, cov_Q[1, 1]))
    except np.linalg.LinAlgError:
        Q_ew_err = np.inf
        Q_ns_err = np.inf

    b_ew_err = np.abs(Q_ew_err * lam / cos_dec)
    b_ns_err = (
        np.abs(Q_ns_err * lam / (sin_lat * cos_dec))
        if abs(sin_lat * cos_dec) > 1e-15
        else np.inf
    )

    # Recover A, B at the solution
    psi = 2.0 * np.pi * (Q_ew * np.sin(ha_valid) + Q_ns * np.cos(ha_valid))
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    cc = np.sum(cos_psi**2)
    ss = np.sum(sin_psi**2)
    cs = np.sum(cos_psi * sin_psi)
    det = cc * ss - cs * cs
    yc = np.sum(vis_real * cos_psi)
    ys = np.sum(vis_real * sin_psi)
    A_fit = (ss * yc - cs * ys) / det if abs(det) > 1e-30 else np.nan
    B_fit = (cc * ys - cs * yc) / det if abs(det) > 1e-30 else np.nan

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=b_ns,
        b_ns_err_m=b_ns_err,
        method="nls",
        chi2_reduced=s2,
        n_points=n,
        metadata={
            "Q_ew": Q_ew,
            "Q_ns": Q_ns,
            "A": A_fit,
            "B": B_fit,
            "amplitude": np.sqrt(A_fit**2 + B_fit**2) if np.isfinite(A_fit) else np.nan,
            "phase_offset_rad": np.arctan2(-B_fit, A_fit) if np.isfinite(A_fit) else np.nan,
            "nfev": sol.nfev,
            "cost": sol.cost,
        },
    )


def nls_baseline_broadband(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float,
    f_sky_hz: np.ndarray,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
    bad_channels: np.ndarray | tuple | None = None,
    band_hz: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[BaselineResult | None, np.ndarray]:
    """Band-averaged NLS fringe-fit baseline across channels.

    Returns (aggregated result, per-channel structured array).
    """
    n_ch = corr_dc.shape[1]
    f_sky_hz = np.asarray(f_sky_hz)

    good = np.ones(n_ch, dtype=bool)
    if bad_channels is not None:
        good[list(bad_channels)] = False
    if band_hz is not None:
        good &= (f_sky_hz >= band_hz[0]) & (f_sky_hz <= band_hz[1])

    per_ch = np.full(
        n_ch,
        np.nan,
        dtype=[("b_ew_m", float), ("b_ew_err_m", float),
               ("b_ns_m", float), ("b_ns_err_m", float)],
    )

    for k in np.where(good)[0]:
        res = nls_baseline_single_channel(
            ha_rad, corr_dc[:, k], dec_rad, f_sky_hz[k],
            lat_rad=lat_rad, **kwargs,
        )
        if res is not None:
            per_ch[k]["b_ew_m"] = res.b_ew_m
            per_ch[k]["b_ew_err_m"] = res.b_ew_err_m
            per_ch[k]["b_ns_m"] = res.b_ns_m
            per_ch[k]["b_ns_err_m"] = res.b_ns_err_m

    valid = np.isfinite(per_ch["b_ew_m"])
    if not valid.any():
        return None, per_ch

    b_ew_vals = per_ch["b_ew_m"][valid]
    b_ns_vals = per_ch["b_ns_m"][valid]

    med_ew = np.median(b_ew_vals)
    p16_ew, p84_ew = np.percentile(b_ew_vals, [16, 84])
    scatter_ew = 0.5 * (p84_ew - p16_ew)

    med_ns = np.nanmedian(b_ns_vals)
    finite_ns = np.isfinite(b_ns_vals)
    if finite_ns.sum() > 1:
        p16_ns, p84_ns = np.percentile(b_ns_vals[finite_ns], [16, 84])
        scatter_ns = 0.5 * (p84_ns - p16_ns)
    else:
        scatter_ns = np.inf

    result = BaselineResult(
        b_ew_m=med_ew,
        b_ew_err_m=scatter_ew,
        b_ns_m=med_ns,
        b_ns_err_m=scatter_ns,
        method="nls_broadband",
        n_points=int(valid.sum()),
    )
    return result, per_ch


# ===================================================================
# Combined estimator
# ===================================================================


def _ivw(values: np.ndarray, errors: np.ndarray, sigma_clip: float = 3.0):
    """Inverse-variance weighted mean with MAD-based sigma clipping."""
    finite = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if not finite.any():
        return np.nan, np.inf, np.zeros(len(values), dtype=bool)

    v, e = values[finite], errors[finite]
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad if mad > 0 else np.inf
    keep_local = np.abs(v - med) <= sigma_clip * scale
    if not keep_local.any():
        keep_local[:] = True

    w = 1.0 / e[keep_local] ** 2
    combined = np.sum(w * v[keep_local]) / np.sum(w)
    err = 1.0 / np.sqrt(np.sum(w))

    # Map back to original indexing
    keep_full = np.zeros(len(values), dtype=bool)
    finite_idx = np.where(finite)[0]
    keep_full[finite_idx[keep_local]] = True
    return combined, err, keep_full


def combined_baseline(
    results: list[BaselineResult],
    *,
    sigma_clip: float = 3.0,
) -> BaselineResult:
    """Inverse-variance weighted combination of multiple baseline estimates.

    Combines both :math:`b_{\\rm ew}` and :math:`b_{\\rm ns}` independently.

    Parameters
    ----------
    results : list of BaselineResult
        Individual estimates (``None`` entries are silently skipped).
    sigma_clip : float
        Reject estimates farther than this many sigma from the median.
    """
    good = [r for r in results if r is not None]
    if not good:
        return BaselineResult(
            b_ew_m=np.nan, b_ew_err_m=np.inf, method="combined", n_points=0
        )

    b_ew = np.array([r.b_ew_m for r in good])
    e_ew = np.array([r.b_ew_err_m for r in good])
    b_ns = np.array([r.b_ns_m for r in good])
    e_ns = np.array([r.b_ns_err_m for r in good])

    ew_combined, ew_err, keep_ew = _ivw(b_ew, e_ew, sigma_clip)
    ns_combined, ns_err, keep_ns = _ivw(b_ns, e_ns, sigma_clip)

    methods = list({good[i].method for i in np.where(keep_ew)[0]})

    return BaselineResult(
        b_ew_m=ew_combined,
        b_ew_err_m=ew_err,
        b_ns_m=ns_combined,
        b_ns_err_m=ns_err,
        method="combined",
        n_points=int(keep_ew.sum()),
        metadata={
            "methods": methods,
            "n_clipped_ew": int((~keep_ew).sum()),
            "n_clipped_ns": int((~keep_ns).sum()),
        },
    )
