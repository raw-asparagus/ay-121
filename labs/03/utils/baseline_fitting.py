"""Baseline determination from interferometric fringe observations.

Seven independent methods are provided, each returning a standardised
:class:`BaselineResult`.  A combined inverse-variance estimator is also
available.

Methods
-------
1. **FFT in x-coordinate** — remap visibility to ``x = cos(dec) sin(h)``
   and FFT to find the fringe spatial frequency (B_EW only).
1a. **STFT fringe-frequency fit** — short-time Fourier transform of the
   band-averaged time series to measure the local fringe frequency at each
   epoch, then OLS fit of f_f(h) to the fringe-frequency equation
   (lab-manual prescription; AY121-Lab3).
2. **Phase slope** — fit unwrapped phase against ``[sin(h), cos(h)]``
   per channel (B_EW and B_NS).
3. **Lag-spectrum delay** — IFFT the cross-spectrum to obtain geometric
   delay, then fit ``tau = A sin(h) + B cos(h) + tau_inst`` (B_EW and B_NS).
4. **NLS fringe fit** — nonlinear least-squares fit of the full complex
   fringe equation with separable linear/nonlinear parameters (B_EW and B_NS).
4a. **NLS real fringe fit** — same as Method 4 but fits only Re[V], matching
   the lab-manual F(h_s) = A cos(psi) + B sin(psi) prescription exactly.
5. **Grid search** — brute-force 2-D grid over (Q_ew, Q_ns); lab-manual
   prescribed technique.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_LIGHT_MS, NCH_LAT_DEG, OMEGA_EARTH_RAD_S
from .geometry import x_coordinate

_LAT_RAD_NCH = np.deg2rad(NCH_LAT_DEG)
_SIN_LAT_NCH = np.sin(_LAT_RAD_NCH)


def _as_dec_array(dec_rad: float | np.ndarray, n: int) -> np.ndarray:
    """Broadcast scalar or array declination to shape (n,)."""
    d = np.asarray(dec_rad, dtype=float)
    if d.ndim == 0:
        return np.broadcast_to(d, n).copy()
    return d

__all__ = [
    "BaselineResult",
    "GridSearchResult",
    "fft_baseline_single_channel",
    "fft_baseline_broadband",
    "stft_fringe_frequency",
    "stft_baseline_from_ff",
    "stft_baseline",
    "phase_slope_baseline_single_channel",
    "phase_slope_baseline_broadband",
    "lag_delay_single_capture",
    "lag_delay_baseline_series",
    "nls_baseline_single_channel",
    "nls_baseline_broadband",
    "nls_real_baseline_single_channel",
    "nls_real_baseline_broadband",
    "grid_search_baseline",
    "brute_force_1d_Qew_sweep",
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
    dec_rad: float | np.ndarray,
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
    3. Zero-pad and FFT.
    4. Find peak frequency via parabolic interpolation.
    5. Convert: ``B_EW = |f_x| * c / freq_hz``.

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

    ivw_ew, ivw_ew_err, _ = _robust_mean(per_ch["b_ew_m"][valid], per_ch["b_ew_err_m"][valid])

    result = BaselineResult(
        b_ew_m=ivw_ew,
        b_ew_err_m=ivw_ew_err,
        method="fft_broadband",
        n_points=int(valid.sum()),
        metadata={
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
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    min_samples: int = 4,
) -> BaselineResult | None:
    r"""Baseline from unwrapped-phase slope.

    Fits the per-capture phase using the physical regressors:

    .. math::

        \phi_i = \frac{2\pi\nu}{c}\bigl[
            b_{\rm ew}\cos\delta_i\,\sin h_i
          + b_{\rm ns}\sin L\,\cos\delta_i\,\cos h_i
        \bigr] + \phi_c

    The design matrix is :math:`[\cos\delta_i\sin h_i,\;\sin L\cos\delta_i
    \cos h_i,\;1]`, so the fit coefficients directly give:

    .. math::

        b_{\rm ew} = \frac{A\,c}{2\pi\nu}, \quad
        b_{\rm ns} = \frac{B\,c}{2\pi\nu}.

    Parameters
    ----------
    dec_rad : float or (N,) array
        Source declination in radians (scalar or per-capture).

    Returns ``None`` if fewer than *min_samples* valid points.
    """
    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    valid = np.isfinite(vis)
    n = int(valid.sum())
    if n < min_samples:
        return None

    dec = _as_dec_array(dec_rad, len(ha))
    phase = np.unwrap(np.angle(vis[valid]))
    cos_dec = np.cos(dec[valid])

    # Physical regressors: cos(δ)sin(h) and sin(L)cos(δ)cos(h)
    r_ew = cos_dec * np.sin(ha[valid])
    r_ns = _SIN_LAT_NCH * cos_dec * np.cos(ha[valid])

    X = np.column_stack([r_ew, r_ns, np.ones(n)])
    result = np.linalg.lstsq(X, phase, rcond=None)
    coeffs = result[0]  # [A, B, phi_0]
    A, B, phi_0 = coeffs

    scale = C_LIGHT_MS / (2.0 * np.pi * freq_hz)
    b_ew = A * scale
    b_ns = B * scale

    # Uncertainty from residual covariance
    resid = phase - X @ coeffs
    resid_var = np.sum(resid**2) / max(1, n - 3)
    try:
        cov = resid_var * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)

    A_err = np.sqrt(max(0, cov[0, 0]))
    B_err = np.sqrt(max(0, cov[1, 1]))
    b_ew_err = np.abs(A_err * scale)
    b_ns_err = np.abs(B_err * scale)

    chi2 = resid_var  # already divided by dof

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=np.abs(b_ns),
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
    dec_rad: float | np.ndarray,
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
        n_ch,
        np.nan,
        dtype=[("b_ew_m", float), ("b_ew_err_m", float),
               ("b_ns_m", float), ("b_ns_err_m", float)],
    )

    for k in np.where(good)[0]:
        res = phase_slope_baseline_single_channel(
            ha_rad, corr_dc[:, k], dec_rad, f_sky_hz[k], **kwargs
        )
        if res is not None:
            per_ch[k]["b_ew_m"] = res.b_ew_m
            per_ch[k]["b_ew_err_m"] = res.b_ew_err_m
            per_ch[k]["b_ns_m"] = res.b_ns_m
            per_ch[k]["b_ns_err_m"] = res.b_ns_err_m

    valid = np.isfinite(per_ch["b_ew_m"])
    if not valid.any():
        return None, per_ch

    ivw_ew, ivw_ew_err, _ = _robust_mean(per_ch["b_ew_m"][valid], per_ch["b_ew_err_m"][valid])
    ivw_ns, ivw_ns_err, _ = _robust_mean(per_ch["b_ns_m"][valid], per_ch["b_ns_err_m"][valid])

    result = BaselineResult(
        b_ew_m=ivw_ew,
        b_ew_err_m=ivw_ew_err,
        b_ns_m=ivw_ns,
        b_ns_err_m=ivw_ns_err,
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
        Channels to zero (flagged data).

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

    # Zero bad channels (no interpolation — avoids injecting fabricated data)
    if bad_channels is not None:
        for ch in bad_channels:
            if 0 <= ch < n_ch:
                vis[ch] = 0.0

    n_pad = n_ch * pad_factor
    lag = np.fft.fftshift(np.fft.ifft(vis, n=n_pad))
    tau_axis_ns = np.fft.fftshift(np.fft.fftfreq(n_pad, d=df_hz)) * 1e9

    amp = np.abs(lag)
    pk = np.argmax(amp)
    peak_amp = amp[pk]

    # Parabolic (3-point) interpolation for sub-bin delay refinement
    if 0 < pk < len(amp) - 1:
        y0, y1, y2 = amp[pk - 1], amp[pk], amp[pk + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-30:
            delta = 0.5 * (y0 - y2) / denom
            dtau = tau_axis_ns[1] - tau_axis_ns[0]
            tau_ns = tau_axis_ns[pk] + delta * dtau
        else:
            tau_ns = tau_axis_ns[pk]
    else:
        tau_ns = tau_axis_ns[pk]

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
    dec_rad: float | np.ndarray,
    df_hz: float,
    *,
    pad_factor: int = 16,
    bad_channels: np.ndarray | tuple | None = None,
    min_snr: float = 0.0,
) -> BaselineResult | None:
    r"""Baseline from lag-spectrum delays fitted across captures.

    For each capture, computes the IFFT delay, then fits using the
    physical regressors (with per-capture declination):

    .. math::

        \tau_i = b_{\rm ew}\,\frac{\cos\delta_i\,\sin h_i}{c}
               + b_{\rm ns}\,\frac{\sin L\,\cos\delta_i\,\cos h_i}{c}
               + \tau_{\rm inst}

    Parameters
    ----------
    dec_rad : float or (N_cap,) array
        Source declination in radians (scalar or per-capture).

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

    dec = _as_dec_array(dec_rad, n_cap)
    cos_dec = np.cos(dec[valid])
    tau_valid = tau_ns[valid]

    # Physical regressors (delay in ns = b/c * geometry * 1e9)
    r_ew = cos_dec * np.sin(ha_rad[valid])  # (b_ew/c) * r_ew gives delay
    r_ns = _SIN_LAT_NCH * cos_dec * np.cos(ha_rad[valid])

    X = np.column_stack([r_ew, r_ns, np.ones(n)])
    result = np.linalg.lstsq(X, tau_valid, rcond=None)
    coeffs = result[0]  # [A_ns, B_ns, tau_inst_ns]
    A_ns, B_ns, tau_inst_ns = coeffs

    # A_ns = b_ew / c * 1e9,  B_ns = b_ns / c * 1e9
    b_ew = A_ns * 1e-9 * C_LIGHT_MS
    b_ns = B_ns * 1e-9 * C_LIGHT_MS

    # Uncertainty from residual covariance
    resid = tau_valid - X @ coeffs
    resid_var = np.sum(resid**2) / max(1, n - 3)
    try:
        cov = resid_var * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)

    A_err = np.sqrt(max(0, cov[0, 0]))
    B_err = np.sqrt(max(0, cov[1, 1]))
    b_ew_err = np.abs(A_err * 1e-9 * C_LIGHT_MS)
    b_ns_err = np.abs(B_err * 1e-9 * C_LIGHT_MS)

    chi2 = resid_var

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=np.abs(b_ns),
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


def _nls_residual_complex(
    b_vec: np.ndarray,
    ha_rad: np.ndarray,
    cos_dec: np.ndarray,
    freq_hz: float,
    vis_re: np.ndarray,
    vis_im: np.ndarray,
) -> np.ndarray:
    r"""Residual function for the complex NLS fringe fit.

    Parameters *b_vec* = ``[b_ew, b_ns]`` in metres.  The phase argument
    uses per-capture declination via *cos_dec*.
    """
    b_ew, b_ns = b_vec
    lam = C_LIGHT_MS / freq_hz
    psi = 2.0 * np.pi * (
        (b_ew / lam) * cos_dec * np.sin(ha_rad)
        + (b_ns / lam) * _SIN_LAT_NCH * cos_dec * np.cos(ha_rad)
    )
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Normal equations: X = [cos_psi, sin_psi], solve X a = y for real and imag
    cc = np.dot(cos_psi, cos_psi)
    ss = np.dot(sin_psi, sin_psi)
    cs = np.dot(cos_psi, sin_psi)
    det = cc * ss - cs * cs
    if abs(det) < 1e-30:
        return np.concatenate([vis_re, vis_im])

    # Real part: Re(V) = A cos_psi + B sin_psi
    yc_r = np.dot(vis_re, cos_psi)
    ys_r = np.dot(vis_re, sin_psi)
    A = (ss * yc_r - cs * ys_r) / det
    B = (cc * ys_r - cs * yc_r) / det

    # Imaginary part: Im(V) = C cos_psi + D sin_psi
    yc_i = np.dot(vis_im, cos_psi)
    ys_i = np.dot(vis_im, sin_psi)
    C = (ss * yc_i - cs * ys_i) / det
    D = (cc * ys_i - cs * yc_i) / det

    resid_re = vis_re - (A * cos_psi + B * sin_psi)
    resid_im = vis_im - (C * cos_psi + D * sin_psi)
    return np.concatenate([resid_re, resid_im])


def nls_baseline_single_channel(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    min_samples: int = 16,
    b_ew_init: float | None = None,
    b_ns_init: float = 0.0,
) -> BaselineResult | None:
    r"""Baseline from nonlinear least-squares fit of the full complex visibility.

    Fits both real and imaginary parts simultaneously.  The nonlinear
    parameters are :math:`(b_{\rm ew}, b_{\rm ns})` in metres; the linear
    parameters :math:`(A, B, C, D)` are solved analytically at each step.
    Per-capture declination is supported.

    Parameters
    ----------
    dec_rad : float or (N,) array
        Source declination in radians (scalar or per-capture).
    b_ew_init : float, optional
        Initial guess for :math:`b_{\rm ew}` in metres (default: seeded
        from the phase-slope result).
    b_ns_init : float
        Initial guess for :math:`b_{\rm ns}` in metres (default: 0).

    Returns ``None`` if fewer than *min_samples* valid points.
    """
    from scipy.optimize import least_squares

    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    valid = np.isfinite(vis)
    n = int(valid.sum())
    if n < min_samples:
        return None

    dec = _as_dec_array(dec_rad, len(ha))
    vis_re = vis[valid].real
    vis_im = vis[valid].imag
    ha_valid = ha[valid]
    cos_dec_valid = np.cos(dec[valid])

    if b_ew_init is None:
        ps_res = phase_slope_baseline_single_channel(
            ha_rad, visibility_dc, dec_rad, freq_hz, min_samples=min_samples,
        )
        if ps_res is not None:
            b_ew_init = ps_res.b_ew_m
            if np.isfinite(ps_res.b_ns_m):
                b_ns_init = ps_res.b_ns_m
        else:
            fft_res = fft_baseline_single_channel(
                ha_rad, visibility_dc, dec_rad, freq_hz, min_samples=min_samples,
            )
            b_ew_init = fft_res.b_ew_m if fft_res is not None else 20.0

    sol = least_squares(
        _nls_residual_complex,
        x0=[b_ew_init, b_ns_init],
        args=(ha_valid, cos_dec_valid, freq_hz, vis_re, vis_im),
        method="lm",
    )

    b_ew, b_ns = sol.x

    # Uncertainty: residual has 2n points, 6 effective params
    resid = sol.fun
    dof = max(1, 2 * n - 6)
    s2 = np.sum(resid**2) / dof
    try:
        JtJ = sol.jac.T @ sol.jac
        cov_b = s2 * np.linalg.inv(JtJ)
        b_ew_err = np.sqrt(max(0, cov_b[0, 0]))
        b_ns_err = np.sqrt(max(0, cov_b[1, 1]))
    except np.linalg.LinAlgError:
        b_ew_err = np.inf
        b_ns_err = np.inf

    # Recover linear params at solution
    lam = C_LIGHT_MS / freq_hz
    psi = 2.0 * np.pi * (
        (b_ew / lam) * cos_dec_valid * np.sin(ha_valid)
        + (b_ns / lam) * _SIN_LAT_NCH * cos_dec_valid * np.cos(ha_valid)
    )
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    cc = np.dot(cos_psi, cos_psi)
    ss = np.dot(sin_psi, sin_psi)
    cs = np.dot(cos_psi, sin_psi)
    det = cc * ss - cs * cs
    if abs(det) > 1e-30:
        A = (ss * np.dot(vis_re, cos_psi) - cs * np.dot(vis_re, sin_psi)) / det
        B = (cc * np.dot(vis_re, sin_psi) - cs * np.dot(vis_re, cos_psi)) / det
        C = (ss * np.dot(vis_im, cos_psi) - cs * np.dot(vis_im, sin_psi)) / det
        D = (cc * np.dot(vis_im, sin_psi) - cs * np.dot(vis_im, cos_psi)) / det
    else:
        A = B = C = D = np.nan

    amplitude = np.sqrt(A**2 + B**2 + C**2 + D**2) / np.sqrt(2) if np.isfinite(A) else np.nan

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=np.abs(b_ns),
        b_ns_err_m=b_ns_err,
        method="nls",
        chi2_reduced=s2,
        n_points=n,
        metadata={
            "A_re": A, "B_re": B, "C_im": C, "D_im": D,
            "amplitude": amplitude,
            "nfev": sol.nfev,
            "cost": sol.cost,
        },
    )


def nls_baseline_broadband(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    f_sky_hz: np.ndarray,
    *,
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
            ha_rad, corr_dc[:, k], dec_rad, f_sky_hz[k], **kwargs,
        )
        if res is not None:
            per_ch[k]["b_ew_m"] = res.b_ew_m
            per_ch[k]["b_ew_err_m"] = res.b_ew_err_m
            per_ch[k]["b_ns_m"] = res.b_ns_m
            per_ch[k]["b_ns_err_m"] = res.b_ns_err_m

    valid = np.isfinite(per_ch["b_ew_m"])
    if not valid.any():
        return None, per_ch

    ivw_ew, ivw_ew_err, _ = _robust_mean(per_ch["b_ew_m"][valid], per_ch["b_ew_err_m"][valid])
    ivw_ns, ivw_ns_err, _ = _robust_mean(per_ch["b_ns_m"][valid], per_ch["b_ns_err_m"][valid])

    result = BaselineResult(
        b_ew_m=ivw_ew,
        b_ew_err_m=ivw_ew_err,
        b_ns_m=ivw_ns,
        b_ns_err_m=ivw_ns_err,
        method="nls_broadband",
        n_points=int(valid.sum()),
    )
    return result, per_ch


# ===================================================================
# Method 5 — Brute-force grid search
# ===================================================================


@dataclass(frozen=True)
class GridSearchResult:
    """Result of a brute-force grid search over (Q_ew, Q_ns).

    Attributes
    ----------
    b_ew_m, b_ew_err_m : float
        Best-fit east-west baseline and uncertainty.
    b_ns_m, b_ns_err_m : float
        Best-fit north-south baseline and uncertainty.
    Q_ew_grid : (M,) array
        Grid values of Q_ew.
    Q_ns_grid : (K,) array
        Grid values of Q_ns.
    S2_map : (K, M) array
        Sum-of-squared residuals at each grid point (rows = Q_ns, cols = Q_ew).
    Q_ew_best, Q_ns_best : float
        Grid point with minimum S^2.
    S2_min : float
        Minimum sum-of-squared residuals.
    alpha_matrix : (2, 2) array
        Curvature matrix (numerical second derivatives of S^2 at the minimum).
    covariance_matrix : (2, 2) array
        Covariance matrix = inverse of the curvature matrix.
    n_points : int
        Number of data points used.
    """

    b_ew_m: float
    b_ew_err_m: float
    b_ns_m: float
    b_ns_err_m: float
    # Coarse grid
    Q_ew_coarse: np.ndarray
    Q_ns_coarse: np.ndarray
    S2_coarse: np.ndarray
    # Fine grid
    Q_ew_grid: np.ndarray
    Q_ns_grid: np.ndarray
    S2_map: np.ndarray
    # Best fit
    Q_ew_best: float
    Q_ns_best: float
    S2_min: float
    alpha_matrix: np.ndarray
    covariance_matrix: np.ndarray
    n_points: int


def _evaluate_S2_grid(
    Q_ew_grid: np.ndarray,
    Q_ns_grid: np.ndarray,
    sin_h: np.ndarray,
    cos_h: np.ndarray,
    cos_dec: np.ndarray,
    vis_re: np.ndarray,
    vis_im: np.ndarray,
) -> np.ndarray:
    """Evaluate S² on a 2-D grid over (Q_ew, Q_ns), with per-capture cos(δ).

    The phase argument at each grid point is:
    ψ_i = 2π (Q_ew cos δ_i sin h_i + Q_ns sin L cos δ_i cos h_i)
    where Q_ew = b_ew/λ and Q_ns = b_ns/λ.
    """
    n_ns = len(Q_ns_grid)
    n_ew = len(Q_ew_grid)
    S2 = np.empty((n_ns, n_ew), dtype=float)

    # Pre-compute per-capture geometry
    cd_sin_h = cos_dec * sin_h                  # cos(δ_i) sin(h_i)
    cd_cos_h = _SIN_LAT_NCH * cos_dec * cos_h   # sin(L) cos(δ_i) cos(h_i)

    for j, Q_ns in enumerate(Q_ns_grid):
        for i, Q_ew in enumerate(Q_ew_grid):
            psi = 2.0 * np.pi * (Q_ew * cd_sin_h + Q_ns * cd_cos_h)
            cp = np.cos(psi)
            sp = np.sin(psi)

            cc = np.dot(cp, cp)
            ss = np.dot(sp, sp)
            cs = np.dot(cp, sp)
            det = cc * ss - cs * cs

            if abs(det) < 1e-30:
                S2[j, i] = np.inf
                continue

            yc_r = np.dot(vis_re, cp)
            ys_r = np.dot(vis_re, sp)
            A = (ss * yc_r - cs * ys_r) / det
            B = (cc * ys_r - cs * yc_r) / det

            yc_i = np.dot(vis_im, cp)
            ys_i = np.dot(vis_im, sp)
            C = (ss * yc_i - cs * ys_i) / det
            D = (cc * ys_i - cs * yc_i) / det

            resid_re = vis_re - (A * cp + B * sp)
            resid_im = vis_im - (C * cp + D * sp)
            S2[j, i] = np.dot(resid_re, resid_re) + np.dot(resid_im, resid_im)

    return S2


def _curvature_and_covariance(
    S2_map: np.ndarray,
    Q_ew_grid: np.ndarray,
    Q_ns_grid: np.ndarray,
    i_best: int,
    j_best: int,
    S2_min: float,
    dof: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute curvature matrix [α] and covariance at the minimum.

    The curvature matrix [α] is built from the second derivatives of S²
    (eqn 4 in the fitting notes).  Since S² is *not* normalised by
    the data variance, the covariance is **not** simply [α]⁻¹.  We
    estimate σ² = S²_min / dof and return:

        cov = σ² · [α]⁻¹ = (S²_min / dof) · [α]⁻¹
    """
    n_ew = len(Q_ew_grid)
    n_ns = len(Q_ns_grid)
    dQ_ew = Q_ew_grid[1] - Q_ew_grid[0] if n_ew > 1 else 1.0
    dQ_ns = Q_ns_grid[1] - Q_ns_grid[0] if n_ns > 1 else 1.0

    if 0 < i_best < n_ew - 1:
        d2_ew = (S2_map[j_best, i_best + 1] - 2 * S2_min + S2_map[j_best, i_best - 1]) / dQ_ew**2
    else:
        d2_ew = np.inf

    if 0 < j_best < n_ns - 1:
        d2_ns = (S2_map[j_best + 1, i_best] - 2 * S2_min + S2_map[j_best - 1, i_best]) / dQ_ns**2
    else:
        d2_ns = np.inf

    if 0 < i_best < n_ew - 1 and 0 < j_best < n_ns - 1:
        d2_cross = (
            S2_map[j_best + 1, i_best + 1]
            - S2_map[j_best + 1, i_best - 1]
            - S2_map[j_best - 1, i_best + 1]
            + S2_map[j_best - 1, i_best - 1]
        ) / (4.0 * dQ_ew * dQ_ns)
    else:
        d2_cross = 0.0

    alpha = np.array([[0.5 * d2_ew, 0.5 * d2_cross], [0.5 * d2_cross, 0.5 * d2_ns]])

    # Scale by estimated variance: σ² = S²_min / dof
    sigma2 = S2_min / max(1, dof)

    try:
        cov = sigma2 * np.linalg.inv(alpha)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)

    return alpha, cov


def brute_force_1d_Qew_sweep(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    q_ew_range: tuple[float, float],
    n_points: int = 4000,
    min_samples: int = 16,
) -> dict | None:
    r"""1-D brute-force sweep of :math:`\mathcal{S}^2` vs :math:`Q_{\rm ew}` at :math:`Q_{\rm ns}=0`.

    Implements **Step 1** of the fitting procedure prescribed in
    ``src/ugradio/lab_interf/fitting_notes_2017.pdf`` (and §8.4.3 of
    ``interf.tex``): for each guessed value of :math:`Q_{\rm ew}`, the
    linear-in-parameters coefficients :math:`(A, B)` in the fringe model

    .. math::

        F(h) = A \cos(2\pi\,Q_{\rm ew}\,\cos\delta\,\sin h)
             + B \sin(2\pi\,Q_{\rm ew}\,\cos\delta\,\sin h)

    are determined analytically by least squares (separable variable
    projection), and the resulting sum of squared residuals
    :math:`\mathcal{S}^2(Q_{\rm ew})` is tabulated. The grid minimum
    locates the east-west baseline; a well-defined, sharp minimum is a
    visual sanity check for the 2-D brute-force search and the proper
    NLS fit that follow.

    The complex visibility is fitted with *separate* (A, B) for the real
    and imaginary parts (i.e. four linear parameters and the single
    nonlinear parameter :math:`Q_{\rm ew}`), matching the treatment in
    :func:`grid_search_baseline`.

    Parameters
    ----------
    ha_rad
        Per-capture hour angle in radians.
    visibility_dc
        Per-capture complex visibility (DC-corrected, single channel).
    dec_rad
        Source declination in radians (scalar or per-capture).
    freq_hz
        Channel frequency in Hz.
    q_ew_range
        ``(lo, hi)`` sweep range in wavelengths.
    n_points
        Number of grid points across ``q_ew_range`` (fine, densely sampled).

    Returns
    -------
    dict or None
        Dict with keys
        ``{"Q_ew_grid", "S2", "Q_ew_best", "b_ew_best_m", "S2_min", "n_points"}``
        or ``None`` if fewer than ``min_samples`` valid captures.
    """
    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)
    valid = np.isfinite(vis)
    n = int(valid.sum())
    if n < min_samples:
        return None

    dec = _as_dec_array(dec_rad, len(ha))
    cos_dec_valid = np.cos(dec[valid])
    ha_valid = ha[valid]
    sin_h = np.sin(ha_valid)
    cos_h = np.cos(ha_valid)
    vis_re = vis[valid].real
    vis_im = vis[valid].imag

    Q_ew_grid = np.linspace(q_ew_range[0], q_ew_range[1], n_points)
    S2 = _evaluate_S2_grid(
        Q_ew_grid,
        np.array([0.0]),
        sin_h,
        cos_h,
        cos_dec_valid,
        vis_re,
        vis_im,
    )[0]  # shape (n_points,)

    idx = int(np.argmin(S2))
    Q_ew_best = float(Q_ew_grid[idx])
    lam = C_LIGHT_MS / freq_hz
    return {
        "Q_ew_grid": Q_ew_grid,
        "S2": S2,
        "Q_ew_best": Q_ew_best,
        "b_ew_best_m": Q_ew_best * lam,
        "S2_min": float(S2[idx]),
        "n_points": n,
    }


def grid_search_baseline(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    q_ew_range: tuple[float, float] | None = None,
    q_ns_range: tuple[float, float] | None = None,
    n_coarse: int = 200,
    n_fine: int = 300,
    fine_halfwidth_ew: float = 1.0,
    fine_halfwidth_ns: float = 10.0,
    min_samples: int = 16,
) -> GridSearchResult | None:
    r"""Two-pass brute-force grid search over :math:`(Q_{\rm ew}, Q_{\rm ns})`.

    The grid is over :math:`Q_{\rm ew} = b_{\rm ew}/\lambda` and
    :math:`Q_{\rm ns} = b_{\rm ns}/\lambda`.  Per-capture declination is
    folded into the phase computation at each grid point.

    **Pass 1 (coarse):** wide grid to locate the global minimum.
    **Pass 2 (fine):** narrow grid centred on the coarse minimum.

    Parameters
    ----------
    dec_rad : float or (N,) array
        Source declination (scalar or per-capture).
    q_ew_range, q_ns_range : (lo, hi), optional
        Coarse search range in Q-units (= b/λ).
    """
    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    valid = np.isfinite(vis)
    n = int(valid.sum())
    if n < min_samples:
        return None

    dec = _as_dec_array(dec_rad, len(ha))
    vis_re = vis[valid].real
    vis_im = vis[valid].imag
    ha_valid = ha[valid]
    sin_h = np.sin(ha_valid)
    cos_h = np.cos(ha_valid)
    cos_dec_valid = np.cos(dec[valid])

    lam = C_LIGHT_MS / freq_hz

    # --- Default coarse search range from phase-slope estimate ---
    if q_ew_range is None or q_ns_range is None:
        ps_res = phase_slope_baseline_single_channel(
            ha_rad, visibility_dc, dec_rad, freq_hz, min_samples=min_samples,
        )
        if ps_res is not None:
            q_ew_center = ps_res.b_ew_m / lam
            q_ns_center = ps_res.b_ns_m / lam if np.isfinite(ps_res.b_ns_m) else 0.0
        else:
            q_ew_center = 20.0 / lam
            q_ns_center = 0.0
        if q_ew_range is None:
            q_ew_range = (q_ew_center - 50.0, q_ew_center + 50.0)
        if q_ns_range is None:
            q_ns_range = (q_ns_center - 100.0, q_ns_center + 100.0)

    # --- Pass 1: coarse grid ---
    Q_ew_coarse = np.linspace(q_ew_range[0], q_ew_range[1], n_coarse)
    Q_ns_coarse = np.linspace(q_ns_range[0], q_ns_range[1], n_coarse)
    S2_coarse = _evaluate_S2_grid(Q_ew_coarse, Q_ns_coarse, sin_h, cos_h, cos_dec_valid, vis_re, vis_im)

    idx_flat = np.argmin(S2_coarse)
    j_c, i_c = np.unravel_index(idx_flat, S2_coarse.shape)
    Q_ew_coarse_best = Q_ew_coarse[i_c]
    Q_ns_coarse_best = Q_ns_coarse[j_c]

    # --- Pass 2: fine grid centred on coarse minimum ---
    Q_ew_fine = np.linspace(
        Q_ew_coarse_best - fine_halfwidth_ew,
        Q_ew_coarse_best + fine_halfwidth_ew,
        n_fine,
    )
    Q_ns_fine = np.linspace(
        Q_ns_coarse_best - fine_halfwidth_ns,
        Q_ns_coarse_best + fine_halfwidth_ns,
        n_fine,
    )
    S2_fine = _evaluate_S2_grid(Q_ew_fine, Q_ns_fine, sin_h, cos_h, cos_dec_valid, vis_re, vis_im)

    idx_flat = np.argmin(S2_fine)
    j_best, i_best = np.unravel_index(idx_flat, S2_fine.shape)
    Q_ew_best = Q_ew_fine[i_best]
    Q_ns_best = Q_ns_fine[j_best]
    S2_min = S2_fine[j_best, i_best]

    # Q = b/λ, so b = Q * λ
    b_ew = Q_ew_best * lam
    b_ns = Q_ns_best * lam

    # --- Step 3: curvature matrix from the fine grid ---
    # dof = 2N (real + imag data points) - 6 (A, B, C, D, Q_ew, Q_ns)
    dof = max(1, 2 * n - 6)
    alpha, cov = _curvature_and_covariance(
        S2_fine, Q_ew_fine, Q_ns_fine, i_best, j_best, S2_min, dof,
    )

    Q_ew_err = np.sqrt(max(0, cov[0, 0]))
    Q_ns_err = np.sqrt(max(0, cov[1, 1]))

    b_ew_err = np.abs(Q_ew_err * lam)
    b_ns_err = np.abs(Q_ns_err * lam)

    return GridSearchResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=np.abs(b_ns),
        b_ns_err_m=b_ns_err,
        Q_ew_coarse=Q_ew_coarse,
        Q_ns_coarse=Q_ns_coarse,
        S2_coarse=S2_coarse,
        Q_ew_grid=Q_ew_fine,
        Q_ns_grid=Q_ns_fine,
        S2_map=S2_fine,
        Q_ew_best=Q_ew_best,
        Q_ns_best=Q_ns_best,
        S2_min=S2_min,
        alpha_matrix=alpha,
        covariance_matrix=cov,
        n_points=n,
    )


# ===================================================================
# Method 1a — STFT fringe-frequency fit
# ===================================================================


def _stft_one_chip(
    ha_chip: np.ndarray,
    vis_chip: np.ndarray,
    *,
    window_size: int,
    step_size: int,
    pad_factor: int,
    min_snr: float,
    predicted_ff_hz_fn=None,
    search_tol_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run an STFT fringe-frequency scan over one contiguous chip.

    Resamples the chip onto a uniform HA grid (mean cadence) before
    sliding the window so that the FFT frequency axis is well defined,
    then converts the dimensionless frequency in cycles-per-radian-of-HA
    back to Hz via :math:`\\omega_\\oplus`.

    If ``predicted_ff_hz_fn`` is supplied (a callable that takes an
    HA-in-radians scalar and returns the *expected* fringe frequency in
    Hz from a baseline prior), the FFT peak search is restricted to a
    band of half-width ``search_tol_hz`` around the prediction. This
    keeps the per-window measurement honest at low SNR — at the Bessel
    nulls of the disk envelope the fringe collapses below noise and the
    unconstrained peak finder otherwise locks onto a sidelobe several
    mHz away from the true location, producing wild outliers.
    """
    ha = np.asarray(ha_chip, dtype=float)
    vis = np.asarray(vis_chip, dtype=complex)

    n_in = len(ha)
    if n_in < window_size:
        return np.array([]), np.array([]), np.array([])

    # --- Uniform-HA resampling --------------------------------------------
    # The NCH cadence has up to ~40% jitter inside a chip; an FFT on the
    # raw samples produces a biased frequency axis. Resample to a uniform
    # HA grid spanning the chip with the same number of points.
    ha_uniform = np.linspace(ha[0], ha[-1], n_in)
    finite = np.isfinite(vis)
    if finite.sum() < window_size:
        return np.array([]), np.array([]), np.array([])
    re_u = np.interp(ha_uniform, ha[finite], vis[finite].real)
    im_u = np.interp(ha_uniform, ha[finite], vis[finite].imag)
    vis_u = re_u + 1j * im_u

    dha = ha_uniform[1] - ha_uniform[0]
    if dha <= 0 or not np.isfinite(dha):
        return np.array([]), np.array([]), np.array([])

    starts = list(range(0, n_in - window_size + 1, step_size))
    if not starts:
        return np.array([]), np.array([]), np.array([])

    ha_centers = np.empty(len(starts))
    ff_hz = np.full(len(starts), np.nan)
    snr_arr = np.full(len(starts), np.nan)

    n_pad = window_size * pad_factor
    # FFT frequency axis in cycles-per-radian-of-HA. Convert to Hz at the
    # end by multiplying by omega_earth (since dHA/dt = omega_earth).
    freq_axis_cyc_per_rad = np.fft.fftshift(np.fft.fftfreq(n_pad, d=dha))
    freq_axis_hz = freq_axis_cyc_per_rad * OMEGA_EARTH_RAD_S
    for idx, s in enumerate(starts):
        seg = vis_u[s : s + window_size].copy()
        ha_centers[idx] = np.mean(ha_uniform[s : s + window_size])

        spectrum = np.fft.fftshift(np.fft.fft(seg, n=n_pad))
        amp = np.abs(spectrum)

        # Search all frequencies except the DC bin itself.
        search_mask = freq_axis_cyc_per_rad != 0.0

        # Optional prior-constrained peak search
        if predicted_ff_hz_fn is not None and search_tol_hz is not None:
            f_pred_hz = float(abs(predicted_ff_hz_fn(ha_centers[idx])))
            search_mask &= (
                (np.abs(freq_axis_hz) >= f_pred_hz - search_tol_hz)
                & (np.abs(freq_axis_hz) <= f_pred_hz + search_tol_hz)
            )

        peaks = _fft_sorted_local_peaks(amp, search_mask)
        if len(peaks) == 0:
            continue

        f_peak_cyc_per_rad = _fft_peak_refined(freq_axis_cyc_per_rad, amp, peaks[0])
        peak_amp = amp[peaks[0]]

        # SNR: peak / median of off-peak background.
        off_mask = np.ones(n_pad, dtype=bool)
        lo = max(0, peaks[0] - 5)
        hi = min(n_pad, peaks[0] + 6)
        off_mask[lo:hi] = False
        snr = peak_amp / np.median(amp[off_mask]) if off_mask.any() else np.nan

        snr_arr[idx] = snr
        # Take absolute value — fringe frequency is a positive quantity;
        # the sign degeneracy (positive vs negative frequency peak) is
        # resolved downstream by the geometric model.
        ff_hz[idx] = abs(f_peak_cyc_per_rad * OMEGA_EARTH_RAD_S)

    return ha_centers, ff_hz, snr_arr


def stft_fringe_frequency(
    ha_rad: np.ndarray,
    vis_band_avg: np.ndarray,
    *,
    window_size: int = 64,
    step_size: int = 16,
    pad_factor: int = 8,
    min_snr: float = 3.0,
    chip_slices: list[slice] | None = None,
    predicted_ff_hz_fn=None,
    search_tol_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Measure the local fringe frequency per window via STFT.

    Slides a segment of *window_size* captures along the
    complex visibility time series. For each window the DFT
    peak (sub-bin refined via parabolic interpolation) gives the local
    fringe frequency :math:`\hat f_{f,i}` at the window centre.

    Two robustness features are essential for the NCH dataset:

    * **Per-chip operation.** When *chip_slices* is supplied, the STFT
      is run independently on each chip so that no window straddles an
      inter-chip dead-time gap (which would otherwise corrupt the FFT
      frequency axis). The window-centre arrays from all chips are
      concatenated.
    * **Uniform-HA resampling.** Inside each chip the captures are
      linearly interpolated onto a uniform HA grid before the FFT runs.
      The NCH capture cadence has up to ~40% intra-chip jitter; without
      resampling the FFT bin width is biased and the recovered fringe
      frequency drifts.

    The fringe frequency is computed in cycles-per-radian-of-HA from the
    uniform-HA FFT and converted to Hz at the end via :math:`\omega_\oplus`.
    This avoids any dependence on the (inaccurate) median capture cadence.

    Parameters
    ----------
    ha_rad : (N,) array
        Hour angle at each capture in radians (monotonically increasing
        within each chip).
    vis_band_avg : (N,) complex array
        Band-averaged complex visibility (DC-corrected).
    window_size : int
        Number of captures per window.
    step_size : int
        Window step in captures.
    pad_factor : int
        Zero-padding factor for sub-bin frequency resolution.
    min_snr : float
        Windows with peak-to-median SNR below this are returned as NaN.
    chip_slices : list of slice, optional
        Row slices that partition ``ha_rad`` / ``vis_band_avg`` into
        contiguous chips. If ``None``, the full series is treated as a
        single chip.

    Returns
    -------
    ha_centers : (M,) array
        Mean hour angle of each window in radians (concatenated across
        chips).
    ff_hz : (M,) array
        Measured fringe frequency in Hz (NaN where SNR < min_snr).
    snr_arr : (M,) array
        Peak-to-median SNR for each window.
    """
    ha = np.asarray(ha_rad, dtype=float)
    vis = np.asarray(vis_band_avg, dtype=complex)

    if chip_slices is None:
        chip_slices = [slice(0, len(ha))]

    parts_h, parts_f, parts_s = [], [], []
    for sl in chip_slices:
        h_c, f_c, s_c = _stft_one_chip(
            ha[sl], vis[sl],
            window_size=window_size,
            step_size=step_size,
            pad_factor=pad_factor,
            min_snr=min_snr,
            predicted_ff_hz_fn=predicted_ff_hz_fn,
            search_tol_hz=search_tol_hz,
        )
        if len(h_c) > 0:
            parts_h.append(h_c)
            parts_f.append(f_c)
            parts_s.append(s_c)

    if not parts_h:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(parts_h),
        np.concatenate(parts_f),
        np.concatenate(parts_s),
    )


def stft_baseline_from_ff(
    ha_centers: np.ndarray,
    ff_hz: np.ndarray,
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    snr: np.ndarray | None = None,
    min_snr: float = 3.0,
    sigma_clip: float = 3.0,
    max_iterations: int = 5,
) -> BaselineResult | None:
    r"""Fit measured fringe frequencies to the fringe-frequency equation.

    The lab manual fringe-frequency equation ([AY121-Lab3] eq. ``fringefreq``,
    boxed) is:

    .. math::

        \frac{f_{f,Hz}}{\omega_\oplus} =
            \frac{b_{\rm ew}}{\lambda}\cos\delta\,\cos h
          - \frac{b_{\rm ns}}{\lambda}\sin L\,\cos\delta\,\sin h

    This is linear in :math:`(b_{\rm ew}, b_{\rm ns})`, so the solution
    is a standard linear least-squares problem. The design matrix is

    .. math::

        X = \bigl[\omega_\oplus\cos\delta_i\cos h_i/\lambda,\;
                  -\omega_\oplus\sin L\cos\delta_i\sin h_i/\lambda\bigr]

    and we fit :math:`\hat f_{f,i} = X_i\,[b_{\rm ew}, b_{\rm ns}]^T`.

    Two robustness layers are applied to handle the well-known failure
    modes of the STFT method:

    * **SNR weighting.** Windows are weighted by their SNR (the per-window
      peak-to-noise ratio from :func:`stft_fringe_frequency`). High-SNR
      windows — those near the fringe-amplitude peaks of the Bessel
      envelope — drive the fit; near-null windows that just made the SNR
      cut still contribute but with reduced influence.
    * **Iterative sigma clipping.** Near the Bessel nulls of the disk
      modulating function the fringe amplitude can collapse to noise and
      the FFT peak finder occasionally locks onto a sidelobe — producing
      a per-window outlier of several mHz. After the first weighted fit
      the residuals are scaled by the weighted residual standard deviation
      and any window outside ``sigma_clip`` is masked; the fit is then
      re-run on the surviving subset until the mask stabilises.

    Both layers can be disabled by passing ``snr=None`` (uniform weights)
    and ``sigma_clip=np.inf``.

    Parameters
    ----------
    ha_centers : (M,) array
        Window-centre hour angles in radians.
    ff_hz : (M,) array
        Measured fringe frequencies in Hz (NaN entries are ignored).
    dec_rad : float or (M,) array
        Source declination at each window centre.
    freq_hz : float
        Band-centre sky frequency in Hz (used to compute λ).
    snr : (M,) array, optional
        Per-window SNR. Windows with ``snr < min_snr`` are excluded;
        the surviving entries are used as weights in the WLS fit.
    min_snr : float
        Minimum SNR threshold (applied only when *snr* is provided).
    sigma_clip : float
        Outlier rejection threshold in units of the weighted residual
        standard deviation. Windows whose residual exceeds
        ``sigma_clip * sigma_resid`` are dropped and the fit is re-run.
    max_iterations : int
        Maximum number of sigma-clipping iterations.

    Returns ``None`` if fewer than 4 valid windows.
    """
    ha = np.asarray(ha_centers)
    ff = np.asarray(ff_hz)
    dec = _as_dec_array(dec_rad, len(ha))

    valid_full = np.isfinite(ff)
    if snr is not None:
        snr_arr = np.asarray(snr, dtype=float)
        valid_full &= np.isfinite(snr_arr) & (snr_arr >= min_snr)
    else:
        snr_arr = np.ones_like(ff)

    if int(valid_full.sum()) < 4:
        return None

    lam = C_LIGHT_MS / freq_hz
    cos_dec_full = np.cos(dec)

    # Full-length design matrix; we'll subset by mask each iteration.
    r_ew_full = OMEGA_EARTH_RAD_S * cos_dec_full * np.cos(ha) / lam
    r_ns_full = -OMEGA_EARTH_RAD_S * _SIN_LAT_NCH * cos_dec_full * np.sin(ha) / lam
    X_full = np.column_stack([r_ew_full, r_ns_full])

    mask = valid_full.copy()
    coeffs = np.zeros(2)
    cov = np.full((2, 2), np.inf)
    resid = np.zeros(int(mask.sum()))
    resid_var = np.inf

    for _it in range(max_iterations):
        n_it = int(mask.sum())
        if n_it < 4:
            break
        Xm = X_full[mask]
        ym = ff[mask]
        wm = snr_arr[mask] if snr is not None else np.ones(n_it)

        # Weighted least squares: solve (W^{1/2} X) b = (W^{1/2} y).
        sw = np.sqrt(wm)
        Xw = Xm * sw[:, None]
        yw = ym * sw
        coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

        resid = ym - Xm @ coeffs
        # Weighted residual std (avoid double-counting weights)
        resid_var = float(np.average(resid ** 2, weights=wm))
        sigma_resid = np.sqrt(max(resid_var, 1e-30))

        new_mask = mask.copy()
        new_mask[mask] = np.abs(resid) <= sigma_clip * sigma_resid
        if int(new_mask.sum()) == n_it:
            break  # converged
        mask = new_mask

    n = int(mask.sum())
    if n < 4:
        return None

    Xm = X_full[mask]
    wm = snr_arr[mask] if snr is not None else np.ones(n)
    sw = np.sqrt(wm)
    Xw = Xm * sw[:, None]

    # Covariance: (X^T W X)^{-1} * (residual variance)
    try:
        cov_unscaled = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        cov_unscaled = np.full((2, 2), np.inf)
    cov = cov_unscaled * resid_var

    b_ew, b_ns = coeffs
    b_ew_err = np.sqrt(max(0, cov[0, 0]))
    b_ns_err = np.sqrt(max(0, cov[1, 1]))

    # Build a full-length model array (NaN at excluded windows) so the
    # caller can plot measured vs model on a single shared axis.
    ff_model_full = np.full_like(ff, np.nan)
    ff_model_full[mask] = X_full[mask] @ coeffs

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=np.abs(b_ns),
        b_ns_err_m=b_ns_err,
        method="stft_ff",
        chi2_reduced=resid_var,
        n_points=n,
        metadata={
            "ha_centers": ha_centers,
            "ff_hz_measured": ff_hz,
            "ff_hz_model": ff_model_full,
            "residuals_hz": resid,
            "valid_mask": mask,
            "snr": snr_arr if snr is not None else None,
        },
    )


def stft_baseline(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    f_sky_hz: np.ndarray,
    *,
    bad_channels: np.ndarray | tuple | None = None,
    band_hz: tuple[float, float] | None = None,
    window_size: int = 64,
    step_size: int = 16,
    pad_factor: int = 8,
    min_snr: float = 3.0,
    chip_slices: list[slice] | None = None,
    b_ew_prior_m: float | None = None,
    b_ns_prior_m: float = 0.0,
    search_tol_hz: float | None = None,
) -> BaselineResult | None:
    """STFT fringe-frequency baseline — band-average then fit.

    Convenience wrapper: band-averages *corr_dc* over good channels, calls
    :func:`stft_fringe_frequency` to extract per-window fringe frequencies,
    then calls :func:`stft_baseline_from_ff` to fit the baseline.

    Parameters
    ----------
    ha_rad : (N_cap,) array
        Hour angles in radians.
    corr_dc : (N_cap, N_ch) complex array
        DC-corrected complex visibilities.
    dec_rad : float or (N_cap,) array
        Source declination(s) in radians.
    f_sky_hz : (N_ch,) array
        Sky frequency per channel in Hz.
    bad_channels, band_hz
        Channel masking (same convention as other broadband methods).
    window_size, step_size, pad_factor, min_snr
        Passed to :func:`stft_fringe_frequency`.

    Returns ``None`` if no valid windows remain after masking.
    """
    n_ch = corr_dc.shape[1]
    f_sky_hz = np.asarray(f_sky_hz)

    good = np.ones(n_ch, dtype=bool)
    if bad_channels is not None:
        good[list(bad_channels)] = False
    if band_hz is not None:
        good &= (f_sky_hz >= band_hz[0]) & (f_sky_hz <= band_hz[1])

    if not good.any():
        return None

    vis_band_avg = np.nanmean(corr_dc[:, good], axis=1)

    # Representative sky frequency for λ
    freq_hz = float(np.nanmedian(f_sky_hz[good]))

    # Scalar or array declination
    dec = _as_dec_array(dec_rad, len(ha_rad))

    # Optional prior-constrained peak search. If a baseline prior is given,
    # we build a closure that returns the predicted f_f at any HA from the
    # prior baseline + per-capture declination, and pass it to the per-chip
    # STFT loop. The peak finder then restricts its search to a band of
    # ±search_tol_hz around the prediction.
    predicted_ff_hz_fn = None
    if b_ew_prior_m is not None and search_tol_hz is not None:
        ha_arr_for_interp = np.asarray(ha_rad, dtype=float)
        if dec.ndim > 0 and len(dec) == len(ha_arr_for_interp):
            def _dec_at(h):
                return float(np.interp(h, ha_arr_for_interp, dec))
        else:
            _dec_at_val = float(dec[0]) if dec.ndim > 0 else float(dec)
            def _dec_at(h):
                return _dec_at_val

        # Inline the fringe-frequency formula (avoids importing geometry to
        # prevent a circular import).
        def predicted_ff_hz_fn(h_rad: float) -> float:
            d = _dec_at(h_rad)
            cd = np.cos(d)
            lam_pred = C_LIGHT_MS / freq_hz
            return float(OMEGA_EARTH_RAD_S * (
                (b_ew_prior_m / lam_pred) * cd * np.cos(h_rad)
                - (b_ns_prior_m / lam_pred) * _SIN_LAT_NCH * cd * np.sin(h_rad)
            ))

    ha_centers, ff_hz, snr_arr = stft_fringe_frequency(
        ha_rad, vis_band_avg,
        window_size=window_size,
        step_size=step_size,
        pad_factor=pad_factor,
        min_snr=min_snr,
        chip_slices=chip_slices,
        predicted_ff_hz_fn=predicted_ff_hz_fn,
        search_tol_hz=search_tol_hz,
    )
    if len(ha_centers) == 0:
        return None

    # Interpolate declination to window centres for per-capture dec support
    if dec.ndim > 0 and len(dec) == len(ha_rad):
        dec_centers = np.interp(ha_centers, ha_rad, dec)
    else:
        dec_centers = float(dec[0]) if dec.ndim > 0 else float(dec)

    return stft_baseline_from_ff(
        ha_centers, ff_hz, dec_centers, freq_hz,
        snr=snr_arr, min_snr=min_snr,
    )


# ===================================================================
# Method 4a — NLS real fringe fit (lab-manual prescription)
# ===================================================================


def _nls_residual_real(
    b_vec: np.ndarray,
    ha_rad: np.ndarray,
    cos_dec: np.ndarray,
    freq_hz: float,
    vis_re: np.ndarray,
) -> np.ndarray:
    r"""Residual for the real-only NLS fringe fit.

    The lab-manual fringe response equation ([AY121-Lab3] eq. ``fringeresponse``,
    boxed):

    .. math::

        F(h_s) = A\cos(2\pi\nu\tau_g') + B\sin(2\pi\nu\tau_g')

    where :math:`2\pi\nu\tau_g' = \psi`.  Parameters *b_vec* = ``[b_ew, b_ns]``
    in metres; the linear parameters :math:`(A, B)` are solved analytically.
    Returns the N-point real residual vector.
    """
    b_ew, b_ns = b_vec
    lam = C_LIGHT_MS / freq_hz
    psi = 2.0 * np.pi * (
        (b_ew / lam) * cos_dec * np.sin(ha_rad)
        + (b_ns / lam) * _SIN_LAT_NCH * cos_dec * np.cos(ha_rad)
    )
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    cc = np.dot(cos_psi, cos_psi)
    ss = np.dot(sin_psi, sin_psi)
    cs = np.dot(cos_psi, sin_psi)
    det = cc * ss - cs * cs
    if abs(det) < 1e-30:
        return vis_re

    yc = np.dot(vis_re, cos_psi)
    ys = np.dot(vis_re, sin_psi)
    A = (ss * yc - cs * ys) / det
    B = (cc * ys - cs * yc) / det

    return vis_re - (A * cos_psi + B * sin_psi)


def nls_real_baseline_single_channel(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    min_samples: int = 16,
    b_ew_init: float | None = None,
    b_ns_init: float = 0.0,
) -> BaselineResult | None:
    r"""Baseline from NLS fit of the real fringe — lab-manual prescription.

    Fits the real part of the complex visibility to the lab-manual fringe
    response equation ([AY121-Lab3] eq. ``fringeresponse``, boxed):

    .. math::

        F(h_s) = A\cos\psi_i + B\sin\psi_i, \quad
        \psi_i = 2\pi\!\left[
            \frac{b_{\rm ew}}{\lambda}\cos\delta_i\sin h_i
          + \frac{b_{\rm ns}}{\lambda}\sin L\cos\delta_i\cos h_i
        \right]

    This is the exact real-valued fit described in the lab manual.  The
    nonlinear parameters :math:`(b_{\rm ew}, b_{\rm ns})` are optimised
    via Levenberg-Marquardt ([scipy] ``least_squares``); the linear
    parameters :math:`(A, B)` are solved analytically at each step
    (Golub-Pereyra separable NLS [Golub73]).

    Compared to :func:`nls_baseline_single_channel` (Method 4), this uses
    only Re[V] — N data points instead of 2N — so the uncertainties are
    larger by roughly :math:`\sqrt{2}` when signal-to-noise is equal.

    Parameters
    ----------
    dec_rad : float or (N,) array
        Source declination in radians (scalar or per-capture).
    b_ew_init : float, optional
        Initial guess for :math:`b_{\rm ew}` in metres (default: seeded
        from the phase-slope result).
    b_ns_init : float
        Initial guess for :math:`b_{\rm ns}` in metres (default: 0).

    Returns ``None`` if fewer than *min_samples* valid points.
    """
    from scipy.optimize import least_squares

    ha = np.asarray(ha_rad)
    vis = np.asarray(visibility_dc, dtype=complex)

    valid = np.isfinite(vis)
    n = int(valid.sum())
    if n < min_samples:
        return None

    dec = _as_dec_array(dec_rad, len(ha))
    vis_re = vis[valid].real
    ha_valid = ha[valid]
    cos_dec_valid = np.cos(dec[valid])

    if b_ew_init is None:
        ps_res = phase_slope_baseline_single_channel(
            ha_rad, visibility_dc, dec_rad, freq_hz, min_samples=min_samples,
        )
        if ps_res is not None:
            b_ew_init = ps_res.b_ew_m
            if np.isfinite(ps_res.b_ns_m):
                b_ns_init = ps_res.b_ns_m
        else:
            fft_res = fft_baseline_single_channel(
                ha_rad, visibility_dc, dec_rad, freq_hz, min_samples=min_samples,
            )
            b_ew_init = fft_res.b_ew_m if fft_res is not None else 20.0

    sol = least_squares(
        _nls_residual_real,
        x0=[b_ew_init, b_ns_init],
        args=(ha_valid, cos_dec_valid, freq_hz, vis_re),
        method="lm",
    )

    b_ew, b_ns = sol.x

    # Uncertainty: residual has N points, 4 effective params (A, B, b_ew, b_ns)
    resid = sol.fun
    dof = max(1, n - 4)
    s2 = np.sum(resid**2) / dof
    try:
        JtJ = sol.jac.T @ sol.jac
        cov_b = s2 * np.linalg.inv(JtJ)
        b_ew_err = np.sqrt(max(0, cov_b[0, 0]))
        b_ns_err = np.sqrt(max(0, cov_b[1, 1]))
    except np.linalg.LinAlgError:
        b_ew_err = np.inf
        b_ns_err = np.inf

    # Recover linear params at solution for metadata
    lam = C_LIGHT_MS / freq_hz
    psi = 2.0 * np.pi * (
        (b_ew / lam) * cos_dec_valid * np.sin(ha_valid)
        + (b_ns / lam) * _SIN_LAT_NCH * cos_dec_valid * np.cos(ha_valid)
    )
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    cc = np.dot(cos_psi, cos_psi)
    ss = np.dot(sin_psi, sin_psi)
    cs = np.dot(cos_psi, sin_psi)
    det = cc * ss - cs * cs
    if abs(det) > 1e-30:
        A = (ss * np.dot(vis_re, cos_psi) - cs * np.dot(vis_re, sin_psi)) / det
        B = (cc * np.dot(vis_re, sin_psi) - cs * np.dot(vis_re, cos_psi)) / det
    else:
        A = B = np.nan

    return BaselineResult(
        b_ew_m=np.abs(b_ew),
        b_ew_err_m=b_ew_err,
        b_ns_m=np.abs(b_ns),
        b_ns_err_m=b_ns_err,
        method="nls_real",
        chi2_reduced=s2,
        n_points=n,
        metadata={
            "A": A, "B": B,
            "amplitude": np.sqrt(A**2 + B**2) if np.isfinite(A) else np.nan,
            "nfev": sol.nfev,
            "cost": sol.cost,
        },
    )


def nls_real_baseline_broadband(
    ha_rad: np.ndarray,
    corr_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    f_sky_hz: np.ndarray,
    *,
    bad_channels: np.ndarray | tuple | None = None,
    band_hz: tuple[float, float] | None = None,
    **kwargs,
) -> tuple[BaselineResult | None, np.ndarray]:
    """Band-averaged real-NLS fringe-fit baseline across channels.

    Identical aggregation to :func:`nls_baseline_broadband` but calls
    :func:`nls_real_baseline_single_channel` (real part only).

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
        res = nls_real_baseline_single_channel(
            ha_rad, corr_dc[:, k], dec_rad, f_sky_hz[k], **kwargs,
        )
        if res is not None:
            per_ch[k]["b_ew_m"] = res.b_ew_m
            per_ch[k]["b_ew_err_m"] = res.b_ew_err_m
            per_ch[k]["b_ns_m"] = res.b_ns_m
            per_ch[k]["b_ns_err_m"] = res.b_ns_err_m

    valid = np.isfinite(per_ch["b_ew_m"])
    if not valid.any():
        return None, per_ch

    ivw_ew, ivw_ew_err, _ = _robust_mean(per_ch["b_ew_m"][valid], per_ch["b_ew_err_m"][valid])
    ivw_ns, ivw_ns_err, _ = _robust_mean(per_ch["b_ns_m"][valid], per_ch["b_ns_err_m"][valid])

    result = BaselineResult(
        b_ew_m=ivw_ew,
        b_ew_err_m=ivw_ew_err,
        b_ns_m=ivw_ns,
        b_ns_err_m=ivw_ns_err,
        method="nls_real_broadband",
        n_points=int(valid.sum()),
    )
    return result, per_ch


# ===================================================================
# Combined estimator
# ===================================================================


def _robust_mean(values: np.ndarray, errors: np.ndarray, sigma_clip: float = 3.0):
    """Unweighted mean with errors added in quadrature, after MAD-based sigma clipping."""
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

    n = int(keep_local.sum())
    combined = float(np.mean(v[keep_local]))
    err = float(np.sqrt(np.sum(e[keep_local] ** 2)) / n)

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
    """Unweighted mean combination of multiple baseline estimates (errors in quadrature).

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

    ew_combined, ew_err, keep_ew = _robust_mean(b_ew, e_ew, sigma_clip)
    ns_combined, ns_err, keep_ns = _robust_mean(b_ns, e_ns, sigma_clip)

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
