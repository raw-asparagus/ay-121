"""Baseline determination from interferometric fringe observations.

Four independent methods are provided, each returning a standardised
:class:`BaselineResult`.  A combined inverse-variance estimator is also
available.

Methods
-------
1. **FFT in x-coordinate** — remap visibility to ``x = cos(dec) sin(h)``
   and FFT to find the fringe spatial frequency (B_EW only).
2. **Phase slope** — fit unwrapped phase against ``[sin(h), cos(h)]``
   per channel (B_EW and B_NS).
3. **Lag-spectrum delay** — IFFT the cross-spectrum to obtain geometric
   delay, then fit ``tau = A sin(h) + B cos(h) + tau_inst`` (B_EW and B_NS).
4. **NLS fringe fit** — nonlinear least-squares fit of the full fringe
   equation with separable linear/nonlinear parameters (B_EW and B_NS).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import C_LIGHT_MS, NCH_LAT_DEG
from .geometry import geometric_delay_s, x_coordinate


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
    "phase_slope_baseline_single_channel",
    "phase_slope_baseline_broadband",
    "lag_delay_single_capture",
    "lag_delay_baseline_series",
    "nls_baseline_single_channel",
    "nls_baseline_broadband",
    "grid_search_baseline",
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
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
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
    sin_lat = np.sin(lat_rad)

    # Physical regressors: cos(δ)sin(h) and sin(L)cos(δ)cos(h)
    r_ew = cos_dec * np.sin(ha[valid])
    r_ns = sin_lat * cos_dec * np.cos(ha[valid])

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

    b_ew_vals = per_ch["b_ew_m"][valid]
    mean_ew = np.mean(b_ew_vals)
    std_ew = np.std(b_ew_vals) / np.sqrt(len(b_ew_vals)) if len(b_ew_vals) > 1 else np.inf

    b_ns_vals = per_ch["b_ns_m"][valid]
    finite_ns = np.isfinite(b_ns_vals)
    if finite_ns.sum() > 1:
        mean_ns = np.nanmean(b_ns_vals[finite_ns])
        std_ns = np.nanstd(b_ns_vals[finite_ns]) / np.sqrt(finite_ns.sum())
    else:
        mean_ns = np.nanmean(b_ns_vals) if finite_ns.any() else np.nan
        std_ns = np.inf

    result = BaselineResult(
        b_ew_m=mean_ew,
        b_ew_err_m=std_ew,
        b_ns_m=mean_ns,
        b_ns_err_m=std_ns,
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
    dec_rad: float | np.ndarray,
    df_hz: float,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
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
    sin_lat = np.sin(lat_rad)
    tau_valid = tau_ns[valid]

    # Physical regressors (delay in ns = b/c * geometry * 1e9)
    r_ew = cos_dec * np.sin(ha_rad[valid])  # (b_ew/c) * r_ew gives delay
    r_ns = sin_lat * cos_dec * np.cos(ha_rad[valid])

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


def _nls_residual_complex(
    b_vec: np.ndarray,
    ha_rad: np.ndarray,
    cos_dec: np.ndarray,
    sin_lat: float,
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
        + (b_ns / lam) * sin_lat * cos_dec * np.cos(ha_rad)
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
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
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
    sin_lat = np.sin(lat_rad)

    if b_ew_init is None:
        ps_res = phase_slope_baseline_single_channel(
            ha_rad, visibility_dc, dec_rad, freq_hz,
            lat_rad=lat_rad, min_samples=min_samples,
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
        args=(ha_valid, cos_dec_valid, sin_lat, freq_hz, vis_re, vis_im),
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
        + (b_ns / lam) * sin_lat * cos_dec_valid * np.cos(ha_valid)
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
        b_ns_m=b_ns,
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
    sin_lat: float,
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
    cd_sin_h = cos_dec * sin_h        # cos(δ_i) sin(h_i)
    cd_cos_h = sin_lat * cos_dec * cos_h  # sin(L) cos(δ_i) cos(h_i)

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

    alpha = np.array([[0.5 * d2_ew, d2_cross], [d2_cross, 0.5 * d2_ns]])

    # Scale by estimated variance: σ² = S²_min / dof
    sigma2 = S2_min / max(1, dof)

    try:
        cov = sigma2 * np.linalg.inv(alpha)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.inf)

    return alpha, cov


def grid_search_baseline(
    ha_rad: np.ndarray,
    visibility_dc: np.ndarray,
    dec_rad: float | np.ndarray,
    freq_hz: float,
    *,
    lat_rad: float = np.deg2rad(NCH_LAT_DEG),
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
    cos_dec_mean = np.mean(cos_dec_valid)
    sin_lat = np.sin(lat_rad)

    # --- Default coarse search range from phase-slope estimate ---
    if q_ew_range is None or q_ns_range is None:
        ps_res = phase_slope_baseline_single_channel(
            ha_rad, visibility_dc, dec_rad, freq_hz,
            lat_rad=lat_rad, min_samples=min_samples,
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
    S2_coarse = _evaluate_S2_grid(Q_ew_coarse, Q_ns_coarse, sin_h, cos_h, cos_dec_valid, sin_lat, vis_re, vis_im)

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
    S2_fine = _evaluate_S2_grid(Q_ew_fine, Q_ns_fine, sin_h, cos_h, cos_dec_valid, sin_lat, vis_re, vis_im)

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
        b_ns_m=b_ns,
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
