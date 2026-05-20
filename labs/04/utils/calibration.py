"""Calibration helpers: per-cell scalars, EBHIS anchoring, Tcal(t) fit.

Three groups of helpers live here:

* :func:`compute_cell_scalars` -- per-(session, cell) band-averaged
  noise-on/off powers, keyed by ``(session, gl, gb)``, used downstream
  by ``main_scan_qa.ipynb`` to apply ``Tcal_pol(t)``:

      T_sys_pol = Tcal_pol(t) * P_off_pol / (P_on_pol - P_off_pol)
      T_B_pol   = R_pol * T_sys_pol

* EBHIS anchoring (Bonn AllSky_profiles fetcher, peak finder,
  per-visit FS aggregation) used by ``main_scan_calibration.ipynb``
  to derive ``Tcal_pol(t)`` from the 2-peak scale-ratio method.

* 24 h PDT Fourier fold (`hour_pdt`, `fourier_design`, `fit_fourier`)
  used to forecast ``Tcal_pol(t)`` for the science load.
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


def compute_cell_scalars(
    records: list[dict],
    *,
    int_mask: np.ndarray,
    lo_freqs: tuple[float, float],
) -> dict:
    """Band-average corr00 / corr11 power per pol for each (session, cell, LO).

    For each science dump that survived upstream outlier filtering, the
    pol-0 and pol-1 powers are averaged over ``int_mask`` (the LO overlap
    with DC excluded), then aggregated per (session, gl, gb) into a tiny
    scalar table.

    Parameters
    ----------
    records : list of dict
        Dump records with ``corr00``, ``corr11``, ``session``, ``gl``,
        ``gb``, ``lo_mhz``, ``noise_on``, ``time``.
    int_mask : np.ndarray
        Boolean integration mask over the 1024-channel grid (overlap with
        DC bin excluded).
    lo_freqs : (float, float)
        The two LO frequencies that define the freq-switched pair.

    Returns
    -------
    dict
        ``(session, gl, gb) -> entry``.  Each entry has ``t_median`` and,
        for each ``lo_mhz`` in ``lo_freqs`` and each pol ``pol0``/``pol1``,
        the keys ``P_on_{pol}_{lo:g}``, ``P_off_{pol}_{lo:g}`` plus
        ``n_on_{lo:g}``, ``n_off_{lo:g}``.  Missing on/off groups yield
        NaN powers (and zero counts) so consumers can detect partial
        coverage.
    """
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        p0 = np.nanmean(r['corr00'][int_mask])
        p1 = np.nanmean(r['corr11'][int_mask])
        bucket[(r['session'], r['gl'], r['gb'])][(r['lo_mhz'], r['noise_on'])].append(
            (r['time'], p0, p1)
        )

    cell_scalars: dict = {}
    for scell, by_lo_noise in bucket.items():
        times = [t for buckets in by_lo_noise.values() for t, _, _ in buckets]
        entry = {'t_median': np.median(times)}
        for lo_mhz in lo_freqs:
            on  = by_lo_noise.get((lo_mhz, True),  [])
            off = by_lo_noise.get((lo_mhz, False), [])
            for pol_idx, pol_label in ((1, 'pol0'), (2, 'pol1')):
                P_on  = np.mean([row[pol_idx] for row in on])  if on  else np.nan
                P_off = np.mean([row[pol_idx] for row in off]) if off else np.nan
                entry[f'P_on_{pol_label}_{lo_mhz:g}']  = P_on
                entry[f'P_off_{pol_label}_{lo_mhz:g}'] = P_off
            entry[f'n_on_{lo_mhz:g}']  = len(on)
            entry[f'n_off_{lo_mhz:g}'] = len(off)
        cell_scalars[scell] = entry
    return cell_scalars


# ---------------------------------------------------------------------------
# EBHIS reference spectra (Bonn AllSky_profiles)
# ---------------------------------------------------------------------------

EBHIS_SERVER = 'https://www.astro.uni-bonn.de/hisurvey/AllSky_profiles'


def fetch_ebhis_spectrum(name, coord, beam_deg, cache_dir):
    """Beam-averaged EBHIS spectrum at ``coord``; returns ``(v_LSR_kms, T_B_K)``.

    Caches the raw ASCII server response per pointing in ``cache_dir`` so
    repeat calls in the same session avoid the network round-trip.  The
    Bonn response concatenates EBHIS, GASS and LAB spectra (each preceded
    by a ``%%<NAME>  N datapoints:`` header); only the EBHIS block is
    parsed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f'ebhis_{name}_beam{beam_deg:.2f}.txt'
    if cache.exists():
        print(f'  cache hit: {cache}')
        data = cache.read_text()
    else:
        gal = coord.galactic
        body = urllib.parse.urlencode({
            'coordinates': 'lb',
            'ral': f'{gal.l.deg:.4f}',
            'decb': f'{gal.b.deg:+.4f}',
            'beam': f'{beam_deg:.3f}',
            'ebhis': 'ebhis',
            'search': 'Search',
        }).encode()
        req = urllib.request.Request(
            f'{EBHIS_SERVER}/index.php', data=body,
            headers={'User-Agent': 'Mozilla/5.0',
                     'Content-Type': 'application/x-www-form-urlencoded'},
        )
        print(f'  querying Bonn server: {name} at '
              f'(l, b) = ({gal.l.deg:.2f}, {gal.b.deg:+.2f}), '
              f'beam = {beam_deg} deg')
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode('utf-8', errors='replace')
        m = re.search(r'href="(download\.php\?[^"]+)"', html)
        if not m:
            raise RuntimeError(f'no download link returned for {name}')
        dl_url = f'{EBHIS_SERVER}/{m.group(1)}'
        with urllib.request.urlopen(dl_url, timeout=30) as r:
            data = r.read().decode('utf-8', errors='replace')
        cache.write_text(data)
        print(f'  cached: {cache} ({len(data)} bytes)')

    rows = []
    in_ebhis = False
    for ln in data.splitlines():
        s = ln.strip()
        if s.startswith('%%') and 'EBHIS' in s:
            in_ebhis = True
            continue
        if s.startswith('%%') and 'EBHIS' not in s:
            in_ebhis = False
            continue
        if not in_ebhis or not s or s.startswith('%'):
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if not rows:
        raise RuntimeError(f'no EBHIS data parsed for {name}')
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def find_top2_peaks(y, x, min_sep, x_lo, x_hi):
    """Greedy: global max of ``y(x)`` inside ``[x_lo, x_hi]``, then the
    next-highest sample at least ``min_sep`` from any already accepted.

    Returns ``(x_peaks, y_at_peaks)`` sorted by ``x`` ascending.
    """
    in_win = (x >= x_lo) & (x <= x_hi) & np.isfinite(y)
    idx = np.where(in_win)[0]
    order = idx[np.argsort(-y[idx])]
    picked = []
    for i in order:
        if all(abs(x[i] - x[j]) >= min_sep for j in picked):
            picked.append(i)
            if len(picked) == 2:
                break
    if len(picked) < 2:
        raise RuntimeError(
            f'Only found {len(picked)} peaks separated by >= {min_sep}')
    picked.sort(key=lambda i: x[i])
    return (np.array([x[i] for i in picked]),
            np.array([y[i] for i in picked]))


# ---------------------------------------------------------------------------
# Per-visit FS aggregation
# ---------------------------------------------------------------------------

def band_mean_sem(scalars):
    """Mean + standard error of the mean (NaN-safe)."""
    n = len(scalars)
    if n == 0:
        return np.nan, np.nan
    mu = np.nanmean(scalars)
    if n < 2:
        return mu, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sigma = np.nanstd(scalars, ddof=1)
    return mu, sigma / np.sqrt(n)


def build_gain_visits(records, *, int_mask, pols, los, visit_gap_sec):
    """Group recal dumps into visits, band-average per (pol, LO, noise state).

    A "visit" is a maximal run of dumps at the same ``(session, target_id)``
    with adjacent dumps separated by no more than ``visit_gap_sec``.  For
    each visit and each (pol, LO) combination, the noise-on and noise-off
    band-averaged powers (over ``int_mask``) are accumulated with their
    SEMs.  ``records`` references are retained on each visit so callers
    can sample per-channel R(c) later.

    Parameters
    ----------
    pols : sequence of (key, pol_index)
        e.g. ``(('corr00', 0), ('corr11', 1))``.
    los : sequence of (lo_id, lo_mhz)
        e.g. ``((1, 1419.86), (2, 1421.14))``.
    """
    groups = defaultdict(list)
    for r in records:
        groups[(r['session'], r['target_id'])].append(r)
    visits = []
    for (sess, tid), grp in groups.items():
        grp = sorted(grp, key=lambda r: r['time'])
        runs = [[grp[0]]]
        for r in grp[1:]:
            if r['time'] - runs[-1][-1]['time'] > visit_gap_sec:
                runs.append([r])
            else:
                runs[-1].append(r)
        for vi, run in enumerate(runs):
            v = dict(session=sess, target_id=tid, visit_idx=vi,
                     t_mid=np.mean([r['time'] for r in run]),
                     alt_mean=np.mean([r['alt'] for r in run]),
                     n_dumps=len(run),
                     records=run)
            for pkey, pi in pols:
                for lid, lmhz in los:
                    on  = [r[pkey] for r in run
                           if r['noise_on'] and r['lo_mhz'] == lmhz]
                    off = [r[pkey] for r in run
                           if (not r['noise_on']) and r['lo_mhz'] == lmhz]
                    pon, spon = band_mean_sem(
                        np.array([np.nanmean(d[int_mask]) for d in on])
                        if on else np.array([]))
                    poff, spoff = band_mean_sem(
                        np.array([np.nanmean(d[int_mask]) for d in off])
                        if off else np.array([]))
                    v[f'p_on_pol{pi}_lo{lid}']   = pon
                    v[f'p_off_pol{pi}_lo{lid}']  = poff
                    v[f'sp_on_pol{pi}_lo{lid}']  = spon
                    v[f'sp_off_pol{pi}_lo{lid}'] = spoff
            visits.append(v)
    return sorted(visits, key=lambda v: v['t_mid'])


def add_per_pol_W_R(visits, bw_kms):
    """Annotate each visit with ``W_R_pol{i}`` and its SEM (in km/s units).

    ``W_R = BW * (P_LO1 - P_LO2) / P_LO2`` for noise-off, propagated from
    per-LO SEMs assuming independent errors.
    """
    for v in visits:
        for pi in (0, 1):
            p1 = v[f'p_off_pol{pi}_lo1']; p2 = v[f'p_off_pol{pi}_lo2']
            s1 = v[f'sp_off_pol{pi}_lo1']; s2 = v[f'sp_off_pol{pi}_lo2']
            if not (np.isfinite(p1) and np.isfinite(p2)
                    and np.isfinite(s1) and np.isfinite(s2)) or p2 == 0:
                v[f'W_R_pol{pi}']  = np.nan
                v[f'sW_R_pol{pi}'] = np.nan
                continue
            v[f'W_R_pol{pi}'] = bw_kms * (p1 - p2) / p2
            var_W = bw_kms ** 2 * ((s1 / p2) ** 2 + (p1 * s2 / p2 ** 2) ** 2)
            v[f'sW_R_pol{pi}'] = np.sqrt(var_W)


def avg_pointing_fs_diff(records, pointing, *, lo_freqs, pol='I'):
    """Per-channel FS ratio averaged over all noise-off dumps for one pointing.

    ``pol='I'`` (default) sums ``corr00 + corr11`` for Stokes I; ``pol=0``
    / ``pol=1`` returns the single-pol ratio used by the per-pol Tcal
    solve (the noise diode couples differently into each pol, so peaks
    must be found per pol independently to avoid mixing the pol-0
    coupling deficit into the pol-1 anchor).

    Returns ``(R, n_lo1, n_lo2)`` where ``R = (avg_LO1 - avg_LO2) / avg_LO2``
    on the 1024-channel grid, or ``(None, n_lo1, n_lo2)`` if either LO
    bucket is empty.  ``lo_freqs`` is ``(f1_mhz, f2_mhz)``.
    """
    if pol == 'I':
        def _extract(r):
            return r['corr00'] + r['corr11']
    elif pol == 0:
        def _extract(r):
            return r['corr00']
    elif pol == 1:
        def _extract(r):
            return r['corr11']
    else:
        raise ValueError(f"pol must be 'I', 0, or 1; got {pol!r}")

    f1, f2 = lo_freqs
    on_lo1 = [_extract(r) for r in records
              if r['target_id'] == pointing and (not r['noise_on'])
              and r['lo_mhz'] == f1]
    on_lo2 = [_extract(r) for r in records
              if r['target_id'] == pointing and (not r['noise_on'])
              and r['lo_mhz'] == f2]
    if not on_lo1 or not on_lo2:
        return None, len(on_lo1), len(on_lo2)
    avg_lo1 = np.nanmean(np.array(on_lo1), axis=0)
    avg_lo2 = np.nanmean(np.array(on_lo2), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        R = (avg_lo1 - avg_lo2) / avg_lo2
    return R, len(on_lo1), len(on_lo2)


def visit_R_pol_at_peaks(visit, pi, v_peaks, *, v_axis, halfwidth_kms, lo_freqs):
    """Per-visit R_pol(c) sampled as the mean over +/- ``halfwidth_kms``
    around each peak velocity in ``v_peaks``.  Returns an array of length
    ``len(v_peaks)`` (NaN where the visit has no LO1 or LO2 dumps).
    """
    f1, f2 = lo_freqs
    key = 'corr00' if pi == 0 else 'corr11'

    def _avg(lmhz):
        specs = [r[key] for r in visit['records']
                 if (not r['noise_on']) and r['lo_mhz'] == lmhz]
        if not specs:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(np.array(specs), axis=0)

    avg1, avg2 = _avg(f1), _avg(f2)
    if avg1 is None or avg2 is None:
        return np.full(len(v_peaks), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        R = (avg1 - avg2) / avg2
    out = np.full(len(v_peaks), np.nan)
    for k, vp in enumerate(v_peaks):
        in_band = (v_axis >= vp - halfwidth_kms) & \
                  (v_axis <= vp + halfwidth_kms)
        if in_band.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                out[k] = np.nanmean(R[in_band])
    return out


def visit_p_lo_avg(visit, pi, kind, *, los):
    """Average a visit's per-LO band power scalar (``kind`` = ``'p_on'`` or
    ``'p_off'``) across the two LOs, propagating SEM in quadrature."""
    ps  = [visit[f'{kind}_pol{pi}_lo{lo}']  for lo, _ in los]
    sps = [visit[f's{kind}_pol{pi}_lo{lo}'] for lo, _ in los]
    pairs = [(p, s) for p, s in zip(ps, sps)
             if np.isfinite(p) and np.isfinite(s)]
    if not pairs:
        return np.nan, np.nan
    p_avg = np.mean([p for p, _ in pairs])
    var   = np.sum([s ** 2 for _, s in pairs]) / len(pairs) ** 2
    return p_avg, var


def visit_dp_avg(visit, pi, *, los):
    """``dp = p_on - p_off`` averaged across LOs, variance summed."""
    pon,  von  = visit_p_lo_avg(visit, pi, 'p_on',  los=los)
    poff, voff = visit_p_lo_avg(visit, pi, 'p_off', los=los)
    return pon - poff, von + voff


# ---------------------------------------------------------------------------
# 24 h PDT Fourier fold
# ---------------------------------------------------------------------------

def hour_pdt(unix_t, tz):
    """Hour-of-day (0-24) in tz-aware ``tz`` for a Unix timestamp."""
    d = dt.datetime.fromtimestamp(unix_t, tz)
    return d.hour + d.minute / 60.0 + d.second / 3600.0


def fourier_design(h, n_harm, period):
    """Design matrix for ``a0 + sum_{k=1..K} [a_k cos + b_k sin]`` of
    period ``period`` (same units as ``h``)."""
    cols = [np.ones_like(h)]
    for k in range(1, n_harm + 1):
        omega = 2 * np.pi * k / period
        cols.append(np.cos(omega * h))
        cols.append(np.sin(omega * h))
    return np.column_stack(cols)


def fit_fourier(h, y, n_harm, period):
    """Unweighted least-squares Fourier fit; returns ``(coef, cov, rms, dof)``.

    Covariance assumes homoscedastic residuals
    (``cov = (RSS/dof) * (X^T X)^-1``).
    """
    X = fourier_design(h, n_harm, period)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    res  = y - yhat
    rms  = np.sqrt(np.mean(res ** 2))
    dof  = len(y) - len(coef)
    s2  = np.sum(res ** 2) / max(dof, 1)
    cov = s2 * np.linalg.inv(X.T @ X)
    return coef, cov, rms, dof
