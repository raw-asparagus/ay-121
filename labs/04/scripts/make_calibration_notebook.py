"""Generate labs/04/main_scan_calibration.ipynb.

Produces per-pol T_cal(t) at the (90 deg, 72 deg) recal pointing via
EBHIS-anchored differential 2-peak scale plus a 24 h PDT Fourier
forecast.  Replaces the older nominal-T_cal calibration notebook.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'main_scan_calibration.ipynb'


def _id():
    return uuid.uuid4().hex[:8]


def md(text):
    return {
        'cell_type': 'markdown',
        'id': _id(),
        'metadata': {},
        'source': text,
    }


def code(text):
    return {
        'cell_type': 'code',
        'id': _id(),
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': text,
    }


CELLS = [
    md(r"""# Anchored Tcal(t) calibration from EBHIS (HI4PI north): 2-peak scale-ratio method

Produces per-pol `Tcal(t)` curves anchored to velocity-resolved EBHIS
spectra (the northern half of HI4PI) at each recal pointing.  Instead of
integrating an area (`W_R` vs `W_HI`), the calibration is set by a
**scale ratio** at the two highest distinct peaks (separated by
`>= 30 km/s`) of each pointing's EBHIS spectrum.  Peaks are local
features with high `T_B / noise` -- they pin the absolute scale far more
cleanly than a wide integral that mixes line core, line wings, and
residual bandpass.

## Workflow

1. Fetch per-pointing velocity-resolved EBHIS spectrum from the
   Bonn AllSky_profiles server, requesting a 3.4 deg beam (matches
   Leuschner).  Cache as ASCII in `artifacts/`.
2. Per pointing, find the two highest distinct peaks of the EBHIS
   spectrum inside the Leuschner `v_LSR` window, requiring a minimum
   separation of `30 km/s`.  Read off `(v_i, T_{B,i})` for `i = 1, 2`.
3. Load + preprocess recal dumps, group into visits.
4. Per visit, per pol: average all noise-off LO1 and LO2 spectra and
   form the per-channel FS ratio
   `R_pol(c) = (P_LO1(c) - P_LO2(c)) / P_LO2(c)`.  Single-pol convention:
   the unpolarised line deposits `T_B/2` in each pol, so at the line
   peak `R_pol = (T_B/2) / T_sys_pol`.
5. Sample `R_pol` in a small `+-2 km/s` window around each of the two
   peak velocities (averaging suppresses single-channel noise).
6. Per peak `i`, per visit, per pol:
   `T_sys_pol_i = T_{B,i} / (2 * R_pol_i)`.  Combine the two peaks via
   simple mean; spread between the two is the systematic uncertainty.
7. `T_cal_pol(t) = T_sys_pol(t) * (dp / p_off)_pol(t)`.

## Assumptions / caveats

- EBHIS is the northern (Dec >= -5 deg) component of HI4PI; both recal
  pointings at Dec=+72 are in EBHIS coverage.  The Bonn server
  beam-averages a 3.4 deg cone around the requested (l, b) and returns
  a velocity-resolved spectrum in K, properly accounting for the
  Effelsberg PSF (no extra convolution needed).
- Optically-thin assumption is implicit in the EBHIS reduction; valid
  for `T_B << T_spin ~ 100 K`, which holds for these high-latitude
  fields.
- The peaks must be inside the Leuschner `v_LSR` window AND inside the
  LO1-mapped INT_MASK channel range (else the FS difference does not
  see the line cleanly).
- Per-pol `R_pol(c)` is sensitive to the `pol-0` diode-coupling deficit
  only via the gain ratios in `dp / p_off`; the FS difference itself
  cancels the bandpass and the diode.
"""),
    code(r"""import sys
import datetime as dt
import warnings
import urllib.request
import urllib.parse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import astropy.units as u
import astropy.coordinates as ac
from astropy.time import Time

# Path: labs/04 (for utils/) and project root (for ugradiolab/).
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
_PROJECT_ROOT = str(Path.cwd().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import (
    load_recal_dumps,
    preprocess_dumps,
    flag_outlier_dumps,
    build_overlap_grid,
    resample_records_to_lsr,
    fetch_ebhis_spectrum,
    find_top2_peaks,
    build_gain_visits,
    add_per_pol_W_R,
    avg_pointing_fs_diff,
    visit_R_pol_at_peaks,
    visit_p_lo_avg,
    visit_dp_avg,
    hour_pdt,
    fit_fourier,
    fourier_design,
)
from plotters import plot_ebhis_vs_leuschner_R_per_pol
from ugradiolab.plotting import (
    TEXTWIDTH_IN, subpanels,
    LW_FINE, LW_LIGHT, ALPHA_FAINT, ALPHA_LIGHT, ALPHA_STANDARD,
    SS_STANDARD,
    NEUTRAL_COLOR,
)

# --- Hardware / signal-processing constants (mirror main_scan_calibration.ipynb) ---
SAMPLE_RATE_HZ = 3.2e6
NFFT = 1024
F1_MHZ = 1419.86
F2_MHZ = 1421.14

RFI_WINDOW = 15
RFI_SIGMA = 10.0
RFI_CHEB_DEGREE = 3
RFI_SAMPLE_FRAC = 0.7
RFI_EXTREMA_ORDER = 2

SHAPE_DEV_THRESH = 0.05
SHAPE_FRAC_THRESH = 0.05
SHAPE_MIN_GROUP_SIZE = 3

EDGE_TRIM_MHZ = 0.256
VISIT_GAP_SEC = 300.0

POLS = (('corr00', 0), ('corr11', 1))
LOS  = ((1, F1_MHZ), (2, F2_MHZ))

DATA_DIRS = [Path('data/main'), Path('data/nps')]
CACHE_DIR = Path('artifacts')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- EBHIS anchoring constants ---
EBHIS_SERVER = 'https://www.astro.uni-bonn.de/hisurvey/AllSky_profiles'
LEUSCHNER_HPBW_DEG = 3.4
NH_TO_TBKMS = 1.823e18                       # cm^-2 per (K km/s), optically thin

# Leuschner v_LSR window (from main_scan_load.ipynb section 3 print).
V_LSR_LO = -165.0   # km/s
V_LSR_HI =  132.0   # km/s

# --- Recal pointings (only (90, 72) used for anchoring) ---
RECAL_POINTINGS = {
    'recal_drift_bk': ac.SkyCoord(ra=90*u.deg,  dec=72*u.deg, frame='icrs'),
}
POINTING_LABELS = {
    'recal_drift_bk': r'$(\alpha,\delta)=(90^\circ,\,72^\circ)$',
}

# --- Session-shading style (mirror main_scan_calibration.ipynb) ---
SESSION_SPAN_ALPHA = 0.08
SESSION_BOUNDARY_ALPHA = 0.45
SESSION_BOUNDARY_COLOR = '0.35'


def _shade_sessions(ax, session_spans):
    boundaries = set()
    for t0, t1 in session_spans.values():
        ax.axvspan(t0, t1, color=NEUTRAL_COLOR,
                   alpha=SESSION_SPAN_ALPHA, zorder=0)
        boundaries.add(t0); boundaries.add(t1)
    for t in boundaries:
        ax.axvline(t, color=SESSION_BOUNDARY_COLOR,
                   alpha=SESSION_BOUNDARY_ALPHA,
                   lw=LW_FINE, zorder=1)


# --- LST / PDT axis helpers (lifted from main_scan_calibration.ipynb) ---
LEUSCHNER_LOC = ac.EarthLocation(lat=37.9183*u.deg, lon=-122.1067*u.deg,
                                 height=304*u.m)
PDT_TZ = dt.timezone(dt.timedelta(hours=-7), name='PDT')
SIDEREAL_RATE = 1.00273790935
SIDEREAL_DAY_S = 86400.0 / SIDEREAL_RATE
SOLAR_DAY_S = 86400.0
LST_TICK_HOURS = (0, 6, 12, 18)
PDT_TICK_HOURS = (0, 6, 12, 18)


def _lst_hour(unix_t):
    return float(Time(unix_t, format='unix', location=LEUSCHNER_LOC)
                 .sidereal_time('apparent').hour) % 24.0


def _ticks_at_lst(t_lo, t_hi, hours):
    lst0 = _lst_hour(t_lo)
    out = []
    for tgt in hours:
        dlst = (tgt - lst0) % 24.0
        t = t_lo + (dlst / SIDEREAL_RATE) * 3600.0
        while t <= t_hi:
            out.append(t); t += SIDEREAL_DAY_S
    return sorted(out)


def _ticks_at_pdt(t_lo, t_hi, hours):
    out = []
    for tgt in hours:
        day0 = dt.datetime.fromtimestamp(t_lo, PDT_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0)
        t = (day0 + dt.timedelta(hours=tgt)).timestamp()
        while t < t_lo: t += SOLAR_DAY_S
        while t <= t_hi:
            out.append(t); t += SOLAR_DAY_S
    return sorted(out)


def time_axes_lst_pdt(axes):
    bottom = axes[-1]
    t_lo, t_hi = bottom.get_xlim()
    lst_ticks = _ticks_at_lst(t_lo, t_hi, LST_TICK_HOURS)
    bottom.set_xticks(lst_ticks)
    bottom.set_xticklabels([f'{int(round(_lst_hour(t))) % 24:d}h'
                            for t in lst_ticks], rotation=30, ha='right')
    bottom.set_xlabel('LST')
    top = axes[0].twiny()
    top.set_xlim(axes[0].get_xlim())
    pdt_ticks = _ticks_at_pdt(t_lo, t_hi, PDT_TICK_HOURS)
    top.set_xticks(pdt_ticks)
    top.set_xticklabels(
        [dt.datetime.fromtimestamp(t, PDT_TZ).strftime('%H:%M')
         for t in pdt_ticks], rotation=30, ha='left')
    top.set_xlabel('PDT')
    return top
"""),
    md(r"""## 1. Recal pointing coordinates

Two recal pointings at Dec=+72 (sidereally tracked, fixed sky positions).
"""),
    code(r"""print('Recal pointings:')
for name, c in RECAL_POINTINGS.items():
    gal = c.galactic
    print(f'  {name:18s}  (alpha, delta) = '
          f'({c.ra.deg:6.1f}, {c.dec.deg:5.1f})  '
          f'(l, b) = ({gal.l.deg:6.1f}, {gal.b.deg:+5.1f})')
"""),
    md(r"""## 2. Fetch velocity-resolved EBHIS spectra

POST to the Bonn `AllSky_profiles` form to retrieve a beam-averaged
(`3.4 deg` HPBW) EBHIS spectrum at each recal pointing's galactic
coordinates.  Cache the ASCII response in `artifacts/` to avoid repeat
downloads.

The response is a two-column ASCII table: `v_LSR [km/s]   T_B [K]`
(with `%`-prefixed header lines).  Channel width ~1 km/s,
velocity range typically `[-400, +400] km/s`.
"""),
    code(r"""ebhis_spectra = {}
for name, coord in RECAL_POINTINGS.items():
    v, T = fetch_ebhis_spectrum(name, coord, LEUSCHNER_HPBW_DEG, CACHE_DIR)
    ebhis_spectra[name] = (v, T)
    print(f'  {name}: N_channels = {len(v)}, '
          f'dv = {np.median(np.diff(v)):.2f} km/s, '
          f'v range = [{v.min():.0f}, {v.max():.0f}]')
"""),
    md(r"""## 3. Reference: integrated EBHIS `W_HI` per pointing

For reference, print full-range and Leuschner-window `W_HI` integrals
per pointing.  Peak-finding is deferred to section 5, where the
Leuschner `R(v)` spectrum is constructed -- peaks are found on **our**
spectrum, not on EBHIS, with EBHIS providing `T_B` only at the chosen
peak velocities.
"""),
    code(r"""W_HI_LIT = {}     # K km/s, truncated to Leuschner window (Stokes I, reference)
W_HI_FULL = {}    # K km/s, full velocity range (reference)

for name, (v, T) in ebhis_spectra.items():
    W_full = float(np.trapezoid(T, v))
    in_win = (v >= V_LSR_LO) & (v <= V_LSR_HI)
    W_win  = float(np.trapezoid(T[in_win], v[in_win]))
    W_HI_LIT[name]  = W_win
    W_HI_FULL[name] = W_full
    print(f'  {name}:')
    print(f'    W_HI (full v range):           {W_full:7.2f} K*km/s')
    print(f'    W_HI (Leuschner [{V_LSR_LO:+.0f}, {V_LSR_HI:+.0f}]):  '
          f'{W_win:7.2f} K*km/s  (lost {(1 - W_win/W_full)*100:.2f}%)')
"""),
    md(r"""## 4. Load + preprocess recal dumps; per-dump LSR resampling; build visits

Mirrors sections 1-6 of `main_scan_load.ipynb`.  The science-load
notebook applies LSR resampling per-pair via `build_lsr_pairs`; the
calibration notebook reuses the same framework via
`resample_records_to_lsr` (in-place per-dump), so by the end of section 4
every record's `corr00` / `corr11` already lives on a common LSR-aligned
channel grid.  Downstream peak finding and per-pol `R(v_LSR)` sampling
therefore see no smearing from different per-dump `v_corr`.

Per-pol propagation is required because the noise diode coupling is
pol-dependent at 3.2 MHz (pol 0 deficit); per-pol band powers are
recorded on each visit via `build_gain_visits` and per-pol integrated
`W_R` via `add_per_pol_W_R`.
"""),
    code(r"""recal_records = load_recal_dumps(DATA_DIRS)
preprocess_dumps(
    recal_records,
    rfi_window=RFI_WINDOW, rfi_sigma=RFI_SIGMA,
    rfi_degree=RFI_CHEB_DEGREE,
    rfi_sample_frac=RFI_SAMPLE_FRAC,
    rfi_extrema_order=RFI_EXTREMA_ORDER,
)
flag_outlier_dumps(
    recal_records,
    dev_thresh=SHAPE_DEV_THRESH,
    frac_thresh=SHAPE_FRAC_THRESH,
    min_group_size=SHAPE_MIN_GROUP_SIZE,
)

# Per-dump LSR resampling: shifts each record's corr00 / corr11 by
# v_corr_d / dvch so all dumps share a single LSR-aligned channel grid.
# Sets r['v_corr'] on each record.
resample_records_to_lsr(
    recal_records, nfft=NFFT, sample_rate_hz=SAMPLE_RATE_HZ,
)
_v_corr_arr = np.array([r['v_corr'] for r in recal_records])
print(f'LSR resampling: v_corr range '
      f'[{_v_corr_arr.min():+.2f}, {_v_corr_arr.max():+.2f}] km/s '
      f'(median {np.median(_v_corr_arr):+.2f} km/s)')

grid = build_overlap_grid(
    F1_MHZ, F2_MHZ, SAMPLE_RATE_HZ, NFFT, edge_trim_mhz=EDGE_TRIM_MHZ,
)
INT_MASK = grid['overlap_mask'].copy()
INT_MASK[NFFT // 2] = False
DV_KMS = grid['dv_kms']
BW_KMS = INT_MASK.sum() * DV_KMS
print(f'INT_MASK: {INT_MASK.sum()} channels  '
      f'dv = {DV_KMS:.3f} km/s/ch  BW = {BW_KMS:.2f} km/s')
print(f'{len(recal_records)} recal dumps after preprocessing + LSR resample')
"""),
    code(r"""gain_visits = build_gain_visits(
    recal_records, int_mask=INT_MASK, pols=POLS, los=LOS,
    visit_gap_sec=VISIT_GAP_SEC,
)
add_per_pol_W_R(gain_visits, BW_KMS)
print(f'{len(gain_visits)} visits total')
print('Per pointing:',
      {tid: sum(1 for v in gain_visits if v['target_id'] == tid)
       for tid in sorted({v["target_id"] for v in gain_visits})})
"""),
    md(r"""## 5. Sanity-check spectra (3 rows, per-pol): EBHIS vs Leuschner $R$

Top row: EBHIS beam-averaged $T_B(v_{\rm LSR})$ with the Leuschner
window shaded.  Vertical lines mark the two anchor peaks found on the
EBHIS spectrum.

Middle / bottom rows: Leuschner per-channel frequency-switched ratio
$R_{\rm pol}(c) = (P^{\rm LO1}_{\rm pol}(c) - P^{\rm LO2}_{\rm pol}(c)) /
P^{\rm LO2}_{\rm pol}(c)$, **per pol independently**, averaged over all
noise-off dumps at the pointing.  Anchor peaks are likewise found per pol
on the pol-specific $R(v_{\rm LSR})$ spectrum.  The diode coupling
deficit on pol 0 can shift the apparent peak heights and locations
relative to pol 1; if peak finding ran on the Stokes-I sum, pol-1
information would dominate and pol-0 peak velocities would be wrong.

LO1 channel mapping is converted to $v_{\rm LSR}$ assuming every dump
has already been LSR-resampled in section 4 (i.e. each record's `corr00`
/ `corr11` already lives on the LSR-aligned grid).  Peak search is
restricted to $v_{\rm LSR} \in [V_{\rm LSR,lo}, 0]$: at
$(\ell, b) = (141.9, +21.9)$ the recal_drift_bk pointing is in the
outer Galaxy with no inner-Galactic emission, so both anchor peaks must
sit at negative $v_{\rm LSR}$.

Each pair of (EBHIS peak, Leuschner peak) is matched by sorted-$v$
index; velocities need not agree between the two surveys, only the
ordering carries through to $T_{\rm sys}_{\rm pol} = (T_{B,1} - T_{B,2}) /
(2 (R_{{\rm pol},1} - R_{{\rm pol},2}))$ in section 6.
"""),
    code(r"""# v_LSR axis using LO1 channel-to-frequency.  Records are already
# LSR-resampled in section 4, so no global V_CORR_MEAN offset is needed.
C_KMS = 299792.458
HI_REST_MHZ = 1420.40575
df_mhz = SAMPLE_RATE_HZ / NFFT / 1e6
freq_lo1_axis = F1_MHZ + (np.arange(NFFT) - NFFT // 2) * df_mhz
v_lsr_axis = C_KMS * (HI_REST_MHZ - freq_lo1_axis) / HI_REST_MHZ

pointing_list = list(RECAL_POINTINGS.keys())

# Per-pointing minimum separation between the two R-spectrum peaks.
MIN_PEAK_SEP_KMS = {
    'recal_drift_bk': 50.0,
}

# At (l, b) = (141.9, +21.9) the recal_drift_bk pointing has no
# inner-Galactic emission, so both anchor peaks must sit at v_LSR <= 0.
PEAK_SEARCH_V_HI = 0.0


# Per-pol peak finding: dicts are keyed by (pointing_name, pol_index).
EBHIS_PEAK_V  = {}            # pointing -> EBHIS peak v_LSR (2,)
EBHIS_PEAK_TB = {}            # pointing -> EBHIS T_B at peaks (2,)
R_PEAK_V       = {}           # (pointing, pi) -> Leuschner peak v (2,)
R_PEAK_REF     = {}           # (pointing, pi) -> Leuschner R at peaks (2,)
POINTING_R_AVG = {}           # (pointing, pi) -> R(v) per-pol avg spectrum
POINTING_R_N   = {}           # (pointing, pi) -> (n_lo1, n_lo2)

for name in pointing_list:
    msep = MIN_PEAK_SEP_KMS[name]

    # EBHIS peaks (on EBHIS's own v_LSR grid).
    v_eb, T_eb = ebhis_spectra[name]
    v_eb_pk, T_eb_pk = find_top2_peaks(
        T_eb, v_eb, msep, x_lo=V_LSR_LO, x_hi=PEAK_SEARCH_V_HI,
    )
    EBHIS_PEAK_V[name]  = v_eb_pk
    EBHIS_PEAK_TB[name] = T_eb_pk

    print(f'  {name}  (min_sep >= {msep:.0f} km/s, '
          f'v <= {PEAK_SEARCH_V_HI:+.0f} km/s):')
    print(f'    EBHIS peaks:  '
          + '   '.join(f'v={vp:+7.2f}  T_B={Tp:6.2f} K'
                       for vp, Tp in zip(v_eb_pk, T_eb_pk)))

    # Per-pol Leuschner R peaks.
    for pi in (0, 1):
        R_c, n1, n2 = avg_pointing_fs_diff(
            recal_records, name, lo_freqs=(F1_MHZ, F2_MHZ), pol=pi,
        )
        POINTING_R_AVG[(name, pi)] = R_c
        POINTING_R_N[(name, pi)]   = (n1, n2)
        if R_c is None:
            raise RuntimeError(f'No FS data for {name} pol {pi}')
        v_R_pk, R_at_R_pk = find_top2_peaks(
            R_c, v_lsr_axis, msep, x_lo=V_LSR_LO, x_hi=PEAK_SEARCH_V_HI,
        )
        R_PEAK_V[(name, pi)]   = v_R_pk
        R_PEAK_REF[(name, pi)] = R_at_R_pk
        print(f'    Leuschner pol {pi} peaks (N={n1}+{n2}):  '
              + '   '.join(f'v={vp:+7.2f}  R={Rp:+.4f}'
                           for vp, Rp in zip(v_R_pk, R_at_R_pk)))


n_visits_by_pointing = {
    name: sum(1 for v in gain_visits if v['target_id'] == name)
    for name in pointing_list
}

fig, axes = plot_ebhis_vs_leuschner_R_per_pol(
    ebhis_spectra,
    POINTING_R_AVG, POINTING_R_N,
    EBHIS_PEAK_V, EBHIS_PEAK_TB,
    R_PEAK_V, R_PEAK_REF,
    v_lsr_axis,
    POINTING_LABELS,
    v_lsr_window=(V_LSR_LO, V_LSR_HI),
    n_visits_by_pointing=n_visits_by_pointing,
)
plt.show()
"""),
    md(r"""## 6. Differential-peak Tcal(t)

For each visit `t` (per pol), evaluate `R_pol(c, t)` and form the
difference between the two anchor peaks.  Any visit-level offset in
`R_pol` (residual bandpass, continuum) drops out:

```
R_pol(c, t)    = (P_LO1_pol(c, t) - P_LO2_pol(c, t)) / P_LO2_pol(c, t)
R_pol_i(t)     = mean R_pol over +/- PEAK_HALFWIDTH_KMS around Leuschner v_i
T_sys_pol(t)   = (T_B_global - T_B_next) / (2 * (R_pol_global(t) - R_pol_next(t)))
T_cal_pol(t)   = T_sys_pol(t) * (dp / p_off)_pol(t)
```

`global` is the pair (by sorted-$v$ index) with the larger EBHIS $T_B$;
`next` is the other pair.  Single estimate per visit per pol -- no
2-peak spread, no errorbar.  Anchor velocities are read off section
5's **per-pol** `R_PEAK_V[(tid, pi)]`, so each pol uses its own
Leuschner peak velocities (pol 0 may differ from pol 1 by 1-2 channels
due to the diode coupling deficit).
"""),
    code(r"""PEAK_HALFWIDTH_KMS = 2.0   # avg R_pol over +/- this many km/s around each peak

rows = []
for v in gain_visits:
    tid = v['target_id']
    if (tid, 0) not in R_PEAK_V:
        continue
    T_peaks = EBHIS_PEAK_TB[tid]  # EBHIS T_B at EBHIS peaks (sorted by v)
    # "global" pair = the one with the larger EBHIS T_B; "next" = the other.
    g = int(np.argmax(T_peaks))
    n = 1 - g
    T_B_global, T_B_next = float(T_peaks[g]), float(T_peaks[n])
    row = {'t_mid': v['t_mid'], 'session': v['session'],
           'target_id': tid, 'alt': v['alt_mean']}
    for pi in (0, 1):
        v_peaks_pol = R_PEAK_V[(tid, pi)]
        R_at = visit_R_pol_at_peaks(
            v, pi, v_peaks_pol,
            v_axis=v_lsr_axis, halfwidth_kms=PEAK_HALFWIDTH_KMS,
            lo_freqs=(F1_MHZ, F2_MHZ),
        )
        R_global = float(R_at[g]) if np.isfinite(R_at[g]) else np.nan
        R_next   = float(R_at[n]) if np.isfinite(R_at[n]) else np.nan
        poff, _ = visit_p_lo_avg(v, pi, 'p_off', los=LOS)
        dp,   _ = visit_dp_avg(v, pi, los=LOS)
        ok = (np.isfinite(R_global) and np.isfinite(R_next)
              and (R_global - R_next) != 0
              and np.isfinite(poff) and poff > 0
              and np.isfinite(dp))
        if not ok:
            row[f'T_sys_pol{pi}'] = np.nan
            row[f'Tcal_pol{pi}']  = np.nan
            row[f'R_pol{pi}_global'] = R_global
            row[f'R_pol{pi}_next']   = R_next
            continue
        T_sys = (T_B_global - T_B_next) / (2.0 * (R_global - R_next))
        Tcal  = T_sys * dp / poff
        row[f'T_sys_pol{pi}']    = T_sys
        row[f'Tcal_pol{pi}']     = Tcal
        row[f'R_pol{pi}_global'] = R_global
        row[f'R_pol{pi}_next']   = R_next
    rows.append(row)

tcal_df = pd.DataFrame(rows)
print(f'{len(tcal_df)} anchored (visit, pol) rows '
      f'(peak halfwidth = +/- {PEAK_HALFWIDTH_KMS:.1f} km/s, '
      f'differential scale).')
for pi in (0, 1):
    col = f'Tcal_pol{pi}'
    series = tcal_df[col].dropna()
    if len(series):
        print(f'  pol {pi}: median Tcal = {series.median():.2f} K  '
              f'(IQR [{series.quantile(0.25):.2f}, '
              f'{series.quantile(0.75):.2f}], N={len(series)})')
"""),
    md(r"""## 7. Plot Tcal(t) per pol, both pointings overlaid

Two stacked panels (pol 0, pol 1).  Marker = pointing
(circle = `recal_drift`, triangle = `recal_drift_bk`); marker face
colour = altitude (reversed viridis; darker = higher).  Both pointings
should overlay if drift is genuinely diode-related; persistent
A vs B offsets signal pointing-dependent systematics
(elevation spillover, atmosphere, partial beam-coupling to the
EBHIS-derived brightness gradient).
"""),
    code(r"""# Plot-only outlier mask: hide pol-0 visits with Tcal_pol0 > this in all
# scatter plots, but keep them in the underlying analysis (medians + fits).
POL0_PLOT_MAX = 10.0  # K


def plot_keep_mask(df, pi):
    if pi == 0:
        return df[f'Tcal_pol{pi}'].fillna(np.inf) <= POL0_PLOT_MAX
    return np.ones(len(df), dtype=bool)


pointings = sorted(tcal_df['target_id'].unique())
markers = dict(zip(pointings, ['o', '^']))

sessions = sorted(tcal_df['session'].unique())
session_spans = {
    s: (tcal_df.loc[tcal_df['session'] == s, 't_mid'].min(),
        tcal_df.loc[tcal_df['session'] == s, 't_mid'].max())
    for s in sessions
}

alt_min, alt_max = 17.0, 83.0
cmap = plt.cm.viridis_r

fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 10 / 16),
                 constrained_layout=True)
axes = subpanels(fig, 2, sharex=True)

sc = None
for ax, pi in zip(axes, (0, 1)):
    for tid in pointings:
        sub = tcal_df[tcal_df['target_id'] == tid].dropna(
            subset=[f'Tcal_pol{pi}'])
        plot_sub = sub[plot_keep_mask(sub, pi)]
        sc = ax.scatter(plot_sub['t_mid'], plot_sub[f'Tcal_pol{pi}'],
                        c=plot_sub['alt'], cmap=cmap,
                        vmin=alt_min, vmax=alt_max,
                        marker=markers[tid], s=SS_STANDARD,
                        edgecolor='k', linewidth=LW_FINE, zorder=3)
    median = tcal_df[f'Tcal_pol{pi}'].median()
    ax.axhline(median, color=f'C{pi}', ls='--', lw=LW_FINE,
               alpha=ALPHA_LIGHT,
               label=rf'median = {median:.2f} K')
    ax.set_ylabel(rf'$T_{{\rm cal}}$ pol {pi} (K)')
    ax.legend(loc='upper right', frameon=True, fontsize='small')

marker_handles = [
    Line2D([], [], marker=markers[tid], color='none',
           markerfacecolor='lightgray', markeredgecolor='k',
           markersize=8, label=POINTING_LABELS[tid])
    for tid in pointings
]
axes[0].legend(
    handles=marker_handles + [
        Line2D([], [], color=f'C{0}', ls='--', lw=LW_FINE,
               label=rf'pol 0 median = '
                     rf'{tcal_df["Tcal_pol0"].median():.2f} K')],
    loc='upper right', frameon=True, fontsize='small',
    title='pointing',
)

time_axes_lst_pdt(axes)
cbar = fig.colorbar(sc, ax=axes, location='right',
                    fraction=0.04, pad=0.02, shrink=0.9)
cbar.set_label('altitude (deg)')

fig.suptitle(
    r'2-peak-anchored $T_{\rm cal}(t)$ per pol (EBHIS $T_B$ paired by sorted-$v$ index with Leuschner $R$ peaks)'
)
plt.show()
"""),
    md(r"""## 8. Fold to 24 h (PDT) and Fourier-fit, per-pol harmonic order

The diode coupling drifts with ambient temperature, which follows the
solar day.  Over the 4.5-day recal_drift_bk span LST and PDT differ by
< 20 min, so the two periods are mathematically degenerate; PDT is
chosen as the physically motivated axis.  Fit a discrete Fourier
series of period 24 h:

```
T_cal(h_PDT) ~ a0 + sum_{k=1..K_pol} [ a_k cos(2 pi k h_PDT / 24)
                                       + b_k sin(2 pi k h_PDT / 24) ]
```

Different `K_pol` per polarisation: K_0 = 2 (small but `>= 3 sigma` 2nd
harmonic on pol 0), K_1 = 1 (pol 1's 2nd harmonic is consistent with
noise).  Unweighted linear least squares.  Residual RMS is the
per-visit scatter about the fit.  Per-harmonic amplitude
`A_k = sqrt(a_k^2 + b_k^2)` and phase `phi_k = atan2(b_k, a_k)` are
reported.
"""),
    code(r"""PERIOD_HOURS = 24.0   # solar day; LST/PDT are degenerate over the 4.5 d span
N_HARMONICS  = {0: 2, 1: 1}   # per-pol number of Fourier harmonics

fold = {}
for pi in (0, 1):
    sub = tcal_df.dropna(subset=[f'Tcal_pol{pi}']).copy()
    if sub.empty:
        print(f'pol {pi}: no usable rows')
        continue
    h = np.array([hour_pdt(t, PDT_TZ) for t in sub['t_mid']])
    y = sub[f'Tcal_pol{pi}'].values
    K = N_HARMONICS[pi]
    coef, cov, rms, dof = fit_fourier(h, y, K, PERIOD_HOURS)
    fold[pi] = dict(h=h, y=y, coef=coef, cov=cov,
                    rms=rms, dof=dof, sub=sub, K=K)
    print(f'pol {pi}: K = {K},  N = {len(y)},  a0 = {coef[0]:.3f} K,  '
          f'residual RMS = {rms:.3f} K   (dof = {dof})')
    sd = np.sqrt(np.diag(cov))
    for k in range(1, K + 1):
        ak = coef[2*k - 1]; bk = coef[2*k]
        s_ak = sd[2*k - 1];  s_bk = sd[2*k]
        amp   = float(np.hypot(ak, bk))
        s_amp = float(np.sqrt((ak * s_ak)**2 + (bk * s_bk)**2) / amp) \
                if amp > 0 else float('nan')
        phase = float(np.degrees(np.arctan2(bk, ak)))
        print(f'   k = {k}:  a = {ak:+.3f} +/- {s_ak:.3f},  '
              f'b = {bk:+.3f} +/- {s_bk:.3f},  '
              f'A = {amp:.3f} +/- {s_amp:.3f} K,  '
              f'phi = {phase:+.1f} deg')
"""),
    code(r"""h_smooth = np.linspace(0.0, 24.0, 481)

fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 10 / 16),
                 constrained_layout=True)
axes = subpanels(fig, 2, sharex=True)

for ax, pi in zip(axes, (0, 1)):
    d = fold.get(pi)
    if d is None:
        continue
    # Plot-only mask: hide pol-0 visits above POL0_PLOT_MAX (still in fit).
    mask = (d['y'] <= POL0_PLOT_MAX) if pi == 0 else np.ones_like(d['y'], dtype=bool)
    ax.scatter(d['h'][mask], d['y'][mask],
               s=20, color=f'C{pi}', edgecolor='k',
               linewidth=LW_FINE, alpha=ALPHA_STANDARD, zorder=2)
    y_fit = fourier_design(h_smooth, d['K'], PERIOD_HOURS) @ d['coef']
    ax.plot(h_smooth, y_fit, color='C3', lw=LW_LIGHT, zorder=3,
            label=rf'$K={d["K"]}$ fit, '
                  rf'RMS = {d["rms"]:.2f} K (dof = {d["dof"]})')
    ax.axhline(d['coef'][0], color='0.3', ls='--', lw=LW_FINE,
               alpha=ALPHA_LIGHT, zorder=1,
               label=rf'$a_0={d["coef"][0]:.2f}$ K')
    ax.set_ylabel(rf'$T_{{\rm cal}}$ pol {pi} (K)')
    ax.legend(loc='upper right', fontsize='small', frameon=True)

axes[-1].set_xlabel('hour of day (PDT)')
axes[-1].set_xlim(0, 24)
axes[-1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

fig.suptitle(rf'24 h PDT fold + Fourier fit ($K_0={N_HARMONICS[0]}$, '
             rf'$K_1={N_HARMONICS[1]}$) at '
             rf'{POINTING_LABELS[list(RECAL_POINTINGS.keys())[0]]}')
plt.show()
"""),
    md(r"""## 8b. Persist Fourier state to `artifacts/tcal_drift_state.pkl`

Save the per-pol Fourier coefficients, harmonic counts, period, and
PDT offset to a pickle for downstream consumption by
`main_scan_qa.ipynb`.  QA evaluates `Tcal_pol(t)` per cell median
time by re-folding `t -> h_PDT mod 24` and applying the same
design matrix.

Schema:

```
{
    'period_hours': 24.0,
    'tz_offset_hours': -7.0,      # PDT
    'n_harmonics': {0: 2, 1: 1},
    'pointing': 'recal_drift_bk',
    'anchor_pointing_coord': SkyCoord(...),
    'fit': {pi: {'coef': np.ndarray, 'cov': np.ndarray,
                  'rms': float, 'dof': int, 'N': int}},
    't_min': float, 't_max': float,
}
```
"""),
    code(r"""import pickle

TCAL_STATE_PATH = Path('artifacts/tcal_drift_state.pkl')

_anchor_name = list(RECAL_POINTINGS.keys())[0]
_t_all = []
for pi, d in fold.items():
    _t_all.extend(d['sub']['t_mid'].tolist())

tcal_drift_state = {
    'period_hours': float(PERIOD_HOURS),
    'tz_offset_hours': -7.0,
    'n_harmonics': {int(pi): int(K) for pi, K in N_HARMONICS.items()},
    'pointing': _anchor_name,
    'anchor_pointing_coord': RECAL_POINTINGS[_anchor_name],
    'fit': {int(pi): {
                'coef': np.asarray(d['coef'], dtype=float),
                'cov':  np.asarray(d['cov'],  dtype=float),
                'rms':  float(d['rms']),
                'dof':  int(d['dof']),
                'N':    int(len(d['y'])),
            } for pi, d in fold.items()},
    't_min': float(min(_t_all)),
    't_max': float(max(_t_all)),
}

with open(TCAL_STATE_PATH, 'wb') as f:
    pickle.dump(tcal_drift_state, f)
print(f'Wrote {TCAL_STATE_PATH} '
      f'(N_pol0={tcal_drift_state["fit"][0]["N"]}, '
      f'N_pol1={tcal_drift_state["fit"][1]["N"]}, '
      f'K_0={tcal_drift_state["n_harmonics"][0]}, '
      f'K_1={tcal_drift_state["n_harmonics"][1]})')
"""),
    md(r"""## 9. Summary

The noise-diode nominal values (`T_cal_pol0 = 58 K`,
`T_cal_pol1 = 79 K`) are unreliable at 3.2 MHz bandwidth (pol-0
coupling deficit; both pols drift with ambient temperature), so the
nominals are no longer used as a reference target.  The
`T_cal_pol(t)` forecast adopted here is:

```
peaks (v_i, T_{B,i})        <- top-2 EBHIS peaks within the Leuschner
                               v_LSR window, separated by >= the
                               per-pointing MIN_PEAK_SEP_KMS
v_R_i, R_i(t)               <- corresponding Leuschner R(v) peaks
                               paired by sorted-v index
global = pair with larger T_B;  next = the other
T_sys_pol(t) = (T_B_global - T_B_next) / (2 * (R_global - R_next))
T_cal_pol(t) = T_sys_pol(t) * (dp / p_off)_pol(t)
T_cal_pol(h) = a0 + sum_{k=1..K_pol} [a_k cos + b_k sin] (period 24 h)
```

The pointing-anchored differential scale gives a single per-visit
`T_cal` (no 2-peak spread); the per-pol Fourier model in `h_PDT` is
the recommended forecast to apply during the science reduction.

To feed back into `main_scan_load.ipynb`, save the Fourier coefficients
(or sampled `T_cal_pol(t)`) to `artifacts/tcal_drift_state.pkl` and
apply per-(session, cell) at reduction time.
"""),
]

nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f'Wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB, {len(CELLS)} cells)')
