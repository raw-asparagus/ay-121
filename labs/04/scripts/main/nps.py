#!/usr/bin/env python3
"""Lab 4 - Leuschner Sky Survey -- North Polar Spur.

North Polar Spur survey using the same machinery as the galactic-plane
``radio.py`` driver but with NPS-shaped grid bounds and dedicated
artifacts/data directories.  ``main()`` runs once and returns, intended
to be invoked as a time-boxed first stage before the galactic-plane
loop in ``radio.py``.

Per pointing the LO sequence is (ABBA interleaved):
    CAL-f1, CAL-f2, OBS-f2, OBS-f1, OBS-f1, OBS-f2, OBS-f2, OBS-f1
where f1=1419.86 MHz, f2=1421.14 MHz.

Usage:
    python main.py

Output:
    data/session_{NNN}/<obs|cal>_{l}_{b}/<...>_{timestamp}.npz

Run from this directory:
    PYTHONPATH=../../.. python3 main.py
"""

import glob
import json
import math
import re
import sys
import threading
import time as _time
from pathlib import Path

import numpy as np
from astropy.coordinates import get_body, get_sun
from astropy.time import Time

# Allow `from utils.timing_stats import ...` when run with cwd=labs/04/.
sys.path.insert(0, '.')
from utils.timing_stats import load as _load_timing_stats  # noqa: E402

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
    compute_radec_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_calibrated_sdr_reader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

F1_MHZ       = 1419.86
F2_MHZ       = 1421.14
SAMPLE_RATE  = 3.2e6
NSAMPLES     = 32768   # 32 chunks * 1024 channels
NBLOCKS      = 1025    # 1024 valid + 1 buffer flush
NFFT         = 1024
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN       =  7.0
AZ_MAX       = 348.0
REPOINT_INTERVAL_SEC = 10.0
SUN_AVOID_DEG  = 30.0   # skip cells within this angular distance of the Sun (Sun sidelobe pickup at 1.4 GHz is non-negligible inside ~30 deg)
MOON_AVOID_DEG = 10.0   # skip cells within this angular distance of the Moon (0 to disable)
OUTPUT_DIR   = 'data/nps'  # relative to cwd (expected: labs/04/)

# Grid bounds (North Polar Spur: l 210-380 wrapping, b 0-70).
# L_CENTER is the phase anchor of the cos(b)-corrected longitude grid; setting
# it to 120 keeps NPS rows on the same phase as the galactic-plane survey so
# their brick interleaves are mutually consistent.  The anchor may sit outside
# [L_MIN, L_MAX] -- _build_l_row filters the final row to stay in range.
L_CENTER     = 120.0
L_MIN, L_MAX = 210.0, 380.0
B_MIN, B_MAX = 0, 70
B_STEP       = 2
PHYSICAL_SPACING_DEG = 2.0

# Per-pointing dump schedule (plane is bright: T_B ~ 50-150 K)
CAL_DUMPS_PER_LO = 2    # 2 cal dumps per LO = 4 cal total
OBS_DUMPS_PER_LO = 3    # 3 obs dumps per LO = 6 obs total
N_LOS = 2
DUMPS_PER_CELL = (CAL_DUMPS_PER_LO + OBS_DUMPS_PER_LO) * N_LOS

# Periodic re-cal drift check: every RECAL_EVERY_N_CELLS successful cells
# the dish points to a fixed circumpolar reference (RA/Dec) and runs the
# normal cell schedule (cal + obs). Because the sky is identical at every
# visit, any drift in (P_on - P_off) or T_sys at this target is purely
# instrumental -- a direct timeline of receiver / diode drift through the
# session.
#
# From Leuschner (lat=37.92 deg N), no fixed (RA, Dec) target is *always*
# within limits: any Dec > 69 deg is circumpolar above alt=17 deg but
# crosses az=0 deg at both transits, hitting the 19 deg north exclusion
# (AZ_MIN=7, AZ_MAX=348). Dec=+72 deg minimises the combined blockage to
# ~8 h/day -- max az excursion ~22.4 deg comfortably clears the exclusion
# at non-transit phases, lower transit alt ~19.9 deg has a safe 2.9 deg
# margin above the 17 deg limit. RA=12h places transit windows away from
# typical evening observing (~LST 12-17h around late spring). Galactic
# (l~120, b~+44) -- extragalactic, very cold sky. The defer-and-retry
# logic in the selector handles transit blockage cleanly: when the recal
# target is in the exclusion, cells_since_recal is rolled back so the
# recal fires within ~5 survey cells of the source clearing.
RECAL_ENABLE = True
# Two targets at the same Dec=+72 deg but RA offset by 6h (90 deg). By
# geometry the two transit-blockage windows of each target fall inside
# the other target's safe window, so at any LST at least one of the two
# is observable. The selector tries primary first, falls back to backup
# if the primary is currently blocked. Both names get distinct cell dirs
# so the drift timeline can be tracked per target.
RECAL_TARGETS = [
    {'name': 'obs_recal_drift',    'ra_deg': 180.0, 'dec_deg': 72.0},
    {'name': 'obs_recal_drift_bk', 'ra_deg':  90.0, 'dec_deg': 72.0},
]
RECAL_EVERY_N_CELLS = 20

def _detect_next_session() -> int:
    """Return one greater than the highest existing session_NNN dir, or 1."""
    highest = 0
    for d in glob.glob(f'{OUTPUT_DIR}/session_*'):
        m = re.search(r'session_(\d+)$', d)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1 if highest else 1


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _build_l_row(b_deg, l_center=L_CENTER):
    """Build non-integer longitude grid at latitude b, centered at l_center.

    Exact spacing: Delta_l = PHYSICAL_SPACING_DEG / cos(b).
    Expands outward from l_center until L_MIN and L_MAX are exceeded.
    Returns sorted list of l values (floats, rounded to 2 decimal places).
    """
    cos_b = math.cos(math.radians(b_deg))
    if cos_b <= 0:
        return [l_center]
    dl = PHYSICAL_SPACING_DEG / cos_b

    l_vals = [round(l_center, 2)]
    l = l_center + dl
    while l <= L_MAX:
        l_vals.append(round(l, 2))
        l += dl
    l = l_center - dl
    while l >= L_MIN:
        l_vals.append(round(l, 2))
        l -= dl

    return sorted(v for v in l_vals if L_MIN <= v <= L_MAX)


def build_galplane_grid(phase='even'):
    """Build column-major grid: sweep constant-l columns from L_MIN to L_MAX.

    For each approximate longitude, sweeps through all b values at that
    longitude.  Adjacent columns alternate b direction (zig-zag).
    Columns are ordered by increasing l, starting from L_MIN.

    Parameters
    ----------
    phase : 'even' or 'odd'
        'even' -- b = [-4, -2, 0, 2, 4], l centered at L_CENTER.
        'odd'  -- b = [-3, -1, 1, 3], l offset by half a physical step
                  from L_CENTER (brick-pattern interleave).

    Returns list of (col_idx, row_idx, l, b) tuples.
    """
    if phase == 'even':
        b_vals = list(range(B_MIN, B_MAX + 1, B_STEP))
    else:
        b_vals = list(range(B_MIN + 1, B_MAX, B_STEP))

    # Build all cells per b row
    all_cells = []
    for b in b_vals:
        if phase == 'odd':
            half_step = PHYSICAL_SPACING_DEG / (2 * math.cos(math.radians(b)))
            l_center = L_CENTER + half_step
        else:
            l_center = L_CENTER

        l_vals = _build_l_row(b, l_center=l_center)
        dl = PHYSICAL_SPACING_DEG / np.cos(np.radians(b))
        print(f'  b={b:+3d}: Delta_l={dl:.2f} deg, {len(l_vals)} cells, '
              f'l=[{l_vals[0]:.1f}, {l_vals[-1]:.1f}]')
        for l in l_vals:
            all_cells.append((l, b))

    # Group cells into columns: sort by l, bin within half-spacing tolerance
    all_cells.sort(key=lambda c: c[0])
    col_tol = PHYSICAL_SPACING_DEG / 2
    columns = [[all_cells[0]]]
    for cell in all_cells[1:]:
        if cell[0] - columns[-1][0][0] <= col_tol:
            columns[-1].append(cell)
        else:
            columns.append([cell])

    # Build output: columns in ascending l, zig-zag b within each column
    cells = []
    for col_idx, col in enumerate(columns):
        col_sorted = sorted(col, key=lambda c: c[1])
        if col_idx % 2 == 1:
            col_sorted = list(reversed(col_sorted))
        for row_idx, (l, b) in enumerate(col_sorted):
            cells.append((col_idx, row_idx, l, b))

    print(f'  Column-major: {len(columns)} columns, {len(cells)} cells total')
    return cells


def classify_cells_by_az_side(cells, unix_t=None):
    """Split cells into rising and setting candidate lists at ``unix_t``.

    Permanently inaccessible cells (max_alt < MIN_ALT_DEG) are dropped.
    Cells currently below the alt limit are still kept on whichever side
    their azimuth puts them, so the forecasted-survivor comparison can
    pick up cells that will rise into view.
    """
    rising = []
    setting = []
    n_permanent = 0
    n_rising_now = 0
    n_setting_now = 0

    t = unix_t if unix_t is not None else _time.time()
    for row, col, l, b in cells:
        ra, dec = _cell_radec(l, b)
        max_alt = 90.0 - abs(LEO_LAT_DEG - dec)
        if max_alt < MIN_ALT_DEG:
            n_permanent += 1
            continue
        alt, az = _fast_altaz(ra, dec, t)

        in_limits = MIN_ALT_DEG <= alt <= MAX_ALT_DEG
        if az <= 180 or az > AZ_MAX:
            rising.append((row, col, l, b))
            if in_limits:
                n_rising_now += 1
        else:
            setting.append((row, col, l, b))
            if in_limits:
                n_setting_now += 1

    if n_permanent:
        print(f'  Dropped {n_permanent} permanently inaccessible cells')
    print(f'  Az classification: {n_rising_now} rising / {n_setting_now} setting '
          f'currently accessible ({len(rising)} / {len(setting)} total)')
    return rising, setting


def _load_reobserve_set():
    """Load reobserve.json and return a set of cell_lb keys to force-reobserve."""
    reobs_path = Path('artifacts/nps_reobserve.json')
    if not reobs_path.exists():
        return set()
    try:
        entries = json.loads(reobs_path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    reobs = set()
    for e in entries:
        l_name = f'{e["l"] % 360.0:.2f}'.replace('.', 'p')
        reobs.add(f'{l_name}_{e["b"]}')
    return reobs


def _bright_body_radec(unix_t):
    """Return (sun_ra_deg, sun_dec_deg, moon_ra_deg, moon_dec_deg) at unix_t.

    Uses astropy ``get_sun`` / ``get_body``.  Moon RA/Dec is geocentric;
    its topocentric parallax (<1 deg) is well below the MOON_AVOID_DEG=10
    threshold and the Sun's parallax is sub-arcsec at this distance, so
    no parallax correction is applied.
    """
    t = Time(unix_t, format='unix')
    sun = get_sun(t)
    moon = get_body('moon', t)
    return (sun.ra.deg, sun.dec.deg, moon.ra.deg, moon.dec.deg)


def _angular_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle angular distance in degrees (vectorised in scalars)."""
    r1 = np.deg2rad(ra1)
    d1 = np.deg2rad(dec1)
    r2 = np.deg2rad(ra2)
    d2 = np.deg2rad(dec2)
    cos_sep = (np.sin(d1) * np.sin(d2)
               + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2))
    cos_sep = max(-1.0, min(1.0, float(cos_sep)))
    return float(np.rad2deg(np.arccos(cos_sep)))


# ---------------------------------------------------------------------------
# Fast planning math
# ---------------------------------------------------------------------------
#
# RA/Dec are time-invariant for galactic (l, b); the bulk of
# ``compute_gal_pointing``'s cost is the SkyCoord+ICRS construction (~21 ms).
# Cache (l, b) -> (ra, dec) on first lookup, then derive alt/az from a
# closed-form numpy formula (~50 us).  Used only on the planner's hot path
# (classify, set-time sort, forward-sim).  The hardware-driving target
# selector keeps using compute_gal_pointing for full astropy fidelity.

_RADEC_CACHE: dict = {}
_LAT_R = math.radians(LEO_LAT_DEG)
_SIN_LAT = math.sin(_LAT_R)
_COS_LAT = math.cos(_LAT_R)


def _cell_radec(l: float, b: float) -> tuple[float, float]:
    """Cached ICRS (ra, dec) in degrees for galactic (l, b)."""
    key = (round(l % 360.0, 4), b)
    cached = _RADEC_CACHE.get(key)
    if cached is not None:
        return cached
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    gc = SkyCoord(l=l * u.deg, b=b * u.deg, frame='galactic')
    ra = float(gc.icrs.ra.deg)
    dec = float(gc.icrs.dec.deg)
    _RADEC_CACHE[key] = (ra, dec)
    return ra, dec


def _gmst_hours(unix_t: float) -> float:
    """Approximate GMST in hours from unix time (accurate to ~1 s)."""
    jd = unix_t / 86400.0 + 2440587.5
    d = jd - 2451545.0
    return (18.697374558 + 24.06570982441908 * d) % 24.0


def _fast_altaz(ra_deg: float, dec_deg: float, unix_t: float) -> tuple[float, float]:
    """Alt/az in degrees for Leuschner from RA/Dec at unix_t.

    Closed-form rotation; no astropy.  Matches compute_gal_pointing to
    well under a degree, which is far below the dish HPBW (3.4 deg) and
    the planner's alt/az limit margins.
    """
    lst_deg = (_gmst_hours(unix_t) * 15.0 + LEO_LON_DEG) % 360.0
    ha_deg = ((lst_deg - ra_deg + 180.0) % 360.0) - 180.0
    ha = math.radians(ha_deg)
    dec = math.radians(dec_deg)
    cos_dec = math.cos(dec)
    sin_dec = math.sin(dec)
    sin_alt = _SIN_LAT * sin_dec + _COS_LAT * cos_dec * math.cos(ha)
    if sin_alt > 1.0:
        sin_alt = 1.0
    elif sin_alt < -1.0:
        sin_alt = -1.0
    alt = math.degrees(math.asin(sin_alt))
    cos_alt_sq = 1.0 - sin_alt * sin_alt
    if cos_alt_sq <= 1e-18:
        return alt, 0.0
    cos_alt = math.sqrt(cos_alt_sq)
    sin_az = -cos_dec * math.sin(ha) / cos_alt
    cos_az = (sin_dec - _SIN_LAT * sin_alt) / (_COS_LAT * cos_alt)
    az = (math.degrees(math.atan2(sin_az, cos_az)) + 360.0) % 360.0
    return alt, az


def _time_until_set_sec(l, b, unix_t, min_alt_deg=MIN_ALT_DEG):
    """Sort key: seconds until cell (l, b) is no longer observable.

    Uses analytic hour-angle geometry (no horizon refraction, no az
    exclusion).  Three regimes:

    * Currently above ``min_alt_deg`` -> seconds until it sets.
    * Not yet risen -> seconds until rise + full above-horizon duration,
      so these cells sort *after* currently-up cells but ahead of
      circumpolar/long-pending ones.
    * Already set this sidereal pass, or never rises above ``min_alt_deg``
      -> ``-1`` (sorts to front; forward-sim will drop them).
    * Circumpolar above the limit (never sets) -> ``inf``.
    """
    ra, dec = _cell_radec(l, b)
    dec_r = math.radians(dec)
    cos_lat_dec = _COS_LAT * math.cos(dec_r)
    if abs(cos_lat_dec) < 1e-12:
        return float('inf')
    cos_H = (
        math.sin(math.radians(min_alt_deg))
        - _SIN_LAT * math.sin(dec_r)
    ) / cos_lat_dec
    if cos_H >= 1.0:
        return -1.0   # never rises above min_alt
    if cos_H <= -1.0:
        return float('inf')   # circumpolar above limit
    H_set_deg = math.degrees(math.acos(cos_H))

    # LST -> current hour angle in (-180, 180].
    lst_deg = (_gmst_hours(unix_t) * 15.0 + LEO_LON_DEG) % 360.0
    ha_now = ((lst_deg - ra + 180.0) % 360.0) - 180.0

    sidereal_rate = 360.0 / 86164.0905   # deg/sec

    if abs(ha_now) <= H_set_deg:
        # Currently above min_alt; setting in (H_set - ha_now) of sidereal time.
        return (H_set_deg - ha_now) / sidereal_rate
    if ha_now < -H_set_deg:
        # Not yet risen; rise then set.  Schedule after currently-up cells.
        time_to_rise = (-H_set_deg - ha_now) / sidereal_rate
        duration_above = 2.0 * H_set_deg / sidereal_rate
        return time_to_rise + duration_above
    # ha_now > H_set_deg: already past setting on this sidereal pass.
    return -1.0


def filter_cells_forward_simulated(cells, cell_total_time_sec, index_offset=0, unix_t=None):
    """Drop cells projected to be outside alt/az limits when the scan reaches them.

    Projects each cell's observation time as
    ``unix_t + (index_offset + i + 0.5) * cell_total_time_sec`` and
    evaluates alt/az + sun/moon separation at that time.  ``unix_t``
    defaults to the current time; pass an explicit value when planning a
    phase that will not begin executing immediately.
    """
    if not cells:
        return cells
    now = unix_t if unix_t is not None else _time.time()
    kept = []
    n_drop_alt = 0
    n_drop_az = 0
    n_drop_sun = 0
    n_drop_moon = 0
    # Periodic re-cal inserts one extra cell every RECAL_EVERY_N_CELLS, so
    # actual elapsed time per scheduled cell is scaled by this factor.
    recal_factor = (1.0 + 1.0 / RECAL_EVERY_N_CELLS) if RECAL_ENABLE else 1.0
    for i, (row, col, l, b) in enumerate(cells):
        proj_t = now + (index_offset + i + 0.5) * cell_total_time_sec * recal_factor
        ra, dec = _cell_radec(l, b)
        alt, az = _fast_altaz(ra, dec, proj_t)
        in_alt = MIN_ALT_DEG <= alt <= MAX_ALT_DEG
        in_az = AZ_MIN <= az <= AZ_MAX
        if not in_alt:
            n_drop_alt += 1
            continue
        if not in_az:
            n_drop_az += 1
            continue
        if SUN_AVOID_DEG > 0 or MOON_AVOID_DEG > 0:
            sun_ra, sun_dec, moon_ra, moon_dec = _bright_body_radec(proj_t)
            if SUN_AVOID_DEG > 0:
                sep = _angular_sep_deg(ra, dec, sun_ra, sun_dec)
                if sep < SUN_AVOID_DEG:
                    n_drop_sun += 1
                    continue
            if MOON_AVOID_DEG > 0:
                sep = _angular_sep_deg(ra, dec, moon_ra, moon_dec)
                if sep < MOON_AVOID_DEG:
                    n_drop_moon += 1
                    continue
        kept.append((row, col, l, b))
    print(f'  Forward sim ({cell_total_time_sec:.0f}s/cell, '
          f'offset {index_offset}): kept {len(kept)}/{len(cells)}, '
          f'dropped {n_drop_alt} alt, {n_drop_az} az, '
          f'{n_drop_sun} sun (<{SUN_AVOID_DEG:.0f} deg), '
          f'{n_drop_moon} moon (<{MOON_AVOID_DEG:.0f} deg)')
    return kept


def _load_completeness_index() -> dict:
    """Return cell_lb -> {(lo_round, kind): count}.

    Caches the per-file LO read in ``artifacts/nps_completeness.json`` keyed
    by mtime so subsequent planning passes are O(new files), not O(all files).
    LO is rounded to 2 decimals to avoid float-key drift.
    """
    cache_path = Path('artifacts/nps_completeness.json')
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    counts = {}   # cell_lb -> {(lo_str, kind): count}
    new_entries = 0
    for npz in glob.glob(f'{OUTPUT_DIR}/session_*/*_*/*.npz'):
        st = Path(npz).stat()
        key = f'{npz}:{int(st.st_mtime)}:{st.st_size}'
        info = cache.get(key)
        if info is None:
            try:
                with np.load(npz, allow_pickle=False) as d:
                    lo = float(d['lo_freq_mhz'])
            except (OSError, KeyError, ValueError):
                continue
            info = {'lo': round(lo, 2)}
            cache[key] = info
            new_entries += 1

        cell_dir = Path(npz).parent.name
        if cell_dir.startswith('obs_'):
            lb = cell_dir[4:]
            kind = 'obs'
        elif cell_dir.startswith('cal_'):
            lb = cell_dir[4:]
            kind = 'cal'
        else:
            continue
        bucket = counts.setdefault(lb, {})
        bucket[(info['lo'], kind)] = bucket.get((info['lo'], kind), 0) + 1

    if new_entries:
        # Drop cache entries whose files no longer exist.
        cache = {k: v for k, v in cache.items()
                 if Path(k.split(':', 1)[0]).exists()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return counts


def _is_cell_complete(bucket: dict) -> bool:
    """A cell is complete iff each LO has the required obs+cal counts."""
    lo_targets = (F1_MHZ, F2_MHZ)
    for lo in lo_targets:
        lo_key = round(lo, 2)
        if bucket.get((lo_key, 'obs'), 0) < OBS_DUMPS_PER_LO:
            return False
        if bucket.get((lo_key, 'cal'), 0) < CAL_DUMPS_PER_LO:
            return False
    return True


def filter_cells_by_existing_data(cells):
    """Skip cells that already have full per-LO obs+cal coverage.

    A cell counts as complete only when every LO in (F1_MHZ, F2_MHZ) has
    at least OBS_DUMPS_PER_LO obs dumps and CAL_DUMPS_PER_LO cal dumps.
    Cells listed in reobserve.json are always treated as incomplete.
    """
    reobserve = _load_reobserve_set()
    if reobserve:
        print(f'  Reobserve list: {len(reobserve)} cells forced incomplete')

    counts = _load_completeness_index()

    kept = []
    n_skipped = 0
    n_reobs = 0
    for row, col, l, b in cells:
        l_name = f'{l % 360.0:.2f}'.replace('.', 'p')
        cell_lb = f'{l_name}_{b}'
        if cell_lb in reobserve:
            kept.append((row, col, l, b))
            n_reobs += 1
            continue
        bucket = counts.get(cell_lb, {})
        if _is_cell_complete(bucket):
            n_skipped += 1
        else:
            kept.append((row, col, l, b))

    if n_skipped or n_reobs:
        print(f'  Existing data: {n_skipped} complete cells skipped, '
              f'{n_reobs} forced reobserve, {len(kept)} remaining')
    else:
        print(f'  No existing data found -- keeping all {len(kept)} cells')

    return kept


# ---------------------------------------------------------------------------
# Target selector + dump notifier
# ---------------------------------------------------------------------------

def make_scan_target_selector(cells, cell_event, cell_done_event, done_event):
    """Create a per-cell target selector for use with per-cell reader mode.

    Cells that are out of alt/az limits when reached are skipped (not
    stalled on).  After one pass through all cells, any skipped cells
    are retried so that cells rising into view during a long run still
    get observed.  If a full retry pass completes with zero successful
    observations, the remaining cells are abandoned and done_event is set.

    Every ``RECAL_EVERY_N_CELLS`` successful cells the selector inserts a
    drift-check pointing at the fixed circumpolar (RA, Dec) reference; the
    reader runs the normal per-cell schedule there, writing data to
    ``obs_recal_drift/`` / ``cal_recal_drift/``.  Recal cells outside the
    alt/az limits at the current sidereal time are simply skipped (the
    survey resumes immediately).

    The selector coordinates with the per-cell reader via events:

    * **cell_done_event** -- set by the reader when its per-cell schedule
      is exhausted.  The selector reacts by advancing to the next cell.
    * **cell_event** -- set by the selector to tell the reader to begin
      a new cell.  This is set *after* returning the new target to the
      pointing thread, so the reader unblocks only once the dish is
      about to settle (pointing state is still None from the preceding
      ``None`` return, guaranteeing the reader waits for the slew to
      complete before capturing).
    """
    cell_list = list(cells)
    current_cell_idx = 0
    skipped = []
    cells_observed_this_pass = 0
    # Start a session with an immediate recal so it doubles as a diode
    # warm-up and provides a t=0 reference point for the drift timeline.
    cells_since_recal = RECAL_EVERY_N_CELLS if RECAL_ENABLE else 0
    in_recal_cell = False  # True while serving a recal pointing
    chosen_recal = None      # (name, ra_deg, dec_deg) of the active recal target
    need_cell_start = True   # signal reader to begin (first cell or after advance)
    need_return_none = False  # return None once to clear pointing state on transition

    _, _, cl, cb = cell_list[0]
    print(f'  [scan] {len(cell_list)} cells, first: l={cl} b={cb}')
    if RECAL_ENABLE:
        targets_str = ' | '.join(
            f'{t["name"]} (RA={t["ra_deg"]:.1f}, Dec={t["dec_deg"]:+.1f})'
            for t in RECAL_TARGETS
        )
        print(f'  [scan] Drift recal every {RECAL_EVERY_N_CELLS} cells, '
              f'targets: {targets_str}')

    def _start_retry_pass():
        nonlocal current_cell_idx, cells_observed_this_pass
        if cells_observed_this_pass == 0:
            print(f'  [scan] No progress -- abandoning {len(skipped)} cells')
            skipped.clear()
            return False
        cell_list[:] = list(skipped)
        skipped.clear()
        current_cell_idx = 0
        cells_observed_this_pass = 0
        print(f'  [scan] Retry pass: {len(cell_list)} cells')
        return True

    def _check_end_of_list():
        """Handle end-of-list: retry skipped cells or signal done."""
        if current_cell_idx >= len(cell_list):
            if skipped:
                if not _start_retry_pass():
                    done_event.set()
                    return True
            else:
                print('  [scan] All cells complete.')
                done_event.set()
                return True
        return False

    def target_selector():
        nonlocal current_cell_idx, cells_observed_this_pass
        nonlocal cells_since_recal, in_recal_cell, chosen_recal
        nonlocal need_cell_start, need_return_none

        if done_event.is_set():
            return None

        # Cell complete -> return None once to clear pointing state,
        # then advance on the next call.
        if cell_done_event.is_set():
            cell_done_event.clear()
            if in_recal_cell:
                in_recal_cell = False
                chosen_recal = None
            else:
                cells_observed_this_pass += 1
                cells_since_recal += 1
                current_cell_idx += 1
            need_return_none = True
            need_cell_start = True
            if not in_recal_cell and _check_end_of_list():
                return None

        if need_return_none:
            need_return_none = False
            return None  # pointing -> None; reader waits for valid state

        # While a recal pointing is active, keep returning it until cell_done.
        # Without this, subsequent polls fall through to the survey-cell
        # branch and the dish slews mid-recal-schedule.
        if in_recal_cell and chosen_recal is not None:
            name, ra, dec = chosen_recal
            alt, az, _ = compute_radec_pointing(
                ra, dec,
                lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
            )
            return name, alt, az, ra, dec

        # Inject recal cell when due, before serving the next survey cell.
        # Try each recal target in order; use the first one currently in
        # alt/az limits. The two targets are chosen so at most one is
        # blocked at any LST.
        if (RECAL_ENABLE and not in_recal_cell
                and cells_since_recal >= RECAL_EVERY_N_CELLS):
            chosen = None
            tried = []
            for tgt in RECAL_TARGETS:
                alt, az, _ = compute_radec_pointing(
                    tgt['ra_deg'], tgt['dec_deg'],
                    lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
                )
                tried.append((tgt['name'], alt, az))
                if MIN_ALT_DEG <= alt <= MAX_ALT_DEG and AZ_MIN <= az <= AZ_MAX:
                    chosen = (tgt, alt, az)
                    break
            if chosen is not None:
                tgt, alt, az = chosen
                in_recal_cell = True
                chosen_recal = (tgt['name'], tgt['ra_deg'], tgt['dec_deg'])
                cells_since_recal = 0
                if need_cell_start:
                    need_cell_start = False
                    cell_event.set()
                    print(f'  [scan] Recal drift check [{tgt["name"]}]: '
                          f'alt={alt:.1f}, az={az:.1f}')
                return tgt['name'], alt, az, tgt['ra_deg'], tgt['dec_deg']
            else:
                # All recal targets out of limits right now -- skip and try
                # again after a few more cells.
                cells_since_recal = max(0, RECAL_EVERY_N_CELLS - 5)
                tried_str = ', '.join(f'{n} (alt={a:.1f},az={z:.1f})'
                                       for n, a, z in tried)
                print(f'  [scan] All recal targets out of limits '
                      f'({tried_str}); deferring.')

        if _check_end_of_list():
            return None

        _, _, cell_l, cell_b = cell_list[current_cell_idx]
        alt, az, ra, dec, _ = compute_gal_pointing(
            cell_l, cell_b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )

        if alt < MIN_ALT_DEG or alt > MAX_ALT_DEG or az < AZ_MIN or az > AZ_MAX:
            skipped.append(cell_list[current_cell_idx])
            current_cell_idx += 1
            # Reader is still blocked on cell_event -- no signal needed.
            _check_end_of_list()
            return None

        if need_cell_start:
            need_cell_start = False
            cell_event.set()
            _, _, cl, cb = cell_list[current_cell_idx]
            print(f'  [scan] Cell {current_cell_idx+1}/{len(cell_list)}: '
                  f'l={cl}, b={cb}')

        l_name = f'{cell_l % 360.0:.2f}'.replace('.', 'p')
        return f'obs_{l_name}_{cell_b}', alt, az, ra, dec

    return target_selector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_hardware():
    from ugradio.leusch import LeuschNoise, LeuschTelescope
    from ugradio.sdr import SDR

    telescope = LeuschTelescope()
    noise_ctrl = LeuschNoise()
    sdr_0 = SDR(device_index=0, direct=False,
                center_freq=F1_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
    sdr_1 = SDR(device_index=1, direct=False,
                center_freq=F1_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
    return telescope, [sdr_0, sdr_1], noise_ctrl


def _sort_set_time(cells, now):
    """Sort by time-until-set ascending (cells about to set first)."""
    return sorted(cells, key=lambda c: _time_until_set_sec(c[2], c[3], now))


def _sort_column_major(cells, l_ascending=True, b_first_ascending=True):
    """Constant-l columns swept in b, alternating direction per column (zigzag).

    Columns are taken from ``col_idx`` produced by ``build_galplane_grid``
    (cos(b)-corrected longitude bins).  ``l_ascending`` chooses whether
    the first column is the lowest-l or highest-l strip;
    ``b_first_ascending`` chooses the b direction in that first column.
    Subsequent columns alternate b direction so adjacent columns stay
    adjacent in slew distance.
    """
    by_col = {}
    for c in cells:
        by_col.setdefault(c[0], []).append(c)
    col_order = sorted(by_col, reverse=not l_ascending)
    ordered = []
    for i, col_idx in enumerate(col_order):
        col = sorted(by_col[col_idx], key=lambda c: c[3])  # by b ascending
        ascending = b_first_ascending if i % 2 == 0 else not b_first_ascending
        if not ascending:
            col = list(reversed(col))
        ordered.extend(col)
    return ordered


def _sort_row_major(cells, b_ascending=True, l_first_ascending=True):
    """Constant-b rows swept in l, alternating direction per row (zigzag).

    ``b_ascending`` chooses whether the first row is the lowest-b or
    highest-b strip; ``l_first_ascending`` chooses whether the first row
    is swept low->high in l or high->low.  Subsequent rows alternate l
    direction so adjacent rows stay adjacent in slew distance.
    """
    by_b = {}
    for c in cells:
        by_b.setdefault(c[3], []).append(c)
    b_order = sorted(by_b, reverse=not b_ascending)
    ordered = []
    for i, b in enumerate(b_order):
        row = sorted(by_b[b], key=lambda c: c[2])
        # Row 0 takes the requested direction; alternate from there.
        ascending = l_first_ascending if i % 2 == 0 else not l_first_ascending
        if not ascending:
            row = list(reversed(row))
        ordered.extend(row)
    return ordered


def plan_phase(phase, cell_total_time_sec):
    """Build + filter one phase's grid evaluated at the current time.

    Forecasts every (side, sweep-order) combination and picks whichever
    has the most forward-sim survivors.  Sweep orders are constrained to
    zigzag/parallelogram patterns since arbitrary orderings tend to add
    slew without improving coverage.
    """
    now = _time.time()
    all_phase = build_galplane_grid(phase=phase)
    print(f'\n  Total grid cells ({phase}): {len(all_phase)}')

    rising_raw, setting_raw = classify_cells_by_az_side(all_phase, unix_t=now)
    rising = filter_cells_by_existing_data(rising_raw) if rising_raw else []
    setting = filter_cells_by_existing_data(setting_raw) if setting_raw else []

    # Constant-l (column-major, cos(b)-corrected) sweeps only.
    # Forward-sim picks the l direction (asc/desc) and the starting b
    # direction; zigzag within columns is fixed.  Four variants per side.
    strategies = [
        ('col l+ b+', lambda cs: _sort_column_major(cs, l_ascending=True,  b_first_ascending=True)),
        ('col l+ b-', lambda cs: _sort_column_major(cs, l_ascending=True,  b_first_ascending=False)),
        ('col l- b+', lambda cs: _sort_column_major(cs, l_ascending=False, b_first_ascending=True)),
        ('col l- b-', lambda cs: _sort_column_major(cs, l_ascending=False, b_first_ascending=False)),
    ]

    plans = []
    for side, side_cells in [('rising', rising), ('setting', setting)]:
        if not side_cells:
            continue
        for name, sort_fn in strategies:
            ordered = sort_fn(side_cells)
            kept = filter_cells_forward_simulated(
                ordered, cell_total_time_sec, index_offset=0, unix_t=now,
            )
            plans.append((side, name, kept))

    if not plans:
        return []
    side, name, kept = max(plans, key=lambda p: len(p[2]))
    summary = ', '.join(f'{s}/{n}={len(k)}' for s, n, k in plans)
    print(f'  Picked {side}/{name}: {len(kept)} forecast cells  [{summary}]')
    return kept


def run_phase(phase, cells, telescope, sdrs, noise, outdir):
    """Run a StreamingCapture pass over one phase's cells.

    All per-pass state (events, reader closure, capture instance) is
    created here so successive phases -- and successive ``main()`` calls
    in the outer ``while True`` loop -- start from a clean slate.
    """
    cell_event = threading.Event()
    cell_done_event = threading.Event()
    done_event = threading.Event()
    pointing_wake_event = threading.Event()

    target_selector = make_scan_target_selector(
        cells, cell_event, cell_done_event, done_event,
    )

    read_fn = make_calibrated_sdr_reader(
        sdrs, noise,
        nsamples=NSAMPLES, nblocks=NBLOCKS, nfft=NFFT,
        lo_freqs_mhz=(F1_MHZ, F2_MHZ),
        cal_dumps_per_lo=CAL_DUMPS_PER_LO,
        cell_event=cell_event,
        obs_dumps_per_lo=OBS_DUMPS_PER_LO,
        cell_done_event=cell_done_event,
        stop_event=done_event,
        pointing_wake_event=pointing_wake_event,
    )

    def on_save(path, dump):
        lo_tag = f'LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
        cal_tag = 'CAL' if dump.get('noise_on') else 'OBS'
        print(f'  [{dump["target_name"]}] {cal_tag} {lo_tag} -> {path}')

    print(f'  Phase {phase} session dir: {outdir}')

    capture = StreamingCapture(
        telescope=telescope,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=outdir,
        n_writers=2,
        repoint_interval_sec=REPOINT_INTERVAL_SEC,
        on_save=on_save,
        wake_event=pointing_wake_event,
    )

    capture.run(done_event=done_event)


def main():
    print('Leuschner Sky Survey -- North Polar Spur')
    print('=' * 60)
    print(f'  Grid: l~[{L_MIN}, {L_MAX}], b=[{B_MIN}, {B_MAX}], b_step={B_STEP} deg')
    print(f'  Longitude: non-integer, centered at l={L_CENTER}, '
          f'Delta_l = {PHYSICAL_SPACING_DEG}/cos(b)')
    print(f'  LO: f1={F1_MHZ} MHz, f2={F2_MHZ} MHz')
    print(f'  Sample rate: {SAMPLE_RATE/1e6} MHz, NFFT={NFFT}, NBLOCKS={NBLOCKS}')
    print(f'  Per pointing: {CAL_DUMPS_PER_LO * N_LOS} cal + '
          f'{OBS_DUMPS_PER_LO * N_LOS} obs = {DUMPS_PER_CELL} dumps')
    print(f'  Track interval: {REPOINT_INTERVAL_SEC} s')

    stats = _load_timing_stats(
        'artifacts/nps_timing_stats.json',
        archive_dir='data/archive/nps',
    )
    # Use the mean for forward-sim instead of p50: the distribution is right-
    # skewed (p95 ~50% above p50) so the mean is a more honest expectation
    # for "how long will N cells take?".
    cell_total_time_sec = stats['cell_total_time_sec']['mean']
    print(f'  Timing stats: {stats["n_cells_observed"]} cells, '
          f'{stats["n_sessions"]} sessions, '
          f'mean cell time {cell_total_time_sec:.0f}s '
          f'(p50 {stats["cell_total_time_sec"]["p50"]:.0f}s, '
          f'p95 {stats["cell_total_time_sec"]["p95"]:.0f}s), '
          f'duty {stats["duty_cycle"]:.2f}')

    dump_time = NBLOCKS * NSAMPLES / SAMPLE_RATE
    print(f'  Integration per dump: {dump_time:.1f} s')

    # Re-detect on every main() call so a new outer-loop iteration picks
    # up sessions that landed during the previous iteration.
    next_session = _detect_next_session()

    def next_session_dir() -> str:
        nonlocal next_session
        path = f'{OUTPUT_DIR}/session_{next_session:03d}'
        next_session += 1
        return path

    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.')

    try:
        for phase in ('even',):  # 'odd' temporarily disabled — re-add to enable interleave phase
            cells = plan_phase(phase, cell_total_time_sec)
            if not cells:
                print(f'  Phase {phase}: no remaining cells, skipping.')
                continue
            recal_factor = (1.0 + 1.0 / RECAL_EVERY_N_CELLS) if RECAL_ENABLE else 1.0
            total_time_h = len(cells) * cell_total_time_sec * recal_factor / 3600
            recal_note = (f' (incl. ~{len(cells) // RECAL_EVERY_N_CELLS} recal cells)'
                          if RECAL_ENABLE else '')
            print(f'  Phase {phase}: {len(cells)} cells, '
                  f'~{total_time_h:.1f} h estimated{recal_note}')
            run_phase(phase, cells, telescope, sdrs, noise, next_session_dir())
    finally:
        noise.off()
        for sdr in sdrs:
            sdr.close()

    print('\n' + '=' * 60)
    print('  North Polar Spur survey pass complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
