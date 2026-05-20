"""Shared survey runner for the Lab-4 main and NPS pipelines.

Each entrypoint script (``main``, ``nps.py``) builds a frozen
``SurveyConfig`` and calls :func:`run`.  Everything else -- grid build,
completeness index, forward-sim filtering, recal injection, session
directory management, and the synchronous coordinator wiring -- lives
here so the two entrypoints stay config-only.

The runtime path is:

    plan_phase(cfg)
        -> build cell list (filtered by accessibility, completeness,
           and forward-sim survival)
    SurveyScheduler(cells, cfg)
        -> generator of Cell objects (recal injection, retry pass,
           final recal)
    SurveyCoordinator(...)
        -> for cell in scheduler: prearm SDR, slew, run schedule, save
"""

from __future__ import annotations

import glob
import json
import re
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Allow `from utils.X import ...` when run with cwd=labs/04/.
sys.path.insert(0, '.')
from utils.timing_stats import load as _load_timing_stats  # noqa: E402
from utils.mapping import (  # noqa: E402
    build_galplane_grid,
    cell_radec as _cell_radec,
    fast_altaz as _fast_altaz,
)

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
    compute_radec_pointing,
)
from ugradiolab.capture import (
    Cell,
    Pointing,
    SDRSession,
    SurveyCoordinator,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecalTarget:
    name: str
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class SurveyConfig:
    # Identity / IO
    title: str
    output_dir: str
    artifacts_prefix: str          # e.g. 'main' or 'nps' -> artifacts/<prefix>_*.json
    timing_archive_dir: str

    # SDR + LO
    f1_mhz: float = 1419.86
    f2_mhz: float = 1421.14
    sample_rate: float = 3.2e6
    nsamples: int = 16384
    nblocks: int = 1025
    nfft: int = 1024

    # Telescope limits
    min_alt_deg: float = 17.0
    max_alt_deg: float = 83.0
    az_min: float = 7.0
    az_max: float = 348.0
    track_interval_s: float = 10.0

    # Grid bounds
    l_center: float = 120.0
    l_min: float = -10.0
    l_max: float = 250.0
    b_min: int = -6
    b_max: int = 6
    b_step: int = 2
    physical_spacing_deg: float = 2.0

    # Planner horizon: drop cells whose projected observation time is more
    # than this many hours past `now` at planning time.  Prevents the
    # planner from keeping cells that come into view many hours later but
    # then get reached by the executor (which iterates at real-time rate)
    # within minutes -- and silently skipped because they're still below
    # the horizon.  With main's outer `while True: run()` loop, the planner
    # re-runs each iteration so cells naturally roll in as they rise.
    max_planning_horizon_h: float = 4.0

    # Per-cell schedule (per-LO; ABBA expansion happens internally)
    cal_dumps_per_lo: int = 2
    obs_dumps_per_lo: int = 3

    # Recal
    recal_enable: bool = True
    recal_targets: tuple[RecalTarget, ...] = (
        RecalTarget('obs_recal_drift',    180.0, 72.0),
        RecalTarget('obs_recal_drift_bk',  90.0, 72.0),
    )
    recal_every_n_cells: int = 10
    # End-to-end recal cycle cost (last survey dump -> first survey dump
    # of the next group): slew-out + 12-dump recal block + slew-back.
    # Measured across data/main and data/nps (10 cycles as of 2026-05-13):
    #   mean=260.5  p50=245.6  p90=305.0  p95=310.2  max=315.5
    # Default is p90 -- pessimistic budget so cells survive forecast only
    # if they really fit at runtime.  Used by the planner's stepwise
    # forward sim to charge real recal cost at each boundary instead of
    # the smoothed `1 + 1/N` factor.
    recal_block_sec: float = 305.0

    # Phases
    phases: tuple[str, ...] = ('even',)

    # Diode settle (inside SDRSession; surfaced for reference)
    noise_settle_s: float = 0.5

    # Writer pool
    n_writers: int = 2

    @property
    def n_los(self) -> int:
        return 2

    @property
    def dumps_per_cell(self) -> int:
        return (self.cal_dumps_per_lo + self.obs_dumps_per_lo) * self.n_los


# ---------------------------------------------------------------------------
# Grid + fast planning math live in utils.mapping (shared with notebooks).
# ---------------------------------------------------------------------------

def classify_cells_by_az_side(cells, cfg: SurveyConfig, unix_t=None):
    """Split cells into rising and setting candidate lists at unix_t."""
    rising = []
    setting = []
    n_permanent = 0
    n_rising_now = 0
    n_setting_now = 0

    t = unix_t if unix_t is not None else _time.time()
    for row, col, l, b in cells:
        ra, dec = _cell_radec(l, b)
        max_alt = 90.0 - abs(LEO_LAT_DEG - dec)
        if max_alt < cfg.min_alt_deg:
            n_permanent += 1
            continue
        alt, az = _fast_altaz(ra, dec, t)

        in_limits = cfg.min_alt_deg <= alt <= cfg.max_alt_deg
        if az <= 180 or az > cfg.az_max:
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


def filter_cells_forward_simulated(
    cells, cfg: SurveyConfig, cell_total_time_sec: float,
    index_offset: int = 0, unix_t=None,
):
    """Drop cells projected to be outside limits when the scan reaches them."""
    if not cells:
        return cells
    now = unix_t if unix_t is not None else _time.time()
    kept = []
    n_drop_alt = n_drop_az = n_drop_horizon = 0
    horizon_s = cfg.max_planning_horizon_h * 3600.0
    # Stepwise recal accounting: every `recal_every_n_cells` survey cells,
    # one recal block (~recal_block_sec) is injected between them. The
    # session also starts with one recal (the initial diode warm-up), so
    # the count of recals completed before cell i is offset by 1 when
    # recal is enabled.
    init_recals = 1 if cfg.recal_enable else 0
    for i, (row, col, l, b) in enumerate(cells):
        global_idx = index_offset + i
        if cfg.recal_enable and cfg.recal_every_n_cells > 0:
            n_recals = init_recals + global_idx // cfg.recal_every_n_cells
        else:
            n_recals = 0
        offset_s = ((global_idx + 0.5) * cell_total_time_sec
                    + n_recals * cfg.recal_block_sec)
        if offset_s > horizon_s:
            n_drop_horizon += 1
            continue
        proj_t = now + offset_s
        ra, dec = _cell_radec(l, b)
        alt, az = _fast_altaz(ra, dec, proj_t)
        if not (cfg.min_alt_deg <= alt <= cfg.max_alt_deg):
            n_drop_alt += 1
            continue
        if not (cfg.az_min <= az <= cfg.az_max):
            n_drop_az += 1
            continue
        kept.append((row, col, l, b))
    print(f'  Forward sim ({cell_total_time_sec:.0f}s/cell + '
          f'{cfg.recal_block_sec:.0f}s/recal every {cfg.recal_every_n_cells}, '
          f'offset {index_offset}, horizon {cfg.max_planning_horizon_h:.1f} h): '
          f'kept {len(kept)}/{len(cells)}, '
          f'dropped {n_drop_alt} alt, {n_drop_az} az, '
          f'{n_drop_horizon} past horizon')
    return kept


# ---------------------------------------------------------------------------
# Completeness / reobserve
# ---------------------------------------------------------------------------

def _load_reobserve_set(cfg: SurveyConfig) -> set:
    reobs_path = Path(f'artifacts/{cfg.artifacts_prefix}_reobserve.json')
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


def _load_completeness_index(cfg: SurveyConfig) -> dict:
    cache_path = Path(f'artifacts/{cfg.artifacts_prefix}_completeness.json')
    cache: dict = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    counts: dict = {}
    new_entries = 0
    for npz in glob.glob(f'{cfg.output_dir}/session_*/*_*/*.npz'):
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
        cache = {k: v for k, v in cache.items()
                 if Path(k.split(':', 1)[0]).exists()}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return counts


def _is_cell_complete(bucket: dict, cfg: SurveyConfig) -> bool:
    for lo in (cfg.f1_mhz, cfg.f2_mhz):
        lo_key = round(lo, 2)
        if bucket.get((lo_key, 'obs'), 0) < cfg.obs_dumps_per_lo:
            return False
        if bucket.get((lo_key, 'cal'), 0) < cfg.cal_dumps_per_lo:
            return False
    return True


def filter_cells_by_existing_data(cells, cfg: SurveyConfig):
    reobserve = _load_reobserve_set(cfg)
    if reobserve:
        print(f'  Reobserve list: {len(reobserve)} cells forced incomplete')

    counts = _load_completeness_index(cfg)

    kept = []
    n_skipped = n_reobs = 0
    for row, col, l, b in cells:
        l_name = f'{l % 360.0:.2f}'.replace('.', 'p')
        cell_lb = f'{l_name}_{b}'
        if cell_lb in reobserve:
            kept.append((row, col, l, b))
            n_reobs += 1
            continue
        bucket = counts.get(cell_lb, {})
        if _is_cell_complete(bucket, cfg):
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
# Plan + sweep ordering
# ---------------------------------------------------------------------------

def _sort_column_major(cells, l_ascending=True, b_first_ascending=True):
    by_col: dict = {}
    for c in cells:
        by_col.setdefault(c[0], []).append(c)
    col_order = sorted(by_col, reverse=not l_ascending)
    ordered = []
    for i, col_idx in enumerate(col_order):
        col = sorted(by_col[col_idx], key=lambda c: c[3])
        ascending = b_first_ascending if i % 2 == 0 else not b_first_ascending
        if not ascending:
            col = list(reversed(col))
        ordered.extend(col)
    return ordered


def plan_phase(phase: str, cfg: SurveyConfig, cell_total_time_sec: float):
    """Build + filter one phase's grid evaluated at the current time.

    Picks the (rising/setting, sweep direction) plan with the most
    forward-sim survivors.
    """
    now = _time.time()
    all_phase = build_galplane_grid(
        b_min=cfg.b_min, b_max=cfg.b_max, b_step=cfg.b_step,
        l_min=cfg.l_min, l_max=cfg.l_max, l_center=cfg.l_center,
        physical_spacing_deg=cfg.physical_spacing_deg,
        phase=phase, verbose=True,
    )
    print(f'\n  Total grid cells ({phase}): {len(all_phase)}')
    if not all_phase:
        return []

    rising_raw, setting_raw = classify_cells_by_az_side(all_phase, cfg, unix_t=now)
    rising = filter_cells_by_existing_data(rising_raw, cfg) if rising_raw else []
    setting = filter_cells_by_existing_data(setting_raw, cfg) if setting_raw else []

    strategies = [
        ('col l+ b+', lambda cs: _sort_column_major(cs, True,  True)),
        ('col l+ b-', lambda cs: _sort_column_major(cs, True,  False)),
        ('col l- b+', lambda cs: _sort_column_major(cs, False, True)),
        ('col l- b-', lambda cs: _sort_column_major(cs, False, False)),
    ]

    # The initial recal is now charged inside filter_cells_forward_simulated
    # (init_recals=1 when recal_enable=True), so no index_offset bias is
    # needed here.
    plans = []
    for side, side_cells in [('rising', rising), ('setting', setting)]:
        if not side_cells:
            continue
        for name, sort_fn in strategies:
            ordered = sort_fn(side_cells)
            kept = filter_cells_forward_simulated(
                ordered, cfg, cell_total_time_sec,
                index_offset=0, unix_t=now,
            )
            plans.append((side, name, kept))

    if not plans:
        return []
    side, name, kept = max(plans, key=lambda p: len(p[2]))
    summary = ', '.join(f'{s}/{n}={len(k)}' for s, n, k in plans)
    print(f'  Picked {side}/{name}: {len(kept)} forecast cells  [{summary}]')
    return kept


# ---------------------------------------------------------------------------
# Per-cell schedule (ABBA interleave across cal/obs phases)
# ---------------------------------------------------------------------------

def build_cell_schedule(
    lo_list: tuple[float, ...],
    cal_dumps_per_lo: int,
    obs_dumps_per_lo: int,
) -> list[tuple[float, bool]]:
    """ABBA-interleaved (lo_mhz, noise_on) schedule for one cell.

    For two LOs A, B with cal=2 obs=3::

        CAL-A, CAL-B, CAL-B, CAL-A, OBS-A, OBS-B, OBS-B, OBS-A, OBS-A, OBS-B

    The obs phase starts on the same LO that cal ended on (no PLL settle
    at the cal-to-obs boundary).
    """
    abba_cycle = list(lo_list) + list(reversed(lo_list))
    cycle_len = len(abba_cycle)
    schedule: list[tuple[float, bool]] = []
    total_cal = cal_dumps_per_lo * len(lo_list)
    for i in range(total_cal):
        schedule.append((abba_cycle[i % cycle_len], True))
    total_obs = obs_dumps_per_lo * len(lo_list)
    for i in range(total_obs):
        schedule.append((abba_cycle[(total_cal + i) % cycle_len], False))
    return schedule


# ---------------------------------------------------------------------------
# Scheduler (cells -> Cell objects with recal injection + retry)
# ---------------------------------------------------------------------------

class SurveyScheduler:
    """Iterator yielding ``Cell`` objects in observation order.

    Behaviour:

    * Out-of-limits cells are appended to a skipped list at iteration
      time and retried after one full pass through the survey list.
      A retry pass that produces zero observations abandons the
      remaining skipped cells.
    * Every ``cfg.recal_every_n_cells`` successful cells the next yield
      is a recal cell.  ``cfg.recal_targets`` are tried in order; the
      first one in alt/az limits is chosen.  If none are accessible the
      recal is deferred (with a small back-off) and the survey continues.
    * Paired recal: when at least two recal targets are accessible *and*
      one of them has just transitioned into the alt/az limits (or it
      is the very first recal of the session), every accessible target
      is yielded back-to-back so the reduction can tie their gain/T_cal
      references together.  Sessions where only one target is ever
      accessible behave as before -- one target is self-consistent.
    * One final recal fires after the last survey cell (or after the
      retry pass abandons), so every session bookends with drift refs.
    """

    def __init__(self, cells, cfg: SurveyConfig) -> None:
        self._raw_cells = list(cells)
        self._cfg = cfg

    def __iter__(self) -> Iterator[Cell]:
        cfg = self._cfg
        cell_list = list(self._raw_cells)
        skipped: list = []
        observed_this_pass = 0
        # Start with a recal so it doubles as a diode warm-up + t=0 reference.
        since_recal = cfg.recal_every_n_cells if cfg.recal_enable else 0
        final_recal_pending = cfg.recal_enable
        idx = 0
        skip_log: list[str] = []  # accumulated skip reasons for this pass
        # Paired-recal bookkeeping: track per-target accessibility so we can
        # fire all currently-accessible targets back-to-back the first time
        # they are simultaneously in limits (session start) or when one of
        # them transitions out-of-limits -> in-limits during the session.
        last_recal_in_limits: dict[str, bool] = {
            t.name: False for t in cfg.recal_targets
        }
        pending_recals: list[Cell] = []

        if cell_list:
            _, _, cl, cb = cell_list[0]
            print(f'  [scan] {len(cell_list)} cells, first: l={cl} b={cb}')
        else:
            print('  [scan] (empty cell list)')
        if cfg.recal_enable:
            tgts = ' | '.join(
                f'{t.name} (RA={t.ra_deg:.1f}, Dec={t.dec_deg:+.1f})'
                for t in cfg.recal_targets
            )
            print(f'  [scan] Drift recal every {cfg.recal_every_n_cells} cells, '
                  f'targets: {tgts}')

        while True:
            # 0) Drain any pending paired-recal cells before anything else.
            if pending_recals:
                next_cell = pending_recals.pop(0)
                since_recal = 0
                yield next_cell
                continue

            # 1) Recal injection.
            if (cfg.recal_enable
                    and cfg.recal_targets
                    and since_recal >= cfg.recal_every_n_cells):
                recal_cells = self._build_recal_cells()
                in_limits_now = {n: (c is not None) for n, c in recal_cells.items()}
                accessible = [
                    recal_cells[t.name] for t in cfg.recal_targets
                    if in_limits_now[t.name]
                ]
                if accessible:
                    newly_accessible = [
                        n for n, ok in in_limits_now.items()
                        if ok and not last_recal_in_limits[n]
                    ]
                    pair = len(accessible) >= 2 and bool(newly_accessible)
                    last_recal_in_limits = in_limits_now
                    if pair:
                        names = ', '.join(c.pointing.target_name for c in accessible)
                        trigger = ('session start' if all(
                            not v for k, v in last_recal_in_limits.items()
                            if k not in newly_accessible
                        ) else f'newly accessible: {",".join(newly_accessible)}')
                        print(f'  [scan] Paired recal ({trigger}); firing '
                              f'{len(accessible)} targets back-to-back: {names}')
                        pending_recals = accessible[1:]
                        since_recal = 0
                        yield accessible[0]
                        continue
                    since_recal = 0
                    yield accessible[0]
                    continue
                # All recal targets out of limits right now -- back off and
                # keep doing survey cells; retry recal in a few cells.
                last_recal_in_limits = in_limits_now
                since_recal = max(0, cfg.recal_every_n_cells - 5)

            # 2) Out of survey cells?
            if idx >= len(cell_list):
                if skip_log:
                    print(f'  [scan] Pass summary: {observed_this_pass} observed, '
                          f'{len(skip_log)} skipped (first few: '
                          f'{"; ".join(skip_log[:5])})')
                    skip_log = []
                if skipped and observed_this_pass > 0:
                    cell_list = skipped
                    skipped = []
                    idx = 0
                    observed_this_pass = 0
                    print(f'  [scan] Retry pass: {len(cell_list)} cells')
                    continue
                if final_recal_pending:
                    final_recal_pending = False
                    since_recal = cfg.recal_every_n_cells
                    if skipped:
                        print(f'  [scan] No more accessible cells '
                              f'({len(skipped)} abandoned) -- final recal.')
                        skipped = []
                    else:
                        print('  [scan] All cells complete -- final recal.')
                    continue
                if skipped:
                    print(f'  [scan] Survey complete; abandoned {len(skipped)} cells.')
                else:
                    print('  [scan] Survey complete.')
                return

            # 3) Serve next survey cell.
            _, _, cell_l, cell_b = cell_list[idx]
            alt, az, ra, dec, _ = compute_gal_pointing(
                cell_l, cell_b,
                lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
            )
            if not self._in_limits(alt, az):
                reasons = []
                if not (cfg.min_alt_deg <= alt <= cfg.max_alt_deg):
                    reasons.append(f'alt={alt:.1f}')
                if not (cfg.az_min <= az <= cfg.az_max):
                    reasons.append(f'az={az:.1f}')
                skip_log.append(f'l={cell_l},b={cell_b} ({",".join(reasons)})')
                skipped.append(cell_list[idx])
                idx += 1
                continue

            l_name = f'{cell_l % 360.0:.2f}'.replace('.', 'p')
            target_name = f'obs_{l_name}_{cell_b}'
            pointing = Pointing(
                target_name=target_name,
                alt_deg=alt, az_deg=az, ra_deg=ra, dec_deg=dec,
            )
            schedule = build_cell_schedule(
                (cfg.f1_mhz, cfg.f2_mhz),
                cfg.cal_dumps_per_lo, cfg.obs_dumps_per_lo,
            )
            recompute = self._make_gal_recompute(cell_l, cell_b)
            print(f'  [scan] Cell {idx + 1}/{len(cell_list)}: '
                  f'l={cell_l}, b={cell_b}')
            yield Cell(pointing=pointing, schedule=schedule,
                       recompute_altaz=recompute)
            idx += 1
            observed_this_pass += 1
            since_recal += 1

    # ------------------------------------------------------------------

    def _in_limits(self, alt: float, az: float) -> bool:
        cfg = self._cfg
        return (cfg.min_alt_deg <= alt <= cfg.max_alt_deg
                and cfg.az_min <= az <= cfg.az_max)

    def _build_recal_cells(self) -> dict[str, 'Cell | None']:
        """Evaluate every recal target's current alt/az and build a Cell
        for each that is in limits.  Returns an ordered dict keyed by
        target name, value=None for out-of-limits targets.
        """
        cfg = self._cfg
        out: dict[str, Cell | None] = {}
        diag = []
        for tgt in cfg.recal_targets:
            alt, az, _ = compute_radec_pointing(
                tgt.ra_deg, tgt.dec_deg,
                lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
            )
            in_lim = self._in_limits(alt, az)
            diag.append(f'{tgt.name} (alt={alt:.1f},az={az:.1f},'
                        f'{"ok" if in_lim else "out"})')
            if in_lim:
                pointing = Pointing(
                    target_name=tgt.name,
                    alt_deg=alt, az_deg=az,
                    ra_deg=tgt.ra_deg, dec_deg=tgt.dec_deg,
                )
                schedule = build_cell_schedule(
                    (cfg.f1_mhz, cfg.f2_mhz),
                    cfg.cal_dumps_per_lo, cfg.obs_dumps_per_lo,
                )
                recompute = self._make_radec_recompute(tgt.ra_deg, tgt.dec_deg)
                out[tgt.name] = Cell(pointing=pointing, schedule=schedule,
                                     recompute_altaz=recompute)
            else:
                out[tgt.name] = None
        n_ok = sum(c is not None for c in out.values())
        if n_ok == 0:
            print(f'  [scan] All recal targets out of limits ({"; ".join(diag)}); '
                  f'deferring.')
        else:
            print(f'  [scan] Recal drift check: {"; ".join(diag)}')
        return out

    @staticmethod
    def _make_gal_recompute(l: float, b: float):
        def _recompute() -> tuple[float, float]:
            alt, az, _, _, _ = compute_gal_pointing(
                l, b,
                lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
            )
            return alt, az
        return _recompute

    @staticmethod
    def _make_radec_recompute(ra: float, dec: float):
        def _recompute() -> tuple[float, float]:
            alt, az, _ = compute_radec_pointing(
                ra, dec,
                lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
            )
            return alt, az
        return _recompute


# ---------------------------------------------------------------------------
# Session directory
# ---------------------------------------------------------------------------

def detect_next_session(output_dir: str) -> int:
    highest = 0
    for d in glob.glob(f'{output_dir}/session_*'):
        m = re.search(r'session_(\d+)$', d)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1 if highest else 1


# ---------------------------------------------------------------------------
# Hardware setup
# ---------------------------------------------------------------------------

def setup_hardware(cfg: SurveyConfig):
    from ugradio.leusch import LeuschNoise, LeuschTelescope
    from ugradio.sdr import SDR

    telescope = LeuschTelescope()
    noise_ctrl = LeuschNoise()
    sdr_0 = SDR(device_index=0, direct=False,
                center_freq=cfg.f1_mhz * 1e6,
                sample_rate=cfg.sample_rate, gain=0.0)
    sdr_1 = SDR(device_index=1, direct=False,
                center_freq=cfg.f1_mhz * 1e6,
                sample_rate=cfg.sample_rate, gain=0.0)
    return telescope, [sdr_0, sdr_1], noise_ctrl


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def _on_save(path: str, dump: dict) -> None:
    lo_tag = f'LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    cal_tag = 'CAL' if dump.get('noise_on') else 'OBS'
    print(f'  [{dump["target_name"]}] {cal_tag} {lo_tag} -> {path}')


def run_phase(
    phase: str,
    cells,
    cfg: SurveyConfig,
    telescope,
    sdrs,
    noise,
    outdir: str,
) -> None:
    print(f'  Phase {phase} session dir: {outdir}')
    sdr_session = SDRSession(
        sdrs=sdrs, noise=noise,
        nsamples=cfg.nsamples, nblocks=cfg.nblocks, nfft=cfg.nfft,
        noise_settle_s=cfg.noise_settle_s,
    )
    scheduler = SurveyScheduler(cells, cfg)
    coordinator = SurveyCoordinator(
        telescope=telescope,
        sdr_session=sdr_session,
        scheduler=scheduler,
        outdir=outdir,
        n_writers=cfg.n_writers,
        track_interval_s=cfg.track_interval_s,
        on_save=_on_save,
    )
    coordinator.run()


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def run(cfg: SurveyConfig) -> None:
    """One full pass through the configured phases."""
    print(f'Leuschner Sky Survey -- {cfg.title}')
    print('=' * 60)
    print(f'  Grid: l~[{cfg.l_min}, {cfg.l_max}], '
          f'b=[{cfg.b_min}, {cfg.b_max}], b_step={cfg.b_step} deg')
    print(f'  Longitude: non-integer, centered at l={cfg.l_center}, '
          f'Delta_l = {cfg.physical_spacing_deg}/cos(b)')
    print(f'  LO: f1={cfg.f1_mhz} MHz, f2={cfg.f2_mhz} MHz')
    print(f'  Sample rate: {cfg.sample_rate / 1e6} MHz, '
          f'NFFT={cfg.nfft}, NBLOCKS={cfg.nblocks}')
    print(f'  Per pointing: {cfg.cal_dumps_per_lo * cfg.n_los} cal + '
          f'{cfg.obs_dumps_per_lo * cfg.n_los} obs = {cfg.dumps_per_cell} dumps')
    print(f'  Track interval: {cfg.track_interval_s} s')

    stats = _load_timing_stats(
        f'artifacts/{cfg.artifacts_prefix}_timing_stats.json',
        archive_dir=cfg.timing_archive_dir,
    )
    cell_total_time_sec = stats['cell_total_time_sec']['mean']
    print(f'  Timing stats: {stats["n_cells_observed"]} cells, '
          f'{stats["n_sessions"]} sessions, '
          f'mean cell time {cell_total_time_sec:.0f}s '
          f'(p50 {stats["cell_total_time_sec"]["p50"]:.0f}s, '
          f'p95 {stats["cell_total_time_sec"]["p95"]:.0f}s), '
          f'duty {stats["duty_cycle"]:.2f}')

    dump_time = cfg.nblocks * cfg.nsamples / cfg.sample_rate
    print(f'  Integration per dump: {dump_time:.1f} s')

    next_session = detect_next_session(cfg.output_dir)

    def next_session_dir() -> str:
        nonlocal next_session
        path = f'{cfg.output_dir}/session_{next_session:03d}'
        next_session += 1
        return path

    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware(cfg)
    print('Hardware ready.')

    try:
        for phase in cfg.phases:
            cells = plan_phase(phase, cfg, cell_total_time_sec)
            if not cells:
                print(f'  Phase {phase}: no remaining cells, skipping.')
                continue
            recal_factor = (
                (1.0 + 1.0 / cfg.recal_every_n_cells) if cfg.recal_enable else 1.0
            )
            total_time_h = len(cells) * cell_total_time_sec * recal_factor / 3600
            recal_note = (f' (incl. ~{len(cells) // cfg.recal_every_n_cells} '
                          f'recal cells)' if cfg.recal_enable else '')
            print(f'  Phase {phase}: {len(cells)} cells, '
                  f'~{total_time_h:.1f} h estimated{recal_note}')
            run_phase(phase, cells, cfg, telescope, sdrs, noise,
                      next_session_dir())
    finally:
        try:
            noise.off()
        except Exception:
            pass
        for sdr in sdrs:
            try:
                sdr.close()
            except Exception:
                pass

    print('\n' + '=' * 60)
    print(f'  {cfg.title} pass complete!')
    print('=' * 60)
