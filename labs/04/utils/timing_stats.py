"""Empirical scan-timing statistics from the main archive.

Used by ``labs/04/scripts/main/radio.py`` and the planning notebooks
(``main_scan_plan.ipynb``, ``01_scan_plan.ipynb``) so the same measured
cadence and slew-time distributions drive the forward-simulating planner
and any visualisations.

Filenames in the archive encode the dump time:
``{obs|cal}_{l_name}_{b}_{YYYYMMDD}_{HHMMSS}.npz``.  We parse these
without opening the npz contents, so a full archive walk on the Pi is
seconds, not minutes.

Public API
----------
``compute(archive_dir)``  -- walk archive filenames, return stats dict.
``save(stats, path)``     -- serialise to JSON.
``load(path, archive_dir=None, max_age_sec=86400)`` -- read cached
    JSON; auto-regenerate if cache is older than the newest session in
    archive_dir.  Falls back to module-level DEFAULTS if neither cache
    nor archive is available.

Stats schema
------------
{
  "generated_at": <unix_t when generated>,
  "n_sessions":   <int>,
  "n_cells_observed": <int>,
  "intra_cell_cadence_sec": {"mean", "p50", "p95"},   # gap between dumps within same cell
  "slew_gap_sec":          {"mean", "p50", "p95"},   # gap between cells
  "cell_total_time_sec":   {"mean", "p50", "p95"},   # cell duration + slew to next
  "duty_cycle":            <float in [0, 1]>,         # capture / wallclock estimate
}
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np


_DUMP_RE = re.compile(
    r'^(?:obs|cal)_(.+?)_(-?\d+)_(\d{8}_\d{6})\.npz$'
)

# Capture duration assuming the radio defaults: NBLOCKS=1025, NSAMPLES=16384,
# SAMPLE_RATE=3.2 MHz.  Used only to derive a rough duty cycle from cadence.
_NOMINAL_CAPTURE_SEC = (1025 * 16384) / 3.2e6  # ~5.25 s

# Priors used until enough new sessions land for compute() to overwrite them.
# Halved-capture regime: ~5 s per dump capture, slew time unchanged.
DEFAULTS = {
    'generated_at': 0.0,
    'n_sessions': 0,
    'n_cells_observed': 0,
    'intra_cell_cadence_sec': {'mean':  6.5, 'p50':  6.0, 'p95': 11.0},
    'slew_gap_sec':           {'mean': 40.0, 'p50': 36.0, 'p95': 57.0},
    'cell_total_time_sec':    {'mean': 95.0, 'p50': 88.0, 'p95': 140.0},
    'duty_cycle': 0.55,
}


def _parse_filename(name: str) -> tuple[str, float] | None:
    """Return (cell_key, unix_timestamp) or None if name doesn't match."""
    m = _DUMP_RE.match(name)
    if m is None:
        return None
    l_name, b_str, dt_str = m.group(1), m.group(2), m.group(3)
    cell_key = f'{l_name}_{b_str}'
    try:
        t = time.mktime(time.strptime(dt_str, '%Y%m%d_%H%M%S'))
    except ValueError:
        return None
    return cell_key, t


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {'mean': float('nan'), 'p50': float('nan'), 'p95': float('nan')}
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(arr.mean()),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
    }


#  Pre-rewrite captures used NSAMPLES=32768 -> ~13 s intra-cell cadence;
#  post-rewrite uses NSAMPLES=16384 -> ~6 s.  Sessions whose median intra-cell
#  gap exceeds this threshold are dropped from the stats so the forecast
#  reflects only the current pipeline.  Auto-segments without a date cutoff.
_POSTREWRITE_MAX_INTRA_CELL_SEC = 8.0


def compute(archive_dir) -> dict:
    """Walk ``archive_dir/session_*/<obs|cal>_*/*.npz`` filenames, compute stats.

    Sessions are filtered to the post-rewrite pipeline by per-session
    median intra-cell cadence (see ``_POSTREWRITE_MAX_INTRA_CELL_SEC``).
    """
    archive_path = Path(archive_dir)
    sessions = sorted(p for p in archive_path.glob('session_*') if p.is_dir())

    intra_cell: list[float] = []
    slew_gap: list[float] = []
    cell_total: list[float] = []
    n_cells = 0
    n_sessions_used = 0
    n_sessions_skipped = 0

    for session_dir in sessions:
        dumps: list[tuple[str, float]] = []
        for dump_path in session_dir.glob('*/*.npz'):
            parsed = _parse_filename(dump_path.name)
            if parsed is not None:
                dumps.append(parsed)
        dumps.sort(key=lambda x: x[1])
        if not dumps:
            continue

        # First pass: collect this session's intra-cell gaps and decide
        # whether it's a post-rewrite session.
        s_intra: list[float] = []
        s_slew: list[float] = []
        s_cell_total: list[float] = []
        s_n_cells = 1
        prev_target, prev_t = dumps[0]
        cell_start = prev_t
        for target, t in dumps[1:]:
            gap = t - prev_t
            if target == prev_target:
                s_intra.append(gap)
            else:
                s_slew.append(gap)
                s_cell_total.append((prev_t - cell_start) + gap)
                cell_start = t
                s_n_cells += 1
            prev_target, prev_t = target, t

        if not s_intra:
            # No multi-dump cells -- can't classify; skip conservatively.
            n_sessions_skipped += 1
            continue

        med = float(np.percentile(s_intra, 50))
        if med > _POSTREWRITE_MAX_INTRA_CELL_SEC:
            n_sessions_skipped += 1
            continue

        intra_cell.extend(s_intra)
        slew_gap.extend(s_slew)
        cell_total.extend(s_cell_total)
        n_cells += s_n_cells
        n_sessions_used += 1

    # End-to-end duty cycle: on-source integration / total wallclock per cell
    # (includes slew + dump overhead).  This is the metric that drives
    # observation-time forecasts, so it must agree with cell_total_time_sec.
    if cell_total and n_cells:
        avg_cell_total = float(np.mean(cell_total))
        avg_dumps_per_cell = (n_cells + len(intra_cell)) / n_cells
        on_source_per_cell = avg_dumps_per_cell * _NOMINAL_CAPTURE_SEC
        duty = on_source_per_cell / avg_cell_total if avg_cell_total > 0 else DEFAULTS['duty_cycle']
    else:
        duty = DEFAULTS['duty_cycle']

    # If no post-rewrite sessions were usable, fall back to DEFAULTS rather
    # than emit NaN quantiles (which would break the planner downstream).
    if n_sessions_used == 0:
        out = dict(DEFAULTS)
        out['generated_at'] = time.time()
        out['n_sessions'] = 0
        out['n_sessions_skipped_prerewrite'] = n_sessions_skipped
        return out

    return {
        'generated_at': time.time(),
        'n_sessions': n_sessions_used,
        'n_sessions_skipped_prerewrite': n_sessions_skipped,
        'n_cells_observed': n_cells,
        'intra_cell_cadence_sec': _quantiles(intra_cell),
        'slew_gap_sec': _quantiles(slew_gap),
        'cell_total_time_sec': _quantiles(cell_total),
        'duty_cycle': float(duty),
    }


def save(stats: dict, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2))


def load(path, archive_dir=None, max_age_sec: float = 86400.0) -> dict:
    """Load cached stats, regenerating if stale and archive_dir is provided.

    Order of preference:
      1. Cache exists and is newer than newest session in archive_dir -> use cache.
      2. Archive available -> recompute, save, return.
      3. Cache exists but archive missing -> use stale cache.
      4. Neither -> return DEFAULTS.
    """
    cache = Path(path)
    cached = None
    if cache.exists():
        try:
            cached = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None

    archive_path = Path(archive_dir) if archive_dir is not None else None

    if cached is not None and archive_path is not None and archive_path.exists():
        newest_session = max(
            (s.stat().st_mtime for s in archive_path.glob('session_*')),
            default=0.0,
        )
        gen = cached.get('generated_at', 0.0)
        if gen >= newest_session - 60.0 and (time.time() - gen) < max_age_sec:
            return cached

    if archive_path is not None and archive_path.exists():
        stats = compute(archive_path)
        try:
            save(stats, cache)
        except OSError:
            pass
        return stats

    if cached is not None:
        return cached

    return dict(DEFAULTS)


def _cli() -> None:
    """Regenerate the default cache from the default archive path."""
    here = Path(__file__).resolve()
    lab_dir = here.parents[1]  # labs/04/
    archive = lab_dir / 'data' / 'archive' / 'main'
    out = lab_dir / 'artifacts' / 'main_timing_stats.json'
    if not archive.exists():
        print(f'archive not found: {archive}')
        return
    stats = compute(archive)
    save(stats, out)
    print(f'wrote {out}')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    _cli()
