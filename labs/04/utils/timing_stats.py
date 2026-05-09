"""Empirical scan-timing statistics from the main archive.

Used by ``labs/04/scripts/main/radio`` and the planning notebooks
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

# Capture duration assuming the radio defaults: NBLOCKS=1025, NSAMPLES=32768,
# SAMPLE_RATE=3.2 MHz.  Used only to derive a rough duty cycle from cadence.
_NOMINAL_CAPTURE_SEC = (1025 * 32768) / 3.2e6  # ~10.49 s

DEFAULTS = {
    'generated_at': 0.0,
    'n_sessions': 0,
    'n_cells_observed': 0,
    'intra_cell_cadence_sec': {'mean': 10.5, 'p50': 10.5, 'p95': 11.0},
    'slew_gap_sec':           {'mean': 15.0, 'p50': 12.0, 'p95': 30.0},
    'cell_total_time_sec':    {'mean': 100.0, 'p50': 95.0, 'p95': 140.0},
    'duty_cycle': 0.78,
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


def compute(archive_dir) -> dict:
    """Walk ``archive_dir/session_*/<obs|cal>_*/*.npz`` filenames, compute stats."""
    archive_path = Path(archive_dir)
    sessions = sorted(p for p in archive_path.glob('session_*') if p.is_dir())

    intra_cell: list[float] = []
    slew_gap: list[float] = []
    cell_total: list[float] = []
    n_cells = 0

    for session_dir in sessions:
        dumps: list[tuple[str, float]] = []
        for dump_path in session_dir.glob('*/*.npz'):
            parsed = _parse_filename(dump_path.name)
            if parsed is not None:
                dumps.append(parsed)
        dumps.sort(key=lambda x: x[1])
        if not dumps:
            continue

        prev_target, prev_t = dumps[0]
        cell_start = prev_t
        n_cells += 1

        for target, t in dumps[1:]:
            gap = t - prev_t
            if target == prev_target:
                intra_cell.append(gap)
            else:
                slew_gap.append(gap)
                cell_total.append((prev_t - cell_start) + gap)
                cell_start = t
                n_cells += 1
            prev_target, prev_t = target, t

    if intra_cell:
        avg_cadence = float(np.mean(intra_cell))
        duty = _NOMINAL_CAPTURE_SEC / avg_cadence if avg_cadence > 0 else DEFAULTS['duty_cycle']
    else:
        duty = DEFAULTS['duty_cycle']

    return {
        'generated_at': time.time(),
        'n_sessions': len(sessions),
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
