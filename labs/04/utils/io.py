"""Raw-dump loading and session summary helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.coordinates as ac
import astropy.units as u

from .cache import _memory


RECAL_CELL_PREFIXES = ('obs_recal_', 'cal_recal_')

# Per-dump scalar fields: (record_key, npz_key, decoder).  Scalars round-trip
# through np.savez as 0-D arrays; the decoder extracts a Python scalar so
# downstream code sees ergonomic native types.  Array fields (corr00, corr11)
# are already np.float64 on disk and need no conversion.
_SCALAR_FIELDS = (
    ('lo_mhz', 'lo_freq_mhz', float),
    ('noise_on', 'noise_on', bool),
    ('time', 'time', float),
    ('alt', 'alt_deg', float),
    ('az', 'az_deg', float),
    ('ra', 'ra_deg', float),
    ('dec', 'dec_deg', float),
)

# Threads for parallel np.load.  np.load releases the GIL during disk I/O,
# so a modest pool gives a near-linear speedup on warm cache without
# overwhelming the storage backend.
_LOAD_WORKERS = 16


def _load_dump(path, survey: str, session_id: str) -> dict:
    """Load a single .npz dump into a record dict (no galactic coords yet)."""
    with np.load(path, allow_pickle=True) as f:
        rec = {
            'path': path,
            'survey': survey,
            'session': session_id,
            'target': str(f['target_name']),
            'corr00': f['corr00'],
            'corr11': f['corr11'],
        }
        for out_key, npz_key, cast in _SCALAR_FIELDS:
            rec[out_key] = cast(f[npz_key])
    return rec


def _load_dumps_parallel(jobs: list[tuple]) -> list[dict]:
    """Run :func:`_load_dump` over ``jobs`` (path, survey, session_id) in a thread pool."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as ex:
        return list(ex.map(lambda a: _load_dump(*a), jobs))


def _assign_galactic(records: list[dict]) -> None:
    """Set 'gl' (2 dp) and 'gb' (integer deg) on each record from RA/Dec.

    Single vectorized SkyCoord transform instead of per-record construction.
    """
    if not records:
        return
    ra = np.fromiter((r['ra'] for r in records), dtype=np.float64, count=len(records))
    dec = np.fromiter((r['dec'] for r in records), dtype=np.float64, count=len(records))
    gal = ac.SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs').galactic
    gl = np.round(gal.l.deg, 2)
    gb = np.round(gal.b.deg).astype(int)
    for r, l_val, b_val in zip(records, gl, gb):
        r['gl'] = float(l_val)
        r['gb'] = int(b_val)


def _strip_obscal_prefix(target: str) -> str:
    """``obs_recal_drift`` / ``cal_recal_drift`` -> ``recal_drift``."""
    if target.startswith(('obs_', 'cal_')):
        return target[4:]
    return target


def _collect_dump_jobs(
    data_dirs: list[Path],
    *,
    want_recal: bool,
) -> list[tuple]:
    """Walk ``data_dirs`` and return (path, survey, session_id) jobs.

    If ``want_recal`` is False, recal cells are skipped.  If True, only recal
    cells are kept.
    """
    jobs: list[tuple] = []
    for data_dir in data_dirs:
        survey = data_dir.name
        for session_dir in sorted(data_dir.glob('session_*')):
            session_id = f'{survey}/{session_dir.name}'
            cell_dirs = (sorted(session_dir.glob('obs_*'))
                         + sorted(session_dir.glob('cal_*')))
            for cell_dir in cell_dirs:
                is_recal = cell_dir.name.startswith(RECAL_CELL_PREFIXES)
                if want_recal != is_recal:
                    continue
                for p in sorted(cell_dir.glob('*.npz')):
                    jobs.append((p, survey, session_id))
    return jobs


def _fingerprint_jobs(jobs: list[tuple]) -> tuple:
    """Cache key derived from each .npz path and its mtime.

    Invalidates the cache whenever a file is added, removed, or rewritten.
    """
    return tuple((str(p), p.stat().st_mtime_ns) for p, *_ in jobs)


@_memory.cache(ignore=['jobs'])
def _load_session_dumps_cached(fingerprint: tuple, jobs: list[tuple]) -> list[dict]:
    records = _load_dumps_parallel(jobs)
    _assign_galactic(records)
    return records


def load_session_dumps(data_dirs: list[Path]) -> list[dict]:
    """Load every science .npz dump under ``data_dirs`` into a flat list of records.

    Each record carries the correlator spectra, observation metadata, and the
    galactic (l, b) of the pointing (rounded to integer b, two decimals on l).
    Recal-drift cells are skipped; load them via :func:`load_recal_dumps`.

    Results are cached on disk via :mod:`joblib`; the cache key includes each
    file path and its mtime, so adding/removing/rewriting a dump invalidates
    the relevant entry automatically.
    """
    jobs = _collect_dump_jobs(data_dirs, want_recal=False)
    fingerprint = _fingerprint_jobs(jobs)
    return _load_session_dumps_cached(fingerprint, jobs)


@_memory.cache(ignore=['jobs'])
def _load_recal_dumps_cached(fingerprint: tuple, jobs: list[tuple]) -> list[dict]:
    records = _load_dumps_parallel(jobs)
    for r in records:
        r['target_id'] = _strip_obscal_prefix(r['target'])
    _assign_galactic(records)
    return records


def load_recal_dumps(data_dirs: list[Path]) -> list[dict]:
    """Load only recal-pointing dumps, tagged with a ``target_id`` field.

    The two recal targets used by the main survey (``recal_drift`` and
    ``recal_drift_bk``) sit at fixed (RA, Dec) and are therefore at a single
    galactic (l, b) per target.  Each record's galactic (l, b) is filled in
    just like science records; ``target_id`` strips the ``obs_``/``cal_``
    prefix so noise-on and noise-off dumps from the same target collapse to
    one key for visit clustering.
    """
    jobs = _collect_dump_jobs(data_dirs, want_recal=True)
    fingerprint = _fingerprint_jobs(jobs)
    return _load_recal_dumps_cached(fingerprint, jobs)


def build_session_summary(
    records: list[dict],
    int_time_s: float,
) -> pd.DataFrame:
    """Per-session diagnostics: wall clock, cell count, dump count, duty cycle."""
    import datetime as dt

    def _utc(ts: float) -> str:
        return dt.datetime.fromtimestamp(ts, dt.UTC).strftime('%Y-%m-%d %H:%M')

    if not records:
        return pd.DataFrame(columns=[
            'Session', 'UTC Start', 'UTC End', 'Wall (min)',
            'Obs. Cells', 'Dumps', 'Duty %',
        ])

    df = pd.DataFrame({
        'session': [r['session'] for r in records],
        'time': [r['time'] for r in records],
        'gl': [r['gl'] for r in records],
        'gb': [r['gb'] for r in records],
        'noise_on': [r['noise_on'] for r in records],
    })

    def _agg(g: pd.DataFrame) -> pd.Series:
        t0, t1 = g['time'].min(), g['time'].max()
        wall_s = t1 - t0
        n = len(g)
        duty_pct = (int_time_s * n / wall_s) * 100 if wall_s > 0 else float('nan')
        sci = g[~g['noise_on']]
        n_cells = len({(gl, gb) for gl, gb in zip(sci['gl'], sci['gb'])})
        return pd.Series({
            't_start': t0,
            't_end': t1,
            'UTC Start': _utc(t0),
            'UTC End': _utc(t1),
            'Wall (min)': round(wall_s / 60, 1),
            'Obs. Cells': n_cells,
            'Dumps': n,
            'Duty %': round(duty_pct, 1),
        })

    summary = (
        df.groupby('session', sort=False)
          .apply(_agg, include_groups=False)
          .reset_index()
          .rename(columns={'session': 'Session'})
          .sort_values(['t_start', 't_end'])
          .reset_index(drop=True)
    )
    return summary[['Session', 'UTC Start', 'UTC End', 'Wall (min)',
                    'Obs. Cells', 'Dumps', 'Duty %']]
