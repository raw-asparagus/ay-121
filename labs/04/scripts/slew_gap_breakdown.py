"""Break down the cell-to-cell slew gap into physical-motion floor vs residual.

Usage:
    python labs/04/scripts/slew_gap_breakdown.py [archive_dir ...]

If no archive is given, scans data/main and data/nps.  Reads each dump's
recorded ``time``, ``alt_deg``, ``az_deg``, and ``target_name`` from the
npz files (no spectrum decode), groups dumps by cell, and for every
cell-to-cell boundary computes:

    measured_gap  = t_first_next - t_last_prev           (capture-end to capture-end)
    first_capture = NSAMPLES * NBLOCKS / SAMPLE_RATE     (~5.25 s for main)
    off_source    = measured_gap - first_capture          (true SDR-idle window)
    d_alt, d_az   = pointing delta between the two dumps
    phys_floor    = max(|d_alt|, |d_az|) / SLEW_RATE_DEG_S
    residual      = off_source - phys_floor               (server wait-poll + prearm + bookkeeping)

Distinguishes survey-to-survey hops from boundaries that touch a recal
cell (target_name starting with ``obs_recal_drift``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CAPTURE_SEC = (1025 * 16384) / 3.2e6      # main pipeline first-capture duration
SLEW_RATE_DEG_S = 1.2                     # implied by 360 deg / 5 min ceiling
RECAL_PREFIX = 'obs_recal_drift'


def _load_dumps(session_dir: Path) -> list[dict]:
    out = []
    for p in sorted(session_dir.glob('*/*.npz')):
        try:
            with np.load(p, allow_pickle=True) as d:
                out.append({
                    'path': p,
                    't': float(d['time']),
                    'alt': float(d['alt_deg']),
                    'az': float(d['az_deg']),
                    'target': str(d['target_name']),
                })
        except (OSError, KeyError, ValueError):
            continue
    out.sort(key=lambda r: r['t'])
    return out


def _cell_boundaries(dumps: list[dict]) -> list[tuple[dict, dict]]:
    """Return (last_dump_of_cell_N, first_dump_of_cell_N+1) pairs."""
    pairs = []
    for i in range(1, len(dumps)):
        if dumps[i]['target'] != dumps[i - 1]['target']:
            pairs.append((dumps[i - 1], dumps[i]))
    return pairs


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {'n': 0}
    a = np.asarray(values, dtype=float)
    return {
        'n': int(a.size),
        'mean': float(a.mean()),
        'p50': float(np.percentile(a, 50)),
        'p90': float(np.percentile(a, 90)),
        'p95': float(np.percentile(a, 95)),
        'max': float(a.max()),
    }


def _fmt(q: dict) -> str:
    if q.get('n', 0) == 0:
        return '  (no samples)'
    return (f'  n={q["n"]:4d}  mean={q["mean"]:6.2f}  p50={q["p50"]:6.2f}  '
            f'p90={q["p90"]:6.2f}  p95={q["p95"]:6.2f}  max={q["max"]:7.2f}')


def analyse_archive(archive_dir: Path) -> None:
    sessions = sorted(p for p in archive_dir.glob('session_*') if p.is_dir())
    if not sessions:
        print(f'{archive_dir}: no session_* directories')
        return

    survey = {'gap': [], 'off': [], 'd_alt': [], 'd_az': [], 'd_max': [],
              'floor': [], 'residual': []}
    to_recal = {'gap': [], 'off': [], 'd_max': []}
    from_recal = {'gap': [], 'off': [], 'd_max': []}
    intra_cadence = []

    worst = []                            # (off_source, residual, d_max, prev, nxt)

    for session_dir in sessions:
        dumps = _load_dumps(session_dir)
        if len(dumps) < 2:
            continue

        # intra-cell cadence (consecutive dumps with same target)
        for i in range(1, len(dumps)):
            if dumps[i]['target'] == dumps[i - 1]['target']:
                intra_cadence.append(dumps[i]['t'] - dumps[i - 1]['t'])

        for prev, nxt in _cell_boundaries(dumps):
            gap = nxt['t'] - prev['t']
            off = gap - CAPTURE_SEC
            d_alt = nxt['alt'] - prev['alt']
            d_az = nxt['az'] - prev['az']
            d_max = max(abs(d_alt), abs(d_az))
            floor = d_max / SLEW_RATE_DEG_S
            residual = off - floor

            prev_recal = prev['target'].startswith(RECAL_PREFIX)
            nxt_recal = nxt['target'].startswith(RECAL_PREFIX)

            if prev_recal and not nxt_recal:
                from_recal['gap'].append(gap)
                from_recal['off'].append(off)
                from_recal['d_max'].append(d_max)
            elif nxt_recal and not prev_recal:
                to_recal['gap'].append(gap)
                to_recal['off'].append(off)
                to_recal['d_max'].append(d_max)
            elif not prev_recal and not nxt_recal:
                survey['gap'].append(gap)
                survey['off'].append(off)
                survey['d_alt'].append(d_alt)
                survey['d_az'].append(d_az)
                survey['d_max'].append(d_max)
                survey['floor'].append(floor)
                survey['residual'].append(residual)
                worst.append((off, residual, d_max, prev, nxt))

    print(f'\n=== {archive_dir} ===')
    print(f'sessions scanned: {len(sessions)}')
    print(f'capture_sec used for first-dump subtraction: {CAPTURE_SEC:.2f} s')
    print(f'slew_rate_deg_s used for physical floor:     {SLEW_RATE_DEG_S:.2f}')
    print()
    print('Intra-cell cadence (sec, consecutive dumps same target):')
    print(_fmt(_quantiles(intra_cadence)))
    print()
    print('Survey -> survey boundaries:')
    print('  measured gap (sec, includes first-capture of new cell):')
    print(_fmt(_quantiles(survey['gap'])))
    print('  off-source window (gap - capture_sec):')
    print(_fmt(_quantiles(survey['off'])))
    print('  |d_alt| (deg):')
    print(_fmt(_quantiles([abs(x) for x in survey['d_alt']])))
    print('  |d_az|  (deg):')
    print(_fmt(_quantiles([abs(x) for x in survey['d_az']])))
    print('  max(|d_alt|,|d_az|) (deg):')
    print(_fmt(_quantiles(survey['d_max'])))
    print('  physical floor at slew_rate (sec):')
    print(_fmt(_quantiles(survey['floor'])))
    print('  residual = off_source - floor (sec):')
    print(_fmt(_quantiles(survey['residual'])))
    print()
    print('Survey -> recal boundaries:')
    print('  measured gap:')
    print(_fmt(_quantiles(to_recal['gap'])))
    print('  off-source:')
    print(_fmt(_quantiles(to_recal['off'])))
    print('  max-axis delta (deg):')
    print(_fmt(_quantiles(to_recal['d_max'])))
    print()
    print('Recal -> survey boundaries:')
    print('  measured gap:')
    print(_fmt(_quantiles(from_recal['gap'])))
    print('  off-source:')
    print(_fmt(_quantiles(from_recal['off'])))
    print('  max-axis delta (deg):')
    print(_fmt(_quantiles(from_recal['d_max'])))
    print()

    worst.sort(key=lambda r: r[1], reverse=True)
    print('Top 5 survey-to-survey boundaries by residual (likely server wait-poll'
          ' or hidden overhead):')
    for off, residual, d_max, prev, nxt in worst[:5]:
        print(f'  off={off:6.2f}s  floor_implied_by_dmax={d_max/SLEW_RATE_DEG_S:5.2f}s'
              f'  residual={residual:6.2f}s'
              f'  d_max={d_max:5.2f} deg'
              f'  prev={prev["target"]:>14s}  next={nxt["target"]:<14s}')
    print()
    print('Bottom 5 by residual (closest to physical floor; should be small or'
          ' negative if floor over-estimates):')
    for off, residual, d_max, prev, nxt in worst[-5:]:
        print(f'  off={off:6.2f}s  floor_implied_by_dmax={d_max/SLEW_RATE_DEG_S:5.2f}s'
              f'  residual={residual:6.2f}s'
              f'  d_max={d_max:5.2f} deg'
              f'  prev={prev["target"]:>14s}  next={nxt["target"]:<14s}')


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        here = Path(__file__).resolve().parents[1]      # labs/04
        args = [here / 'data' / 'main', here / 'data' / 'nps']
    for ad in args:
        if ad.exists():
            analyse_archive(ad)
        else:
            print(f'skip missing: {ad}')


if __name__ == '__main__':
    main()
