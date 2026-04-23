#!/usr/bin/env python3
"""
migrate_streaming.py — Restructure streaming survey data from
  streaming/DR{N}[ab]/scan_r{row}_c{col}/*.npz
to
  streaming/session_{NNN}/<obs|cal>_{l}_{b}/<obs|cal>_{l}_{b}_{YYYYMMDD}_{HHMMSS}.npz

Session boundaries: a new session starts at the first dump, and at every
noise_on=True dump that immediately follows a noise_on=False dump.

Usage (from project root):
    python src/migrate_streaming.py                  # dry-run (no writes)
    python src/migrate_streaming.py --execute        # perform migration
    python src/migrate_streaming.py --streaming-dir PATH
"""

import argparse
import datetime
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


DR_RE = re.compile(r'^DR\d+[ab]?$')


def gal_int(ra_deg: float, dec_deg: float) -> tuple[int, int]:
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    l = round(coord.galactic.l.deg) % 360
    b = round(coord.galactic.b.deg)
    return l, b


def cell_name(is_cal: bool, l: int, b: int) -> str:
    return f"{'cal' if is_cal else 'obs'}_{l}_{b}"


def session_name(n: int) -> str:
    return f"session_{n:03d}"


def utc_str(unix_t: float) -> str:
    return datetime.datetime.utcfromtimestamp(unix_t).strftime('%Y%m%d_%H%M%S')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true',
                        help='Perform migration. Default is dry-run (no writes).')
    parser.add_argument('--streaming-dir', default='data/lab04/streaming',
                        help='Path to the streaming root directory.')
    args = parser.parse_args()

    dry_run = not args.execute
    streaming_dir = Path(args.streaming_dir)

    if not streaming_dir.exists():
        sys.exit(f'ERROR: streaming dir not found: {streaming_dir}')

    print(f"{'[DRY RUN] ' if dry_run else ''}Streaming dir: {streaming_dir.resolve()}\n")

    # ── Phase 1: collect minimal metadata ────────────────────────────────────
    print('Phase 1: Scanning dumps...')

    dr_dirs = sorted(d for d in streaming_dir.iterdir()
                     if d.is_dir() and DR_RE.match(d.name))
    if not dr_dirs:
        sys.exit('ERROR: No DR* directories found.')

    dumps = []
    for dr_dir in dr_dirs:
        for cell_dir in sorted(dr_dir.glob('scan_r*_c*')):
            for npz in sorted(cell_dir.glob('*.npz')):
                try:
                    with np.load(npz, allow_pickle=True) as f:
                        dumps.append({
                            'path':     npz,
                            'time':     float(f['time']),
                            'noise_on': bool(f['noise_on']),
                            'ra_deg':   float(f['ra_deg']),
                            'dec_deg':  float(f['dec_deg']),
                        })
                except Exception as exc:
                    print(f'  WARNING: cannot read {npz}: {exc}')

    print(f'  {len(dumps)} dumps across {len(dr_dirs)} DR directories')

    # ── Phase 2: sort chronologically ────────────────────────────────────────
    dumps.sort(key=lambda d: d['time'])

    # ── Phase 3: assign sessions ──────────────────────────────────────────────
    print('\nPhase 3: Detecting sessions...')

    sid = 0
    prev_noise = None
    for dump in dumps:
        curr_noise = dump['noise_on']
        if prev_noise is None or (curr_noise and not prev_noise):
            sid += 1
        dump['session_id'] = sid
        prev_noise = curr_noise

    n_sessions = sid
    by_session: dict[int, list] = defaultdict(list)
    for dump in dumps:
        by_session[dump['session_id']].append(dump)

    edge_cases: list[str] = []
    for s in range(1, n_sessions + 1):
        sdumps = by_session[s]
        n_cal = sum(1 for d in sdumps if d['noise_on'])
        n_obs = sum(1 for d in sdumps if not d['noise_on'])
        t0 = utc_str(sdumps[0]['time'])
        t1 = utc_str(sdumps[-1]['time'])
        print(f'  {session_name(s)}: {n_cal:4d} cal  {n_obs:5d} obs | {t0} → {t1}')
        if n_cal == 0:
            edge_cases.append(f'{session_name(s)}: no calibration dumps')
        if n_obs == 0:
            edge_cases.append(f'{session_name(s)}: no observation dumps')

    # ── Phase 4: compute galactic coords and new paths ────────────────────────
    print('\nPhase 4: Building path map...')

    seen_new_paths: dict[Path, list] = defaultdict(list)
    orig_cell_coords: dict[str, set] = defaultdict(set)

    for dump in dumps:
        l, b = gal_int(dump['ra_deg'], dump['dec_deg'])
        ts = utc_str(dump['time'])
        cname = cell_name(dump['noise_on'], l, b)
        sname = session_name(dump['session_id'])
        fname = f'{cname}_{ts}.npz'
        new_path = streaming_dir / sname / cname / fname

        dump['gal_l'] = l
        dump['gal_b'] = b
        dump['new_cell'] = cname
        dump['new_path'] = new_path

        seen_new_paths[new_path].append(dump['path'])
        orig_cell_coords[dump['path'].parent.name].add((l, b))

    # timestamp collisions
    for new_path, srcs in seen_new_paths.items():
        if len(srcs) > 1:
            edge_cases.append(
                f'COLLISION: {len(srcs)} dumps → {new_path.relative_to(streaming_dir)}'
                f' (sources: {[str(s.name) for s in srcs]})'
            )

    # within-cell galactic disagreement (dumps in same scan_r*_c* round to different coords)
    for orig_cell, coord_set in orig_cell_coords.items():
        if len(coord_set) > 1:
            edge_cases.append(
                f'SPLIT: {orig_cell} contains {len(coord_set)} distinct galactic pointings:'
                f' {sorted(coord_set)}'
            )

    # ── Phase 5: edge-case report ─────────────────────────────────────────────
    if edge_cases:
        print(f'\n*** {len(edge_cases)} EDGE CASE(S) DETECTED — review before proceeding ***')
        for ec in edge_cases:
            print(f'  ! {ec}')
    else:
        print('  No edge cases.')

    # ── Dry-run summary ───────────────────────────────────────────────────────
    if dry_run:
        print('\n--- DRY RUN SUMMARY ---')
        for dump in dumps[:8]:
            old = dump['path'].relative_to(streaming_dir)
            new = dump['new_path'].relative_to(streaming_dir)
            print(f'  {old}')
            print(f'    → {new}')
        print(f'  ... ({len(dumps)} total)')
        print(f'\n{len(dumps)} dumps → {n_sessions} sessions')
        print('Re-run with --execute to perform migration.')
        return

    # ── Phase 6: execute ──────────────────────────────────────────────────────
    print(f'\nPhase 6: Migrating {len(dumps)} dumps...')
    ok = failed = 0
    for i, dump in enumerate(dumps):
        if i % 1000 == 0:
            print(f'  {i}/{len(dumps)}')
        new_path: Path = dump['new_path']
        new_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with np.load(dump['path'], allow_pickle=True) as f:
                data = dict(f)
            data['target_name'] = np.str_(dump['new_cell'])
            np.savez(new_path, **data)
            ok += 1
        except Exception as exc:
            print(f'  ERROR: {dump["path"]}: {exc}')
            failed += 1

    print(f'\nDone: {ok} succeeded, {failed} failed.')
    if edge_cases:
        print(f'Note: {len(edge_cases)} edge case(s) flagged above.')


if __name__ == '__main__':
    main()
