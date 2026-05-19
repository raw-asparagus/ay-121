#!/usr/bin/env python3
"""Lab 4 - Leuschner Sky Survey -- Edge-clipped recheck pass.

Reads ``artifacts/edge_clipped_recheck.json`` (written by Section 7b of
``main_scan_load.ipynb``) and re-observes each flagged cell at the
recommended frequency-switched LO pair (lower or higher than the main
survey's pair) so the line wings that were clipped get covered.

Single-shot driver.  Cells are split by suggested LO pair; within each
group they are observed in |b|-ascending order (manifest order).
Sessions land in ``data/edges/session_NNN/``.

Run from this directory:
    PYTHONPATH=../../.. python3 edges.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _survey import (  # noqa: E402
    RecalTarget,
    SurveyConfig,
    detect_next_session,
    run_phase,
    setup_hardware,
)


MANIFEST_PATH = Path('artifacts/edge_clipped_recheck.json')


# LO frequencies are overridden per group from the manifest -- the values
# below are placeholders so SurveyConfig validates.  The grid bounds are
# also placeholders because the cell list comes from the manifest, not
# from build_galplane_grid.
BASE_CONFIG = SurveyConfig(
    title='Edge-clipped recheck',
    output_dir='data/edges',
    artifacts_prefix='edges',
    timing_archive_dir='data/archive/edges',

    f1_mhz=1419.86,
    f2_mhz=1421.14,
    sample_rate=3.2e6,
    nsamples=16384,
    nblocks=1025,
    nfft=1024,

    min_alt_deg=17.0,
    max_alt_deg=83.0,
    az_min=7.0,
    az_max=348.0,
    track_interval_s=10.0,

    # Placeholder grid bounds (unused -- cells come from manifest).
    l_center=120.0,
    l_min=-180.0,
    l_max=360.0,
    b_min=-90,
    b_max=90,
    b_step=2,
    physical_spacing_deg=2.0,

    # Same per-pointing dump schedule as the main survey.
    cal_dumps_per_lo=2,
    obs_dumps_per_lo=4,

    recal_enable=True,
    recal_targets=(
        RecalTarget('obs_recal_drift',    180.0, 72.0),
        RecalTarget('obs_recal_drift_bk',  90.0, 72.0),
    ),
    recal_every_n_cells=10,

    phases=('even',),
)


def cells_from_manifest(entries: list[dict]) -> list[tuple]:
    """Convert manifest entries to ``(col_idx, row_idx, l, b)`` tuples
    expected by SurveyScheduler.  Column-major scan order does not apply
    here; |b|-ascending order is preserved from the manifest.
    """
    return [(0, i, e['gl'], e['gb']) for i, e in enumerate(entries)]


def run_group(label: str, lo_pair: tuple, entries: list[dict],
              session_dir: str) -> None:
    cells = cells_from_manifest(entries)
    if not cells:
        print(f'[edges] {label} group empty; skipping.')
        return
    cfg = replace(BASE_CONFIG, f1_mhz=float(lo_pair[0]), f2_mhz=float(lo_pair[1]))
    print(f'\n[edges] {label} group: {len(cells)} cells, '
          f'LO=({cfg.f1_mhz}, {cfg.f2_mhz}) MHz -> {session_dir}')
    telescope, sdrs, noise = setup_hardware(cfg)
    try:
        run_phase('edges', cells, cfg, telescope, sdrs, noise, session_dir)
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


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f'[edges] No manifest at {MANIFEST_PATH}; nothing to do.')
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    cells_all = manifest.get('cells', [])
    if not cells_all:
        print(f'[edges] Manifest at {MANIFEST_PATH} has zero cells; nothing to do.')
        return

    # Manifest is |b|-ascending already; split by suggested_pair.
    lower_cells  = [c for c in cells_all if c['suggested_pair'] == 'lower']
    higher_cells = [c for c in cells_all if c['suggested_pair'] == 'higher']

    print(f'[edges] Loaded {len(cells_all)} cells from {MANIFEST_PATH}')
    print(f'  Lower  pair (LO0, LO1) = {tuple(manifest["lower_pair_mhz"])} MHz: '
          f'{len(lower_cells)} cells')
    print(f'  Higher pair (LO2, LO3) = {tuple(manifest["higher_pair_mhz"])} MHz: '
          f'{len(higher_cells)} cells')

    next_session = detect_next_session(BASE_CONFIG.output_dir)

    def session_dir() -> str:
        nonlocal next_session
        path = f'{BASE_CONFIG.output_dir}/session_{next_session:03d}'
        next_session += 1
        return path

    if lower_cells:
        run_group('lower',  tuple(manifest['lower_pair_mhz']),
                  lower_cells,  session_dir())
    if higher_cells:
        run_group('higher', tuple(manifest['higher_pair_mhz']),
                  higher_cells, session_dir())

    print('\n[edges] All edge-clipped groups complete.')


if __name__ == '__main__':
    main()
