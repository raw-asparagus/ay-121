#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI OTF raster scan (DR3).

Simple raster in galactic coordinates at 2° spacing.
Cells are filtered to one side of the az exclusion zone (rising or
setting) to prevent the telescope from crossing the north gap.

Two parts run sequentially:
  Part 1: Orion-Eridanus / wide plane intersection l=160–220, b=-20 to -10
  Part 2: Galactic plane l=120–180, b=-4 to +4

Usage:
    python observe_otf.py

Output:
    data/lab04/streaming/DR3/scan_r<row>_c<col>/...npz
"""

import threading

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_calibrated_sdr_reader

# ---------------------------------------------------------------------------
# Survey parts (run sequentially)
# ---------------------------------------------------------------------------

SURVEY_PARTS = [
    {   # Phase 1: Finish Orion-Eri (setting, 6-9:30 PM)
        'name': 'Phase 1: Orion-Eridanus (setting)',
        'l_min': 160, 'l_max': 220,
        'b_min': -20, 'b_max': -10,
        'step': 2,
        'dumps_per_band': 4,
    },
    {   # Phase 2: Plane l=130-180 (setting, 9:30 PM-1 AM)
        'name': 'Phase 2: Gal. plane l=130-180 (setting)',
        'l_min': 130, 'l_max': 180,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
    },
    {   # Phase 3: Plane l=120-160 (rising, 1-6 AM) --- includes canonical (120,0)
        'name': 'Phase 3: Gal. plane l=120-160 (rising)',
        'l_min': 120, 'l_max': 160,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
    },
    {   # Phase 4: Plane l=60-120 (rising, 6-8 AM)
        'name': 'Phase 4: Gal. plane l=60-120 (rising)',
        'l_min': 60, 'l_max': 120,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
    },
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LO_ON_MHZ   = 1420.0
LO_OFF_MHZ  = 1421.0
SAMPLE_RATE  = 2.56e6
NSAMPLES     = 32768
NBLOCKS      = 1025
NFFT         = 1024
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN       =  7.0
AZ_MAX       = 348.0
CAL_DUMPS    = 2
REPOINT_TRACK_SEC = 60.0
OUTDIR       = 'data/lab04/streaming/DR3'
MANIFEST_PATH = 'survey_manifest.json'  # relative to labs/04/


# ---------------------------------------------------------------------------
# Grid builder with az-side filtering
# ---------------------------------------------------------------------------

def build_raster_cells(l_min, l_max, b_min, b_max, step):
    """Build a raster grid in galactic coordinates.

    Boustrophedon ordering: even rows scan in decreasing l.
    Returns list of (row_idx, col_idx, l, b) tuples.
    """
    b_vals = list(range(b_min, b_max + 1, step))
    l_vals = list(range(l_min, l_max + 1, step))

    cells = []
    for row_idx, b in enumerate(b_vals):
        row = [(row_idx, j, l_vals[j], b) for j in range(len(l_vals))]
        if row_idx % 2 == 0:
            row = list(reversed(row))  # decreasing l
        cells.extend(row)

    return cells


def filter_cells_by_az_side(cells):
    """Filter cells to one side of the az exclusion zone.

    Computes current alt/az for each cell, classifies as rising (az 7-180)
    or setting (az 180-348), and keeps only the side with more accessible
    cells. Also removes cells outside alt limits.
    """
    classified = []
    for row, col, l, b in cells:
        alt, az, ra, dec, _ = compute_gal_pointing(
            l, b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )
        in_limits = MIN_ALT_DEG <= alt <= MAX_ALT_DEG
        is_rising = AZ_MIN <= az <= 180
        is_setting = 180 < az <= AZ_MAX
        classified.append((row, col, l, b, alt, az, in_limits, is_rising, is_setting))

    n_rising = sum(1 for c in classified if c[6] and c[7])
    n_setting = sum(1 for c in classified if c[6] and c[8])

    if n_rising >= n_setting:
        side = 'rising'
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if ok and rising]
    else:
        side = 'setting'
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if ok and setting]

    print(f'  Az filter: {side} ({n_rising} rising / {n_setting} setting)')
    print(f'  {len(kept)} cells kept, {len(cells) - len(kept)} dropped')

    return kept


def filter_cells_by_manifest(cells):
    """Remove cells that are already marked complete in the survey manifest."""
    import sys
    from pathlib import Path

    # Try to find manifest relative to the script location
    script_dir = Path(__file__).resolve().parent.parent  # labs/04/
    manifest_path = script_dir / MANIFEST_PATH

    if not manifest_path.exists():
        print(f'  Manifest not found at {manifest_path} --- skipping manifest filter')
        return cells

    sys.path.insert(0, str(script_dir))
    from manifest import get_complete_cells

    complete = get_complete_cells(manifest_path)
    if not complete:
        print(f'  Manifest has no complete cells --- keeping all')
        return cells

    kept = [(r, c, l, b) for r, c, l, b in cells if (l, b) not in complete]
    n_skipped = len(cells) - len(kept)
    print(f'  Manifest filter: {n_skipped} complete cells skipped, {len(kept)} remaining')

    return kept


# ---------------------------------------------------------------------------
# Target selector factory
# ---------------------------------------------------------------------------

def make_scan_target_selector(cells, dumps_per_cell):
    """Create target selector, dump notifier, and done_event."""
    total_cells = len(cells)
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    transitioning = False
    done_event = threading.Event()

    _, _, cl, cb = cells[0]
    print(f'  [scan] {total_cells} cells')
    print(f'  [scan] First: l={cl}, b={cb}')
    _, _, cl, cb = cells[-1]
    print(f'  [scan] Last:  l={cl}, b={cb}')

    def dump_notifier():
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning

        if current_cell_idx >= total_cells:
            done_event.set()
            return None

        with lock:
            count = cell_dump_count
            if count >= dumps_per_cell and not transitioning:
                transitioning = True
                return None
            if transitioning:
                transitioning = False
                current_cell_idx += 1
                cell_dump_count = 0
                if current_cell_idx >= total_cells:
                    print('  [scan] All cells complete.')
                    done_event.set()
                    return None
                _, _, cl, cb = cells[current_cell_idx]
                print(f'  [scan] Cell {current_cell_idx+1}/{total_cells}: '
                      f'l={cl}, b={cb}')

        _, _, cell_l, cell_b = cells[current_cell_idx]
        alt, az, ra, dec, _ = compute_gal_pointing(
            cell_l, cell_b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )

        if alt < MIN_ALT_DEG or alt > MAX_ALT_DEG or az < AZ_MIN or az > AZ_MAX:
            return None

        row, col = cells[current_cell_idx][0], cells[current_cell_idx][1]
        return f'scan_r{row}_c{col}', alt, az, ra, dec

    return target_selector, dump_notifier, done_event


# ---------------------------------------------------------------------------

def setup_hardware():
    from ugradio.leusch import LeuschNoise, LeuschTelescope
    from ugradio.sdr import SDR

    telescope = LeuschTelescope()
    noise = LeuschNoise()
    sdr_0 = SDR(device_index=0, direct=False,
                center_freq=LO_ON_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
    sdr_1 = SDR(device_index=1, direct=False,
                center_freq=LO_ON_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
    return telescope, [sdr_0, sdr_1], noise


def main():
    print('Lab 4 - Leuschner 21 cm HI OTF raster scan (DR3)')
    print('=' * 60)

    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.')

    for part_idx, part in enumerate(SURVEY_PARTS):
        l_min, l_max = part['l_min'], part['l_max']
        b_min, b_max = part['b_min'], part['b_max']
        step = part['step']
        dumps_per_band = part['dumps_per_band']
        dumps_per_cell = dumps_per_band * 2

        print(f'\n{"="*60}')
        print(f'  {part["name"]}')
        print(f'  l=[{l_min}, {l_max}], b=[{b_min}, {b_max}], step={step}°')
        print(f'{"="*60}')

        # Build grid, filter by az side, then skip complete cells
        all_cells = build_raster_cells(l_min, l_max, b_min, b_max, step)
        cells = filter_cells_by_az_side(all_cells)
        cells = filter_cells_by_manifest(cells)

        if not cells:
            print('  No remaining cells --- skipping.')
            continue

        dump_cadence = 20.0  # conservative estimate with pipelined reader (measured: 17s)
        cell_time = dumps_per_cell * dump_cadence + 5
        pass_time = len(cells) * cell_time / 3600
        print(f'  Dumps per cell: {dumps_per_cell} ({dumps_per_band}/band)')
        print(f'  Estimated: {pass_time:.1f} h')

        target_selector, dump_notifier, done_event = \
            make_scan_target_selector(cells, dumps_per_cell)

        read_fn = make_calibrated_sdr_reader(
            sdrs, noise,
            nsamples=NSAMPLES, nblocks=NBLOCKS, nfft=NFFT,
            lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
            cal_dumps_per_lo=CAL_DUMPS,
        )

        def on_save(path, dump, _notifier=dump_notifier):
            _notifier()
            noise_tag = ' [CAL]' if dump.get('noise_on') else ''
            lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
            print(f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
                  f'{lo_tag}{noise_tag}  -> {path}')

        capture = StreamingCapture(
            telescope=telescope,
            read_fn=read_fn,
            target_selector=target_selector,
            outdir=OUTDIR,
            n_writers=2,
            repoint_interval_sec=REPOINT_TRACK_SEC,
            on_save=on_save,
        )
        capture.run(done_event=done_event)

        print(f'  {part["name"]} complete.')

    print('\n' + '=' * 60)
    print('  All survey parts complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
