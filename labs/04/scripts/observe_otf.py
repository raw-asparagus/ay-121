#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI OTF raster scan (DR9: 8-phase galactic plane).

Simple raster in galactic coordinates at 2-deg spacing.
Cells are filtered to one side of the az exclusion zone (rising or
setting) to prevent the telescope from crossing the north gap.
Below-horizon cells on the chosen side are included and retried as
they rise into view during long runs.

DR9 8-phase survey (2026-04-22+):

  Even-Even Plane Fills (DR9a) — 4 phases:
    Phase 1: l=-10 to 120, rising (inner galaxy)
    Phase 2: l=-10 to 120, setting
    Phase 3: l=120 to 260, rising (outer galaxy)
    Phase 4: l=120 to 260, setting
    ~120 cells total, ~2h combined

  Odd-Odd Galactic Plane (DR9b) — 4 phases:
    Phase 5: l=-9 to 119, rising (inner galaxy)
    Phase 6: l=-9 to 119, setting
    Phase 7: l=121 to 259, rising (outer galaxy)
    Phase 8: l=121 to 259, setting
    ~540 cells total, ~6h combined (after DR8b manifest update)

Usage:
    python observe_otf.py

Output:
    data/lab04/streaming/DR9a/scan_r<row>_c<col>/...npz  (phases 1-4, even-even)
    data/lab04/streaming/DR9b/scan_r<row>_c<col>/...npz  (phases 5-8, odd-odd)
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
    {   # Phase 1: Even-even inner rising
        'name': 'Even-even plane fills — inner rising (DR9a-1)',
        'l_min': -10, 'l_max': 120,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'outdir': 'data/lab04/streaming/DR9a',
    },
    # {   # Phase 2: Even-even inner setting
    #     'name': 'Even-even plane fills — inner setting (DR9a-2)',
    #     'l_min': -10, 'l_max': 120,
    #     'b_min': -4, 'b_max': 4,
    #     'step': 2,
    #     'dumps_per_band': 4,
    #     'outdir': 'data/lab04/streaming/DR9a',
    # },
    {   # Phase 3: Even-even outer rising
        'name': 'Even-even plane fills — outer rising (DR9a-3)',
        'l_min': 120, 'l_max': 260,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'outdir': 'data/lab04/streaming/DR9a',
    },
    # {   # Phase 4: Even-even outer setting
    #     'name': 'Even-even plane fills — outer setting (DR9a-4)',
    #     'l_min': 120, 'l_max': 260,
    #     'b_min': -4, 'b_max': 4,
    #     'step': 2,
    #     'dumps_per_band': 4,
    #     'outdir': 'data/lab04/streaming/DR9a',
    # },
    {   # Phase 5: Odd-odd inner rising
        'name': 'Odd-odd galactic plane — inner rising (DR9b-1)',
        'l_min': -9, 'l_max': 119,
        'b_min': -3, 'b_max': 3,
        'step': 2,
        'dumps_per_band': 4,
        'outdir': 'data/lab04/streaming/DR9b',
    },
    # {   # Phase 6: Odd-odd inner setting
    #     'name': 'Odd-odd galactic plane — inner setting (DR9b-2)',
    #     'l_min': -9, 'l_max': 119,
    #     'b_min': -3, 'b_max': 3,
    #     'step': 2,
    #     'dumps_per_band': 4,
    #     'outdir': 'data/lab04/streaming/DR9b',
    # },
    {   # Phase 7: Odd-odd outer rising
        'name': 'Odd-odd galactic plane — outer rising (DR9b-3)',
        'l_min': 121, 'l_max': 259,
        'b_min': -3, 'b_max': 3,
        'step': 2,
        'dumps_per_band': 4,
        'outdir': 'data/lab04/streaming/DR9b',
    },
    # {   # Phase 8: Odd-odd outer setting
    #     'name': 'Odd-odd galactic plane — outer setting (DR9b-4)',
    #     'l_min': 121, 'l_max': 259,
    #     'b_min': -3, 'b_max': 3,
    #     'step': 2,
    #     'dumps_per_band': 4,
    #     'outdir': 'data/lab04/streaming/DR9b',
    # },
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
OUTDIR       = 'data/lab04/streaming/DR9a'  # default; overridden per part
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

    First removes permanently inaccessible cells: those whose upper transit
    altitude (90 - |lat - dec|) is below MIN_ALT_DEG and can never be reached.

    Then classifies remaining cells as rising (az 7-180) or setting (az 180-348).
    The side with more *currently accessible* cells is chosen, but all cells on
    that side are kept -- including those temporarily below the horizon -- so
    they can be retried later during a long run.
    """
    classified = []
    n_permanent = 0
    for row, col, l, b in cells:
        alt, az, ra, dec, _ = compute_gal_pointing(
            l, b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )
        # Drop cells that never reach MIN_ALT_DEG from this latitude.
        # Upper transit altitude = 90 - |lat - dec|.
        max_alt = 90.0 - abs(LEO_LAT_DEG - dec)
        if max_alt < MIN_ALT_DEG:
            n_permanent += 1
            continue

        in_limits = MIN_ALT_DEG <= alt <= MAX_ALT_DEG
        # Cells in the az exclusion zone (348-360, 0-7) are transiting
        # near north and will enter the rising side next.
        is_rising = (AZ_MIN <= az <= 180) or az > AZ_MAX or az < AZ_MIN
        is_setting = 180 < az <= AZ_MAX
        classified.append((row, col, l, b, alt, az, in_limits, is_rising, is_setting))

    if n_permanent:
        print(f'  Permanent inaccessibility: {n_permanent} cells dropped '
              f'(never reach {MIN_ALT_DEG}deg from lat={LEO_LAT_DEG}deg)')

    # Pick side based on cells *currently* in limits
    n_rising = sum(1 for c in classified if c[6] and c[7])
    n_setting = sum(1 for c in classified if c[6] and c[8])

    # Keep ALL cells on the chosen side (including temporarily below-horizon ones)
    if n_rising >= n_setting:
        side = 'rising'
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if rising]
    else:
        side = 'setting'
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if setting]

    n_now = n_rising if side == 'rising' else n_setting
    print(f'  Az filter: {side} ({n_rising} rising / {n_setting} setting)')
    print(f'  {len(kept)} cells kept ({n_now} currently accessible),'
          f' {len(cells) - len(kept) - n_permanent} dropped (wrong az side)')

    return kept


def filter_cells_by_manifest(cells, fills_only=False):
    """Remove cells based on the survey manifest.

    If fills_only is False (default): remove cells marked complete.
    If fills_only is True: keep ONLY cells that are in the manifest
    and incomplete (i.e., only fill gaps in existing data).
    """
    import sys
    from pathlib import Path

    # Try to find manifest relative to the script location
    script_dir = Path(__file__).resolve().parent.parent  # labs/04/
    manifest_path = script_dir / MANIFEST_PATH

    if not manifest_path.exists():
        print(f'  Manifest not found at {manifest_path} --- skipping manifest filter')
        return cells

    sys.path.insert(0, str(script_dir))
    from manifest import get_complete_cells, get_incomplete_cells

    if fills_only:
        incomplete = get_incomplete_cells(manifest_path)
        kept = [(r, c, l, b) for r, c, l, b in cells if (l, b) in incomplete]
        n_skipped = len(cells) - len(kept)
        print(f'  Manifest filter (fills only): {len(kept)} incomplete cells kept, '
              f'{n_skipped} skipped')
    else:
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
    """Create target selector, dump notifier, and done_event.

    Cells that are out of alt/az limits when reached are *skipped* (not
    stalled on).  After one pass through all cells, any skipped cells
    are retried so that cells rising into view during a long run still
    get observed.  If a full retry pass completes with zero successful
    observations, the remaining cells are abandoned (they never became
    accessible) and the done_event is set.
    """
    cell_list = list(cells)  # mutable working copy
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    transitioning = False
    done_event = threading.Event()
    skipped = []
    cells_observed_this_pass = 0  # non-zero means progress was made

    _, _, cl, cb = cell_list[0]
    print(f'  [scan] {len(cell_list)} cells')
    print(f'  [scan] First: l={cl}, b={cb}')
    _, _, cl, cb = cell_list[-1]
    print(f'  [scan] Last:  l={cl}, b={cb}')

    def _start_retry_pass():
        """Swap skipped cells back into the work list for another pass.

        Returns False if no progress was made last pass (all cells were
        skipped), signalling that the remaining cells should be abandoned.
        """
        nonlocal current_cell_idx, cells_observed_this_pass
        if cells_observed_this_pass == 0:
            print(f'  [scan] No progress in last pass - abandoning '
                  f'{len(skipped)} unreachable cell(s).')
            return False
        cell_list[:] = list(skipped)
        skipped.clear()
        current_cell_idx = 0
        cells_observed_this_pass = 0
        print(f'  [scan] Retry pass: {len(cell_list)} cells')
        return True

    def dump_notifier():
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning, cells_observed_this_pass

        if current_cell_idx >= len(cell_list):
            if skipped:
                if not _start_retry_pass():
                    done_event.set()
                    return None
            else:
                print('  [scan] All cells complete.')
                done_event.set()
                return None

        with lock:
            count = cell_dump_count
            if count >= dumps_per_cell and not transitioning:
                transitioning = True
                return None
            if transitioning:
                transitioning = False
                cells_observed_this_pass += 1
                current_cell_idx += 1
                cell_dump_count = 0
                if current_cell_idx >= len(cell_list):
                    if skipped:
                        if not _start_retry_pass():
                            done_event.set()
                            return None
                    else:
                        print('  [scan] All cells complete.')
                        done_event.set()
                        return None
                _, _, cl, cb = cell_list[current_cell_idx]
                print(f'  [scan] Cell {current_cell_idx+1}/{len(cell_list)}: '
                      f'l={cl}, b={cb}')

        _, _, cell_l, cell_b = cell_list[current_cell_idx]
        alt, az, ra, dec, _ = compute_gal_pointing(
            cell_l, cell_b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )

        if alt < MIN_ALT_DEG or alt > MAX_ALT_DEG or az < AZ_MIN or az > AZ_MAX:
            with lock:
                skipped.append(cell_list[current_cell_idx])
                current_cell_idx += 1
                cell_dump_count = 0
                if current_cell_idx >= len(cell_list):
                    if skipped:
                        if not _start_retry_pass():
                            done_event.set()
                            return None
                    else:
                        print('  [scan] All cells complete.')
                        done_event.set()
                        return None
            return None

        row, col = cell_list[current_cell_idx][0], cell_list[current_cell_idx][1]
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
    print('Lab 4 - Leuschner 21 cm HI OTF raster scan (DR8a/DR8b)')
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
        print(f'  l=[{l_min}, {l_max}], b=[{b_min}, {b_max}], step={step}deg')
        print(f'{"="*60}')

        # Build grid, filter by az side, then skip complete cells
        all_cells = build_raster_cells(l_min, l_max, b_min, b_max, step)
        cells = filter_cells_by_az_side(all_cells)
        cells = filter_cells_by_manifest(cells, fills_only=part.get('fills_only', False))

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

        part_outdir = part.get('outdir', OUTDIR)
        capture = StreamingCapture(
            telescope=telescope,
            read_fn=read_fn,
            target_selector=target_selector,
            outdir=part_outdir,
            n_writers=2,
            repoint_interval_sec=REPOINT_TRACK_SEC,
            on_save=on_save,
        )

        max_hours = part.get('max_hours')
        if max_hours:
            timer = threading.Timer(max_hours * 3600, done_event.set)
            timer.daemon = True
            timer.start()
            print(f'  Time limit: {max_hours:.1f} h')

        capture.run(done_event=done_event)

        if max_hours:
            timer.cancel()

        print(f'  {part["name"]} complete.')

    print('\n' + '=' * 60)
    print('  All survey parts complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
