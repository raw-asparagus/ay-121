#!/usr/bin/env python3
"""Lab 4 - Leuschner Sky Survey -- Galactic Plane.

Galactic plane survey with per-pointing noise-diode calibration and
interleaved frequency switching.  Uses the StreamingCapture framework
for continuous three-thread operation (pointing, reading, writing).

Grid: b=[-4, +4], l~[-10, 250] in 2-deg latitude steps.
      Longitude uses non-integer tessellation centered at l=120:
      Delta_l = 2/cos(b) exactly, expanded outward from center.
      At |b| <= 4 the foreshortening is minimal (cos(4) = 0.998),
      so Delta_l ~ 2.004 deg -- effectively integer spacing.

Per pointing the LO sequence is:
    cal-f1, cal-f2, then 4x alternating obs-f1/obs-f2
where f1=1419.86 MHz, f2=1421.14 MHz.

Usage:
    python main.py

Output:
    data/session_{NNN}/<obs|cal>_{l}_{b}/<...>_{timestamp}.npz

Run from this directory:
    PYTHONPATH=../../.. python3 main.py
"""

import threading

import numpy as np

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
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
NBLOCKS      = 1281    # 1280 valid + 1 buffer flush
NFFT         = 1024
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN       =  7.0
AZ_MAX       = 348.0
REPOINT_INTERVAL_SEC = 10.0
OUTPUT_DIR   = 'data'  # relative to script directory

# Grid bounds (galactic plane)
L_CENTER     = 120.0
L_MIN, L_MAX = -10.0, 250.0
B_MIN, B_MAX = -4, 4
B_STEP       = 2
PHYSICAL_SPACING_DEG = 2.0

# Per-pointing dump schedule (plane is bright: T_B ~ 50-150 K)
CAL_DUMPS_PER_LO = 1    # 1 cal dump per LO = 2 cal total
OBS_DUMPS_PER_LO = 3    # 3 obs dumps per LO = 6 obs total
N_LOS = 2
DUMPS_PER_CELL = (CAL_DUMPS_PER_LO + OBS_DUMPS_PER_LO) * N_LOS

NEXT_SESSION = 6


def _next_session_dir() -> str:
    global NEXT_SESSION
    path = f'{OUTPUT_DIR}/session_{NEXT_SESSION:03d}'
    NEXT_SESSION += 1
    return path


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _build_l_row(b_deg):
    """Build non-integer longitude grid at latitude b, centered at L_CENTER.

    Exact spacing: Delta_l = PHYSICAL_SPACING_DEG / cos(b).
    Expands outward from L_CENTER until L_MIN and L_MAX are exceeded.
    Returns sorted list of l values (floats, rounded to 2 decimal places).
    """
    import math
    cos_b = math.cos(math.radians(b_deg))
    if cos_b <= 0:
        return [L_CENTER]
    dl = PHYSICAL_SPACING_DEG / cos_b

    l_vals = [L_CENTER]
    l = L_CENTER + dl
    while l <= L_MAX:
        l_vals.append(round(l, 2))
        l += dl
    l = L_CENTER - dl
    while l >= L_MIN:
        l_vals.append(round(l, 2))
        l -= dl

    return sorted(l_vals)


def build_galplane_grid():
    """Build non-integer tessellation for galactic plane, boustrophedon order.

    Returns list of (row_idx, col_idx, l, b) tuples.
    """
    cells = []
    b_vals = list(range(B_MIN, B_MAX + 1, B_STEP))

    for row_idx, b in enumerate(b_vals):
        l_vals = _build_l_row(b)

        row = [(row_idx, j, l_vals[j], b) for j in range(len(l_vals))]
        if row_idx % 2 == 0:
            row = list(reversed(row))
        cells.extend(row)

        dl = PHYSICAL_SPACING_DEG / np.cos(np.radians(b))
        print(f'  b={b:+3d}: Delta_l={dl:.2f} deg, {len(l_vals)} cells, '
              f'l=[{l_vals[0]:.1f}, {l_vals[-1]:.1f}]')

    return cells


def filter_cells_by_az_side(cells):
    """Keep cells on the az side with more currently accessible cells."""
    classified = []
    n_permanent = 0

    for row, col, l, b in cells:
        alt, az, ra, dec, _ = compute_gal_pointing(
            l, b, lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )
        max_alt = 90.0 - abs(LEO_LAT_DEG - dec)
        if max_alt < MIN_ALT_DEG:
            n_permanent += 1
            continue

        in_limits = MIN_ALT_DEG <= alt <= MAX_ALT_DEG
        is_rising = (AZ_MIN <= az <= 180) or az > AZ_MAX or az < AZ_MIN
        is_setting = 180 < az <= AZ_MAX
        classified.append((row, col, l, b, alt, az, in_limits, is_rising, is_setting))

    if n_permanent:
        print(f'  Dropped {n_permanent} permanently inaccessible cells')

    n_rising = sum(1 for c in classified if c[6] and c[7])
    n_setting = sum(1 for c in classified if c[6] and c[8])

    if n_rising >= n_setting:
        side = 'rising'
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if rising]
    else:
        side = 'setting'
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if setting]

    n_now = n_rising if side == 'rising' else n_setting
    print(f'  Az side: {side} ({n_rising} rising / {n_setting} setting)')
    print(f'  {len(kept)} cells kept ({n_now} currently accessible)')
    return kept


def filter_cells_by_existing_data(cells):
    """Skip cells that already have enough obs dumps across all sessions.

    Scans OUTPUT_DIR/session_*/obs_{l_name}_{b}/ for .npz files.
    A cell is complete when it has >= OBS_DUMPS obs files total.
    """
    from pathlib import Path
    import glob

    counts = {}  # cell_lb -> {'obs': n, 'cal': n}
    for npz in glob.glob(f'{OUTPUT_DIR}/session_*/*_*/*.npz'):
        cell_dir = Path(npz).parent.name
        if cell_dir.startswith('obs_'):
            lb = cell_dir[4:]  # strip 'obs_'
            counts.setdefault(lb, {'obs': 0, 'cal': 0})['obs'] += 1
        elif cell_dir.startswith('cal_'):
            lb = cell_dir[4:]  # strip 'cal_'
            counts.setdefault(lb, {'obs': 0, 'cal': 0})['cal'] += 1

    kept = []
    n_skipped = 0
    for row, col, l, b in cells:
        l_name = f'{l:.2f}'.replace('.', 'p')
        cell_lb = f'{l_name}_{b}'
        c = counts.get(cell_lb, {'obs': 0, 'cal': 0})
        if (c['obs'] >= OBS_DUMPS_PER_LO * N_LOS
                and c['cal'] >= CAL_DUMPS_PER_LO * N_LOS):
            n_skipped += 1
        else:
            kept.append((row, col, l, b))

    if n_skipped:
        print(f'  Existing data: {n_skipped} complete cells skipped, '
              f'{len(kept)} remaining')
    else:
        print(f'  No existing data found -- keeping all {len(kept)} cells')

    return kept


# ---------------------------------------------------------------------------
# Target selector + dump notifier
# ---------------------------------------------------------------------------

def make_scan_target_selector(cells, dumps_per_cell):
    """Create target selector, dump notifier, and done_event.

    Cells that are out of alt/az limits when reached are skipped (not
    stalled on).  After one pass through all cells, any skipped cells
    are retried so that cells rising into view during a long run still
    get observed.  If a full retry pass completes with zero successful
    observations, the remaining cells are abandoned and done_event is set.

    ``dump_notifier`` should be passed as ``on_read`` to
    ``StreamingCapture`` so that counting happens in the reader thread.
    """
    cell_list = list(cells)
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    done_event = threading.Event()
    skipped = []
    cells_observed_this_pass = 0

    _, _, cl, cb = cell_list[0]
    print(f'  [scan] {len(cell_list)} cells, first: l={cl} b={cb}')

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

    def dump_notifier(_dump):
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, cells_observed_this_pass

        if done_event.is_set():
            return None

        if _check_end_of_list():
            return None

        # Cell complete -- advance immediately and return None to
        # pause the reader during the slew to the new target.
        with lock:
            count = cell_dump_count
            if count >= dumps_per_cell:
                cells_observed_this_pass += 1
                current_cell_idx += 1
                cell_dump_count = 0
                if _check_end_of_list():
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
                if _check_end_of_list():
                    return None
            return None

        l_name = f'{cell_l:.2f}'.replace('.', 'p')
        return f'obs_{l_name}_{cell_b}', alt, az, ra, dec

    return target_selector, dump_notifier, done_event


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


def main():
    print('Leuschner Sky Survey -- Galactic Plane')
    print('=' * 60)
    print(f'  Grid: l~[{L_MIN}, {L_MAX}], b=[{B_MIN}, {B_MAX}], b_step={B_STEP} deg')
    print(f'  Longitude: non-integer, centered at l={L_CENTER}, '
          f'Delta_l = {PHYSICAL_SPACING_DEG}/cos(b)')
    print(f'  LO: f1={F1_MHZ} MHz, f2={F2_MHZ} MHz')
    print(f'  Sample rate: {SAMPLE_RATE/1e6} MHz, NFFT={NFFT}, NBLOCKS={NBLOCKS}')
    print(f'  Per pointing: {CAL_DUMPS_PER_LO * N_LOS} cal + '
          f'{OBS_DUMPS_PER_LO * N_LOS} obs = {DUMPS_PER_CELL} dumps')
    print(f'  Track interval: {REPOINT_INTERVAL_SEC} s')

    all_cells = build_galplane_grid()
    print(f'\n  Total grid cells: {len(all_cells)}')

    cells = filter_cells_by_az_side(all_cells)
    cells = filter_cells_by_existing_data(cells)
    if not cells:
        print('  No remaining cells. Exiting.')
        return

    dump_time = NBLOCKS * NSAMPLES / SAMPLE_RATE
    cell_time = DUMPS_PER_CELL * (dump_time + 2) + 3
    total_time_h = len(cells) * cell_time / 3600
    print(f'\n  Integration per dump: {dump_time:.1f} s')
    print(f'  Estimated total: {total_time_h:.1f} h for {len(cells)} cells')

    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.')

    target_selector, dump_notifier, done_event = \
        make_scan_target_selector(cells, DUMPS_PER_CELL)

    read_fn = make_calibrated_sdr_reader(
        sdrs, noise,
        nsamples=NSAMPLES, nblocks=NBLOCKS, nfft=NFFT,
        lo_freqs_mhz=(F1_MHZ, F2_MHZ),
        cal_dumps_per_lo=CAL_DUMPS_PER_LO,
        obs_dumps_per_lo=OBS_DUMPS_PER_LO,
    )

    def on_save(path, dump):
        lo_tag = f'LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
        cal_tag = 'CAL' if dump.get('noise_on') else 'OBS'
        print(f'  [{dump["target_name"]}] {cal_tag} {lo_tag} -> {path}')

    outdir = _next_session_dir()
    print(f'  Session dir: {outdir}')

    capture = StreamingCapture(
        telescope=telescope,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=outdir,
        n_writers=2,
        repoint_interval_sec=REPOINT_INTERVAL_SEC,
        on_save=on_save,
        on_read=dump_notifier,
    )

    capture.run(done_event=done_event)

    noise.off()
    print('\n' + '=' * 60)
    print('  Galactic plane survey complete!')
    print('=' * 60)


if __name__ == '__main__':
    while True:
        main()
