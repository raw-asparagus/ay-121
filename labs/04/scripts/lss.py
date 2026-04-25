#!/usr/bin/env python3
"""Lab 4 - Leuschner Sky Survey (LSS).

Galactic plane survey with per-pointing noise-diode calibration and
interleaved frequency switching.  Uses the StreamingCapture framework
for continuous three-thread operation (pointing, reading, writing).

Grid: b=[-4, +4], l=[-10, 250] in 2-deg steps on a checkerboard
      ((even,even) + (odd,odd) coordinates).

Per pointing the LO sequence is:
    cal-f1, cal-f2, obs-f1, obs-f2, obs-f1, obs-f2, obs-f1, obs-f2
where f1=1419.86 MHz, f2=1421.14 MHz.

Usage:
    python lss.py

Output:
    data/lab04/galacticplane/session_{NNN}/<obs|cal>_{l}_{b}/<...>_{timestamp}.npz
"""

import threading
import time

import numpy as np

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import _PipelinedSDR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

F1_MHZ       = 1419.86
F2_MHZ       = 1421.14
SAMPLE_RATE  = 3.2e6
NSAMPLES     = 98304   # 64 chunks * 1536 channels
NBLOCKS      = 513     # 512 valid + 1 buffer flush
NFFT         = 1536
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN       =  7.0
AZ_MAX       = 348.0
REPOINT_INTERVAL_SEC = 10.0
OUTPUT_DIR   = 'data/lab04/galacticplane'

# Grid bounds
L_MIN, L_MAX = -10, 250
B_MIN, B_MAX = -4, 4
STEP         = 2

# Per-pointing dump schedule
CAL_DUMPS  = 2   # cal-f1, cal-f2
OBS_DUMPS  = 8   # (obs-f1, obs-f2) x 4
DUMPS_PER_CELL = CAL_DUMPS + OBS_DUMPS

NEXT_SESSION = 1


def _next_session_dir() -> str:
    global NEXT_SESSION
    path = f'{OUTPUT_DIR}/session_{NEXT_SESSION:03d}'
    NEXT_SESSION += 1
    return path


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_checkerboard_grid():
    """Checkerboard raster: (even,even) + (odd,odd), boustrophedon order."""
    cells = []
    b_vals = list(range(B_MIN, B_MAX + 1, STEP))

    for row_idx, b in enumerate(b_vals):
        if b % 2 == 0:
            l_start = L_MIN if L_MIN % 2 == 0 else L_MIN + 1
        else:
            l_start = L_MIN if L_MIN % 2 != 0 else L_MIN + 1
        l_vals = list(range(l_start, L_MAX + 1, STEP))

        row = [(row_idx, j, l_vals[j], b) for j in range(len(l_vals))]
        if row_idx % 2 == 0:
            row = list(reversed(row))
        cells.extend(row)

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


# ---------------------------------------------------------------------------
# Target selector + dump notifier
# ---------------------------------------------------------------------------

def make_scan_target_selector(cells):
    """Create target_selector, dump_notifier, done_event, and cell_event.

    cell_event is set whenever the target advances to a new cell, so the
    reader can reset its per-cell LO/cal schedule.
    """
    cell_list = list(cells)
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    transitioning = False
    done_event = threading.Event()
    cell_event = threading.Event()  # fires on each new cell
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

    def dump_notifier():
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning, cells_observed_this_pass

        if done_event.is_set():
            return None

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
            if count >= DUMPS_PER_CELL and not transitioning:
                transitioning = True
                return None
            if transitioning:
                transitioning = False
                cells_observed_this_pass += 1
                current_cell_idx += 1
                cell_dump_count = 0
                cell_event.set()  # signal new cell to reader
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
                cell_event.set()
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

        return f'obs_{cell_l}_{cell_b}', alt, az, ra, dec

    return target_selector, dump_notifier, done_event, cell_event


# ---------------------------------------------------------------------------
# Reader with per-cell cal + interleaved LO
# ---------------------------------------------------------------------------

def make_lss_reader(sdrs, noise, cell_event, nsamples=NSAMPLES,
                    nblocks=NBLOCKS, nfft=NFFT):  # noqa: E501
    """Create a reader that runs per-cell cal-f1, cal-f2, then obs cycles.

    On each new cell (signalled by cell_event), the schedule resets to:
        cal-f1, cal-f2, obs-f1, obs-f2, obs-f1, obs-f2, obs-f1, obs-f2

    Uses pipelined capture (FFT of previous dump overlaps USB transfer
    of current dump).
    """
    # Per-cell schedule: (lo_mhz, noise_on)
    CELL_SCHEDULE = [
        (F1_MHZ, True),   # cal f1
        (F2_MHZ, True),   # cal f2
        (F1_MHZ, False),  # obs f1
        (F2_MHZ, False),  # obs f2
        (F1_MHZ, False),  # obs f1
        (F2_MHZ, False),  # obs f2
        (F1_MHZ, False),  # obs f1
        (F2_MHZ, False),  # obs f2
    ]

    pipeline = _PipelinedSDR(sdrs, nsamples, nblocks, nfft)

    schedule_idx = 0
    noise_on_log = []  # tracks noise_on for each submitted capture
    call_count = 0
    submit_count = 0
    current_noise_state = None

    def _set_noise(on):
        nonlocal current_noise_state
        if on != current_noise_state:
            if on:
                noise.on()
            else:
                noise.off()
            current_noise_state = on

    def read(prev_cnt):
        nonlocal schedule_idx, call_count, submit_count

        # Check if we advanced to a new cell
        if cell_event.is_set():
            cell_event.clear()
            schedule_idx = 0

        # Determine LO and noise state for this submission
        lo, is_cal = CELL_SCHEDULE[schedule_idx % len(CELL_SCHEDULE)]
        _set_noise(is_cal)

        # Submit capture, get back previous dump
        dump = pipeline.next_dump(lo)
        noise_on_log.append(is_cal)
        submit_count += 1
        schedule_idx += 1

        if dump is None:
            # First call -- no previous dump. Submit one more.
            lo2, is_cal2 = CELL_SCHEDULE[schedule_idx % len(CELL_SCHEDULE)]
            _set_noise(is_cal2)
            dump = pipeline.next_dump(lo2)
            noise_on_log.append(is_cal2)
            submit_count += 1
            schedule_idx += 1

        if dump is not None:
            dump['noise_on'] = noise_on_log[call_count]
            call_count += 1

        return dump

    return read


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
    print('Leuschner Sky Survey (LSS)')
    print('=' * 60)
    print(f'  Grid: l=[{L_MIN}, {L_MAX}], b=[{B_MIN}, {B_MAX}], step={STEP} deg')
    print(f'  Checkerboard: (even,even) + (odd,odd)')
    print(f'  LO: f1={F1_MHZ} MHz, f2={F2_MHZ} MHz')
    print(f'  Sample rate: {SAMPLE_RATE/1e6} MHz, NFFT={NFFT}, NBLOCKS={NBLOCKS}')
    print(f'  Per pointing: {CAL_DUMPS} cal + {OBS_DUMPS} obs = {DUMPS_PER_CELL} dumps')
    print(f'  Track interval: {REPOINT_INTERVAL_SEC} s')

    all_cells = build_checkerboard_grid()
    print(f'\n  Total grid cells: {len(all_cells)}')

    cells = filter_cells_by_az_side(all_cells)
    if not cells:
        print('  No accessible cells. Exiting.')
        return

    dump_time = NBLOCKS * NSAMPLES / SAMPLE_RATE
    cell_time = DUMPS_PER_CELL * (dump_time + 2) + 3  # +2s LO settle, +3s slew
    total_time_h = len(cells) * cell_time / 3600
    print(f'\n  Integration per dump: {dump_time:.1f} s')
    print(f'  Estimated total: {total_time_h:.1f} h for {len(cells)} cells')

    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.')

    target_selector, dump_notifier, done_event, cell_event = \
        make_scan_target_selector(cells)

    read_fn = make_lss_reader(sdrs, noise, cell_event)

    def on_save(path, dump, _notifier=dump_notifier):
        _notifier()
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
    )

    capture.run(done_event=done_event)

    noise.off()
    print('\n' + '=' * 60)
    print('  LSS complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
