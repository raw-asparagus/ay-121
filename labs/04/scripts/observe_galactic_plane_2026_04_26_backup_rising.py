#!/usr/bin/env python3
"""BACKUP: GALACTIC PLANE SURVEY - RISING-SIDE ONLY (4 PHASES)
Leuschner Observatory, April 26, 2026

========================================================================
If the April 25/26 marathon fails or weather prevents completion,
use this backup plan: Focus on rising-side observations only.
This captures the same regions as the marathon rising phases.
========================================================================

SCHEDULE (PDT):
  22:00 Fri Apr 25 to 10:30 Sat Apr 26  (if April 25 abort)
  OR
  20:00 Sat Apr 26 to 06:00 Sun Apr 27  (standalone evening/morning)

PHASES (4 rising-side only):
  22:00–23:30  Phase 1: Even-even INNER RISING (1.5 h, 38 cells)
  23:30–01:40  Phase 2: Even-even OUTER RISING (2.2 h, 40–50 cells)
  01:40–06:20  Phase 3: Odd-odd INNER RISING (4.7 h, 115 cells)
  06:20–10:30  Phase 4: Odd-odd OUTER RISING (4.2 h, 109 cells)

TOTAL:  10.5 hours  |  ~300 cells  |  ~300 MB

COVERAGE:
  [OK] Even-even l in [-10°, 260°], b in [-4°, 4°]
  [OK] Odd-odd l in [-9°, 259°], b in [-3°, 3°]
  [OK] Priority region l in [10°, 240°] COMPLETE
  [OK] Extension to l = 260° COMPLETE
  [NO] Setting-side observations DEFERRED

MISSING (if you use this backup only):
  - Setting-side observations for same regions
  - Use this if marathon had hardware failure mid-way
  - Or use for initial data if weather prevents April 25 evening

USAGE:
    python observe_galactic_plane_2026_04_26_backup_rising.py

========================================================================
"""

import glob
import sys
import threading
import time
import traceback
import signal
from pathlib import Path
from datetime import datetime

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_calibrated_sdr_reader

# =========================================================================
# LOGGING
# =========================================================================

class RobustLogger:
    """Thread-safe logger."""
    def __init__(self, log_path='obs_backup_2026_04_26.log'):
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        with open(self.log_path, 'w') as f:
            f.write(f"=== BACKUP SURVEY START: {datetime.now().isoformat()} ===\n\n")

    def log(self, msg, level='INFO'):
        with self.lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] [{level:7}] {msg}"
            print(log_line)
            with open(self.log_path, 'a') as f:
                f.write(log_line + '\n')

logger = RobustLogger()
abort_event = threading.Event()

def handle_abort(signum, frame):
    logger.log("ABORT SIGNAL - gracefully stopping", 'WARN')
    abort_event.set()

signal.signal(signal.SIGINT, handle_abort)

# =========================================================================
# 4-PHASE RISING-SIDE SURVEY
# =========================================================================

SURVEY_PARTS = [
    {
        'name': 'Even-even INNER RISING',
        'phase_num': 1,
        'l_min': -10, 'l_max': 120,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'expected_cells': 38,
        'expected_duration_h': 1.5,
    },
    {
        'name': 'Even-even OUTER RISING',
        'phase_num': 2,
        'l_min': 120, 'l_max': 260,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'expected_cells': 40,
        'expected_duration_h': 1.6,
    },
    {
        'name': 'Odd-odd INNER RISING',
        'phase_num': 3,
        'l_min': -9, 'l_max': 119,
        'b_min': -3, 'b_max': 3,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'expected_cells': 115,
        'expected_duration_h': 4.7,
    },
    {
        'name': 'Odd-odd OUTER RISING',
        'phase_num': 4,
        'l_min': 121, 'l_max': 259,
        'b_min': -3, 'b_max': 3,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'expected_cells': 109,
        'expected_duration_h': 4.2,
    },
]

# =========================================================================
# HARDWARE & PARAMETERS (same as marathon)
# =========================================================================

LO_ON_MHZ   = 1420.0
LO_OFF_MHZ  = 1421.0
SAMPLE_RATE  = 2.56e6
NSAMPLES     = 32768
NBLOCKS      = 1025
NFFT         = 1024
CAL_DUMPS    = 2
REPOINT_TRACK_SEC = 60.0
STREAMING_DIR = 'data/lab04/streaming'
MANIFEST_PATH = 'survey_manifest.json'


def _next_session_dir() -> str:
    existing = sorted(glob.glob(f'{STREAMING_DIR}/session_???'))
    n = len(existing) + 1
    return f'{STREAMING_DIR}/session_{n:03d}'

MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN       = 7.0
AZ_MAX       = 348.0

DUMP_CADENCE_S = 17.0
SLEW_TIME_S = 5.0

# =========================================================================
# GRID BUILDING & FILTERING (simplified version)
# =========================================================================

def build_raster_cells(l_min, l_max, b_min, b_max, step):
    b_vals = list(range(b_min, b_max + 1, step))
    l_vals = list(range(l_min, l_max + 1, step))
    cells = []
    for row_idx, b in enumerate(b_vals):
        row = [(row_idx, j, l_vals[j], b) for j in range(len(l_vals))]
        if row_idx % 2 == 0:
            row = list(reversed(row))
        cells.extend(row)
    return cells


def filter_cells_by_az_side(cells, side):
    classified = []
    n_permanent = 0
    for row, col, l, b in cells:
        alt, az, ra, dec, _ = compute_gal_pointing(
            l, b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
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
        logger.log(f'  {n_permanent} permanently inaccessible', 'INFO')

    if side == 'rising':
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if rising]
    else:
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if setting]
    return kept


def filter_cells_by_manifest(cells):
    script_dir = Path(__file__).resolve().parent.parent
    manifest_path = script_dir / MANIFEST_PATH
    if not manifest_path.exists():
        return cells
    sys.path.insert(0, str(script_dir))
    from manifest import get_complete_cells
    complete = get_complete_cells(manifest_path)
    if not complete:
        return cells
    kept = [(r, c, l, b) for r, c, l, b in cells if (l, b) not in complete]
    logger.log(f'  Manifest: {len(cells) - len(kept)} skipped, {len(kept)} remaining', 'INFO')
    return kept


def make_scan_target_selector(cells, dumps_per_cell, phase_num):
    """Simplified target selector."""
    cell_list = list(cells)
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    transitioning = False
    done_event = threading.Event()
    skipped = []
    cells_observed_this_pass = 0

    logger.log(f'  Phase {phase_num}: {len(cell_list)} cells', 'INFO')

    def _start_retry_pass():
        nonlocal current_cell_idx, cells_observed_this_pass
        if cells_observed_this_pass == 0:
            logger.log(f'  Phase {phase_num}: No progress, abandoning {len(skipped)}', 'WARN')
            return False
        cell_list[:] = list(skipped)
        skipped.clear()
        current_cell_idx = 0
        cells_observed_this_pass = 0
        return True

    def dump_notifier():
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning, cells_observed_this_pass

        if abort_event.is_set():
            done_event.set()
            return None

        if current_cell_idx >= len(cell_list):
            if skipped:
                if not _start_retry_pass():
                    done_event.set()
                    return None
            else:
                logger.log(f'  Phase {phase_num}: Complete', 'OK')
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
                        logger.log(f'  Phase {phase_num}: Complete', 'OK')
                        done_event.set()
                        return None

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
                        logger.log(f'  Phase {phase_num}: Complete', 'OK')
                        done_event.set()
                        return None
            return None

        row, col = cell_list[current_cell_idx][0], cell_list[current_cell_idx][1]
        return f'obs_{cell_l}_{cell_b}', alt, az, ra, dec

    return target_selector, dump_notifier, done_event


# =========================================================================
# MAIN
# =========================================================================

def main():
    logger.log("="*70, 'INFO')
    logger.log("GALACTIC PLANE SURVEY - BACKUP (RISING ONLY)", 'INFO')
    logger.log("April 26, 2026", 'INFO')
    logger.log("="*70, 'INFO')

    logger.log("\nUse if:", 'INFO')
    logger.log("  - April 25/26 marathon failed/aborted", 'INFO')
    logger.log("  - Weather prevented April 25 evening session", 'INFO')
    logger.log("  - You want rising-side coverage before attempting setting", 'INFO')

    logger.log("\nInitializing hardware...", 'INFO')
    try:
        from ugradio.leusch import LeuschNoise, LeuschTelescope
        from ugradio.sdr import SDR
        telescope = LeuschTelescope()
        noise = LeuschNoise()
        sdr_0 = SDR(device_index=0, direct=False,
                    center_freq=LO_ON_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
        sdr_1 = SDR(device_index=1, direct=False,
                    center_freq=LO_ON_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
        logger.log('Hardware ready.', 'OK')
    except Exception as e:
        logger.log(f'HARDWARE INIT FAILED: {e}', 'FATAL')
        return

    n_complete = 0
    for part in SURVEY_PARTS:
        if abort_event.is_set():
            break

        phase_num = part['phase_num']
        side = part['side']
        l_min, l_max = part['l_min'], part['l_max']
        b_min, b_max = part['b_min'], part['b_max']
        step = part['step']
        dumps_per_band = part['dumps_per_band']
        dumps_per_cell = dumps_per_band * 2

        logger.log("\n" + "="*70, 'INFO')
        logger.log(f"PHASE {phase_num}: {part['name']}", 'INFO')
        logger.log("="*70, 'INFO')

        try:
            all_cells = build_raster_cells(l_min, l_max, b_min, b_max, step)
            cells = filter_cells_by_az_side(all_cells, side)
            cells = filter_cells_by_manifest(cells)

            if not cells:
                logger.log("No remaining cells - skipping", 'WARN')
                continue

            cell_time = dumps_per_cell * DUMP_CADENCE_S + SLEW_TIME_S
            logger.log(f'Ready: {len(cells)} cells, {len(cells)*cell_time/3600:.1f}h', 'INFO')

            target_selector, dump_notifier, done_event = \
                make_scan_target_selector(cells, dumps_per_cell, phase_num)

            read_fn = make_calibrated_sdr_reader(
                sdrs, noise,
                nsamples=NSAMPLES, nblocks=NBLOCKS, nfft=NFFT,
                lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
                cal_dumps_per_lo=CAL_DUMPS,
            )

            def on_save(path, dump, _notifier=dump_notifier):
                _notifier()
                path_obj = Path(path) if isinstance(path, str) else path
                logger.log(f"  [{dump['target_name']}] -> {path_obj.name}", 'DATA')

            capture = StreamingCapture(
                telescope=telescope,
                read_fn=read_fn,
                target_selector=target_selector,
                outdir=_next_session_dir(),
                n_writers=2,
                repoint_interval_sec=REPOINT_TRACK_SEC,
                on_save=on_save,
            )

            logger.log('Starting...', 'INFO')
            capture.run(done_event=done_event)
            logger.log('Phase complete', 'OK')
            n_complete += 1

        except Exception as e:
            logger.log(f'PHASE FAILED: {e}', 'ERROR')
            logger.log(f'{traceback.format_exc()}', 'ERROR')

    logger.log("\n" + "="*70, 'INFO')
    logger.log(f"BACKUP SURVEY COMPLETE ({n_complete}/{len(SURVEY_PARTS)} phases)", 'OK')
    logger.log("="*70, 'INFO')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.log("Interrupted", 'WARN')
        abort_event.set()
    except Exception as e:
        logger.log(f"FATAL: {e}", 'FATAL')
        logger.log(traceback.format_exc(), 'FATAL')
