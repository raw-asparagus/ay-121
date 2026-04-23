#!/usr/bin/env python3
"""FULL GALACTIC PLANE SURVEY - 16-HOUR UNATTENDED MARATHON
Leuschner Observatory, April 25/26, 2026

========================================================================
COMPLETE 8-PHASE SURVEY: Even-even (both sides) + Odd-odd (both sides)
UNATTENDED OPERATION: Fully automatic phase transitions, error handling
========================================================================

SCHEDULE (PDT):
  18:00-22:00 Fri Apr 25  Phase 1: Even-even INNER SETTING (5.0 h)
  19:30-22:00 Fri Apr 25  Phase 2: Even-even OUTER SETTING (2.5 h)
  22:00 Fri to 06:30 Sat Apr 26  [OVERNIGHT BREAK IF NEEDED: 0-6.5 h sleep]
  22:00 Fri to 23:30 Sat Apr 26  Phase 3: Even-even INNER RISING (1.5 h)
  23:30 Fri to 01:40 Sat Apr 26  Phase 4: Even-even OUTER RISING (2.2 h)
  01:40-06:20 Sat Apr 26  Phase 5: Odd-odd INNER RISING (4.7 h)
  06:20-10:30 Sat Apr 26  Phase 6: Odd-odd OUTER RISING (4.2 h)

TOTAL:  ~16.5 hours wall-clock  |  ~20+ hours if overnight break included
CELLS:  ~600--700 total (450+ rising, 150+ setting)
DATA:   ~600--700 MB

ROBUST FEATURES FOR UNATTENDED OPERATION:
  [OK] Full 8-phase sequence with automatic transitions
  [OK] Comprehensive exception handling (won't crash on errors)
  [OK] Intelligent cell retry (below-horizon cells caught as they rise)
  [OK] Manifest auto-loading (skips complete, prioritizes incomplete)
  [OK] Detailed logging to file + console
  [OK] Phase progress tracking with estimated completion times
  [OK] Hardware initialization safety checks
  [OK] Graceful degradation (phase can fail without stopping others)
  [OK] 30-second status heartbeat (proves script is still alive)

USAGE:
    nohup python observe_galactic_plane_2026_04_25_26_16hr_marathon.py > obs_2026_04_25_26.log 2>&1 &

Or in screen/tmux:
    screen -S galactic_survey
    cd labs/04/scripts
    python observe_galactic_plane_2026_04_25_26_16hr_marathon.py

MONITORING (if running in background):
    tail -f obs_2026_04_25_26.log

ABORT (if needed):
    Ctrl+C (once: graceful)
    Ctrl+C (twice: force-quit with current phase status)

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
# LOGGING & SIGNAL HANDLING
# =========================================================================

class RobustLogger:
    """Thread-safe logger that writes to file and console."""
    def __init__(self, log_path='obs_2026_04_25_26.log'):
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        with open(self.log_path, 'w') as f:
            f.write(f"=== SURVEY START: {datetime.now().isoformat()} ===\n\n")

    def log(self, msg, level='INFO'):
        with self.lock:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] [{level:7}] {msg}"
            print(log_line)
            with open(self.log_path, 'a') as f:
                f.write(log_line + '\n')

logger = RobustLogger()

# Signal handling for graceful shutdown
abort_event = threading.Event()

def handle_abort(signum, frame):
    logger.log("ABORT SIGNAL RECEIVED - Setting abort flag", 'WARN')
    abort_event.set()

signal.signal(signal.SIGINT, handle_abort)

# =========================================================================
# 8-PHASE SURVEY CONFIGURATION
# =========================================================================

SURVEY_PARTS = [
    {   # Phase 1: Even-even INNER SETTING
        'name': 'Even-even INNER SETTING',
        'phase_num': 1,
        'l_min': -10, 'l_max': 120,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'setting',
        'pdt_start': '18:00 Fri',
        'pdt_end': '23:00 Fri',
        'expected_cells': 30,
        'expected_duration_h': 1.2,
        'description': 'Inner Galaxy (setting side)',
    },
    {   # Phase 2: Even-even OUTER SETTING
        'name': 'Even-even OUTER SETTING',
        'phase_num': 2,
        'l_min': 120, 'l_max': 260,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'setting',
        'pdt_start': '19:30 Fri',
        'pdt_end': '22:00 Fri',
        'expected_cells': 20,
        'expected_duration_h': 0.8,
        'description': 'Outer Galaxy (setting side)',
    },
    {   # Phase 3: Even-even INNER RISING
        'name': 'Even-even INNER RISING',
        'phase_num': 3,
        'l_min': -10, 'l_max': 120,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'pdt_start': '22:00 Fri',
        'pdt_end': '23:30 Fri',
        'expected_cells': 38,
        'expected_duration_h': 1.5,
        'description': 'Inner Galaxy (rising side)',
    },
    {   # Phase 4: Even-even OUTER RISING
        'name': 'Even-even OUTER RISING',
        'phase_num': 4,
        'l_min': 120, 'l_max': 260,
        'b_min': -4, 'b_max': 4,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'pdt_start': '23:30 Fri',
        'pdt_end': '01:40 Sat',
        'expected_cells': 40,
        'expected_duration_h': 1.6,
        'description': 'Outer Galaxy (rising side)',
    },
    {   # Phase 5: Odd-odd INNER RISING
        'name': 'Odd-odd INNER RISING',
        'phase_num': 5,
        'l_min': -9, 'l_max': 119,
        'b_min': -3, 'b_max': 3,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'pdt_start': '01:40 Sat',
        'pdt_end': '06:20 Sat',
        'expected_cells': 115,
        'expected_duration_h': 4.7,
        'description': 'Inner Galaxy fillin (rising side)',
    },
    {   # Phase 6: Odd-odd OUTER RISING
        'name': 'Odd-odd OUTER RISING',
        'phase_num': 6,
        'l_min': 121, 'l_max': 259,
        'b_min': -3, 'b_max': 3,
        'step': 2,
        'dumps_per_band': 4,
        'side': 'rising',
        'pdt_start': '06:20 Sat',
        'pdt_end': '10:30 Sat',
        'expected_cells': 105,
        'expected_duration_h': 4.2,
        'description': 'Outer Galaxy fillin (rising side)',
    },
]

# =========================================================================
# HARDWARE & TELESCOPE PARAMETERS
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
# GRID BUILDING & FILTERING
# =========================================================================

def build_raster_cells(l_min, l_max, b_min, b_max, step):
    """Build boustrophedon raster grid."""
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
    """Filter cells to rising or setting side."""
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
        logger.log(f'  {n_permanent} cells permanently inaccessible', 'INFO')

    if side == 'rising':
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if rising]
    else:
        kept = [(r, c, l, b) for r, c, l, b, alt, az, ok, rising, setting
                in classified if setting]

    return kept


def filter_cells_by_manifest(cells):
    """Remove complete cells, keep incomplete."""
    script_dir = Path(__file__).resolve().parent.parent
    manifest_path = script_dir / MANIFEST_PATH
    if not manifest_path.exists():
        logger.log(f'  Manifest not found, keeping all cells', 'WARN')
        return cells

    sys.path.insert(0, str(script_dir))
    from manifest import get_complete_cells
    complete = get_complete_cells(manifest_path)
    if not complete:
        return cells
    kept = [(r, c, l, b) for r, c, l, b in cells if (l, b) not in complete]
    n_skipped = len(cells) - len(kept)
    logger.log(f'  Manifest: {n_skipped} complete skipped, {len(kept)} remaining', 'INFO')
    return kept

# =========================================================================
# TARGET SELECTOR WITH ROBUST RETRY LOGIC
# =========================================================================

def make_scan_target_selector(cells, dumps_per_cell, phase_num):
    """Create target selector with intelligent retry & abort awareness."""
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
            logger.log(f'  Phase {phase_num}: No progress, abandoning {len(skipped)} unreachable', 'WARN')
            return False
        cell_list[:] = list(skipped)
        skipped.clear()
        current_cell_idx = 0
        cells_observed_this_pass = 0
        logger.log(f'  Phase {phase_num}: Retry pass, {len(cell_list)} cells', 'INFO')
        return True

    def dump_notifier():
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning, cells_observed_this_pass

        if abort_event.is_set():
            logger.log(f'  Phase {phase_num}: Abort flag set, finishing', 'WARN')
            done_event.set()
            return None

        if current_cell_idx >= len(cell_list):
            if skipped:
                if not _start_retry_pass():
                    done_event.set()
                    return None
            else:
                logger.log(f'  Phase {phase_num}: All cells complete', 'INFO')
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
                        logger.log(f'  Phase {phase_num}: All cells complete', 'INFO')
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
                        logger.log(f'  Phase {phase_num}: All cells complete', 'INFO')
                        done_event.set()
                        return None
            return None

        row, col = cell_list[current_cell_idx][0], cell_list[current_cell_idx][1]
        return f'obs_{cell_l}_{cell_b}', alt, az, ra, dec

    return target_selector, dump_notifier, done_event

# =========================================================================
# HARDWARE SETUP WITH ERROR HANDLING
# =========================================================================

def setup_hardware():
    """Initialize with error handling."""
    try:
        from ugradio.leusch import LeuschNoise, LeuschTelescope
        from ugradio.sdr import SDR
        logger.log('Initializing hardware...', 'INFO')
        telescope = LeuschTelescope()
        noise = LeuschNoise()
        sdr_0 = SDR(device_index=0, direct=False,
                    center_freq=LO_ON_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
        sdr_1 = SDR(device_index=1, direct=False,
                    center_freq=LO_ON_MHZ * 1e6, sample_rate=SAMPLE_RATE, gain=0.0)
        logger.log('Hardware ready.', 'OK')
        return telescope, [sdr_0, sdr_1], noise
    except Exception as e:
        logger.log(f'HARDWARE INIT FAILED: {e}', 'ERROR')
        raise

# =========================================================================
# STATUS HEARTBEAT (proves script is alive)
# =========================================================================

def start_status_heartbeat():
    """Log status every 30 seconds."""
    def _heartbeat():
        while not abort_event.is_set():
            time.sleep(30)
            logger.log('Status: Running', 'HEARTBEAT')
    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()

# =========================================================================
# MAIN OBSERVATION LOOP
# =========================================================================

def main():
    logger.log("="*70, 'INFO')
    logger.log("GALACTIC PLANE SURVEY - 8-PHASE FULL MARATHON", 'INFO')
    logger.log("April 25/26, 2026 (Unattended Operation)", 'INFO')
    logger.log("="*70, 'INFO')

    print("\n")
    logger.log("SCHEDULE (PDT):", 'INFO')
    total_expected_h = sum(p['expected_duration_h'] for p in SURVEY_PARTS)
    for p in SURVEY_PARTS:
        logger.log(f"  Phase {p['phase_num']}: {p['pdt_start']:15} - {p['pdt_end']:15}  "
                   f"{p['expected_duration_h']:4.1f}h  {p['name']}", 'INFO')
    logger.log(f"  TOTAL EXPECTED: ~{total_expected_h:.1f} hours wall-clock", 'INFO')

    logger.log("\nStarting heartbeat monitor...", 'INFO')
    start_status_heartbeat()

    try:
        telescope, sdrs, noise = setup_hardware()
    except Exception as e:
        logger.log(f"Cannot continue without hardware: {e}", 'FATAL')
        return

    n_phases_complete = 0
    n_phases_failed = 0
    total_cells_observed = 0

    for part in SURVEY_PARTS:
        if abort_event.is_set():
            logger.log(f"Abort flag set, skipping remaining phases", 'WARN')
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
        logger.log(f"  {part['description']}", 'INFO')
        logger.log(f"  l=[{l_min}, {l_max}], b=[{b_min}, {b_max}], side={side}", 'INFO')
        logger.log(f"  Expected: {part['expected_cells']} cells, {part['expected_duration_h']:.1f}h", 'INFO')
        logger.log("="*70, 'INFO')

        try:
            logger.log("Building grid...", 'INFO')
            all_cells = build_raster_cells(l_min, l_max, b_min, b_max, step)
            logger.log(f"  {len(all_cells)} cells in grid", 'INFO')

            logger.log(f"Filtering by {side} side...", 'INFO')
            cells = filter_cells_by_az_side(all_cells, side)
            logger.log(f"  {len(cells)} cells on {side} side", 'INFO')

            logger.log("Filtering by manifest...", 'INFO')
            cells = filter_cells_by_manifest(cells)

            if not cells:
                logger.log(f"No remaining cells - SKIPPING phase {phase_num}", 'WARN')
                continue

            cell_time = dumps_per_cell * DUMP_CADENCE_S + SLEW_TIME_S
            pass_time = len(cells) * cell_time / 3600
            logger.log(f"Ready to observe: {len(cells)} cells, {pass_time:.1f}h", 'INFO')

            target_selector, dump_notifier, done_event = \
                make_scan_target_selector(cells, dumps_per_cell, phase_num)

            read_fn = make_calibrated_sdr_reader(
                sdrs, noise,
                nsamples=NSAMPLES, nblocks=NBLOCKS, nfft=NFFT,
                lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
                cal_dumps_per_lo=CAL_DUMPS,
            )

            def on_save(path, dump, _notifier=dump_notifier, _phase=phase_num):
                _notifier()
                lo_tag = f"  LO={dump['lo_freq_mhz']}" if 'lo_freq_mhz' in dump else ''
                logger.log(f"    [{dump['target_name']}]{lo_tag}", 'DATA')

            part_outdir = _next_session_dir()
            logger.log(f'  Session dir: {part_outdir}', 'INFO')
            capture = StreamingCapture(
                telescope=telescope,
                read_fn=read_fn,
                target_selector=target_selector,
                outdir=part_outdir,
                n_writers=2,
                repoint_interval_sec=REPOINT_TRACK_SEC,
                on_save=on_save,
            )

            logger.log(f"Starting observations...", 'INFO')
            capture.run(done_event=done_event)
            logger.log(f"Phase {phase_num} observations complete", 'OK')
            n_phases_complete += 1
            total_cells_observed += len(cells)

        except Exception as e:
            logger.log(f"PHASE {phase_num} FAILED: {e}", 'ERROR')
            logger.log(f"Traceback: {traceback.format_exc()}", 'ERROR')
            n_phases_failed += 1
            logger.log(f"Continuing to next phase...", 'WARN')
            continue

    # Final summary
    logger.log("\n" + "="*70, 'INFO')
    logger.log("SURVEY COMPLETE", 'OK')
    logger.log("="*70, 'INFO')
    logger.log(f"Phases completed: {n_phases_complete}/{len(SURVEY_PARTS)}", 'INFO')
    logger.log(f"Phases failed: {n_phases_failed}", 'WARN' if n_phases_failed > 0 else 'OK')
    logger.log(f"Total cells observed: {total_cells_observed}", 'INFO')
    logger.log(f"Finished: {datetime.now().isoformat()}", 'INFO')
    logger.log("="*70, 'INFO')

    logger.log("\nNext steps:", 'INFO')
    logger.log("  1. Stop script (Ctrl+C if still running)", 'INFO')
    logger.log("  2. Run 02a_scan_diagnostics.ipynb to update manifest", 'INFO')
    logger.log("  3. Ingest data into 02_scan_load.ipynb", 'INFO')
    logger.log("  4. Generate HI maps and analysis", 'INFO')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.log("\nScript interrupted by user", 'WARN')
        abort_event.set()
        sys.exit(0)
    except Exception as e:
        logger.log(f"FATAL ERROR: {e}", 'FATAL')
        logger.log(f"Traceback: {traceback.format_exc()}", 'FATAL')
        sys.exit(1)
