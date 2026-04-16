#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI OTF grid scan.

Grid-based boustrophedon scan with exact dump counting per cell.

Observation cycle per cell:
  1. Slew to grid cell (no data collected)
  2. Track cell (repoint every REPOINT_TRACK_SEC for sky drift)
  3. Collect exactly DUMPS_PER_BAND ON + DUMPS_PER_BAND OFF dumps
  4. Move to next cell in boustrophedon order -> back to step 1

Calibration:
  - Noise diode ON for first CAL_DUMPS dumps per LO at the start
  - Then frequency-switching science loop

Usage:
    python observe_otf.py

Output:
    data/lab04/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
"""

import threading
import time

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_calibrated_sdr_reader

# ---------------------------------------------------------------------------
# Scan grid (galactic coordinates)
# ---------------------------------------------------------------------------

# Center of the scan region (Cygnus, galactic plane)
GAL_L_CENTER = 90.0
GAL_B_CENTER =  0.0

# Grid parameters (defined in galactic coordinates)
SCAN_STEP_DEG   = 1.0     # cell spacing in l and b (< HPBW/2, oversampled)
N_L_CELLS       = 29      # number of cells in galactic longitude
N_B_CELLS       = 7       # number of cells in galactic latitude
DUMPS_PER_BAND  = 4       # dumps per LO frequency per cell

# Scan direction: 'l' = rows in b sweeping l, 'b' = rows in l sweeping b
# Use 'l' for rising pass, 'b' for setting pass (orthogonal cross-linking)
SCAN_ALONG      = 'l'

# Total dumps per cell: ON + OFF alternating
DUMPS_PER_CELL  = DUMPS_PER_BAND * 2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LO_ON_MHZ   = 1420.0
LO_OFF_MHZ  = 1421.0
SAMPLE_RATE  = 2.56e6    # Hz
NSAMPLES     = 32768
NBLOCKS      = 1025      # block 0 discarded -> 1024 valid
NFFT         = 1024
MIN_ALT_DEG  = 15.5      # Leuschner limit 15 deg + margin
CAL_DUMPS    = 2         # dumps per LO frequency during noise cal
REPOINT_TRACK_SEC = 60.0 # tracking repoint interval (sky drift)
OUTDIR       = 'data/lab04/streaming'


# ---------------------------------------------------------------------------

def setup_hardware():
    """Initialise telescope, SDRs, and noise diode.

    Returns (telescope, sdrs, noise).
    """
    from ugradio.leusch import LeuschNoise, LeuschTelescope
    from ugradio.sdr import SDR

    telescope = LeuschTelescope()
    noise = LeuschNoise()

    sdr_0 = SDR(
        device_index=0, direct=False,
        center_freq=LO_ON_MHZ * 1e6,
        sample_rate=SAMPLE_RATE, gain=0.0,
    )
    sdr_1 = SDR(
        device_index=1, direct=False,
        center_freq=LO_ON_MHZ * 1e6,
        sample_rate=SAMPLE_RATE, gain=0.0,
    )
    return telescope, [sdr_0, sdr_1], noise


def make_scan_target_selector():
    """Create a grid-based scan target selector with dump counting.

    The grid is defined in **galactic coordinates** (l, b) so that both
    rising and setting passes cover the same sky. Each cell's galactic
    position is converted to alt/az at the current time for pointing.

    Returns (target_selector, dump_notifier):
      - target_selector: callable for PointingThread
      - dump_notifier: callable to invoke from on_save after each dump
    """
    # Build galactic offsets centered on scan center
    b_offsets = [
        (i - (N_B_CELLS - 1) / 2) * SCAN_STEP_DEG
        for i in range(N_B_CELLS)
    ]
    l_offsets = [
        (j - (N_L_CELLS - 1) / 2) * SCAN_STEP_DEG
        for j in range(N_L_CELLS)
    ]

    # Build ordered cell list (boustrophedon)
    # SCAN_ALONG='l': rows in b, sweep along l (rising pass)
    # SCAN_ALONG='b': rows in l, sweep along b (setting pass, orthogonal)
    cells = []
    if SCAN_ALONG == 'l':
        for row_idx in range(N_B_CELLS):
            cols = range(N_L_CELLS) if row_idx % 2 == 0 else reversed(range(N_L_CELLS))
            for col_idx in cols:
                cell_l = GAL_L_CENTER + l_offsets[col_idx]
                cell_b = GAL_B_CENTER + b_offsets[row_idx]
                cells.append((row_idx, col_idx, cell_l, cell_b))
    else:  # SCAN_ALONG == 'b'
        for col_idx in range(N_L_CELLS):
            rows = range(N_B_CELLS) if col_idx % 2 == 0 else reversed(range(N_B_CELLS))
            for row_idx in rows:
                cell_l = GAL_L_CENTER + l_offsets[col_idx]
                cell_b = GAL_B_CENTER + b_offsets[row_idx]
                cells.append((row_idx, col_idx, cell_l, cell_b))

    total_cells = len(cells)

    # Shared state
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    transitioning = False

    print(f'  [scan] Grid: {N_L_CELLS} l x {N_B_CELLS} b = {total_cells} cells')
    print(f'  [scan] Galactic coverage: l=[{cells[0][2]:.1f}, {cells[-1][2]:.1f}], '
          f'b=[{b_offsets[0] + GAL_B_CENTER:.1f}, {b_offsets[-1] + GAL_B_CENTER:.1f}]')
    print(f'  [scan] Dumps per cell: {DUMPS_PER_CELL} '
          f'({DUMPS_PER_BAND} ON + {DUMPS_PER_BAND} OFF)')

    def dump_notifier():
        """Call from on_save after each dump is written."""
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning

        if current_cell_idx >= total_cells:
            return None

        # Atomic check: dump count and transition state
        with lock:
            count = cell_dump_count
            if count >= DUMPS_PER_CELL and not transitioning:
                transitioning = True
                return None

            if transitioning:
                transitioning = False
                current_cell_idx += 1
                cell_dump_count = 0
                if current_cell_idx >= total_cells:
                    print('  [scan] All cells complete.')
                    return None
                row, col, cell_l, cell_b = cells[current_cell_idx]
                print(f'  [scan] Cell {current_cell_idx + 1}/{total_cells}: '
                      f'row={row}, col={col}, l={cell_l:.1f}, b={cell_b:.1f}')

        # Current cell: convert galactic -> alt/az at current time
        row, col, cell_l, cell_b = cells[current_cell_idx]

        alt, az, ra, dec, _ = compute_gal_pointing(
            cell_l, cell_b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )

        # Enforce Leuschner limits
        if alt < MIN_ALT_DEG or alt > 85.0:
            return None
        if az < 5.0 or az > 350.0:
            return None

        # Stable name per cell
        name = f'scan_r{row}_c{col}'
        return name, alt, az, ra, dec

    return target_selector, dump_notifier


def main():
    dump_cadence_s = NBLOCKS * NSAMPLES / SAMPLE_RATE + 1.5
    cell_dwell_s = DUMPS_PER_CELL * dump_cadence_s
    total_cells = N_L_CELLS * N_B_CELLS
    total_time_s = total_cells * cell_dwell_s

    scan_dir = 'along l (rows in b)' if SCAN_ALONG == 'l' else 'along b (rows in l)'
    print('Lab 4 - Leuschner 21 cm HI OTF grid scan')
    print(f'  Center: l={GAL_L_CENTER}, b={GAL_B_CENTER}')
    print(f'  Grid: {N_L_CELLS} l x {N_B_CELLS} b = {total_cells} cells (galactic)')
    print(f'  Scan direction: {scan_dir}')
    print(f'  Cell spacing: {SCAN_STEP_DEG} deg')
    print(f'  Coverage: l=[{GAL_L_CENTER - (N_L_CELLS-1)/2*SCAN_STEP_DEG:.1f}, '
          f'{GAL_L_CENTER + (N_L_CELLS-1)/2*SCAN_STEP_DEG:.1f}], '
          f'b=[{GAL_B_CENTER - (N_B_CELLS-1)/2*SCAN_STEP_DEG:.1f}, '
          f'{GAL_B_CENTER + (N_B_CELLS-1)/2*SCAN_STEP_DEG:.1f}]')
    print(f'  Dumps per cell: {DUMPS_PER_CELL} '
          f'({DUMPS_PER_BAND} per band)')
    print(f'  Dwell per cell: ~{cell_dwell_s:.0f}s')
    print(f'  Estimated total: ~{total_time_s / 60:.1f} min '
          f'(+ cal + slews)')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.\n')

    target_selector, dump_notifier = make_scan_target_selector()

    read_fn = make_calibrated_sdr_reader(
        sdrs,
        noise,
        nsamples=NSAMPLES,
        nblocks=NBLOCKS,
        nfft=NFFT,
        lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
        cal_dumps_per_lo=CAL_DUMPS,
    )

    def on_save(path, dump):
        dump_notifier()
        noise_tag = ' [CAL]' if dump.get('noise_on') else ''
        lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
        print(
            f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
            f'{lo_tag}{noise_tag}  -> {path}'
        )

    capture = StreamingCapture(
        telescope=telescope,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=OUTDIR,
        n_writers=2,
        repoint_interval_sec=REPOINT_TRACK_SEC,
        on_save=on_save,
    )
    capture.run()


if __name__ == '__main__':
    main()
