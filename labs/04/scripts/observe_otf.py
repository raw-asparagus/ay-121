#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI OTF parallelogram scan.

Two-pass parallelogram scan with opposite slants for cross-linking:
  - Rising pass: parallelogram slanted along the iso-HA direction
  - Setting pass: parallelogram slanted in the opposite direction
Both passes run automatically with a noise cal at the start of each.

The parallelogram slant is computed from the HA gradient at the scan
center, ensuring the starting edge of each parallelogram is at uniform
hour angle. All pointings are at whole-degree galactic coordinates.

Usage:
    python observe_otf.py

Output:
    data/lab04/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
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
from ugradiolab.capture.readers import make_calibrated_sdr_reader

# ---------------------------------------------------------------------------
# Parallelogram scan parameters (galactic coordinates)
# ---------------------------------------------------------------------------

GAL_L_CENTER = 90.0       # galactic longitude of scan center
GAL_B_CENTER =  0.0       # galactic latitude of scan center
L_EXTENT     = 17         # cells per row (setting may skip ~3 cells near az limit)
B_EXTENT     = 9          # number of b-rows
DUMPS_PER_BAND = 4        # dumps per LO frequency per cell
DUMPS_PER_CELL = DUMPS_PER_BAND * 2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LO_ON_MHZ   = 1420.0
LO_OFF_MHZ  = 1421.0
SAMPLE_RATE  = 2.56e6
NSAMPLES     = 32768
NBLOCKS      = 1025
NFFT         = 1024
# Leuschner hardware: alt 15-85, az 5-350. Conservative guard: +2 deg margin.
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN       =  7.0
AZ_MAX       = 348.0
CAL_DUMPS    = 2
REPOINT_TRACK_SEC = 60.0
OUTDIR       = 'data/lab04/streaming'


# ---------------------------------------------------------------------------
# Parallelogram grid builder
# ---------------------------------------------------------------------------

def compute_iso_ha_slant():
    """Compute the iso-HA slant at the scan center (deg l per deg b)."""
    import astropy.coordinates as _ac
    import astropy.units as _u
    from astropy.time import Time as _Time

    _now = _Time.now()
    _lst = _now.sidereal_time('apparent', longitude=LEO_LON_DEG * _u.deg)

    def _ha(l_deg, b_deg):
        gc = _ac.SkyCoord(l=l_deg * _u.deg, b=b_deg * _u.deg, frame="galactic")
        icrs = gc.transform_to(_ac.ICRS())
        return (_lst - icrs.ra).wrap_at(12 * _u.hourangle).deg

    dha_dl = (_ha(GAL_L_CENTER + 1, GAL_B_CENTER) -
              _ha(GAL_L_CENTER - 1, GAL_B_CENTER)) / 2
    dha_db = (_ha(GAL_L_CENTER, GAL_B_CENTER + 1) -
              _ha(GAL_L_CENTER, GAL_B_CENTER - 1)) / 2
    return -dha_db / dha_dl


def build_parallelogram_cells(slant):
    """Build a parallelogram grid at whole-degree galactic coordinates.

    Returns list of (row_idx, col_idx, l, b) tuples in boustrophedon order,
    starting from the highest-HA edge.
    """
    import astropy.coordinates as _ac
    import astropy.units as _u
    from astropy.time import Time as _Time

    b_vals = [int(GAL_B_CENTER + (i - (B_EXTENT - 1) / 2)) for i in range(B_EXTENT)]

    # Build all cells per row with integer l-shift
    rows = {}
    for row_idx, b_val in enumerate(b_vals):
        l_shift = round(slant * (b_val - GAL_B_CENTER))
        l_center_row = int(GAL_L_CENTER) + l_shift
        l_vals = [l_center_row + j - (L_EXTENT - 1) // 2 for j in range(L_EXTENT)]
        rows[row_idx] = [(row_idx, j, l_vals[j], b_val) for j in range(L_EXTENT)]

    # Compute HA of the first cell in each possible starting corner
    _now = _Time.now()
    _lst = _now.sidereal_time('apparent', longitude=LEO_LON_DEG * _u.deg)

    def _ha(l_deg, b_deg):
        gc = _ac.SkyCoord(l=l_deg * _u.deg, b=b_deg * _u.deg, frame="galactic")
        icrs = gc.transform_to(_ac.ICRS())
        return (_lst - icrs.ra).wrap_at(12 * _u.hourangle).deg

    # Determine starting direction: first row's first cell vs last cell
    first_row = rows[0]
    last_row = rows[B_EXTENT - 1]
    corners = [
        (0, False, _ha(first_row[0][2], first_row[0][3])),     # row 0, forward
        (0, True, _ha(first_row[-1][2], first_row[-1][3])),     # row 0, reversed
        (B_EXTENT-1, False, _ha(last_row[0][2], last_row[0][3])),
        (B_EXTENT-1, True, _ha(last_row[-1][2], last_row[-1][3])),
    ]
    best = max(corners, key=lambda x: x[2])
    start_from_last_row = (best[0] == B_EXTENT - 1)
    start_l_reversed = best[1]

    # Build boustrophedon
    cells = []
    b_order = range(B_EXTENT - 1, -1, -1) if start_from_last_row else range(B_EXTENT)
    for i, row_idx in enumerate(b_order):
        row = list(rows[row_idx])
        reverse_this = (i % 2 == 0) == start_l_reversed
        if reverse_this:
            row = list(reversed(row))
        cells.extend(row)

    return cells


# ---------------------------------------------------------------------------
# Target selector factory
# ---------------------------------------------------------------------------

def make_scan_target_selector(cells):
    """Create target selector and dump notifier for a cell list."""
    total_cells = len(cells)
    lock = threading.Lock()
    cell_dump_count = 0
    current_cell_idx = 0
    transitioning = False

    print(f'  [scan] {total_cells} cells')
    print(f'  [scan] First: l={cells[0][2]}, b={cells[0][3]}')
    print(f'  [scan] Last:  l={cells[-1][2]}, b={cells[-1][3]}')

    def dump_notifier():
        nonlocal cell_dump_count
        with lock:
            cell_dump_count += 1

    def target_selector():
        nonlocal current_cell_idx, cell_dump_count, transitioning

        if current_cell_idx >= total_cells:
            return None

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
                _, _, cl, cb = cells[current_cell_idx]
                print(f'  [scan] Cell {current_cell_idx+1}/{total_cells}: l={cl}, b={cb}')

        _, _, cell_l, cell_b = cells[current_cell_idx]
        alt, az, ra, dec, _ = compute_gal_pointing(
            cell_l, cell_b,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )

        if alt < MIN_ALT_DEG or alt > MAX_ALT_DEG or az < AZ_MIN or az > AZ_MAX:
            return None

        row, col = cells[current_cell_idx][0], cells[current_cell_idx][1]
        return f'scan_r{row}_c{col}', alt, az, ra, dec

    return target_selector, dump_notifier


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


def run_pass(telescope, sdrs, noise, cells, pass_name):
    """Run one scan pass (rising or setting) with calibration."""
    print(f'\n{"="*60}')
    print(f'  {pass_name}')
    print(f'{"="*60}')

    target_selector, dump_notifier = make_scan_target_selector(cells)

    read_fn = make_calibrated_sdr_reader(
        sdrs, noise,
        nsamples=NSAMPLES, nblocks=NBLOCKS, nfft=NFFT,
        lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
        cal_dumps_per_lo=CAL_DUMPS,
    )

    def on_save(path, dump):
        dump_notifier()
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
    capture.run()


def main():
    print('Lab 4 - Leuschner 21 cm HI OTF parallelogram scan')
    print(f'  Center: l={GAL_L_CENTER}, b={GAL_B_CENTER}')
    print(f'  Grid: {L_EXTENT} l x {B_EXTENT} b = {L_EXTENT * B_EXTENT} cells/pass')
    print(f'  Dumps per cell: {DUMPS_PER_CELL} ({DUMPS_PER_BAND}/band)')

    # Compute iso-HA slant
    iso_ha_slant = compute_iso_ha_slant()
    print(f'  Iso-HA slant: {iso_ha_slant:+.2f} deg l per deg b')
    print()

    # Build both parallelogram cell lists
    rising_slant = iso_ha_slant
    setting_slant = -iso_ha_slant

    print(f'Rising parallelogram (slant={rising_slant:+.2f}):')
    rising_cells = build_parallelogram_cells(rising_slant)
    for b in sorted(set(c[3] for c in rising_cells)):
        row = [c for c in rising_cells if c[3] == b]
        ls = sorted(c[2] for c in row)
        print(f'  b={b:+d}: l=[{ls[0]}, {ls[-1]}]')

    print(f'\nSetting parallelogram (slant={setting_slant:+.2f}):')
    setting_cells = build_parallelogram_cells(setting_slant)
    for b in sorted(set(c[3] for c in setting_cells)):
        row = [c for c in setting_cells if c[3] == b]
        ls = sorted(c[2] for c in row)
        print(f'  b={b:+d}: l=[{ls[0]}, {ls[-1]}]')

    dump_cadence = NBLOCKS * NSAMPLES / SAMPLE_RATE + 1.5
    cell_time = DUMPS_PER_CELL * dump_cadence + 5
    pass_time = len(rising_cells) * cell_time / 3600
    print(f'\nEstimated {pass_time:.1f} h per pass, {2*pass_time + 0.5:.1f} h total')

    # Initialise hardware once
    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.')

    # ---- RISING PASS ----
    run_pass(telescope, sdrs, noise, rising_cells, 'RISING PASS (along l)')

    # ---- Wait for az gap to clear, then SETTING PASS ----
    print('\n' + '='*60)
    print('  Rising pass complete. Waiting for setting window...')
    print('  (The script will auto-start when the target clears az exclusion)')
    print('='*60 + '\n')

    run_pass(telescope, sdrs, noise, setting_cells, 'SETTING PASS (along l, opposite slant)')

    print('\n' + '='*60)
    print('  Both passes complete!')
    print('='*60)


if __name__ == '__main__':
    main()
