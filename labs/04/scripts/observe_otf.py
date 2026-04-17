#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI OTF parallelogram scan.

Parallelogram raster in galactic coordinates, slanted along the
iso-hour-angle direction. Boustrophedon ordering with even rows
scanning in decreasing l to follow the sky rotation.

Usage:
    python observe_otf.py

Output:
    data/lab04/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
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
# Scan grid (galactic coordinates)
# ---------------------------------------------------------------------------

GAL_L_CENTER = 111.0      # Cepheus/Cassiopeia, extends DR1 toward l=120
GAL_B_CENTER =   0.0      # galactic plane
L_EXTENT     =  17        # cells per row
B_EXTENT     =   9        # number of b-rows
DUMPS_PER_BAND = 4        # dumps per LO per cell (bright plane)
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

    For each b-row, the l-range is shifted by round(slant * (b - b_center)).
    Returns list of (row_idx, col_idx, l, b) tuples in boustrophedon order,
    with even rows scanning in decreasing l.
    """
    b_vals = [int(GAL_B_CENTER + (i - (B_EXTENT - 1) / 2))
              for i in range(B_EXTENT)]

    cells_by_row = {}
    for row_idx, b_val in enumerate(b_vals):
        l_shift = round(slant * (b_val - GAL_B_CENTER))
        l_row_center = int(GAL_L_CENTER) + l_shift
        l_row = [l_row_center + j - (L_EXTENT - 1) // 2
                 for j in range(L_EXTENT)]
        cells_by_row[row_idx] = [
            (row_idx, j, l_row[j], b_val) for j in range(L_EXTENT)
        ]

    # Boustrophedon: even rows decreasing l
    cells = []
    for row_idx in range(B_EXTENT):
        row = list(cells_by_row[row_idx])
        if row_idx % 2 == 0:
            row = list(reversed(row))
        cells.extend(row)

    return cells


# ---------------------------------------------------------------------------
# Target selector factory
# ---------------------------------------------------------------------------

def make_scan_target_selector(cells):
    """Create target selector, dump notifier, and done_event for a cell list.

    Returns (target_selector, dump_notifier, done_event).
    """
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
            if count >= DUMPS_PER_CELL and not transitioning:
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
    print('Lab 4 - Leuschner 21 cm HI OTF parallelogram scan')
    print(f'  Center: l={GAL_L_CENTER}, b={GAL_B_CENTER}')
    print(f'  Grid: {L_EXTENT} l x {B_EXTENT} b = {L_EXTENT * B_EXTENT} cells')
    print(f'  Dumps per cell: {DUMPS_PER_CELL} ({DUMPS_PER_BAND}/band)')

    # Compute iso-HA slant
    iso_ha_slant = compute_iso_ha_slant()
    print(f'  Iso-HA slant: {iso_ha_slant:+.2f} deg l per deg b')

    cells = build_parallelogram_cells(iso_ha_slant)

    # Print grid layout
    b_vals = sorted(set(c[3] for c in cells))
    for b in b_vals:
        row = sorted([c for c in cells if c[3] == b], key=lambda c: c[2])
        shift = round(iso_ha_slant * (b - GAL_B_CENTER))
        print(f'  b={b:+d}: l=[{row[0][2]}, {row[-1][2]}] (shift={shift:+d})')

    dump_cadence = NBLOCKS * NSAMPLES / SAMPLE_RATE + 1.5
    cell_time = DUMPS_PER_CELL * dump_cadence + 5
    pass_time = len(cells) * cell_time / 3600
    print(f'\nEstimated {pass_time:.1f} h')

    print('\nInitialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.')

    target_selector, dump_notifier, done_event = make_scan_target_selector(cells)

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
    capture.run(done_event=done_event)

    print('\n' + '=' * 60)
    print('  Scan complete!')
    print('=' * 60)


if __name__ == '__main__':
    main()
