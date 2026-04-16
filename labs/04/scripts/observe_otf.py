#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI OTF scan.

On-the-fly boustrophedon scan: constant-elevation azimuth sweeps,
stepping in elevation between rows. Frequency-switched between two
LO settings for bandpass removal.

Scan strategy:
  - Sweep SCAN_THROW_DEG in azimuth at fixed elevation
  - Step SCAN_STEP_DEG in elevation between sweeps
  - Boustrophedon: alternating sweep direction per row
  - Scan rate set by SCAN_RATE_DEG_S

Calibration:
  - Noise diode ON for first CAL_DUMPS dumps per LO at the start
  - Then frequency-switching science loop until scan complete or Ctrl-C

Usage:
    python observe_otf.py

Output:
    data/lab04/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
"""

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

# Center of the scan region
GAL_L_CENTER = 180.0
GAL_B_CENTER =   0.0

# Scan parameters
SCAN_THROW_DEG  = 5.0     # azimuth sweep width (degrees)
SCAN_STEP_DEG   = 1.7     # elevation step between sweeps (~HPBW/2)
N_EL_ROWS       = 5       # number of elevation rows (centered on target)
SCAN_RATE_DEG_S = 0.03    # azimuth scan rate (degrees/sec), ~1.8'/s

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
    """Create an OTF scan target selector.

    Returns a callable that advances through a boustrophedon scan pattern
    in (alt, az) at constant elevation rows. The scan starts from the
    initial alt/az of the galactic center coordinates and sweeps in azimuth.

    Each call returns (name, alt, az, ra, dec) or None.
    """
    # Compute the initial (alt, az) for the center position
    center_alt, center_az, center_ra, center_dec, _ = compute_gal_pointing(
        GAL_L_CENTER, GAL_B_CENTER,
        lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
    )

    # Build elevation row offsets centered on the target
    el_offsets = [
        (i - (N_EL_ROWS - 1) / 2) * SCAN_STEP_DEG
        for i in range(N_EL_ROWS)
    ]

    current_row = 0
    row_start_time = time.time()
    scan_direction = 1  # +1 = forward, -1 = reverse (boustrophedon)

    def target_selector():
        nonlocal current_row, row_start_time, scan_direction

        if current_row >= N_EL_ROWS:
            return None  # scan complete

        # Current elevation = center + row offset
        el_offset = el_offsets[current_row]

        # Recompute center alt/az (sky rotates over time)
        c_alt, c_az, c_ra, c_dec, _ = compute_gal_pointing(
            GAL_L_CENTER, GAL_B_CENTER,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )

        target_alt = c_alt + el_offset

        # Azimuth sweep: linear in time from -throw/2 to +throw/2
        elapsed_in_row = time.time() - row_start_time
        az_offset = scan_direction * (
            -SCAN_THROW_DEG / 2 + SCAN_RATE_DEG_S * elapsed_in_row
        )

        # Check if sweep is complete
        if abs(az_offset) > SCAN_THROW_DEG / 2:
            current_row += 1
            row_start_time = time.time()
            scan_direction *= -1  # reverse for next row
            if current_row >= N_EL_ROWS:
                print('  [scan] All rows complete.')
                return None
            el_offset = el_offsets[current_row]
            target_alt = c_alt + el_offset
            az_offset = scan_direction * (-SCAN_THROW_DEG / 2)
            print(f'  [scan] Row {current_row + 1}/{N_EL_ROWS}, '
                  f'el offset = {el_offset:+.1f} deg')

        target_az = c_az + az_offset

        # Enforce Leuschner limits
        if target_alt < MIN_ALT_DEG or target_alt > 85.0:
            return None
        if target_az < 5.0 or target_az > 350.0:
            return None

        name = f'scan_r{current_row}_el{target_alt:.1f}_az{target_az:.1f}'
        return name, target_alt, target_az, c_ra, c_dec

    return target_selector


def on_save(path, dump):
    noise_tag = ' [CAL]' if dump.get('noise_on') else ''
    lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
        f'{lo_tag}{noise_tag}  -> {path}'
    )


def main():
    sweep_time = SCAN_THROW_DEG / SCAN_RATE_DEG_S
    total_time = sweep_time * N_EL_ROWS

    print('Lab 4 - Leuschner 21 cm HI OTF scan')
    print(f'  Center: l={GAL_L_CENTER}, b={GAL_B_CENTER}')
    print(f'  Throw: {SCAN_THROW_DEG} deg az, '
          f'{N_EL_ROWS} rows x {SCAN_STEP_DEG} deg el')
    print(f'  Scan rate: {SCAN_RATE_DEG_S * 60:.1f} arcmin/s')
    print(f'  Estimated time: {sweep_time:.0f}s/row x {N_EL_ROWS} rows '
          f'= {total_time / 60:.1f} min')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.\n')

    target_selector = make_scan_target_selector()

    read_fn = make_calibrated_sdr_reader(
        sdrs,
        noise,
        nsamples=NSAMPLES,
        nblocks=NBLOCKS,
        nfft=NFFT,
        lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
        cal_dumps_per_lo=CAL_DUMPS,
    )
    capture = StreamingCapture(
        telescope=telescope,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=OUTDIR,
        n_writers=2,
        repoint_interval_sec=10.0,  # repoint frequently for OTF tracking
        on_save=on_save,
    )
    capture.run()


if __name__ == '__main__':
    main()
