#!/usr/bin/env python3
"""Lab 4 — Leuschner 21 cm HI observation.

Dual-polarisation SDR capture with on-board FFT and correlation.
Frequency-switched between two LO settings for bandpass removal.

Calibration protocol (per target):
  1. Slew to target
  2. Noise diode ON  → 32 dumps at LO = 1420 MHz, 32 at LO = 1421 MHz
  3. Noise diode OFF → frequency-switching science loop (1420 / 1421 alternating)

Usage:
    python observe.py

Output:
    data/lab04/streaming/calibration/noisecal_<timestamp>.npz
    data/lab04/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
"""

import os
import time

import numpy as np

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_sdr_reader

# ---------------------------------------------------------------------------
# Source catalog (galactic coordinates)
# ---------------------------------------------------------------------------

GAL_L_DEG = 180.0   # galactic longitude
GAL_B_DEG =   0.0   # galactic latitude (plane)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LO_ON_MHZ   = 1420.0
LO_OFF_MHZ  = 1421.0
SAMPLE_RATE  = 2.56e6    # Hz
NSAMPLES     = 32768
NBLOCKS      = 65        # block 0 discarded → 64 valid
NFFT         = 1024
MIN_ALT_DEG  = 15.5      # Leuschner limit 15° + margin
CAL_DUMPS    = 32        # dumps per LO frequency during noise cal
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


def _capture_cal_dumps(sdrs, lo_mhz, n_dumps):
    """Capture *n_dumps* correlation dumps at a single LO frequency.

    Returns lists of corr00, corr01, corr11 arrays and timestamps.
    """
    reader = make_sdr_reader(
        sdrs,
        nsamples=NSAMPLES,
        nblocks=NBLOCKS,
        nfft=NFFT,
        lo_freqs_mhz=(lo_mhz,),
    )
    corr00_list, corr01_list, corr11_list, times = [], [], [], []
    for i in range(n_dumps):
        d = reader(None)
        corr00_list.append(d['corr00'])
        corr01_list.append(d['corr01'])
        corr11_list.append(d['corr11'])
        times.append(d['time'])
        print(f'    cal dump {i + 1}/{n_dumps}  (LO={lo_mhz} MHz)')
    return corr00_list, corr01_list, corr11_list, times


def collect_noise_cal(sdrs, noise, n_dumps=CAL_DUMPS):
    """Noise diode calibration bracket.

    Collects *n_dumps* at each LO frequency with the noise diode ON,
    then turns the diode OFF.  Returns the path to the saved .npz file.
    """
    noise.on()
    print('  Noise diode ON')

    print(f'  Capturing {n_dumps} dumps at LO = {LO_ON_MHZ} MHz ...')
    c00_on, c01_on, c11_on, t_on = _capture_cal_dumps(sdrs, LO_ON_MHZ, n_dumps)

    print(f'  Capturing {n_dumps} dumps at LO = {LO_OFF_MHZ} MHz ...')
    c00_off, c01_off, c11_off, t_off = _capture_cal_dumps(sdrs, LO_OFF_MHZ, n_dumps)

    noise.off()
    print('  Noise diode OFF')

    # Save.
    cal_dir = os.path.join(OUTDIR, 'calibration')
    os.makedirs(cal_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(cal_dir, f'noisecal_{ts}.npz')
    np.savez(
        path,
        # LO = 1420 MHz (noise ON)
        corr00_on=np.array(c00_on),
        corr01_on=np.array(c01_on),
        corr11_on=np.array(c11_on),
        times_on=np.array(t_on),
        lo_on_mhz=LO_ON_MHZ,
        # LO = 1421 MHz (noise ON)
        corr00_off=np.array(c00_off),
        corr01_off=np.array(c01_off),
        corr11_off=np.array(c11_off),
        times_off=np.array(t_off),
        lo_off_mhz=LO_OFF_MHZ,
        n_dumps=n_dumps,
    )
    return path


def target_selector():
    """Return galactic (l=180, b=0) pointing if above minimum altitude, else None."""
    alt, az, ra, dec, _ = compute_gal_pointing(
        GAL_L_DEG, GAL_B_DEG,
        lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
    )
    if alt >= MIN_ALT_DEG:
        return f'gal_{GAL_L_DEG:.0f}_{GAL_B_DEG:.0f}', alt, az, ra, dec
    return None


def on_save(path, dump):
    lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
        f'{lo_tag}  → {path}'
    )


def main():
    print(f'Lab 4 — Leuschner 21 cm HI observation  (l={GAL_L_DEG}, b={GAL_B_DEG})')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.\n')

    # --- Noise diode calibration ---
    print(f'Collecting noise diode calibration ({CAL_DUMPS} dumps per LO) ...')
    cal_path = collect_noise_cal(sdrs, noise)
    print(f'  Calibration saved → {cal_path}\n')

    # --- Frequency-switching science streaming ---
    read_fn = make_sdr_reader(
        sdrs,
        nsamples=NSAMPLES,
        nblocks=NBLOCKS,
        nfft=NFFT,
        lo_freqs_mhz=(LO_ON_MHZ, LO_OFF_MHZ),
    )
    capture = StreamingCapture(
        telescope=telescope,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=OUTDIR,
        n_writers=2,
        repoint_interval_sec=60.0,
        on_save=on_save,
    )
    capture.run()


if __name__ == '__main__':
    main()
