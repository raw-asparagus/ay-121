#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI observation.

Dual-polarisation SDR capture with on-board FFT and correlation.
Frequency-switched between two LO settings for bandpass removal.

Unified calibration + science pipeline:
  1. Slew to target (handled by StreamingCapture pointing thread)
  2. Noise diode ON  - 32 dumps at LO = 1420 MHz, 32 at LO = 1421 MHz
  3. Noise diode OFF - frequency-switching science loop (1420 / 1421 alternating)

All dumps (calibration and science) flow through the same reader/writer
pipeline.  Calibration dumps are tagged with noise_on=True.

Usage:
    python observe.py

Output:
    data/lab04/streaming/<target>/<target>_dump_<timestamp>_<seq>.npz
"""

from ugradiolab.astronomy import (
    LEO_LAT_DEG,
    LEO_LON_DEG,
    LEO_OBS_ALT_M,
    compute_gal_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_calibrated_sdr_reader

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
NBLOCKS      = 2049      # block 0 discarded -> 2048 valid
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
    noise_tag = ' [CAL]' if dump.get('noise_on') else ''
    lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
        f'{lo_tag}{noise_tag}  -> {path}'
    )


def main():
    print(f'Lab 4 - Leuschner 21 cm HI observation  (l={GAL_L_DEG}, b={GAL_B_DEG})')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    telescope, sdrs, noise = setup_hardware()
    print('Hardware ready.\n')

    # --- Unified calibration + science reader ---
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
        repoint_interval_sec=60.0,
        on_save=on_save,
    )
    capture.run()


if __name__ == '__main__':
    main()
