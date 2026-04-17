#!/usr/bin/env python3
"""Lab 4 - Leuschner 21 cm HI observation of M31.

Dual-polarisation SDR capture with on-board FFT and correlation.
Frequency-switched between two LO settings for bandpass removal.

Unified calibration + science pipeline:
  1. Slew to target (handled by StreamingCapture pointing thread)
  2. Noise diode ON - CAL_DUMPS dumps per LO
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
    compute_radec_pointing,
)
from ugradiolab.capture import StreamingCapture
from ugradiolab.capture.readers import make_calibrated_sdr_reader

# ---------------------------------------------------------------------------
# Source catalog (J2000 equatorial, SIMBAD)
# ---------------------------------------------------------------------------

M31_RA_DEG  = 10.6847   # 00h 42m 44.3s
M31_DEC_DEG = 41.2687   # +41d 16' 07"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LO_FREQS_MHZ = (1420.0, 1421.0, 1422.0, 1423.0)  # 4 LOs: MW + M31 (v ~ 0 to −550 km/s)
SAMPLE_RATE  = 2.56e6    # Hz
NSAMPLES     = 32768
NBLOCKS      = 1025      # block 0 discarded -> 1024 valid
NFFT         = 1024
# Leuschner hardware: alt 15-85, az 5-350. Conservative guard: +2 deg margin.
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN_DEG   =  7.0
AZ_MAX_DEG   = 348.0
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
        center_freq=LO_FREQS_MHZ[0] * 1e6,
        sample_rate=SAMPLE_RATE, gain=0.0,
    )
    sdr_1 = SDR(
        device_index=1, direct=False,
        center_freq=LO_FREQS_MHZ[0] * 1e6,
        sample_rate=SAMPLE_RATE, gain=0.0,
    )
    return telescope, [sdr_0, sdr_1], noise


def target_selector():
    """Return M31 pointing if within telescope limits, else None."""
    alt, az, _ = compute_radec_pointing(
        M31_RA_DEG, M31_DEC_DEG,
        lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
    )
    if (MIN_ALT_DEG <= alt <= MAX_ALT_DEG and AZ_MIN_DEG <= az <= AZ_MAX_DEG):
        return 'm31', alt, az, M31_RA_DEG, M31_DEC_DEG
    return None


def on_save(path, dump):
    noise_tag = ' [CAL]' if dump.get('noise_on') else ''
    lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
        f'{lo_tag}{noise_tag}  -> {path}'
    )


def main():
    print(f'Lab 4 - Leuschner 21 cm HI observation  (M31)')
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
        lo_freqs_mhz=LO_FREQS_MHZ,
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
