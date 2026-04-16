#!/usr/bin/env python3
"""Lab 4 - Single scan of M31 with cal OFF at two LO settings.

Two-frequency observation (1420.33 / 1419.66 MHz) for frequency-switched
bandpass removal centred on the Milky Way HI line.  No noise diode.

Usage:
    python observe_m31_cal_off.py

Output:
    data/lab04/streaming/m31_cal_off/m31_cal_off_dump_<timestamp>_<seq>.npz
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

LO_FREQS_MHZ = (1420.33, 1419.66)  # symmetric about HI rest (1420.405 MHz)
SAMPLE_RATE  = 2.56e6    # Hz
NSAMPLES     = 32768
NBLOCKS      = 1025      # block 0 discarded -> 1024 valid
NFFT         = 1024
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN_DEG   =  7.0
AZ_MAX_DEG   = 348.0
CAL_DUMPS    = 0         # no noise diode calibration
OUTDIR       = 'data/lab04/streaming'


# ---------------------------------------------------------------------------

def setup_hardware():
    """Initialise telescope and SDRs (no noise diode)."""
    from ugradio.leusch import LeuschTelescope
    from ugradio.sdr import SDR

    telescope = LeuschTelescope()

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
    return telescope, [sdr_0, sdr_1]


def target_selector():
    """Return M31 pointing if within telescope limits, else None."""
    alt, az, _ = compute_radec_pointing(
        M31_RA_DEG, M31_DEC_DEG,
        lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
    )
    if (MIN_ALT_DEG <= alt <= MAX_ALT_DEG and AZ_MIN_DEG <= az <= AZ_MAX_DEG):
        return 'm31_cal_off', alt, az, M31_RA_DEG, M31_DEC_DEG
    return None


def on_save(path, dump):
    lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    print(
        f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}'
        f'{lo_tag}  -> {path}'
    )


def main():
    print('Lab 4 - M31 cal-off scan  (1420.33 / 1419.66 MHz)')
    print('=' * 60)
    print()
    print('Initialising hardware ...')
    telescope, sdrs = setup_hardware()
    print('Hardware ready.\n')

    read_fn = make_calibrated_sdr_reader(
        sdrs,
        None,  # no noise diode
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
