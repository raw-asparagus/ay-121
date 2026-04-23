#!/usr/bin/env python3
"""Lab 4 - Fill missing dumps at (l=112, b=+1).

Stare observation to complete coverage at this pointing.
Collects 3 more dumps to reach 8 total (4 per band).

Usage:
    python observe_fill_112_1.py

Output:
    data/lab04/streaming/session_{NNN}/obs_112_1/obs_112_1_{timestamp}.npz
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
# Target: (l=112, b=+1)
# ---------------------------------------------------------------------------

GAL_L = 112.0
GAL_B =   1.0

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LO_FREQS_MHZ = (1420.0, 1421.0)
SAMPLE_RATE  = 2.56e6
NSAMPLES     = 32768
NBLOCKS      = 1025
NFFT         = 1024
MIN_ALT_DEG  = 17.0
MAX_ALT_DEG  = 83.0
AZ_MIN_DEG   =  7.0
AZ_MAX_DEG   = 348.0
import glob

CAL_DUMPS    = 0
DUMPS_TO_COLLECT = 3
STREAMING_DIR = 'data/lab04/streaming'


def _next_session_dir() -> str:
    existing = sorted(glob.glob(f'{STREAMING_DIR}/session_???'))
    n = len(existing) + 1
    return f'{STREAMING_DIR}/session_{n:03d}'

# ---------------------------------------------------------------------------

import threading


def setup_hardware():
    from ugradio.leusch import LeuschTelescope
    from ugradio.sdr import SDR

    telescope = LeuschTelescope()
    sdr_0 = SDR(device_index=0, direct=False,
                center_freq=LO_FREQS_MHZ[0] * 1e6,
                sample_rate=SAMPLE_RATE, gain=0.0)
    sdr_1 = SDR(device_index=1, direct=False,
                center_freq=LO_FREQS_MHZ[0] * 1e6,
                sample_rate=SAMPLE_RATE, gain=0.0)
    return telescope, [sdr_0, sdr_1]


def make_counting_target_selector(max_dumps):
    """Target selector that stops after max_dumps."""
    lock = threading.Lock()
    dump_count = 0
    done_event = threading.Event()

    def dump_notifier():
        nonlocal dump_count
        with lock:
            dump_count += 1
            if dump_count >= max_dumps:
                done_event.set()

    def target_selector():
        if done_event.is_set():
            return None
        alt, az, ra, dec, _ = compute_gal_pointing(
            GAL_L, GAL_B,
            lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
        )
        if (MIN_ALT_DEG <= alt <= MAX_ALT_DEG and
                AZ_MIN_DEG <= az <= AZ_MAX_DEG):
            return 'obs_112_1', alt, az, ra, dec
        return None

    return target_selector, dump_notifier, done_event


def on_save(path, dump, notifier):
    lo_tag = f'  LO={dump["lo_freq_mhz"]}' if 'lo_freq_mhz' in dump else ''
    print(f'  [{dump["target_name"]}]  seq={dump["seq"]:05d}{lo_tag}  -> {path}')
    notifier()


def main():
    print(f'Lab 4 - Fill (l={GAL_L}, b={GAL_B}): {DUMPS_TO_COLLECT} dumps')
    print('=' * 50)

    alt, az, _, _, _ = compute_gal_pointing(
        GAL_L, GAL_B,
        lat=LEO_LAT_DEG, lon=LEO_LON_DEG, obs_alt=LEO_OBS_ALT_M,
    )
    print(f'  Current alt={alt:.1f}, az={az:.1f}')

    print('\nInitialising hardware ...')
    telescope, sdrs = setup_hardware()
    print('Hardware ready.\n')

    target_selector, dump_notifier, done_event = make_counting_target_selector(
        DUMPS_TO_COLLECT)

    read_fn = make_calibrated_sdr_reader(
        sdrs,
        None,
        nsamples=NSAMPLES,
        nblocks=NBLOCKS,
        nfft=NFFT,
        lo_freqs_mhz=LO_FREQS_MHZ,
        cal_dumps_per_lo=CAL_DUMPS,
    )

    outdir = _next_session_dir()
    print(f'  Session dir: {outdir}')
    capture = StreamingCapture(
        telescope=telescope,
        read_fn=read_fn,
        target_selector=target_selector,
        outdir=outdir,
        n_writers=2,
        repoint_interval_sec=60.0,
        on_save=lambda path, dump: on_save(path, dump, dump_notifier),
    )
    capture.run(done_event=done_event)

    print('\n' + '=' * 50)
    print(f'  Done -- {DUMPS_TO_COLLECT} dumps collected at (l={GAL_L}, b={GAL_B})')
    print('=' * 50)


if __name__ == '__main__':
    main()
