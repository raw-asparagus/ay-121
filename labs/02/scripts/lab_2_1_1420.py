#!/usr/bin/env python3
"""Lab 2 galactic-plane observation @ 1420 MHz.

Usage:
    python lab_2_1_vary_n.py
"""

import sys
import time

import astropy.coordinates as ac
import astropy.units as u
import ugradio.coord as coord
import ugradio.nch as nch
import ugradio.timing as timing
from ugradio.sdr import SDR

from ugradiolab import SignalGenerator
from ugradiolab.experiment import CalExperiment, ObsExperiment
from ugradiolab.queue import QueueRunner
from ugradiolab.utils import get_unix_time

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_1_1420'

GAL_L = 120.0   # degrees
GAL_B = 0.0     # degrees

LO_FREQ = 1420.0e6
NBLOCKS_STEPS = [2, 8, 32, 128, 512, 2048]

MIN_ALT_DEG = 10.0     # elevation floor; warn below this

SIGGEN_FREQ_MHZ = 1420.405751768
SIGGEN_AMP_DBM = -80.0

COMMON = dict(
    outdir=OUTDIR,
    nsamples=16384,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
)


# ---------------------------------------------------------------------------

def compute_pointing():
    """Return (alt_deg, az_deg, ra_deg, dec_deg, jd) for (l=120°, b=0°) now."""
    unix_t = get_unix_time()
    jd = timing.julian_date(unix_t)

    gc = ac.SkyCoord(l=GAL_L * u.deg, b=GAL_B * u.deg, frame='galactic')
    ra = gc.icrs.ra.deg
    dec = gc.icrs.dec.deg

    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=nch.lat, lon=nch.lon, alt=nch.alt)
    return alt, az, ra, dec, jd


def build_plan(alt_deg, az_deg):
    """Build one ObsExperiment per nblocks step."""
    pointing = dict(alt_deg=alt_deg, az_deg=az_deg)
    return [
        ObsExperiment(prefix=f'GAL-1420-n{n}', center_freq=LO_FREQ, nblocks=n, **pointing, **COMMON)
        for n in NBLOCKS_STEPS
    ]


def main():
    print('Lab 2 galactic observation — computing pointing for (l=120°, b=0°) ...')
    print()

    alt, az, ra, dec, jd = compute_pointing()

    print(f'  Galactic        :  l = {GAL_L:.1f}°,  b = {GAL_B:.1f}°')
    print(f'  Equatorial J2000:  RA = {ra:.4f}°,  Dec = {dec:.4f}°')
    print(f'  Local alt/az    :  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Julian date     :  {jd:.5f}')
    print()

    if alt < MIN_ALT_DEG:
        print(f'  WARNING: target is only {alt:.1f}° above the horizon '
              f'(minimum recommended: {MIN_ALT_DEG}°).')
        print('  Consider waiting until the target rises or choose a different LST.')
        print()
        cont = input('  Continue anyway? [y/N] ').strip().lower()
        if cont != 'y':
            print('Aborted.')
            sys.exit(0)
        print()

    print(f'  >>> Point the horn to:  Alt = {alt:.2f}°,  Az = {az:.2f}° <<<')
    print()
    input('  Press Enter once the horn is pointed and you are ready to begin: ')
    print()

    experiments = build_plan(alt, az)
    total = len(experiments)

    print(f'Starting {total} captures (nblocks steps: {NBLOCKS_STEPS})...')
    print(f'  LO:   {LO_FREQ / 1e6:.1f} MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=LO_FREQ, sample_rate=2.56e6, gain=0.0)

    try:
        runner = QueueRunner(experiments=experiments, sdr=sdr, confirm=False)
        t0 = time.time()
        paths = runner.run()
        elapsed = time.time() - t0
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
