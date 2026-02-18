#!/usr/bin/env python3
"""Lab 2 galactic-plane frequency-switched observation — §4.

Computes the current alt/az for galactic (l=120°, b=0°) at NCH using
NTP time, prints pointing instructions, then runs a frequency-switched
observation sequence after operator confirmation.

The same two-LO frequency-switching scheme as the zenith observation is
used so the resulting files are analysed identically.

  LO-A: center_freq = 1420.0 MHz  →  HI line at +0.406 MHz (upper half)
  LO-B: center_freq = 1421.0 MHz  →  HI line at −0.594 MHz (lower half)

Usage:
    python lab_2_galactic_obs.py
"""

import sys

import astropy.coordinates as ac
import astropy.units as u
import ntplib
import ugradio.coord as coord
import ugradio.nch as nch
import ugradio.timing as timing
from ugradio.sdr import SDR

from ugradiolab.experiment import ObsExperiment
from ugradiolab.queue import QueueRunner

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_galactic_obs'
N_PAIRS = 200

GAL_L = 120.0   # degrees
GAL_B = 0.0     # degrees

LO_A_FREQ = 1420.0e6   # HI line lands at +0.406 MHz in baseband
LO_B_FREQ = 1421.0e6   # HI line lands at −0.594 MHz in baseband

MIN_ALT_DEG = 10.0     # elevation floor; warn below this

COMMON = dict(
    outdir=OUTDIR,
    nsamples=32768,
    nblocks=10,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
)


# ---------------------------------------------------------------------------

def _get_unix_time():
    """NTP time with system-time fallback."""
    try:
        return ntplib.NTPClient().request('pool.ntp.org', version=3).tx_time
    except Exception:
        return timing.unix_time()


def compute_pointing():
    """Return (alt_deg, az_deg, ra_deg, dec_deg, jd) for (l=120°, b=0°) now."""
    unix_t = _get_unix_time()
    jd = timing.julian_date(unix_t)

    gc = ac.SkyCoord(l=GAL_L * u.deg, b=GAL_B * u.deg, frame='galactic')
    ra = gc.icrs.ra.deg
    dec = gc.icrs.dec.deg

    alt, az = coord.get_altaz(ra, dec, jd=jd, lat=nch.lat, lon=nch.lon, alt=nch.alt)
    return alt, az, ra, dec, jd


def build_plan(alt_deg, az_deg):
    """Build interleaved [A, B, A, B, ...] experiment list."""
    pointing = dict(alt_deg=alt_deg, az_deg=az_deg)
    experiments = []
    for _ in range(N_PAIRS):
        experiments.append(ObsExperiment(prefix='GAL-A', center_freq=LO_A_FREQ, **pointing, **COMMON))
        experiments.append(ObsExperiment(prefix='GAL-B', center_freq=LO_B_FREQ, **pointing, **COMMON))
    return experiments


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

    print(f'Starting {total} captures ({N_PAIRS} pairs)...')
    print(f'  LO-A: {LO_A_FREQ / 1e6:.1f} MHz  →  HI line at +0.406 MHz')
    print(f'  LO-B: {LO_B_FREQ / 1e6:.1f} MHz  →  HI line at -0.594 MHz')
    print(f'  Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=LO_A_FREQ, sample_rate=2.56e6, gain=0.0)

    try:
        runner = QueueRunner(experiments=experiments, sdr=sdr, confirm=False)
        paths = runner.run()
    finally:
        sdr.close()

    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
