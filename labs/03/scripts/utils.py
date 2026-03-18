"""Shared helpers for Lab 3 interferometer calibration scripts."""

import math

import ugradio.interf as interf
import ugradio.nch as nch
from snap_spec.snap import UGRadioSnap


def lst_deg(jd):
    """Return Local Sidereal Time in degrees for the NCH site."""
    T = (jd - 2451545.0) / 36525.0
    g = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
         + T ** 2 * 0.000387933 - T ** 3 / 38710000.0)
    return (g + nch.lon) % 360.0


def optimal_duration(ha_deg, dec_deg, baseline_m, target_phase_deg,
                     obs_freq_hz=9.915e9):
    """Return capture duration (s) giving target_phase_deg of fringe phase advance.

    Uses the instantaneous fringe rate:
        rate = f_RF * (B_ew/c) * cos(dec) * ω_Earth * |cos(HA)|   [cycles/s]
        τ    = (target_phase_deg / 360) / rate
    Clamped to [5, 60] s.
    """
    omega_e = 2 * math.pi / 86164.0
    rate = (obs_freq_hz * baseline_m / 299792458.0
            * math.cos(math.radians(dec_deg))
            * omega_e
            * abs(math.cos(math.radians(ha_deg))))
    if rate < 1e-12:
        return 60.0
    tau = (target_phase_deg / 360.0) / rate
    return max(2.0, min(60.0, tau))


def setup_hardware(snap_retries=5):
    """Initialise interferometer and SNAP correlator.  Returns (interferometer, snap).

    snap.initialize() calls align_adc(), which is non-deterministic: the ADC
    ramp test occasionally fails on the first attempt.  snap_retries controls
    how many times initialization is retried before raising.
    """
    interferometer = interf.Interferometer()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    for attempt in range(1, snap_retries + 1):
        try:
            snap.initialize(mode='corr', sample_rate=500, force=True)
            snap.input.use_adc()
            if attempt > 1:
                print(f'  SNAP initialized on attempt {attempt}.')
            return interferometer, snap
        except AssertionError as exc:
            print(f'  SNAP init attempt {attempt}/{snap_retries} failed ({exc}), retrying...')
    raise RuntimeError(f'SNAP initialization failed after {snap_retries} attempts.')


def reinit_snap(snap, retries=5):
    """Reinitialise the SNAP correlator after a capture failure (force=True).

    Retries up to *retries* times because align_adc() is non-deterministic.
    Raises RuntimeError if all attempts fail so the caller knows the SNAP is
    in an unusable state (rather than silently collecting corrupted data).
    """
    for attempt in range(1, retries + 1):
        try:
            snap.initialize(mode='corr', sample_rate=500, force=True)
            snap.input.use_adc()
            if attempt > 1:
                print(f'  SNAP re-initialized on attempt {attempt}.')
            return
        except Exception as exc:
            print(f'  SNAP reinit attempt {attempt}/{retries} failed ({exc}), retrying...')
    raise RuntimeError(f'SNAP re-initialization failed after {retries} attempts.')
