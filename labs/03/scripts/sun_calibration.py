#!/usr/bin/env python3
"""Lab 3 Phase 1 — Sun fringe calibration.

Tracks the Sun over a 30-minute window without delay-line compensation.
The raw fringe pattern as a function of hour angle is used to fit the
interferometer baseline (BASELINE_EW_M, BASELINE_NS_M).  The fitted values
go into sun_observe.py for Phase 2.

Baseline estimate: 10–20 m.  At X-band (~10 GHz) this gives a fringe period
of roughly 21–41 seconds (depending on the Sun's declination), so 10 s
captures resolve 2–4 points per fringe cycle.  Over 30 minutes (~1800 s) you
accumulate 44–86 fringe cycles — more than enough for a reliable fit.

Pointing offset note
--------------------
The two antennas may not be perfectly co-pointed; an offset of up to ~1° is
expected.  This has two effects:

  1. Reduced fringe amplitude — a 1° separation across a ~20° primary beam
     costs only ~5 % in visibility, so fringes remain clearly detectable.

  2. Constant phase offset — the geometric delay τ_g is computed from the
     commanded pointing direction, not the true per-antenna pointing.  Any
     residual pointing error between the two antennas introduces a slowly
     varying (and partly constant) additive phase term φ_off that cannot be
     removed by the geometric delay alone.

Consequence for fitting: include a free constant phase offset φ₀ in the
fringe model (see "Next steps" below).

RF chain (NCH X-band interferometer):
  two X-band horn antennas → coaxial combiner → SNAP digital correlator

Usage:
    python sun_calibration.py

Output:
    data/lab03/sun_calibration/sun_cal_corr_<timestamp>.npz  (one file per capture)

Fit baseline from the output in a notebook by modelling fringe phase:
    φ(t) = 2π f₀ (B_ew / c) cos(dec) sin(ha(t)) + φ₀
where φ₀ absorbs pointing-offset and instrumental phase.
"""

import math
import sys
import time

import ugradio.interf as interf

# NOTE: confirm snap_spec import path before running
# import snap_spec

from ugradiolab import SunExperiment, compute_sun_pointing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTDIR      = 'data/lab03/sun_calibration'
MIN_ALT_DEG = 15.0   # elevation floor; abort below this

DURATION_SEC = 10.0  # integration time per SNAP capture
OBS_WINDOW_SEC = 30 * 60  # 30-minute observation window
N_CAPTURES = round(OBS_WINDOW_SEC / DURATION_SEC)  # = 180

# Estimated baseline range (metres) — used only for pre-run diagnostics.
# Exact value is *fitted* from the fringe data; Phase 2 uses that fitted value.
BASELINE_EW_EST_MIN_M = 10.0
BASELINE_EW_EST_MAX_M = 20.0

# Expected inter-antenna pointing offset (degrees).
# Used to estimate fringe visibility reduction in the pre-run summary.
ANTENNA_POINTING_OFFSET_DEG = 1.0

# Approximate X-band primary beam FWHM (degrees).
# A 10 GHz horn with ~30 cm aperture → FWHM ~ 15–20°; conservative estimate.
PRIMARY_BEAM_FWHM_DEG = 20.0

# Observing frequency (Hz) — NCH X-band interferometer centre frequency.
OBS_FREQ_HZ = 10.0e9

# ---------------------------------------------------------------------------


def _fringe_period_sec(baseline_m: float, dec_deg: float, freq_hz: float) -> float:
    """Expected fringe period (seconds) for a given EW baseline and Sun declination."""
    lam = 3e8 / freq_hz                     # wavelength (m)
    omega_earth = 2 * math.pi / 86164.0     # sidereal rate (rad/s)
    cos_dec = math.cos(math.radians(dec_deg))
    if cos_dec < 1e-6:
        return float('inf')
    return lam / (baseline_m * cos_dec * omega_earth)


def _visibility_factor(offset_deg: float, beam_fwhm_deg: float) -> float:
    """Approximate fringe visibility reduction from inter-antenna pointing offset.

    Models each antenna beam as a Gaussian.  The cross-correlation visibility
    for a source at the beam centre of antenna 1 but offset by `offset_deg`
    from antenna 2's beam centre is:

        V = exp(-offset_deg² / (4 * sigma²))

    where sigma = FWHM / (2 * sqrt(2 * ln 2)).
    """
    sigma_deg = beam_fwhm_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return math.exp(-(offset_deg ** 2) / (4.0 * sigma_deg ** 2))


def check_sun_visible(min_alt_deg):
    """Print current Sun position and return (alt, az, dec).  Exits if below horizon."""
    alt, az, ra, dec, jd = compute_sun_pointing()
    print(f'  Sun position:  Alt = {alt:.2f}°,  Az = {az:.2f}°')
    print(f'  Equatorial  :  RA = {ra:.4f}°,  Dec = {dec:.4f}°')
    print(f'  Julian date :  {jd:.5f}')
    print()
    if alt < min_alt_deg:
        print(f'  ERROR: Sun is only {alt:.1f}° above the horizon '
              f'(minimum: {min_alt_deg}°).')
        print('  Wait until the Sun rises higher and re-run.')
        sys.exit(1)
    return alt, az, dec


def print_observation_plan(dec_deg: float) -> None:
    """Print a diagnostic summary of the planned observation."""
    t_min = _fringe_period_sec(BASELINE_EW_EST_MIN_M, dec_deg, OBS_FREQ_HZ)
    t_max = _fringe_period_sec(BASELINE_EW_EST_MAX_M, dec_deg, OBS_FREQ_HZ)
    n_cycles_min = OBS_WINDOW_SEC / t_min
    n_cycles_max = OBS_WINDOW_SEC / t_max

    vis = _visibility_factor(ANTENNA_POINTING_OFFSET_DEG, PRIMARY_BEAM_FWHM_DEG)

    print('  Observation plan')
    print(f'    Duration       : {N_CAPTURES} × {DURATION_SEC:.0f}s = {OBS_WINDOW_SEC / 60:.0f} min')
    print(f'    Baseline range : {BASELINE_EW_EST_MIN_M:.0f}–{BASELINE_EW_EST_MAX_M:.0f} m (estimate; fitted from data)')
    print(f'    Fringe period  : {t_max:.0f}–{t_min:.0f}s  (at Dec = {dec_deg:.1f}°)')
    print(f'    Fringe cycles  : ~{n_cycles_max:.0f}–{n_cycles_min:.0f} over {OBS_WINDOW_SEC / 60:.0f} min')
    print(f'    Points/cycle   : ~{DURATION_SEC / t_max:.1f}–{DURATION_SEC / t_min:.1f} (10 s captures)')
    print()
    print(f'  Pointing offset : {ANTENNA_POINTING_OFFSET_DEG:.1f}° between antennas (expected)')
    print(f'  Beam FWHM est.  : {PRIMARY_BEAM_FWHM_DEG:.0f}°')
    print(f'  Visibility loss : {(1.0 - vis) * 100:.1f}%  '
          f'(fringes remain detectable; φ₀ absorbs residual phase)')
    print()
    print('  No delay-line compensation — recording raw fringes for baseline fit.')
    print()


def main():
    print('Lab 3 Phase 1 — Sun fringe calibration')
    print('=' * 50)
    print()
    print('Checking Sun position ...')
    print()
    alt, az, dec = check_sun_visible(MIN_ALT_DEG)

    print_observation_plan(dec)

    input('  Connect hardware, then press Enter to begin: ')
    print()

    # --- Hardware setup ---
    interferometer = interf.Interferometer()
    # snap = snap_spec.Snap(...)   # TODO: confirm snap_spec API

    # --- Capture loop (back-to-back, no sleep) ---
    paths  = []
    t0     = time.time()

    for i in range(N_CAPTURES):
        exp = SunExperiment(
            interferometer = interferometer,
            snap           = None,  # replace None with snap object
            duration_sec   = DURATION_SEC,
            outdir         = OUTDIR,
            prefix         = f'sun-cal-{i:03d}',
            # baseline_ew_m=None (default) → no delay applied in Phase 1
        )

        print(f'[{i + 1:3d}/{N_CAPTURES}] ', end='', flush=True)
        path = exp.run()
        paths.append(path)
        elapsed = time.time() - t0
        print(f'Alt={exp.alt_deg:.2f}°  Az={exp.az_deg:.2f}°  t={elapsed:.0f}s  → {path}')

    total_elapsed = time.time() - t0
    print()
    print(f'Done. {len(paths)} files saved to {OUTDIR}/ in {total_elapsed:.0f}s')
    print()
    print('Next steps:')
    print('  1. Load the .npz files in a notebook and extract fringe phase vs time.')
    print()
    print('  2. Convert capture timestamps to hour angle ha(t) using LST and Sun RA.')
    print()
    print('  3. Fit the fringe phase with a 3-parameter model:')
    print('       φ(t) = 2π f₀ (B_ew / c) cos(dec) sin(ha(t)) + φ₀')
    print()
    print('     φ₀ is a free constant phase offset that absorbs:')
    print('       - Instrumental phase (cable length difference, LO offsets)')
    print('       - Residual phase from the ~1° inter-antenna pointing offset')
    print('     Do NOT force φ₀ = 0; fitting it is essential with uncalibrated pointing.')
    print()
    print('  4. Optionally extend to a 2D fit if NS baseline is non-negligible:')
    print('       φ(t) = 2π f₀ [(B_ew/c) cos(dec) sin(ha(t))')
    print('                    + (B_ns/c) (sin(dec) cos(lat) - cos(dec) sin(lat) cos(ha(t)))]')
    print('              + φ₀')
    print()
    print(f'  5. Expected B_ew range: {BASELINE_EW_EST_MIN_M:.0f}–{BASELINE_EW_EST_MAX_M:.0f} m.')
    print('     Copy the fitted BASELINE_EW_M (and BASELINE_NS_M if non-zero)')
    print('     into labs/03/scripts/sun_observe.py for Phase 2.')


if __name__ == '__main__':
    main()
