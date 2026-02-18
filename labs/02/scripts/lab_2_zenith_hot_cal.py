#!/usr/bin/env python3
"""Lab 2 Y-factor calibration — blackbody hot load at zenith, two gain settings.

Horn points straight up (alt=90°) throughout so the cold-sky baseline
sees the CMB + atmosphere (~10 K), giving a usable Y-factor contrast
with the human-body hot load (~310 K).

Two complete Y-factor sequences are run back-to-back, one at gain=0 and
one at gain=20.  Each sequence is self-contained with its own cold-sky
pre- and post-baselines so they can be analysed independently.

Sequence (24 steps total):
  --- Gain = 0 ---
  1.  COLD-SKY-PRE-G00   horn up, aperture open   → s_cold (gain 0)
  --- Enter: fill the aperture ---
  2.  HOT-LOAD-G00-01    horn up, aperture covered
  ...
  11. HOT-LOAD-G00-10    horn up, aperture covered
  --- Enter: step away ---
  12. COLD-SKY-POST-G00  horn up, aperture open   → drift check (gain 0)

  --- Gain = 20 ---
  13. COLD-SKY-PRE-G20   horn up, aperture open   → s_cold (gain 20)
  --- Enter: fill the aperture ---
  14. HOT-LOAD-G20-01    horn up, aperture covered
  ...
  23. HOT-LOAD-G20-10    horn up, aperture covered
  --- Enter: step away ---
  24. COLD-SKY-POST-G20  horn up, aperture open   → drift check (gain 20)

Usage:
    python lab_2_zenith_hot_cal.py
"""

from ugradio.sdr import SDR
from ugradiolab.experiment import ObsExperiment

# ---------------------------------------------------------------------------
OUTDIR = 'data/lab2_zenith_hot_cal'
N_HOT = 10
GAINS = [0.0, 20.0]

SDR_BASE = dict(
    outdir=OUTDIR,
    nsamples=32768,
    nblocks=10,
    direct=False,
    center_freq=1420e6,
    sample_rate=2.56e6,
    alt_deg=90.0,
    az_deg=0.0,
)

STEPS_PER_GAIN = 1 + N_HOT + 1      # cold_pre + hot_loads + cold_post
TOTAL = STEPS_PER_GAIN * len(GAINS)


def _build_gain_set(gain):
    """Return (cold_pre, hot_loads, cold_post) experiments for one gain value."""
    tag = f'G{int(gain):02d}'
    common = dict(**SDR_BASE, gain=gain)
    cold_pre = ObsExperiment(prefix=f'COLD-SKY-PRE-{tag}', **common)
    hot_loads = [
        ObsExperiment(prefix=f'HOT-LOAD-{tag}-{i + 1:02d}', **common)
        for i in range(N_HOT)
    ]
    cold_post = ObsExperiment(prefix=f'COLD-SKY-POST-{tag}', **common)
    return cold_pre, hot_loads, cold_post


def _run_gain_set(sdr, gain, step_offset):
    """Run one complete Y-factor sequence; returns final step number."""
    cold_pre, hot_loads, cold_post = _build_gain_set(gain)
    tag = f'G{int(gain):02d}'
    step = step_offset

    print(f'--- Gain = {int(gain)} dB ---')
    print()

    # Cold-sky pre-baseline
    step += 1
    print(f'[{step}/{TOTAL}] {cold_pre.prefix}  (cold sky, aperture open)')
    print(f'  -> {cold_pre.run(sdr)}')
    print()

    input(f'  [{tag}] Fill the aperture from above, then press Enter to begin hot-load captures: ')
    print()

    # Hot-load captures
    for i, exp in enumerate(hot_loads):
        step += 1
        print(f'[{step}/{TOTAL}] {exp.prefix}  (aperture covered)')
        print(f'  -> {exp.run(sdr)}')

    print()
    input(f'  [{tag}] Step away from the aperture, then press Enter to take the post-baseline: ')
    print()

    # Cold-sky post-baseline
    step += 1
    print(f'[{step}/{TOTAL}] {cold_post.prefix}  (cold sky, aperture open)')
    print(f'  -> {cold_post.run(sdr)}')
    print()

    return step


def main():
    print(f'Lab 2 Y-factor calibration (zenith): {TOTAL} steps across {len(GAINS)} gain settings')
    print(f'Output: {OUTDIR}/')
    print()

    sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)

    try:
        step = 0
        for gain in GAINS:
            step = _run_gain_set(sdr, gain, step)
    finally:
        sdr.close()

    print(f'Done. {TOTAL} files saved to {OUTDIR}/')


if __name__ == '__main__':
    main()
