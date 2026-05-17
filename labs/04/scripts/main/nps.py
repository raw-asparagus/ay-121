#!/usr/bin/env python3
"""Lab 4 - Leuschner Sky Survey -- North Polar Spur.

NPS pass driven by the synchronous SurveyCoordinator.  Single-shot --
intended to be run as a time-boxed first stage before launching the
galactic-plane loop in ``main``.

L_CENTER=120 sits outside [L_MIN, L_MAX] and is kept that way on
purpose: anchoring the cos(b)-corrected longitude grid at the same
phase as the galactic-plane survey keeps the brick-interleave layout
mutually consistent.  ``_build_l_row`` filters the row to the survey
window after expansion.

Run from this directory:
    PYTHONPATH=../../.. python3 nps.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _survey import RecalTarget, SurveyConfig, run  # noqa: E402


CONFIG = SurveyConfig(
    title='North Polar Spur',
    output_dir='data/nps',
    artifacts_prefix='nps',
    timing_archive_dir='data/archive/nps',

    f1_mhz=1419.86,
    f2_mhz=1421.14,
    sample_rate=3.2e6,
    nsamples=16384,    # 16 chunks * 1024 channels (halved for faster cadence)
    nblocks=1025,
    nfft=1024,

    min_alt_deg=17.0,
    max_alt_deg=83.0,
    az_min=7.0,
    az_max=348.0,
    track_interval_s=10.0,

    # Full-sky scan, anchored on main's l_center for brick-interleave
    # consistency.  Forward-sim filter drops unreachable cells at runtime.
    l_center=120.0,
    l_min=-60.0,
    l_max=300.0,
    b_min=-20,
    b_max=20,
    b_step=2,
    physical_spacing_deg=2.0,

    # ABBA ABBA ABBA: 4 cal then 8 obs.
    cal_dumps_per_lo=2,
    obs_dumps_per_lo=4,

    recal_enable=True,
    recal_targets=(
        RecalTarget('obs_recal_drift',    180.0, 72.0),
        RecalTarget('obs_recal_drift_bk',  90.0, 72.0),
    ),
    recal_every_n_cells=10,

    phases=('even',),
)


if __name__ == '__main__':
    run(CONFIG)
