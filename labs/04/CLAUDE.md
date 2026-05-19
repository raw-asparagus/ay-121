# Lab 4 -- 21 cm HI Survey: Claude Instructions

## Style Rules

Use ASCII-only text in this lab unless a Unicode character is required by a quoted source or upstream data. Do not introduce special symbols such as degree signs, arrows, em dashes, en dashes, or other Unicode punctuation in code, comments, docstrings, or notebooks.

## Repository layout

Notebooks are flat at the lab root (no `notebooks/` subdir). The streaming
pipeline and M31 stare analysis are retired; their data lives in
`data/backup/` and is not part of the active pipeline.

```
labs/04/
  main_scan_plan.ipynb               # forward-simulating planner for `scripts/main/main`
  main_scan_calibration.ipynb        # noise-diode drift calibration (recal pointings)
  main_scan_load.ipynb               # load + reduce science pointings -> T_B(l,b,v_LSR)
  main_scan_load_diagnostics.ipynb   # per-session per-cell T_sys diagnostics
  main_scan_qa.ipynb                 # neighbour-based QA on calibrated T_B
  ARCHITECTURE.md                    # high-level data flow (partially stale)
  PIPELINE.md                        # detailed stage-by-stage walkthrough of main_scan_load
  pipeline.md                        # legacy streaming-era summary (reference only)
  manifest.py                        # manifest read/write + completeness thresholds
  plotters.py                        # reusable plot functions
  artifacts/
    scan_load_state.pkl              # state handoff: cell_combined, qa_flagged_set, v_lsr_overlap, ...
  scripts/
    main/
      main                           # galactic-plane survey driver (executable, no .py)
      nps.py                         # North Polar Spur driver (single-shot first stage)
      _survey.py                     # shared SurveyConfig, scheduler, recal logic
    slew_gap_breakdown.py            # timing post-mortem helper
    trial_gap_breakdown.py           # timing post-mortem helper
  utils/
    cache.py freqswitch.py io.py lsr.py mapping.py qa.py rfi.py timing_stats.py
  data/
    main/session_NNN/                # 3.2 MHz pipelined galactic-plane sessions
    nps/session_NNN/                 # 3.2 MHz pipelined NPS sessions
    archive/{main,nps}/              # timing archives written by the survey driver
    backup/{main,streaming,...}/     # retired datasets (not loaded by active pipeline)
```

State handoff between notebooks is via `artifacts/scan_load_state.pkl`,
written by `main_scan_load.ipynb` and read by `main_scan_qa.ipynb` and
`main_scan_load_diagnostics.ipynb`. There is no `survey_manifest.json`,
`truncation_manifest.json`, or `reduced_survey.npz` checked into the tree
right now -- these are generated on demand if needed.

## Survey drivers

The active survey runs two driver scripts under `scripts/main/`, both
backed by `_survey.py` (which owns `SurveyConfig`, the brick-interleave
scheduler, recal injection, and the `SurveyCoordinator` glue).

### `scripts/main/main` -- galactic plane (loops forever)

- `output_dir = data/main`, `artifacts_prefix = main`
- LO pair 1419.86 / 1421.14 MHz, sample rate 3.2 MHz, NSAMPLES=16384
- Grid: `l in [-60, 300]`, `b in [-20, 20]`, `b_step = 2`,
  `physical_spacing_deg = 2.0`, brick interleave centered on `l = 120`
- Phases currently restricted to `('even',)` -- odd phase temporarily
  disabled at the top of `main`
- Per pointing: `cal_dumps_per_lo=2` (4 cal total) + `obs_dumps_per_lo=4`
  (8 obs total), ABBA-interleaved across cal/obs slots
- Periodic recal: two `RecalTarget`s at Dec=+72 (l=180 and l=90),
  injected every 10 cells

### `scripts/main/nps.py` -- North Polar Spur (single-shot)

- `output_dir = data/nps`, `artifacts_prefix = nps`
- Same SDR/LO config and per-pointing dump schedule as `main`
- Grid: `l in [210, 380]`, `b in [0, 70]`, anchored on `l_center = 120`
  so the cos(b)-corrected longitude phase matches the plane survey
- Intended to be run as a time-boxed first stage before launching the
  plane loop

Run from the script directory with `PYTHONPATH=../../.. python3 main`
(or `nps.py`).

## Grid geometry

Two-phase brick-pattern tessellation with `Delta_l = 2 / cos(b)` exact:

- Even phase: `b in {-6, -4, -2, 0, +2, +4, +6}`
- Odd phase:  `b in {-5, -3, -1, +1, +3, +5}`, longitude offset by half a step
- Scan order: columns of constant `l`, ascending `L_MIN -> L_MAX`,
  zig-zagging in `b` within each column

When tuning QA radii in `utils.qa`, the relevant neighbour ring for the
brick grid contains 8 ring-1 cells (4 same-axis at sep 2.0, 4 half-row
diagonals at sep ~sqrt(2)) -- use `NEIGHBOR_MAX_SEP_DEG = 2.1`.

## Pipeline cycle

1. **Plan** -- `main_scan_plan.ipynb` forward-simulates the driver loop
   using measured cadence and slew stats from `data/archive/main/`.
2. **Observe** -- run `scripts/main/main` (and/or `nps.py`) on the Pi.
   Each session is created automatically under `data/main/session_NNN/`
   (or `data/nps/`).
3. **Ingest** -- transfer sessions from the Pi to `data/main/` and clean
   up any Zone.Identifier files from Windows.
4. **Calibrate** -- run `main_scan_calibration.ipynb` for noise-diode
   drift QA on the recal pointings.
5. **Reduce** -- run `main_scan_load.ipynb` on science pointings
   (recal pointings are skipped via `skip_recal=True`). Writes
   `artifacts/scan_load_state.pkl`.
6. **QA + diagnostics** -- run `main_scan_qa.ipynb` and
   `main_scan_load_diagnostics.ipynb` against the state pickle.

`PIPELINE.md` is the authoritative stage-by-stage description of
`main_scan_load.ipynb` -- consult it before refactoring reduction code.
`ARCHITECTURE.md` still references the retired `notebooks/` layout in
several places; treat it as historical until it is refreshed.

## Data organisation

- Each survey driver session writes `data/{main,nps}/session_NNN/`
  (sequential, timestamp-ordered).
- Within each session: `obs_{l_name}_{b}/obs_*.npz` (science, noise-off)
  and `cal_{l_name}_{b}/cal_*.npz` (calibration, noise-on). Recal-drift
  pointings use `obs_recal_drift[_bk]_{...}/...` naming and are handled
  by `main_scan_calibration.ipynb`, not the science load.
- Galactic coordinates `(l, b)` are integer degrees (with `l_name`
  reflecting wrap-around for negative `l`).
- Each `.npz` carries `corr00`, `corr11` (1024-channel float64 power
  spectra), plus scalars `lo_freq_mhz`, `noise_on`, `time`,
  `target_name`, `alt_deg`, `az_deg`, `ra_deg`, `dec_deg`, `seq`.
- Calibration scope: **per-cell**. `cal_{l}_{b}` and `obs_{l}_{b}` at
  the same pointing are paired; each science cell has its own gain and
  `T_sys`. `cal_*/` contains noise-on dumps; the noise-off reference is
  the matching `obs_*/` at the same `(l, b)`.

## Frequency switching and calibration

- LO pair 1419.86 / 1421.14 MHz, 3.2 MHz sample rate, Stokes I =
  `corr00 + corr11`. The known pol-0 noise-diode coupling issue at
  3.2 MHz affects `T_cal` accuracy only; the dimensionless ratio
  `R = (I_LO1 - I_LO2)/I_LO2` carries no `T_cal` dependence, so both
  pols are summed for ~sqrt(2) SNR gain.
- Edge trim 256 kHz each side; the post-trim overlap defines the LSR
  velocity grid (`v_lsr_overlap`).
- `R = (I_LO1 - I_LO2) / I_LO2`, averaged over surviving dump pairs.
- `T_B = R * T_sys` per cell.
- The DC bin (channel 512 after fftshift) must be masked. For LO=1420
  data the two bins below DC are also masked to suppress leakage near
  the HI line.
- RFI flagging: sliding-window Chebyshev pseudo-continuum with sigma MAD
  threshold (`utils/rfi.py`); local extrema excluded from the fit so the
  baseline is not biased by RFI spikes.
- `main_scan_calibration.ipynb` runs an **anchor-free naive** noise-diode
  drift solve at the two recal pointings, assuming a single global
  `T_B_ASSUMED`. Output is diagnostic (visit-level `Tcal_pol(t)` traces);
  it does not feed back into the science load yet.

## Reduction QA (T_B-based)

All QA steps run on calibrated `T_B` (Kelvin), so integrated `W` is in
`K * km/s` and `noise_rms` is in K. This makes residual thresholds
physical and unbiased by cell-to-cell `T_sys`.

- **Cross-session pair filter** (`utils.flag_outlier_pairs`): builds
  per-pair `T_B_lsr = R_lsr * T_sys` and rejects pairs whose deviant-
  channel fraction exceeds `PAIR_FRAC_THRESH` against the cell's
  population MAD. Catches sessions with anomalous gain/T_sys, not just
  shape outliers.
- **Neighbour QA** (`utils.neighbor_qa`): beam-weighted local plane fit
  of `W` and `peak_v` over neighbours within `NEIGHBOR_MAX_SEP_DEG`.
  `W_SCALE_FLOOR` is in `K * km/s` (default 1000). Use
  `NEIGHBOR_MAX_SEP_DEG = 2.1` for the brick grid (captures the 8
  ring-1 neighbours).
- **T_sys QA**: per-cell. Flags cells outside `T_SYS_NSIGMA` robust
  sigma or absolute bounds.

State handoff to downstream notebooks is via
`artifacts/scan_load_state.pkl` (`cell_combined`, `qa_flagged_set`,
`v_lsr_overlap`, `dv_kms`, plus per-(session, cell) `T_sys` / gain
dictionaries).

## SDR pipeline internals

The survey runs through a synchronous coordinator in
`ugradiolab/capture/coordinator.py` driving `SDRSession`
(`ugradiolab/capture/sdr_session.py`):

- `SDRSession.run_schedule(schedule)` is a generator that yields one
  correlated dump per `(lo_mhz, noise_on)` slot. FFT/correlation of dump
  N runs in a background thread while USB capture of dump N+1 is in
  flight. The schedule drains synchronously, so no capture or
  correlation carries between schedules -- there is no priming penalty
  at cell boundaries.
- `SurveyCoordinator` runs one cell at a time: prearm the SDR's first
  LO + diode state during the slew, blocking
  `telescope.point(wait=True)`, start a sidereal `TrackingThread`,
  iterate `run_schedule`, hand each dump to a `WriterPool`, repeat.
- `SurveyScheduler` (in `labs/04/scripts/main/_survey.py`) is a
  generator yielding `Cell` objects with recal injection, retry-pass,
  and final recal logic. No cross-thread events.

The streaming-pipeline driver `ugradiolab/capture/streaming.py` and
`make_snap_reader` are kept for `labs/02` and `labs/03` only.

## Hardware parameters

- Telescope: Leuschner 4.5 m dish, HPBW = 3.4 deg
- Receiver: dual-polarization RTL-SDR, 1024 channels
- Sample rate 3.2 MHz, NSAMPLES = 16384, NBLOCKS = 1025 (block 0 discarded)
- LO frequencies: 1419.86 / 1421.14 MHz (frequency switching)
- FFT: NFFT = 1024
- Alt limits: 17 - 83 deg
- Az limits: 7 - 348 deg (exclusion near north)
- Noise diode (nominal): `T_cal = 58 K` (pol 0), `79 K` (pol 1), average 68.5 K
- Integration per dump: `16384 / 3.2e6 * 1025` ~= 5.25 s
- Dump cadence (post-rewrite): ~6 s (pipelined within cell)
- Duty cycle: ~0.55 (capture-bound; slew dominates the gap budget)

## SNR guidelines

Standard schedule: `obs_dumps_per_lo = 4` (8 obs dumps total per cell,
4 surviving LO1/LO2 pairs). Increase via the `SurveyConfig` field if a
follow-up pass is needed for low-SNR cells.

## Lab manual requirements

From `src/ugradio/lab_dish/HI1.tex`:

- **Required**: frequency-switched spectrum at (l=120, b=0) for
  cross-group comparison
- **Required**: position-velocity images, colour images
- **Required**: choose one of 8 projects for the report
- Manual spatial sampling: 2 deg spacing (Nyquist for 3.4 deg beam)
- Foreshortening correction: at high `|b|`, use `delta_l = 2 / cos(b)`
  (already applied by the brick-interleave grid)

Supplementary calibration context is in
`src/ugradio/lab_bighorn/cal_intensity.tex`.
