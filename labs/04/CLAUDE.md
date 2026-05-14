# Lab 4 — 21 cm HI Survey: Claude Instructions

## Style Rules

Use ASCII-only text in this lab unless a Unicode character is required by a quoted source or upstream data. Do not introduce special symbols such as degree signs, arrows, em dashes, en dashes, or other Unicode punctuation in code, comments, docstrings, or notebooks.

## Survey Planning Workflow

When the user asks to plan, optimize, or start a new survey observation, follow
this procedure:

### 1. Determine the best survey area

Check which candidate regions are currently accessible from Leuschner Observatory
(lat 37.9183, lon -122.1067):

```python
import astropy.units as u
import astropy.coordinates as ac
from astropy.time import Time

location = ac.EarthLocation(lat=37.9183*u.deg, lon=-122.1067*u.deg, height=304*u.m)
now = Time.now()
```

**Candidate regions** (from `01_scan_plan.ipynb` Section 0, sourced from
`src/ugradio/lab_dish/HI1.tex`):

| Project | l range | b range | T_B | dumps/band |
|---------|---------|---------|-----|------------|
| Gal. plane (narrow) | -10 to 250 | -4 to +4 | 50-150 K | 4 |
| Gal. plane (wide) | 0 to 360 | -20 to +20 | 20-100 K | 4 |
| Great circle l=220/40 | 38-42, 218-222 | -90 to +90 | varies | 4-16 |
| Great circle l=130/310 | 128-132, 308-312 | -90 to +90 | varies | 4-16 |
| NCP | 105 to 160 | +15 to +50 | 10-20 K | 8-16 |
| Orion-Eridanus | 160 to 220 | -70 to -10 | 10-20 K | 4-8 |
| North Polar Spur | 210 to 380 | 0 to +90 | varies | 4-16 |
| HVC | 60 to 180 | +20 to +60 | 1-5 K | 16-32 |
| Magellanic Stream | 60 to 110 | -90 to -30 | 0.1-0.5 K | 32+ |

For each region, compute how many cells are on the rising side (az 7-180) vs
setting side (az 180-348) at the current time. Subtract manifest-complete cells.
Rank by number of remaining accessible cells. Report the results to the user
with a recommendation for which region and side to observe. Then proceed to
update the plan notebook and observe script without further prompting — the
user expects the full pipeline to be set up ready to run.

### 2. Check the manifest

Read `labs/04/survey_manifest.json` to determine which cells are already complete:

```python
from manifest import read_manifest
manifest = read_manifest('labs/04/survey_manifest.json')
```

Subtract complete cells from the candidate count. Only plan for incomplete or
new cells. If the manifest is empty or stale, warn the user to run
`02_scan_load.ipynb` first.

### 3. Apply constraints

All plans must satisfy:
1. **No azimuth crossing** — all cells on one side (rising OR setting), never
   crossing the 348-7 deg exclusion zone
2. **Altitude limits** — 17 deg < alt < 83 deg at the projected observation time
3. **Rising or setting only** — pick the side with more accessible cells
4. **2 deg grid spacing** — Nyquist sampling for 3.4 deg HPBW beam
5. **Manifest-aware** — skip complete cells, flag cells needing extra data
6. **Timing** — main dataset measured ~6 s/dump cadence post-rewrite (10 dumps/cell ~60 s capture); streaming dataset historical ~17 s/dump
7. **Sky rotation** — simulate timing sequentially; cells observed hours into
   the run may have rotated out of limits

### 4. Update the plan notebook

Edit `labs/04/notebooks/01_scan_plan.ipynb` Section 1 (`config` cell):

- Set `SURVEY_PARTS` to the chosen region(s)
- Set `T_START = Time.now()`
- Set `SIDE = 'auto'` (or force 'rising'/'setting' if needed)
- Set `parallelogram = True` for galactic plane surveys near transit
- Verify the grid builder output in Section 2 shows 0 bad cells

### 5. Update the observe script

Edit `labs/04/scripts/observe_otf.py`:

- Update `SURVEY_PARTS` to match the plan notebook exactly (same order,
  same l/b ranges, same step and dumps_per_band)
- The script reads `survey_manifest.json` and filters cells at runtime
- Each survey part auto-creates the next `session_{NNN}` directory under
  `data/lab04/streaming/` — no manual DR naming needed

### 6. Verify before running

- Check that the time estimate fits within the observing window
- Verify all cells are on one side of the az exclusion
- Confirm the manifest has been generated from the latest data
  (run `02a_scan_diagnostics.ipynb` Section 5 if needed)
- Check for the canonical (l=120, b=0) spectrum — required by the lab manual

## Data Pipeline

### Observation cycle

1. **Plan** — run `01_scan_plan.ipynb` (reads manifest, checks sky, simulates)
2. **Sync** — update `observe_otf.py` to match the plan
3. **Push manifest** — copy `survey_manifest.json` to the Pi
4. **Observe** — run `observe_otf.py` on the Pi; each part auto-creates `session_{NNN}/`
5. **Ingest** — transfer data from Pi to `data/lab04/streaming/session_{NNN}/`
6. **Reduce** — run `02_scan_load.ipynb` (saves `reduced_survey.npz`)
7. **Diagnose + manifest** — run `02a_scan_diagnostics.ipynb` (QA plots, writes manifest)
8. **Science products** — run `02b_survey_results.ipynb` (l-v diagrams, maps, spectra)
9. **Repeat** from step 1

### Data organization

- Each observation part creates a session: `session_{NNN}/` (sequential, timestamp-ordered)
- Within each session: `obs_{l}_{b}/obs_{l}_{b}_{timestamp}.npz` (science dumps,
  noise-off) and `cal_{l}_{b}/cal_{l}_{b}_{timestamp}.npz` (calibration dumps)
- Galactic coordinates (l, b) are integer degrees
- Sessions are delimited by calibration blocks: a new session starts whenever a
  `noise_on=True` dump follows a `noise_on=False` dump chronologically
- Calibration scope depends on dataset:
  - `data/lab04/main/` -- per-cell calibration. `cal_{l}_{b}` and `obs_{l}_{b}`
    at the same pointing are paired; each science cell has its own gain and
    T_sys. `cal_*/` contains noise-on dumps; the noise-off reference is the
    matching `obs_*/` at the same (l, b).
  - `data/lab04/streaming/` -- per-session calibration. `cal_*/` cells contain
    only noise-on dumps; the noise-off reference is the session-wide pool of
    `obs_*/` noise-off dumps. All cells in a session share one scalar T_sys.
- The analysis notebook loads all `session_*/obs_*` cells and keys on (l, b)
- m31 stare data lives in `m31/` (flat, legacy format) -- not session-structured
- Clean up Zone.Identifier files after data transfer from Windows

### Frequency switching

- LO pair and bandwidth depend on dataset:
  - `data/lab04/streaming/` -- 1420.0 / 1421.0 MHz, sample rate 2.56 MHz,
    Stokes I = corr00 + corr11
  - `data/lab04/main/`      -- 1419.86 / 1421.14 MHz, sample rate 3.2 MHz,
    Stokes I = corr00 + corr11. The known pol-0 noise-diode coupling
    issue at 3.2 MHz affects T_cal accuracy only; the dimensionless
    ratio R = (I_LO1 - I_LO2)/I_LO2 carries no T_cal dependence, so
    both pols are summed for ~sqrt(2) SNR gain.
- Edge trim 256 kHz each side; the post-trim overlap defines the LSR
  velocity grid (`v_lsr_overlap`)
- R = (I_LO1 - I_LO2) / I_LO2, averaged over surviving dump pairs
- T_B = R * T_sys (per-cell for main, per-session for streaming)
- The DC bin (channel 512 after fftshift) must be masked
- RFI flagging: sliding-window Chebyshev pseudo-continuum with sigma MAD
  threshold (`utils/rfi.py`); local extrema excluded from the fit so the
  baseline is not biased by RFI spikes

### Reduction QA (T_B-based)

All quality-assurance steps in both pipelines run on calibrated T_B (Kelvin),
so integrated W is in K * km/s and noise_rms is in K. This makes residual
thresholds physical and unbiased by cell-to-cell or session-to-session T_sys.

- **Cross-session pair filter** (`utils.flag_outlier_pairs`): builds per-pair
  `T_B_lsr = R_lsr * T_sys` and rejects pairs whose deviant-channel fraction
  exceeds `PAIR_FRAC_THRESH` against the cell's population MAD. Catches
  sessions with anomalous gain/T_sys, not just shape outliers.
- **Neighbor QA** (`utils.neighbor_qa`): beam-weighted local plane fit of
  W and peak_v over neighbors within `NEIGHBOR_MAX_SEP_DEG`. `W_SCALE_FLOOR`
  is in K * km/s (default 1000). The radius depends on the grid:
  - `main/` brick interleave (even b in {-4,-2,0,2,4}, odd b in {-3,-1,1,3}
    offset by Delta_l/2): use `NEIGHBOR_MAX_SEP_DEG = 2.1` -- captures the
    8 ring-1 neighbors (4 half-row diagonals at sep ~sqrt(2), 4 same-axis
    at sep = 2.0).
  - `streaming/` integer 2 deg grid: use `NEIGHBOR_MAX_SEP_DEG = 3.0` --
    captures 4 axis-aligned at 2.0 and 4 diagonals at ~2.83.
- **T_sys QA**: per-cell for main, per-session for streaming. Flags cells
  outside `T_SYS_NSIGMA` robust sigma or absolute bounds; per-session flag
  excludes every cell observed in a deviant session.

State handoff to downstream notebooks is via `scan_load_state.pkl` (pickled
`cell_combined`, `qa_flagged_set`, `v_lsr_overlap`, `dv_kms`, plus per-session
T_sys/gain dictionaries).


## Key files

| File | Role |
|------|------|
| `notebooks/00_streaming_load.ipynb` | M31 stare observation analysis |
| `notebooks/01_scan_plan.ipynb` | Survey planning, visualization, constraint checking |
| `notebooks/02_scan_load.ipynb` | Streaming pipeline: load, reduce, per-session calibration, QA on T_B; writes `scan_load_state.pkl` and `reduced_survey.npz` |
| `notebooks/02_scan_load_lv.ipynb` | Streaming l-v products (Mollweide, b~0 strip, tangent-point V(R), spiral overlay); reads `scan_load_state.pkl` |
| `notebooks/02a_scan_diagnostics.ipynb` | QA visualization, coverage maps, truncation flagging, manifest generation |
| `notebooks/02b_survey_results.ipynb` | Science products: l-v diagrams, maps, reference spectrum |
| `notebooks/03_m31_sensitivity.ipynb` | M31 SNR predictions and sensitivity analysis |
| `notebooks/cal_frames.ipynb` | Side-by-side noise-on/off spectra at four pointings (start/end of each dataset) |
| `notebooks/main/scan_load.ipynb` | Main pipeline: per-cell calibration variant of `02_scan_load.ipynb` |
| `notebooks/main/scan_load_lv.ipynb` | Main l-v products (parallels `02_scan_load_lv.ipynb`) |
| `notebooks/utils/qa.py` | Neighbor-based QA: cell metrics, local plane fits, cross-session pair filter |
| `scripts/observe_otf.py` | OTF raster survey (manifest-aware, az-filtered, pipelined) |
| `scripts/observe.py` | M31 stare (4 LOs: 1420-1423, noise cal) |
| `scripts/observe_m31_cal_off.py` | M31 cal-off at 1420.33/1419.66 MHz |
| `scripts/main.py` | Follow-up observations for truncated spectra (redwards/bluewards) |
| `manifest.py` | Manifest read/write, completeness thresholds, truncation manifest |
| `truncation_manifest.json` | Truncation follow-up completeness state (redwards/bluewards) |
| `ugradiolab/plotting.py` | Matplotlib style constants and figure factories |
| `plotters.py` | Reusable plot functions |
| `reduced_survey.npz` | Serialized reduced data (spectra, metrics, QA flags) |
| `survey_manifest.json` | Current survey completeness state |

## Hardware parameters

- Telescope: Leuschner 4.5m dish, HPBW = 3.4 deg
- Receiver: dual-polarization RTL-SDR, 1024 channels
- Streaming dataset: 2.56 MHz bandwidth, NSAMPLES=32768; main dataset: 3.2 MHz, NSAMPLES=16384
- LO frequencies: streaming 1420.0/1421.0 MHz; main 1419.86/1421.14 MHz (frequency switching)
- FFT: NFFT=1024, NBLOCKS=1025 (block 0 discarded)
- Alt limits: 17 - 83 deg
- Az limits: 7 - 348 deg (exclusion near north)
- Noise diode: T_cal = 58 K (pol 0), 79 K (pol 1), average 68.5 K
- Integration per dump: 13.1 s streaming (32768 / 2.56e6 * 1025); 5.25 s main (16384 / 3.2e6 * 1025)
- Dump cadence (main, post-rewrite): ~6 s (pipelined within cell, no priming between cells)
- Duty cycle (main): ~0.55 (capture-bound; slew dominates the gap budget)

## SDR pipeline (main dataset)

The galactic-plane / NPS surveys run through a synchronous coordinator
in `ugradiolab/capture/coordinator.py` driving `SDRSession`
(`ugradiolab/capture/sdr_session.py`).

* `SDRSession.run_schedule(schedule)` is a generator that yields one
  correlated dump per `(lo_mhz, noise_on)` slot. FFT/correlation of dump
  N runs in a background thread while USB capture of dump N+1 is in
  flight. The schedule drains synchronously, so no capture or
  correlation carries between schedules -- there is no priming penalty
  at cell boundaries.
* `SurveyCoordinator` runs one cell at a time: prearm the SDR's first
  LO + diode state during the slew, blocking `telescope.point(wait=True)`,
  start a sidereal `TrackingThread`, iterate `run_schedule`, hand each
  dump to a `WriterPool`, repeat.
* `SurveyScheduler` (in `labs/04/scripts/main/_survey.py`) is a generator
  yielding `Cell` objects with recal injection, retry-pass, and final
  recal logic. No cross-thread events.

The streaming-pipeline driver `ugradiolab/capture/streaming.py` and
`make_snap_reader` are kept for `labs/02` and `labs/03`.

## SNR guidelines

Current standard: 4 pairs for all latitudes. Can be increased later if needed.

Measured SNR (actual, peak-to-noise in overlap region):
- Plane (b~0): median SNR ~4.6-12.9 with 4 pairs
- |b|~18-29: median SNR ~11-18 with 4 pairs

SNR per dump pair varies by region (~15 for bright plane, ~3 at |b|=20,
~1.5 at |b|=30).

## Lab manual requirements

From `src/ugradio/lab_dish/HI1.tex`:

- **Required**: frequency-switched spectrum at (l=120, b=0) for cross-group
  comparison — NOT YET TAKEN
- **Required**: position-velocity images, color images
- **Required**: choose one of 8 projects for the report
- Manual spatial sampling: 2 deg spacing (Nyquist for 3.4 deg beam)
- Foreshortening correction: at high |b|, use delta_l = 2/cos(b)

## Survey sessions

Two parallel datasets accumulate in separate trees:

- `data/lab04/streaming/` -- legacy 2.56 MHz streaming survey, sessions
  001-037 (2026-04-16 to 2026-04-24), ~13900 dumps. Per-session calibration.
- `data/lab04/main/` -- current 3.2 MHz pipelined survey using the brick
  interleave grid, sessions 001-029 and growing (2026-04-25 onward),
  ~8500 dumps as of 2026-04-29. Per-cell calibration.

Coverage spans the galactic plane (b=-4 to +4), extended latitude (b up to
+28), and scattered fills. Use the appropriate `scan_load.ipynb` (02_ for
streaming, main/ for main) to see current cell-level completion. Counts
above are snapshots and drift as new sessions land -- check `git log` or
the notebook session table for current totals.

## M31 observation

Separate from the survey. Analysis in `00_streaming_load.ipynb`.
- 5 LOs: 1420, 1421, 1422, 1423 (1424 dropped — beyond M31 velocity range)
- M31 systemic velocity: -300 km/s, rotation +-250 km/s
- HI range: 1420.4 to 1423.0 MHz
- Marginal detection: chip 1 peak at v=-319 km/s, SNR ~5-7
- Milky Way HI dominates chip 0 (SNR ~177)
- Expected antenna temp for M31: 0.2-1 K (beam-diluted on 4.5m dish)
