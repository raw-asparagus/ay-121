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
6. **Timing** — use 20 s/dump conservative cadence (measured: 17 s pipelined)
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
- Within each session: `obs_{l}_{b}/obs_{l}_{b}_{timestamp}.npz` (science dumps)
  and `cal_{l}_{b}/cal_{l}_{b}_{timestamp}.npz` (noise-diode calibration dumps)
- Galactic coordinates (l, b) are integer degrees; `obs_` vs `cal_` reflects `noise_on` state
- Sessions are delimited by calibration blocks: a new session starts whenever a
  `noise_on=True` dump follows a `noise_on=False` dump chronologically
- The analysis notebook loads all `session_*/obs_*` cells and keys on (l, b)
- m31 stare data lives in `m31/` (flat, legacy format) — not session-structured
- Clean up Zone.Identifier files after data transfer from Windows

### Frequency switching

- Two LO frequencies: 1420.0 and 1421.0 MHz
- Overlap after edge trim (256 kHz each side): 1419.98 to 1421.02 MHz
- Velocity coverage: -130 to +90 km/s
- R = (I_LO1 - I_LO2) / I_LO2, averaged over dump pairs
- The DC bin (channel 512 after fftshift) must be masked
- Sky frequency 1420.0 MHz + 2 bins below must be masked per LO
- RFI flagging: scipy.signal.medfilt (kernel=15) + 5-sigma MAD threshold


## Key files

| File | Role |
|------|------|
| `notebooks/00_streaming_load.ipynb` | M31 stare observation analysis |
| `notebooks/01_scan_plan.ipynb` | Survey planning, visualization, constraint checking |
| `notebooks/02_scan_load.ipynb` | Data loading, reduction, QA, saves `reduced_survey.npz` |
| `notebooks/02a_scan_diagnostics.ipynb` | QA visualization, coverage maps, truncation flagging, manifest generation |
| `notebooks/02b_survey_results.ipynb` | Science products: l-v diagrams, maps, reference spectrum |
| `notebooks/03_m31_sensitivity.ipynb` | M31 SNR predictions and sensitivity analysis |
| `notebooks/utils/qa.py` | Neighbor-based QA: cell metrics and local plane fits |
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
- Receiver: dual-polarization RTL-SDR, 2.56 MHz bandwidth, 1024 channels
- LO frequencies: 1420.0 / 1421.0 MHz (frequency switching)
- FFT: NFFT=1024, NBLOCKS=1025 (block 0 discarded), NSAMPLES=32768
- Alt limits: 17 - 83 deg
- Az limits: 7 - 348 deg (exclusion near north)
- Noise diode: T_cal = 79 K (pol 0), 58 K (pol 1), average 68.5 K
- Dump cadence: ~17 s (pipelined reader)
- Integration per dump: 13.1 s (1025 * 32768 / 2.56e6)
- Duty cycle: ~77% (pipelined), ~44% (sequential)

## Pipelined reader

`ugradiolab/capture/readers.py` contains `_PipelinedSDR` which overlaps
FFT/correlation of the previous capture with USB data transfer of the next
capture. This improved duty cycle from 44% to 77% (cadence 29s -> 17s).
The bottleneck is now LO frequency switching (~2-3s per dump).

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

Historical data migrated to session_001 through session_027 (2026-04-16 to 2026-04-23).
New observations start from session_028. Total: 12833 science+cal dumps across 27 sessions.

Coverage spans galactic plane (b=-4 to +4), extended latitude (b up to +28),
and scattered fills. Use `02_scan_load.ipynb` to see current cell-level completion.

## M31 observation

Separate from the survey. Analysis in `00_streaming_load.ipynb`.
- 5 LOs: 1420, 1421, 1422, 1423 (1424 dropped — beyond M31 velocity range)
- M31 systemic velocity: -300 km/s, rotation +-250 km/s
- HI range: 1420.4 to 1423.0 MHz
- Marginal detection: chip 1 peak at v=-319 km/s, SNR ~5-7
- Milky Way HI dominates chip 0 (SNR ~177)
- Expected antenna temp for M31: 0.2-1 K (beam-diluted on 4.5m dish)
