# Lab 4 — 21 cm HI Survey: Architecture

End-to-end pipeline from observation planning through data products.

## Data flow

```
Plan (01)                Observe (otf.py)              Spectra & Maps (02)
─────────               ────────────────               ───────────────────
SURVEY_PARTS ──→ grid ──→ az/manifest    ┌─ .npz ──→ load
                  filter ──→ cell list   │           fftshift + mask DC
manifest.json ◄──────────────────────┐   │           RFI flag + outlier
  ▲                                  │   │           LSR correct
  │           target_selector ──→    │   │           freq-switch R
  │           SDR capture ──→ FFT ──┤   │           session consistency
  │           correlate ──→ .npz ───┘   │           neighbor QA
  │                                     │    ┌──→ W(l,b) heatmaps
  │                                     ├──→ │   Mollweide map
  │                                     │    └──→ spectra grids
  │                                     │
  │         Diagnostics (02a)           │
  │         ─────────────────           │
  │         load + fftshift + DC mask   │
  │         RFI flag + outlier (+ plots)│
  │         freq-switch + QA            │
  │         pointing summary            │
  │         coverage Mollweide          │
  │         pointing traces per session │
  └──────── manifest update ◄───────────┘

M31 stare (observe.py)        M31 analysis
──────────────────────        ────────────
5 LOs ──→ .npz ──→ 00_streaming_load ──→ freq-switched profiles
                   03_m31_sensitivity ──→ SEFD / integration time
```

## 1. Observation planning (`notebooks/01_scan_plan.ipynb`)

**Input**: Survey region definition (`SURVEY_PARTS`), manifest of completed cells

**Process**:

1. **Project footprints** — define 11 candidate survey regions in galactic
   (l, b); compute a never-observable sky mask by sweeping all hour angles
   through the alt/az constraints; Mollweide visualization with hatched
   inaccessible region
2. **Grid construction** — build raster in galactic coordinates at 2 deg spacing
   (Nyquist for 3.4 deg HPBW); boustrophedon ordering (even rows: decreasing l);
   optional parallelogram mode aligns rows with constant hour-angle surfaces
   via `compute_iso_ha_slant(l_center, b_center, T_START)`
3. **Manifest filter** — read `survey_manifest.json` via `read_manifest()`;
   skip cells where `complete=True`
4. **Sequential timing simulation** — for each cell, compute alt/az at
   projected mid-observation time using astropy galactic -> ICRS -> AltAz;
   clock advances by `dumps_per_cell * DUMP_CADENCE_S + SLEW_TIME_S` per cell
5. **Az-side selection** — classify cells as rising (az 7-180 deg) or setting
   (az 180-348 deg); pick side with more accessible cells (or force via `SIDE`)
6. **Alt/az cuts** — 17 deg < alt < 83 deg, 7 deg < az < 348 deg

**Output**: Ordered cell list (`sim`), time estimate, 4 diagnostic plots
(Mollweide footprints, galactic scan pattern with HPBW circles colored by LST,
alt/az Mollweide trace, timeline)

## 2. Data acquisition (`scripts/observe_otf.py`)

**Input**: `SURVEY_PARTS` config, manifest, hardware (telescope, 2x SDR, noise diode)

**Process per survey part**:

1. `build_raster_cells` -> boustrophedon grid at `step`-degree spacing
2. `filter_cells_by_az_side` -> keeps one side; includes below-horizon cells
   (classified as rising if in exclusion zone 348-7 deg) for retry during long runs
3. `filter_cells_by_manifest` -> removes cells where `n_pairs >= 4`
4. `_next_session_dir()` -> auto-creates `session_{NNN}/` under `data/lab04/streaming/`
5. `make_scan_target_selector` -> skip-and-retry state machine:
   - Returns `obs_{l}_{b}` as target name (WriterPool flips to `cal_` for noise-on dumps)
   - Tracks dump count per cell via `dump_notifier` callback
   - After `dumps_per_cell` dumps (8 = 4/band x 2 LOs), transitions to next cell
   - Out-of-limits cells are skipped and retried after one full pass
   - `done_event` signals completion; `max_hours` timer can force early stop
6. `make_calibrated_sdr_reader` (in `ugradiolab/capture/readers.py`):
   - **Cal phase**: noise diode ON, `cal_dumps_per_lo` (2) dumps per LO
     (block switching: all dumps at LO1, then all at LO2)
   - **Science phase**: diode OFF, block LO switching
     (`[1420]*4 + [1421]*4 + ...`)
   - **Pipelined capture** (`_PipelinedSDR`): overlaps FFT/correlation of
     previous dump with USB transfer of current dump (77% duty cycle vs
     44% sequential)
7. Per dump, the reader does:
   - `_sdr_capture`: set LO (skipped if unchanged), `capture_data` from both
     SDRs -> raw IQ int8 `(nblocks, nsamples, 2)`
   - `_sdr_correlate`: discard block 0, reshape into `nfft`-length chunks,
     FFT each chunk, accumulate `corr00 = |V0|^2`, `corr11 = |V1|^2`,
     `corr01 = V0V1*`, divide by total windows
8. `StreamingCapture` orchestrates: calls `target_selector` for pointing,
   slews telescope, calls `read_fn` for data, writes `.npz` via 2-thread
   writer pool, calls `on_save` callback (dump_notifier)

**Output per dump**: `<obs|cal>_{l}_{b}/<obs|cal>_{l}_{b}_{YYYYMMDD}_{HHMMSS}.npz`
containing `corr00`, `corr11`, `corr01` (1024 channels each), `time`,
`lo_freq_mhz`, `noise_on`, `target_name`, `alt_deg`, `az_deg`, `ra_deg`,
`dec_deg`, `seq`

## 3. Spectra & maps (`notebooks/02_scan_load.ipynb`)

### 3a. Load
- Glob all `.npz` from `session_*/obs_*/`
- Parse galactic (l, b) from target name (`obs_{l}_{b}`), store all metadata
  in `records` list
- Separate calibration (`noise_on=True`) from science dumps

### 3b. fftshift + mask DC + RFI flagging
- `np.fft.fftshift` on `corr00`, `corr11` -> reorder to [-f_s/2, +f_s/2]
- Mask DC bin (channel 512) -> NaN
- For LO=1420 only: mask channels 510-511 (DC leakage near HI line)
- Stokes I = corr00 + corr11
- Per-channel RFI flagging: `scipy.signal.medfilt` (kernel=15) + MAD sigma
  estimate (`sigma = MAD / 0.6745`); flag both positive and negative outliers
  > 5 sigma -> NaN (in-place via `flag_rfi_channels`)
- Outlier dump filter via `flag_outlier_dumps`: per (session, cell, LO) group,
  compare each dump's spectral shape to group median; flag dump if >20% of
  channels deviate >10% from median ratio; remove flagged dumps from records

### 3c. LSR correction + frequency switching
- Galactic coords parsed directly from target name (integer l, b)
- Compute `v_corr_lsr` per (session, cell) group using `vlsr_correction()`:
  - Heliocentric: `astropy.radial_velocity_correction(kind='heliocentric')`
  - Solar motion: 20 km/s toward (RA 270 deg, Dec +30 deg) FK4 B1900.0,
    projected onto line of sight
- Overlap region: 1419.98-1421.02 MHz (256 kHz edge trim each side), 416 channels
- R = (I1 - I2) / I2 per dump pair via `compute_R_for_dumps()`, `nanmean`
  across pairs within each session
- Combined results: per-session spectra interpolated onto common LSR velocity
  grid before merging

### 3d. Quality assurance

**Session-level spectral consistency** (for cells with 2+ sessions):
- Channel-wise median across sessions -> reference spectrum
- Per-session residual = R_session - R_median
- Broadband z = |mean(residual)| / (noise_rms / sqrt(N_ch)) -- catches gain offsets
- Narrowband z = max(|residual|) / noise_rms -- catches localized features
- noise_rms from off-signal channels (v_LSR < -100 km/s)
- Drop session if broadband_z > 4 or narrowband_z > 5
- Recompute combined R from surviving sessions

**Neighbor-based spatial QA**:
- W and peak velocity compared to beam-weighted local plane fit
- W flag: fractional residual > 30% and z > 3.5
- peak_v flag: absolute residual > 15 km/s and z > 4.0

### 3e. Velocity-integrated map
- `W = sum(R(v) * dv)` per cell via `compute_cell_W()` (NaN-interpolated
  before summation)
- Per-session flat heatmaps via `plot_heatmap()` (integer-degree gridding)
- Combined all-sky Mollweide via `plot_survey_mollweide()` (centered at l=120 deg)

### 3f. Spectra grid
- Per-session: `R_overlap` vs topocentric velocity for every cell, arranged in
  (l, b) grid pages (5 columns) via `plot_spectra_grid()`

## 3A. Diagnostics & manifest (`notebooks/02a_scan_diagnostics.ipynb`)

Independent load + fftshift + DC mask + RFI + outlier pipeline (same
steps as 3a-3b), plus frequency switching and full QA (same as 3c-3d), then:

- **Outlier diagnostic plots**: overplot all dumps per (cell, LO) with
  flagged outliers highlighted in red, median spectrum in black
- **Pointing summary**: per-(l, b) dump counts, bad pixel stats
  (pandas DataFrame)
- **Survey coverage**: Mollweide with pointings colored by dump count,
  overlaid on 11 candidate project footprints and never-observable mask
- **Pointing traces per session**: galactic (l, b) scan pattern with HPBW beam
  circles + alt/az Mollweide, both colored by LST
- **Manifest update**: tally `n_1420`, `n_1421` per (l, b), excluding QA-flagged
  cells; compute `n_pairs = min(n_1420, n_1421)`, complete if >= 4 (via
  `pairs_target()`); write `survey_manifest.json` via `write_manifest()` ->
  feeds back to planning (01) and observation (otf.py)

## 4. M31 stare reduction (`notebooks/00_streaming_load.ipynb`)

**Input**: `.npz` dumps from `data/lab04/streaming/m31/`
(5 LOs: 1420, 1421, 1422, 1423, 1424 MHz)

**Process**:
1. Load 714 dumps (20 cal + 694 science), separate by noise diode state
2. `np.fft.fftshift`, mask DC bin (ch 512) -> NaN; for LO=1420, also mask
   channels 510-511 (DC leakage near HI line)
3. Stokes I = corr00 + corr11
4. Per-channel RFI flagging: `scipy.signal.medfilt` (kernel=15) + 5 sigma MAD
5. Per-dump outlier filter: spectral shape comparison (same thresholds as survey)
6. Mean Stokes I per LO band (diagnostic plot: science + cal overlaid)
7. Build 4 adjacent LO pairs: R = (I1 - I2) / I2, averaged over dump pairs
   (140, 138, 138, 138 pairs respectively)
8. LSR velocity correction: heliocentric + solar motion -> v_corr = +3.22 km/s
9. Overlap trimming: 256 kHz from each edge per pair -> ~264 usable channels

**Output**: Frequency-switched HI profiles per LO pair (chips 0-3),
covering v_LSR approx -760 to +93 km/s. Milky Way HI dominates chip 0
(SNR ~177); M31 marginal detection on chip 1 (SNR ~5-7 at v approx -319 km/s).

| Pair (MHz) | Overlap (MHz) | v_LSR range (km/s) |
|------------|---------------|--------------------|
| 1420 / 1421 | 1419.98-1421.02 | -127 to +93 |
| 1421 / 1422 | 1420.98-1422.02 | -338 to -118 |
| 1422 / 1423 | 1421.98-1423.02 | -549 to -329 |
| 1423 / 1424 | 1422.98-1424.02 | -760 to -540 |

## 5. M31 sensitivity analysis (`notebooks/03_m31_sensitivity.ipynb`)

**Input**: Survey data (all sessions), telescope parameters, M31 literature values

**Process**:
1. Telescope gain: G = eta_A * A_geom / (2k_B) = 0.00288 K/Jy
   (D = 4.5 m, eta_A = 0.5, A_eff = 7.95 m^2)
2. Load survey cells (same fftshift + DC mask + RFI pipeline),
   pair LO dumps per cell, compute per-dump-pair SNR in overlap region
   (peak/rms, noise from line-free edge channels)
3. Bin SNR by |b|, back-calculate T_sys from measured SNR vs reference T_B
   per latitude band
4. M31 flux density: Rayleigh-Jeans conversion, beam dilution
5. Radiometer equation -> required integration time for target SNR
6. Cross-check: predicted SNR for existing M31 observation matches observed

**Output**: T_sys = 381 K (median from survey), SEFD = 132 kJy,
integration time tables, SNR vs smoothing plots

## Data products

| Product | Frame | Scope | Notebook |
|---------|-------|-------|----------|
| W(l, b) heatmap per session | topocentric | per-session | 02 |
| W(l, b) Mollweide all-sky | LSR | combined | 02 |
| R(v) spectra grid per session | topocentric | per-session | 02 |
| Outlier dump diagnostic plots | -- | per-session | 02a |
| Pointing summary table | -- | cumulative | 02a |
| Coverage Mollweide (dump count) | galactic | cumulative | 02a |
| (l, b) scan pattern per session | galactic | per-session | 02a |
| (az, alt) pointing trace per session | topographic | per-session | 02a |
| `survey_manifest.json` | -- | cumulative | 02a |
| M31 HI profiles per LO pair | LSR | M31 stare | 00 |
| M31 SNR vs integration time | -- | M31 | 03 |
| T_sys from survey SNR | -- | all sessions | 03 |

## Shared utilities (`notebooks/utils/`)

| Module | Functions | Purpose |
|--------|-----------|---------|
| `rfi.py` | `flag_rfi_channels`, `flag_outlier_dumps` | Per-channel RFI: medfilt + MAD, flag >5 sigma -> NaN; per-dump outlier: spectral shape vs group median per (session, cell, LO) |
| `lsr.py` | `vlsr_correction` | LSR velocity correction: heliocentric (astropy) + solar motion 20 km/s toward RA 270 deg Dec +30 deg FK4 B1900.0 |
| `freqswitch.py` | `compute_R_for_dumps` | Frequency-switched R from paired LO dumps; groups by session, pairs I1/I2, optional LSR interpolation; returns per-session spectra for consistency checks |
| `mapping.py` | `compute_cell_W`, `build_heatmap` | Velocity-integrated W with NaN interpolation; integer-degree gridding |

## Support modules

| File | Purpose |
|------|---------|
| `manifest.py` | `read_manifest`, `write_manifest`, `get_complete_cells`, `get_incomplete_cells`, `pairs_target`; JSON I/O for per-(l, b) completeness tracking |
| `plotting.py` | Matplotlib rcParams, page dimensions, line weights, marker sizes, alpha levels, figure factories |
| `plotters.py` | Reusable plot functions: `plot_hi_spectrum`, `plot_spectra_grid`, `plot_heatmap`, `plot_survey_mollweide`, etc. |

## Other scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `observe_otf.py` | OTF raster survey (manifest-aware, az-filtered, pipelined) |
| `observe.py` | M31 stare (4 LOs: 1420-1423 MHz, noise cal) |
| `observe_m31_cal_off.py` | M31 cal-off reference at 1420.33/1419.66 MHz |
| `observe_fill_112_1.py` | Fill missing dumps at (l=112, b=+1) |
| `observe_fill_183_29.py` | Fill missing dumps at (l=183, b=+29) |

## Hardware parameters

| Parameter | Value |
|-----------|-------|
| Telescope | Leuschner 4.5 m, HPBW = 3.4 deg |
| Receiver | Dual-pol RTL-SDR, 2.56 MHz BW, 1024 channels |
| LO frequencies | 1420.0 / 1421.0 MHz (frequency switching) |
| FFT | NFFT=1024, NBLOCKS=1025 (block 0 discarded), NSAMPLES=32768 |
| Alt limits | 17-83 deg |
| Az limits | 7-348 deg (exclusion near north) |
| Noise diode | T_cal = 79 K (pol 0), 58 K (pol 1), mean 68.5 K |
| Integration per dump | 13.1 s (1025 * 32768 / 2.56e6) |
| Dump cadence | ~17 s (pipelined) |
| Duty cycle | ~77% (pipelined) |

## Derived system parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Geometric area A_geom | 15.90 m^2 | pi(D/2)^2 |
| Effective area A_eff | 7.95 m^2 | eta_A * A_geom, eta_A = 0.5 |
| Gain G | 0.00288 K/Jy | A_eff / (2k_B) |
| Channel width dnu | 2.50 kHz | f_s / NFFT |
| Velocity resolution dv | 0.528 km/s | c * dnu / f_HI |
| T_sys | 381 K | median from survey SNR back-calculation (03) |
| SEFD | 132 kJy | T_sys / G |
| Beam solid angle Omega_beam | 13.1 deg^2 | 1.133 * HPBW^2 |

## Velocity coverage

| Quantity | Value |
|----------|-------|
| Overlap band (survey) | 1419.98-1421.02 MHz (416 channels) |
| Channel width | 2.50 kHz = 0.528 km/s |
| Topocentric range | -130 to +90 km/s |
| LSR range | varies by +/-30 km/s with date/direction |
| M31 coverage (4 chips) | -760 to +93 km/s (LSR) |
