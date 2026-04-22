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
  │           SDR capture ──→ FFT ──┤   │
  │           correlate ──→ .npz ───┘   │    ┌──→ W(l,b) heatmaps
  │                                     ├──→ │   Mollweide map
  │                                     │    └──→ spectra grids
  │                                     │
  │         Diagnostics (02a)           │
  │         ─────────────────           │
  │         load + fftshift + DC mask   │
  │         RFI flag + outlier (+ plots)│
  │         pointing summary            │
  │         coverage Mollweide          │
  │         pointing traces per DR      │
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
2. **Grid construction** — build raster in galactic coordinates at 2° spacing
   (Nyquist for 3.4° HPBW); boustrophedon ordering (even rows: decreasing l);
   optional parallelogram mode aligns rows with constant hour-angle surfaces
   via `compute_iso_ha_slant(l_center, b_center, T_START)`
3. **Manifest filter** — read `survey_manifest.json` via `read_manifest()`;
   skip cells where `complete=True`
4. **Sequential timing simulation** — for each cell, compute alt/az at
   projected mid-observation time using astropy galactic → ICRS → AltAz;
   clock advances by `dumps_per_cell × DUMP_CADENCE_S + SLEW_TIME_S` per cell
5. **Az-side selection** — classify cells as rising (az 7–180°) or setting
   (az 180–348°); pick side with more accessible cells (or force via `SIDE`)
6. **Alt/az cuts** — 17° < alt < 83°, 7° < az < 348°

**Output**: Ordered cell list (`sim`), time estimate, 4 diagnostic plots
(Mollweide footprints, galactic scan pattern with HPBW circles colored by LST,
alt/az Mollweide trace, timeline)

## 2. Data acquisition (`scripts/observe_otf.py`)

**Input**: `SURVEY_PARTS` config, manifest, hardware (telescope, 2× SDR, noise diode)

**Process per survey part**:

1. `build_raster_cells` → boustrophedon grid at `step`-degree spacing
2. `filter_cells_by_az_side` → keeps one side; includes below-horizon cells
   (classified as rising if in exclusion zone 348–7°) for retry during long runs
3. `filter_cells_by_manifest` → removes cells where `n_pairs ≥ 4`
4. `make_scan_target_selector` → skip-and-retry state machine:
   - Tracks dump count per cell via `dump_notifier` callback
   - After `dumps_per_cell` dumps (8 = 4/band × 2 LOs), transitions to next cell
   - Out-of-limits cells are skipped and retried after one full pass
   - `done_event` signals completion; `max_hours` timer can force early stop
5. `make_calibrated_sdr_reader` (in `ugradiolab/capture/readers.py`):
   - **Cal phase**: noise diode ON, `cal_dumps_per_lo` (2) dumps per LO
     (block switching: all dumps at LO₁, then all at LO₂)
   - **Science phase**: diode OFF, block LO switching
     (`[1420]×4 + [1421]×4 + ...`)
   - **Pipelined capture** (`_PipelinedSDR`): overlaps FFT/correlation of
     previous dump with USB transfer of current dump (77% duty cycle vs
     44% sequential)
6. Per dump, the reader does:
   - `_sdr_capture`: set LO (skipped if unchanged), `capture_data` from both
     SDRs → raw IQ int8 `(nblocks, nsamples, 2)`
   - `_sdr_correlate`: discard block 0, reshape into `nfft`-length chunks,
     FFT each chunk, accumulate `corr00 = |V₀|²`, `corr11 = |V₁|²`,
     `corr01 = V₀V₁*`, divide by total windows
7. `StreamingCapture` orchestrates: calls `target_selector` for pointing,
   slews telescope, calls `read_fn` for data, writes `.npz` via 2-thread
   writer pool, calls `on_save` callback (dump_notifier)

**Output per dump**: `scan_r{row}_c{col}/scan_r{row}_c{col}_dump_{unix_time}_{seq}.npz`
containing `corr00`, `corr11`, `corr01` (1024 channels each), `time`,
`lo_freq_mhz`, `noise_on`, `target_name`, `alt_deg`, `az_deg`, `ra_deg`,
`dec_deg`, `seq`

## 3. Spectra & maps (`notebooks/02_scan_load.ipynb`)

### 3a. Load (cell `load`)
- Glob all `.npz` from 11 DRs (`DR1` through `DR7`)
- Parse row/col from target name (`scan_r(\d+)_c(\d+)`), store all metadata
  in `records` list
- Separate calibration (`noise_on=True`) from science dumps

### 3b. fftshift + mask DC + RFI flagging (cells `fft`, `171eb962`)
- `np.fft.fftshift` on `corr00`, `corr11` → reorder to [−f_s/2, +f_s/2]
- Mask DC bin (channel 512) → NaN
- For LO=1420 only: mask channels 510–511 (DC leakage near HI line)
- Stokes I = corr00 + corr11
- Per-channel RFI flagging: `scipy.signal.medfilt` (kernel=15) + MAD σ
  estimate (`σ = MAD / 0.6745`); flag both positive and negative outliers
  > 5σ → NaN (in-place via `flag_rfi_channels`)
- Outlier dump filter via `flag_outlier_dumps`: per (DR, cell, LO) group,
  compare each dump's spectral shape to group median; flag dump if >20% of
  channels deviate >10% from median ratio; remove flagged dumps from records

### 3c. LSR correction + frequency switching (cell `fsw`)
- Assign galactic coords: ICRS → galactic via astropy, round to nearest degree
- Compute `v_corr_lsr` per (DR, cell) group using `vlsr_correction()`:
  - Heliocentric: `astropy.radial_velocity_correction(kind='heliocentric')`
  - Solar motion: 20 km/s toward (RA 270°, Dec +30°) FK4 B1900.0, projected
    onto line of sight
- Overlap region: 1419.98–1421.02 MHz (256 kHz edge trim each side), 416 channels
- R = (I₁ − I₂) / I₂ per dump pair via `compute_R_for_dumps()`, `nanmean`
  across pairs
- Combined results: per-DR spectra interpolated onto common LSR velocity grid
  before merging

### 3d. Velocity-integrated map (cell `map`)
- `W = Σ R(v) × Δv` per cell via `compute_cell_W()` (NaN-interpolated before
  summation)
- Per-DR flat heatmaps via `plot_heatmap()` (integer-degree gridding)
- Combined all-sky Mollweide via `plot_survey_mollweide()` (centered at l=120°)

### 3e. Spectra grid (cell `spectra`)
- Per-DR: `R_overlap` vs topocentric velocity for every cell, arranged in
  (l, b) grid pages (5 columns) via `plot_spectra_grid()`

## 3A. Diagnostics & manifest (`notebooks/02a_scan_diagnostics.ipynb`)

Independent load + fftshift + DC mask + RFI + outlier pipeline (same
steps as §3a–3b), plus:

- **Outlier diagnostic plots**: overplot all dumps per (cell, LO) with
  flagged outliers highlighted in red, median spectrum in black
- **Pointing summary**: per-(l, b) dump counts, bad pixel stats
  (pandas DataFrame)
- **Survey coverage**: Mollweide with pointings colored by dump count,
  overlaid on 11 candidate project footprints and never-observable mask
- **Pointing traces per DR**: galactic (l, b) scan pattern with HPBW beam
  circles + alt/az Mollweide, both colored by LST
- **Manifest update**: tally `n_1420`, `n_1421` per (l, b); compute
  `n_pairs = min(n_1420, n_1421)`, complete if ≥ 4 (via `pairs_target()`);
  write `survey_manifest.json` via `write_manifest()` → feeds back to
  planning (01) and observation (otf.py)

## 4. M31 stare reduction (`notebooks/00_streaming_load.ipynb`)

**Input**: `.npz` dumps from `data/lab04/streaming/m31/`
(5 LOs: 1420, 1421, 1422, 1423, 1424 MHz)

**Process**:
1. Load 714 dumps (20 cal + 694 science), separate by noise diode state
2. `np.fft.fftshift`, mask DC bin (ch 512) → NaN; for LO=1420, also mask
   channels 510–511 (DC leakage near HI line)
3. Stokes I = corr00 + corr11
4. Per-channel RFI flagging: `scipy.signal.medfilt` (kernel=15) + 5σ MAD
5. Per-dump outlier filter: spectral shape comparison (same thresholds as survey)
6. Mean Stokes I per LO band (diagnostic plot: science + cal overlaid)
7. Build 4 adjacent LO pairs: R = (I₁ − I₂) / I₂, averaged over dump pairs
   (140, 138, 138, 138 pairs respectively)
8. LSR velocity correction: heliocentric + solar motion → v_corr = +3.22 km/s
9. Overlap trimming: 256 kHz from each edge per pair → ~264 usable channels

**Output**: Frequency-switched HI profiles per LO pair (chips 0–3),
covering v_LSR ≈ −760 to +93 km/s. Milky Way HI dominates chip 0
(SNR ~177); M31 marginal detection on chip 1 (SNR ~5–7 at v ≈ −319 km/s).

| Pair (MHz) | Overlap (MHz) | v_LSR range (km/s) |
|------------|---------------|--------------------|
| 1420 / 1421 | 1419.98–1421.02 | −127 to +93 |
| 1421 / 1422 | 1420.98–1422.02 | −338 to −118 |
| 1422 / 1423 | 1421.98–1423.02 | −549 to −329 |
| 1423 / 1424 | 1422.98–1424.02 | −760 to −540 |

## 5. M31 sensitivity analysis (`notebooks/03_m31_sensitivity.ipynb`)

**Input**: Survey data (all DRs), telescope parameters, M31 literature values

**Process**:
1. Telescope gain: G = η_A × A_geom / (2k_B) = 0.00288 K/Jy
   (D = 4.5 m, η_A = 0.5, A_eff = 7.95 m²)
2. Load 1082 survey cells (same fftshift + DC mask + RFI pipeline as §3b),
   pair LO dumps per cell, compute per-dump-pair SNR in overlap region
   (peak/rms, noise from line-free edge channels) → 4518 pairs
3. Bin SNR by |b|, back-calculate T_sys from measured SNR vs reference T_B
   per latitude band (100 K at b=0 tapering to 5 K at b=28)
4. M31 flux density: Rayleigh–Jeans conversion of brightness temperature
   in iso-velocity slices, accounting for beam dilution
   (fiducial: T_B = 10 K, Ω_source = 1.5 deg², η_ff = 0.11 → S_ν = 283 Jy,
   T_A = 0.816 K)
5. Radiometer equation → required integration time for target SNR at
   various spectral smoothing resolutions (5–106 km/s)
6. Cross-check: predicted SNR for existing M31 observation (chip 1,
   138 pairs, τ = 1809 s per LO) matches observed SNR ~5–7 at ~79 km/s
   smoothing
7. Sensitivity analysis: integration time vs T_B and η_A assumptions

**Output**: T_sys = 381 K (median from survey), SEFD = 132 kJy,
integration time tables, SNR vs smoothing plots, observing budget
(all goals achievable within existing ~30 min/LO integration).

## Data products

| Product | Frame | Scope | Notebook |
|---------|-------|-------|----------|
| W(l, b) heatmap per DR | topocentric | per-DR | 02 `map` |
| W(l, b) Mollweide all-sky | LSR | combined | 02 `map` |
| R(v) spectra grid per DR | topocentric | per-DR | 02 `spectra` |
| Outlier dump diagnostic plots | — | per-DR | 02a `21a05e63` |
| Pointing summary table | — | cumulative | 02a `0e44f1fc` |
| Coverage Mollweide (dump count) | galactic | cumulative | 02a `c185c95d` |
| (l, b) scan pattern per DR | galactic | per-DR | 02a `a82fe1a9` |
| (az, alt) pointing trace per DR | topographic | per-DR | 02a `a82fe1a9` |
| `survey_manifest.json` | — | cumulative | 02a `0a6eeb76` |
| M31 HI profiles per LO pair | LSR | M31 stare | 00 `plot` |
| M31 SNR vs integration time | — | M31 | 03 |
| T_sys from survey SNR | — | all DRs | 03 |

## Shared utilities (`notebooks/utils/`)

| Module | Functions | Purpose |
|--------|-----------|---------|
| `rfi.py` | `flag_rfi_channels`, `flag_outlier_dumps` | Per-channel RFI: `scipy.signal.medfilt` (kernel=15) + MAD σ (`MAD / 0.6745`), flag >5σ outliers → NaN in-place; per-dump outlier: spectral shape vs group median, flag if >20% channels deviate >10% |
| `lsr.py` | `vlsr_correction` | LSR velocity correction: heliocentric (astropy) + solar motion 20 km/s toward RA 270° Dec +30° FK4 B1900.0, projected onto line of sight |
| `freqswitch.py` | `compute_R_for_dumps` | Frequency-switched R from paired LO dumps; groups by (DR, LO), pairs I₁/I₂, optional LSR interpolation onto common grid before cross-DR merge |
| `mapping.py` | `compute_cell_W`, `build_heatmap` | Velocity-integrated W with NaN interpolation before summation; integer-degree gridding via `searchsorted` onto 2-D map |

## Support modules

| File | Purpose |
|------|---------|
| `manifest.py` | `read_manifest`, `write_manifest`, `get_complete_cells`, `get_incomplete_cells`, `pairs_target`; JSON I/O for per-(l, b) completeness tracking; `n_pairs = min(n_1420, n_1421)`, complete if ≥ `pairs_target()` (4) |
| `plotting.py` | Matplotlib rcParams (LaTeX, Computer Modern Roman), page dimensions (textwidth 7.59 in, columnwidth 3.73 in), line weights, marker sizes, alpha levels, figure factories (`textwidth_figure`, `columnwidth_figure`, `landscapewidth_figure`, `subpanels`, `zero_line`) |
| `plotters.py` | Reusable plot functions: `plot_hi_spectrum`, `plot_spectra_grid`, `plot_heatmap`, `plot_survey_mollweide`, `plot_altaz_mollweide`, `plot_scan_pattern`, `plot_timeline`, `plot_calibrated_profiles` |

## Other scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `observe_otf.py` | OTF raster survey (manifest-aware, az-filtered, pipelined) |
| `observe.py` | M31 stare (4 LOs: 1420–1423 MHz, noise cal) |
| `observe_m31_cal_off.py` | M31 cal-off reference at 1420.33/1419.66 MHz |
| `observe_fill_112_1.py` | Fill 3 missing dumps at (l=112, b=+1) for DR2b |
| `observe_fill_183_29.py` | Fill 17 missing dumps at (l=183, b=+29) for DR2a |

## Hardware parameters

| Parameter | Value |
|-----------|-------|
| Telescope | Leuschner 4.5 m, HPBW = 3.4° |
| Receiver | Dual-pol RTL-SDR, 2.56 MHz BW, 1024 channels |
| LO frequencies | 1420.0 / 1421.0 MHz (frequency switching) |
| FFT | NFFT=1024, NBLOCKS=1025 (block 0 discarded), NSAMPLES=32768 |
| Alt limits | 17–83° |
| Az limits | 7–348° (exclusion near north) |
| Noise diode | T_cal = 79 K (pol 0), 58 K (pol 1), mean 68.5 K |
| Integration per dump | 13.1 s (1025 × 32768 / 2.56 × 10⁶) |
| Dump cadence | ~17 s (pipelined), ~29 s (sequential/old DRs) |
| Duty cycle | ~77% (pipelined), ~44% (sequential) |

## Derived system parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Geometric area A_geom | 15.90 m² | π(D/2)² |
| Effective area A_eff | 7.95 m² | η_A × A_geom, η_A = 0.5 |
| Gain G | 0.00288 K/Jy | A_eff / (2k_B) |
| Channel width Δν | 2.50 kHz | f_s / NFFT |
| Velocity resolution Δv | 0.528 km/s | c × Δν / f_HI |
| T_sys | 381 K | median from survey SNR back-calculation (03) |
| SEFD | 132 kJy | T_sys / G |
| Beam solid angle Ω_beam | 13.1 deg² | 1.133 × HPBW² |

## Velocity coverage

| Quantity | Value |
|----------|-------|
| Overlap band (survey) | 1419.98–1421.02 MHz (416 channels) |
| Channel width | 2.50 kHz = 0.528 km/s |
| Topocentric range | −130 to +90 km/s |
| LSR range | varies by ±30 km/s with date/direction |
| M31 coverage (4 chips) | −760 to +93 km/s (LSR) |
