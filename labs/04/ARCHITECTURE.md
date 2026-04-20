# Lab 4 — 21 cm HI Survey: Architecture

End-to-end pipeline from observation planning through data products.

## Data flow

```
Plan (01)                Observe (otf.py)              Reduce (02)
─────────               ────────────────               ──────────
SURVEY_PARTS ──→ grid ──→ az/manifest    ┌─ .npz ──→ load
                  filter ──→ cell list   │           fftshift
manifest.json ◄────────────────────────┐ │           mask DC/RFI
  ▲                                    │ │           outlier filter
  │              target_selector ──→   │ │           LSR correct
  │              SDR capture ──→ FFT ──┤ │           freq-switch R
  │              correlate ──→ .npz ───┘ │
  │                                      │    ┌──→ W(l,b) heatmaps
  │                                      ├──→ │   Mollweide map
  │                                      │    └──→ spectra grids
  └──────────────── manifest update ◄────┘         pointing traces
```

## 1. Observation planning (`notebooks/01_scan_plan.ipynb`)

**Input**: Survey region definition (`SURVEY_PARTS`), manifest of completed cells

**Process**:
- Define grid in galactic (l, b) at 2° spacing (Nyquist for 3.4° HPBW)
- Boustrophedon ordering (even rows: decreasing l)
- Filter by manifest: skip complete cells (`l % 360, b` lookup)
- Simulate sequential timing: for each cell, compute alt/az at projected
  observation time using astropy galactic → ICRS → AltAz
- Az-side filter: classify cells as rising (7–180°) or setting (180–348°),
  pick side with more accessible cells
- Alt/az cuts: 17° < alt < 83°, 7° < az < 348°

**Output**: Ordered cell list, time estimate, diagnostic plots

## 2. Data acquisition (`scripts/observe_otf.py`)

**Input**: `SURVEY_PARTS` config, manifest, hardware (telescope, 2× SDR, noise diode)

**Process per survey part**:

1. `build_raster_cells` → boustrophedon grid
2. `filter_cells_by_az_side` → keeps one side; includes below-horizon cells
   (classified as rising if in exclusion zone 348–7°)
3. `filter_cells_by_manifest` → removes complete cells
4. `make_scan_target_selector` → skip-and-retry state machine:
   - Tracks dump count per cell via `dump_notifier` callback
   - After `dumps_per_cell` dumps (8 = 4/band × 2 LOs), transitions to next cell
   - Out-of-limits cells are skipped and retried after one full pass
   - `done_event` signals completion; `max_hours` timer can force early stop
5. `make_calibrated_sdr_reader` (in `ugradiolab/capture/readers.py`):
   - **Cal phase**: noise diode ON, `cal_dumps_per_lo` dumps per LO
     (block switching: all dumps at LO₁, then all at LO₂)
   - **Science phase**: diode OFF, block LO switching
     (`[1420]×4 + [1421]×4 + ...`)
   - **Pipelined capture** (`_PipelinedSDR`): overlaps FFT/correlation of
     previous dump with USB transfer of current dump (80% duty cycle vs
     44% sequential)
6. Per dump, the reader does:
   - `_sdr_capture`: set LO (skipped if unchanged), `capture_data` from both
     SDRs → raw IQ int8 `(nblocks, nsamples, 2)`
   - `_sdr_correlate`: discard block 0, reshape into `nfft`-length chunks,
     FFT each chunk, accumulate `corr00 = |V₀|²`, `corr11 = |V₁|²`,
     `corr01 = V₀V₁*`, divide by total windows
7. `StreamingCapture` orchestrates: calls `target_selector` for pointing,
   slews telescope, calls `read_fn` for data, writes `.npz` via writer pool,
   calls `on_save` callback

**Output per dump**: `scan_r{row}_c{col}_dump_{unix_time}_{seq}.npz` containing
`corr00`, `corr11`, `corr01` (1024 channels each), `time`, `lo_freq_mhz`,
`noise_on`, `target_name`, `alt_deg`, `az_deg`, `ra_deg`, `dec_deg`, `seq`

## 3. Data reduction (`notebooks/02_scan_load.ipynb`)

### 3a. Load (cell `load`)
- Glob all `.npz` from `DR1/` through `DR5b/`
- Parse row/col from target name, store all metadata in `records` list

### 3b. fftshift + masking (cell `fft`)
- `fftshift` on `corr00`, `corr11` → reorder to [−f_s/2, +f_s/2]
- Mask DC bin (channel 512) → NaN (covers the LO spike for all LOs)
- For LO=1420 only: mask 2 bins below DC (channels 510–511) to suppress
  leakage from the LO spike near the HI line.  Other LOs need no extra
  mask — their LO spike is at DC (already masked) and sky 1420.0 MHz
  falls at a normal science channel.
- Stokes I = corr00 + corr11

### 3c. RFI flagging (cell `fft`)
- Per dump: NaN-aware rolling median (pandas, window=15, ignores NaN in
  each window so masked DC/leakage bins don't create blind zones)
- MAD σ estimate from residuals
- Flag both positive and negative outliers > 5σ from local median → NaN
  (catches RFI spikes and SDR dropouts)

### 3d. Outlier dump filter — spectral shape (cell `171eb962`)
- Group by (DR, target, LO)
- Compute median spectrum per group
- For each dump, compute ratio = spectrum / median
- Flag dump if >20% of channels deviate >10% from the median ratio
- Catches gain ripple, wrong pointing, partial RFI contamination, and
  broadband glitches that alter spectral shape (not just amplitude)

### 3e. LSR correction + frequency switching (cell `fsw`)
- Assign galactic coords: ICRS → galactic, round to nearest degree
- Compute `v_corr_lsr` per (DR, cell) group using:
  - `astropy.radial_velocity_correction(kind='heliocentric')` (Earth motion)
  - Solar motion projection: 20 km/s toward (18h, +30°) B1900
- Overlap region: 1419.98–1421.02 MHz (256 kHz edge trim), 418 channels
- Topocentric velocity: `v = c × (1 − f_sky / 1420.405)`

**Per-DR results** (topocentric, `cell_results`):
- Within each DR, pair LO₁ and LO₂ dumps positionally
- R = (I₁ − I₂) / I₂ per pair, then `nanmean` across pairs
- Store full-band `R_mean` (1024 ch) and `R_overlap` (418 ch)

**Combined results** (LSR-corrected, `cell_results_combined`):
- Pair within each DR separately (no cross-night pairing)
- Per-DR `R_overlap` sits at `v_topo + v_corr_dr` in LSR frame
- Interpolate each DR's R onto common LSR grid `v_lsr_overlap`
- Average interpolated spectra across DRs

### 3f. Velocity-integrated map (cell `map`)
- NaN channels (DC mask, RFI) filled by linear interpolation from neighbors
- `W = Σ R(v) × Δv` over the interpolated spectrum
- `dv` = 0.528 km/s per channel

### 3g. Spectra grid (cell `spectra`)
- Per-DR: plot `R_overlap` vs topocentric velocity for every cell,
  arranged in (l, b) grid pages

### 3h. Pointing traces (cell `1fcc1c7f`)
Per DR, two plots:
1. **Galactic scan pattern** — (l, b) scatter + beam circles, colored by LST
2. **Alt/az Mollweide** — (az, alt) scatter colored by LST, with min-alt
   and az-exclusion lines

### 3i. Manifest update (cell `eab8d950`)
- Count `n_1420`, `n_1421` per (l, b) from surviving records
- `n_pairs = min(n_1420, n_1421)`, complete if `n_pairs ≥ target` (currently 4)
- Write `survey_manifest.json` → feeds back to steps 1 and 2

## Data products

| Product | Frame | Scope | Cell |
|---------|-------|-------|------|
| W(l, b) heatmap per DR | topocentric | per-DR | `map` |
| W(l, b) Mollweide all-sky | LSR | combined | `map` |
| R(v) spectra grid per DR | topocentric | per-DR | `spectra` |
| (l, b) beam coverage per DR | galactic | per-DR | `1fcc1c7f` |
| (az, alt) pointing trace per DR | topographic | per-DR | `1fcc1c7f` |
| `survey_manifest.json` | — | cumulative | `eab8d950` |

## Shared utilities (`notebooks/utils/`)

| Module | Functions | Purpose |
|--------|-----------|---------|
| `rfi.py` | `flag_rfi_channels`, `flag_outlier_dumps` | Per-channel RFI flagging (rolling median + MAD); per-dump outlier filter (spectral shape) |
| `lsr.py` | `vlsr_correction` | LSR velocity correction (heliocentric + solar motion) for Leuschner |
| `freqswitch.py` | `compute_R_for_dumps` | Frequency-switched R from paired LO dumps, with optional cross-DR LSR interpolation |
| `mapping.py` | `compute_cell_W`, `build_heatmap` | Velocity-integrated W (NaN interpolation), integer-degree gridding |

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
| Dump cadence | ~16.5 s (pipelined), ~29 s (sequential/old DRs) |
| Duty cycle | ~80% (pipelined), ~44% (sequential) |

## Velocity coverage

| Quantity | Value |
|----------|-------|
| Overlap band | 1419.98–1421.02 MHz |
| Channel width | 2.50 kHz = 0.528 km/s |
| Topocentric range | −130 to +90 km/s |
| LSR range | varies by ±30 km/s with date/direction |
