# `main_scan_load.ipynb` — Reduction pipeline

End-to-end walk through the cells in `labs/04/main_scan_load.ipynb` that
take raw SDR dumps under `data/main/` and `data/nps/` to a calibrated
`T_B(l, b, v_LSR)` survey product.

## Inputs

- `data/main/session_*/{obs|cal}_{l_name}_{b}/*.npz`
- `data/nps/session_*/{obs|cal}_{l_name}_{b}/*.npz`
- Each `.npz` carries `corr00`, `corr11` (1024-channel power spectra,
  `float64`), plus scalars `lo_freq_mhz`, `noise_on`, `time`,
  `target_name`, `alt_deg`, `az_deg`, `ra_deg`, `dec_deg`, `seq`.

The pipeline reduces both surveys jointly. Per-cell calibration scope —
each `(l, b)` cell carries its own cal-on/off pair, and gain + T_sys are
solved per `(session, cell)`.

## Stage 0 — Configuration (cell 0)

Imports and constants. Notable parameters:

- **Hardware**: `SAMPLE_RATE_HZ = 3.2e6`, `NFFT = 1024`, `HI_REST_MHZ = 1420.405751768`
- **LO pair**: `F1_MHZ = 1419.86`, `F2_MHZ = 1421.14`
- **Noise diode**: `T_CAL_00 = 58 K` (pol 0, treated as scalar), `T_CAL_11 = 79 K` (pol 1, used)
- **Beam**: `HPBW_DEG = 3.4`
- **RFI flagging**: `RFI_WINDOW = 15`, `RFI_SIGMA = 10.0`, `RFI_CHEB_DEGREE = 3`,
  `RFI_SAMPLE_FRAC = 0.7`, `RFI_EXTREMA_ORDER = 2`
- **Outlier-dump filter**: `SHAPE_DEV_THRESH = 0.05`, `SHAPE_FRAC_THRESH = 0.05`,
  `SHAPE_MIN_GROUP_SIZE = 3`
- **Edge trim**: `EDGE_TRIM_MHZ = 0.256` (post-trim defines the LSR overlap region)
- **Calibration**: `CAL_SPECTRUM_KEY = 'corr11'`, `CAL_GAIN_KEY = 'gain_11'`
- **Pair filter** (cross-session): `PAIR_NSIGMA = 5.0`, `PAIR_FRAC_THRESH = 0.15`,
  `MIN_VIABLE_PAIRS = 3`, `PAIR_SPECTRUM_KEY = 'T_B_lsr'`
- **Per-cell metric thresholds** for QA (`METRIC_*` block)
- **Neighbor QA** (`NEIGHBOR_*` block, currently disabled)
- **Paths**: `DATA_DIRS = [Path('data/main'), Path('data/nps')]`,
  `REOBSERVE_PATH = Path('artifacts/main_reobserve.json')`,
  `MOLL_CENTER_L = 120.0`

## Stage 1 — Load raw dumps (cells 1-2)

For every `.npz` under both `DATA_DIRS`:

```python
records.append({
    'path':    p,
    'survey':  'main' or 'nps',
    'session': f'{survey}/{session_dir.name}',   # e.g. 'main/session_007'
    'target':  f['target_name'],
    'corr00':  f['corr00'].astype(float),
    'corr11':  f['corr11'].astype(float),
    'lo_mhz':  f['lo_freq_mhz'],
    'noise_on': f['noise_on'],
    'time': f['time'],
    'alt': f['alt_deg'], 'az': f['az_deg'],
    'ra':  f['ra_deg'],  'dec': f['dec_deg'],
})
```

After loading, the icrs `(ra, dec)` is converted to galactic `(gl, gb)`
(rounded to 2 d.p. and integer respectively) and stored in each record.
A per-session summary is printed (UT span, dump count, duty cycle).

## Stage 1b — Sun + Moon proximity check (cells 3-4)

Defensive QA: for every dump, compute great-circle separation from the
Sun and Moon at the dump's UTC time and verify `sep > SUN_AVOID_DEG = 30°`
and `sep > MOON_AVOID_DEG = 10°`. Sun/Moon RA/Dec are computed on a
10-minute grid and linear-interpolated in `(cos RA, sin RA, dec)` to
avoid the 0/360° wrap. Reports min/median separation and lists up to 10
worst violations per body.

This catches cells the planner failed to filter out (e.g. observed
through old code with the lower 10° threshold).

## Stage 2 — fftshift + DC bin mask (cell 6)

Per dump:
- `corr00, corr11 ← np.fft.fftshift(corr0X)` — DC bin moved to channel 512
- `dc_mask` is a 1024-channel `True/False` array where channel 512 is `False`
- `corr11[~dc_mask] = np.nan`

The `dc_mask` is reused later in gain computation.

## Stage 3 — RFI channel flagging (cell 6)

`flag_rfi_channels` from `utils/rfi.py`:

1. **Sliding-window Chebyshev pseudo-continuum** (`_cheb_pseudocontinuum`):
   for each channel, fit a degree-3 Chebyshev polynomial to the
   surrounding 15-channel window after dropping local extrema and
   subsampling 70% of the survivors (seeded RNG). The polynomial value
   at the window center becomes the pseudo-continuum.
2. **Residual + sigma-MAD clip** at `RFI_SIGMA = 10σ`. Outlier channels
   are NaN'd in the dump's spectrum.
3. The mask is dump-specific; channels NaN'd here propagate through all
   downstream means.

Reflect-padded so every window is full-width.

## Stage 3b — Outlier dump filter (cell 7)

`flag_outlier_dumps` removes whole dumps whose spectral shape differs
from the rest of the cell's group by more than `SHAPE_DEV_THRESH = 0.05`
in `SHAPE_FRAC_THRESH = 5%` of channels. Group size below
`SHAPE_MIN_GROUP_SIZE = 3` skips the filter (insufficient population
statistics).

`outlier_records` is the reduced list — used for everything downstream.

## Stage 4 — Frequency switching (cell 9)

Per dump-pair `(LO1, LO2)`:

```
I_LO1, I_LO2  = corr11 at the two LOs (after RFI mask)
R_native(ν)   = (I_LO1 − I_LO2) / I_LO2
```

The two LOs (`F1_MHZ = 1419.86`, `F2_MHZ = 1421.14`) provide a 1.28 MHz
shift; after `EDGE_TRIM_MHZ = 256 kHz` is removed from each side, the
overlap region defines a common `v_lsr_overlap` velocity grid. `R` is
computed in this overlap region.

Pairs with too few viable channels after RFI flagging are dropped.

## Stage 5 — Per-cell gain calibration (cell 11)

`utils.compute_cell_gains` solves for the per-channel gain from the
noise-on / noise-off pair within each cell:

```
P_on(ν)  = mean(corr11 of cal-on dumps for this cell)
P_off(ν) = mean(corr11 of cal-off dumps for this cell)
G_raw(ν) = (P_on − P_off) / T_CAL_11           # treated as scalar 79 K
G(ν)     = smooth_gain_spectrum(G_raw, dc_mask)
```

`smooth_gain_spectrum` is a NaN-safe running median (`scipy.ndimage.median_filter`,
kernel `GAIN_SMOOTH_CHANNELS`) with NaNs interpolated, then re-masked,
plus a `GAIN_FLOOR_FRAC × band_median` floor. Channels with `G_raw ≤ 0`
become NaN.

Output keyed on `(session, gl, gb)`: `{gain_11: G(ν), gain_11_scalar: median(G)}`.

## Stage 6 — LSR correction, cross-session pair filter, combine (cell 13)

### 6a. LSR correction
`utils.vlsr_correction` shifts each pair's `R_native(ν)` onto the common
`v_lsr_overlap` grid using Earth's velocity component along the cell's
RA/Dec at the dump's `time`. Output: `R_lsr(v_LSR)`.

### 6b. Per-pair T_B in K (for population stats only)
For each pair, compute `T_B_lsr = R_lsr · T_sys_session_estimate`. This
is just a temporary unit conversion so the next filter operates on
absolute K rather than ratios — protects against mistaking a session
with anomalous T_sys for a shape outlier.

### 6c. Cross-session pair filter
`utils.flag_outlier_pairs` rejects pairs whose `T_B_lsr` deviates from
the cell's pair population by more than `PAIR_NSIGMA = 5σ` (robust MAD)
*or* whose deviant-channel fraction exceeds `PAIR_FRAC_THRESH = 0.15`.

### 6d. Insufficient-pair flag
Cells whose surviving pair count is below `MIN_VIABLE_PAIRS = 3` go
into `insufficient_set`. They keep their data but are excluded from
science maps via `qa_flagged_set`.

### 6e. Combine
Per-cell mean of surviving `R_lsr` pairs becomes `R_overlap(v_LSR)`.
Pair count is recorded. This is the input to T_sys calibration.

## Stage 7 — T_sys calibration (cell 13, closing block)

`utils.apply_tsys_calibration`:

```
P_off_nu    = mean(corr11 of cal-off dumps)        # full NFFT band
T_sys_raw   = P_off_nu / G(ν)                       # NaN where dc_mask
T_sys(ν)    = smooth_tsys_spectrum(T_sys_raw)       # bandpass only
T_sys_overlap = T_sys(ν)[overlap_mask]
T_B_overlap = R_overlap · T_sys_overlap
T_sys_scalar = median(T_sys_overlap)
```

`smooth_tsys_spectrum`:
- Single global Chebyshev fit over `x ∈ [-1, 1]`, degree `TSYS_FIT_DEGREE = 2`.
- Iterative one-sided sigma clip at `TSYS_CLIP_SIGMA = 2.0σ`
  (`TSYS_CLIP_ITERS = 5`). Positive residuals are masked as HI line
  emission; negatives are kept so the polynomial can drop into
  absorption.
- Returns a smooth bandpass-only T_sys; the HI line is clipped out so
  that `T_B = R · T_sys` doesn't double-count it.

Per-cell record now contains:
```
cell_combined[(gl, gb)] = {
    'T_B':      <NFFT_overlap,>     # K, line spectrum
    'R':        <NFFT_overlap,>     # ratio
    'T_sys':    scalar              # K, band-median
    'T_sys_nu': <NFFT,>              # K, full-band
    'gain_11': ...
    'n_pairs': N,
}
```

## Stage 8 — Per-cell metrics (cell 15, partly disabled)

`utils.compute_cell_metrics` computes per-cell QA metrics from `T_B`:
integrated `W = Σ T_B · dv`, peak velocity, peak amplitude, noise
floor, peak prominence, etc. Used as input to neighbor QA (currently
disabled).

## Stage 9 — Neighbor QA (cell 15) — **DISABLED**

`utils.neighbor_qa` would compare each cell's `W` and `peak_v` to a
beam-weighted local plane fit, flagging deviations beyond
`W_Z_THRESH = 3σ`, `W_FRAC_THRESH = 0.30`, etc. Currently the entire
cell is line-commented and `neighbor_cells = []` is a no-op stub.
Re-enable: search for `Neighbor QA disabled`.

## Stage 10 — T_sys QA (cell 17) — **DISABLED**

Would flag cells whose scalar `T_sys` deviates from the population
beyond `T_SYS_NSIGMA = 4σ` or falls outside `[T_SYS_ABS_LO = 80 K,
T_SYS_ABS_HI = 300 K]`. Currently line-commented and `tsys_flagged = []`
is a no-op stub. Re-enable: search for `T_sys QA disabled`.

## Stage 11 — `qa_flagged_set` aggregation (cells 18-19, 21, 23)

`qa_flagged_set` is the union of:
- `neighbor_cells` (currently empty)
- `tsys_flagged` (currently empty)
- `insufficient_set` (cells with `n_pairs < 3`)

So with QA in its current state, **the only effective exclusion criterion
is "fewer than 3 viable pairs after dump+pair filtering"**.

A `reobs_map` is built: `(l, b) -> {'l', 'b', 'reason'}` listing every
QA-flagged or insufficient cell with its reason. Written to
`artifacts/main_reobserve.json` so the next planning pass forces those
cells to be re-observed.

## Stage 12 — Integrated maps (cells 21, 23)

### 7a. R-map (cell 21)
`W_R(l, b) = Σ R_overlap(v) · dv` for v in the line window. Mollweide
plot via `plotters.plot_survey_mollweide_gridded` with beam-weighted
gridding (no QA-flagged cells). Diagnostic: should be roughly
proportional to the temperature map.

### 7. T_B-map (cell 23)
`W(l, b) = Σ T_B(v) · dv` in K·km/s. Same plot style. The science
product. Includes the never-observable overlay (red shading) from
`plotters.add_never_observable_overlay`.

## Stage 13 — State export (cell 24)

The reduced-survey state is pickled to `artifacts/scan_load_state.pkl`:

```python
{
    'cell_combined': {(gl, gb): {T_B, R, T_sys, T_sys_nu, n_pairs, ...}, ...},
    'qa_flagged_set': set of (gl, gb),
    'v_lsr_overlap': 1-D array,
    'dv_kms': float,
    'tsys_by_session_cell': {(session, gl, gb): T_sys, ...},
    'sessions': [list],
    'T_CAL_11': 79.0,
    ...
}
```

Consumers:
- `main_scan_load_lv.ipynb` — l-v products (Mollweide, b≈0 strip)
- `main_scan_load_diagnostics.ipynb` — per-session T_sys histogram,
  random-pointing cal vs obs, alt/az pointing trace

## Final data products

| product | what | where |
|---|---|---|
| `cell_combined` | per-cell calibrated T_B, R, T_sys, T_sys_nu, n_pairs | `scan_load_state.pkl` |
| Mollweide W-map | beam-weighted integrated HI intensity over the sky | inline plot, cell 23 |
| Mollweide W_R-map | same in R units (sanity check) | inline plot, cell 21 |
| `survey_manifest.json` (legacy) | completeness state | written elsewhere |
| `main_reobserve.json` | force-reobserve list driven by QA flags | `artifacts/` |

## Approximations / caveats

- **Scalar `T_cal`**: real T_cal has 5-15% spectral structure; folded
  into a constant absorbs into G(ν) and feeds back into T_sys.
- **`TSYS_FIT_DEGREE = 2`**: assumes a smooth quadratic bandpass over
  3.2 MHz. Edge channels are pre-trimmed (`EDGE_TRIM_MHZ = 0.256`) so the
  fit sees only the central well-behaved region.
- **DC bin** masked everywhere via `dc_mask`.
- **Pol 0 (`corr00`)** is loaded but not used for science: the noise
  diode coupling is too non-linear at 3.2 MHz. Stokes I = `corr11`.
- **NPS l-wrap**: cells stored with `l_name = (l % 360)` so the same sky
  position has the same filename across both surveys.
- **Currently disabled** QA: T_sys outliers and neighbor-plane W/peak_v
  deviations. Only `insufficient_set` actively excludes cells.
