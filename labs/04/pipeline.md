# Lab 4 HI Survey Pipeline

This document summarizes the end-to-end Lab 4 pipeline from data collection to the combined HI map after QA flags.

## 1. Collection

The raw inputs are OTF scan dumps stored under `data/lab04/streaming/session_*/`. Each dump is a `.npz` file containing the dual-polarization spectra and metadata such as target name, LO frequency, timestamps, sky coordinates, and noise-diode state.

The collection is organized by session and cell. Science dumps live in `obs_{l}_{b}/` subdirectories; calibration dumps (`noise_on=True`) live in `cal_{l}_{b}/` subdirectories. The notebook loads every available dump from the discovered `obs_*` directories.

## 2. Raw load and metadata parsing

The loader builds a `records` list, one entry per dump. For each record it stores:

- `corr00` and `corr11`
- LO frequency
- timestamp
- azimuth, altitude, RA, Dec
- target name
- session label

The target name is parsed as `obs_{l}_{b}` so every dump can be grouped into a survey cell by its integer galactic coordinates. The notebook counts the available dumps per session.

## 3. Spectral preprocessing

Each dump is shifted with `np.fft.fftshift` so the spectrum is centered on baseband zero. The pipeline then:

- masks the DC bin at channel 512
- masks the two bins below DC for LO = 1420 MHz to suppress leakage near the HI line
- forms Stokes I as `corr00 + corr11`
- applies RFI flagging with a rolling median and MAD estimator

The RFI step flags both positive spikes and negative dropouts. Flagged channels are replaced with `NaN` in place.

## 4. Dump-level spectral outlier rejection

After channel-level RFI handling, the notebook removes entire dumps whose spectral shape is inconsistent with the group median. The grouping is by data release, cell, and LO frequency.

The rule is:

- compare each dump to the group median spectrum
- compute the fraction of channels whose ratio to the median deviates by more than 10 percent
- reject the dump if more than 20 percent of channels fail that test

This removes gross spectral outliers before any frequency-switching or mapping is done.

## 5. Galactic coordinates and LSR correction

The integer galactic coordinates `(l, b)` are parsed directly from the target name (`obs_{l}_{b}`) and define the map grid.

For the frequency-switched spectra, each dump also receives an LSR velocity correction computed from the observation time and pointing direction. The correction combines the heliocentric velocity correction with the solar motion toward the local standard of rest.

The notebook keeps two versions of the frequency-switched result:

- per-session spectra in the topocentric frame
- a combined spectrum interpolated onto a common LSR velocity grid

## 6. Frequency switching

The two LO settings are paired to form a common overlap band. The notebook trims the band edges by 256 kHz on each side and keeps only the overlap channels shared by both LO settings.

For each cell, it computes:

$$R = (I_1 - I_2) / I_2$$

where `I1` and `I2` are the two LO spectra.

The result is stored as `R_overlap`, which is the common spectrum used downstream for integrated intensity, peak velocity, and QA.

## 7. Quality assurance

The QA stage has two layers, applied in sequence before any plotting.

### 7a. Session-level spectral consistency (disabled)

This check is implemented but currently commented out -- visual inspection
showed that all multi-session cells are consistent and the flagging was
producing false positives.  The code remains in both notebooks for future
use if needed.

When active, it compares each session's frequency-switched R spectrum to the
channel-wise median across all sessions at that cell, using broadband and
narrowband z-scores normalized by off-signal noise.

### 7b. Neighbor-based spatial QA

The pipeline applies spatial QA to the combined
LSR-aligned cell results.  From each cell's overlap spectrum it computes:

- velocity-integrated intensity: $W = \sum R(v)\,\Delta v$
- peak velocity from a peak finder on the smoothed overlap profile

The reference for each cell is built from nearby sky cells, not from scan
order.  Neighbor selection uses actual angular separation on the sky and keeps
cells within about 4.5 deg, which is wide enough to include the beam-coupled
local neighborhood for a 3.4 deg HPBW dish.

The QA model does two things:

1. It weights neighbors by beam overlap using a Gaussian beam model.
2. It fits a local plane so smooth spatial gradients are not mistaken for
   anomalies.

For each cell, the QA stage computes residuals for W and peak velocity.
A cell is flagged if:

- `W`: fractional residual greater than 30 percent and z-score greater
  than 3.5
- `v_peak`: absolute residual greater than 15 km/s and z-score greater
  than 4.0

### Manifest integration

The diagnostics notebook (`02a_scan_diagnostics.ipynb`) runs the neighbor QA
and excludes flagged cells from the survey manifest, so they are re-observed
in subsequent sessions.

## 8. Cell-level map products and plotting

After QA, the notebook produces all visualization products from the clean (unflagged) cells:

- per-session heatmaps of `W`
- a combined Mollweide HI map (excluding flagged cells)
- per-session spectra grids saved to `spectra_per_dr.pdf`
- summary counts for total dumps, flagged cells, and surviving map cells

## 9. Final outputs

The pipeline produces:

- cleaned combined HI Mollweide map (post-QA)
- per-session HI heatmaps
- per-session spectra PDF
- updated survey manifest (QA-flagged cells excluded from completeness counts)
- summary counts for total dumps, flagged cells, and surviving map cells

## Notes on interpretation

- Fractional residuals are best for plotting and human interpretation.
- z-scores are best for automated flagging.
- Peak velocity should be interpreted in km/s residuals, not as a fractional quantity.
- Beam overlap matters because the scan step is smaller than the beam size, so neighboring cells are not independent.

The implementation lives in [02_scan_load.ipynb](/home/ikaros/projects/ay-121/labs/04/notebooks/02_scan_load.ipynb) and is consistent with the survey architecture described in [ARCHITECTURE.md](/home/ikaros/projects/ay-121/labs/04/ARCHITECTURE.md).