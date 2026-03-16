# API: Models

Source: `ugradiolab/models/record.py`, `ugradiolab/models/spectrum.py`

---

## Record

`@dataclass(frozen=True)`

Unified capture metadata record for both observation and calibration captures. Stores raw I/Q data in memory as `complex64` but serialises to `int8` on disk.

> **int8 / complex64 split**: In memory `data` is `complex64` with shape `(nblocks, nsamples)`. On disk (`.npz`) it is `int8` with shape `(nblocks, nsamples, 2)` — the last axis is `[I, Q]`. `Record.load` reconstructs complex64 automatically. If you open the file with `np.load` directly, you will see the int8 representation.

### Fields

| Field | Type (in-memory) | Required | Units | Constraint | Description |
|---|---|---|---|---|---|
| `data` | `np.ndarray` complex64 `(nblocks, nsamples)` | yes | — | finite; components in `[-128, 127]` | Raw I/Q samples |
| `sample_rate` | `float` | yes | Hz | > 0 | SDR sample rate |
| `center_freq` | `float` | yes | Hz | finite | SDR LO centre frequency |
| `gain` | `float` | yes | dB | finite | SDR gain |
| `direct` | `bool` | yes | — | — | Direct sampling mode |
| `unix_time` | `float` | yes | s (Unix epoch) | finite | Capture timestamp |
| `jd` | `float` | yes | days | finite | Julian Date at capture |
| `lst` | `float` | yes | **radians** | finite | Local Sidereal Time |
| `alt` | `float` | yes | degrees | finite | Telescope altitude |
| `az` | `float` | yes | degrees | finite | Telescope azimuth |
| `obs_lat` | `float` | yes | degrees | finite | Observer latitude |
| `obs_lon` | `float` | yes | degrees | finite | Observer longitude |
| `obs_alt` | `float` | yes | metres | finite | Observer altitude |
| `nblocks` | `int` | yes | — | > 0 | Number of captured blocks |
| `nsamples` | `int` | yes | — | > 0 | Samples per block |
| `siggen_freq` | `float \| None` | no | Hz | finite if set | Signal generator frequency |
| `siggen_amp` | `float \| None` | no | dBm | finite if set | Signal generator amplitude |
| `siggen_rf_on` | `bool \| None` | no | — | — | Signal generator RF output state |

> **LST is in radians**: `lst` is the output of `ugradio.timing.lst(jd, lon)` which returns radians. All other angular fields are degrees.

### Properties

| Property | Return type | Description |
|---|---|---|
| `uses_synth` | `bool` | `True` if all three `siggen_*` fields are populated |

### Methods

#### `Record.from_sdr(data, sdr, alt_deg, az_deg, lat, lon, obs_alt, synth)`

Class method. Builds a `Record` from hardware state and raw captured data.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | array-like `(nblocks, nsamples, 2)` int8 | required | Raw [I, Q] from SDR |
| `sdr` | `ugradio.sdr.SDR` | required | Configured SDR instance |
| `alt_deg` | `float` | required | Telescope altitude in degrees |
| `az_deg` | `float` | required | Telescope azimuth in degrees |
| `lat` | `float` | `nch.lat` | Observer latitude in degrees |
| `lon` | `float` | `nch.lon` | Observer longitude in degrees |
| `obs_alt` | `float` | `nch.alt` | Observer altitude in metres |
| `synth` | `SignalGenerator \| None` | `None` | Connected signal generator |

Returns `Record`.

#### `Record.load(filepath)`

Class method. Loads a `.npz` file and returns a `Record`.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str \| Path` | Path to a `.npz` file written by `save` |

Returns `Record`. Raises `ValueError` if required keys are missing, data shape/dtype is unexpected, or `nblocks`/`nsamples` are inconsistent.

#### `Record.save(filepath)`

Instance method. Saves this `Record` to a `.npz` file. Serialises `data` as int8 `(nblocks, nsamples, 2)`.

| Parameter | Type | Description |
|---|---|---|
| `filepath` | `str \| Path` | Destination path |

### Validation Rules (`__post_init__`)

1. `data` must be 2-D and numeric; coerced to `complex64`
2. Both real and imaginary components must be finite
3. Both components must be integer-valued and in `[-128, 127]`
4. `data.shape` must equal `(nblocks, nsamples)`
5. All scalar float fields must be finite real scalars
6. `sample_rate` must be > 0
7. `nblocks` and `nsamples` must be positive integers

---

## Spectrum

`@dataclass(frozen=True)`

Integrated power spectrum with full observation metadata. Produced by `Spectrum.from_record` from a `Record`; shares all metadata fields with `Record` but stores only reduced data (no raw I/Q).

### Additional Fields (not in Record)

| Field | Type | Required | Units | Constraint | Description |
|---|---|---|---|---|---|
| `psd` | `np.ndarray` float64 `(nsamples,)` | yes | per-bin | finite | Mean power spectrum across blocks |
| `std` | `np.ndarray` float64 `(nsamples,)` | yes | per-bin | finite; ≥ 0 | Standard error of mean PSD |
| `freqs` | `np.ndarray` float64 `(nsamples,)` | yes | Hz | finite | Frequency axis, DC-centred, absolute |

All metadata fields (`sample_rate`, `center_freq`, `gain`, `direct`, `unix_time`, `jd`, `lst`, `alt`, `az`, `obs_lat`, `obs_lon`, `obs_alt`, `nblocks`, `nsamples`, `siggen_freq`, `siggen_amp`, `siggen_rf_on`) are identical to `Record` — see table above.

> **PSD normalisation**: `psd` is **per-bin**, not per-Hz density. Computed as `|FFT(block)|² / nsamples²`. Total power `sum(psd)` equals `mean(|x[n]|²)` by Parseval's theorem and is independent of `nsamples`. To compare PSDs from captures with different `nsamples`, divide by `bin_width = sample_rate / nsamples` to get a per-Hz density.

> **DC removal**: `Spectrum.from_record` subtracts `data.mean(axis=1, keepdims=True)` from each block before the FFT. The DC bin is explicitly zeroed as a consequence of the measurement process, not just masked for display. `mask_dc_bin` exists for plotting convenience.

### Properties

| Property | Return type | Description |
|---|---|---|
| `uses_synth` | `bool` | `True` if all three `siggen_*` fields are populated |
| `freqs_mhz` | `np.ndarray` | Frequency axis in MHz |
| `bin_width` | `float` | Frequency resolution in Hz: `sample_rate / nsamples` |
| `total_power` | `float` | `sum(psd)` — mean square of time samples (Parseval) |
| `total_power_db` | `float` | `10 * log10(total_power)` — raises `ValueError` if ≤ 0 |
| `total_power_sigma` | `float` | Uncertainty on total power: `sqrt(sum(std²))` |

### Methods

#### `Spectrum.from_record(record)`

Class method. Computes a `Spectrum` from a `Record`.

| Parameter | Type | Description |
|---|---|---|
| `record` | `Record` | A validated `Record` instance |

DC-removes each block, FFTs, computes `|FFT|² / nsamples²`, then averages across blocks.

Returns `Spectrum`.

#### `Spectrum.from_data(filepath)`

Class method. Loads a `Record` from a `.npz` file and immediately computes a `Spectrum`.

Returns `Spectrum`.

#### `Spectrum.load(filepath)` / `Spectrum.save(filepath)`

Identical semantics to `Record.load`/`Record.save`. No raw I/Q data — only `psd`, `std`, `freqs` and metadata.

#### `bin_at(freq_hz)`

Returns `int` — index of the frequency bin closest to `freq_hz` (Hz).

#### `frequency_axis_mhz(mode='absolute')`

| `mode` | Returns |
|---|---|
| `'absolute'` | Sky frequency axis in MHz |
| `'baseband'` | Offset from LO centre in MHz |

#### `velocity_axis_kms(rest_freq_hz, velocity_shift_kms=0.0)`

Radio-definition Doppler velocity axis in km/s. Positive = receding.

#### `psd_values(*, smooth_kwargs=None, mask_dc=False)`

Returns a `float64` copy of `psd`, optionally smoothed (passes `smooth_kwargs` to `smooth`) and/or with the DC bin masked as `NaN`.

#### `std_bounds(values=None, *, floor=None)`

Returns `(lo, hi)` — one-sigma envelopes around `values` (defaults to `psd`). `floor` clips both envelopes to a minimum value.

#### `mask_dc_bin(values=None)`

Returns a copy with the bin nearest `center_freq` set to `NaN`.

#### `smooth(method='gaussian', **kwargs)`

Returns a smoothed copy of `psd`. The original `psd` is unchanged.

| `method` | kwargs | Default |
|---|---|---|
| `'gaussian'` | `sigma` (bins) | `sigma=32` |
| `'savgol'` | `window_length`, `polyorder` | `window_length=129`, `polyorder=3` |
| `'boxcar'` | `M` (bins) | `M=64` |

#### `ratio_to(other, *, smooth_kwargs=None)`

Returns channel-wise ratio `self.psd / other.psd` as `float64` array. Shapes must match.

#### `ratio_std_to(other)`

Propagates raw PSD standard errors into the ratio uncertainty using standard error propagation. Returns `float64` array.
