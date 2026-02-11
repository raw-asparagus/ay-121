# ugradiolab API Reference

## Package Layout

```
ugradiolab/
    __init__.py              # re-exports all public symbols
    experiment.py            # Experiment dataclasses + queue runner
    lab.py                   # combined SDR+siggen workflows
    drivers/
        __init__.py
        siggen.py            # SignalGenerator (USBTMC/SCPI)
        sdr_utils.py         # SDR capture helpers
    data/
        __init__.py
        schema.py            # .npz save/load (cal + obs schemas)
```

**External dependencies** (installed separately):

| Package | Purpose |
|---------|---------|
| `ugradio.sdr` | RTL-SDR dongle interface (wraps pyrtlsdr/librtlsdr) |
| `ugradio.timing` | Timestamps: `unix_time()`, `julian_date()`, `lst()` |
| `ugradio.nch` | Observer location constants (lat, lon, alt) |
| `numpy` | Array operations and `.npz` file I/O |

---

## 1. Signal Generator — `drivers/siggen.py`

### `SignalGenerator` class

Communicates with an Agilent/Keysight N9310A via USBTMC, sending
short-form SCPI commands directly to `/dev/usbtmc0`.

```python
from ugradiolab.drivers.siggen import SignalGenerator

sg = SignalGenerator(device='/dev/usbtmc0')
```

The constructor opens the device file and validates the instrument
identity via `*IDN?` (checks for `"N9310A"` in the response).

#### Methods

| Method | SCPI | Description |
|--------|------|-------------|
| `set_freq_mhz(freq)` | `FREQ:CW <freq> MHz` | Set CW frequency in MHz |
| `get_freq()` | `FREQ:CW?` | Returns current frequency in **Hz** (float) |
| `set_ampl_dbm(amp)` | `AMPL:CW <amp> dBm` | Set CW amplitude in dBm |
| `get_ampl()` | `AMPL:CW?` | Returns current amplitude in **dBm** (float) |
| `rf_on()` | `RFO:STAT ON` | Enable RF output |
| `rf_off()` | `RFO:STAT OFF` | Disable RF output |
| `rf_state()` | `RFO:STAT?` | Returns `True` if RF is on |
| `close()` | — | Close the USBTMC device handle |

**Timing**: A 0.3 s delay is enforced after each write to avoid
overwhelming the instrument's command buffer.

### Convenience functions

```python
from ugradiolab.drivers.siggen import connect, set_signal, freq_sweep
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `connect(device)` | `connect(device='/dev/usbtmc0') -> SignalGenerator` | Open and validate a connection |
| `set_signal(synth, freq_mhz, amp_dbm, rf_on=True)` | — | Set frequency, amplitude, and RF state in one call |
| `freq_sweep(synth, start, stop, step)` | yields `float` | Generator: steps frequency from `start` to `stop` (MHz), yielding each value after setting it |

---

## 2. SDR Capture — `drivers/sdr_utils.py`

Wraps `ugradio.sdr.SDR` with helpers that handle the **stale first
block** (the RTL-SDR ring buffer retains data from before the capture
request — block 0 is always discarded).

```python
from ugradiolab.drivers.sdr_utils import capture_and_fft, power_spectrum, collect_time_series
```

### Internal: `_capture(sdr, nsamples, nblocks)`

Requests `nblocks + 1` blocks from `sdr.capture_data()` and returns
`data[1:]`, discarding the stale buffer.  All public functions below
use `_capture` internally.

### Functions

#### `capture_and_fft(sdr, nsamples=2048, nblocks=1)`

Capture raw samples and compute the FFT per block.

- **Direct mode** (`sdr.direct=True`): real-valued → `np.fft.rfft`,
  frequency axis via `rfftfreq`.
- **I/Q mode** (`sdr.direct=False`): complex I+jQ → `np.fft.fft` with
  `fftshift`, frequency axis centered on `center_freq`.

**Returns**: `(freqs, fft_data)` — frequency axis in Hz, complex FFT
array with shape `(nblocks, ...)`.

#### `power_spectrum(sdr, nsamples=2048, nblocks=1)`

Calls `capture_and_fft`, then computes `mean(|FFT|^2)` over blocks.

**Returns**: `(freqs, psd)` — frequency axis in Hz, averaged PSD array.

#### `collect_time_series(sdr, nsamples=2048, nblocks=1)`

Capture raw voltage data with a time axis.

**Returns**: `(t, data)` — time in seconds (`np.arange(nsamples) / sample_rate`),
raw int8 array with shape `(nblocks, nsamples)` or `(nblocks, nsamples, 2)`.

---

## 3. Data Schemas — `data/schema.py`

All captures are stored as `.npz` files (NumPy compressed archives).
Two schemas exist, sharing a common core of SDR parameters and
timestamps, with type-specific metadata.

```python
from ugradiolab.data.schema import save_cal, save_obs, load
```

### Common fields (both schemas)

| Key | dtype | Description |
|-----|-------|-------------|
| `data` | int8 | Raw samples: `(nblocks, nsamples)` or `(nblocks, nsamples, 2)` |
| `sample_rate` | float64 | SDR sample rate (Hz) |
| `center_freq` | float64 | SDR LO frequency (Hz); 0 if direct mode |
| `gain` | float64 | SDR gain (dB) |
| `direct` | bool | True = direct sampling, False = I/Q |
| `unix_time` | float64 | Capture timestamp (seconds since epoch) |
| `jd` | float64 | Julian Date |
| `lst` | float64 | Local Sidereal Time (radians) |
| `alt` | float64 | Telescope altitude (degrees) |
| `az` | float64 | Telescope azimuth (degrees) |
| `observer_lat` | float64 | Observer latitude (degrees) |
| `observer_lon` | float64 | Observer longitude (degrees) |
| `observer_alt` | float64 | Observer altitude (meters) |
| `nblocks` | int64 | Number of blocks |
| `nsamples` | int64 | Samples per block |

### Calibration schema — additional fields

| Key | dtype | Description |
|-----|-------|-------------|
| `siggen_freq` | float64 | Signal generator frequency (Hz) |
| `siggen_amp` | float64 | Signal generator amplitude (dBm) |
| `siggen_rf_on` | bool | RF output state at capture time |

### `save_cal(filepath, data, sdr, synth, alt_deg, az_deg, lat, lon, observer_alt)`

Queries the `SignalGenerator` for its current state (`get_freq()`,
`get_ampl()`, `rf_state()`), reads SDR parameters from the `SDR`
object, captures timestamps via `ugradio.timing`, and writes
everything to a `.npz` file.

### `save_obs(filepath, data, sdr, alt_deg, az_deg, lat, lon, observer_alt)`

Same as `save_cal` but without signal generator fields.  Observer
location defaults to `ugradio.nch` (New Campbell Hall).

### `load(filepath)`

Returns `np.load(filepath, allow_pickle=False)` — a dict-like
`NpzFile` object.  Access fields by key:

```python
f = load('data/obs.npz')
f['data']          # (nblocks, nsamples, 2) int8
f['alt']           # 90.0
f['jd']            # 2461082.73
f['sample_rate']   # 2560000.0
```

---

## 4. Experiment Specification — `experiment.py`

Experiments are Python **dataclasses** that bundle all parameters
needed for a single capture.  They form an inheritance hierarchy:

```
Experiment (base)
├── CalExperiment      — signal generator + pointing
└── ObsExperiment      — pointing only (no signal generator)
```

### `Experiment` (base — not instantiated directly)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nsamples` | int | 2048 | Samples per block |
| `nblocks` | int | 1 | Blocks to capture |
| `sample_rate` | float | 2.56e6 | SDR sample rate (Hz) |
| `center_freq` | float | 0.0 | SDR LO frequency (Hz) |
| `gain` | float | 0.0 | SDR gain (dB) |
| `direct` | bool | True | Sampling mode |
| `outdir` | str | `'.'` | Output directory |
| `prefix` | str | `'exp'` | Filename prefix |

Method `_configure_sdr(sdr)` applies these parameters to an existing
`SDR` object (sets direct/I/Q mode, center frequency, sample rate, gain).

### `CalExperiment(Experiment)`

Additional fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `siggen_freq_mhz` | float | 0.0 | Tone frequency (MHz) |
| `siggen_amp_dbm` | float | -10.0 | Tone amplitude (dBm) |
| `alt_deg` | float | 0.0 | Telescope altitude (deg) |
| `az_deg` | float | 0.0 | Telescope azimuth (deg) |
| `lat` | float | nch.lat | Observer latitude |
| `lon` | float | nch.lon | Observer longitude |
| `observer_alt` | float | nch.alt | Observer altitude (m) |

`run(sdr, synth)`:
1. Reconfigures the SDR
2. Sets the signal generator via `set_signal()`
3. Captures data (discards stale block)
4. Saves to `.npz` via `save_cal()`
5. Returns the output filepath

### `ObsExperiment(Experiment)`

Additional fields: `alt_deg`, `az_deg`, `lat`, `lon`, `observer_alt`
(same defaults as `CalExperiment`).

`run(sdr, synth=None)`:
1. Reconfigures the SDR
2. Captures data (discards stale block)
3. Saves to `.npz` via `save_obs()`
4. Returns the output filepath

### `run_queue(experiments, sdr, synth=None, confirm=True)`

Executes a list of experiments sequentially on shared hardware.

**Before each step**, prints a summary:

```
[3/13] Z-TONE-PWR1 (CalExperiment)
  alt=90.0  az=0.0
  nsamples=2048  nblocks=10  sample_rate=2.56 MHz
  siggen: 1421.2058 MHz, -41 dBm
  [Enter]=run  s=skip  q=quit:
```

**Interactive controls** (when `confirm=True`):
- **Enter** — run the experiment
- **s** — skip it (no file produced)
- **q** — abort the rest of the queue

**Returns**: list of output filepaths (one per executed experiment).

### Output filename convention

```
{prefix}_{type}_{YYYYMMDD}_{HHMMSS}.npz
```

- `type`: `cal` or `obs`
- Timestamp from `time.strftime` at capture time

Example: `Z-TONE-PWR1_cal_20260211_143022.npz`

---

## 5. Pipeline: Configuration to Data Collection

The end-to-end workflow has four stages:

```
Define  ──>  Initialize  ──>  Execute  ──>  Load
params       hardware         queue         data
```

### Stage 1 — Define experiment parameters

```python
from ugradiolab.experiment import CalExperiment, ObsExperiment

# Shared SDR configuration
sdr_cfg = dict(
    direct=False,
    center_freq=1420e6,
    sample_rate=2.56e6,
    gain=0.0,
    nsamples=4096,
    nblocks=10,
    outdir='data/cal',
)

# Build an experiment list
experiments = [
    # Baseline (no tone)
    ObsExperiment(prefix='BASE-PRE', alt_deg=90, az_deg=0, **sdr_cfg),

    # Calibration tone
    CalExperiment(prefix='TONE-1420', siggen_freq_mhz=1421.2058,
                  siggen_amp_dbm=-35, alt_deg=90, az_deg=0, **sdr_cfg),

    # Post-baseline
    ObsExperiment(prefix='BASE-POST', alt_deg=90, az_deg=0, **sdr_cfg),
]
```

Each experiment is a **self-contained specification** — all SDR
parameters, pointing, signal generator settings, and output
configuration are captured in the dataclass fields.  No mutable
state is shared between experiments.

### Stage 2 — Initialize hardware

```python
from ugradio.sdr import SDR
from ugradiolab.drivers.siggen import connect

sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6, gain=0.0)
synth = connect()  # opens /dev/usbtmc0
```

A single `SDR` and `SignalGenerator` instance are reused across all
experiments.  Each experiment's `run()` method reconfigures the SDR
to match its own parameters before capturing.

### Stage 3 — Execute the queue

```python
from ugradiolab.experiment import run_queue

try:
    paths = run_queue(experiments, sdr=sdr, synth=synth, confirm=True)
finally:
    synth.rf_off()
    sdr.close()
```

For each experiment in the list, `run_queue`:

1. **Prints** a summary of the experiment parameters
2. **Prompts** for confirmation (if `confirm=True`)
3. **Reconfigures** the SDR to the experiment's parameters
4. **Sets** the signal generator (CalExperiment only)
5. **Captures** `nblocks + 1` blocks, discards block 0 (stale buffer)
6. **Timestamps** the capture (`unix_time`, `jd`, `lst` via `ugradio.timing`)
7. **Saves** to `.npz` with full metadata (SDR state, siggen state, pointing, observer location)
8. **Returns** the filepath

The `finally` block ensures the signal generator RF output is
disabled and the SDR is closed regardless of errors or early abort.

### Stage 4 — Load and inspect data

```python
from ugradiolab.data.schema import load

f = load('data/cal/TONE-1420_cal_20260211_143022.npz')

# Raw I/Q samples
iq = f['data']                    # (10, 4096, 2) int8

# Metadata
print(f['siggen_freq'])           # 1421205800.0 (Hz)
print(f['siggen_amp'])            # -35.0 (dBm)
print(f['sample_rate'])           # 2560000.0 (Hz)
print(f['jd'])                    # 2461082.73...
print(f['lst'])                   # 1.785... (radians)
print(f['alt'])                   # 90.0 (degrees)
```

---

## 6. CLI Scripts

Scripts live in `labs/02/scripts/` and auto-configure `sys.path` to
find `ugradiolab`.

### `lab_2_calibration.py`

13-step plan: baselines at zenith/horizontal, power sweep, frequency
offsets, and post-calibration baselines.

```bash
python lab_2_calibration.py --outdir data/lab2_cal --nsamples 2048 --nblocks 10
python lab_2_calibration.py --no-confirm   # skip interactive prompts
```

### `lab_2_cold_cal.py`

44-step plan: horn horizontal, alternating SIGGEN OFF baseline and
SIGGEN ON from -50 to -30 dBm in 1 dB steps.

```bash
python lab_2_cold_cal.py --outdir data/lab2_cold_cal --nsamples 2048 --nblocks 10
python lab_2_cold_cal.py --no-confirm
```

### Writing your own script

```python
#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ugradio.sdr import SDR
from ugradiolab.drivers.siggen import connect
from ugradiolab.experiment import ObsExperiment, CalExperiment, run_queue

def build_plan(outdir, nsamples, nblocks):
    common = dict(nsamples=nsamples, nblocks=nblocks, outdir=outdir,
                  direct=False, center_freq=1420e6, sample_rate=2.56e6)
    return [
        ObsExperiment(prefix='MY-OBS', alt_deg=45, az_deg=180, **common),
        # ... add more experiments
    ]

sdr = SDR(direct=False, center_freq=1420e6, sample_rate=2.56e6)
synth = connect()
try:
    paths = run_queue(build_plan('data/my_run', 2048, 10), sdr=sdr, synth=synth)
finally:
    synth.rf_off()
    sdr.close()
```