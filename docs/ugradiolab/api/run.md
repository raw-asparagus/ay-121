# API: Run

Source: `ugradiolab/run/experiment.py`, `ugradiolab/run/sdr_experiment.py`,
`ugradiolab/run/interf_experiment.py`, `ugradiolab/run/queue.py`

---

## Experiment (abstract base)

`@dataclass` — abstract base class for all experiment types.

### Fields and Defaults

| Field | Type | Default | Units | Description |
|---|---|---|---|---|
| `alt_deg` | `float` | `0.0` | degrees | Telescope altitude for metadata |
| `az_deg` | `float` | `0.0` | degrees | Telescope azimuth for metadata |
| `outdir` | `str` | `'data/'` | — | Output directory (created if absent) |
| `prefix` | `str` | `'exp'` | — | Output filename prefix |
| `lat` | `float` | `nch.lat` | degrees | Observer latitude |
| `lon` | `float` | `nch.lon` | degrees | Observer longitude |
| `obs_alt` | `float` | `nch.alt` | metres | Observer altitude |

### `run()` (abstract)

Subclasses implement this zero-argument method. Returns `str` — the path to the saved `.npz` file.

---

## SDRExperiment (abstract)

`@dataclass` — subclass of `Experiment`. Adds SDR hardware fields and capture helpers.

### Additional Fields

| Field | Type | Default | Units | Description |
|---|---|---|---|---|
| `sdr` | `object` | `None` | — | Initialized SDR instance (not in repr) |
| `nsamples` | `int` | `32768` | — | Samples per SDR capture block |
| `nblocks` | `int` | `1` | — | Number of blocks to capture |
| `sample_rate` | `float` | `2.56e6` | Hz | SDR sample rate |
| `center_freq` | `float` | `1420e6` | Hz | SDR LO centre frequency |
| `gain` | `float` | `0.0` | dB | SDR gain |
| `direct` | `bool` | `False` | — | Direct sampling mode |

### Output Filename Format

```
{outdir}/{prefix}_{tag}_{YYYYMMDD_HHMMSS}.npz
```

where `tag` is `'cal'` for calibration experiments and `'obs'` for sky observations.

**Note**: `_capture` discards the first block to flush the stale buffer. `nblocks+1` blocks are requested; only the last `nblocks` are stored.

---

## CalExperiment

`@dataclass` — subclass of `SDRExperiment`.

Calibration experiment that drives a signal generator and captures with the SDR.

### Additional Fields

| Field | Type | Default | Units | Description |
|---|---|---|---|---|
| `synth` | `object` | `None` | — | Initialized SignalGenerator instance (not in repr) |
| `siggen_freq_mhz` | `float` | `1420.405751768` | MHz | Signal generator CW frequency |
| `siggen_amp_dbm` | `float` | `-80.0` | dBm | Signal generator amplitude |

### `run()`

Executes the calibration using `self.sdr` and `self.synth`:
1. Reconfigures the SDR to match experiment parameters
2. Sets signal generator frequency and amplitude
3. Enables RF output (`self.synth.rf_on()`)
4. Captures data and saves as `Record`
5. **Always** calls `self.synth.rf_off()` in a `finally` block, even if capture raises

Raises `ValueError` if `self.synth` is `None`.

Returns `str` — path to the saved `.npz` file.

---

## ObsExperiment

`@dataclass` — subclass of `SDRExperiment`.

Sky observation experiment. No additional fields beyond `SDRExperiment`.

### `run()`

Reconfigures `self.sdr` and captures data.

Returns `str` — path to the saved `.npz` file.

---

## StreamingCapture

Producer-consumer streaming capture for the SNAP correlator. Saves every individual accumulator dump to its own `.npz` file.

### Constructor

```python
StreamingCapture(interferometer, snap, target_selector, outdir='data/',
                 n_writers=2, queue_maxsize=200, repoint_interval_sec=30.0,
                 on_save=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `interferometer` | `object` | required | Pointing controller |
| `snap` | `object` | required | SNAP correlator interface |
| `target_selector` | `callable` | required | Returns `(name, alt, az, ra, dec)` or `None` |
| `outdir` | `str` | `'data/'` | Root output directory |
| `n_writers` | `int` | `2` | Number of writer threads |
| `queue_maxsize` | `int` | `200` | Bounded queue size (backpressure when full) |
| `repoint_interval_sec` | `float` | `30.0` | Max time between repoints for same target |
| `on_save` | `callable` | `None` | `on_save(path, dump)` callback |

### `run()`

Starts all threads and blocks until `KeyboardInterrupt`. Orderly shutdown drains the queue so no dumps are lost.

---

## QueueRunner

Manages and executes an ordered sequence of experiments. Hardware-agnostic — each experiment carries its own hardware references.

### Constructor

```python
QueueRunner(experiments, confirm=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiments` | iterable of `Experiment` | required | Ordered list of experiments to run |
| `confirm` | `bool` | `True` | Whether to prompt for confirmation before each experiment |

### `run()`

Iterates the experiment queue. Returns `list[str]` — paths of all saved `.npz` files.

**Interactive confirmation**: when `confirm=True`, before each experiment the runner prints a summary and waits for keyboard input:

| Key | Action |
|---|---|
| Enter | Run the experiment |
| `s` | Skip this experiment |
| `q` | Abort the remaining queue |
