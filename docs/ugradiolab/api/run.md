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

## InterfExperiment

`@dataclass` — subclass of `Experiment`. Interferometric observation.

### Additional Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `interferometer` | `object` | `None` | Interferometer controller (not in repr) |
| `snap` | `object` | `None` | SNAP correlator (not in repr) |
| `delay_line` | `object` | `None` | Delay-line client (not in repr) |
| `duration_sec` | `float` | `10.0` | Integration time passed to snap.get_corr() |
| `baseline_ew_m` | `float\|None` | `None` | East-west baseline in metres; `None` disables delay |
| `baseline_ns_m` | `float` | `0.0` | North-south baseline in metres |
| `delay_max_ns` | `float\|None` | `None` | Hardware delay limit |

### `run()`

Points both antennas, optionally applies delay, captures correlation data.

Returns `str` — path to the saved `.npz` file.

---

## SunExperiment / MoonExperiment

`@dataclass` — subclasses of `InterfExperiment`.

Compute the current Sun/Moon position at `run()` time and delegate to `InterfExperiment.run()`.

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
