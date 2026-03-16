# API: Run

Source: `ugradiolab/run/experiment.py`, `ugradiolab/run/queue.py`

---

## Experiment (abstract base)

`@dataclass` — abstract base class for all experiment types.

### Fields and Defaults

| Field | Type | Default | Units | Description |
|---|---|---|---|---|
| `nsamples` | `int` | `32768` | — | Samples per SDR capture block |
| `nblocks` | `int` | `1` | — | Number of blocks to capture |
| `sample_rate` | `float` | `2.56e6` | Hz | SDR sample rate |
| `center_freq` | `float` | `1420e6` | Hz | SDR LO centre frequency |
| `gain` | `float` | `0.0` | dB | SDR gain |
| `direct` | `bool` | `False` | — | Direct sampling mode |
| `outdir` | `str` | `'data/'` | — | Output directory (created if absent) |
| `prefix` | `str` | `'exp'` | — | Output filename prefix |
| `alt_deg` | `float` | `0.0` | degrees | Telescope altitude for Record metadata |
| `az_deg` | `float` | `0.0` | degrees | Telescope azimuth for Record metadata |
| `lat` | `float` | `nch.lat` | degrees | Observer latitude |
| `lon` | `float` | `nch.lon` | degrees | Observer longitude |
| `obs_alt` | `float` | `nch.alt` | metres | Observer altitude |

### `run(sdr, synth=None)` (abstract)

Subclasses implement this method. Returns `str` — the path to the saved `.npz` file.

### Output Filename Format

```
{outdir}/{prefix}_{tag}_{YYYYMMDD_HHMMSS}.npz
```

where `tag` is `'cal'` for calibration experiments and `'obs'` for sky observations.

**Note**: `_capture` discards the first block of the SDR capture to flush the stale buffer. This means `nblocks+1` blocks are actually requested from the SDR; only the last `nblocks` are stored.

---

## CalExperiment

`@dataclass(frozen=False)` — subclass of `Experiment`.

Calibration experiment that drives a signal generator and captures with the SDR.

### Additional Fields

| Field | Type | Default | Units | Description |
|---|---|---|---|---|
| `siggen_freq_mhz` | `float` | `1420.405751768` | MHz | Signal generator CW frequency |
| `siggen_amp_dbm` | `float` | `-80.0` | dBm | Signal generator amplitude |

### `run(sdr, synth)`

Executes the calibration:
1. Reconfigures the SDR to match experiment parameters
2. Sets signal generator frequency and amplitude
3. Enables RF output (`synth.rf_on()`)
4. Captures data and saves as `Record`
5. **Always** calls `synth.rf_off()` in a `finally` block, even if capture raises an exception

```python
try:
    synth.set_freq_mhz(...)
    synth.set_ampl_dbm(...)
    synth.rf_on()
    record = self._capture(sdr, synth=synth)
    record.save(path)
finally:
    synth.rf_off()   # guaranteed cleanup
```

Raises `ValueError` if `synth` is `None`.

Returns `str` — path to the saved `.npz` file.

---

## ObsExperiment

`@dataclass(frozen=False)` — subclass of `Experiment`.

Sky observation experiment. No additional fields beyond `Experiment`.

### `run(sdr, synth=None)`

Reconfigures the SDR and captures data. `synth` is accepted but ignored.

Returns `str` — path to the saved `.npz` file.

---

## QueueRunner

Manages and executes an ordered sequence of experiments.

### Constructor

```python
QueueRunner(experiments, sdr, synth=None, confirm=True)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiments` | iterable of `Experiment` | required | Ordered list of experiments to run |
| `sdr` | `ugradio.sdr.SDR` | required | SDR instance shared by all experiments |
| `synth` | `SignalGenerator \| None` | `None` | Signal generator passed to each `run` call |
| `confirm` | `bool` | `True` | Whether to prompt for confirmation before each experiment |

### `run()`

Iterates the experiment queue. Returns `list[str]` — paths of all saved `.npz` files.

**Interactive confirmation**: when `confirm=True`, before each experiment the runner prints a summary and waits for keyboard input:

| Key | Action |
|---|---|
| Enter | Run the experiment |
| `s` | Skip this experiment |
| `q` | Abort the remaining queue |
