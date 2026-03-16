# Architecture

## Module Dependency Diagram

```
ugradiolab
├── __init__.py          (re-exports everything below)
│
├── models/
│   ├── record.py        → numpy, ugradio.nch, ugradio.timing, utils
│   └── spectrum.py      → numpy, scipy.ndimage, scipy.signal, models.record
│
├── pointing.py          → numpy, ugradio.coord, ugradio.nch, ugradio.timing,
│                          astropy.coordinates, astropy.units, utils
│
├── run/
│   ├── experiment.py        → abc, ugradio.nch
│   ├── sdr_experiment.py    → experiment, models.record, utils
│   ├── interf_experiment.py → experiment, drivers.interferometer, utils
│   └── queue.py             → (no hardware deps)
│
├── drivers/
│   └── signal_generator.py  → time
│
└── utils.py             → ntplib, ugradio.timing

External dependencies (hardware):
  ugradio.sdr.SDR        — RTL-SDR capture hardware
  SignalGenerator        — Keysight N9310A via /dev/usbtmc0
```

## Data Flow

```
Hardware
  └─ SDR (ugradio.sdr.SDR)
       │  capture_data(nsamples, nblocks+1)
       │  returns int8 array (nblocks+1, nsamples, 2)
       ↓
  Record.from_sdr(data[1:], sdr, alt_deg, az_deg, ...)
       │  • discards first block (stale buffer flush)
       │  • converts int8 [I,Q] → complex64 in-memory
       │  • stamps unix_time / JD / LST via get_unix_time()
       │  • queries SDR for sample_rate, center_freq, gain, direct
       │  • optionally queries SignalGenerator for siggen_* fields
       ↓
  Record (frozen dataclass, complex64 in-memory)
       │
       ├─ Record.save(path)
       │    • stacks [I,Q] back to int8, shape (nblocks, nsamples, 2)
       │    • np.savez → .npz on disk
       │
       └─ Spectrum.from_record(record)
              │  • DC removal: data -= data.mean(axis=1, keepdims=True)
              │  • FFT each block, fftshift
              │  • PSD = |FFT|² / nsamples²  (per-bin, not per-Hz)
              │  • psd = mean across blocks
              │  • std = std / sqrt(nblocks)  (standard error of mean)
              ↓
         Spectrum (frozen dataclass, float64)
              │
              └─ Spectrum.save(path)
                   • np.savez → .npz on disk (no raw I/Q data)

Offline path:
  Record.load(path)    → Record
  Spectrum.load(path)  → Spectrum
  Spectrum.from_data(record_path)  → Spectrum  (load + compute in one step)
```

## Experiment Layer

The experiment hierarchy is:

```
Experiment (ABC)              ← shared fields: alt_deg, az_deg, outdir, prefix, lat, lon, obs_alt
├── SDRExperiment (ABC)       ← adds sdr, nsamples, nblocks, sample_rate, center_freq, gain, direct
│   ├── CalExperiment         ← adds synth, siggen_freq_mhz, siggen_amp_dbm
│   └── ObsExperiment
└── InterfExperiment          ← adds interferometer, snap, delay_line, duration_sec, baseline_*
    ├── SunExperiment
    └── MoonExperiment
```

Hardware is bound at construction time (e.g. `ObsExperiment(sdr=sdr, ...)`), so `run()` takes no arguments. The calling script opens hardware before the loop and closes it in `finally` — experiment objects hold a *reference* to already-open hardware, not ownership.

`SDRExperiment` provides:
- `_configure_sdr()` — pushes all parameters to `self.sdr`
- `_capture(synth)` — calls `self.sdr.capture_data`, discards block 0, returns a `Record`

`QueueRunner` iterates a list of experiments back-to-back, optionally prompting for confirmation before each one. It is hardware-agnostic — hardware is embedded in each experiment.

Output filenames follow the pattern: `{outdir}/{prefix}_{tag}_{YYYYMMDD_HHMMSS}.npz` where `tag` is `'cal'`, `'obs'`, or `'corr'`.

## Coordinate Pipeline

Two separate pipelines co-exist:

### Internal matrix pipeline (6 scalar functions)

Uses the IAU 1958 / Hipparcos 3×3 rotation matrices:
```
_EQ_TO_GAL  =  3×3 float64  (equatorial → galactic)
_GAL_TO_EQ  =  _EQ_TO_GAL.T
```

Function chain for galactic → alt/az:
```
(l, b) → unit vector → _GAL_TO_EQ @ v → (RA, Dec)
(RA, Dec) → unit vector → M_eq_to_altaz(lst_rad, lat_deg) @ v → (alt, az)
```

where `M_eq_to_altaz = M_lat @ Rz(-LST)`, with `Rz(-LST)` converting RA to hour angle and `M_lat` converting HA-Dec to North-East-Up horizontal coordinates.

All inputs and outputs are **degrees**, except `lst_rad` which is **radians** (as returned by `ugradio.timing.lst(jd, lon)`).

### Astropy pipeline (`compute_pointing` only)

`compute_pointing` calls `astropy.coordinates.SkyCoord` for the galactic→ICRS conversion, then delegates to `ugradio.coord.get_altaz` for ICRS→alt/az. This path uses the current time and site location to give an accurate pointing for telescope slewing. The internal matrix is not used here.

## Immutability Model

Both `Record` and `Spectrum` are `@dataclass(frozen=True)`. Field mutation is prevented by Python's frozen dataclass mechanism. The `__post_init__` method uses `object.__setattr__` to perform type coercion and validation before the object is fully constructed — this is the only context in which attributes are written after `__init__`.

**Practical notebook impact**: you cannot do `rec.gain = 30.0`. Create a new instance instead:
```python
import dataclasses
rec2 = dataclasses.replace(rec, gain=30.0)
```

`__post_init__` enforces:
- `data` is coerced to `complex64` and validated to have integer-valued components in `[-128, 127]`
- `data.shape == (nblocks, nsamples)`
- All scalar float fields are finite real numbers
- `sample_rate > 0`
- `nblocks`, `nsamples` are positive integers
- All optional synth fields are either all `None` or all populated (checked via `uses_synth`)

## Hardware Dependency Table

| Class / Function | Hardware required |
|---|---|
| `Record.from_sdr` | RTL-SDR (`ugradio.sdr.SDR`) |
| `CalExperiment.run` | RTL-SDR (`self.sdr`) + Keysight N9310A (`self.synth`) |
| `ObsExperiment.run` | RTL-SDR (`self.sdr`) |
| `InterfExperiment.run` | Interferometer (`self.interferometer`) + SNAP (`self.snap`) |
| `QueueRunner.run` | None — hardware is embedded in each experiment |
| `compute_pointing` | None (reads system clock only) |
| `Record.load`, `Spectrum.load` | None |
| `Spectrum.from_record`, `Spectrum.from_data` | None |
| `SignalGenerator.__init__` | Keysight N9310A at `/dev/usbtmc0` |
