# ugradiolab

A purpose-built radio astronomy capture-and-processing framework for UC Berkeley's AY121 lab course. Wraps the course `ugradio` package with typed data models, coordinate utilities, hardware drivers, and an experiment runner.

## Requirements

- Python ≥ 3.12
- `ugradio` course package (provides `ugradio.sdr`, `ugradio.coord`, `ugradio.timing`, `ugradio.nch`)

| Dependency | Purpose |
|---|---|
| `numpy` | Array storage, FFT, `.npz` I/O |
| `scipy` | Spectrum smoothing (`gaussian_filter1d`, `savgol_filter`) |
| `ntplib` | NTP time synchronisation in `get_unix_time` |
| `astropy` | High-accuracy galactic→ICRS conversion in `compute_pointing` |
| `ugradio` | SDR capture, coordinate helpers, timing, site constants |

## Public Exports

All names below are importable directly from `ugradiolab`.

| Name | Type | Source module |
|---|---|---|
| `Record` | dataclass | `ugradiolab.models.record` |
| `Spectrum` | dataclass | `ugradiolab.models.spectrum` |
| `SmoothMethod` | `Literal` type alias | `ugradiolab.models.spectrum` |
| `SignalGenerator` | class | `ugradiolab.drivers.signal_generator` |
| `Experiment` | abstract dataclass | `ugradiolab.run.experiment` |
| `CalExperiment` | dataclass | `ugradiolab.run.experiment` |
| `ObsExperiment` | dataclass | `ugradiolab.run.experiment` |
| `QueueRunner` | class | `ugradiolab.run.queue` |
| `compute_pointing` | function | `ugradiolab.pointing` |
| `galactic_to_equatorial_matrix` | function | `ugradiolab.pointing` |
| `equatorial_to_altaz_matrix` | function | `ugradiolab.pointing` |
| `galactic_to_equatorial` | function | `ugradiolab.pointing` |
| `equatorial_to_galactic` | function | `ugradiolab.pointing` |
| `equatorial_to_altaz` | function | `ugradiolab.pointing` |
| `altaz_to_equatorial` | function | `ugradiolab.pointing` |
| `galactic_to_altaz` | function | `ugradiolab.pointing` |
| `altaz_to_galactic` | function | `ugradiolab.pointing` |
| `get_unix_time` | function | `ugradiolab.utils` |

## Documentation

| File | Contents |
|---|---|
| [architecture.md](architecture.md) | Module dependency diagram, data flow, experiment layer, coordinate pipeline, immutability model |
| [quickstart.md](quickstart.md) | Step-by-step guide from hardware setup to saved spectra |
| [api/models.md](api/models.md) | `Record` and `Spectrum` field tables, methods, validation rules |
| [api/pointing.md](api/pointing.md) | Coordinate conversion functions, `compute_pointing`, angle conventions |
| [api/run.md](api/run.md) | `Experiment`, `CalExperiment`, `ObsExperiment`, `QueueRunner` |
| [api/drivers.md](api/drivers.md) | `SignalGenerator` SCPI interface and hardware notes |
| [api/utils.md](api/utils.md) | `get_unix_time` NTP fallback behaviour |
| [schemas/README.md](schemas/README.md) | How to read the schema YAML files |
| [schemas/record.schema.yaml](schemas/record.schema.yaml) | Formal schema for `Record` `.npz` files |
| [schemas/spectrum.schema.yaml](schemas/spectrum.schema.yaml) | Formal schema for `Spectrum` `.npz` files |
