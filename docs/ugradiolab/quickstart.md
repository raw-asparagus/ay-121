# Quickstart

A practical walkthrough from hardware setup to saved, reduced spectra.

## Prerequisites

- [ ] RTL-SDR dongle connected and recognised by the OS
- [ ] `ugradio` course package installed and importable
- [ ] `ugradiolab` installed (`pip install -e .` from repo root)
- [ ] (For calibration) Keysight N9310A connected via USB; check `/dev/usbtmc0` exists and your user has read/write permission

---

## 1. Minimal Sky Observation

```python
import ugradio.sdr as sdr_mod
from ugradiolab import ObsExperiment, QueueRunner

# Open the SDR (reconfigured per-experiment by the runner)
sdr = sdr_mod.SDR()

exp = ObsExperiment(
    nsamples    = 32768,
    nblocks     = 64,
    sample_rate = 2.56e6,   # Hz
    center_freq = 1420.4e6, # Hz — HI line
    gain        = 30.0,
    outdir      = 'data/obs/',
    prefix      = 'galactic_plane',
    alt_deg     = 45.0,
    az_deg      = 180.0,
)

runner = QueueRunner([exp], sdr, confirm=False)
paths = runner.run()
print(paths)  # ['data/obs/galactic_plane_obs_20260316_123456.npz']
```

---

## 2. Load and Inspect a Record

```python
from ugradiolab import Record

rec = Record.load('data/obs/galactic_plane_obs_20260316_123456.npz')

print(rec.nblocks, rec.nsamples)    # e.g. 64, 32768
print(rec.center_freq / 1e6)        # 1420.4 MHz
print(rec.data.shape, rec.data.dtype)  # (64, 32768) complex64
# Note: on disk this is stored as int8 shape (64, 32768, 2)
```

> **Note**: `rec.data` in memory is `complex64` with shape `(nblocks, nsamples)`.
> The `.npz` file stores `data` as `int8` with shape `(nblocks, nsamples, 2)` — the
> last axis is `[I, Q]`. `Record.load` reconstructs complex64 automatically.

---

## 3. Compute and Plot a Spectrum

```python
import matplotlib.pyplot as plt
from ugradiolab import Spectrum

spec = Spectrum.from_record(rec)

# Or equivalently, in one step:
# spec = Spectrum.from_data('data/obs/galactic_plane_obs_20260316_123456.npz')

freqs_mhz = spec.frequency_axis_mhz('absolute')
psd       = spec.psd_values(mask_dc=True)  # NaN at DC bin

fig, ax = plt.subplots()
ax.plot(freqs_mhz, psd)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('PSD (per bin)')
ax.set_title('HI spectrum')
plt.show()
```

To convert to velocity:
```python
HI_FREQ = 1420.405751768e6  # Hz
vel_kms = spec.velocity_axis_kms(rest_freq_hz=HI_FREQ)
ax.set_xlabel('LSR velocity (km/s)')
```

---

## 4. Calibration Run

```python
from ugradiolab import CalExperiment, SignalGenerator

synth = SignalGenerator()  # opens /dev/usbtmc0

cal = CalExperiment(
    nsamples        = 32768,
    nblocks         = 64,
    sample_rate     = 2.56e6,
    center_freq     = 1420.4e6,
    gain            = 30.0,
    outdir          = 'data/cal/',
    prefix          = 'cal',
    siggen_freq_mhz = 1420.405751768,
    siggen_amp_dbm  = -80.0,
)

# CalExperiment.run uses a finally block to ensure synth.rf_off()
# is always called, even if capture raises an exception.
path = cal.run(sdr, synth=synth)

synth.close()  # always close when done
```

---

## 5. Coordinate Conversion

```python
from ugradiolab import compute_pointing, galactic_to_altaz
import ugradio.timing as timing

# Live pointing (uses current time + astropy)
alt, az, ra, dec, jd = compute_pointing(gal_l=120.0, gal_b=0.0)
print(f'Point to alt={alt:.2f}°, az={az:.2f}°')

# Offline conversion (uses internal Hipparcos matrix)
import ugradio.nch as nch
lst_rad = timing.lst(jd, nch.lon)
alt2, az2 = galactic_to_altaz(120.0, 0.0, lst_rad=lst_rad, lat_deg=nch.lat)
```

> **Warning**: `lst_rad` is in **radians**. All other angle arguments in the six
> scalar functions (`alt_deg`, `az_deg`, `l_deg`, `b_deg`, `ra_deg`, `dec_deg`,
> `lat_deg`) are in degrees.

---

## 6. Save and Reload a Spectrum

```python
spec.save('data/spec/galactic_plane_spec.npz')

spec2 = Spectrum.load('data/spec/galactic_plane_spec.npz')
assert (spec2.psd == spec.psd).all()
```
