# API: Drivers

Source: `ugradiolab/drivers/signal_generator.py`

---

## SignalGenerator

Direct USBTMC interface to an Agilent/Keysight N9310A signal generator.

### Hardware Notes

- **Device**: Agilent/Keysight N9310A RF signal generator
- **Interface**: USB Test and Measurement Class (USBTMC), via `/dev/usbtmc0`
- **Permissions**: your user must have read/write access to `/dev/usbtmc0`. On Linux, add yourself to the `usbtmc` group or adjust udev rules.
- **Validation on open**: the constructor sends `*IDN?` and asserts `'N9310A'` is in the response. Connection to a different instrument will raise `AssertionError`.

### Constructor

```python
SignalGenerator(device='/dev/usbtmc0')
```

Opens the USBTMC device and verifies instrument identity. Raises `AssertionError` if the instrument is not an N9310A.

### Method Table

| Method | SCPI command | Returns | Description |
|---|---|---|---|
| `set_freq_mhz(freq_mhz)` | `FREQ:CW {freq} MHz` | `None` | Set CW frequency in MHz |
| `get_freq()` | `FREQ:CW?` | `float` (Hz) | Query current CW frequency |
| `set_ampl_dbm(amp_dbm)` | `AMPL:CW {amp} dBm` | `None` | Set CW amplitude in dBm |
| `get_ampl()` | `AMPL:CW?` | `float` (dBm) | Query current CW amplitude |
| `rf_on()` | `RFO:STAT ON` | `None` | Enable RF output |
| `rf_off()` | `RFO:STAT OFF` | `None` | Disable RF output |
| `rf_state()` | `RFO:STAT?` | `bool` | Query RF output state (`True` = on) |
| `close()` | — | `None` | Turn RF off and close USBTMC handle |

### Inter-Command Delay

Every write operation (`_write`) sleeps for **250 ms** after flushing the command. This is required for reliable communication with the N9310A and cannot be removed without risking partial responses or command loss.

### Lifecycle

Always call `close()` when finished. Safe to call multiple times (`close()` is idempotent — it checks whether the device handle is already `None` before attempting shutdown).

Recommended pattern:

```python
synth = SignalGenerator()
try:
    synth.set_freq_mhz(1420.4)
    synth.set_ampl_dbm(-80.0)
    synth.rf_on()
    # ... capture ...
finally:
    synth.close()
```

`close()` calls `rf_off()` internally (with exceptions silently swallowed), then closes the file handle. This matches the behaviour of `CalExperiment.run`, which uses a `finally` block to call `rf_off()` directly.
