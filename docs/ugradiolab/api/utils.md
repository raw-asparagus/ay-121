# API: Utils

Source: `ugradiolab/io/clock.py`

---

## `get_unix_time`

```python
get_unix_time(timeout: float = 2.0, local: bool = False) -> float
```

Returns the current Unix time in seconds (float).

### Behaviour

1. Attempts to query `pool.ntp.org` (NTP version 3) with the given timeout.
2. If the NTP request succeeds, returns `response.tx_time` (the NTP transmit timestamp).
3. If the NTP request fails for any reason (`NTPException` or `OSError` — e.g., no network, DNS failure, timeout), prints `"Unable to connect to NTP! Using system time."` and falls back to `ugradio.timing.unix_time()`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timeout` | `float` | `2.0` | NTP request timeout in seconds |
| `local` | `bool` | `False` | If `True`, bypass NTP and use the local system clock immediately |

### Why Accuracy Matters

`get_unix_time` is used in `Record.from_sdr` and `compute_pointing` to timestamp captures. The timestamp is converted to a Julian Date (`ugradio.timing.julian_date`), which is then used to compute Local Sidereal Time via `ugradio.timing.lst`. A 1-second timing error translates to roughly 15 arcseconds of LST error (360°/86400 s), which is negligible for most AY121 applications but can matter for precise pointing or spectral calibration. NTP accuracy is typically ≤ 10 ms over a local network.

### Usage

```python
from ugradiolab import get_unix_time

t = get_unix_time()            # NTP first, system clock fallback
t_fast = get_unix_time(0.5)    # 500 ms timeout
t_local = get_unix_time(local=True)  # local system clock only
```
