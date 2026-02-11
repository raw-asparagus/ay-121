"""Hardware driver utilities for SDR and signal generator."""

from .siggen import SignalGenerator, connect, set_signal, freq_sweep

try:
    from .sdr_utils import capture_and_fft, power_spectrum, collect_time_series
except (ImportError, AttributeError):
    pass