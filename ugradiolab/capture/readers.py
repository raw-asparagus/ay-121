"""Reader factory functions for the streaming capture pipeline.

Each factory returns a callable ``read_fn(prev_cnt) -> dict`` suitable for
:class:`~ugradiolab.capture.streaming.ReaderThread`.

* :func:`make_snap_reader` -- wraps a SNAP FPGA correlator
* :func:`make_sdr_reader` -- dual-polarisation SDR with on-board FFT,
  correlation, and frequency switching
* :func:`make_calibrated_sdr_reader` -- same, with a noise-diode
  calibration phase at the start
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Callable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# SNAP reader
# ---------------------------------------------------------------------------

def make_snap_reader(snap) -> Callable[[int | None], dict]:
    """Wrap a SNAP correlator into a streaming reader callable.

    Parameters
    ----------
    snap : UGRadioSnap
        Initialised SNAP correlator in ``corr`` mode.

    Returns
    -------
    callable
        ``read_fn(prev_cnt) -> dict`` with keys
        ``corr01``, ``time``, ``acc_cnt``.
    """

    def read(prev_cnt: int | None) -> dict:
        return snap.read_data(prev_cnt=prev_cnt)

    return read


# ---------------------------------------------------------------------------
# SDR helpers
# ---------------------------------------------------------------------------

_last_lo_mhz = [None]  # mutable cache for LO frequency skipping


def _sdr_capture(sdrs, nsamples, nblocks, lo_mhz):
    """Set LO and capture raw IQ data from both SDRs.

    Skips set_center_freq if the LO hasn't changed since the last call,
    avoiding the ~1.5s PLL settle time on the R820T2 tuner.

    Returns (raw_data_dict, capture_time, lo_mhz).
    """
    from ugradio.sdr import capture_data

    if lo_mhz != _last_lo_mhz[0]:
        for sdr in sdrs:
            sdr.set_center_freq(lo_mhz * 1e6)
        _last_lo_mhz[0] = lo_mhz

    data = capture_data(sdrs, nsamples=nsamples, nblocks=nblocks)
    t = time.time()

    return data, t, lo_mhz


def _sdr_correlate(data, nsamples, nblocks, nfft, t, lo_mhz):
    """FFT and correlate raw SDR data. Returns dump dict."""
    dev_ids = sorted(data.keys())

    raw_0 = data[dev_ids[0]]  # (nblocks, nsamples, 2)
    raw_1 = data[dev_ids[1]]

    n_valid = nblocks - 1
    n_chunks = nsamples // nfft

    # Accumulate correlations block-by-block to avoid allocating the full
    # (n_valid, n_chunks, nfft) FFT array, which can exceed Pi memory.
    corr00 = np.zeros(nfft, dtype=np.float64)
    corr11 = np.zeros(nfft, dtype=np.float64)
    corr01 = np.zeros(nfft, dtype=np.complex128)

    for b in range(1, nblocks):  # skip block 0
        block_0 = raw_0[b]  # (nsamples, 2) int8
        block_1 = raw_1[b]

        iq_0 = block_0[:, 0].astype(np.float32) + 1j * block_0[:, 1].astype(np.float32)
        iq_1 = block_1[:, 0].astype(np.float32) + 1j * block_1[:, 1].astype(np.float32)

        V0 = np.fft.fft(iq_0.reshape(n_chunks, nfft), axis=-1)
        V1 = np.fft.fft(iq_1.reshape(n_chunks, nfft), axis=-1)

        corr00 += np.sum((V0 * np.conj(V0)).real, axis=0)
        corr11 += np.sum((V1 * np.conj(V1)).real, axis=0)
        corr01 += np.sum(V0 * np.conj(V1), axis=0)

    total_windows = n_valid * n_chunks
    corr00 /= total_windows
    corr11 /= total_windows
    corr01 /= total_windows

    return {
        'corr00': corr00,
        'corr01': corr01,
        'corr11': corr11,
        'time': t,
        'lo_freq_mhz': lo_mhz,
    }


def _sdr_capture_and_correlate(sdrs, nsamples, nblocks, nfft, lo_mhz):
    """Set LO, capture both polarisations, FFT, correlate, return dump dict.

    This is the non-pipelined version, kept for compatibility.
    """
    data, t, lo = _sdr_capture(sdrs, nsamples, nblocks, lo_mhz)
    return _sdr_correlate(data, nsamples, nblocks, nfft, t, lo)


# ---------------------------------------------------------------------------
# Pipelined SDR capture + correlate
# ---------------------------------------------------------------------------

class _PipelinedSDR:
    """Overlaps FFT/correlation with the next SDR capture.

    On each call to ``next_dump(lo_mhz)``:
      1. If there's a previous capture pending correlation, start correlating
         it in a background thread.
      2. Start the next SDR capture (this blocks for ~13s on USB transfer).
      3. Wait for the background correlation to finish.
      4. Return the correlated dump from the *previous* capture.

    The first call has no previous data, so it just captures and returns None.
    The caller must call ``flush()`` at the end to get the last dump.
    """

    def __init__(self, sdrs, nsamples, nblocks, nfft):
        self._sdrs = sdrs
        self._nsamples = nsamples
        self._nblocks = nblocks
        self._nfft = nfft

        self._pending_data = None   # raw data awaiting correlation
        self._pending_t = None
        self._pending_lo = None
        self._corr_thread = None
        self._corr_result = None

    def next_dump(self, lo_mhz):
        """Capture at lo_mhz, return the *previous* dump (or None if first call)."""
        # Start correlating previous capture in background
        if self._pending_data is not None:
            self._start_correlate()

        # Capture new data (blocks for USB transfer duration)
        data, t, lo = _sdr_capture(self._sdrs, self._nsamples, self._nblocks, lo_mhz)

        # Wait for background correlation to finish
        prev_dump = None
        if self._corr_thread is not None:
            self._corr_thread.join()
            prev_dump = self._corr_result
            self._corr_thread = None
            self._corr_result = None

        # Store new capture for next round's correlation
        self._pending_data = data
        self._pending_t = t
        self._pending_lo = lo

        return prev_dump

    def flush(self):
        """Correlate and return the last pending capture."""
        if self._pending_data is None:
            return None
        return _sdr_correlate(
            self._pending_data, self._nsamples, self._nblocks,
            self._nfft, self._pending_t, self._pending_lo,
        )

    def _start_correlate(self):
        """Start correlation of pending data in a background thread."""
        data = self._pending_data
        t = self._pending_t
        lo = self._pending_lo
        nsamples = self._nsamples
        nblocks = self._nblocks
        nfft = self._nfft

        def _run():
            self._corr_result = _sdr_correlate(data, nsamples, nblocks, nfft, t, lo)

        self._corr_thread = threading.Thread(target=_run, daemon=True)
        self._corr_thread.start()


# ---------------------------------------------------------------------------
# SDR reader
# ---------------------------------------------------------------------------

def make_sdr_reader(
    sdrs: list,
    nsamples: int = 32768,
    nblocks: int = 65,
    nfft: int = 1024,
    lo_freqs_mhz: Sequence[float] = (1420.0, 1421.0),
) -> Callable[[int | None], dict]:
    """Create a streaming reader for dual-polarisation SDR capture.

    Each call to the returned function:

    1. Sets both SDRs to the next LO frequency in the cycle.
    2. Captures *nblocks* blocks from both SDRs simultaneously.
    3. Discards block 0 (stale USB buffer).
    4. Reshapes each block into chunks of *nfft* samples and FFTs.
    5. Computes ``corr00``, ``corr01``, ``corr11`` averaged over all windows.

    Parameters
    ----------
    sdrs : list of SDR
        Two initialised SDR objects (polarisation 0 and 1).
    nsamples : int
        Samples per capture block.
    nblocks : int
        Total blocks to capture (block 0 is discarded).
    nfft : int
        FFT length (number of spectral channels).
    lo_freqs_mhz : sequence of float
        LO frequencies to cycle through (e.g. ``(1420.0, 1421.0)``).

    Returns
    -------
    callable
        ``read_fn(prev_cnt) -> dict`` with keys
        ``corr00``, ``corr01``, ``corr11``, ``time``, ``lo_freq_mhz``.
    """
    # Block LO switching: all dumps at one LO before switching.
    # Default block size = 4 (matches typical dumps_per_band).
    _block = [lo for lo in lo_freqs_mhz for _ in range(4)]
    freq_cycle = itertools.cycle(_block)
    pipeline = _PipelinedSDR(sdrs, nsamples, nblocks, nfft)

    def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
        lo = next(freq_cycle)
        dump = pipeline.next_dump(lo)

        if dump is not None:
            return dump

        # First call returned None — capture again to get the first dump
        lo = next(freq_cycle)
        dump = pipeline.next_dump(lo)
        if dump is not None:
            return dump

        # Shouldn't reach here, but fall back to non-pipelined
        return _sdr_capture_and_correlate(sdrs, nsamples, nblocks, nfft, lo)

    return read


# ---------------------------------------------------------------------------
# SDR reader with noise-diode calibration phase
# ---------------------------------------------------------------------------

def make_calibrated_sdr_reader(
    sdrs: list,
    noise,
    nsamples: int = 32768,
    nblocks: int = 65,
    nfft: int = 1024,
    lo_freqs_mhz: Sequence[float] = (1420.0, 1421.0),
    cal_dumps_per_lo: int = 32,
) -> Callable[[int | None], dict]:
    """SDR reader that runs a noise-diode calibration phase, then science.

    The first ``cal_dumps_per_lo * len(lo_freqs_mhz)`` calls capture with the
    noise diode ON (each LO frequency for ``cal_dumps_per_lo`` dumps in
    sequence).  After the calibration phase the diode is turned OFF and
    subsequent calls use block LO switching — all dumps at one LO before
    switching to the next, minimising tuner PLL settle overhead.

    Uses pipelined capture: FFT/correlation of the previous dump runs in a
    background thread while the next USB capture is in progress.  Combined
    with LO caching (skip set_center_freq when frequency unchanged), this
    achieves ~87% duty cycle vs ~77% with alternating LO switching.

    Every returned dict includes a ``'noise_on'`` boolean flag.

    Parameters
    ----------
    sdrs : list of SDR
        Two initialised SDR objects (polarisation 0 and 1).
    noise : object
        Noise diode controller with ``.on()`` / ``.off()`` methods.
    nsamples, nblocks, nfft, lo_freqs_mhz
        Forwarded to the SDR capture core.
    cal_dumps_per_lo : int
        Number of calibration dumps per LO frequency.
    """
    lo_list = list(lo_freqs_mhz)
    total_cal_dumps = cal_dumps_per_lo * len(lo_list)

    # Cal schedule: [lo0]*N + [lo1]*N + ...
    cal_lo_schedule = [lo for lo in lo_list for _ in range(cal_dumps_per_lo)]

    # Science schedule: block LO switching — all dumps at one LO before
    # switching to the next.  This minimises tuner PLL settle overhead
    # (set_center_freq is skipped when LO is unchanged).
    # Block size matches cal_dumps_per_lo so each LO gets equal coverage
    # per cell.  Example with 2 LOs, 4 dumps/band:
    #   [1420]*4 + [1421]*4 + [1420]*4 + [1421]*4 + ...
    _block_size = max(1, cal_dumps_per_lo)
    _science_block = [lo for lo in lo_list for _ in range(_block_size)]
    science_cycle = itertools.cycle(_science_block)

    pipeline = _PipelinedSDR(sdrs, nsamples, nblocks, nfft)

    call_count = 0
    cal_started = False
    # Buffer for dumps produced by the pipeline (previous dump comes back
    # when we submit the next capture).
    noise_on_flags = []  # tracks the noise_on state for each submitted capture
    submit_count = 0

    def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
        nonlocal call_count, cal_started, submit_count

        # Determine LO and noise state for this submission
        if submit_count < total_cal_dumps:
            if not cal_started:
                noise.on()
                print('  [reader] Noise diode ON - calibration phase')
                cal_started = True
            lo = cal_lo_schedule[submit_count]
            is_noise_on = True
        else:
            lo = next(science_cycle)
            is_noise_on = False

        # Submit capture, get back previous dump
        dump = pipeline.next_dump(lo)
        noise_on_flags.append(is_noise_on)
        submit_count += 1

        if dump is None:
            # First call — no previous dump yet. Submit one more.
            if submit_count < total_cal_dumps:
                lo2 = cal_lo_schedule[submit_count]
                is_noise_on2 = True
            else:
                lo2 = next(science_cycle)
                is_noise_on2 = False
            dump = pipeline.next_dump(lo2)
            noise_on_flags.append(is_noise_on2)
            submit_count += 1

        if dump is not None:
            # Tag with the noise_on state of the dump being returned
            # (which is from call_count, not submit_count)
            dump['noise_on'] = noise_on_flags[call_count]
            call_count += 1

            # Check if we just finished the cal phase
            if call_count == total_cal_dumps and noise is not None:
                noise.off()
                print('  [reader] Noise diode OFF - entering science mode')

        return dump

    return read
