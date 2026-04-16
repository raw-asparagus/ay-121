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

def _sdr_capture_and_correlate(sdrs, nsamples, nblocks, nfft, lo_mhz):
    """Set LO, capture both polarisations, FFT, correlate, return dump dict.

    This is the shared DSP core used by all SDR reader factories.
    """
    from ugradio.sdr import capture_data

    for sdr in sdrs:
        sdr.set_center_freq(lo_mhz * 1e6)

    data = capture_data(sdrs, nsamples=nsamples, nblocks=nblocks)
    t = time.time()

    dev_ids = sorted(data.keys())

    # Discard block 0 (stale USB buffer), convert int8 I/Q to complex float32
    raw_0 = data[dev_ids[0]][1:]  # (nblocks-1, nsamples, 2)
    raw_1 = data[dev_ids[1]][1:]

    iq_0 = raw_0[..., 0].astype(np.float32) + 1j * raw_0[..., 1].astype(np.float32)
    iq_1 = raw_1[..., 0].astype(np.float32) + 1j * raw_1[..., 1].astype(np.float32)

    # Reshape into FFT chunks and transform
    n_valid = nblocks - 1
    n_chunks = nsamples // nfft
    V0 = np.fft.fft(iq_0.reshape(n_valid, n_chunks, nfft), axis=-1)
    V1 = np.fft.fft(iq_1.reshape(n_valid, n_chunks, nfft), axis=-1)

    # Correlations, averaged across all windows
    corr00 = np.mean((V0 * np.conj(V0)).real, axis=(0, 1))  # (nfft,) float
    corr11 = np.mean((V1 * np.conj(V1)).real, axis=(0, 1))  # (nfft,) float
    corr01 = np.mean(V0 * np.conj(V1), axis=(0, 1))         # (nfft,) complex

    return {
        'corr00': corr00,
        'corr01': corr01,
        'corr11': corr11,
        'time': t,
        'lo_freq_mhz': lo_mhz,
    }


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
    freq_cycle = itertools.cycle(lo_freqs_mhz)

    def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
        lo = next(freq_cycle)
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
    subsequent calls alternate LO frequencies for science.

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
    science_cycle = itertools.cycle(lo_list)

    call_count = 0
    cal_started = False

    def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
        nonlocal call_count, cal_started

        if call_count < total_cal_dumps:
            if not cal_started:
                noise.on()
                print('  [reader] Noise diode ON - calibration phase')
                cal_started = True

            lo = cal_lo_schedule[call_count]
            dump = _sdr_capture_and_correlate(sdrs, nsamples, nblocks, nfft, lo)
            dump['noise_on'] = True
            call_count += 1

            if call_count == total_cal_dumps:
                noise.off()
                print('  [reader] Noise diode OFF - entering science mode')
        else:
            lo = next(science_cycle)
            dump = _sdr_capture_and_correlate(sdrs, nsamples, nblocks, nfft, lo)
            dump['noise_on'] = False

        return dump

    return read
