"""Reader factory functions for the streaming capture pipeline.

Each factory returns a callable ``read_fn(prev_cnt) -> dict`` suitable for
:class:`~ugradiolab.capture.streaming.ReaderThread`.

* :func:`make_snap_reader` — wraps a SNAP FPGA correlator
* :func:`make_sdr_reader` — dual-polarisation SDR with on-board FFT,
  correlation, and frequency switching
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
    5. Computes ``corr00 = |V₀|²``, ``corr01 = V₀·conj(V₁)``,
       ``corr11 = |V₁|²`` averaged over all windows.

    Parameters
    ----------
    sdrs : list of SDR
        Two initialised SDR objects (polarisation 0 and 1).
    nsamples : int
        Samples per capture block.
    nblocks : int
        Total blocks to capture (block 0 is discarded → *nblocks − 1* used).
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
    from ugradio.sdr import capture_data

    freq_cycle = itertools.cycle(lo_freqs_mhz)

    def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
        # --- Frequency switching ---
        lo = next(freq_cycle)
        for sdr in sdrs:
            sdr.set_center_freq(lo * 1e6)

        # --- Capture both polarisations simultaneously ---
        data = capture_data(sdrs, nsamples=nsamples, nblocks=nblocks)
        t = time.time()

        dev_ids = sorted(data.keys())

        # --- Discard block 0, convert int8 I/Q to complex float32 ---
        raw_0 = data[dev_ids[0]][1:]  # (nblocks-1, nsamples, 2)
        raw_1 = data[dev_ids[1]][1:]

        iq_0 = raw_0[..., 0].astype(np.float32) + 1j * raw_0[..., 1].astype(np.float32)
        iq_1 = raw_1[..., 0].astype(np.float32) + 1j * raw_1[..., 1].astype(np.float32)

        # --- Reshape into FFT chunks and transform ---
        n_valid = nblocks - 1
        n_chunks = nsamples // nfft
        # (n_valid, n_chunks, nfft)
        V0 = np.fft.fft(iq_0.reshape(n_valid, n_chunks, nfft), axis=-1)
        V1 = np.fft.fft(iq_1.reshape(n_valid, n_chunks, nfft), axis=-1)

        # --- Correlations, averaged across all windows ---
        corr00 = np.mean((V0 * np.conj(V0)).real, axis=(0, 1))  # (nfft,) float
        corr11 = np.mean((V1 * np.conj(V1)).real, axis=(0, 1))  # (nfft,) float
        corr01 = np.mean(V0 * np.conj(V1), axis=(0, 1))         # (nfft,) complex

        return {
            'corr00': corr00,
            'corr01': corr01,
            'corr11': corr11,
            'time': t,
            'lo_freq_mhz': lo,
        }

    return read
