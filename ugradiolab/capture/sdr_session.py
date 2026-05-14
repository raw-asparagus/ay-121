"""Pipelined dual-polarisation SDR capture and correlation.

A single ``SDRSession`` owns two SDR devices, the noise-diode controller,
and the FFT/correlation core.  ``run_schedule(schedule)`` is a generator
that yields one correlated dump per ``(lo_mhz, noise_on)`` entry.

Pipeline behaviour
------------------
Within a schedule the FFT/correlation of dump N runs in a background
thread while the USB capture of dump N+1 is in flight.  The schedule
drains synchronously at the end (the last correlation completes before
``run_schedule`` returns), so there is no carry-over state between
schedules: the next schedule starts cleanly with no priming capture.

LO and noise-diode state are cached on the session and only re-issued
when they change, which lets a caller pre-arm the first slot's LO and
diode state during a telescope slew so the first capture starts
immediately on settle.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable

import numpy as np
from scipy.fft import fft as _fft


class _Future:
    """Minimal one-shot future used to hand a correlated dump from the
    background correlation thread back to the schedule generator."""

    __slots__ = ('_event', '_value', '_exc')

    def __init__(self) -> None:
        self._event = threading.Event()
        self._value = None
        self._exc: BaseException | None = None

    def set_result(self, value: dict) -> None:
        self._value = value
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exc = exc
        self._event.set()

    def result(self) -> dict:
        self._event.wait()
        if self._exc is not None:
            raise self._exc
        return self._value  # type: ignore[return-value]


def _correlate(data, nsamples, nblocks, nfft, t, lo_mhz, _batch=64):
    """FFT and cross-correlate one capture from two SDRs.

    Block 0 of every capture is discarded (USB pipe flush).  Blocks are
    processed in batches to amortise Python loop / GIL overhead so the
    correlation can actually run in parallel with the next USB transfer.
    """
    dev_ids = sorted(data.keys())
    raw_0 = data[dev_ids[0]]
    raw_1 = data[dev_ids[1]]

    n_valid = nblocks - 1
    n_chunks = nsamples // nfft

    corr00 = np.zeros(nfft, dtype=np.float64)
    corr11 = np.zeros(nfft, dtype=np.float64)
    corr01 = np.zeros(nfft, dtype=np.complex128)

    for b_start in range(1, nblocks, _batch):
        b_end = min(b_start + _batch, nblocks)
        chunk_0 = raw_0[b_start:b_end]
        chunk_1 = raw_1[b_start:b_end]

        iq_0 = chunk_0[..., 0].astype(np.float32) + 1j * chunk_0[..., 1].astype(np.float32)
        iq_1 = chunk_1[..., 0].astype(np.float32) + 1j * chunk_1[..., 1].astype(np.float32)

        V0 = _fft(iq_0.reshape(-1, nfft), axis=-1, workers=-1)
        V1 = _fft(iq_1.reshape(-1, nfft), axis=-1, workers=-1)

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


class SDRSession:
    """Pipelined dual-polarisation SDR with cached LO and noise state.

    Parameters
    ----------
    sdrs : list
        Two initialised SDR objects (polarisation 0 and 1).
    noise : object
        Noise-diode controller exposing ``.on()`` / ``.off()``.
    nsamples, nblocks, nfft : int
        Forwarded to the SDR capture core.  ``nblocks`` includes the
        block-0 buffer flush; ``n_valid = nblocks - 1`` blocks contribute
        to the correlation.
    noise_settle_s : float
        Sleep after a real diode state change before the next capture
        starts.  Skipped if the requested state matches the cached state.
    """

    def __init__(
        self,
        sdrs: list,
        noise,
        nsamples: int = 32768,
        nblocks: int = 1025,
        nfft: int = 1024,
        noise_settle_s: float = 0.5,
    ) -> None:
        self._sdrs = sdrs
        self._noise = noise
        self._nsamples = nsamples
        self._nblocks = nblocks
        self._nfft = nfft
        self._noise_settle_s = noise_settle_s
        self._cached_lo: float | None = None
        self._cached_noise: bool | None = None

    # ------------------------------------------------------------------
    # State setters (idempotent; safe to call from a slew-overlapped path)
    # ------------------------------------------------------------------

    def set_lo(self, lo_mhz: float) -> None:
        """Re-tune both SDRs.  No-op if already tuned to ``lo_mhz``."""
        if lo_mhz != self._cached_lo:
            for sdr in self._sdrs:
                sdr.set_center_freq(lo_mhz * 1e6)
            self._cached_lo = lo_mhz

    def set_noise(self, on: bool) -> None:
        """Toggle the noise diode and sleep ``noise_settle_s`` if state changed."""
        if on != self._cached_noise:
            (self._noise.on if on else self._noise.off)()
            self._cached_noise = on
            time.sleep(self._noise_settle_s)

    def prearm(self, lo_mhz: float, noise_on: bool) -> None:
        """Set LO and diode state ahead of a capture.

        Used by the coordinator to overlap LO PLL settle and diode warm-up
        with the telescope slew, so the first capture of a cell starts
        immediately on settle.
        """
        self.set_lo(lo_mhz)
        self.set_noise(noise_on)

    # ------------------------------------------------------------------
    # Schedule runner
    # ------------------------------------------------------------------

    def run_schedule(self, schedule: Iterable[tuple[float, bool]]):
        """Yield one correlated dump per ``(lo_mhz, noise_on)`` entry.

        The schedule drains synchronously: when ``run_schedule`` returns,
        no capture or correlation is left pending.  The next call starts
        from a clean state.
        """
        schedule = list(schedule)
        if not schedule:
            return

        bg: _Future | None = None
        for lo, noise_on in schedule:
            self.set_noise(noise_on)
            self.set_lo(lo)
            data, t = self._capture()
            if bg is not None:
                yield bg.result()
            bg = self._submit_correlate(data, t, lo, noise_on)

        if bg is not None:
            yield bg.result()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture(self) -> tuple[dict, float]:
        from ugradio.sdr import capture_data

        data = capture_data(self._sdrs, nsamples=self._nsamples, nblocks=self._nblocks)
        return data, time.time()

    def _submit_correlate(self, data, t: float, lo: float, noise_on: bool) -> _Future:
        fut = _Future()
        nsamples = self._nsamples
        nblocks = self._nblocks
        nfft = self._nfft

        def _run() -> None:
            try:
                d = _correlate(data, nsamples, nblocks, nfft, t, lo)
                d['noise_on'] = noise_on
                fut.set_result(d)
            except BaseException as exc:  # propagate to caller via result()
                fut.set_exception(exc)

        threading.Thread(target=_run, name='sdr-correlate', daemon=True).start()
        return fut
