"""Reader factory functions for the streaming capture pipeline.

Each factory returns a callable ``read_fn(prev_cnt) -> dict`` suitable for
:class:`~ugradiolab.capture.streaming.ReaderThread`.

* :func:`make_snap_reader` -- wraps a SNAP FPGA correlator
* :func:`make_calibrated_sdr_reader` -- dual-polarisation SDR with
  on-board FFT, correlation, frequency switching, and noise-diode
  calibration
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Callable, Sequence

import numpy as np
from scipy.fft import fft as _fft


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


def _sdr_correlate(data, nsamples, nblocks, nfft, t, lo_mhz,
                   _batch=64):
    """FFT and correlate raw SDR data. Returns dump dict.

    Processes blocks in batches to reduce Python loop iterations (and
    GIL re-acquisitions), enabling real overlap with USB capture when
    called from a background thread.  Default batch size of 64 blocks
    uses ~16 MB per polarisation (complex64), safe on a 4 GB Pi.
    """
    dev_ids = sorted(data.keys())

    raw_0 = data[dev_ids[0]]  # (nblocks, nsamples, 2)
    raw_1 = data[dev_ids[1]]

    n_valid = nblocks - 1
    n_chunks = nsamples // nfft

    corr00 = np.zeros(nfft, dtype=np.float64)
    corr11 = np.zeros(nfft, dtype=np.float64)
    corr01 = np.zeros(nfft, dtype=np.complex128)

    for b_start in range(1, nblocks, _batch):
        b_end = min(b_start + _batch, nblocks)

        chunk_0 = raw_0[b_start:b_end]          # (batch, nsamples, 2)
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
        result = _sdr_correlate(
            self._pending_data, self._nsamples, self._nblocks,
            self._nfft, self._pending_t, self._pending_lo,
        )
        self._pending_data = None
        return result

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
# SDR reader with noise-diode calibration phase
# ---------------------------------------------------------------------------

def _build_cell_schedule(lo_list, cal_dumps_per_lo, obs_dumps_per_lo):
    """Build an interleaved cal+obs LO schedule for one cell.

    Cal phase: one dump per LO in list order (``LO1, LO2, ...``),
    repeated ``cal_dumps_per_lo`` times per LO.

    Obs phase: ABBA interleaving starting from the last cal LO.
    The unit ``reversed(lo_list) + lo_list`` is cycled for
    ``obs_dumps_per_lo * len(lo_list)`` dumps, ensuring maximum
    temporal mixing::

        cal_dumps_per_lo=1, obs_dumps_per_lo=3, 2 LOs (A, B):
          CAL-A, CAL-B, OBS-B, OBS-A, OBS-A, OBS-B, OBS-B, OBS-A

    Returns list of ``(lo_mhz, noise_on)`` tuples.
    """
    schedule = []
    for lo in lo_list:
        schedule.extend([(lo, True)] * cal_dumps_per_lo)

    abba = list(reversed(lo_list)) + list(lo_list)
    total_obs = obs_dumps_per_lo * len(lo_list)
    for i in range(total_obs):
        schedule.append((abba[i % len(abba)], False))

    return schedule


def make_calibrated_sdr_reader(
    sdrs: list,
    noise,
    nsamples: int = 32768,
    nblocks: int = 65,
    nfft: int = 1024,
    lo_freqs_mhz: Sequence[float] = (1420.0, 1421.0),
    cal_dumps_per_lo: int = 32,
    cell_event: 'threading.Event | None' = None,
    obs_dumps_per_lo: int | None = None,
    cell_done_event: 'threading.Event | None' = None,
    stop_event: 'threading.Event | None' = None,
    pointing_wake_event: 'threading.Event | None' = None,
) -> Callable[[int | None], dict]:
    """SDR reader with noise-diode calibration and block LO switching.

    Supports two modes:

    **Per-session mode** (default, ``cell_event=None``):
    The first ``cal_dumps_per_lo * len(lo_freqs_mhz)`` calls capture with
    the noise diode ON, then all subsequent calls run with the diode OFF
    using block LO switching.

    **Per-cell mode** (``cell_event`` provided):
    Each cell gets its own calibration phase followed by science dumps.
    When the schedule is exhausted the reader flushes the pipeline (no
    new USB capture), sets ``cell_done_event``, and blocks until
    ``cell_event`` signals a new cell.  This keeps the reader idle
    during slews and eliminates extra dumps at cell boundaries.

    The per-cell schedule uses ABBA LO interleaving for maximum
    temporal mixing of the two frequency bands::

        CAL: LO1, LO2              (noise ON)
        OBS: LO2, LO1, LO1, LO2, LO2, LO1  (noise OFF, ABBA cycle)

    Obs starts at the last cal LO to avoid a PLL settle at the
    cal-to-obs boundary.

    Uses pipelined capture: FFT/correlation of the previous dump runs
    in a background thread while the next USB capture is in progress.

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
    cell_event : threading.Event, optional
        When provided, enables per-cell calibration.  The schedule resets
        to the cal phase each time this event is set.
    obs_dumps_per_lo : int, optional
        Number of science dumps per LO per cell.  Required when
        ``cell_event`` is not None.
    cell_done_event : threading.Event, optional
        Set by the reader when the per-cell schedule is exhausted.
        The target selector should react to this by advancing to the
        next cell and setting ``cell_event``.
    stop_event : threading.Event, optional
        Checked while the reader is blocked between cells so it can
        exit cleanly on shutdown.
    """
    lo_list = list(lo_freqs_mhz)
    pipeline = _PipelinedSDR(sdrs, nsamples, nblocks, nfft)

    noise_on_flags = []  # tracks noise_on state for each submitted capture
    call_count = 0
    submit_count = 0
    current_noise_state = None
    # Settle time after a diode state change, before the next integration
    # starts. Lets the noise diode and any thermal tail in the LNA chain
    # reach steady state so the first cal/obs dump after a transition is
    # not biased by the transient.
    NOISE_SETTLE_S = 0.5

    def _set_noise(on):
        nonlocal current_noise_state
        if on != current_noise_state:
            if on:
                noise.on()
            else:
                noise.off()
            current_noise_state = on
            time.sleep(NOISE_SETTLE_S)

    # ----- Per-cell mode -----
    if cell_event is not None:
        if obs_dumps_per_lo is None:
            raise ValueError('obs_dumps_per_lo is required with cell_event')

        cell_schedule = _build_cell_schedule(
            lo_list, cal_dumps_per_lo, obs_dumps_per_lo,
        )

        schedule_len = len(cell_schedule)
        schedule_idx = 0
        flushed = False  # True after pipeline.flush() at end of cell

        def _wait_for_new_cell():
            """Block until cell_event or stop_event fires."""
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                if cell_event.wait(timeout=0.5):
                    return True

        def _reset_schedule():
            """Flush stale pipeline data and reset schedule for a new cell.

            Does NOT prime the pipeline — returns {} so ReaderThread
            re-checks pointing state before the next capture.  This
            ensures the first capture of the new cell happens only
            after the dish has settled on the new target.
            """
            nonlocal schedule_idx, flushed, call_count
            cell_event.clear()
            stale = pipeline.flush()
            if stale is not None:
                call_count += 1  # skip the discarded dump's noise flag
            flushed = False
            schedule_idx = 0

        def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
            nonlocal schedule_idx, call_count, flushed

            # --- Cell complete: flush pipeline then block ---
            if schedule_idx >= schedule_len:
                if not flushed:
                    # Correlate last pending capture (no new USB transfer).
                    # cell_done is NOT set here — that would let PointingThread
                    # clear pointing state during flush and the post-state
                    # check in ReaderThread would discard this dump.
                    flushed = True
                    last = pipeline.flush()
                    if last is not None:
                        last['noise_on'] = noise_on_flags[call_count]
                        call_count += 1
                        return last

                # Last dump has been returned and queued.  Now safe to
                # signal cell-done so the selector + PointingThread can
                # begin the slew.
                if cell_done_event is not None and not cell_done_event.is_set():
                    cell_done_event.set()
                    if pointing_wake_event is not None:
                        pointing_wake_event.set()

                # Pipeline already flushed — block until next cell
                if not _wait_for_new_cell():
                    return {}  # stop_event fired
                _reset_schedule()
                # Return empty so ReaderThread re-checks pointing state
                # before the first capture of the new cell.
                return {}

            # --- Cell transition signalled externally (e.g. skip) ---
            if cell_event.is_set():
                _reset_schedule()
                return {}

            # --- Normal schedule entry ---
            lo, is_cal = cell_schedule[schedule_idx]
            _set_noise(is_cal)

            dump = pipeline.next_dump(lo)
            noise_on_flags.append(is_cal)
            schedule_idx += 1

            if dump is None:
                # First call or after reset — pipeline priming, capture once more
                lo2, is_cal2 = cell_schedule[schedule_idx]
                _set_noise(is_cal2)
                dump = pipeline.next_dump(lo2)
                noise_on_flags.append(is_cal2)
                schedule_idx += 1

            if dump is not None:
                dump['noise_on'] = noise_on_flags[call_count]
                call_count += 1

            return dump

        return read

    # ----- Per-session mode (original behavior) -----
    total_cal_dumps = cal_dumps_per_lo * len(lo_list)

    # Cal schedule: [lo0]*N + [lo1]*N + ...
    cal_lo_schedule = [lo for lo in lo_list for _ in range(cal_dumps_per_lo)]

    # Science schedule: block LO switching -- all dumps at one LO before
    # switching to the next.  This minimises tuner PLL settle overhead
    # (set_center_freq is skipped when LO is unchanged).
    # Block size matches cal_dumps_per_lo so each LO gets equal coverage
    # per cell.  Example with 2 LOs, 4 dumps/band:
    #   [1420]*4 + [1421]*4 + [1420]*4 + [1421]*4 + ...
    _block_size = max(1, cal_dumps_per_lo)
    _science_block = [lo for lo in lo_list for _ in range(_block_size)]
    science_cycle = itertools.cycle(_science_block)

    cal_started = False

    def read(prev_cnt: int | None) -> dict:  # noqa: ARG001
        nonlocal call_count, cal_started, submit_count

        # Determine LO and noise state for this submission.
        # Turn diode off BEFORE the first science submission so the
        # capture doesn't contain residual diode power.
        if submit_count < total_cal_dumps:
            if not cal_started:
                noise.on()
                print('  [reader] Noise diode ON - calibration phase')
                cal_started = True
            lo = cal_lo_schedule[submit_count]
            is_noise_on = True
        else:
            if submit_count == total_cal_dumps and noise is not None:
                noise.off()
                print('  [reader] Noise diode OFF - entering science mode')
            lo = next(science_cycle)
            is_noise_on = False

        # Submit capture, get back previous dump
        dump = pipeline.next_dump(lo)
        noise_on_flags.append(is_noise_on)
        submit_count += 1

        if dump is None:
            # First call -- no previous dump yet. Submit one more.
            if submit_count < total_cal_dumps:
                lo2 = cal_lo_schedule[submit_count]
                is_noise_on2 = True
            else:
                if submit_count == total_cal_dumps and noise is not None:
                    noise.off()
                    print('  [reader] Noise diode OFF - entering science mode')
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

        return dump

    return read
