"""Producer-consumer streaming capture pipeline.

Saves every single dump to disk individually.  Three independent threads
handle pointing, reading, and writing:

    PointingThread  -- manages target selection and telescope repointing
    ReaderThread    -- calls a read_fn callable; enqueues each dump
    WriterPool      -- N workers dequeue and save .npz files

StreamingCapture wires them together and provides a single ``run()`` entry
point that blocks until Ctrl-C.

Hardware-agnostic: works with any telescope that exposes
``point(alt, az, wait=True)`` and any reader callable produced by the
factories in :mod:`ugradiolab.capture.readers`.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointingState:
    """Immutable snapshot of the current dish pointing."""

    target_name: str
    alt_deg: float
    az_deg: float
    ra_deg: float
    dec_deg: float
    updated_at: float  # unix timestamp


# ---------------------------------------------------------------------------
# Pointing thread
# ---------------------------------------------------------------------------

class PointingThread:
    """Periodically selects a target and repoints the telescope.

    Parameters
    ----------
    telescope : object
        Pointing controller -- any object with
        ``point(alt, az, wait=True)``.
    target_selector : callable
        Returns ``(name, alt, az, ra, dec)`` for the highest-priority
        visible target, or ``None`` when nothing is up.
    repoint_interval_sec : float
        Maximum time between repoints for the same target.
    """

    def __init__(
        self,
        telescope,
        target_selector: Callable[[], tuple[str, float, float, float, float] | None],
        repoint_interval_sec: float = 30.0,
    ):
        self._telescope = telescope
        self._target_selector = target_selector
        self._repoint_interval_sec = repoint_interval_sec

        self._state: PointingState | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name='pointing', daemon=False)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def get_state(self) -> PointingState | None:
        with self._lock:
            return self._state

    def _run(self) -> None:
        last_repoint_time = 0.0
        current_target: str | None = None

        while not self._stop_event.is_set():
            result = self._target_selector()
            if result is None:
                with self._lock:
                    self._state = None
                current_target = None
                self._stop_event.wait(timeout=1.0)
                continue

            name, alt, az, ra, dec = result
            now = time.time()
            need_repoint = (
                name != current_target
                or (now - last_repoint_time) >= self._repoint_interval_sec
            )

            if need_repoint:
                try:
                    self._telescope.point(alt, az, wait=True)
                except (AssertionError, TimeoutError, OSError) as exc:
                    print(f'  [pointing] slew failed: {exc}')
                    self._stop_event.wait(timeout=5.0)
                    continue
                last_repoint_time = time.time()
                current_target = name

            new_state = PointingState(
                target_name=name,
                alt_deg=alt,
                az_deg=az,
                ra_deg=ra,
                dec_deg=dec,
                updated_at=time.time(),
            )
            with self._lock:
                self._state = new_state

            self._stop_event.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Reader thread (producer)
# ---------------------------------------------------------------------------

class ReaderThread:
    """Calls a reader function and pushes every dump onto a queue.

    The reader function is a callable ``read_fn(prev_cnt) -> dict`` produced
    by one of the factory functions in :mod:`ugradiolab.capture.readers`.

    Parameters
    ----------
    read_fn : callable
        ``read_fn(prev_cnt) -> dict``.  Must return a dict containing at
        least a ``'time'`` key.  May contain ``'acc_cnt'`` (used as the
        state token for the next call; defaults to ``None`` if absent).
    pointing_thread : PointingThread
        Provides the current pointing state to tag each dump.
    dump_queue : queue.Queue
        Output queue for raw dumps.
    max_consecutive_errors : int
        Number of consecutive ``AssertionError`` failures before giving up.
    """

    def __init__(
        self,
        read_fn: Callable,
        pointing_thread: PointingThread,
        dump_queue: queue.Queue,
        max_consecutive_errors: int = 3,
    ):
        self._read_fn = read_fn
        self._pointing = pointing_thread
        self._queue = dump_queue
        self._max_consecutive_errors = max_consecutive_errors

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name='reader', daemon=False)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        prev_cnt = None
        seq = 0
        consecutive_errors = 0

        while not self._stop_event.is_set():
            # Wait for a valid pointing before capturing.
            state = self._pointing.get_state()
            if state is None:
                self._stop_event.wait(timeout=0.5)
                prev_cnt = None  # reset state token (SNAP may miss acc_cnt ticks)
                continue

            try:
                d = self._read_fn(prev_cnt)
                consecutive_errors = 0
            except AssertionError:
                consecutive_errors += 1
                if consecutive_errors >= self._max_consecutive_errors:
                    print(
                        f'  [reader] {consecutive_errors} '
                        'consecutive failures, stopping reader.'
                    )
                    break
                prev_cnt = None
                continue

            dump = {
                **d,
                'target_name': state.target_name,
                'alt_deg':     state.alt_deg,
                'az_deg':      state.az_deg,
                'ra_deg':      state.ra_deg,
                'dec_deg':     state.dec_deg,
                'seq':         seq,
            }
            self._queue.put(dump)  # blocks if queue full (backpressure)
            prev_cnt = d.get('acc_cnt')
            seq += 1


# ---------------------------------------------------------------------------
# Writer pool (consumers)
# ---------------------------------------------------------------------------

class WriterPool:
    """N worker threads that pull dumps from a queue and save them as .npz.

    Parameters
    ----------
    dump_queue : queue.Queue
        Shared input queue.
    outdir : str
        Root output directory.  Files are saved under ``outdir/<target>/``.
    n_workers : int
        Number of writer threads.
    on_save : callable, optional
        ``on_save(path, dump)`` called after each successful save.
    """

    def __init__(
        self,
        dump_queue: queue.Queue,
        outdir: str,
        n_workers: int = 2,
        on_save: Callable[[str, dict], None] | None = None,
    ):
        self._queue = dump_queue
        self._outdir = outdir
        self._n_workers = n_workers
        self._on_save = on_save

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for i in range(self._n_workers):
            t = threading.Thread(target=self._worker, name=f'writer-{i}', daemon=False)
            t.start()
            self._threads.append(t)

    def stop_and_drain(self) -> None:
        """Signal workers to stop, then join them after they drain the queue."""
        self._stop_event.set()
        for t in self._threads:
            t.join()
        self._threads.clear()

    def _worker(self) -> None:
        while True:
            try:
                dump = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            try:
                name = dump['target_name']
                # Calibration dumps go into cal_* cell, obs dumps into obs_* cell
                if dump.get('noise_on') and name.startswith('obs_'):
                    name = 'cal_' + name[4:]
                    dump = dict(dump)
                    dump['target_name'] = name
                ts = time.strftime('%Y%m%d_%H%M%S', time.gmtime(dump['time']))
                filename = f'{name}_{ts}.npz'
                dest = os.path.join(self._outdir, name, filename)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                np.savez(dest, **dump)
                if self._on_save is not None:
                    self._on_save(dest, dump)
            except Exception as exc:
                print(f'  [writer] save failed: {exc}')
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class StreamingCapture:
    """Wire pointing, reading, and writing threads into a streaming pipeline.

    Parameters
    ----------
    telescope : object
        Pointing controller --any object with
        ``point(alt, az, wait=True)``.  Works with both
        ``ugradio.interf.Interferometer`` and
        ``ugradio.leusch.LeuschTelescope``.
    read_fn : callable
        ``read_fn(prev_cnt) -> dict`` --produced by a factory in
        :mod:`ugradiolab.capture.readers`.
    target_selector : callable
        Returns ``(name, alt, az, ra, dec)`` or ``None``.
    outdir : str
        Root output directory.
    n_writers : int
        Number of writer threads.
    queue_maxsize : int
        Bounded queue size (backpressure when full).
    repoint_interval_sec : float
        How often the pointing thread repoints the same target.
    on_save : callable, optional
        ``on_save(path, dump)`` called after each dump is written.
    """

    def __init__(
        self,
        telescope,
        read_fn: Callable,
        target_selector: Callable,
        outdir: str = 'data/',
        n_writers: int = 2,
        queue_maxsize: int = 200,
        repoint_interval_sec: float = 30.0,
        on_save: Callable[[str, dict], None] | None = None,
    ):
        self._queue = queue.Queue(maxsize=queue_maxsize)
        self._pointing = PointingThread(
            telescope, target_selector, repoint_interval_sec,
        )
        self._reader = ReaderThread(read_fn, self._pointing, self._queue)
        self._writer = WriterPool(self._queue, outdir, n_writers, on_save)

    def run(self, done_event: 'threading.Event | None' = None) -> None:
        """Start all threads and block until KeyboardInterrupt or done_event.

        Parameters
        ----------
        done_event : threading.Event, optional
            If provided, exit automatically when this event is set
            (e.g. by the target_selector when the scan grid is complete).
            If None (default), run until KeyboardInterrupt.
        """
        self._pointing.start()

        # Wait for the first valid pointing before starting the reader.
        print('Waiting for target acquisition ...')
        while self._pointing.get_state() is None:
            time.sleep(0.5)
        state = self._pointing.get_state()
        print(f'Acquired target: {state.target_name}  '
              f'(alt={state.alt_deg:.1f} deg, az={state.az_deg:.1f} deg)')

        self._reader.start()
        self._writer.start()
        print('Streaming.  Press Ctrl-C to stop.\n')

        try:
            while True:
                time.sleep(1.0)
                if done_event is not None and done_event.is_set():
                    print('\nScan complete (done_event set).')
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Orderly shutdown: reader -> drain writers -> pointing."""
        print('\nShutting down ...')
        self._reader.stop()
        print(f'  Reader stopped.  {self._queue.qsize()} dumps queued.')
        self._writer.stop_and_drain()
        print('  Writers drained and stopped.')
        self._pointing.stop()
        print('  Pointing stopped.  Shutdown complete.')
