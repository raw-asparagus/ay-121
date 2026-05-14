"""Synchronous survey driver.

A single coordinator thread drives the survey loop:

    for cell in scheduler:
        sdr.prearm(cell.first_lo, cell.first_noise_on)   # overlaps slew
        telescope.point(cell.alt, cell.az, wait=True)
        with tracking:
            for dump in sdr.run_schedule(cell.schedule):
                writer.submit(tagged(dump, cell))

There are no cross-thread state machines, no events between selector and
reader, and no priming captures.  ``WriterPool`` continues to live in a
small thread pool so disk writes don't block correlation/capture.

A ``TrackingThread`` re-issues ``telescope.point(..., wait=False)`` on a
fixed cadence using a per-cell ``recompute_altaz`` callable, so the dish
follows sidereal motion during long cells.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Cell + pointing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pointing:
    """Initial (slew-target) sky coordinates for a cell."""

    target_name: str
    alt_deg: float
    az_deg: float
    ra_deg: float
    dec_deg: float


@dataclass
class Cell:
    """One unit of work for the coordinator.

    ``schedule`` is an ordered list of ``(lo_mhz, noise_on)`` tuples that
    is fed straight to ``SDRSession.run_schedule``.

    ``recompute_altaz`` is an optional callable returning fresh
    ``(alt, az)`` for sidereal tracking.  When ``None`` the tracker is
    not started (e.g. for stare cells with no sky-rate).
    """

    pointing: Pointing
    schedule: list = field(default_factory=list)
    recompute_altaz: Optional[Callable[[], tuple[float, float]]] = None


# ---------------------------------------------------------------------------
# Tracking thread
# ---------------------------------------------------------------------------

class TrackingThread:
    """Re-issue ``telescope.point(alt, az, wait=False)`` periodically.

    Started just after the initial slew, stopped before the next slew.
    Errors during the lightweight track-update are swallowed (logged
    once) so a transient socket error doesn't poison the cell.
    """

    def __init__(
        self,
        telescope,
        recompute_altaz: Callable[[], tuple[float, float]],
        interval_s: float = 10.0,
    ) -> None:
        self._telescope = telescope
        self._recompute = recompute_altaz
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name='survey-tracker', daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        warned = False
        while not self._stop.wait(self._interval_s):
            try:
                alt, az = self._recompute()
                self._telescope.point(alt, az, wait=False)
            except Exception as exc:
                if not warned:
                    print(f'  [tracker] update failed: {exc} (further errors silenced)')
                    warned = True


# ---------------------------------------------------------------------------
# Writer pool
# ---------------------------------------------------------------------------

class WriterPool:
    """Asynchronous dump writer.

    Save latency on the Pi is small (~tens of ms) but variable; running
    saves on a tiny pool keeps the coordinator's capture/correlate loop
    free of disk-jitter.

    ``noise_on`` dumps from a target named ``obs_*`` are rewritten to
    the ``cal_*`` cell directory at save time, matching the on-disk
    layout the analysis notebooks expect.
    """

    def __init__(
        self,
        outdir: str,
        n_workers: int = 2,
        on_save: Optional[Callable[[str, dict], None]] = None,
        queue_maxsize: int = 200,
    ) -> None:
        self._outdir = outdir
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._stop = threading.Event()
        self._on_save = on_save
        self._threads: list[threading.Thread] = []
        for i in range(n_workers):
            t = threading.Thread(
                target=self._worker, name=f'writer-{i}', daemon=False,
            )
            t.start()
            self._threads.append(t)

    def submit(self, dump: dict) -> None:
        self._q.put(dump)

    def stop_and_drain(self) -> None:
        """Wait for all queued dumps to flush, then shut down workers."""
        self._q.join()
        self._stop.set()
        for t in self._threads:
            t.join()
        self._threads.clear()

    def _worker(self) -> None:
        while True:
            try:
                dump = self._q.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self._save(dump)
            except Exception as exc:
                print(f'  [writer] save failed: {exc}')
            finally:
                self._q.task_done()

    def _save(self, dump: dict) -> None:
        name = dump['target_name']
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


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class SurveyCoordinator:
    """Drive a survey by iterating a scheduler over a hardware stack.

    Parameters
    ----------
    telescope : object
        Pointing controller exposing ``point(alt, az, wait=True|False)``.
    sdr_session : SDRSession
        Owns the dual-pol SDRs, noise diode, and pipelined correlate.
    scheduler : iterable of Cell
        Yields one ``Cell`` per pointing in observation order.
    outdir : str
        Root output directory; per-cell subdirectories are created lazily.
    n_writers : int
        Writer-pool size.
    track_interval_s : float
        Cadence for sidereal track updates within a cell.
    on_save, on_cell_start, on_cell_end : callable, optional
        Lifecycle hooks for logging / progress reporting.
    """

    def __init__(
        self,
        telescope,
        sdr_session,
        scheduler: Iterable[Cell],
        outdir: str,
        n_writers: int = 2,
        track_interval_s: float = 10.0,
        on_save: Optional[Callable[[str, dict], None]] = None,
        on_cell_start: Optional[Callable[[Pointing], None]] = None,
        on_cell_end: Optional[Callable[[Pointing], None]] = None,
    ) -> None:
        self._telescope = telescope
        self._sdr = sdr_session
        self._scheduler = scheduler
        self._writer = WriterPool(outdir, n_workers=n_writers, on_save=on_save)
        self._track_interval_s = track_interval_s
        self._on_cell_start = on_cell_start
        self._on_cell_end = on_cell_end

    def run(self) -> None:
        try:
            for cell in self._scheduler:
                self._run_cell(cell)
        except KeyboardInterrupt:
            print('\n[coordinator] KeyboardInterrupt -- draining writers.')
        finally:
            self._writer.stop_and_drain()

    # ------------------------------------------------------------------
    # Per-cell sequence
    # ------------------------------------------------------------------

    def _slew_with_monitor(
        self,
        alt: float,
        az: float,
        poll_sec: float = 0.5,
        min_slew_deg: float = 1.0,
    ) -> None:
        """Run blocking ``point(alt, az, wait=True)`` while a watcher thread
        prints live dish coordinates.

        Tracking nudges (current position already within ``min_slew_deg`` on
        both axes) skip the watcher so the log only shows real slews.  The
        watcher uses fresh sockets so it doesn't interfere with the in-flight
        ``wait``.
        """
        get_pointing = getattr(self._telescope, 'get_pointing', None)
        if get_pointing is None:
            self._telescope.point(alt, az, wait=True)
            return

        try:
            cur_alt, cur_az = get_pointing()
        except (OSError, TimeoutError, ValueError, AssertionError):
            cur_alt = cur_az = None

        is_tracking_nudge = (
            cur_alt is not None
            and abs(cur_alt - alt) < min_slew_deg
            and abs(cur_az - az) < min_slew_deg
        )
        if is_tracking_nudge:
            self._telescope.point(alt, az, wait=True)
            return

        stop = threading.Event()
        printed = threading.Event()

        def watch() -> None:
            while not stop.wait(poll_sec):
                try:
                    p_alt, p_az = get_pointing()
                except (OSError, TimeoutError, ValueError, AssertionError):
                    continue
                d_alt = p_alt - alt
                d_az = p_az - az
                print(
                    f'\r  [pointing] slew -> alt={alt:6.2f} az={az:6.2f} | '
                    f'now alt={p_alt:6.2f} az={p_az:6.2f} | '
                    f'd_alt={d_alt:+6.2f} d_az={d_az:+6.2f}   ',
                    end='', flush=True,
                )
                printed.set()

        watcher = threading.Thread(target=watch, name='slew-monitor', daemon=True)
        watcher.start()
        try:
            self._telescope.point(alt, az, wait=True)
        finally:
            stop.set()
            watcher.join(timeout=poll_sec * 4)
            if printed.is_set():
                print()

    def _run_cell(self, cell: Cell) -> None:
        p = cell.pointing
        if not cell.schedule:
            return

        # Pre-arm: start LO PLL settle and diode warm-up while the dish slews.
        first_lo, first_noise_on = cell.schedule[0]
        try:
            self._sdr.prearm(first_lo, first_noise_on)
        except Exception as exc:
            print(f'  [coordinator] prearm failed: {exc}')

        # Slew (blocking) with live pointing monitor.
        try:
            self._slew_with_monitor(p.alt_deg, p.az_deg)
        except Exception as exc:
            print(f'  [coordinator] slew failed for {p.target_name}: {exc}')
            return

        # Sidereal tracking during the cell (only if scheduler provided a recompute).
        tracker: TrackingThread | None = None
        if cell.recompute_altaz is not None:
            tracker = TrackingThread(
                self._telescope, cell.recompute_altaz,
                interval_s=self._track_interval_s,
            )
            tracker.start()

        if self._on_cell_start is not None:
            self._on_cell_start(p)

        try:
            for dump in self._sdr.run_schedule(cell.schedule):
                dump['target_name'] = p.target_name
                dump['alt_deg'] = p.alt_deg
                dump['az_deg'] = p.az_deg
                dump['ra_deg'] = p.ra_deg
                dump['dec_deg'] = p.dec_deg
                self._writer.submit(dump)
        finally:
            if tracker is not None:
                tracker.stop()
            if self._on_cell_end is not None:
                self._on_cell_end(p)
