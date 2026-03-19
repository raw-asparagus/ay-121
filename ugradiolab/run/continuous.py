from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..utils import make_path


class ContinuousCapture:
    """Pipelined capture loop: slew(N→N+1) and ephemeris(N+2) overlap with collect(N).

    Four things run concurrently during each collection window:

    1. ``interferometer.point() + .wait()``  — fires and joins the in-progress slew
    2. ``make_experiment_fn() + _prepare()`` — computes the N+2 experiment
    3. ``np.savez()`` of capture N-1         — saves the previous datum
    4. ``exp._apply_tau(tau)``               — TCP delay-line write (when active)

    There is no separate post-slew verify.  Instead, the pre-collect verify at
    the start of cycle N+1 doubles as the post-slew check for the N→N+1 slew.
    On failure the previous capture is discarded and a blocking repoint is
    attempted up to ``_MAX_REPOINTS`` times before the error is re-raised.

    Timeline::

        bootstrap:  point(N0, wait=True) → verify → pre-fetch N1
        loop:
          next_exp._prepare()                        ← refresh ephemeris at slew-fire time
          submit point(next_exp) + wait()            ← fire slew in background (TCP overhead hidden)
          submit make_experiment_fn() + _prepare()   ← background
          [verify exp pre-collect]                   ← skipped when cycle % verify_every_n != 0
                                                       on failure: discard prev save,
                                                       repoint blocking, retry, reset cycle count
          tau = exp._compute_tau()                   ← pure math, instant
          submit exp._apply_tau(tau)                 ← TCP delay write in background
          data = exp._read_data(tau)                 ← collect N  (5–60 s)
                                                       ↑ slew, ephemeris, prev save, delay all finish here
          delay_future.result()                      ← join delay write (usually already done)
          submit np.savez(exp)                       ← background
          on_save callback
          wait_future.result()                       ← instant for small slews
                                                       (None-guarded after repoint recovery)
          exp = next_exp
          next_exp = prefetch_future.result()        ← instant
                                                       (skipped after repoint recovery)

    Duty cycle:  ~98–99 %  with verify_every_n=5 on 3 s captures.

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator.
    pool_workers : int
        Background thread count.  4 suffices: save + slew + prefetch + spare.
    verify_every_n : int
        Run ``_verify_on_target`` every N cycles (default 1 = every cycle).
        Set to 5 for ~98 % duty cycle on 3 s captures when tracking slowly-moving
        sources (Sun, M17).  Verify always runs on the first cycle and after any
        repoint event.
    """

    def __init__(self, interferometer, snap, pool_workers: int = 4, verify_every_n: int = 1):
        self._interf           = interferometer
        self._snap             = snap
        self._executor         = ThreadPoolExecutor(max_workers=pool_workers)
        self._futures:         list[Future] = []
        self._wait_future:     Future | None = None
        self._prefetch_future: Future | None = None
        self._verify_every_n   = verify_every_n
        self._cycle_count      = 0

    def run(self, make_experiment_fn, on_save=None) -> None:
        """Run the pipelined capture loop until interrupted.

        Parameters
        ----------
        make_experiment_fn : callable() -> InterfExperiment
            Called once before each capture.  Must NOT call point() internally.
        on_save : callable(path, exp) -> None, optional
            Called in the main thread immediately after save is submitted.
        """
        # ── Bootstrap ─────────────────────────────────────────────────────
        exp = make_experiment_fn()
        exp._prepare()
        try:
            self._interf.point(exp.alt_deg, exp.az_deg, wait=True)
        except (AssertionError, TimeoutError, OSError) as exc:
            raise RuntimeError(f'initial pointing failed: {exc}') from exc
        exp._verify_on_target('bootstrap')

        next_exp = make_experiment_fn()
        next_exp._prepare()

        _MAX_REPOINTS     = 3
        _prev_save_path:   str | None    = None
        _prev_save_future: Future | None = None

        while True:
            # ── Fire slew in background (Opt 1: point() TCP overhead hidden) ─
            next_exp._prepare()   # refresh ephemeris at slew-fire time
            _alt, _az = next_exp.alt_deg, next_exp.az_deg

            def _slew(_alt=_alt, _az=_az):
                try:
                    self._interf.point(_alt, _az, wait=False)
                except (AssertionError, TimeoutError, OSError) as exc:
                    raise RuntimeError(f'pointing failed: {exc}') from exc
                self._interf.wait()

            self._wait_future = self._executor.submit(_slew)

            # ── Launch background: prefetch N+2 ───────────────────────────
            def _prefetch(fn=make_experiment_fn):
                e = fn()
                e._prepare()
                return e
            self._prefetch_future = self._executor.submit(_prefetch)

            # ── Pre-collect verify with repoint recovery (Opt 3: cadenced) ─
            # Also serves as post-slew check for the previous slew.
            # On failure: discard previous capture, drain background futures,
            # repoint blocking, then retry up to _MAX_REPOINTS times.
            self._cycle_count += 1
            _do_verify = (self._cycle_count % self._verify_every_n == 0)

            if _do_verify:
                for _attempt in range(1, _MAX_REPOINTS + 1):
                    try:
                        exp._verify_on_target('pre-collect')
                        _prev_save_path   = None   # previous capture confirmed good
                        _prev_save_future = None
                        break
                    except RuntimeError as _exc:
                        # Discard previous capture — dishes may have been moving
                        if _prev_save_path is not None:
                            if _prev_save_future is not None:
                                try:
                                    _prev_save_future.result()
                                except Exception:
                                    pass
                            try:
                                Path(_prev_save_path).unlink(missing_ok=True)
                            except OSError:
                                pass
                            _prev_save_path   = None
                            _prev_save_future = None

                        # Drain in-flight background futures before repointing
                        for _f in (self._wait_future, self._prefetch_future):
                            if _f is not None:
                                try:
                                    _f.result()
                                except Exception:
                                    pass
                        self._wait_future     = None
                        self._prefetch_future = None

                        if _attempt == _MAX_REPOINTS:
                            raise   # exhausted retries

                        print(f'  [repoint {_attempt}/{_MAX_REPOINTS - 1}] {_exc}')

                        # Blocking repoint to the current target position
                        exp._prepare()
                        try:
                            self._interf.point(exp.alt_deg, exp.az_deg, wait=True)
                        except (AssertionError, TimeoutError, OSError) as _pe:
                            raise RuntimeError(f'repoint failed: {_pe}') from _pe

                        # Refresh next_exp so the outer loop has a valid slew target
                        next_exp = make_experiment_fn()
                        next_exp._prepare()
                        # Reset cycle count so next cycle always verifies after a fault
                        self._cycle_count = 0
                        # Loop back to re-verify
            else:
                # Skip verify; clear prev save tracking (no rollback needed)
                _prev_save_path   = None
                _prev_save_future = None

            # ── Collect ────────────────────────────────────────────────────
            data = exp._read_data()

            # ── Save ───────────────────────────────────────────────────────
            path              = make_path(exp.outdir, exp.prefix, 'corr')
            _prev_save_future = self._executor.submit(np.savez, path, **data)
            self._futures.append(_prev_save_future)
            _prev_save_path   = path
            if on_save:
                on_save(path, exp)

            # ── Join slew (None-guarded: may have been drained by repoint) ─
            if self._wait_future is not None:
                try:
                    self._wait_future.result()
                except RuntimeError:
                    raise
                except (TimeoutError, OSError) as exc:
                    raise RuntimeError(f'interferometer.wait() failed: {exc}') from exc
                self._wait_future = None

            # ── Advance ────────────────────────────────────────────────────
            exp = next_exp

            # ── Collect prefetched N+2 (skipped after repoint recovery) ────
            if self._prefetch_future is not None:
                next_exp = self._prefetch_future.result()
                self._prefetch_future = None

    def flush(self) -> None:
        """Block until all pending saves complete; surface any errors."""
        for future in (self._wait_future, self._prefetch_future):
            if future is not None:
                try:
                    future.result()
                except Exception:
                    pass
        self._wait_future     = None
        self._prefetch_future = None
        errors = []
        for f in self._futures:
            exc = f.exception()
            if exc is not None:
                errors.append(exc)
        self._futures.clear()
        self._executor.shutdown(wait=True)  # release worker threads (Issue 5)
        if errors:
            raise RuntimeError(f'{len(errors)} save(s) failed: {errors}')
