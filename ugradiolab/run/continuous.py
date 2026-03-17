from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ..utils import make_path


class ContinuousCapture:
    """Pipelined capture loop: slew(N→N+1) overlaps with collect(N).

    The slew to the *next* target fires at the **start** of the current
    collection.  For same-source tracking (Sun, M17), the slew is ~0.02°
    and completes in <1 s; the antenna is within tolerance for the entire
    5 s collection window.  By the time the collection finishes the slew
    has been done for ~4 s, so ``wait()`` returns instantly — no dead time.

    Timeline::

        bootstrap:  point(N0, wait=True) → verify
        pre-fetch:  make_fn() → next_exp(N1) → _prepare()
        loop:
          point(next_exp, wait=False)        ← fire slew at collection start
          submit interferometer.wait()        ← background thread
          verify exp pre-collect             ← antenna still on exp (0.02° slew)
          compute_tau + read_data(exp)        ← collect N  (5–60 s)
          submit np.savez(exp)               ← background thread
          on_save callback
          wait_future.result()               ← join slew (instant for small moves)
          exp._verify_on_target('post-slew') ← confirm settled on next target
          exp = next_exp
          next_exp = make_fn() → _prepare() ← prepare N+2 for next iteration

    Duty cycle:  ~97 %  (only ~100–200 ms of overhead per capture for
    post-slew verify + ephemeris, vs ~5 s gap in the blocking design).

    Source-switch note
    ------------------
    For a large slew (e.g. Sun → M17, ~30°) the antennas move ~15° during
    a 5 s collection.  ``_verify_on_target('post-collect')`` inside
    ``_read_data`` will raise ``RuntimeError``, correctly flagging the datum
    as invalid.  ``ContinuousCapture.run()`` propagates this; the caller
    should catch it, call ``flush()``, and restart.

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator (carried for symmetry).
    pool_workers : int
        Background thread count.  3 suffices: one for save, one for wait(),
        one spare.
    """

    def __init__(self, interferometer, snap, pool_workers: int = 3):
        self._interf         = interferometer
        self._snap           = snap
        self._executor       = ThreadPoolExecutor(max_workers=pool_workers)
        self._futures:       list[Future] = []
        self._wait_future:   Future | None = None

    def run(self, make_experiment_fn, on_save=None) -> None:
        """Run the pipelined capture loop until interrupted.

        Parameters
        ----------
        make_experiment_fn : callable() -> InterfExperiment
            Called once before each capture.  Must NOT call point() internally.
        on_save : callable(path, exp) -> None, optional
            Called in the main thread immediately after save is submitted.
        """
        # ── Bootstrap: blocking slew to the very first target ─────────────
        exp = make_experiment_fn()
        exp._prepare()
        try:
            self._interf.point(exp.alt_deg, exp.az_deg, wait=True)
        except AssertionError as exc:
            raise RuntimeError(f'initial pointing out of range: {exc}') from exc
        exp._verify_on_target('bootstrap')

        # Pre-fetch the second experiment so its position is ready before the
        # first collection loop iteration fires the slew.
        next_exp = make_experiment_fn()
        next_exp._prepare()

        while True:
            # ── Fire slew to NEXT target at the START of collecting CURRENT ─
            # The slew runs concurrently with the collection below.
            # For same-source tracking (~0.02° move) the antenna barely moves
            # during the 5 s window and stays within pointing_tol_deg.
            try:
                self._interf.point(next_exp.alt_deg, next_exp.az_deg, wait=False)
            except AssertionError as exc:
                raise RuntimeError(f'pointing out of range: {exc}') from exc
            self._wait_future = self._executor.submit(self._interf.wait)

            # ── Collect current experiment ─────────────────────────────────
            exp._verify_on_target('pre-collect')
            tau  = exp._compute_tau()
            data = exp._read_data(tau)         # ← 5–60 s; slew runs here

            # ── Save in background ─────────────────────────────────────────
            path = make_path(exp.outdir, exp.prefix, 'corr')
            self._futures.append(self._executor.submit(np.savez, path, **data))
            if on_save:
                on_save(path, exp)

            # ── Join slew (returns instantly for small moves) ──────────────
            self._wait_future.result()         # re-raises on hardware error
            self._wait_future = None

            # ── Advance to next experiment ─────────────────────────────────
            exp = next_exp
            exp._verify_on_target('post-slew')

            # ── Prepare the experiment after next (N+2) ─────────────────────
            next_exp = make_experiment_fn()
            next_exp._prepare()

    def flush(self) -> None:
        """Block until all pending saves complete; surface any errors."""
        if self._wait_future is not None:
            try:
                self._wait_future.result()
            except Exception:
                pass
            self._wait_future = None
        errors = []
        for f in self._futures:
            exc = f.exception()
            if exc is not None:
                errors.append(exc)
        self._futures.clear()
        if errors:
            raise RuntimeError(f'{len(errors)} save(s) failed: {errors}')
