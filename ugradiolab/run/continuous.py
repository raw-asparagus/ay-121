from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ..utils import make_path


class ContinuousCapture:
    """Pipelined capture loop: slew(N→N+1) and ephemeris(N+2) overlap with collect(N).

    Three things run concurrently during each collection window:

    1. ``interferometer.wait()``         — joins the in-progress slew
    2. ``make_experiment_fn() + _prepare()`` — computes the N+2 experiment
    3. ``np.savez()`` of capture N-1    — saves the previous datum

    After collection ends, the only remaining serial work is ``post_slew_verify``
    (~200 ms) before the next collection starts.

    Timeline::

        bootstrap:  point(N0, wait=True) → verify → pre-fetch N1
        loop:
          point(next_exp, wait=False)             ← fire slew (concurrent with collect)
          submit interferometer.wait()             ← background
          submit make_experiment_fn() + _prepare() ← background
          verify exp pre-collect
          compute_tau + read_data(exp)             ← collect N  (5–60 s)
                                                     ↑ slew, ephemeris, prev save all finish here
          submit np.savez(exp)                     ← background
          on_save callback
          wait_future.result()                     ← instant for small slews
          exp._verify_on_target('post-slew')       ← ~200 ms
          next_exp._prepare() already done
          exp = next_exp
          next_exp = prefetch_future.result()      ← instant

    Duty cycle:  ~93 %  (only ~300 ms of serial overhead per capture).

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator.
    pool_workers : int
        Background thread count.  4 suffices: save + wait + prefetch + spare.
    """

    def __init__(self, interferometer, snap, pool_workers: int = 4):
        self._interf            = interferometer
        self._snap              = snap
        self._executor          = ThreadPoolExecutor(max_workers=pool_workers)
        self._futures:          list[Future] = []
        self._wait_future:      Future | None = None
        self._prefetch_future:  Future | None = None

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
        except AssertionError as exc:
            raise RuntimeError(f'initial pointing out of range: {exc}') from exc
        exp._verify_on_target('bootstrap')

        # Pre-fetch the second experiment so its position is known before the
        # first collection fires the slew.
        next_exp = make_experiment_fn()
        next_exp._prepare()

        while True:
            # ── Fire slew to NEXT target at start of current collection ────
            try:
                self._interf.point(next_exp.alt_deg, next_exp.az_deg, wait=False)
            except AssertionError as exc:
                raise RuntimeError(f'pointing out of range: {exc}') from exc

            # ── Launch background work that runs during collection ─────────
            # 1. wait() for the slew just fired
            self._wait_future = self._executor.submit(self._interf.wait)

            # 2. Pre-compute the experiment after next (N+2): ephemeris is the
            #    bottleneck (~1–2 s on the Pi); doing it here costs nothing.
            def _prefetch(fn=make_experiment_fn):
                e = fn()
                e._prepare()
                return e
            self._prefetch_future = self._executor.submit(_prefetch)

            # ── Collect ────────────────────────────────────────────────────
            exp._verify_on_target('pre-collect')
            tau  = exp._compute_tau()
            data = exp._read_data(tau)   # ← 5–60 s; all background work lands here

            # ── Save current capture in background ─────────────────────────
            path = make_path(exp.outdir, exp.prefix, 'corr')
            self._futures.append(self._executor.submit(np.savez, path, **data))
            if on_save:
                on_save(path, exp)

            # ── Join slew (instant for small same-source moves) ────────────
            self._wait_future.result()
            self._wait_future = None

            # ── Advance ────────────────────────────────────────────────────
            exp = next_exp
            exp._verify_on_target('post-slew')   # ~200 ms — only serial overhead

            # ── Collect prefetched N+2 experiment (instant) ────────────────
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
        if errors:
            raise RuntimeError(f'{len(errors)} save(s) failed: {errors}')
