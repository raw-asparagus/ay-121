from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ..utils import make_path


class ContinuousCapture:
    """Pipelined capture loop: slew(N+1) and save(N) overlap in parallel.

    Computation of the next experiment (pointing, duration) runs during the slew.

    Timeline::

        [_read_data N] → point(N+1, wait=False)  ← fires immediately
                       → submit savez(N)           ← background thread
                       → make_experiment_fn()       ← compute N+2 params
                       → interferometer.wait()      ← blocks until slew done
                       → _verify → [_read_data N+1] → ...

    This eliminates file-write latency from the inter-capture gap: the only dead
    time is the slew itself.

    Parameters
    ----------
    interferometer : ugradio.interf.Interferometer
        Connected interferometer controller.
    snap : snap_spec.snap.UGRadioSnap
        Initialised SNAP correlator (not used directly here; passed for symmetry
        and potential future use).
    save_workers : int
        Number of background threads for np.savez. Default 2 is sufficient since
        saves are short (<1 s) and slews are the bottleneck.
    """

    def __init__(self, interferometer, snap, save_workers: int = 2):
        self._interf   = interferometer
        self._snap     = snap
        self._executor = ThreadPoolExecutor(max_workers=save_workers)
        self._futures: list[Future] = []

    def run(self, make_experiment_fn, on_save=None) -> None:
        """Run the pipelined capture loop until interrupted.

        Parameters
        ----------
        make_experiment_fn : callable() -> InterfExperiment
            Called once before each capture.  Must NOT call point() internally.
            ContinuousCapture calls exp._prepare() to set alt_deg/az_deg.
        on_save : callable(path, exp) -> None, optional
            Called in the main thread immediately after the save is submitted
            (before interferometer.wait() returns).  Use for logging.
        """
        # Bootstrap: blocking slew for the very first capture.
        exp = make_experiment_fn()
        exp._prepare()
        try:
            self._interf.point(exp.alt_deg, exp.az_deg, wait=True)
        except AssertionError as exc:
            raise RuntimeError(f'initial pointing out of range: {exc}') from exc

        while True:
            exp._verify_on_target('pre-collect')
            tau  = exp._compute_tau()
            data = exp._read_data(tau)
            exp._verify_on_target('post-collect')

            # Prepare the NEXT experiment (ephemeris, duration — no hardware).
            next_exp = make_experiment_fn()
            next_exp._prepare()

            # Fire slew immediately (non-blocking).
            try:
                self._interf.point(next_exp.alt_deg, next_exp.az_deg, wait=False)
            except AssertionError as exc:
                raise RuntimeError(f'pointing out of range: {exc}') from exc

            # Submit save of current capture in background.
            path = make_path(exp.outdir, exp.prefix, 'corr')
            self._futures.append(self._executor.submit(np.savez, path, **data))
            if on_save:
                on_save(path, exp)

            # Block until slew completes — save runs concurrently.
            self._interf.wait()

            exp = next_exp

    def flush(self) -> None:
        """Block until all pending background saves complete.

        Raises RuntimeError summarising any save failures encountered.
        Clears the pending future list on exit.
        """
        errors = [f.exception() for f in self._futures if f.exception()]
        self._futures.clear()
        if errors:
            raise RuntimeError(f'{len(errors)} save(s) failed: {errors}')
