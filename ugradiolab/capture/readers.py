"""Reader factory functions for the streaming capture pipeline.

Currently exports:

* :func:`make_snap_reader` -- wraps a SNAP FPGA correlator for use with
  :class:`~ugradiolab.capture.streaming.StreamingCapture` (used by
  ``labs/03``).

The dual-polarisation SDR reader that previously lived here was
superseded by :class:`~ugradiolab.capture.sdr_session.SDRSession` and
:class:`~ugradiolab.capture.coordinator.SurveyCoordinator`.
"""

from __future__ import annotations

from typing import Callable


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
