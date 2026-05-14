"""Mock trial of SurveyCoordinator to localise the ~14 s gap residual.

Runs N synthetic cells through the REAL SurveyCoordinator + SDRSession
path with fakes for telescope / SDR USB / noise diode.  Per-cell phases
are timed via perf_counter hooks so the boundary off-source budget can
be attributed without involving hardware.

Fakes (realistic timings only):
  - telescope.point(wait=True) sleeps d_max / SLEW_RATE_DEG_S
  - sdr capture sleeps CAPTURE_SEC (1025 * 16384 / 3.2e6 ~= 5.25 s)
  - sdr correlate sleeps CORR_SEC (~0.30 s, measured-ish)
  - noise diode on/off is instant; set_noise honours real noise_settle_s
  - sdr.set_center_freq sleeps RETUNE_SEC (RTL PLL settle)

Run:
    python labs/04/scripts/trial_gap_breakdown.py [n_cells]
"""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ugradiolab.capture.coordinator import (  # noqa: E402
    Cell, Pointing, SurveyCoordinator,
)
from ugradiolab.capture.sdr_session import SDRSession, _Future  # noqa: E402


CAPTURE_SEC = float(os.environ.get('TRIAL_CAPTURE_SEC', '0.5'))
CORR_SEC = float(os.environ.get('TRIAL_CORR_SEC', '0.05'))
RETUNE_SEC = float(os.environ.get('TRIAL_RETUNE_SEC', '0.05'))
SLEW_RATE_DEG_S = float(os.environ.get('TRIAL_SLEW_DEG_S', '1.2'))
NOISE_SETTLE_S = float(os.environ.get('TRIAL_NOISE_SETTLE_S', '0.5'))
TRACK_INTERVAL_S = 10.0
# NB: capture is compressed for fast iteration; the residual = off - slew is
# what we read, not absolute gap.  Set TRIAL_CAPTURE_SEC=5.36 to match field.


EVENTS = []  # list of (t, cell_tag, label)
_lock = threading.Lock()


def log(cell_tag: str, label: str) -> None:
    with _lock:
        EVENTS.append((time.perf_counter(), cell_tag, label))


@contextmanager
def stage(cell_tag: str, label: str):
    log(cell_tag, f'{label}.start')
    try:
        yield
    finally:
        log(cell_tag, f'{label}.end')


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSDR:
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self._cell_tag = '-'

    def set_center_freq(self, hz: float) -> None:
        log(self._cell_tag, f'sdr{self.idx}.set_center_freq.start')
        time.sleep(RETUNE_SEC)
        log(self._cell_tag, f'sdr{self.idx}.set_center_freq.end')


class FakeNoise:
    def __init__(self) -> None:
        self._cell_tag = '-'

    def on(self) -> None:
        log(self._cell_tag, 'noise.on')

    def off(self) -> None:
        log(self._cell_tag, 'noise.off')


class FakeTelescope:
    def __init__(self, alt0: float, az0: float) -> None:
        self.alt = alt0
        self.az = az0
        self._cell_tag = '-'

    def point(self, alt: float, az: float, wait: bool = True) -> None:
        d_alt = abs(alt - self.alt)
        d_az = abs(az - self.az)
        d_max = max(d_alt, d_az)
        if wait:
            with stage(self._cell_tag, f'point.wait[d_max={d_max:.2f}]'):
                time.sleep(d_max / SLEW_RATE_DEG_S)
        else:
            log(self._cell_tag, 'point.nowait')
        self.alt = alt
        self.az = az

    def get_pointing(self):
        return self.alt, self.az


# ---------------------------------------------------------------------------
# Instrumented SDRSession
# ---------------------------------------------------------------------------

class TrialSDRSession(SDRSession):
    """Logging wrapper around SDRSession.

    When ``real=False`` (default), ``_capture`` and ``_submit_correlate`` are
    stubbed with sleeps so this runs without hardware.  When ``real=True``
    they delegate to the real SDRSession methods and only add log markers,
    so the wrapper is safe to use on the Pi with live SDRs.
    """

    cell_tag = '-'
    real = False

    def prearm(self, lo_mhz: float, noise_on: bool) -> None:
        with stage(self.cell_tag, f'prearm[lo={lo_mhz},on={noise_on}]'):
            super().prearm(lo_mhz, noise_on)

    def set_lo(self, lo_mhz: float) -> None:
        with stage(self.cell_tag, f'set_lo[{lo_mhz}]'):
            super().set_lo(lo_mhz)

    def set_noise(self, on: bool) -> None:
        with stage(self.cell_tag, f'set_noise[{on}]'):
            super().set_noise(on)

    def run_schedule(self, schedule):
        with stage(self.cell_tag, 'run_schedule'):
            yielded = 0
            for dump in super().run_schedule(schedule):
                yielded += 1
                log(self.cell_tag, f'run_schedule.yield[{yielded}]')
                yield dump

    def _capture(self):
        log(self.cell_tag, 'capture.start')
        if self.real:
            try:
                data, t = super()._capture()
            finally:
                log(self.cell_tag, 'capture.end')
            return data, t
        time.sleep(CAPTURE_SEC)
        log(self.cell_tag, 'capture.end')
        return {}, time.time()

    def _submit_correlate(self, data, t, lo, noise_on):
        tag = self.cell_tag
        if self.real:
            log(tag, 'correlate.submit')
            return super()._submit_correlate(data, t, lo, noise_on)

        fut = _Future()

        def _run() -> None:
            log(tag, 'correlate.start')
            time.sleep(CORR_SEC)
            log(tag, 'correlate.end')
            fut.set_result({
                'time': t, 'lo_freq_mhz': lo, 'noise_on': noise_on,
                'corr00': np.zeros(1024), 'corr01': np.zeros(1024, dtype=complex),
                'corr11': np.zeros(1024),
            })

        threading.Thread(target=_run, name='trial-correlate', daemon=True).start()
        return fut


# ---------------------------------------------------------------------------
# Real-hardware wrappers (add log markers without changing behaviour)
# ---------------------------------------------------------------------------

class LoggingTelescope:
    """Wrap a real telescope so point()/get_pointing() are logged."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cell_tag = '-'

    def point(self, alt: float, az: float, wait: bool = True) -> None:
        if wait:
            with stage(self._cell_tag, f'point.wait[alt={alt:.2f},az={az:.2f}]'):
                self._inner.point(alt, az, wait=True)
        else:
            log(self._cell_tag, f'point.nowait[alt={alt:.2f},az={az:.2f}]')
            self._inner.point(alt, az, wait=False)

    def get_pointing(self):
        return self._inner.get_pointing()


class LoggingNoise:
    def __init__(self, inner) -> None:
        self._inner = inner
        self._cell_tag = '-'

    def on(self) -> None:
        with stage(self._cell_tag, 'noise.on'):
            self._inner.on()

    def off(self) -> None:
        with stage(self._cell_tag, 'noise.off'):
            self._inner.off()


class LoggingSDR:
    """Wrap a real ugradio.sdr.SDR; logs set_center_freq duration."""

    def __init__(self, inner, idx: int) -> None:
        self._inner = inner
        self.idx = idx
        self._cell_tag = '-'

    def set_center_freq(self, hz: float) -> None:
        with stage(self._cell_tag, f'sdr{self.idx}.set_center_freq[{hz/1e6:.2f}MHz]'):
            self._inner.set_center_freq(hz)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Cell construction
# ---------------------------------------------------------------------------

def make_cells(n):
    """N obs cells offset by ~2 deg each (brick interleave proxy)."""
    cells = []
    alt = 40.0
    az = 180.0
    f1, f2 = 1419.86, 1421.14
    n_per_cell = int(os.environ.get('TRIAL_DUMPS_PER_CELL', '4'))
    schedule = [(f1, False), (f2, False)] * (n_per_cell // 2)
    for i in range(n):
        # Alternate small d_alt vs d_az like the real grid.
        if i % 2 == 0:
            az += 2.0
        else:
            alt += 2.0
        p = Pointing(
            target_name=f'obs_{int(az):03d}_{int(alt):+d}',
            alt_deg=alt, az_deg=az,
            ra_deg=0.0, dec_deg=0.0,
        )
        cells.append(Cell(pointing=p, schedule=schedule, recompute_altaz=None))
    return cells


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _make_real_cells(telescope, n):
    """Build N cells stepping ~2 deg in az from the current dish pointing.

    Keeps inside Leuschner limits (alt 17-83, az 7-348) so the trial is
    safe to run unattended.  Schedule is short to keep total time small.
    """
    cur_alt, cur_az = telescope.get_pointing()
    if not (20.0 < cur_alt < 80.0):
        cur_alt = 45.0
    if not (15.0 < cur_az < 340.0):
        cur_az = 180.0
    cells = []
    alt = float(cur_alt)
    az = float(cur_az)
    f1, f2 = 1419.86, 1421.14
    n_per_cell = int(os.environ.get('TRIAL_DUMPS_PER_CELL', '4'))
    schedule = [(f1, False), (f2, False)] * (n_per_cell // 2)
    for i in range(n):
        if i % 2 == 0:
            az = az + 2.0 if az < 340.0 else az - 2.0
        else:
            alt = alt + 2.0 if alt < 80.0 else alt - 2.0
        p = Pointing(
            target_name=f'obs_trial_{i:02d}',
            alt_deg=alt, az_deg=az,
            ra_deg=0.0, dec_deg=0.0,
        )
        cells.append(Cell(pointing=p, schedule=schedule, recompute_altaz=None))
    return cells


def _build_real_hardware():
    """Import and construct real Leuschner hardware.  Called only when --real."""
    from ugradio.leusch import LeuschNoise, LeuschTelescope
    from ugradio.sdr import SDR

    f1_mhz = 1419.86
    sample_rate = 3.2e6
    raw_sdrs = [
        SDR(device_index=0, direct=False,
            center_freq=f1_mhz * 1e6, sample_rate=sample_rate, gain=0.0),
        SDR(device_index=1, direct=False,
            center_freq=f1_mhz * 1e6, sample_rate=sample_rate, gain=0.0),
    ]
    telescope = LoggingTelescope(LeuschTelescope())
    noise = LoggingNoise(LeuschNoise())
    sdrs = [LoggingSDR(s, i) for i, s in enumerate(raw_sdrs)]
    return telescope, sdrs, noise


def main() -> None:
    argv = list(sys.argv[1:])
    use_real = '--real' in argv
    if use_real:
        argv.remove('--real')
    n_cells = int(argv[0]) if argv else 6

    outdir = Path('/tmp/trial_gap_breakdown')
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in outdir.iterdir():
        if sub.is_dir():
            for f in sub.iterdir():
                f.unlink()
            sub.rmdir()

    if use_real:
        telescope, sdrs, noise = _build_real_hardware()
        nsamples, nblocks = 16384, 1025
        print('[trial] REAL hardware mode -- Leuschner dish + 2x RTL-SDR')
    else:
        telescope = FakeTelescope(alt0=40.0, az0=180.0)
        sdrs = [FakeSDR(0), FakeSDR(1)]
        noise = FakeNoise()
        nsamples, nblocks = 16384, 1025
        print('[trial] MOCK mode -- sleep-based fakes')

    sdr_session = TrialSDRSession(
        sdrs=sdrs, noise=noise,
        nsamples=nsamples, nblocks=nblocks, nfft=1024,
        noise_settle_s=NOISE_SETTLE_S,
    )
    sdr_session.real = use_real

    if use_real:
        cells = _make_real_cells(telescope, n_cells)
    else:
        cells = make_cells(n_cells)

    # Bind cell tag onto every fake before each cell runs.
    def _scheduler():
        for i, c in enumerate(cells):
            tag = f'C{i}'
            telescope._cell_tag = tag
            noise._cell_tag = tag
            for s in sdrs:
                s._cell_tag = tag
            sdr_session.cell_tag = tag
            log(tag, 'cell.scheduler_yield')
            yield c

    coord = SurveyCoordinator(
        telescope=telescope,
        sdr_session=sdr_session,
        scheduler=_scheduler(),
        outdir=str(outdir),
        n_writers=2,
        track_interval_s=TRACK_INTERVAL_S,
        on_cell_start=lambda p: log(f'?{p.target_name}', 'on_cell_start'),
        on_cell_end=lambda p: log(f'?{p.target_name}', 'on_cell_end'),
    )

    t0 = time.perf_counter()
    log('-', 'run.start')
    coord.run()
    log('-', 'run.end')
    t_total = time.perf_counter() - t0

    print(f'\n=== trial completed in {t_total:.2f} s for {n_cells} cells ===')
    print(f'CAPTURE_SEC={CAPTURE_SEC:.3f}  CORR_SEC={CORR_SEC}  '
          f'RETUNE_SEC={RETUNE_SEC}  SLEW_RATE={SLEW_RATE_DEG_S} deg/s')

    # Group events by cell.
    by_cell = {}
    for t, tag, lbl in EVENTS:
        by_cell.setdefault(tag, []).append((t - t0, lbl))

    print('\n--- event log (rel seconds) ---')
    for tag in sorted(by_cell):
        if not tag.startswith('C'):
            continue
        evs = by_cell[tag]
        print(f'\n[{tag}]')
        for ts, lbl in evs:
            print(f'  t={ts:7.3f}   {lbl}')

    # Per-cell phase summary.
    print('\n--- per-cell phase durations (s) ---')
    print(f'{"cell":>4} {"sched_yld":>9} {"prearm":>7} {"slew":>6} '
          f'{"on_start":>8} {"sched":>6} {"first_cap":>9} {"cell_total":>10}')
    cell_starts = {}
    for tag in sorted(by_cell):
        if not tag.startswith('C'):
            continue
        evs = dict()  # label -> time
        for ts, lbl in by_cell[tag]:
            evs.setdefault(lbl, ts)
        # Find phase markers (prefix match).
        def find(prefix):
            for lbl, ts in evs.items():
                if lbl.startswith(prefix):
                    return ts
            return None

        sched_yld = evs.get('cell.scheduler_yield')
        prearm_s = find('prearm[')
        prearm_e = next(
            (ts for lbl, ts in evs.items()
             if lbl.startswith('prearm[') and lbl.endswith('.end')),
            None,
        )
        # stage() emits "label.start" and "label.end".  The lbl above for the
        # start was returned without the trailing .start because we stored the
        # raw label.  Adjust:
        prearm_s = next(
            (ts for lbl, ts in evs.items()
             if lbl.startswith('prearm[') and lbl.endswith('.start')),
            None,
        )
        slew_s = next(
            (ts for lbl, ts in evs.items()
             if lbl.startswith('point.wait[') and lbl.endswith('.start')),
            None,
        )
        slew_e = next(
            (ts for lbl, ts in evs.items()
             if lbl.startswith('point.wait[') and lbl.endswith('.end')),
            None,
        )
        on_start_s = next(
            (ts for lbl, ts in evs.items() if lbl == 'on_cell_start'),
            None,
        )
        sched_s = evs.get('run_schedule.start')
        sched_e = evs.get('run_schedule.end')
        first_cap_s = evs.get('capture.start')
        first_cap_e = evs.get('capture.end')

        def dur(a, b):
            if a is None or b is None:
                return float('nan')
            return b - a

        print(f'{tag:>4} '
              f'{(sched_yld or float("nan")):>9.3f} '
              f'{dur(prearm_s, prearm_e):>7.3f} '
              f'{dur(slew_s, slew_e):>6.3f} '
              f'{dur(on_start_s, sched_s) if on_start_s else 0.0:>8.3f} '
              f'{dur(sched_s, sched_e):>6.2f} '
              f'{dur(first_cap_s, first_cap_e):>9.3f} '
              f'{dur(sched_yld, sched_e):>10.2f}')
        cell_starts[tag] = (sched_yld, first_cap_s, first_cap_e, sched_e)

    # Boundary breakdown: gap between last capture.end of cell N and first
    # capture.end of cell N+1 (matches the field analysis's prev/next .t).
    print('\n--- inter-cell boundaries (matches slew_gap_breakdown) ---')
    print(f'{"prev->next":>10}  {"measured_gap":>12}  {"off=gap-cap":>11}  '
          f'{"residual=off-slew":>17}')
    tags = [t for t in sorted(by_cell) if t.startswith('C')]
    # last capture.end per cell:
    last_cap_end = {}
    first_cap_end = {}
    slew_dur = {}
    for tag in tags:
        caps_end = [ts for ts, lbl in by_cell[tag] if lbl == 'capture.end']
        caps_start = [ts for ts, lbl in by_cell[tag] if lbl == 'capture.start']
        if caps_end:
            first_cap_end[tag] = caps_end[0]
            last_cap_end[tag] = caps_end[-1]
        # slew duration
        s_s = next((ts for ts, lbl in by_cell[tag]
                    if lbl.startswith('point.wait[') and lbl.endswith('.start')),
                   None)
        s_e = next((ts for ts, lbl in by_cell[tag]
                    if lbl.startswith('point.wait[') and lbl.endswith('.end')),
                   None)
        slew_dur[tag] = (s_e - s_s) if (s_s and s_e) else 0.0

    for prev, nxt in zip(tags[:-1], tags[1:]):
        if prev not in last_cap_end or nxt not in first_cap_end:
            continue
        gap = first_cap_end[nxt] - last_cap_end[prev]
        off = gap - CAPTURE_SEC
        residual = off - slew_dur[nxt]
        print(f'{prev:>4}->{nxt:<4}  {gap:>12.3f}  {off:>11.3f}  '
              f'{residual:>17.3f}   (slew={slew_dur[nxt]:.2f})')


if __name__ == '__main__':
    main()
