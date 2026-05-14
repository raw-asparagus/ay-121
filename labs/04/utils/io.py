"""Raw-dump loading and session summary helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import astropy.coordinates as ac
import astropy.units as u


RECAL_CELL_PREFIXES = ('obs_recal_', 'cal_recal_')

# Per-dump scalar fields: (record_key, npz_key, decoder).  Scalars round-trip
# through np.savez as 0-D arrays; the decoder extracts a Python scalar so
# downstream code sees ergonomic native types.  Array fields (corr00, corr11)
# are already np.float64 on disk and need no conversion.
_SCALAR_FIELDS = (
    ('lo_mhz', 'lo_freq_mhz', float),
    ('noise_on', 'noise_on', bool),
    ('time', 'time', float),
    ('alt', 'alt_deg', float),
    ('az', 'az_deg', float),
    ('ra', 'ra_deg', float),
    ('dec', 'dec_deg', float),
)


def load_session_dumps(
    data_dirs: list[Path],
    *,
    skip_recal: bool = True,
) -> list[dict]:
    """Load every .npz dump under ``data_dirs`` into a flat list of records.

    Each record carries the correlator spectra, observation metadata, and the
    galactic (l, b) of the pointing (rounded to integer b, two decimals on l).
    Recal-drift cells are skipped by default since they only feed the (now
    disabled) T_cal drift correction.
    """
    records: list[dict] = []
    for data_dir in data_dirs:
        survey = data_dir.name
        for session_dir in sorted(data_dir.glob('session_*')):
            session_id = f'{survey}/{session_dir.name}'
            cell_dirs = (sorted(session_dir.glob('obs_*'))
                         + sorted(session_dir.glob('cal_*')))
            for cell_dir in cell_dirs:
                if skip_recal and cell_dir.name.startswith(RECAL_CELL_PREFIXES):
                    continue
                for p in sorted(cell_dir.glob('*.npz')):
                    with np.load(p, allow_pickle=True) as f:
                        rec = {
                            'path': p,
                            'survey': survey,
                            'session': session_id,
                            'target': str(f['target_name']),
                            'corr00': f['corr00'],
                            'corr11': f['corr11'],
                        }
                        for out_key, npz_key, cast in _SCALAR_FIELDS:
                            rec[out_key] = cast(f[npz_key])
                    records.append(rec)

    # Galactic coordinates from RA/Dec. Round l to 2 dp and b to integer to
    # match the grid step used by the survey scheduler.
    for r in records:
        c = ac.SkyCoord(ra=r['ra'] * u.deg, dec=r['dec'] * u.deg, frame='icrs')
        r['gl'] = round(c.galactic.l.deg, 2)
        r['gb'] = round(c.galactic.b.deg)

    return records


def build_session_summary(
    records: list[dict],
    int_time_s: float,
) -> pd.DataFrame:
    """Per-session diagnostics: wall clock, cell count, dump count, duty cycle."""
    import datetime as dt

    def _utc(ts: float) -> str:
        return dt.datetime.fromtimestamp(ts, dt.UTC).strftime('%Y-%m-%d %H:%M')

    sessions = sorted({r['session'] for r in records})
    rows = []
    for s in sessions:
        sr = [r for r in records if r['session'] == s]
        if not sr:
            continue
        n_cells = len({(r['gl'], r['gb']) for r in sr if not r['noise_on']})
        times = [r['time'] for r in sr]
        t0, t1 = min(times), max(times)
        wall_s = t1 - t0
        duty_pct = (int_time_s * len(sr) / wall_s) * 100 if wall_s > 0 else float('nan')
        rows.append({
            'Session': s,
            'UTC Start': _utc(t0),
            'UTC End': _utc(t1),
            'Wall (min)': round(wall_s / 60, 1),
            'Obs. Cells': n_cells,
            'Dumps': len(sr),
            'Duty %': round(duty_pct, 1),
        })
    return pd.DataFrame(rows)
