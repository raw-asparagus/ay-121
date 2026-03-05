"""
migrate_records.py — one-off migration of old-schema .npz files to the
current Record schema.

Old schema:
  data       complex128, shape (nblocks, nsamples)
  observer_lat / observer_lon / observer_alt

New schema (Record.save / Record.load):
  data       int8, shape (nblocks, nsamples, 2)   [I, Q stacked on axis=-1]
  obs_lat / obs_lon / obs_alt

Usage
-----
  python scripts/migrate_records.py              # migrate ./data/**/*.npz
  python scripts/migrate_records.py ./data/      # explicit root
  python scripts/migrate_records.py --dry-run    # validate without writing
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_REQUIRED_KEYS = frozenset({
    'data', 'sample_rate', 'center_freq', 'gain', 'direct',
    'unix_time', 'jd', 'lst', 'alt', 'az',
    'obs_lat', 'obs_lon', 'obs_alt',
    'nblocks', 'nsamples',
})


def _verify(path: Path) -> None:
    """Re-open the file and assert it satisfies Record.load() requirements."""
    with np.load(path, allow_pickle=False) as f:
        missing = _REQUIRED_KEYS - f.keys()
        if missing:
            raise ValueError(f'missing required keys: {missing}')
        data = f['data']
        if data.dtype != np.dtype(np.int8):
            raise ValueError(f'data dtype is {data.dtype}, expected int8')
        if data.ndim != 3 or data.shape[-1] != 2:
            raise ValueError(f'data shape is {data.shape}, expected (nblocks, nsamples, 2)')
        nblocks  = int(f['nblocks'])
        nsamples = int(f['nsamples'])
        if data.shape[:2] != (nblocks, nsamples):
            raise ValueError(
                f'data shape {data.shape[:2]} inconsistent with '
                f'nblocks={nblocks}, nsamples={nsamples}'
            )


def migrate_file(path: Path, dry_run: bool) -> str:
    """Migrate a single .npz file in-place.

    Returns a status string: 'migrated', 'skipped', or raises on error.
    """
    with np.load(path, allow_pickle=False) as f:
        data_c = f['data']

        # Already on the new schema — nothing to do.
        if data_c.dtype != np.dtype(np.complex128):
            return 'skipped'

        # ---- 1. Convert data: complex128 (nblocks, nsamples)
        #              → int8 (nblocks, nsamples, 2)
        iq = np.stack([
            data_c.real.astype(np.int8),
            data_c.imag.astype(np.int8),
        ], axis=-1)

        # ---- 2. Build the new NPZ dict (mirrors Record._to_npz_dict)
        npz = dict(
            data        = iq,
            sample_rate = np.float64(f['sample_rate']),
            center_freq = np.float64(f['center_freq']),
            gain        = np.float64(f['gain']),
            direct      = np.bool_(f['direct']),
            unix_time   = np.float64(f['unix_time']),
            jd          = np.float64(f['jd']),
            lst         = np.float64(f['lst']),
            alt         = np.float64(f['alt']),
            az          = np.float64(f['az']),
            obs_lat     = np.float64(f['observer_lat']),   # renamed
            obs_lon     = np.float64(f['observer_lon']),   # renamed
            obs_alt     = np.float64(f['observer_alt']),   # renamed
            nblocks     = np.int64(f['nblocks']),
            nsamples    = np.int64(f['nsamples']),
        )
        # Pass through optional siggen fields if present
        for key in ('siggen_freq', 'siggen_amp', 'siggen_rf_on'):
            if key in f:
                npz[key] = f[key]

    if dry_run:
        return 'dry-run ok'

    # ---- 3. Overwrite with canonical schema
    np.savez(str(path), **npz)

    # ---- 4. Verify the written file satisfies Record.load() requirements
    _verify(path)

    return 'migrated'


def main():
    parser = argparse.ArgumentParser(description='Migrate old-schema .npz files.')
    parser.add_argument(
        'data_root', nargs='?', default='./data',
        help='Root directory to search for .npz files (default: ./data)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Validate conversion logic without writing any files.',
    )
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        sys.exit(f'ERROR: data root not found: {root}')

    paths = sorted(root.rglob('*.npz'))
    if not paths:
        sys.exit(f'No .npz files found under {root}')

    migrated = skipped = errors = 0

    for path in paths:
        try:
            status = migrate_file(path, dry_run=args.dry_run)
        except Exception as exc:
            print(f'[ERROR] {path}: {exc}')
            errors += 1
            continue

        tag = {
            'migrated':    '[OK] migrated',
            'skipped':     '[--] skipped (already new schema)',
            'dry-run ok':  '[DRY] would migrate',
        }[status]

        print(f'{tag}: {path}')

        if status == 'migrated':
            migrated += 1
        elif status == 'skipped':
            skipped += 1

    print()
    if args.dry_run:
        dry = len(paths) - errors
        print(f'Summary (dry run): {dry} would be migrated, {errors} errors')
    else:
        print(f'Summary: {migrated} migrated, {skipped} already up-to-date, {errors} errors')

    if errors:
        sys.exit(1)


if __name__ == '__main__':
    main()
