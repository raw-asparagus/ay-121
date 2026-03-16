# Data Product Schemas

Formal specifications for the `.npz` files produced and consumed by `ugradiolab`.

## Schema Files

| File | Describes |
|---|---|
| [record.schema.yaml](record.schema.yaml) | `.npz` files written by `Record.save` |
| [spectrum.schema.yaml](spectrum.schema.yaml) | `.npz` files written by `Spectrum.save` |

## YAML Field Entry Format

Each field entry in the schema files has the following keys:

```yaml
field_name:
  dtype: numpy dtype string (e.g. float64, int64, int8, bool)
  shape: scalar | [dim1, dim2, ...]   # symbolic names refer to scalar fields in the same file
  required: true | false
  units: string or "—" if dimensionless
  constraints: list of constraint strings, or "—"
  description: human-readable description
```

## Symbolic Shape Notation

Shape entries use symbolic names that refer to scalar fields stored in the same `.npz` file:

| Symbol | Meaning |
|---|---|
| `scalar` | 0-D array (shape `()`) — a single number |
| `[nblocks, nsamples, 2]` | 3-D array whose first two dimensions equal the `nblocks` and `nsamples` scalars stored in the same file; last axis is always length 2 |
| `[nsamples]` | 1-D array whose length equals the `nsamples` scalar stored in the same file |

## Key Differences Between Record and Spectrum Schemas

| | Record | Spectrum |
|---|---|---|
| Raw I/Q data | yes — `data` as `int8 (nblocks, nsamples, 2)` | no |
| Reduced PSD | no | yes — `psd`, `std`, `freqs` as `float64 (nsamples,)` |
| In-memory dtype of `data` | `complex64` | N/A |
| Typical file size | large (≫ 1 MB) | small (≈ 1 MB per spectrum) |

## Programmatic Validation

```python
import numpy as np

def validate_record(path):
    required = {
        'data', 'sample_rate', 'center_freq', 'gain', 'direct',
        'unix_time', 'jd', 'lst', 'alt', 'az',
        'obs_lat', 'obs_lon', 'obs_alt', 'nblocks', 'nsamples',
    }
    with np.load(path, allow_pickle=False) as f:
        missing = required - set(f.keys())
        if missing:
            raise ValueError(f'Missing keys: {missing}')
        assert f['data'].dtype == np.int8, f"data dtype: {f['data'].dtype}"
        assert f['data'].ndim == 3 and f['data'].shape[2] == 2
        nblocks  = int(f['nblocks'])
        nsamples = int(f['nsamples'])
        assert f['data'].shape[:2] == (nblocks, nsamples)
        print(f'OK: {nblocks} blocks × {nsamples} samples')

def validate_spectrum(path):
    required = {
        'psd', 'std', 'freqs',
        'sample_rate', 'center_freq', 'gain', 'direct',
        'unix_time', 'jd', 'lst', 'alt', 'az',
        'obs_lat', 'obs_lon', 'obs_alt', 'nblocks', 'nsamples',
    }
    with np.load(path, allow_pickle=False) as f:
        missing = required - set(f.keys())
        if missing:
            raise ValueError(f'Missing keys: {missing}')
        assert f['psd'].dtype == np.float64
        assert f['psd'].ndim == 1
        assert f['psd'].shape == f['std'].shape == f['freqs'].shape
        nsamples = int(f['nsamples'])
        assert f['psd'].size == nsamples
        print(f'OK: {nsamples} frequency bins')
```
