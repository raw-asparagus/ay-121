from __future__ import annotations

from typing import Any

import numpy as np

COMMON_REQUIRED_METADATA_KEYS = frozenset({
    "sample_rate",
    "center_freq",
    "gain",
    "direct",
    "unix_time",
    "jd",
    "lst",
    "alt",
    "az",
    "obs_lat",
    "obs_lon",
    "obs_alt",
    "nblocks",
    "nsamples",
})

COMMON_SCALAR_FLOAT_FIELDS = (
    "sample_rate",
    "center_freq",
    "gain",
    "unix_time",
    "jd",
    "lst",
    "alt",
    "az",
    "obs_lat",
    "obs_lon",
    "obs_alt",
)
POSITIVE_FLOAT_FIELDS = frozenset({"sample_rate"})
OPTIONAL_FLOAT_FIELDS = ("siggen_freq", "siggen_amp")
OPTIONAL_BOOL_FIELDS = ("siggen_rf_on",)


def as_scalar(name: str, value: Any, *, kind: str) -> float | int | bool:
    """Coerce an array-like value into a validated scalar.

    Parameters
    ----------
    name : str
        Field name used in validation error messages.
    value : Any
        Candidate scalar value.
    kind : {'float', 'int', 'bool'}
        Target scalar type and validation rule set.

    Returns
    -------
    scalar : float or int or bool
        Validated scalar value converted to a builtin Python scalar type.

    Raises
    ------
    ValueError
        If ``value`` is not scalar, cannot be coerced to the requested type,
        is non-finite when ``kind='float'``, is not a positive integer when
        ``kind='int'``, or if ``kind`` is unknown.
    """
    arr = np.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {arr.shape}")
    item = arr.item()
    if kind == "float":
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} must be a real scalar, got boolean {item!r}")
        try:
            out = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real scalar, got {item!r}") from exc
        if not np.isfinite(out):
            raise ValueError(f"{name} must be finite, got {out!r}")
        return out
    if kind == "int":
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer, got boolean {item!r}")
        if isinstance(item, (int, np.integer)):
            out = int(item)
        else:
            try:
                numeric = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a positive integer, got {item!r}") from exc
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{name} must be a positive integer, got {item!r}")
            out = int(numeric)
        if out <= 0:
            raise ValueError(f"{name} must be > 0, got {out!r}")
        return out
    if kind == "bool":
        if isinstance(item, (bool, np.bool_)):
            return bool(item)
        if isinstance(item, (int, np.integer)) and item in (0, 1):
            return bool(item)
        raise ValueError(f"{name} must be boolean, got {item!r}")
    raise ValueError(f"Unknown scalar kind {kind!r}")


def missing_required_keys(keys: Any, required_keys: frozenset[str]) -> set[str]:
    """Return the required metadata keys that are absent.

    Parameters
    ----------
    keys : Any
        Iterable of keys present in the serialized payload.
    required_keys : frozenset of str
        Required key set for the payload schema.

    Returns
    -------
    missing : set of str
        Required keys that are not present in ``keys``.
    """
    return set(required_keys) - set(keys)


def optional_npz_value(npz: Any, key: str) -> Any:
    """Return an optional value from an ``np.load`` handle.

    Parameters
    ----------
    npz : Any
        ``NpzFile``-like object exposing membership checks and item lookup.
    key : str
        Optional field name.

    Returns
    -------
    value : Any
        Stored value if ``key`` is present, otherwise ``None``.
    """
    return npz[key] if key in npz else None


def set_common_metadata_fields(instance: Any) -> None:
    """Validate and normalize shared metadata fields on an object.

    Parameters
    ----------
    instance : Any
        Object exposing the common metadata attributes defined in this module.

    Returns
    -------
    None
        The object is mutated in place via ``object.__setattr__``.

    Raises
    ------
    ValueError
        If any required metadata field is not scalar, is non-finite, or fails
        the positivity checks enforced for the shared schema.
    """
    object.__setattr__(instance, "direct", as_scalar("direct", instance.direct, kind="bool"))

    for name in COMMON_SCALAR_FLOAT_FIELDS:
        value = as_scalar(name, getattr(instance, name), kind="float")
        if name in POSITIVE_FLOAT_FIELDS and value <= 0:
            raise ValueError(f"{name} must be > 0, got {value!r}")
        object.__setattr__(instance, name, value)

    for name in OPTIONAL_FLOAT_FIELDS:
        value = getattr(instance, name)
        if value is None:
            continue
        object.__setattr__(instance, name, as_scalar(name, value, kind="float"))

    for name in OPTIONAL_BOOL_FIELDS:
        value = getattr(instance, name)
        if value is None:
            continue
        object.__setattr__(instance, name, as_scalar(name, value, kind="bool"))
