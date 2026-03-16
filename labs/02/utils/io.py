from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import EXPECTED_EQUIPMENT_SCHEMA_VERSION
from .paths import (
    ATTENUATION_MANIFEST_PATH,
    EQUIPMENT_ARTIFACT_PATH,
    REPO_ROOT,
    SDR_GAIN_SWEEP_MANIFEST_PATH,
    TEMPERATURE_ARTIFACT_PATH,
    UNKNOWN_LENGTH_MANIFEST_PATH,
)

_MANIFEST_BASE_COLUMNS = [
    "set_id",
    "lo1420_path",
    "lo1421_path",
    "lo1420_total_power",
    "lo1421_total_power",
    "power_meter_dbm",
    "siggen_freq_mhz",
    "siggen_amp_dbm",
]
_MANIFEST_PATH_COLUMNS = ("lo1420_path", "lo1421_path")
_TEMPERATURE_REQUIRED_KEYS = [
    "t_rx_1420",
    "sigma_t_rx_1420",
    "t_rx_1421",
    "sigma_t_rx_1421",
    "t_cold",
    "t_hot",
    "sigma_hw_fraction",
    "cold_ref_method",
    "freq_hz_1420",
    "freq_hz_1421",
    "cold_ref_profile_1420",
    "cold_ref_profile_1421",
    "cold_ref_mask_1420",
    "cold_ref_mask_1421",
    "highest_unclipped_setpoint_dbm",
    "first_clipped_setpoint_dbm",
    "clip_threshold",
    "hardware_response_floor",
    "hardware_response_1420",
    "hardware_response_1421",
    "hardware_mask_1420",
    "hardware_mask_1421",
]
_SWEEP_REQUIRED_COLUMNS = [
    "point_id",
    "lo_mhz",
    "siggen_amp_dbm",
    "manual_meter_dbm",
    "total_power_db",
    "i_clip_frac",
    "q_clip_frac",
]


@dataclass(frozen=True)
class EquipmentArtifact:
    schema_version: str
    alpha_db_per_m: float
    sigma_alpha_db_per_m: float
    unknown_cable_length_m: float
    unknown_cable_length_sigma_m: float
    freq_offset_mhz: np.ndarray
    fir_response_norm: np.ndarray
    sum_response_norm: np.ndarray
    combined_response_norm: np.ndarray
    passband_mask: np.ndarray
    combined_eval_mask: np.ndarray
    highest_unclipped_setpoint_dbm: float
    first_clipped_setpoint_dbm: float
    clip_threshold: float
    sweep_rmse_db: float
    response_floor: float

    def to_compat_dict(self, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(raw or {})
        out.update(
            {
                "schema_version": self.schema_version,
                "alpha_db_per_m": self.alpha_db_per_m,
                "sigma_alpha_db_per_m": self.sigma_alpha_db_per_m,
                "unknown_cable_length_m": self.unknown_cable_length_m,
                "unknown_cable_length_sigma_m": self.unknown_cable_length_sigma_m,
                "freq_offset_mhz": self.freq_offset_mhz,
                "fir_response_norm": self.fir_response_norm,
                "sum_response_norm": self.sum_response_norm,
                "combined_response_norm": self.combined_response_norm,
                "passband_mask": self.passband_mask,
                "combined_eval_mask": self.combined_eval_mask,
                "highest_unclipped_setpoint_dbm": self.highest_unclipped_setpoint_dbm,
                "first_clipped_setpoint_dbm": self.first_clipped_setpoint_dbm,
                "clip_threshold": self.clip_threshold,
                "sweep_rmse_db": self.sweep_rmse_db,
                "response_floor": self.response_floor,
            }
        )
        return out


@dataclass(frozen=True)
class TemperatureArtifact:
    t_rx_1420: float
    sigma_t_rx_1420: float
    t_rx_1421: float
    sigma_t_rx_1421: float
    t_cold: float
    t_hot: float
    sigma_hw_fraction: float
    highest_unclipped_setpoint_dbm: float
    first_clipped_setpoint_dbm: float
    clip_threshold: float
    hardware_response_floor: float
    cold_ref_method: str
    freq_hz_1420: np.ndarray
    freq_hz_1421: np.ndarray
    cold_ref_profile_1420: np.ndarray
    cold_ref_profile_1421: np.ndarray
    cold_ref_mask_1420: np.ndarray
    cold_ref_mask_1421: np.ndarray
    hardware_response_1420: np.ndarray
    hardware_response_1421: np.ndarray
    hardware_mask_1420: np.ndarray
    hardware_mask_1421: np.ndarray

    def to_compat_dict(self, raw: dict[str, Any] | None = None) -> dict[str, Any]:
        out = dict(raw or {})
        out.update(
            {
                "t_rx_1420": self.t_rx_1420,
                "sigma_t_rx_1420": self.sigma_t_rx_1420,
                "t_rx_1421": self.t_rx_1421,
                "sigma_t_rx_1421": self.sigma_t_rx_1421,
                "t_cold": self.t_cold,
                "t_hot": self.t_hot,
                "sigma_hw_fraction": self.sigma_hw_fraction,
                "highest_unclipped_setpoint_dbm": self.highest_unclipped_setpoint_dbm,
                "first_clipped_setpoint_dbm": self.first_clipped_setpoint_dbm,
                "clip_threshold": self.clip_threshold,
                "hardware_response_floor": self.hardware_response_floor,
                "cold_ref_method": self.cold_ref_method,
                "freq_hz_1420": self.freq_hz_1420,
                "freq_hz_1421": self.freq_hz_1421,
                "cold_ref_profile_1420": self.cold_ref_profile_1420,
                "cold_ref_profile_1421": self.cold_ref_profile_1421,
                "cold_ref_mask_1420": self.cold_ref_mask_1420,
                "cold_ref_mask_1421": self.cold_ref_mask_1421,
                "hardware_response_1420": self.hardware_response_1420,
                "hardware_response_1421": self.hardware_response_1421,
                "hardware_mask_1420": self.hardware_mask_1420,
                "hardware_mask_1421": self.hardware_mask_1421,
            }
        )
        return out


def _available_keys(npz_obj: Any) -> list[str]:
    if hasattr(npz_obj, "files"):
        return list(npz_obj.files)
    return list(npz_obj.keys())


def require_keys(npz_obj: Any, required_keys: list[str], label: str) -> None:
    available = _available_keys(npz_obj)
    missing = [key for key in required_keys if key not in available]
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")


def npz_to_dict(npz_obj: Any) -> dict[str, np.ndarray]:
    return {name: np.asarray(npz_obj[name]) for name in _available_keys(npz_obj)}


def save_npz(path: Path, payload: dict[str, Any]) -> Path:
    np.savez(str(path), **payload)
    return path


def load_manifest(
    path: Path,
    *,
    require_cable_length: bool | None = None,
    label: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in _MANIFEST_PATH_COLUMNS:
        if col in df.columns:
            bad_rows = df.index[df[col].isna() | df[col].astype(str).str.strip().eq("")].tolist()
            if bad_rows:
                raise ValueError(f"{label or path}: empty path values in column {col!r} for rows {bad_rows}")
            df[col] = df[col].map(lambda s: (REPO_ROOT / str(s)).resolve())
            missing_rows = df.index[~df[col].map(Path.exists)].tolist()
            if missing_rows:
                raise FileNotFoundError(
                    f"{label or path}: missing files in column {col!r} for rows {missing_rows}"
                )
    if require_cable_length is None:
        return df
    return validate_manifest_frame(
        df,
        require_cable_length=require_cable_length,
        label=label or str(path),
    )


def attenuation_manifest() -> pd.DataFrame:
    df = load_manifest(
        ATTENUATION_MANIFEST_PATH,
        require_cable_length=True,
        label="attenuation manifest",
    )
    if df.empty:
        raise ValueError("attenuation manifest must contain at least one row.")
    return df


def unknown_length_manifest() -> pd.DataFrame:
    df = load_manifest(
        UNKNOWN_LENGTH_MANIFEST_PATH,
        require_cable_length=False,
        label="unknown-length manifest",
    )
    if len(df) != 1:
        raise ValueError(
            f"unknown-length manifest must contain exactly one row, got {len(df)}"
        )
    return df


def sdr_gain_sweep_manifest() -> pd.DataFrame:
    df = pd.read_csv(SDR_GAIN_SWEEP_MANIFEST_PATH)
    missing = [column for column in _SWEEP_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"sdr_gain_sweep manifest: missing required columns {missing}")
    out = df.copy()
    for column in _SWEEP_REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    bad_rows = out.index[out[_SWEEP_REQUIRED_COLUMNS].isna().any(axis=1)].tolist()
    if bad_rows:
        raise ValueError(
            f"sdr_gain_sweep manifest: non-finite required numeric fields in rows {bad_rows}"
        )
    if out.empty:
        raise ValueError("sdr_gain_sweep manifest must contain at least one row.")
    return out


def validate_manifest_frame(
    df: pd.DataFrame,
    *,
    require_cable_length: bool,
    label: str,
) -> pd.DataFrame:
    required = list(_MANIFEST_BASE_COLUMNS)
    if require_cable_length:
        required.append("cable_length_m")

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing required columns {missing}")

    out = df.copy()
    for column in _MANIFEST_PATH_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(lambda path: Path(path).resolve())
            blank_rows = out.index[out[column].astype(str).str.strip().eq("")].tolist()
            if blank_rows:
                raise ValueError(f"{label}: empty path values in column {column!r} for rows {blank_rows}")

    numeric_cols = [
        "set_id",
        "lo1420_total_power",
        "lo1421_total_power",
        "power_meter_dbm",
        "siggen_freq_mhz",
        "siggen_amp_dbm",
    ]
    if "cable_length_m" in out.columns:
        out["cable_length_m"] = pd.to_numeric(out["cable_length_m"], errors="coerce")
    if require_cable_length:
        numeric_cols.append("cable_length_m")

    for column in numeric_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    bad_rows = out.index[out[numeric_cols].isna().any(axis=1)].tolist()
    if bad_rows:
        raise ValueError(f"{label}: non-finite required numeric fields in rows {bad_rows}")
    if (out["lo1420_total_power"] <= 0).any() or (out["lo1421_total_power"] <= 0).any():
        raise ValueError(f"{label}: total_power must be positive for log-normalization")
    if require_cable_length and (out["cable_length_m"] < 0).any():
        raise ValueError(f"{label}: cable_length_m must be non-negative")
    return out


def _scalar_str(value: Any, *, label: str) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{label} must be scalar-like, got shape {arr.shape}")
    return str(arr.reshape(-1)[0])


def _scalar_float(value: Any, *, label: str, allow_nan: bool = False) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"{label} must be scalar-like, got shape {arr.shape}")
    scalar = float(arr.reshape(-1)[0])
    if not allow_nan and not np.isfinite(scalar):
        raise ValueError(f"{label} must be finite.")
    return scalar


def _vector_float(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {arr.shape}")
    return arr


def _vector_bool(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {arr.shape}")
    return arr


def _validate_same_shape(label: str, *arrays: np.ndarray) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"{label} arrays must share one shape, got {sorted(shapes)}")


def normalize_equipment_artifact(eq_dict: dict[str, Any]) -> dict[str, Any]:
    return equipment_artifact_from_dict(eq_dict).to_compat_dict(eq_dict)


def equipment_artifact_from_dict(eq_dict: dict[str, Any]) -> EquipmentArtifact:
    eq = dict(eq_dict)
    require_keys(
        eq,
        [
            "schema_version",
            "model.alpha_db_per_m",
            "model.sigma_alpha_db_per_m",
            "length.unknown_m",
            "length.sigma_unknown_m",
            "response.freq_offset_mhz",
            "response.fir_power_norm",
            "response.sum_power_norm",
            "response.combined_power_norm",
            "response.passband_mask",
            "response.eval_mask",
            "response.floor",
            "linearity.highest_unclipped_setpoint_dbm",
            "linearity.first_clipped_setpoint_dbm",
            "linearity.sweep_rmse_db",
            "linearity.clip_threshold",
        ],
        "equipment_calibration_results_v2.npz",
    )

    schema_version = _scalar_str(eq["schema_version"], label="schema_version")
    if schema_version != EXPECTED_EQUIPMENT_SCHEMA_VERSION:
        raise ValueError(
            f"equipment_calibration_results_v2.npz schema_version={schema_version!r} "
            f"!= expected {EXPECTED_EQUIPMENT_SCHEMA_VERSION!r}"
        )

    freq_offset_mhz = _vector_float(eq["response.freq_offset_mhz"], label="response.freq_offset_mhz")
    fir_response_norm = _vector_float(eq["response.fir_power_norm"], label="response.fir_power_norm")
    sum_response_norm = _vector_float(eq["response.sum_power_norm"], label="response.sum_power_norm")
    combined_response_norm = _vector_float(eq["response.combined_power_norm"], label="response.combined_power_norm")
    passband_mask = _vector_bool(eq["response.passband_mask"], label="response.passband_mask")
    combined_eval_mask = _vector_bool(eq["response.eval_mask"], label="response.eval_mask")
    _validate_same_shape(
        "equipment response",
        freq_offset_mhz,
        fir_response_norm,
        sum_response_norm,
        combined_response_norm,
        passband_mask,
        combined_eval_mask,
    )
    if not np.any(passband_mask):
        raise ValueError("equipment response.passband_mask must keep at least one channel.")
    if not np.any(combined_eval_mask):
        raise ValueError("equipment response.eval_mask must keep at least one channel.")
    eval_support = combined_eval_mask & np.isfinite(combined_response_norm)
    if not np.any(eval_support):
        raise ValueError("equipment response.eval_mask must include finite response values.")
    if np.any(combined_response_norm[eval_support] <= 0):
        raise ValueError("equipment combined response must be positive on eval_mask.")

    alpha_db_per_m = _scalar_float(eq["model.alpha_db_per_m"], label="model.alpha_db_per_m")
    sigma_alpha_db_per_m = _scalar_float(eq["model.sigma_alpha_db_per_m"], label="model.sigma_alpha_db_per_m")
    unknown_cable_length_m = _scalar_float(eq["length.unknown_m"], label="length.unknown_m")
    unknown_cable_length_sigma_m = _scalar_float(eq["length.sigma_unknown_m"], label="length.sigma_unknown_m")
    response_floor = _scalar_float(eq["response.floor"], label="response.floor")
    clip_threshold = _scalar_float(eq["linearity.clip_threshold"], label="linearity.clip_threshold")
    sweep_rmse_db = _scalar_float(eq["linearity.sweep_rmse_db"], label="linearity.sweep_rmse_db")
    if alpha_db_per_m <= 0:
        raise ValueError("model.alpha_db_per_m must be positive.")
    if sigma_alpha_db_per_m < 0:
        raise ValueError("model.sigma_alpha_db_per_m must be non-negative.")
    if unknown_cable_length_m < 0:
        raise ValueError("length.unknown_m must be non-negative.")
    if unknown_cable_length_sigma_m < 0:
        raise ValueError("length.sigma_unknown_m must be non-negative.")
    if response_floor <= 0:
        raise ValueError("response.floor must be positive.")
    if clip_threshold < 0:
        raise ValueError("linearity.clip_threshold must be non-negative.")
    if sweep_rmse_db < 0:
        raise ValueError("linearity.sweep_rmse_db must be non-negative.")

    return EquipmentArtifact(
        schema_version=schema_version,
        alpha_db_per_m=alpha_db_per_m,
        sigma_alpha_db_per_m=sigma_alpha_db_per_m,
        unknown_cable_length_m=unknown_cable_length_m,
        unknown_cable_length_sigma_m=unknown_cable_length_sigma_m,
        freq_offset_mhz=freq_offset_mhz,
        fir_response_norm=fir_response_norm,
        sum_response_norm=sum_response_norm,
        combined_response_norm=combined_response_norm,
        passband_mask=passband_mask,
        combined_eval_mask=combined_eval_mask,
        highest_unclipped_setpoint_dbm=_scalar_float(
            eq["linearity.highest_unclipped_setpoint_dbm"],
            label="linearity.highest_unclipped_setpoint_dbm",
        ),
        first_clipped_setpoint_dbm=_scalar_float(
            eq["linearity.first_clipped_setpoint_dbm"],
            label="linearity.first_clipped_setpoint_dbm",
        ),
        clip_threshold=clip_threshold,
        sweep_rmse_db=sweep_rmse_db,
        response_floor=float(response_floor),
    )


def validate_equipment_schema(eq: dict[str, Any]) -> None:
    schema_str = _scalar_str(eq["schema_version"], label="schema_version")
    if schema_str != EXPECTED_EQUIPMENT_SCHEMA_VERSION:
        raise ValueError(
            f"equipment_calibration_results_v2.npz schema_version={schema_str!r} "
            f"!= expected {EXPECTED_EQUIPMENT_SCHEMA_VERSION!r}"
        )


def temperature_artifact_from_dict(cal_dict: dict[str, Any]) -> TemperatureArtifact:
    cal = dict(cal_dict)
    require_keys(
        cal,
        _TEMPERATURE_REQUIRED_KEYS,
        "calibration_results_v2.npz",
    )

    freq_hz_1420 = _vector_float(cal["freq_hz_1420"], label="freq_hz_1420")
    freq_hz_1421 = _vector_float(cal["freq_hz_1421"], label="freq_hz_1421")
    cold_ref_profile_1420 = _vector_float(cal["cold_ref_profile_1420"], label="cold_ref_profile_1420")
    cold_ref_profile_1421 = _vector_float(cal["cold_ref_profile_1421"], label="cold_ref_profile_1421")
    cold_ref_mask_1420 = _vector_bool(cal["cold_ref_mask_1420"], label="cold_ref_mask_1420")
    cold_ref_mask_1421 = _vector_bool(cal["cold_ref_mask_1421"], label="cold_ref_mask_1421")
    hardware_response_1420 = _vector_float(cal["hardware_response_1420"], label="hardware_response_1420")
    hardware_response_1421 = _vector_float(cal["hardware_response_1421"], label="hardware_response_1421")
    hardware_mask_1420 = _vector_bool(cal["hardware_mask_1420"], label="hardware_mask_1420")
    hardware_mask_1421 = _vector_bool(cal["hardware_mask_1421"], label="hardware_mask_1421")
    _validate_same_shape(
        "LO 1420 calibration",
        freq_hz_1420,
        cold_ref_profile_1420,
        cold_ref_mask_1420,
        hardware_response_1420,
        hardware_mask_1420,
    )
    _validate_same_shape(
        "LO 1421 calibration",
        freq_hz_1421,
        cold_ref_profile_1421,
        cold_ref_mask_1421,
        hardware_response_1421,
        hardware_mask_1421,
    )

    if not np.all(np.isfinite(freq_hz_1420)) or not np.all(np.isfinite(freq_hz_1421)):
        raise ValueError("Calibration frequency axes must be finite.")
    if not np.any(cold_ref_mask_1420) or not np.any(cold_ref_mask_1421):
        raise ValueError("Cold-reference masks must keep at least one channel per LO.")
    if not np.any(hardware_mask_1420) or not np.any(hardware_mask_1421):
        raise ValueError("Hardware masks must keep at least one channel per LO.")
    for label, values in (
        ("cold_ref_profile_1420", cold_ref_profile_1420[cold_ref_mask_1420]),
        ("cold_ref_profile_1421", cold_ref_profile_1421[cold_ref_mask_1421]),
        ("hardware_response_1420", hardware_response_1420[hardware_mask_1420]),
        ("hardware_response_1421", hardware_response_1421[hardware_mask_1421]),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} must be finite on its support mask.")
        if np.any(values <= 0):
            raise ValueError(f"{label} must be positive on its support mask.")

    sigma_hw_fraction = _scalar_float(cal["sigma_hw_fraction"], label="sigma_hw_fraction")
    hardware_response_floor = _scalar_float(
        cal["hardware_response_floor"],
        label="hardware_response_floor",
    )
    if sigma_hw_fraction < 0:
        raise ValueError("sigma_hw_fraction must be non-negative.")
    if hardware_response_floor <= 0:
        raise ValueError("hardware_response_floor must be positive.")
    return TemperatureArtifact(
        t_rx_1420=_scalar_float(cal["t_rx_1420"], label="t_rx_1420"),
        sigma_t_rx_1420=_scalar_float(cal["sigma_t_rx_1420"], label="sigma_t_rx_1420"),
        t_rx_1421=_scalar_float(cal["t_rx_1421"], label="t_rx_1421"),
        sigma_t_rx_1421=_scalar_float(cal["sigma_t_rx_1421"], label="sigma_t_rx_1421"),
        t_cold=_scalar_float(cal["t_cold"], label="t_cold"),
        t_hot=_scalar_float(cal["t_hot"], label="t_hot"),
        sigma_hw_fraction=sigma_hw_fraction,
        highest_unclipped_setpoint_dbm=_scalar_float(
            cal["highest_unclipped_setpoint_dbm"],
            label="highest_unclipped_setpoint_dbm",
        ),
        first_clipped_setpoint_dbm=_scalar_float(
            cal["first_clipped_setpoint_dbm"],
            label="first_clipped_setpoint_dbm",
        ),
        clip_threshold=_scalar_float(cal["clip_threshold"], label="clip_threshold"),
        hardware_response_floor=hardware_response_floor,
        cold_ref_method=_scalar_str(cal["cold_ref_method"], label="cold_ref_method"),
        freq_hz_1420=freq_hz_1420,
        freq_hz_1421=freq_hz_1421,
        cold_ref_profile_1420=cold_ref_profile_1420,
        cold_ref_profile_1421=cold_ref_profile_1421,
        cold_ref_mask_1420=cold_ref_mask_1420,
        cold_ref_mask_1421=cold_ref_mask_1421,
        hardware_response_1420=hardware_response_1420,
        hardware_response_1421=hardware_response_1421,
        hardware_mask_1420=hardware_mask_1420,
        hardware_mask_1421=hardware_mask_1421,
    )


def load_equipment_artifact_typed(
    path: Path = EQUIPMENT_ARTIFACT_PATH,
) -> tuple[Path, EquipmentArtifact]:
    with np.load(path, allow_pickle=False) as data:
        eq = equipment_artifact_from_dict(npz_to_dict(data))
    return path, eq


def load_temperature_artifact_typed(
    path: Path = TEMPERATURE_ARTIFACT_PATH,
) -> tuple[Path, TemperatureArtifact]:
    with np.load(path, allow_pickle=False) as data:
        cal = temperature_artifact_from_dict(npz_to_dict(data))
    return path, cal


def load_equipment_artifact(path: Path = EQUIPMENT_ARTIFACT_PATH) -> tuple[Path, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        raw = npz_to_dict(data)
    eq = equipment_artifact_from_dict(raw)
    return path, eq.to_compat_dict(raw)


def load_temperature_artifact(
    path: Path = TEMPERATURE_ARTIFACT_PATH,
) -> tuple[Path, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        raw = npz_to_dict(data)
    cal = temperature_artifact_from_dict(raw)
    return path, cal.to_compat_dict(raw)
