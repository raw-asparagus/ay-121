from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import EXPECTED_EQUIPMENT_SCHEMA_VERSION
from .paths import (
    ATTENUATION_MANIFEST_PATH,
    EQUIPMENT_ARTIFACT_PATH,
    REPO_ROOT,
    TEMPERATURE_ARTIFACT_PATH,
    UNKNOWN_LENGTH_MANIFEST_PATH,
)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("lo1420_path", "lo1421_path"):
        if col in df.columns:
            df[col] = df[col].apply(lambda s: (REPO_ROOT / str(s)).resolve())
    return df


def attenuation_manifest() -> pd.DataFrame:
    return load_manifest(ATTENUATION_MANIFEST_PATH)


def unknown_length_manifest() -> pd.DataFrame:
    return load_manifest(UNKNOWN_LENGTH_MANIFEST_PATH)


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


def normalize_equipment_artifact(eq_dict: dict[str, Any]) -> dict[str, Any]:
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
            "response.passband_mask",
            "response.eval_mask",
            "linearity.highest_unclipped_setpoint_dbm",
            "linearity.first_clipped_setpoint_dbm",
            "linearity.sweep_rmse_db",
        ],
        "equipment_calibration_results_v2.npz",
    )
    eq.setdefault("alpha_db_per_m", eq["model.alpha_db_per_m"])
    eq.setdefault("sigma_alpha_db_per_m", eq["model.sigma_alpha_db_per_m"])
    eq.setdefault("unknown_cable_length_m", eq["length.unknown_m"])
    eq.setdefault("unknown_cable_length_sigma_m", eq["length.sigma_unknown_m"])
    eq.setdefault("freq_offset_mhz", eq["response.freq_offset_mhz"])
    eq.setdefault("fir_response_norm", eq["response.fir_power_norm"])
    eq.setdefault("sum_response_norm", eq["response.sum_power_norm"])
    eq.setdefault(
        "combined_response_norm",
        eq.get(
            "response.combined_power_norm",
            np.asarray(eq["response.fir_power_norm"], float)
            * np.asarray(eq["response.sum_power_norm"], float),
        ),
    )
    eq.setdefault("passband_mask", eq["response.passband_mask"])
    eq.setdefault("combined_eval_mask", eq["response.eval_mask"])
    eq.setdefault(
        "highest_unclipped_setpoint_dbm",
        eq["linearity.highest_unclipped_setpoint_dbm"],
    )
    eq.setdefault("first_clipped_setpoint_dbm", eq["linearity.first_clipped_setpoint_dbm"])
    eq.setdefault("clip_threshold", eq.get("linearity.clip_threshold", np.nan))
    eq.setdefault("sweep_rmse_db", eq["linearity.sweep_rmse_db"])

    response_floor = eq.get("response.floor", np.nan)
    if not np.isfinite(response_floor):
        combined = np.asarray(eq["combined_response_norm"], float)
        mask = np.asarray(eq["combined_eval_mask"], bool)
        finite = np.isfinite(combined) & mask
        if np.any(finite):
            response_floor = float(np.nanmin(combined[finite]))
    if not np.isfinite(response_floor) or response_floor <= 0:
        response_floor = 10 ** (-20.0 / 10.0)
    eq["response_floor"] = float(response_floor)
    return eq


def validate_equipment_schema(eq: dict[str, Any]) -> None:
    schema_value = np.asarray(eq["schema_version"])
    schema_str = str(schema_value.item() if schema_value.ndim == 0 else schema_value)
    if schema_str != EXPECTED_EQUIPMENT_SCHEMA_VERSION:
        raise ValueError(
            f"equipment_calibration_results_v2.npz schema_version={schema_str!r} "
            f"!= expected {EXPECTED_EQUIPMENT_SCHEMA_VERSION!r}"
        )


def load_equipment_artifact(path: Path = EQUIPMENT_ARTIFACT_PATH) -> tuple[Path, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        eq = normalize_equipment_artifact(npz_to_dict(data))
    validate_equipment_schema(eq)
    return path, eq


def load_temperature_artifact(
    path: Path = TEMPERATURE_ARTIFACT_PATH,
) -> tuple[Path, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        cal = npz_to_dict(data)
    return path, cal
