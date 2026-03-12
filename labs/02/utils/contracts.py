from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    artifact: dict[str, Any] = field(default_factory=dict)
    artifact_path: Path | None = None
    tables: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class EquipmentCalibrationResult(StageResult):
    pass


@dataclass
class TemperatureCalibrationResult(StageResult):
    pass


@dataclass
class AnalysisResult(StageResult):
    pass
