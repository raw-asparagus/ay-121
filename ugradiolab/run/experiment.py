from abc import ABC, abstractmethod
from dataclasses import dataclass

import ugradio.nch as nch


@dataclass
class Experiment(ABC):
    """Shared base class for all experiment types."""
    alt_deg: float = 0.0
    az_deg:  float = 0.0
    outdir:  str   = 'data/'
    prefix:  str   = 'exp'
    lat:     float = nch.lat
    lon:     float = nch.lon
    obs_alt: float = nch.alt

    def _run_summary(self) -> list[str]:
        return []

    @abstractmethod
    def run(self) -> str: ...
