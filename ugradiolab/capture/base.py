from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..astronomy.site import NCH_LAT_DEG, NCH_LON_DEG, NCH_OBS_ALT_M


@dataclass
class Experiment(ABC):
    """Shared base class for all experiment types.

    Attributes
    ----------
    alt_deg : float
        Target altitude in degrees.
    az_deg : float
        Target azimuth in degrees.
    outdir : str
        Directory where output files are written.
    prefix : str
        Filename prefix for generated products.
    lat : float
        Observer latitude in degrees.
    lon : float
        Observer longitude in degrees.
    obs_alt : float
        Observer altitude in meters.
    """
    alt_deg: float = 0.0
    az_deg:  float = 0.0
    outdir:  str   = 'data/'
    prefix:  str   = 'exp'
    lat:     float = NCH_LAT_DEG
    lon:     float = NCH_LON_DEG
    obs_alt: float = NCH_OBS_ALT_M

    def _run_summary(self) -> list[str]:
        """Return extra status lines for interactive runner output."""
        return []

    @abstractmethod
    def run(self) -> str:
        """Execute the experiment and persist its output.

        Returns
        -------
        path : str
            Path to the file written by the experiment.
        """
        ...
