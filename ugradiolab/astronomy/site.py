import ugradio.leo as leo
import ugradio.nch as nch

NCH_LAT_DEG = nch.lat
NCH_LON_DEG = nch.lon
NCH_OBS_ALT_M = nch.alt

LEO_LAT_DEG = leo.lat
LEO_LON_DEG = leo.lon
LEO_OBS_ALT_M = leo.alt

__all__ = [
    "LEO_LAT_DEG",
    "LEO_LON_DEG",
    "LEO_OBS_ALT_M",
    "NCH_LAT_DEG",
    "NCH_LON_DEG",
    "NCH_OBS_ALT_M",
]
