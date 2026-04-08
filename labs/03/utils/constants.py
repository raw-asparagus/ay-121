"""Shared Lab 03 observing and correlator constants."""

F_S_HZ = 500e6
N_FFT = 2048
LO1_HZ = 8750e6
LO2_HZ = 1540e6
IF1_BPF_CENTER_HZ = 1700e6
IF1_BPF_HALF_BW_HZ = 35e6

F_RF0_HZ = LO1_HZ + LO2_HZ
PLOT_BAND_GHZ = (
    (LO1_HZ + IF1_BPF_CENTER_HZ - IF1_BPF_HALF_BW_HZ) / 1e9,
    (LO1_HZ + IF1_BPF_CENTER_HZ + IF1_BPF_HALF_BW_HZ) / 1e9,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT_MS = 299_792_458.0  # speed of light [m/s]
SIDEREAL_DAY_S = 86_164.090_5  # sidereal day [s]
OMEGA_EARTH_RAD_S = 7.292_115_9e-5  # Earth sidereal rotation rate [rad/s]

# ---------------------------------------------------------------------------
# New Campbell Hall (NCH) observatory
# ---------------------------------------------------------------------------
NCH_LAT_DEG = 37.873_199
NCH_LON_DEG = -122.257_063

# ---------------------------------------------------------------------------
# Nominal interferometer parameters
# ---------------------------------------------------------------------------
NOMINAL_B_EW_M = 20.0  # approximate east-west baseline [m]
NOMINAL_B_NS_M = 0.0  # approximate north-south baseline [m]
BAD_CHANNELS = (0, 256, 512, 768)  # DC / FPGA harmonic channels

# ---------------------------------------------------------------------------
# Solar reference
# ---------------------------------------------------------------------------
SOLAR_DIAMETER_ARCMIN_NOMINAL = 31.6  # approximate solar angular diameter

__all__ = [
    "F_S_HZ",
    "N_FFT",
    "LO1_HZ",
    "LO2_HZ",
    "IF1_BPF_CENTER_HZ",
    "IF1_BPF_HALF_BW_HZ",
    "F_RF0_HZ",
    "PLOT_BAND_GHZ",
    "C_LIGHT_MS",
    "SIDEREAL_DAY_S",
    "OMEGA_EARTH_RAD_S",
    "NCH_LAT_DEG",
    "NCH_LON_DEG",
    "NOMINAL_B_EW_M",
    "NOMINAL_B_NS_M",
    "BAD_CHANNELS",
    "SOLAR_DIAMETER_ARCMIN_NOMINAL",
]
