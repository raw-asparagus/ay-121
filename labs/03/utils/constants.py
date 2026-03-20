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

__all__ = [
    "F_S_HZ",
    "N_FFT",
    "LO1_HZ",
    "LO2_HZ",
    "IF1_BPF_CENTER_HZ",
    "IF1_BPF_HALF_BW_HZ",
    "F_RF0_HZ",
    "PLOT_BAND_GHZ",
]
