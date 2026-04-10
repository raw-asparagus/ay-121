"""Shared Lab 03 observing and correlator constants.

Every magic number in this file is attributed below to the entry in
``labs/03/REFERENCES.md`` from which it derives. The keys (e.g. [BIPM-SI],
[IERS2010], [BCD98]) resolve to full bibliographic entries in that file.
"""

# ---------------------------------------------------------------------------
# Correlator / receiver
#
# Source: AY 121 Lab 3 manual [AY121-Lab3]. The IF chain is a double-conversion
# heterodyne with first-LO 8.75 GHz and second-LO 1.54 GHz, sky band centred
# at LO1+LO2 = 10.29 GHz, with a 70 MHz analysis band defined by the first-IF
# bandpass-filter half-width of 35 MHz around 1.7 GHz.
# ---------------------------------------------------------------------------
F_S_HZ = 500e6                       # sample rate [Hz] — back-end FPGA, [AY121-Lab3]
N_FFT = 2048                          # FFT length per spectrum, [AY121-Lab3]
_LO1_HZ = 8750e6                      # first-stage local oscillator [Hz], [AY121-Lab3]
_LO2_HZ = 1540e6                      # second-stage local oscillator [Hz], [AY121-Lab3]
_IF1_BPF_CENTER_HZ = 1700e6           # first-IF bandpass centre [Hz], [AY121-Lab3]
_IF1_BPF_HALF_BW_HZ = 35e6            # first-IF bandpass half-width [Hz], [AY121-Lab3]

F_RF0_HZ = _LO1_HZ + _LO2_HZ          # sky frequency at IF=0
PLOT_BAND_GHZ = (                     # analysis band edges (sky frequency)
    (_LO1_HZ + _IF1_BPF_CENTER_HZ - _IF1_BPF_HALF_BW_HZ) / 1e9,
    (_LO1_HZ + _IF1_BPF_CENTER_HZ + _IF1_BPF_HALF_BW_HZ) / 1e9,
)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# Defined exactly by the SI [BIPM-SI 2019].
C_LIGHT_MS = 299_792_458.0
# Sidereal day and Earth's sidereal rotation rate from the IERS Conventions
# (2010) Technical Note 36 [IERS2010]. The two are related by
# omega_earth = 2*pi / T_sid.
SIDEREAL_DAY_S = 86_164.090_5
OMEGA_EARTH_RAD_S = 7.292_115_9e-5

# ---------------------------------------------------------------------------
# New Campbell Hall (NCH) observatory
#
# Coordinates from the AY 121 lab manual [AY121-Lab3, NCH-coords]. Berkeley,
# California; latitude is positive north, longitude positive east, so the
# negative longitude reflects the western hemisphere.
# ---------------------------------------------------------------------------
NCH_LAT_DEG = 37.873_199
NCH_LON_DEG = -122.257_063

# ---------------------------------------------------------------------------
# Nominal interferometer parameters
#
# Baseline measured in situ with an iPhone 13 Pro lidar between the two
# antenna phase centres on the NCH roof. Used as the prior for the nonlinear
# baseline fits in nb 04 and as the cross-check against the recovered values.
# The 30 cm 1-sigma is a deliberately conservative floor: Apple does not
# publish a metric accuracy spec for the iPhone lidar at multi-metre range;
# published independent benchmarks (e.g. Spreafico et al. 2021, Geosciences
# 11, 478) report few-cm random error growing with distance, plus dm-level
# systematic offsets on poorly-textured surfaces. The 30 cm folds in this
# single-shot lidar uncertainty AND an unmodelled contribution from
# identifying the antenna phase centre by eye on a real dish. See
# REFERENCES.md "Lidar prior" section for the full discussion.
# ---------------------------------------------------------------------------
NOMINAL_B_EW_M = 15.17  # east-west baseline [m] (iPhone 13 Pro lidar, 2026)
NOMINAL_B_NS_M = 1.36   # north-south baseline [m] (iPhone 13 Pro lidar, 2026)
NOMINAL_B_EW_ERR_M = 0.30
NOMINAL_B_NS_ERR_M = 0.30

# Bad channels — DC bin and FPGA pipeline harmonics at multiples of 256.
# Identified empirically from the time-averaged spectrum (see notebook 02a)
# and consistent with the back-end design notes in [AY121-Lab3].
BAD_CHANNELS = (0, 256, 512, 768)

# ---------------------------------------------------------------------------
# Solar reference
#
# Optical photospheric diameter at 1 AU. The 31.6' value here is the
# original value from the AY 121 lab manual / pre-existing constants.py
# (i.e. [AY121-Lab3]); it predates this analysis. It is appropriate for
# an observation near solar aphelion (May-August), when the apparent
# diameter is ~31.5'-31.7'.
#
# The *physical* mean diameter at 1 AU is closer to ~31.99', from
# helioseismic and direct-imaging determinations such as [BCD98] and
# [Meftah18]. Earth's orbital eccentricity (~1.7%) modulates the apparent
# diameter from ~31.46' (aphelion, ~Jul 4) to ~32.53' (perihelion, ~Jan 3).
#
# The radio diameter at cm wavelengths is *larger* than the optical value
# by a few percent due to the chromospheric/low-coronal optical-depth-unity
# surface lying a few thousand km above the photosphere -- a textbook claim
# [Stix] §10.3, [Dulk85] §III. The closest *measured* published radio
# radius to our 10 GHz observation is the Selhorst, Silva-Valio & Costa
# 2004 (A&A 420, 1117) value from the Nobeyama Radioheliograph at 17 GHz:
#     R_sun(17 GHz) = 976.6 +/- 1.5 arcsec
# i.e. theta_sun(17 GHz) = 32.55 +/- 0.05 arcmin. This is the radio
# reference value used in nb 05. The 10 GHz radius is expected to be
# *slightly larger* than at 17 GHz (longer wavelength -> higher
# chromosphere -> larger apparent radius); the K-band 18-26 GHz
# measurements of Marongiu et al. 2024 (A&A 684, A122) give R = 976-982
# arcsec, consistent with this picture. The student should treat the
# Selhorst+04 17 GHz value as the closest definitive reference.
#
# The student should verify 31.6' against the *current* AY 121 lab manual
# and the actual observation date (recoverable from `unix_mid` in nb 05).
# ---------------------------------------------------------------------------
SOLAR_DIAMETER_ARCMIN_NOMINAL = 31.6  # operational value from [AY121-Lab3]
