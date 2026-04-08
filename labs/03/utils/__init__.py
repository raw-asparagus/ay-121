from .constants import (
    F_S_HZ,
    N_FFT,
    LO1_HZ,
    LO2_HZ,
    IF1_BPF_CENTER_HZ,
    IF1_BPF_HALF_BW_HZ,
    F_RF0_HZ,
    PLOT_BAND_GHZ,
)
from .plotting import (
    TEXTWIDTH_IN, COLUMNWIDTH_IN,
    LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, ANNOTATION_SIZE, EMPHASIS_SIZE,
    LW_NONE, LW_GRID, LW_FINE, LW_GUIDE, LW_LIGHT, LW_STANDARD,
    LW_MEDIUM, LW_STRONG, LW_FIT, LW_MODEL, LW_EMPHASIS, LW_CALLOUT, LW_LEVEL,
    LW_HAIRLINE,
    SCATTER_S_FINE, SCATTER_S_STANDARD, SCATTER_S_EMPHASIS, SCATTER_S_CALLOUT,
    MARKER_MS_FINE, MARKER_MS_SMALL, MARKER_MS_STANDARD, MARKER_MS_MEDIUM, MARKER_MS_LARGE,
    PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_COLOR, QUATERNARY_COLOR,
    QUINARY_COLOR, SENARY_COLOR, SEPTENARY_COLOR, NEUTRAL_COLOR,
    LIGHT_NEUTRAL_COLOR, NONARY_COLOR, COMPONENT_COLORS,
    ALPHA_SPAN_LIGHT, ALPHA_SHADE_LIGHT, ALPHA_SHADE_STANDARD, ALPHA_FILL,
    ALPHA_BAR, ERRORBAR_CAPSIZE_SMALL, BAR_HEIGHT_STANDARD,
    WATERFALL_HA_MIN_PER_IN, WATERFALL_PANEL_HEIGHT_IN, WATERFALL_GAP_FACTOR,
)
from .chips import (
    ChipSegmentation,
    segment_capture_times_by_gap,
)
from .captures import (
    CaptureSeries,
    load_capture_series,
)
from .chip_exports import (
    ProcessedSunChipSeries,
    load_processed_sun_chip_series,
)
from .dc import (
    LocalDCResult,
    local_real_dc_correction,
    AdaptiveDCResult,
    adaptive_real_dc_correction,
)
from .constants import (
    C_LIGHT_MS,
    SIDEREAL_DAY_S,
    OMEGA_EARTH_RAD_S,
    NCH_LAT_DEG,
    NCH_LON_DEG,
    NOMINAL_B_EW_M,
    NOMINAL_B_NS_M,
    BAD_CHANNELS,
    SOLAR_DIAMETER_ARCMIN_NOMINAL,
)
from .geometry import (
    jd_from_unix,
    lst_deg,
    hour_angle_deg,
    hour_angle_rad,
    geometric_delay_s,
    effective_geometric_delay_s,
    projected_baseline_lambda,
    fringe_frequency_hz,
    fringe_period_s,
    x_coordinate,
)
from .fringe_model import (
    FringeModelParams,
    SolarDiskParams,
    SunspotParams,
    point_source_visibility,
    point_source_fringes,
    uniform_disk_visibility_amplitude,
    uniform_disk_visibility_signed,
    uniform_disk_zeros,
    solar_visibility,
    fringe_envelope,
)
from .baseline_fitting import (
    BaselineResult,
    fft_baseline_single_channel,
    fft_baseline_broadband,
    phase_slope_baseline_single_channel,
    phase_slope_baseline_broadband,
    lag_delay_single_capture,
    lag_delay_baseline_series,
    nls_baseline_single_channel,
    nls_baseline_broadband,
    combined_baseline,
)
from .solar_analysis import (
    SolarDiameterResult,
    SunspotDetection,
    extract_fringe_envelope,
    find_bessel_zeros_in_envelope,
    solar_diameter_from_zeros,
    fit_solar_diameter_bessel,
    detect_sunspot_anomalies,
    characterize_sunspot_flux,
)
