from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
LAB02_DIR = UTILS_DIR.parent
REPO_ROOT = LAB02_DIR.parent.parent

DATA_ROOT = REPO_ROOT / "data" / "lab02"
CACHE_DIR = LAB02_DIR / "cache"
REPORT_DIR = LAB02_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"

EQUIPMENT_ARTIFACT_PATH = CACHE_DIR / "equipment_calibration_results_v2.npz"
TEMPERATURE_ARTIFACT_PATH = CACHE_DIR / "calibration_results_v2.npz"

ATTENUATION_MANIFEST_PATH = DATA_ROOT / "attenuation" / "manifest.csv"
UNKNOWN_LENGTH_MANIFEST_PATH = DATA_ROOT / "unknown_length" / "manifest.csv"
SDR_GAIN_SWEEP_MANIFEST_PATH = DATA_ROOT / "sdr_gain_sweep" / "manifest.csv"

HUMAN_SPECTRA_DIR = DATA_ROOT / "human_combined_spectra"
COLD_REF_SPECTRA_DIR = DATA_ROOT / "cold_ref_combined_spectra"
STANDARD_SPECTRA_DIR = DATA_ROOT / "standard_combined_spectra"
CYGNUS_X_SPECTRA_DIR = DATA_ROOT / "cygnus-x_combined_spectra"

COLD_REF_1420_PATH = COLD_REF_SPECTRA_DIR / "GAL-1420_combined.npz"
ETA_EFF_ESTIMATE_PATH = REPORT_DIR / "eta_eff_estimate.csv"


def ensure_output_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
