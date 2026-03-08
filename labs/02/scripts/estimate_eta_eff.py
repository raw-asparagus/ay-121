#!/usr/bin/env python3
"""Estimate effective horn efficiency from existing calibrated lab-02 data.

This script computes an *effective* efficiency
    eta_eff = eta_ap * eta_bf = T_A,line,obs / T_B,line,ref
using:
1) the existing scalar/profile calibration in ``labs/02/calibration_results.npz``, and
2) the standard-field LO1420 combined spectrum.

Because this uses an assumed reference brightness, the result is explicitly
provisional and should be reported with large systematic uncertainty.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ugradiolab import Spectrum


DEFAULT_CALIBRATION_NPZ = Path("labs/02/calibration_results.npz")
DEFAULT_STANDARD_1420 = Path("data/lab02/standard_combined_spectra/GAL-1420_combined.npz")
DEFAULT_OUTPUT_CSV = Path("labs/02/report/eta_eff_estimate.csv")

REST_HI_HZ = 1_420_405_751.768

# Reference brightness for l=120, b=0 is model-assumed here.
# Keep a wide sigma because this is a provisional estimate from existing data only.
TB_REF_LINE_K = 80.0
TB_REF_LINE_SIGMA_K = 30.0


def _mad_sigma(values: np.ndarray) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_NPZ)
    parser.add_argument("--standard-1420", type=Path, default=DEFAULT_STANDARD_1420)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    cal = np.load(args.calibration)
    spectrum = Spectrum.load(args.standard_1420)

    trx_k = float(cal["t_rx_1420"])
    sigma_trx_k = float(cal["sigma_t_rx_1420"])
    tcold_k = float(cal["t_cold"])
    cold_ref = np.asarray(cal["cold_ref_profile_1420"], float)
    cold_mask = np.asarray(cal["cold_ref_mask_1420"], bool)

    psd = np.asarray(spectrum.psd, float)
    freqs_hz = np.asarray(spectrum.freqs, float)
    if psd.shape != cold_ref.shape:
        raise ValueError(
            f"Shape mismatch: standard psd has {psd.shape}, cold_ref has {cold_ref.shape}."
        )

    # Profile-based calibrated antenna temperature.
    ta_k = psd * (trx_k + tcold_k) / np.clip(cold_ref, 1e-12, None) - trx_k

    in_center = (freqs_hz > REST_HI_HZ - 0.6e6) & (freqs_hz < REST_HI_HZ + 0.6e6) & cold_mask
    in_off = (
        (
            ((freqs_hz > REST_HI_HZ - 1.2e6) & (freqs_hz < REST_HI_HZ - 0.6e6))
            | ((freqs_hz > REST_HI_HZ + 0.6e6) & (freqs_hz < REST_HI_HZ + 1.2e6))
        )
        & cold_mask
    )
    if not np.any(in_center) or not np.any(in_off):
        raise ValueError("Failed to construct center/offline windows for eta_eff estimate.")

    baseline_k = float(np.median(ta_k[in_off]))
    line_excess_k = ta_k[in_center] - baseline_k
    line_peak_est_k = float(np.quantile(line_excess_k, 0.95))
    line_noise_k = float(_mad_sigma(ta_k[in_off]))

    # Provisional systematic inflation:
    # - receiver calibration uncertainty from sigma_trx_k,
    # - residual transfer/model mismatch as 10% of estimated line peak.
    line_sigma_total_k = float(
        np.sqrt(line_noise_k**2 + sigma_trx_k**2 + (0.10 * line_peak_est_k) ** 2)
    )

    eta_eff = line_peak_est_k / TB_REF_LINE_K
    eta_eff_sigma = float(
        abs(eta_eff)
        * np.sqrt(
            (line_sigma_total_k / max(abs(line_peak_est_k), 1e-9)) ** 2
            + (TB_REF_LINE_SIGMA_K / TB_REF_LINE_K) ** 2
        )
    )
    eta_eff_ci95_lo = eta_eff - 1.96 * eta_eff_sigma
    eta_eff_ci95_hi = eta_eff + 1.96 * eta_eff_sigma

    result = {
        "trx_1420_k": trx_k,
        "sigma_trx_1420_k": sigma_trx_k,
        "tcold_k": tcold_k,
        "baseline_offline_k": baseline_k,
        "line_peak_est_k_q95": line_peak_est_k,
        "line_noise_offline_k_mad": line_noise_k,
        "line_sigma_total_k": line_sigma_total_k,
        "tb_ref_line_k": TB_REF_LINE_K,
        "tb_ref_line_sigma_k": TB_REF_LINE_SIGMA_K,
        "eta_eff": eta_eff,
        "eta_eff_sigma": eta_eff_sigma,
        "eta_eff_ci95_lo": eta_eff_ci95_lo,
        "eta_eff_ci95_hi": eta_eff_ci95_hi,
        "assumptions": (
            "eta_eff = eta_ap*eta_bf; standard-field LO1420; profile-calibrated TA; "
            "line peak from center-window q95 over offline-median baseline; "
            "Tb_ref assumed 80±30 K."
        ),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(f"Wrote effective-efficiency estimate to {args.output_csv}")
    print()
    print("Provisional eta_eff result:")
    print(f"  eta_eff           = {eta_eff:.3f} ± {eta_eff_sigma:.3f} (1σ)")
    print(f"  eta_eff (95% CI)  = [{eta_eff_ci95_lo:.3f}, {eta_eff_ci95_hi:.3f}]")
    print(f"  line_peak_est_k   = {line_peak_est_k:.2f} K  (q95 center-window excess)")
    print(f"  baseline_offline  = {baseline_k:.2f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
