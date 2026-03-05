from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..io import select_spectrum_by_center_freq
from ..models import Spectrum


@dataclass(frozen=True)
class YFactorMeasurement:
    center_freq_hz: float
    p_hot: float
    sigma_p_hot: float
    p_cold: float
    sigma_p_cold: float
    y: float
    sigma_y: float
    y_db: float
    sigma_y_db: float
    t_rx_k: float
    sigma_t_rx_meas_k: float
    sigma_t_rx_total_k: float
    t_hot_k: float
    sigma_t_hot_k: float
    t_cold_k: float
    sigma_t_cold_k: float

    @property
    def center_freq_mhz(self) -> int:
        return int(round(self.center_freq_hz / 1e6))

    def as_row(self) -> dict[str, float]:
        return {
            'LO_MHz': self.center_freq_mhz,
            'P_hot': self.p_hot,
            'sigma_P_hot': self.sigma_p_hot,
            'P_cold': self.p_cold,
            'sigma_P_cold': self.sigma_p_cold,
            'Y': self.y,
            'sigma_Y': self.sigma_y,
            'Y_dB': self.y_db,
            'sigma_Y_dB': self.sigma_y_db,
            'T_rx_K': self.t_rx_k,
            'sigma_T_rx_meas_K': self.sigma_t_rx_meas_k,
            'sigma_T_rx_total_K': self.sigma_t_rx_total_k,
            'T_hot_K': self.t_hot_k,
            'sigma_T_hot_K': self.sigma_t_hot_k,
            'T_cold_K': self.t_cold_k,
            'sigma_T_cold_K': self.sigma_t_cold_k,
        }


def receiver_temperature_from_y(y: float, t_hot_k: float, t_cold_k: float) -> float:
    """Receiver temperature from the Y-factor relation."""
    return (t_hot_k - y * t_cold_k) / (y - 1.0)


def measure_y_factor(
        hot: Spectrum,
        cold: Spectrum,
        *,
        t_hot_k: float,
        sigma_t_hot_k: float,
        t_cold_k: float,
        sigma_t_cold_k: float,
) -> YFactorMeasurement:
    """Measure Y-factor and propagated receiver temperature uncertainties."""
    p_hot = float(hot.total_power)
    p_cold = float(cold.total_power)
    sigma_p_hot = hot.total_power_sigma
    sigma_p_cold = cold.total_power_sigma

    y = p_hot / p_cold
    sigma_y = y * np.sqrt(
        (sigma_p_hot / p_hot) ** 2 + (sigma_p_cold / p_cold) ** 2
    )
    y_db = 10.0 * np.log10(y)
    sigma_y_db = (10.0 / np.log(10.0)) * (sigma_y / y)

    t_rx_k = receiver_temperature_from_y(y, t_hot_k, t_cold_k)
    d_tdy = (t_cold_k - t_hot_k) / (y - 1.0) ** 2
    d_tdth = 1.0 / (y - 1.0)
    d_tdtc = -y / (y - 1.0)

    sigma_t_rx_meas_k = abs(d_tdy) * sigma_y
    sigma_t_rx_total_k = float(np.sqrt(
        (d_tdy * sigma_y) ** 2
        + (d_tdth * sigma_t_hot_k) ** 2
        + (d_tdtc * sigma_t_cold_k) ** 2
    ))

    return YFactorMeasurement(
        center_freq_hz=hot.center_freq,
        p_hot=p_hot,
        sigma_p_hot=sigma_p_hot,
        p_cold=p_cold,
        sigma_p_cold=sigma_p_cold,
        y=y,
        sigma_y=float(sigma_y),
        y_db=float(y_db),
        sigma_y_db=float(sigma_y_db),
        t_rx_k=float(t_rx_k),
        sigma_t_rx_meas_k=float(sigma_t_rx_meas_k),
        sigma_t_rx_total_k=sigma_t_rx_total_k,
        t_hot_k=float(t_hot_k),
        sigma_t_hot_k=float(sigma_t_hot_k),
        t_cold_k=float(t_cold_k),
        sigma_t_cold_k=float(sigma_t_cold_k),
    )


def measure_y_factor_series(
        hot_spectra: Sequence[Spectrum],
        cold_spectra: Sequence[Spectrum],
        *,
        center_freqs_hz: Sequence[float],
        t_hot_k: float,
        sigma_t_hot_k: float,
        t_cold_k: Mapping[int | float, float],
        sigma_t_cold_k: Mapping[int | float, float],
        tol_hz: float = 0.5e6,
) -> list[YFactorMeasurement]:
    """Measure Y-factor quantities across multiple LO settings."""
    measurements = []
    for center_freq_hz in center_freqs_hz:
        hot = select_spectrum_by_center_freq(
            hot_spectra,
            center_freq_hz,
            tol_hz=tol_hz,
        )
        cold = select_spectrum_by_center_freq(
            cold_spectra,
            center_freq_hz,
            tol_hz=tol_hz,
        )
        lo_mhz = int(round(center_freq_hz / 1e6))
        t_cold = _lookup_temperature(t_cold_k, center_freq_hz, lo_mhz)
        sigma_t_cold = _lookup_temperature(sigma_t_cold_k, center_freq_hz, lo_mhz)
        measurements.append(
            measure_y_factor(
                hot,
                cold,
                t_hot_k=t_hot_k,
                sigma_t_hot_k=sigma_t_hot_k,
                t_cold_k=t_cold,
                sigma_t_cold_k=sigma_t_cold,
            )
        )
    return measurements


def _lookup_temperature(
        values: Mapping[int | float, float],
        center_freq_hz: float,
        lo_mhz: int,
) -> float:
    if lo_mhz in values:
        return float(values[lo_mhz])
    if center_freq_hz in values:
        return float(values[center_freq_hz])
    raise KeyError(
        f'No temperature entry found for center_freq={center_freq_hz / 1e6:.0f} MHz'
    )
