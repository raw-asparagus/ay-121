from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

LAB03_PATH = Path(__file__).resolve().parents[1] / "labs" / "03"


@contextmanager
def _lab03_utils():
    lab03_path = str(LAB03_PATH)
    sys.path.insert(0, lab03_path)
    try:
        chips = importlib.import_module("utils.chips")
        plotter = importlib.import_module("utils.plotter")
        yield chips, plotter
    finally:
        if sys.path and sys.path[0] == lab03_path:
            sys.path.pop(0)
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)


def _capture(start: float, end: float) -> dict[str, float]:
    return {"unix_time_start": start, "unix_time_end": end}


def test_segment_captures_by_gap_splits_at_all_large_breaks():
    files = [
        _capture(77.6, 87.6),
        _capture(0.0, 10.0),
        _capture(169.9, 179.9),
        _capture(10.3, 20.3),
        _capture(67.3, 77.3),
    ]
    paths = [Path(f"capture_{idx}.npz") for idx in range(len(files))]
    ha_deg_unsorted = np.array([2.0, -3.0, 4.0, -2.0, 1.0])

    with _lab03_utils() as (chips, _plotter):
        sorted_caps = chips.sort_captures_by_mid_unix(files, paths)
        ha_deg = ha_deg_unsorted[sorted_caps.order]

        chip_info = chips.segment_captures_by_gap(sorted_caps.files_sorted, sorted_caps.unix_sorted, ha_deg)

        assert chip_info.gap_break_indices.tolist() == [2, 4]
        assert np.allclose(chip_info.gap_break_values_sec, [47.0, 82.3], atol=0.2)
        assert chip_info.N_caps == [2, 2, 1]
        assert chip_info.cap_chip.tolist() == [0, 0, 1, 1, 2]
        assert chip_info.split_gap_threshold_sec == 5.0


def test_plot_interval_baseline_handles_three_chips():
    ha_chips = [
        np.array([-80.0, -70.0]),
        np.array([-55.0, -45.0]),
        np.array([-30.0, -20.0]),
    ]
    interval_results = [
        {"ha_ctr": -72.5, "B_EW": 10.0, "B_EW_err": 0.5, "chip": 0, "lo": -75.0, "hi": -70.0},
        {"ha_ctr": -72.5, "B_EW": 11.0, "B_EW_err": 0.6, "chip": 1, "lo": -75.0, "hi": -70.0},
        {"ha_ctr": -72.5, "B_EW": 12.0, "B_EW_err": 0.7, "chip": 2, "lo": -75.0, "hi": -70.0},
    ]

    with _lab03_utils() as (_chips, plotter):
        fig, ax = plotter.plot_interval_baseline(
            ha_chips=ha_chips,
            baseline_lag_chips=[10.0, 11.0, 12.0],
            interval_results=interval_results,
        )

        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "chip 0 interval",
            "chip 1 interval",
            "chip 2 interval",
        ]
        fig.canvas.draw()


def test_plot_lag_delay_summary_accepts_nonzero_example_chip():
    ha_chips = [
        np.array([-80.0, -70.0]),
        np.array([-55.0, -45.0]),
        np.array([-30.0, -20.0]),
    ]
    tau_lag_chips = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
    ]
    sin_h_lag_chips = [np.sin(np.radians(ha)) for ha in ha_chips]
    coeffs_lag_chips = [
        np.array([0.0, 1.0]),
        np.array([0.5, 0.8]),
        np.array([1.0, 0.6]),
    ]

    with _lab03_utils() as (_chips, plotter):
        fig, axes = plotter.plot_lag_delay_summary(
            tau_axis_ns=np.linspace(-10.0, 10.0, 101),
            lag_amp=np.linspace(0.0, 1.0, 101),
            tau_peak_ns=5.0,
            lag_snr=12.0,
            example_chip_idx=2,
            capture_index=1,
            ha_chips=ha_chips,
            tau_lag_chips=tau_lag_chips,
            baseline_lag_chips=[10.0, 11.0, 12.0],
            sin_h_lag_chips=sin_h_lag_chips,
            coeffs_lag_chips=coeffs_lag_chips,
            ha_limits_deg=(-85.0, -15.0),
        )

        assert "chip 2" in axes[0].get_title()
        fig.canvas.draw()
