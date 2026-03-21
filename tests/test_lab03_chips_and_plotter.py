from __future__ import annotations

import importlib
import sys
import tempfile
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
        captures = importlib.import_module("utils.captures")
        chips = importlib.import_module("utils.chips")
        plotter = importlib.import_module("utils.plotter")
        dc = importlib.import_module("utils.dc")
        yield captures, chips, plotter, dc
    finally:
        if sys.path and sys.path[0] == lab03_path:
            sys.path.pop(0)
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)


def test_segment_capture_times_by_gap_splits_at_all_large_breaks():
    unix_time_start = np.array([77.6, 0.0, 169.9, 10.3, 67.3], dtype=float)
    unix_time_end = np.array([87.6, 10.0, 179.9, 20.3, 77.3], dtype=float)
    unix_mid = 0.5 * (unix_time_start + unix_time_end)
    ha_deg_unsorted = np.array([2.0, -3.0, 4.0, -2.0, 1.0], dtype=float)
    order = np.argsort(unix_mid)

    with _lab03_utils() as (_captures, chips, _plotter, _dc):
        chip_info = chips.segment_capture_times_by_gap(
            unix_time_start_sorted=unix_time_start[order],
            unix_time_end_sorted=unix_time_end[order],
            unix_sorted=unix_mid[order],
            ha_deg=ha_deg_unsorted[order],
        )

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
    unix_chips = [
        np.array([100.0, 110.0]),
        np.array([200.0, 210.0]),
        np.array([300.0, 310.0]),
    ]
    interval_results = [
        {"ha_ctr": -72.5, "B_EW": 10.0, "B_EW_err": 0.5, "chip": 0, "lo": -75.0, "hi": -70.0},
        {"ha_ctr": -72.5, "B_EW": 11.0, "B_EW_err": 0.6, "chip": 1, "lo": -75.0, "hi": -70.0},
        {"ha_ctr": -72.5, "B_EW": 12.0, "B_EW_err": 0.7, "chip": 2, "lo": -75.0, "hi": -70.0},
    ]

    with _lab03_utils() as (_captures, _chips, plotter, _dc):
        fig, ax = plotter.plot_interval_baseline(
            ha_chips=ha_chips,
            baseline_lag_chips=[10.0, 11.0, 12.0],
            interval_results=interval_results,
            unix_chips=unix_chips,
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
    unix_chips = [
        np.array([100.0, 110.0]),
        np.array([200.0, 210.0]),
        np.array([300.0, 310.0]),
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

    with _lab03_utils() as (_captures, _chips, plotter, _dc):
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
            unix_chips=unix_chips,
        )

        assert "chip 2" in axes[0].get_title()
        fig.canvas.draw()


def test_plot_channel_time_series_accepts_time_axis():
    ha_chips = [np.array([-10.0, -5.0, 0.0]), np.array([5.0, 10.0])]
    unix_chips = [np.array([100.0, 110.0, 120.0]), np.array([130.0, 140.0])]

    with _lab03_utils() as (_captures, _chips, plotter, _dc):
        fig, axes = plotter.plot_channel_time_series(
            ha_chips=ha_chips,
            amp_norm_chips=[np.array([1.0, 1.1, 0.9]), np.array([1.0, 1.2])],
            phase_deg_chips=[np.array([0.0, 10.0, 20.0]), np.array([30.0, 40.0])],
            real_chips=[np.array([1.0, 0.5, 0.0]), np.array([-0.5, -1.0])],
            imag_chips=[np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.5])],
            channel_index=123,
            channel_freq_ghz=10.5,
            ha_limits_deg=(-12.0, 12.0),
            unix_chips=unix_chips,
        )

        assert len(axes) == 4
        fig.canvas.draw()


def test_local_real_dc_correction_uses_chip_local_odd_window():
    unix_chip = np.arange(7, dtype=float) * 10.0
    corr_chip = np.column_stack((np.arange(7, dtype=float), 100.0 + np.arange(7, dtype=float))).astype(complex)

    with _lab03_utils() as (_captures, _chips, _plotter, dc):
        result = dc.local_real_dc_correction(
            corr_chips=[corr_chip],
            unix_chips=[unix_chip],
            bad_channels=[1],
            nominal_fringe_period_sec=20.0,
            window_periods=2.5,
            min_window_caps=5,
        )

    expected_offset = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0])
    assert result.window_sec == 50.0
    assert result.median_cadence_sec_chips == [10.0]
    assert result.window_caps_chips == [5]
    assert np.allclose(result.real_offset_chips[0][:, 0], expected_offset)
    assert np.all(np.isnan(result.real_offset_chips[0][:, 1]))
    assert np.allclose(result.corr_dc_chips[0][:, 0], np.arange(7) - expected_offset)
    assert np.all(np.isnan(result.corr_dc_chips[0][:, 1]))


def test_local_real_dc_correction_uses_full_chip_when_window_is_longer():
    unix_chip = np.arange(4, dtype=float) * 10.0
    corr_chip = np.array([[0.0], [2.0], [10.0], [20.0]], dtype=complex)

    with _lab03_utils() as (_captures, _chips, _plotter, dc):
        result = dc.local_real_dc_correction(
            corr_chips=[corr_chip],
            unix_chips=[unix_chip],
            bad_channels=[],
            nominal_fringe_period_sec=40.0,
            window_periods=2.5,
            min_window_caps=5,
        )

    expected_offset = np.full((4,), 6.0)
    assert result.window_sec == 100.0
    assert result.window_caps_chips == [4]
    assert np.allclose(result.real_offset_chips[0][:, 0], expected_offset)
    assert np.allclose(result.corr_dc_chips[0][:, 0], np.array([-6.0, -4.0, 4.0, 14.0]))


def test_load_capture_series_collates_scalar_and_spectral_fields():
    with _lab03_utils() as (captures, _chips, _plotter, _dc):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path_a = tmp / "cap_a.npz"
            path_b = tmp / "cap_b.npz"
            np.savez(
                path_a,
                corr=np.array([1.0 + 2.0j, 3.0 + 4.0j]),
                corr_std=np.array([0.1, 0.2]),
                unix_time_start=20.0,
                unix_time_end=24.0,
                n_acc=4,
                alt_deg=21.0,
                az_deg=31.0,
                ra_deg=np.nan,
                dec_deg=np.nan,
                duration_sec=4.0,
            )
            np.savez(
                path_b,
                corr=np.array([5.0 + 6.0j, 7.0 + 8.0j]),
                corr_std=np.array([0.3, 0.4]),
                unix_time_start=10.0,
                unix_time_end=15.0,
                n_acc=3,
                alt_deg=20.0,
                az_deg=30.0,
                ra_deg=np.nan,
                dec_deg=np.nan,
                duration_sec=5.0,
            )

            series = captures.load_capture_series([path_a, path_b])
            series_sorted = series.sorted_by_unix_mid()

        assert series.N_cap == 2
        assert series.N_CH == 2
        assert np.allclose(series.corr, np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]))
        assert np.allclose(series.corr_std, np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert np.allclose(series.unix_time_start, np.array([20.0, 10.0]))
        assert np.allclose(series.unix_time_end, np.array([24.0, 15.0]))
        assert np.allclose(series.n_acc, np.array([4, 3]))
        assert np.allclose(series.alt_deg, np.array([21.0, 20.0]))
        assert np.allclose(series.az_deg, np.array([31.0, 30.0]))
        assert np.allclose(series.duration_sec, np.array([4.0, 5.0]))
        assert np.isnan(series.ra_deg).all()
        assert np.isnan(series.dec_deg).all()
        assert [path.name for path in series_sorted.paths] == ["cap_b.npz", "cap_a.npz"]
        assert np.allclose(series_sorted.unix_mid, np.array([12.5, 22.0]))
