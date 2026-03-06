import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ugradiolab import Spectrum

REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_21_NOTEBOOK = REPO_ROOT / "labs" / "02" / "lab_2_1.ipynb"
LAB_21_ANALYSIS_NOTEBOOK = REPO_ROOT / "labs" / "02" / "lab_2_1_analysis.ipynb"


def make_spectrum(
    *,
    psd=None,
    std=None,
    freqs=None,
    center_freq=1420.0e6,
):
    if psd is None:
        psd = np.array([2.0, 4.0, 8.0], dtype=float)
    if std is None:
        std = np.array([0.2, 0.4, 0.8], dtype=float)
    if freqs is None:
        freqs = center_freq + np.array([-1.0, 0.0, 1.0]) * 1.0e6
    return Spectrum(
        psd=np.array(psd, dtype=float),
        std=np.array(std, dtype=float),
        freqs=np.array(freqs, dtype=float),
        sample_rate=2.56e6,
        center_freq=float(center_freq),
        gain=0.0,
        direct=False,
        unix_time=1234.5,
        jd=2460000.125,
        lst=1.75,
        alt=90.0,
        az=0.0,
        obs_lat=37.8732,
        obs_lon=-122.2573,
        obs_alt=100.0,
        nblocks=4,
        nsamples=3,
    )


def load_namespace_until_marker(notebook_path: Path, marker: str) -> dict:
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": f"notebook_helpers::{notebook_path.name}"}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        if not source_text.strip():
            continue
        exec(source_text, namespace)
        if marker in source_text:
            return namespace
    raise RuntimeError(f"Marker {marker!r} not found in {notebook_path}")


class NotebookIOCalibrationHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_namespace_until_marker(
            LAB_21_NOTEBOOK,
            "NOTEBOOK_HELPERS_IO_CAL",
        )

    def test_load_spectra_cached_from_spectra_dir(self):
        load_spectra_cached = self.ns["load_spectra_cached"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "example"
            spectra_dir = root / "example_spectra"
            data_dir.mkdir()
            spectra_dir.mkdir()

            make_spectrum(center_freq=1420.0e6).save(spectra_dir / "a.npz")
            make_spectrum(center_freq=1421.0e6).save(spectra_dir / "b.npz")

            loaded = load_spectra_cached(data_dir)

        self.assertEqual(len(loaded), 2)
        self.assertEqual([int(s.center_freq / 1e6) for s in loaded], [1420, 1421])

    def test_select_spectrum_helpers(self):
        select_spectrum_by_center_freq = self.ns["select_spectrum_by_center_freq"]
        select_spectra_by_center_freq = self.ns["select_spectra_by_center_freq"]
        spectra = [
            make_spectrum(center_freq=1419.95e6),
            make_spectrum(center_freq=1421.02e6),
        ]

        selected = select_spectrum_by_center_freq(spectra, 1420.0e6, tol_hz=0.1e6)
        selected_map = select_spectra_by_center_freq(
            spectra,
            [1420.0e6, 1421.0e6],
            tol_hz=0.1e6,
        )

        self.assertAlmostEqual(selected.center_freq, 1419.95e6)
        self.assertEqual(set(selected_map), {1420.0e6, 1421.0e6})

    def test_measure_y_factor(self):
        measure_y_factor = self.ns["measure_y_factor"]
        measure_y_factor_series = self.ns["measure_y_factor_series"]
        receiver_temperature_from_y = self.ns["receiver_temperature_from_y"]

        hot = make_spectrum(
            psd=[10.0, 10.0, 10.0],
            std=[1.0, 1.0, 1.0],
            center_freq=1420.0e6,
        )
        cold = make_spectrum(
            psd=[5.0, 5.0, 5.0],
            std=[0.5, 0.5, 0.5],
            center_freq=1420.0e6,
        )

        measurement = measure_y_factor(
            hot,
            cold,
            t_hot_k=310.0,
            sigma_t_hot_k=1.0,
            t_cold_k=30.0,
            sigma_t_cold_k=10.0,
        )
        series = measure_y_factor_series(
            [hot],
            [cold],
            center_freqs_hz=[1420.0e6],
            t_hot_k=310.0,
            sigma_t_hot_k=1.0,
            t_cold_k={1420: 30.0},
            sigma_t_cold_k={1420: 10.0},
        )

        self.assertAlmostEqual(receiver_temperature_from_y(2.0, 310.0, 30.0), 250.0)
        self.assertAlmostEqual(measurement.y, 2.0)
        self.assertAlmostEqual(measurement.t_rx_k, 250.0)
        self.assertEqual(series[0].center_freq_mhz, 1420)


class NotebookHIHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_namespace_until_marker(
            LAB_21_ANALYSIS_NOTEBOOK,
            "NOTEBOOK_HELPERS_IO_HI",
        )

    def test_extract_hi_ratio_profile(self):
        extract_hi_ratio_profile = self.ns["extract_hi_ratio_profile"]
        rest_freq_hz = self.ns["HI_REST_FREQ_HZ"]

        numerator = make_spectrum(
            psd=[2.0, 4.0, 8.0],
            std=[0.2, 0.4, 0.8],
            center_freq=1420.0e6,
        )
        denominator = make_spectrum(
            psd=[1.0, 2.0, 4.0],
            std=[0.1, 0.2, 0.4],
            center_freq=1421.0e6,
        )

        profile = extract_hi_ratio_profile(
            numerator,
            denominator,
            rest_freq_hz=rest_freq_hz,
            smooth_kwargs=dict(method="boxcar", M=1),
            velocity_shift_kms=5.0,
        )

        np.testing.assert_allclose(profile.raw_ratio, np.array([2.0, 2.0, 2.0]))
        np.testing.assert_allclose(profile.peak_profile, np.array([1.0, 1.0, 1.0]))
        shift = profile.numerator_velocity_kms[1] - profile.numerator_topo_velocity_kms[1]
        self.assertAlmostEqual(float(shift), 5.0)

    def test_fit_hi_profile(self):
        fit_hi_profile = self.ns["fit_hi_profile"]
        GaussianComponentGuess = self.ns["GaussianComponentGuess"]

        vel = np.linspace(-80.0, 80.0, 161)
        baseline = 0.02 + 0.01 * (vel / 100.0)
        profile = baseline + 0.3 * np.exp(-0.5 * ((vel - 5.0) / 12.0) ** 2)
        sigma = np.full_like(vel, 0.02)

        fit_result = fit_hi_profile(
            vel,
            profile,
            sigma,
            initial_guesses=[GaussianComponentGuess(0.2, 0.0, 10.0)],
            baseline_poly_order=1,
            fit_min_kms=-60.0,
            fit_max_kms=60.0,
            label="unit-test fit",
        )

        self.assertEqual(fit_result.n_components, 1)
        self.assertAlmostEqual(fit_result.components[0].center_kms, 5.0, places=1)
        self.assertLess(fit_result.chi2_red, 0.1)

    def test_zenith_lsr_correction(self):
        zenith_lsr_correction = self.ns["zenith_lsr_correction"]
        spectrum = make_spectrum()

        with mock.patch("ugradio.doppler.get_projected_velocity", return_value=1234.0):
            correction = zenith_lsr_correction(spectrum)

        self.assertAlmostEqual(correction.velocity_kms, 1.234)
        self.assertAlmostEqual(correction.ra_deg, np.degrees(spectrum.lst))
        self.assertAlmostEqual(correction.dec_deg, spectrum.obs_lat)

    def test_simulate_hi_ratio_signature(self):
        simulate_hi_ratio_signature = self.ns["simulate_hi_ratio_signature"]
        rest_freq_hz = self.ns["HI_REST_FREQ_HZ"]

        simulation = simulate_hi_ratio_signature(
            make_spectrum(center_freq=1420.0e6),
            [0.3, 5.0, 10.0, 0.2, -35.0, 20.0],
            signal_center_freq_hz=1420.0e6,
            reference_center_freq_hz=1421.0e6,
            rest_freq_hz=rest_freq_hz,
        )

        self.assertEqual(simulation.ratio.shape, (3,))
        self.assertEqual(simulation.inverse_ratio.shape, (3,))
        self.assertTrue(np.all(np.isfinite(simulation.ratio)))
        self.assertGreater(simulation.lo_separation_kms, 0.0)


if __name__ == "__main__":
    unittest.main()
