import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from ugradiolab import Record, Spectrum


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "labs"
    / "02"
    / "scripts"
    / "attenuation_collect.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("attenuation_collect", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_record(data: np.ndarray) -> Record:
    return Record(
        data=data.astype(np.complex64),
        sample_rate=2.56e6,
        center_freq=1420.0e6,
        gain=0.0,
        direct=False,
        unix_time=1234.5,
        jd=2460000.125,
        lst=1.75,
        alt=0.0,
        az=0.0,
        obs_lat=37.0,
        obs_lon=-122.0,
        obs_alt=100.0,
        nblocks=data.shape[0],
        nsamples=data.shape[1],
    )


class _FakeDev:
    def close(self):
        return None


class FakeSynth:
    def __init__(self, device=None):
        self.device = device
        self._dev = _FakeDev()
        self.freq_mhz = None
        self.amp_dbm = None

    def set_freq_mhz(self, value):
        self.freq_mhz = value

    def set_ampl_dbm(self, value):
        self.amp_dbm = value

    def rf_off(self):
        return None


class FakeSDR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


class AttenuationCollectTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_script_module()

    def test_next_set_id_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.mod.MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow({"set_id": "2"})
                writer.writerow({"set_id": "7"})
                writer.writerow({"set_id": "3"})
            self.assertEqual(self.mod.next_set_id_from_manifest(path), 8)

    def test_compute_capture_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            iq = np.array(
                [[-127 + 127j, 0 + 10j, 127 - 127j, 5 - 5j]],
                dtype=np.complex64,
            )
            make_record(iq).save(path)

            metrics = self.mod.compute_capture_metrics(path)
            expected_total_power = float(Spectrum.from_data(path).total_power)

        self.assertAlmostEqual(metrics["i_min"], -127.0)
        self.assertAlmostEqual(metrics["i_max"], 127.0)
        self.assertAlmostEqual(metrics["q_min"], -127.0)
        self.assertAlmostEqual(metrics["q_max"], 127.0)
        self.assertAlmostEqual(metrics["i_clip_frac"], 0.5)
        self.assertAlmostEqual(metrics["q_clip_frac"], 0.5)
        self.assertAlmostEqual(metrics["total_power"], expected_total_power)

    def test_main_appends_rows_and_allows_duplicate_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "raw"
            manifest = Path(tmp) / "manifest.csv"
            args = SimpleNamespace(
                outdir=str(outdir),
                manifest=str(manifest),
                siggen_device="/dev/usbtmc0",
                siggen_amp_dbm=-35.0,
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                start_set_id=None,
            )
            metric_a = {
                "total_power": 10.0,
                "i_min": -20.0,
                "i_max": 20.0,
                "i_median": 0.0,
                "i_rms": 5.0,
                "i_clip_frac": 0.0,
                "q_min": -21.0,
                "q_max": 21.0,
                "q_median": 0.0,
                "q_rms": 6.0,
                "q_clip_frac": 0.0,
            }
            metric_b = {
                "total_power": 8.0,
                "i_min": -10.0,
                "i_max": 10.0,
                "i_median": 0.0,
                "i_rms": 3.0,
                "i_clip_frac": 0.0,
                "q_min": -11.0,
                "q_max": 11.0,
                "q_median": 0.0,
                "q_rms": 4.0,
                "q_clip_frac": 0.0,
            }

            with mock.patch.object(self.mod, "parse_args", return_value=args), mock.patch.object(
                self.mod, "SDR", FakeSDR
            ), mock.patch.object(
                self.mod, "SignalGenerator", FakeSynth
            ), mock.patch.object(
                self.mod,
                "run_capture_for_lo",
                side_effect=[
                    "set1_lo1420.npz",
                    "set1_lo1421.npz",
                    "set2_lo1420.npz",
                    "set2_lo1421.npz",
                ],
            ) as run_mock, mock.patch.object(
                self.mod,
                "compute_capture_metrics",
                side_effect=[metric_a, metric_b, metric_a, metric_b],
            ), mock.patch(
                "builtins.input",
                side_effect=[
                    "10",  # length set 1
                    "-20",  # power meter set 1
                    "",  # run set 1
                    "10",  # length set 2 (duplicate)
                    "-21",  # power meter set 2
                    "",  # run set 2
                    "q",  # quit at next set
                ],
            ):
                rc = self.mod.main()

            self.assertEqual(rc, 0)
            self.assertEqual(run_mock.call_count, 4)
            with manifest.open("r", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["set_id"], "1")
            self.assertEqual(rows[1]["set_id"], "2")
            self.assertEqual(rows[0]["cable_length_m"], "10.0")
            self.assertEqual(rows[1]["cable_length_m"], "10.0")
            self.assertEqual(rows[0]["power_meter_dbm"], "-20.0")
            self.assertEqual(rows[1]["power_meter_dbm"], "-21.0")

    def test_main_skip_does_not_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "raw"
            manifest = Path(tmp) / "manifest.csv"
            args = SimpleNamespace(
                outdir=str(outdir),
                manifest=str(manifest),
                siggen_device="/dev/usbtmc0",
                siggen_amp_dbm=-40.0,
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                start_set_id=None,
            )

            with mock.patch.object(self.mod, "parse_args", return_value=args), mock.patch.object(
                self.mod, "SDR", FakeSDR
            ), mock.patch.object(
                self.mod, "SignalGenerator", FakeSynth
            ), mock.patch.object(
                self.mod, "run_capture_for_lo"
            ) as run_mock, mock.patch(
                "builtins.input",
                side_effect=[
                    "5",  # length
                    "-30",  # power meter
                    "s",  # skip
                    "q",  # quit
                ],
            ):
                rc = self.mod.main()

            self.assertEqual(rc, 0)
            run_mock.assert_not_called()
            self.assertFalse(manifest.exists())

    def test_main_continues_set_id_from_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "raw"
            manifest = Path(tmp) / "manifest.csv"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.mod.MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow({"set_id": "5"})

            args = SimpleNamespace(
                outdir=str(outdir),
                manifest=str(manifest),
                siggen_device="/dev/usbtmc0",
                siggen_amp_dbm=-35.0,
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                start_set_id=None,
            )
            metric = {
                "total_power": 10.0,
                "i_min": -20.0,
                "i_max": 20.0,
                "i_median": 0.0,
                "i_rms": 5.0,
                "i_clip_frac": 0.0,
                "q_min": -20.0,
                "q_max": 20.0,
                "q_median": 0.0,
                "q_rms": 5.0,
                "q_clip_frac": 0.0,
            }

            with mock.patch.object(self.mod, "parse_args", return_value=args), mock.patch.object(
                self.mod, "SDR", FakeSDR
            ), mock.patch.object(
                self.mod, "SignalGenerator", FakeSynth
            ), mock.patch.object(
                self.mod,
                "run_capture_for_lo",
                side_effect=["set6_lo1420.npz", "set6_lo1421.npz"],
            ), mock.patch.object(
                self.mod,
                "compute_capture_metrics",
                side_effect=[metric, metric],
            ), mock.patch(
                "builtins.input",
                side_effect=[
                    "12",
                    "-25",
                    "",
                    "q",
                ],
            ):
                rc = self.mod.main()

            self.assertEqual(rc, 0)
            with manifest.open("r", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["set_id"], "6")


if __name__ == "__main__":
    unittest.main()
