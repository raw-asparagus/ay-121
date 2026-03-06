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
    / "unknown_length_collect.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("unknown_length_collect", SCRIPT_PATH)
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


class FakeSDR:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        type(self).instances.append(self)

    def close(self):
        self.closed = True


class UnknownLengthCollectTests(unittest.TestCase):
    def setUp(self):
        FakeSDR.instances = []
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

    def test_run_capture_for_lo_uses_obs_experiment(self):
        fake_exp = mock.Mock()
        fake_exp.run.return_value = "fake.npz"

        with mock.patch.object(self.mod, "ObsExperiment", return_value=fake_exp) as obs_cls:
            path = self.mod.run_capture_for_lo(
                set_id=3,
                lo_hz=self.mod.LO_1420_HZ,
                outdir="tmp",
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                sdr="SDR-OBJECT",
            )

        self.assertEqual(path, "fake.npz")
        obs_cls.assert_called_once()
        kwargs = obs_cls.call_args.kwargs
        self.assertEqual(kwargs["center_freq"], self.mod.LO_1420_HZ)
        self.assertTrue(kwargs["prefix"].startswith("UNKNOWN-set0003-LO1420"))
        fake_exp.run.assert_called_once_with("SDR-OBJECT")

    def test_main_writes_one_manifest_row_with_nan_legacy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "raw"
            manifest = Path(tmp) / "manifest.csv"
            args = SimpleNamespace(
                outdir=str(outdir),
                manifest=str(manifest),
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                setup_seconds=0,
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
                self.mod,
                "run_capture_for_lo",
                side_effect=["set1_lo1420.npz", "set1_lo1421.npz"],
            ) as run_mock, mock.patch.object(
                self.mod,
                "compute_capture_metrics",
                side_effect=[metric_a, metric_b],
            ), mock.patch.object(
                self.mod, "run_setup_countdown"
            ) as timer_mock, mock.patch(
                "builtins.input",
                side_effect=[""],
            ):
                rc = self.mod.main()

            self.assertEqual(rc, 0)
            self.assertEqual(run_mock.call_count, 2)
            timer_mock.assert_called_once_with(0)
            with manifest.open("r", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["set_id"], "1")
            self.assertEqual(row["lo1420_path"], "set1_lo1420.npz")
            self.assertEqual(row["lo1421_path"], "set1_lo1421.npz")
            self.assertEqual(row["cable_length_m"], "nan")
            self.assertEqual(row["power_meter_dbm"], "nan")
            self.assertEqual(row["siggen_freq_mhz"], "nan")
            self.assertEqual(row["siggen_amp_dbm"], "nan")
            self.assertEqual(len(FakeSDR.instances), 1)
            self.assertTrue(FakeSDR.instances[0].closed)

    def test_main_quit_before_timer_does_not_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "raw"
            manifest = Path(tmp) / "manifest.csv"
            args = SimpleNamespace(
                outdir=str(outdir),
                manifest=str(manifest),
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                setup_seconds=300,
                start_set_id=None,
            )

            with mock.patch.object(self.mod, "parse_args", return_value=args), mock.patch.object(
                self.mod, "SDR", FakeSDR
            ), mock.patch.object(
                self.mod, "run_capture_for_lo"
            ) as run_mock, mock.patch.object(
                self.mod, "run_setup_countdown"
            ) as timer_mock, mock.patch(
                "builtins.input",
                side_effect=["q"],
            ):
                rc = self.mod.main()

            self.assertEqual(rc, 0)
            run_mock.assert_not_called()
            timer_mock.assert_not_called()
            self.assertFalse(manifest.exists())
            self.assertEqual(len(FakeSDR.instances), 0)

    def test_main_start_set_id_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "raw"
            manifest = Path(tmp) / "manifest.csv"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.mod.MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow({"set_id": "7"})

            args = SimpleNamespace(
                outdir=str(outdir),
                manifest=str(manifest),
                sample_rate=2.56e6,
                nsamples=8192,
                nblocks=2048,
                gain=0.0,
                direct=False,
                setup_seconds=0,
                start_set_id=50,
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
                self.mod,
                "run_capture_for_lo",
                side_effect=["set50_lo1420.npz", "set50_lo1421.npz"],
            ), mock.patch.object(
                self.mod,
                "compute_capture_metrics",
                side_effect=[metric, metric],
            ), mock.patch.object(
                self.mod, "run_setup_countdown"
            ), mock.patch(
                "builtins.input",
                side_effect=[""],
            ):
                rc = self.mod.main()

            self.assertEqual(rc, 0)
            with manifest.open("r", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["set_id"], "50")


if __name__ == "__main__":
    unittest.main()
