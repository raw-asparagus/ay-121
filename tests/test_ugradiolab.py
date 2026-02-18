import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ugradiolab.data.record import Record
from ugradiolab.experiment import CalExperiment, ObsExperiment
from ugradiolab.queue import QueueRunner


class FakeSDR:
    def __init__(self):
        self.direct = False
        self._sample_rate = 2.56e6
        self._center_freq = 1420e6
        self._gain = 0.0

    def set_direct_sampling(self, mode):
        self._direct_mode = mode

    def set_center_freq(self, value):
        self._center_freq = value

    def set_sample_rate(self, value):
        self._sample_rate = value

    def set_gain(self, value):
        self._gain = value

    def get_sample_rate(self):
        return self._sample_rate

    def get_center_freq(self):
        return self._center_freq

    def get_gain(self):
        return self._gain

    def capture_data(self, nsamples, nblocks):
        if self.direct:
            return np.zeros((nblocks, nsamples), dtype=np.int8)
        return np.zeros((nblocks, nsamples, 2), dtype=np.int8)


class FakeSynth:
    def __init__(self):
        self._on = False
        self._freq_mhz = 0.0
        self._amp_dbm = 0.0

    def set_freq_mhz(self, value):
        self._freq_mhz = float(value)

    def set_ampl_dbm(self, value):
        self._amp_dbm = float(value)

    def rf_on(self):
        self._on = True

    def rf_off(self):
        self._on = False

    def get_freq(self):
        return self._freq_mhz * 1e6

    def get_ampl(self):
        return self._amp_dbm

    def rf_state(self):
        return self._on


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.sdr = FakeSDR()
        self.synth = FakeSynth()
        self.raw = np.zeros((3, 8, 2), dtype=np.int8)
        self.time_patch = mock.patch.multiple(
            "ugradiolab.data.record.timing",
            unix_time=mock.DEFAULT,
            julian_date=mock.DEFAULT,
            lst=mock.DEFAULT,
        )
        patched = self.time_patch.start()
        patched["unix_time"].return_value = 1234.5
        patched["julian_date"].return_value = 2460000.125
        patched["lst"].return_value = 1.75

    def tearDown(self):
        self.time_patch.stop()

    def test_from_sdr_observation_has_no_siggen_fields(self):
        record = Record.from_sdr(
            self.raw, self.sdr, alt_deg=90.0, az_deg=0.0, synth=None
        )
        self.assertEqual(record.nblocks, 3)
        self.assertEqual(record.nsamples, 8)
        self.assertEqual(record.alt, 90.0)
        self.assertEqual(record.az, 0.0)
        self.assertEqual(record.unix_time, 1234.5)
        self.assertIsNone(record.siggen_freq)
        self.assertIsNone(record.siggen_amp)
        self.assertIsNone(record.siggen_rf_on)

    def test_from_sdr_calibration_has_siggen_fields(self):
        self.synth.set_freq_mhz(1421.2058)
        self.synth.set_ampl_dbm(-35.0)
        self.synth.rf_on()
        record = Record.from_sdr(
            self.raw, self.sdr, alt_deg=45.0, az_deg=180.0, synth=self.synth
        )
        self.assertAlmostEqual(record.siggen_freq, 1421.2058e6)
        self.assertAlmostEqual(record.siggen_amp, -35.0)
        self.assertTrue(record.siggen_rf_on)

    def test_from_sdr_rejects_bad_shape(self):
        with self.assertRaises(ValueError):
            Record.from_sdr(
                np.array([1, 2, 3], dtype=np.int8), self.sdr, alt_deg=0, az_deg=0
            )

    def test_save_and_load_obs_round_trip(self):
        record = Record.from_sdr(self.raw, self.sdr, alt_deg=90.0, az_deg=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.npz"
            record.save(path)
            loaded = Record.load(path)
            self.assertIsNotNone(loaded.data)
            self.assertIsNotNone(loaded.sample_rate)
            self.assertIsNone(loaded.siggen_freq)
            self.assertIsNone(loaded.siggen_amp)
            self.assertIsNone(loaded.siggen_rf_on)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.sdr = FakeSDR()
        self.synth = FakeSynth()
        self.time_patch = mock.patch.multiple(
            "ugradiolab.data.record.timing",
            unix_time=mock.DEFAULT,
            julian_date=mock.DEFAULT,
            lst=mock.DEFAULT,
        )
        patched = self.time_patch.start()
        patched["unix_time"].return_value = 1234.5
        patched["julian_date"].return_value = 2460000.125
        patched["lst"].return_value = 1.75

    def tearDown(self):
        self.time_patch.stop()

    def test_cal_experiment_records_rf_on_then_turns_off(self):
        recorded = {}

        def _capture_save(self, _path):
            recorded["record"] = self

        exp = CalExperiment(
            outdir=".",
            prefix="CAL-UNIT",
            nsamples=16,
            nblocks=2,
            direct=False,
            siggen_freq_mhz=1421.2058,
            siggen_amp_dbm=-35.0,
            alt_deg=90.0,
            az_deg=0.0,
        )

        with mock.patch.object(Record, 'save', autospec=True, side_effect=_capture_save):
            outpath = exp.run(self.sdr, synth=self.synth)

        self.assertTrue(outpath.endswith(".npz"))
        self.assertIn("_cal_", outpath)
        self.assertTrue(recorded["record"].siggen_rf_on)
        self.assertFalse(self.synth.rf_state())

    def test_obs_experiment_record_has_no_siggen(self):
        recorded = {}

        def _capture_save(self, _path):
            recorded["record"] = self

        exp = ObsExperiment(
            outdir=".",
            prefix="OBS-UNIT",
            nsamples=16,
            nblocks=2,
            direct=False,
            alt_deg=45.0,
            az_deg=180.0,
        )

        with mock.patch.object(Record, 'save', autospec=True, side_effect=_capture_save):
            outpath = exp.run(self.sdr, synth=None)

        self.assertTrue(outpath.endswith(".npz"))
        self.assertIn("_obs_", outpath)
        self.assertIsNone(recorded["record"].siggen_freq)
        self.assertIsNone(recorded["record"].siggen_amp)
        self.assertIsNone(recorded["record"].siggen_rf_on)

    def test_cal_experiment_turns_rf_off_if_capture_fails(self):
        exp = CalExperiment(
            outdir=".",
            prefix="CAL-FAIL",
            nsamples=16,
            nblocks=2,
            direct=False,
            siggen_freq_mhz=1421.2058,
            siggen_amp_dbm=-35.0,
            alt_deg=90.0,
            az_deg=0.0,
        )

        def _raise_capture(*_args, **_kwargs):
            raise RuntimeError("capture failed")

        self.sdr.capture_data = _raise_capture

        with self.assertRaises(RuntimeError):
            exp.run(self.sdr, synth=self.synth)

        self.assertFalse(self.synth.rf_state())

    def test_queue_runner_requires_synth_for_calibration(self):
        experiments = [CalExperiment(prefix="NEEDS-SYNTH")]
        runner = QueueRunner(experiments, sdr=self.sdr, synth=None, confirm=False)
        with self.assertRaises(ValueError):
            runner.run()


if __name__ == "__main__":
    unittest.main()
