import pytest

from ugradiolab.capture.interferometer import InterfExperiment, PointingError


class _FakeInterferometer:
    def __init__(self, positions):
        self._positions = positions

    def get_pointing(self):
        return self._positions


def test_verify_on_target_wraps_azimuth_across_zero():
    exp = InterfExperiment(
        interferometer=_FakeInterferometer({'east': (45.0, 0.5)}),
        alt_deg=45.0,
        az_deg=359.5,
        pointing_tol_deg=1.0,
    )

    exp._verify_on_target('wrap-test')


def test_verify_on_target_still_rejects_large_wrapped_error():
    exp = InterfExperiment(
        interferometer=_FakeInterferometer({'east': (45.0, 2.5)}),
        alt_deg=45.0,
        az_deg=359.5,
        pointing_tol_deg=1.0,
    )

    with pytest.raises(PointingError, match='off-target'):
        exp._verify_on_target('wrap-test')
