import pytest

import ugradiolab.pointing as pointing
from ugradiolab.drivers import interferometer as interf_driver
from ugradiolab.run.continuous import ContinuousCapture


def test_pointing_helpers_do_not_eagerly_bind_ugradio_coord():
    assert 'coord' not in pointing.__dict__
    assert 'coord' not in interf_driver.__dict__


def test_flush_surfaces_background_on_save_errors():
    capture = ContinuousCapture(interferometer=None, snap=None, pool_workers=1)
    future = capture._executor.submit(
        lambda: (_ for _ in ()).throw(RuntimeError('boom'))
    )
    capture._callback_futures.append(future)

    with pytest.raises(RuntimeError, match='on_save callback'):
        capture.flush()
