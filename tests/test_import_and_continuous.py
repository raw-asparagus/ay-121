import pytest

import ugradiolab.astronomy.coordinates as coordinates
import ugradiolab.astronomy.ephemeris as ephemeris
from ugradiolab.capture.pipelined import PipelinedCapture


def test_pointing_helpers_do_not_eagerly_bind_ugradio_coord():
    assert 'coord' not in coordinates.__dict__
    assert 'coord' not in ephemeris.__dict__


def test_flush_surfaces_background_on_save_errors():
    capture = PipelinedCapture(interferometer=None, snap=None, pool_workers=1)
    future = capture._executor.submit(
        lambda: (_ for _ in ()).throw(RuntimeError('boom'))
    )
    capture._callback_futures.append(future)

    with pytest.raises(RuntimeError, match='on_save callback'):
        capture.flush()
