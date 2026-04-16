"""Shared helpers for Lab 3 interferometer calibration scripts."""


def setup_hardware(snap_retries=5):
    """Initialise interferometer and SNAP correlator.  Returns (interferometer, snap).

    snap.initialize() calls align_adc(), which is non-deterministic: the ADC
    ramp test occasionally fails on the first attempt.  snap_retries controls
    how many times initialization is retried before raising.
    """
    import ugradio.interf as interf
    from snap_spec.snap import UGRadioSnap

    interferometer = interf.Interferometer()
    snap = UGRadioSnap(host='localhost', stream_1=0, stream_2=1)
    for attempt in range(1, snap_retries + 1):
        try:
            snap.initialize(mode='corr', sample_rate=500, force=True)
            snap.input.use_adc()
            if attempt > 1:
                print(f'  SNAP initialized on attempt {attempt}.')
            return interferometer, snap
        except AssertionError as exc:
            print(f'  SNAP init attempt {attempt}/{snap_retries} failed ({exc}), retrying...')
    raise RuntimeError(f'SNAP initialization failed after {snap_retries} attempts.')
