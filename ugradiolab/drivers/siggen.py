"""Agilent N9310A signal generator control utilities wrapping ugradio.agilent."""

import numpy as np
from ugradio.agilent import SynthDirect

DEVICE = '/dev/usbtmc0'


def connect(device=DEVICE):
    """Open a USB connection to the Agilent N9310A signal generator.

    Parameters
    ----------
    device : str
        USB TMC device path. Default: '/dev/usbtmc0'.

    Returns
    -------
    SynthDirect
        Connected and validated synthesizer instance.
    """
    return SynthDirect(device=device)


def set_signal(synth, freq_mhz, amp_dbm, rf_on=True):
    """Set the signal generator frequency, amplitude, and RF output in one call.

    Parameters
    ----------
    synth : SynthDirect
        Connected synthesizer.
    freq_mhz : float
        CW frequency in MHz.
    amp_dbm : float
        CW amplitude in dBm.
    rf_on : bool
        Whether to enable RF output. Default: True.
    """
    synth.set_frequency(freq_mhz, 'MHz')
    synth.set_amplitude(amp_dbm, 'dBm')
    synth.set_rfout(on=rf_on)


def freq_sweep(synth, start_mhz, stop_mhz, step_mhz):
    """Generator that steps the signal generator through a frequency range.

    At each step, sets the frequency and yields it so the caller can
    perform measurements (e.g., capture SDR data).

    Parameters
    ----------
    synth : SynthDirect
        Connected synthesizer.
    start_mhz : float
        Start frequency in MHz (inclusive).
    stop_mhz : float
        Stop frequency in MHz (inclusive).
    step_mhz : float
        Frequency step size in MHz.

    Yields
    ------
    freq_mhz : float
        The frequency that was just set, in MHz.
    """
    freqs = np.arange(start_mhz, stop_mhz + step_mhz / 2, step_mhz)
    for freq in freqs:
        synth.set_frequency(float(freq), 'MHz')
        yield float(freq)