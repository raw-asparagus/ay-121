"""Combined SDR + signal generator workflows."""

from .drivers import sdr_utils
from .drivers.siggen import freq_sweep


def sweep_and_capture(synth, sdr, start_mhz, stop_mhz, step_mhz,
                      nsamples=2048, nblocks=1):
    """Sweep the signal generator and capture a power spectrum at each frequency.

    Parameters
    ----------
    synth : ugradio.agilent.SynthDirect
        Connected signal generator.
    sdr : ugradio.sdr.SDR
        Initialized SDR object.
    start_mhz : float
        Start frequency in MHz.
    stop_mhz : float
        Stop frequency in MHz.
    step_mhz : float
        Frequency step in MHz.
    nsamples : int
        Samples per SDR capture block.
    nblocks : int
        Number of blocks to average for each PSD.

    Returns
    -------
    results : dict
        Mapping of ``{freq_mhz: (freqs_hz, psd)}`` where *freqs_hz* is
        the SDR frequency axis and *psd* is the averaged power spectrum.
    """
    results = {}
    for freq in freq_sweep(synth, start_mhz, stop_mhz, step_mhz):
        freqs_hz, psd = sdr_utils.power_spectrum(sdr, nsamples=nsamples,
                                                  nblocks=nblocks)
        results[freq] = (freqs_hz, psd)
    return results