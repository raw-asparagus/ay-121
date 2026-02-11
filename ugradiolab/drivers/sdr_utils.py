"""SDR data collection utilities wrapping ugradio.sdr."""

import numpy as np
import ugradio.sdr as _sdr


def _capture(sdr, nsamples, nblocks):
    """Capture data, discarding the first block (stale ring-buffer contents)."""
    data = sdr.capture_data(nsamples=nsamples, nblocks=nblocks + 1)
    return data[1:]


def capture_and_fft(sdr, nsamples=2048, nblocks=1):
    """Capture data from an SDR and compute the FFT for each block.

    Parameters
    ----------
    sdr : ugradio.sdr.SDR
        Initialized SDR object.
    nsamples : int
        Number of samples per block.
    nblocks : int
        Number of blocks to capture.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz, length nsamples.
    fft_data : np.ndarray
        Complex FFT array with shape (nblocks, nsamples).
    """
    data = _capture(sdr, nsamples, nblocks)
    sample_rate = sdr.get_sample_rate()

    if sdr.direct:
        # Real-valued data: use rfft then pad to nsamples for uniform shape
        fft_data = np.fft.rfft(data.astype(np.float64), axis=-1)
        freqs = np.fft.rfftfreq(nsamples, d=1.0 / sample_rate)
    else:
        # I/Q data: form complex signal, use full fft
        iq = data[..., 0].astype(np.float64) + 1j * data[..., 1].astype(np.float64)
        fft_data = np.fft.fftshift(np.fft.fft(iq, axis=-1), axes=-1)
        center_freq = sdr.get_center_freq()
        freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1.0 / sample_rate)) + center_freq

    return freqs, fft_data


def power_spectrum(sdr, nsamples=2048, nblocks=1):
    """Capture data and compute the averaged power spectral density.

    Parameters
    ----------
    sdr : ugradio.sdr.SDR
        Initialized SDR object.
    nsamples : int
        Number of samples per block.
    nblocks : int
        Number of blocks to average over.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz.
    psd : np.ndarray
        Power spectral density, averaged over blocks.
    """
    freqs, fft_data = capture_and_fft(sdr, nsamples=nsamples, nblocks=nblocks)
    psd = np.mean(np.abs(fft_data) ** 2, axis=0)
    return freqs, psd


def collect_time_series(sdr, nsamples=2048, nblocks=1):
    """Capture raw voltage samples and return with a time axis.

    Parameters
    ----------
    sdr : ugradio.sdr.SDR
        Initialized SDR object.
    nsamples : int
        Number of samples per block.
    nblocks : int
        Number of blocks to capture.

    Returns
    -------
    t : np.ndarray
        Time axis in seconds, length nsamples.
    data : np.ndarray
        Raw int8 voltage data with shape (nblocks, nsamples) for direct
        mode or (nblocks, nsamples, 2) for I/Q mode.
    """
    data = _capture(sdr, nsamples, nblocks)
    sample_rate = sdr.get_sample_rate()
    t = np.arange(nsamples) / sample_rate
    return t, data