"""Agilent N9310A signal generator control via USBTMC (SCPI)."""

import time
import numpy as np

DEVICE = '/dev/usbtmc0'
WAIT = 0.3  # seconds between writes


class SignalGenerator:
    """Direct USBTMC interface to an Agilent N9310A signal generator.

    Parameters
    ----------
    device : str
        USB TMC device path. Default: '/dev/usbtmc0'.
    """

    def __init__(self, device=DEVICE):
        self._device = device
        self._dev = open(device, 'rb+')
        self._validate()

    def _validate(self):
        """Verify the connected instrument is an Agilent N9310A."""
        resp = self._query('*IDN?')
        assert 'N9310A' in resp, f'Unexpected instrument: {resp}'

    def _write(self, cmd):
        """Send a SCPI command to the instrument."""
        if isinstance(cmd, str):
            cmd = cmd.encode()
        self._dev.write(cmd)
        self._dev.flush()
        time.sleep(WAIT)

    def _read(self):
        """Read a response from the instrument."""
        chunks = []
        while True:
            try:
                chunks.append(self._dev.read(1))
            except TimeoutError:
                break
        return b''.join(chunks).decode().strip()

    def _query(self, cmd):
        """Send a SCPI query and return the response string."""
        self._write(cmd)
        return self._read()

    # ---- Frequency --------------------------------------------------------

    def set_freq_mhz(self, freq_mhz):
        """Set CW frequency in MHz.

        Parameters
        ----------
        freq_mhz : float
            Frequency in MHz.
        """
        self._write(f'FREQ:CW {freq_mhz} MHz')

    def get_freq(self):
        """Query the current CW frequency.

        Returns
        -------
        float
            Frequency in Hz.
        """
        resp = self._query('FREQ:CW?')
        val, unit, *_ = resp.split()
        multiplier = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1.0}
        return float(val) * multiplier.get(unit, 1.0)

    # ---- Amplitude --------------------------------------------------------

    def set_ampl_dbm(self, amp_dbm):
        """Set CW amplitude in dBm.

        Parameters
        ----------
        amp_dbm : float
            Amplitude in dBm.
        """
        self._write(f'AMPL:CW {amp_dbm} dBm')

    def get_ampl(self):
        """Query the current CW amplitude.

        Returns
        -------
        float
            Amplitude in dBm.
        """
        resp = self._query('AMPL:CW?')
        val, unit, *_ = resp.split()
        return float(val)

    # ---- RF output --------------------------------------------------------

    def rf_on(self):
        """Enable RF output."""
        self._write('RFO:STAT ON')

    def rf_off(self):
        """Disable RF output."""
        self._write('RFO:STAT OFF')

    def rf_state(self):
        """Query RF output state.

        Returns
        -------
        bool
            True if RF output is on.
        """
        resp = self._query('RFO:STAT?')
        return bool(int(resp.strip()[0]))

    # ---- Cleanup ----------------------------------------------------------

    def close(self):
        """Close the USBTMC device."""
        self._dev.close()

    def __del__(self):
        try:
            self._dev.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def connect(device=DEVICE):
    """Open a USB connection to the Agilent N9310A signal generator.

    Parameters
    ----------
    device : str
        USB TMC device path. Default: '/dev/usbtmc0'.

    Returns
    -------
    SignalGenerator
        Connected and validated instance.
    """
    return SignalGenerator(device=device)


def set_signal(synth, freq_mhz, amp_dbm, rf_on=True):
    """Set frequency, amplitude, and RF output in one call.

    Parameters
    ----------
    synth : SignalGenerator
        Connected signal generator.
    freq_mhz : float
        CW frequency in MHz.
    amp_dbm : float
        CW amplitude in dBm.
    rf_on : bool
        Whether to enable RF output. Default: True.
    """
    synth.set_freq_mhz(freq_mhz)
    synth.set_ampl_dbm(amp_dbm)
    if rf_on:
        synth.rf_on()
    else:
        synth.rf_off()


def freq_sweep(synth, start_mhz, stop_mhz, step_mhz):
    """Generator that steps through a frequency range.

    Parameters
    ----------
    synth : SignalGenerator
        Connected signal generator.
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
        synth.set_freq_mhz(float(freq))
        yield float(freq)