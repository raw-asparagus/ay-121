"""Agilent/Keysight N9310A signal generator control via USBTMC (SCPI)."""

import time

DEVICE = '/dev/usbtmc0'
WAIT = 0.25  # seconds between writes


class SignalGenerator:
    """Direct USBTMC interface to an Agilent/Keysight N9310A signal generator.

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
        """Verify the connected instrument is an Agilent/Keysight N9310A."""
        resp = self._query('*IDN?')
        assert 'N9310A' in resp, f'Unexpected instrument: {resp}'

    def _write(self, cmd):
        """Send a SCPI command to the instrument."""
        cmd = cmd.encode()
        self._dev.write(cmd)
        self._dev.flush()
        time.sleep(WAIT)

    def _read(self):
        """Read a response from the instrument."""
        return self._dev.read(4096).decode().strip()

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
        parts = resp.split()
        multiplier = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1.0}
        if len(parts) >= 2 and parts[1] in multiplier:
            return float(parts[0]) * multiplier[parts[1]]
        return float(parts[0])

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
        parts = resp.split()
        return float(parts[0])

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
