"""Agilent/Keysight N9310A signal generator control via USBTMC (SCPI)."""

import time

DEVICE = '/dev/usbtmc0'
WAIT   = 0.25  # seconds between writes


class SignalGenerator:
    """Direct USBTMC interface to an Agilent/Keysight N9310A signal generator.

    Parameters
    ----------
    device : str
        USB TMC device path. Default: '/dev/usbtmc0'.
    """

    def __init__(self, device=DEVICE):
        self._device = device
        self._dev    = open(device, 'rb+')
        self._validate()

    def _validate(self):
        """Verifies the connected instrument is an Agilent/Keysight N9310A."""
        resp = self._query('*IDN?')
        assert 'N9310A' in resp, f'Unexpected instrument: {resp}'

    def _write(self, cmd):
        """Sends a SCPI command to the instrument."""
        cmd = cmd.encode()
        self._dev.write(cmd)
        self._dev.flush()
        time.sleep(WAIT)

    def _read(self):
        """Reads a response from the instrument."""
        chunks = []
        while True:
            try:
                chunks.append(self._dev.read(1))
            except TimeoutError:
                break
        return b''.join(chunks).decode().strip()

    def _query(self, cmd):
        """Sends a SCPI query and return the response string."""
        self._write(cmd)
        return self._read()

    # ---- Frequency --------------------------------------------------------

    def set_freq_mhz(self, freq_mhz):
        """Sets CW frequency in MHz.

        Parameters
        ----------
        freq_mhz : float
            Frequency in MHz.
        """
        self._write(f'FREQ:CW {freq_mhz} MHz')

    def get_freq(self):
        """Queries the current CW frequency.

        Returns
        -------
        float
            Frequency in Hz.
        """
        resp  = self._query('FREQ:CW?')
        parts = resp.split()

        multiplier = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1.0}
        if len(parts) >= 2 and parts[1] in multiplier:
            return float(parts[0]) * multiplier[parts[1]]
        return float(parts[0])

    # ---- Amplitude --------------------------------------------------------

    def set_ampl_dbm(self, amp_dbm):
        """Sets CW amplitude in dBm.

        Parameters
        ----------
        amp_dbm : float
            Amplitude in dBm.
        """
        self._write(f'AMPL:CW {amp_dbm} dBm')

    def get_ampl(self):
        """Queries the current CW amplitude.

        Returns
        -------
        float
            Amplitude in dBm.
        """
        resp  = self._query('AMPL:CW?')
        parts = resp.split()
        return float(parts[0])

    # ---- RF output --------------------------------------------------------

    def rf_on(self):
        """Enables RF output."""
        self._write('RFO:STAT ON')

    def rf_off(self):
        """Disables RF output."""
        self._write('RFO:STAT OFF')

    def rf_state(self):
        """Queries RF output state.

        Returns
        -------
        bool
            True if RF output is on.
        """
        resp = self._query('RFO:STAT?')
        return bool(int(resp.strip()[0]))
